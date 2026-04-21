"""
CausalBench Benchmark Runner
Generates identification + backdoor problems from ADMGs,
queries a Claude model via LangChain (ChatAnthropic), and scores against
ananke ground truth.

Usage:
    # Run on existing stress test results
    python benchmark_runner.py --results stress_test_results.json

    # Generate fresh graphs and run
    python benchmark_runner.py --generate 50

    # Use a specific model
    python benchmark_runner.py --results stress_test_results.json --model claude-sonnet-4-20250514

    # Filter to small graphs only (easier to reason about)
    python benchmark_runner.py --results stress_test_results.json --max-nodes 8

    # Dry run: generate prompts without calling the API
    python benchmark_runner.py --results stress_test_results.json --dry-run

    # Run only identification or only backdoor questions
    python benchmark_runner.py --results stress_test_results.json --task id
    python benchmark_runner.py --results stress_test_results.json --task backdoor
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    AnthropicRateLimitError = None

from pydantic import BaseModel, field_validator


class Answer(BaseModel):
    question_id: str
    answer: bool

    @field_validator("question_id", mode="before")
    @classmethod
    def normalize_question_id(cls, value):
        if isinstance(value, str):
            return value.strip("()")
        return value


class BenchmarkResponse(BaseModel):
    answers: list[Answer]


def normalized_question_id(value):
    if isinstance(value, str):
        return value.strip("()")
    return value


def iter_answers(response):
    """Yield answer entries from either a BenchmarkResponse or a raw response dict."""
    if response is None:
        return

    if isinstance(response, BenchmarkResponse):
        for answer in response.answers:
            yield answer
        return

    if isinstance(response, dict):
        for answer in response.get("answers", []):
            if isinstance(answer, dict):
                yield answer


def answer_question_id(answer):
    if isinstance(answer, Answer):
        return normalized_question_id(answer.question_id)
    if isinstance(answer, dict):
        return normalized_question_id(answer.get("question_id"))
    return None


def answer_value(answer):
    if isinstance(answer, Answer):
        return answer.answer
    if isinstance(answer, dict):
        return answer.get("answer")
    return None


def find_answer_by_label(response, label):
    for answer in iter_answers(response):
        if answer_question_id(answer) == label:
            return answer
    return None


from backdoor_oracle import generate_backdoor_questions
from frontdoor_oracle import generate_frontdoor_questions
from prompt_templates import (
    SYSTEM_PROMPT,
    build_identification_prompt,
    build_backdoor_prompt,
    build_combined_prompt,
    build_frontdoor_prompt,
    build_full_prompt,
)

DEFAULT_MODEL = "claude-sonnet-4-6"
OUTPUT_DIR = Path(__file__).parent


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"] if isinstance(data, dict) else data


def generate_fresh(n, seed_offset=0):
    """Generate fresh ADMG specs using the stress test generator."""
    # Import from the same directory or from ADMG-gen repo
    try:
        from admg_stress_test import generate_random_admg, test_identification
    except ImportError:
        sys.exit(
            "Could not import admg_stress_test. "
            "Make sure admg_stress_test.py is on PYTHONPATH or in the current directory."
        )

    results = []
    for i in range(n):
        seed = i + seed_offset
        spec = generate_random_admg(seed)
        result = test_identification(i, spec)
        result["seed"] = seed
        results.append(result)
    return results


def call_claude(llm, system, prompt, max_retries=3):
    """Call Claude via LangChain structured output with retries."""
    structured_llm = llm.with_structured_output(BenchmarkResponse)
    for attempt in range(max_retries):
        try:
            return structured_llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ])
        except Exception as e:
            if AnthropicRateLimitError and isinstance(e, AnthropicRateLimitError):
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
    return None


def score_identification(response, ground_truth_identified):
    """Score the identification question (a)."""
    if response is None:
        return {"question": "identification", "correct": None, "error": "no response"}
    q_a = find_answer_by_label(response, "a")
    if not q_a:
        return {"question": "identification", "correct": None, "error": "question (a) not found"}
    q_a_value = answer_value(q_a)
    return {
        "question": "identification",
        "model_answer": q_a_value,
        "ground_truth": ground_truth_identified,
        "correct": q_a_value == ground_truth_identified,
        "source": "ananke",
    }


def score_backdoor(response, backdoor_ground_truth, label_offset=0):
    """Score backdoor criterion questions."""
    if response is None:
        return [{"question": "backdoor", "correct": None, "error": "no response"}]
    results = []
    for i, gt in enumerate(backdoor_ground_truth):
        label = chr(ord("a") + i + label_offset)
        q = find_answer_by_label(response, label)
        if not q:
            results.append({
                "question": f"backdoor_{label}",
                "candidate_set": gt["candidate_set"],
                "correct": None,
                "error": f"question ({label}) not found",
            })
            continue
        q_value = answer_value(q)
        results.append({
            "question": f"backdoor_{label}",
            "candidate_set": gt["candidate_set"],
            "model_answer": q_value,
            "ground_truth": gt["satisfies_backdoor"],
            "correct": q_value == gt["satisfies_backdoor"],
            "source": gt.get("source", "dowhy"),
        })
    return results


def score_frontdoor(response, frontdoor_ground_truth, label_offset=0):
    """Score front-door criterion questions."""
    if response is None:
        return [{"question": "frontdoor", "correct": None, "error": "no response"}]
    results = []
    for i, gt in enumerate(frontdoor_ground_truth):
        label = chr(ord("a") + i + label_offset)
        q = find_answer_by_label(response, label)
        if not q:
            results.append({
                "question": f"frontdoor_{label}",
                "candidate_set": gt["candidate_set"],
                "correct": None,
                "error": f"question ({label}) not found",
            })
            continue
        q_value = answer_value(q)
        results.append({
            "question": f"frontdoor_{label}",
            "candidate_set": gt["candidate_set"],
            "model_answer": q_value,
            "ground_truth": gt["satisfies_frontdoor"],
            "correct": q_value == gt["satisfies_frontdoor"],
            "source": gt.get("source", "dowhy"),
        })
    return results


def validate_admg(spec):
    """
    Validate an ADMG spec before benchmarking.
    Returns a list of error strings; empty list means the spec is valid.
    """
    import networkx as nx

    vertices = set(spec.get("vertices", []))
    treatment = spec.get("treatment")
    outcome = spec.get("outcome")
    di_edges = [tuple(e) for e in spec.get("di_edges", [])]
    bi_edges = [tuple(e) for e in spec.get("bi_edges", [])]
    errors = []

    if not vertices:
        errors.append("empty vertices list")
    if treatment not in vertices:
        errors.append(f"treatment '{treatment}' not in vertices")
    if outcome not in vertices:
        errors.append(f"outcome '{outcome}' not in vertices")
    if treatment == outcome:
        errors.append("treatment and outcome are the same node")

    for u, v in di_edges:
        if u == v:
            errors.append(f"self-loop in directed edges: {u} -> {v}")
        if u not in vertices:
            errors.append(f"directed edge source '{u}' not in vertices")
        if v not in vertices:
            errors.append(f"directed edge target '{v}' not in vertices")

    for u, v in bi_edges:
        if u == v:
            errors.append(f"self-loop in bidirected edges: {u} <-> {v}")
        if u not in vertices:
            errors.append(f"bidirected edge endpoint '{u}' not in vertices")
        if v not in vertices:
            errors.append(f"bidirected edge endpoint '{v}' not in vertices")

    # Check that directed edges form a DAG
    if not errors:
        g_di = nx.DiGraph()
        g_di.add_nodes_from(vertices)
        g_di.add_edges_from(di_edges)
        if not nx.is_directed_acyclic_graph(g_di):
            errors.append("directed edges contain a cycle (graph is not a DAG)")

    return errors


def run_benchmark(
    results,
    model=DEFAULT_MODEL,
    max_nodes=None,
    task="combined",
    dry_run=False,
    output_path=None,
):
    """
    Main benchmark loop.

    Args:
        results: list of ADMG result dicts (from stress test)
        model: Claude model string
        max_nodes: filter to graphs with <= this many nodes
        task: "id", "backdoor", or "combined"
        dry_run: if True, print prompts without calling API
        output_path: where to save benchmark results
    """
    if max_nodes:
        results = [r for r in results if r["n_nodes"] <= max_nodes]

    # Skip graphs that already errored during generation
    results = [r for r in results if r["error"] is None]

    # Validate graph specs and skip any that are malformed
    valid_results = []
    for r in results:
        errs = validate_admg(r)
        if errs:
            print(f"  Skipping graph {r['id']}: {'; '.join(errs)}")
        else:
            valid_results.append(r)
    results = valid_results

    print(f"Benchmarking {len(results)} graphs with model={model}, task={task}")
    if dry_run:
        print("(DRY RUN - no API calls)")

    if dry_run:
        llm = None
    else:
        llm = ChatAnthropic(model=model, max_tokens=2048)

    benchmark_results = []

    for i, spec in enumerate(results):
        gid = spec["id"]
        gt_identified = spec["is_identified"]
        print(f"\n[{i + 1}/{len(results)}] Graph {gid} "
              f"({spec['n_nodes']} nodes, {spec['n_di_edges']} di, {spec['n_bi_edges']} bi) "
              f"— GT: {'ID' if gt_identified else 'NOT ID'}")

        di_edges = [tuple(e) for e in spec["di_edges"]]
        bi_edges = [tuple(e) for e in spec["bi_edges"]]

        # Generate ground truths based on task
        bd_questions = []
        fd_questions = []

        if task in ("backdoor", "combined", "full"):
            bd_questions = generate_backdoor_questions(
                spec["vertices"], di_edges, bi_edges,
                spec["treatment"], spec["outcome"],
            )

        if task in ("frontdoor", "full"):
            fd_questions = generate_frontdoor_questions(
                spec["vertices"], di_edges, bi_edges,
                spec["treatment"], spec["outcome"],
            )

        entry = {
            "graph_id": gid,
            "seed": spec.get("seed"),
            "n_nodes": spec["n_nodes"],
            "treatment": spec["treatment"],
            "outcome": spec["outcome"],
            "ground_truth_identified": gt_identified,
            "backdoor_ground_truth": bd_questions,
            "frontdoor_ground_truth": fd_questions,
            "scores": [],
        }

        # Build prompt based on task type
        if task == "id":
            prompt = build_identification_prompt(spec)
        elif task == "backdoor":
            prompt = build_backdoor_prompt(spec, bd_questions)
        elif task == "combined":
            prompt = build_combined_prompt(spec, bd_questions)
        elif task == "frontdoor":
            prompt = build_frontdoor_prompt(spec, fd_questions)
        elif task == "full":
            prompt = build_full_prompt(spec, bd_questions, fd_questions)

        if dry_run:
            print(f"  Prompt length: {len(prompt)} chars")
            if bd_questions:
                print(f"  Backdoor questions: {len(bd_questions)}")
                for q in bd_questions:
                    cset = q["candidate_set"]
                    if cset == "__any__":
                        label = "any valid set?"
                    elif cset:
                        label = "{" + ", ".join(cset) + "}"
                    else:
                        label = "{}"
                    print(f"    {label} -> {q['satisfies_backdoor']}")
            if fd_questions:
                print(f"  Frontdoor questions: {len(fd_questions)}")
                for q in fd_questions:
                    cset = q["candidate_set"]
                    if cset == "__any__":
                        label = "any valid set?"
                    elif cset:
                        label = "{" + ", ".join(cset) + "}"
                    else:
                        label = "{}"
                    print(f"    {label} -> {q['satisfies_frontdoor']}")
            entry["prompt"] = prompt
            benchmark_results.append(entry)
            continue

        # Call Claude
        response = call_claude(llm, SYSTEM_PROMPT, prompt)
        entry["raw_response"] = response.model_dump() if response else None

        if response is None:
            print("  Failed to get response")
            entry["api_error"] = True
            benchmark_results.append(entry)
            continue

        # Compute label offsets for each question type upfront, based on the
        # fixed task layout. This avoids fragile incremental accumulation.
        if task == "id":
            id_offset, bd_offset, fd_offset = 0, None, None
        elif task == "backdoor":
            id_offset, bd_offset, fd_offset = None, 0, None
        elif task == "frontdoor":
            id_offset, bd_offset, fd_offset = None, None, 0
        elif task == "combined":
            id_offset, bd_offset, fd_offset = 0, 1, None
        elif task == "full":
            id_offset = 0
            bd_offset = 1
            fd_offset = 1 + len(bd_questions)

        if id_offset is not None:
            id_score = score_identification(response, gt_identified)
            entry["scores"].append(id_score)
            status = "CORRECT" if id_score.get("correct") else "WRONG"
            print(f"  Identification: {status}")

        if bd_offset is not None:
            bd_scores = score_backdoor(response, bd_questions, label_offset=bd_offset)
            entry["scores"].extend(bd_scores)
            for s in bd_scores:
                status = "CORRECT" if s.get("correct") else ("WRONG" if s.get("correct") is False else "ERROR")
                print(f"  Backdoor {s['question']}: {status}")

        if fd_offset is not None:
            fd_scores = score_frontdoor(response, fd_questions, label_offset=fd_offset)
            entry["scores"].extend(fd_scores)
            for s in fd_scores:
                status = "CORRECT" if s.get("correct") else ("WRONG" if s.get("correct") is False else "ERROR")
                print(f"  Frontdoor {s['question']}: {status}")

        benchmark_results.append(entry)

        # Small delay to avoid rate limits
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print_summary(benchmark_results, task)

    # Save
    if output_path is None:
        output_path = OUTPUT_DIR / f"benchmark_{task}_{model.replace('/', '_')}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": model,
            "task": task,
            "n_graphs": len(results),
            "max_nodes": max_nodes,
            "results": benchmark_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return benchmark_results


def print_summary(benchmark_results, task):
    # Report API failures first
    api_errors = sum(1 for e in benchmark_results if e.get("api_error"))
    if api_errors:
        print(f"API errors: {api_errors}/{len(benchmark_results)} graphs failed to get a response")

    all_scores = []
    for entry in benchmark_results:
        all_scores.extend(entry.get("scores", []))

    if not all_scores:
        if api_errors == len(benchmark_results):
            print("No scores — every graph had an API error.")
        else:
            print("No scores to summarize (dry run?)")
        return

    # Overall
    scored = [s for s in all_scores if s.get("correct") is not None]
    errored = [s for s in all_scores if s.get("correct") is None]
    correct = sum(1 for s in scored if s["correct"])
    total = len(scored)
    if errored:
        print(f"Parse errors: {len(errored)} question(s) had unparseable or missing answers")
    print(f"Overall: {correct}/{total} ({100 * correct / total:.1f}%)" if total else "No scoreable answers")

    # By question type
    if task in ("id", "combined", "full"):
        id_scores = [s for s in scored if s["question"] == "identification"]
        id_correct = sum(1 for s in id_scores if s["correct"])
        print(f"Identification: {id_correct}/{len(id_scores)} "
              f"({100 * id_correct / len(id_scores):.1f}%)" if id_scores else "")

        # Breakdown: identified vs not
        id_pos = [s for s in id_scores if s["ground_truth"]]
        id_neg = [s for s in id_scores if not s["ground_truth"]]
        if id_pos:
            tp = sum(1 for s in id_pos if s["correct"])
            print(f"  True positives (GT=identified):     {tp}/{len(id_pos)}")
        if id_neg:
            tn = sum(1 for s in id_neg if s["correct"])
            print(f"  True negatives (GT=not identified):  {tn}/{len(id_neg)}")
        print(f"  Ground truth: ananke OneLineID")

    if task in ("backdoor", "combined", "full"):
        bd_scores = [s for s in scored if s["question"].startswith("backdoor")]
        bd_correct = sum(1 for s in bd_scores if s["correct"])
        print(f"Backdoor: {bd_correct}/{len(bd_scores)} "
              f"({100 * bd_correct / len(bd_scores):.1f}%)" if bd_scores else "")

        # By ground truth source
        for src in ("ananke", "dowhy", "d-sep"):
            src_scores = [s for s in bd_scores if s.get("source") == src]
            if src_scores:
                src_correct = sum(1 for s in src_scores if s["correct"])
                print(f"  [{src}] {src_correct}/{len(src_scores)} "
                      f"({100 * src_correct / len(src_scores):.1f}%)")

    if task in ("frontdoor", "full"):
        fd_scores = [s for s in scored if s["question"].startswith("frontdoor")]
        fd_correct = sum(1 for s in fd_scores if s["correct"])
        print(f"Frontdoor: {fd_correct}/{len(fd_scores)} "
              f"({100 * fd_correct / len(fd_scores):.1f}%)" if fd_scores else "")

    # By graph size
    print("\nBy graph size:")
    size_groups = {}
    for entry in benchmark_results:
        n = entry["n_nodes"]
        bucket = f"{n} nodes" if n <= 10 else f"{(n // 5) * 5}-{(n // 5) * 5 + 4} nodes"
        if bucket not in size_groups:
            size_groups[bucket] = {"correct": 0, "total": 0}
        for s in entry.get("scores", []):
            if s.get("correct") is not None:
                size_groups[bucket]["total"] += 1
                if s["correct"]:
                    size_groups[bucket]["correct"] += 1

    for bucket in sorted(size_groups.keys()):
        g = size_groups[bucket]
        pct = 100 * g["correct"] / g["total"] if g["total"] else 0
        print(f"  {bucket}: {g['correct']}/{g['total']} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="CausalBench benchmark runner")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=str, help="Path to stress_test_results.json")
    source.add_argument("--generate", type=int, metavar="N", help="Generate N fresh ADMGs")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Claude model to benchmark (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Filter to graphs with at most this many nodes")
    parser.add_argument("--task",
                        choices=["id", "backdoor", "frontdoor", "combined", "full"],
                        default="combined",
                        help="Which questions to ask (default: combined)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate prompts without calling the API")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for benchmark results JSON")

    args = parser.parse_args()

    if args.results:
        results = load_results(args.results)
    else:
        results = generate_fresh(args.generate)

    run_benchmark(
        results,
        model=args.model,
        max_nodes=args.max_nodes,
        task=args.task,
        dry_run=args.dry_run,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

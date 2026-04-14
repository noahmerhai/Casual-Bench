"""
Prompt formatting for causal inference benchmark.
Converts ADMG specs and question sets into structured prompts for the Claude API.
"""


def format_graph_description(vertices, di_edges, bi_edges, treatment, outcome):
    """Render graph as a human-readable text block."""
    lines = []
    lines.append(f"Vertices: {', '.join(sorted(vertices))}")
    lines.append(f"Directed edges (parent -> child):")
    for u, v in di_edges:
        lines.append(f"  {u} -> {v}")
    if bi_edges:
        lines.append(f"Bidirected edges (latent common cause):")
        for u, v in bi_edges:
            lines.append(f"  {u} <-> {v}")
    else:
        lines.append("Bidirected edges: none")
    lines.append(f"Treatment: {treatment}")
    lines.append(f"Outcome: {outcome}")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are an expert in causal inference. You will be given a causal graph \
(an Acyclic Directed Mixed Graph / ADMG) and asked yes/no questions about \
identifiability, the backdoor criterion, and/or the front-door criterion.

Rules:
- Directed edges (A -> B) represent direct causal effects.
- Bidirected edges (A <-> B) represent latent common causes (unobserved confounders).
- All vertices listed are OBSERVED unless connected only via bidirected edges to represent hidden variables.
- The backdoor criterion (Pearl 1993) for a set Z relative to (X, Y) requires:
  1. No node in Z is a descendant of X.
  2. Z blocks every path between X and Y that contains an arrow into X \
(i.e., Z d-separates X from Y in the graph with all edges out of X removed).
- The front-door criterion (Pearl 1995) for a set M relative to (X, Y) requires:
  (i)   M intercepts all directed paths from X to Y.
  (ii)  There is no unblocked backdoor path from X to M \
(i.e., X is d-separated from M in the graph with all edges out of X removed).
  (iii) All backdoor paths from M to Y are blocked by X \
(i.e., M is d-separated from Y given {X} in the graph with all edges out of M removed).

Answer each question with true (yes) or false (no) for its question_id label.
"""


def build_identification_prompt(spec):
    """
    Build a prompt asking whether P(outcome | do(treatment)) is identifiable
    from the observational distribution over observed variables.
    """
    graph_desc = format_graph_description(
        spec["vertices"], spec["di_edges"], spec["bi_edges"],
        spec["treatment"], spec["outcome"],
    )
    prompt = f"""\
Consider the following ADMG:

{graph_desc}

Answer the following questions:

(a) Is P({spec['outcome']} | do({spec['treatment']})) identifiable from the \
observational distribution P({', '.join(sorted(spec['vertices']))}) in this graph?

(b) Is there any subset of the observed variables (excluding {spec['treatment']} \
and {spec['outcome']}) that satisfies the backdoor criterion for \
({spec['treatment']}, {spec['outcome']})?\
"""
    return prompt


def build_backdoor_prompt(spec, backdoor_questions):
    """
    Build a prompt with specific backdoor criterion questions
    for candidate adjustment sets.
    """
    graph_desc = format_graph_description(
        spec["vertices"], spec["di_edges"], spec["bi_edges"],
        spec["treatment"], spec["outcome"],
    )

    q_lines = []
    for i, q in enumerate(backdoor_questions):
        label = chr(ord("a") + i)
        cset = q["candidate_set"]
        if cset == "__any__":
            q_lines.append(
                f"({label}) Does any subset of observed variables (excluding "
                f"{spec['treatment']} and {spec['outcome']}) satisfy the backdoor "
                f"criterion for ({spec['treatment']}, {spec['outcome']})?"
            )
        else:
            cset_str = "{" + ", ".join(cset) + "}" if cset else "{} (empty set)"
            q_lines.append(
                f"({label}) Does the set {cset_str} satisfy the backdoor criterion "
                f"for ({spec['treatment']}, {spec['outcome']})?"
            )

    questions_block = "\n".join(q_lines)

    prompt = f"""\
Consider the following ADMG:

{graph_desc}

Answer the following questions about the backdoor criterion:

{questions_block}\
"""
    return prompt


def build_combined_prompt(spec, backdoor_questions):
    """
    Build a single prompt with both identification and backdoor questions.
    """
    graph_desc = format_graph_description(
        spec["vertices"], spec["di_edges"], spec["bi_edges"],
        spec["treatment"], spec["outcome"],
    )

    q_lines = []
    q_lines.append(
        f"(a) Is P({spec['outcome']} | do({spec['treatment']})) identifiable "
        f"from the observational distribution over all observed variables in this graph?"
    )

    for i, q in enumerate(backdoor_questions):
        label = chr(ord("b") + i)
        cset = q["candidate_set"]
        if cset == "__any__":
            q_lines.append(
                f"({label}) Does any subset of observed variables (excluding "
                f"{spec['treatment']} and {spec['outcome']}) satisfy the backdoor "
                f"criterion for ({spec['treatment']}, {spec['outcome']})?"
            )
        else:
            cset_str = "{" + ", ".join(cset) + "}" if cset else "{} (empty set)"
            q_lines.append(
                f"({label}) Does the set {cset_str} satisfy the backdoor criterion "
                f"for ({spec['treatment']}, {spec['outcome']})?"
            )

    questions_block = "\n".join(q_lines)

    prompt = f"""\
Consider the following ADMG:

{graph_desc}

Answer the following questions:

{questions_block}\
"""
    return prompt


def _format_frontdoor_question(label, spec, q):
    """Format a single front-door question line."""
    cset = q["candidate_set"]
    if cset == "__any__":
        return (
            f"({label}) Does any subset of observed variables (excluding "
            f"{spec['treatment']} and {spec['outcome']}) satisfy the front-door "
            f"criterion for ({spec['treatment']}, {spec['outcome']})?"
        )
    else:
        cset_str = "{" + ", ".join(cset) + "}"
        return (
            f"({label}) Does the set {cset_str} satisfy the front-door criterion "
            f"for ({spec['treatment']}, {spec['outcome']})?"
        )


def build_frontdoor_prompt(spec, frontdoor_questions):
    """
    Build a prompt with front-door criterion questions.
    """
    graph_desc = format_graph_description(
        spec["vertices"], spec["di_edges"], spec["bi_edges"],
        spec["treatment"], spec["outcome"],
    )

    q_lines = []
    for i, q in enumerate(frontdoor_questions):
        label = chr(ord("a") + i)
        q_lines.append(_format_frontdoor_question(label, spec, q))

    questions_block = "\n".join(q_lines)

    prompt = f"""\
Consider the following ADMG:

{graph_desc}

Answer the following questions about the front-door criterion:

{questions_block}\
"""
    return prompt


def build_full_prompt(spec, backdoor_questions, frontdoor_questions):
    """
    Build a single prompt with identification, backdoor, and front-door questions.
    """
    graph_desc = format_graph_description(
        spec["vertices"], spec["di_edges"], spec["bi_edges"],
        spec["treatment"], spec["outcome"],
    )

    q_lines = []
    label_idx = 0

    # (a) identification
    label = chr(ord("a") + label_idx)
    q_lines.append(
        f"({label}) Is P({spec['outcome']} | do({spec['treatment']})) identifiable "
        f"from the observational distribution over all observed variables in this graph?"
    )
    label_idx += 1

    # backdoor questions
    for q in backdoor_questions:
        label = chr(ord("a") + label_idx)
        cset = q["candidate_set"]
        if cset == "__any__":
            q_lines.append(
                f"({label}) Does any subset of observed variables (excluding "
                f"{spec['treatment']} and {spec['outcome']}) satisfy the backdoor "
                f"criterion for ({spec['treatment']}, {spec['outcome']})?"
            )
        else:
            cset_str = "{" + ", ".join(cset) + "}" if cset else "{} (empty set)"
            q_lines.append(
                f"({label}) Does the set {cset_str} satisfy the backdoor criterion "
                f"for ({spec['treatment']}, {spec['outcome']})?"
            )
        label_idx += 1

    # frontdoor questions
    for q in frontdoor_questions:
        label = chr(ord("a") + label_idx)
        q_lines.append(_format_frontdoor_question(label, spec, q))
        label_idx += 1

    questions_block = "\n".join(q_lines)

    prompt = f"""\
Consider the following ADMG:

{graph_desc}

Answer the following questions:

{questions_block}\
"""
    return prompt


"""
Backdoor criterion ground-truth oracle.

Ground truth: DoWhy 0.14 AutoIdentifier with BACKDOOR_EXHAUSTIVE.
All candidate sets checked for membership in DoWhy's exhaustive enumeration.

Requires: pip install dowhy pandas
"""

from __future__ import annotations
import warnings

import pandas as pd

from graph_utils import admg_to_gml

try:
    from dowhy import CausalModel
    from dowhy.causal_identifier import BackdoorAdjustment
    _DOWHY_AVAILABLE = True
except ImportError:
    _DOWHY_AVAILABLE = False


def find_backdoor_sets_dowhy(vertices, di_edges, bi_edges, treatment, outcome):
    """
    Use DoWhy AutoIdentifier with BACKDOOR_EXHAUSTIVE to enumerate all valid
    backdoor adjustment sets.

    Returns: deduplicated list of frozensets (one per valid set).
    The empty frozenset is included when no adjustment is needed.
    Returns [] when DoWhy is unavailable or no valid set exists.
    """
    if not _DOWHY_AVAILABLE:
        return []

    gml = admg_to_gml(vertices, di_edges, bi_edges)
    dummy_data = pd.DataFrame({v: [0.0] for v in vertices})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CausalModel(
            data=dummy_data,
            treatment=treatment,
            outcome=outcome,
            graph=gml,
            proceed_when_unidentifiable=True,
        )
        estimand = model.identify_effect(
            proceed_when_unidentifiable=True,
            method_name=BackdoorAdjustment.BACKDOOR_EXHAUSTIVE.value,
        )

    if estimand is None or not estimand.backdoor_variables:
        return []

    # backdoor_variables is a dict {'backdoor1': [...], 'backdoor': [...], ...}
    # Deduplicate by converting each list to a frozenset (preserves insertion order).
    return list(dict.fromkeys(frozenset(v) for v in estimand.backdoor_variables.values()))


def generate_backdoor_questions(vertices, di_edges, bi_edges, treatment, outcome):
    """
    Generate backdoor criterion questions with DoWhy ground-truth labels.

    Each candidate is checked for membership in DoWhy's exhaustive set list.

    Returns list of dicts:
        candidate_set      : list[str] | "__any__"
        satisfies_backdoor : bool
        source             : "dowhy"
    """
    valid_sets = find_backdoor_sets_dowhy(vertices, di_edges, bi_edges, treatment, outcome)
    valid_frozen = set(valid_sets)

    questions = []
    asked: set[frozenset] = set()

    # Q: empty set — valid when there are no unblocked backdoor paths
    empty_key = frozenset()
    questions.append({
        "candidate_set": [],
        "satisfies_backdoor": empty_key in valid_frozen,
        "source": "dowhy",
    })
    asked.add(empty_key)

    # Q: smallest valid set found by DoWhy (positive example)
    if valid_sets:
        best = min(valid_sets, key=len)
        key = frozenset(best)
        if key not in asked:
            questions.append({
                "candidate_set": sorted(best),
                "satisfies_backdoor": True,
                "source": "dowhy",
            })
            asked.add(key)

    # Q: outcome as sole candidate — always invalid (outcome never in adjustment set)
    outcome_key = frozenset({outcome})
    if outcome_key not in asked:
        questions.append({
            "candidate_set": [outcome],
            "satisfies_backdoor": outcome_key in valid_frozen,
            "source": "dowhy",
        })
        asked.add(outcome_key)

    # Q: does any valid backdoor set exist?
    questions.append({
        "candidate_set": "__any__",
        "satisfies_backdoor": len(valid_sets) > 0,
        "source": "dowhy",
    })

    return questions

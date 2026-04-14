"""
Front-door criterion ground-truth oracle.

Ground truth: DoWhy 0.14 CausalModel.identify_effect()
via estimand.get_frontdoor_variables().

Requires: pip install dowhy pandas
"""

from __future__ import annotations
import warnings

import pandas as pd

from graph_utils import admg_to_gml

try:
    from dowhy import CausalModel
    _DOWHY_AVAILABLE = True
except ImportError:
    _DOWHY_AVAILABLE = False


def find_frontdoor_sets_dowhy(vertices, di_edges, bi_edges, treatment, outcome):
    """
    Use DoWhy identify_effect to find the front-door variable set.

    Returns: list containing one frozenset (the front-door set), or [] if
    DoWhy is unavailable or no front-door set is identified.
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
        estimand = model.identify_effect(proceed_when_unidentifiable=True)

    if estimand is None:
        return []

    raw = estimand.get_frontdoor_variables()
    if not raw:
        return []

    # raw is a flat list of variable names representing one front-door set
    return [frozenset(raw)]


def generate_frontdoor_questions(vertices, di_edges, bi_edges, treatment, outcome):
    """
    Generate front-door criterion questions with DoWhy ground-truth labels.

    Returns list of dicts:
        candidate_set       : list[str] | "__any__"
        satisfies_frontdoor : bool
        source              : "dowhy"
    """
    valid_sets = find_frontdoor_sets_dowhy(vertices, di_edges, bi_edges, treatment, outcome)
    valid_frozen = set(valid_sets)

    questions = []
    asked: set[frozenset] = set()

    # Q: DoWhy's front-door set if one exists (positive example)
    if valid_sets:
        best = min(valid_sets, key=len)
        key = frozenset(best)
        questions.append({
            "candidate_set": sorted(best),
            "satisfies_frontdoor": True,
            "source": "dowhy",
        })
        asked.add(key)

    # Q: outcome as sole candidate — always invalid (outcome cannot be a mediator)
    outcome_key = frozenset({outcome})
    if outcome_key not in asked:
        questions.append({
            "candidate_set": [outcome],
            "satisfies_frontdoor": outcome_key in valid_frozen,
            "source": "dowhy",
        })
        asked.add(outcome_key)

    # Q: does any valid front-door set exist?
    questions.append({
        "candidate_set": "__any__",
        "satisfies_frontdoor": len(valid_sets) > 0,
        "source": "dowhy",
    })

    return questions

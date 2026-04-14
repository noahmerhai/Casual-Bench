"""
Shared graph utilities for converting ADMG specs to formats used by
DoWhy (GML) and networkx (augmented DiGraph).

All functions verified against:
  - networkx 3.x: DiGraph, generate_gml, is_d_separator, descendants, has_path
  - DoWhy 0.10+: CausalModel accepts GML with observed="yes"/"no" node attributes
"""

import networkx as nx


def admg_to_gml(vertices, di_edges, bi_edges):
    """
    Convert ADMG to GML string for DoWhy.
    Bidirected U <-> V becomes an unobserved node U_U_V with edges to both.
    DoWhy reads the 'observed' node attribute to distinguish latents.
    """
    g = nx.DiGraph()

    for v in vertices:
        g.add_node(v, observed="yes")

    for u, v in di_edges:
        g.add_edge(u, v)

    seen_bi = set()
    for u, v in bi_edges:
        # Normalize order so (V0,V1) and (V1,V0) map to the same latent node
        key = tuple(sorted([u, v]))
        if key in seen_bi:
            continue
        seen_bi.add(key)
        h = f"U_{key[0]}_{key[1]}"
        g.add_node(h, observed="no")
        g.add_edge(h, key[0])
        g.add_edge(h, key[1])

    return "\n".join(nx.generate_gml(g))



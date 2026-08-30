# MERci/analysis/cell_mapping.py
"""
Per-cell identity matching between two segmentations of the same tissue
(e.g. the same sample imaged on two different microscopes, after
`acquisition.alignment.fit_similarity_alignment` has already brought both
experiments' cell centroids into one shared coordinate frame).

This implements the cheapest layer of a staged cross-microscope cell-
identity-mapping plan (coarse tissue-boundary alignment, then per-cell
matching, then optional segmentation cleanup) developed for `BC555_sample_05`
(`epi` vs `disk`, see `notebooks/after_imaging/06_map_cells_across_
microscopes.ipynb`): mutual-nearest-neighbour matching on cell centroids,
refined by a cell-contact-graph consistency filter — no new dependency
(`scipy.spatial.cKDTree` covers both nearest-neighbour queries and
contact-graph construction).

Functions
---------
build_contact_graph        – {cell_index: {neighbour_index, ...}} from
                              centroids within a distance threshold (the
                              "cells in contact" graph)
match_mutual_nearest_neighbor – 1:1 candidate matches between two centroid
                              sets (injective by construction — no separate
                              assignment-optimisation step needed, see its
                              own docstring)
graph_consistency_filter   – keep only candidate matches whose contact-graph
                              neighbours are also (mostly) mutually matched
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree


def build_contact_graph(centroids_xy: np.ndarray, contact_dist: float) -> Dict[int, Set[int]]:
    """
    Build a "cells in contact" graph: an edge between cell *i* and cell *j*
    whenever their centroids are within *contact_dist* of each other.

    Parameters
    ----------
    centroids_xy : (N, 2) array of cell centroid (x, y) coordinates
    contact_dist : distance threshold (same units as the coordinates)

    Returns
    -------
    ``{i: {neighbour indices of i}}`` for every ``i`` in range(N) (isolated
    cells map to an empty set).
    """
    centroids_xy = np.asarray(centroids_xy, dtype=float)
    n = len(centroids_xy)
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n)}
    if n < 2:
        return adjacency

    tree = cKDTree(centroids_xy)
    for i, j in tree.query_pairs(r=contact_dist):
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def match_mutual_nearest_neighbor(
    src_xy: np.ndarray, tgt_xy: np.ndarray, max_dist: float,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    1:1 candidate matches between two already-aligned centroid point sets:
    a pair ``(i, j)`` is kept only when *j* is *i*'s nearest *tgt* neighbour
    **and** *i* is *j*'s nearest *src* neighbour, and that nearest-neighbour
    distance is within *max_dist*.

    A mutual-nearest-neighbour pair is automatically 1:1 with no separate
    assignment-optimisation (e.g. the Hungarian algorithm) needed: if *j* is
    the unique nearest neighbour of *i* and vice versa, no other point can
    also form a mutual pair with either — it would have to be nearer to *j*
    than *i* is, contradicting *i* being *j*'s nearest neighbour. Cheap,
    dependency-free (`scipy.spatial.cKDTree`), but *not* the fancier
    Coherent-Point-Drift/graph-matching refinement worth reaching for if this
    alone leaves the residual misalignment too large (e.g. real local
    elastic distortion between mountings) — start here and escalate only if
    needed.

    Parameters
    ----------
    src_xy, tgt_xy : (N, 2) / (M, 2) centroid coordinates, already in one
                     shared coordinate frame (see module docstring)
    max_dist       : maximum nearest-neighbour distance to accept a match

    Returns
    -------
    ``(pairs, distances)`` — ``pairs`` is a list of ``(src_index, tgt_index)``
    tuples; ``distances[k]`` is the centroid distance for ``pairs[k]``.
    """
    src_xy = np.asarray(src_xy, dtype=float)
    tgt_xy = np.asarray(tgt_xy, dtype=float)
    if len(src_xy) == 0 or len(tgt_xy) == 0:
        return [], np.empty(0)

    src_tree = cKDTree(src_xy)
    tgt_tree = cKDTree(tgt_xy)

    tgt_dist, tgt_nn = tgt_tree.query(src_xy, k=1)   # nearest tgt point for each src point
    src_dist, src_nn = src_tree.query(tgt_xy, k=1)   # nearest src point for each tgt point

    pairs: List[Tuple[int, int]] = []
    distances: List[float] = []
    for i, (j, d) in enumerate(zip(tgt_nn, tgt_dist)):
        if d > max_dist:
            continue
        if src_nn[j] == i:   # mutual
            pairs.append((i, int(j)))
            distances.append(float(d))
    return pairs, np.asarray(distances)


def graph_consistency_filter(
    pairs: List[Tuple[int, int]],
    src_adjacency: Dict[int, Set[int]],
    tgt_adjacency: Dict[int, Set[int]],
    min_fraction: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Keep only the candidate matches in *pairs* whose contact-graph
    neighbours are consistent with the match: for a pair ``(i, j)``, look at
    how many of *i*'s contact neighbours (in *src_adjacency*) are themselves
    matched (by another entry in *pairs*) to one of *j*'s contact neighbours
    (in *tgt_adjacency*) — keep the pair only when that fraction is at least
    *min_fraction*.

    A lightweight, dependency-free stand-in for full inexact graph matching
    (e.g. the coherent-point-drift-then-graph-matching approach used for a
    closely related histopathology cross-modality alignment problem,
    arXiv:2410.00152): it only *filters* the mutual-nearest-neighbour
    candidates from :func:`match_mutual_nearest_neighbor`, rather than
    solving matching from scratch, so it is cheap and has no extra
    dependency. Isolated cells (no contact neighbours in *either* graph)
    have nothing to check against and are kept as-is — this filter only
    removes matches contradicted by their neighbourhood, not ones it merely
    lacks context for.

    Parameters
    ----------
    pairs         : candidate ``(src_index, tgt_index)`` matches (e.g. from
                    :func:`match_mutual_nearest_neighbor`)
    src_adjacency, tgt_adjacency : contact graphs (:func:`build_contact_graph`)
                    for the *src* and *tgt* point sets respectively
    min_fraction  : minimum fraction of a matched cell's neighbours that must
                    also be consistently matched to keep the pair

    Returns
    -------
    The subset of *pairs* that pass the consistency check, in the same order.
    """
    src_to_tgt = dict(pairs)
    kept: List[Tuple[int, int]] = []
    for i, j in pairs:
        src_neighbors = src_adjacency.get(i, set())
        if not src_neighbors:
            kept.append((i, j))
            continue
        tgt_neighbors = tgt_adjacency.get(j, set())
        consistent = sum(
            1 for n in src_neighbors
            if n in src_to_tgt and src_to_tgt[n] in tgt_neighbors
        )
        if consistent / len(src_neighbors) >= min_fraction:
            kept.append((i, j))
    return kept

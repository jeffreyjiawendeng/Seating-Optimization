# seating_opt/distance.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Tuple, List


def pairwise_distances(
    seats: pd.DataFrame, max_dist: int | None = None
) -> Dict[Tuple[int, int], int]:
    """
    Manhattan distances for i<j; optionally truncate pairs by max_dist to keep the ILP small.
    """
    coords = seats.set_index("Seat_ID")[["X", "Y"]].astype(int).to_dict("index")
    keys = list(coords.keys())
    d = {}
    for i_idx in range(len(keys)):
        i = keys[i_idx]
        xi, yi = coords[i]["X"], coords[i]["Y"]
        for j_idx in range(i_idx + 1, len(keys)):
            j = keys[j_idx]
            xj, yj = coords[j]["X"], coords[j]["Y"]
            dist = abs(xi - xj) + abs(yi - yj)
            if max_dist is None or dist <= max_dist:
                d[(i, j)] = dist
    return d


def build_pair_set(
    seats: pd.DataFrame, max_dist: int | None = None
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    Returns (pair_list, distance_map) where pair_list has (i,j) with i<j.
    """
    dmap = pairwise_distances(seats, max_dist=max_dist)
    pairs = list(dmap.keys())
    return pairs, dmap
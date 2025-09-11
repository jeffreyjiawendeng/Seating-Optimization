# seating_opt/utils.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List


REQUIRED_SEAT_COLS = [
    "Seat_ID", "Room_ID", "Table_ID", "X", "Y",
    "Brightness", "Noise", "Seat_Available"
]

REQUIRED_GROUP_COLS = [
    "Group_ID", "Group_Size", "Brightness_Min", "Objective"  # Objective == "Q1"
]


def validate_seats_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_SEAT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"seats_df missing columns: {missing}")
    if df["Seat_Available"].dtype != bool:
        df["Seat_Available"] = df["Seat_Available"].astype(bool)


def validate_groups_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_GROUP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"groups_df missing columns: {missing}")
    bad = set(df["Objective"].unique()) - {"Q1"}
    if bad:
        raise ValueError("This prototype supports Objective == 'Q1' (minimize noise).")


def manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def build_adjacent(df: pd.DataFrame) -> Dict[int, List[int]]:
    """
    4-neighbor adjacency on stitched (X,Y) grid; across-table borders work if coordinates touch.
    (Not required for the ILP—handy for diagnostics.)
    """
    by_coord = {(int(r.X), int(r.Y)): int(r.Seat_ID) for _, r in df.iterrows()}
    adj = {int(r.Seat_ID): [] for _, r in df.iterrows()}
    for (x, y), sid in by_coord.items():
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = by_coord.get((x + dx, y + dy))
            if nb is not None:
                adj[sid].append(nb)
    return adj


def avg_pairwise_distance(seat_ids: List[int], coords: pd.DataFrame) -> float:
    """coords is a DataFrame indexed by Seat_ID with columns X,Y (ints)."""
    if len(seat_ids) <= 1:
        return 0.0
    pts = coords.loc[seat_ids, ["X", "Y"]].astype(int).values
    n = len(pts)
    s = 0
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += abs(pts[i, 0] - pts[j, 0]) + abs(pts[i, 1] - pts[j, 1])
            c += 1
    return s / c
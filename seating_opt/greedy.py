# seating_opt/greedy.py
from __future__ import annotations
import pandas as pd
import heapq
from typing import Dict, List, Tuple
from .utils import validate_seats_df
from .distance import pairwise_distances


def greedy_pairwise(
    seats_df: pd.DataFrame,
    group_size: int,
    brightness_min: float,
    lam_pair: float = 0.3,
) -> dict:
    """
    Greedy baseline with closeness in the objective (no radius constraint).
      - Start at the quietest seat (lowest Noise).
      - Repeatedly add the seat that minimizes incremental cost:
            Δcost(j) = Noise_j + λ * sum_{i in chosen} d(i,j)
        while preserving a forward feasibility check for average brightness.
    Returns {'status': 'ok'|'infeasible', 'seat_ids': [...]}.
    """
    validate_seats_df(seats_df)
    cand = seats_df[seats_df["Seat_Available"]].copy()
    if len(cand) < group_size:
        return {"status": "infeasible", "reason": "insufficient capacity"}

    # Precompute pairwise Manhattan distances (full, since greedy is local anyway)
    dmap = pairwise_distances(cand[["Seat_ID", "X", "Y"]])

    cand = cand.sort_values("Noise")  # seed order
    seat_list = list(cand["Seat_ID"].astype(int))

    # For fast brightness forward check
    brightness_by_seat = cand.set_index("Seat_ID")["Brightness"].astype(float).to_dict()
    sorted_bright = sorted(brightness_by_seat.values(), reverse=True)

    def best_possible_brightness_sum(current_sum: float, picked: int) -> float:
        """Upper bound on total brightness sum if we can add the brightest remaining seats."""
        top_needed = max(0, group_size - picked)
        return current_sum + sum(sorted_bright[:top_needed])

    for seed in seat_list:
        chosen = [seed]
        chosen_set = {seed}
        bsum = brightness_by_seat[seed]

        # priority queue over all remaining seats by incremental cost
        pq: List[Tuple[float, int]] = []
        for j in seat_list:
            if j == seed:
                continue
            inc = float(cand.set_index("Seat_ID").loc[j, "Noise"])
            # add λ*sum distances to current chosen set (just seed now)
            if (min(seed, j), max(seed, j)) in dmap:
                inc += lam_pair * dmap[(min(seed, j), max(seed, j))]
            heapq.heappush(pq, (inc, j))

        # greedy expansion
        while len(chosen) < group_size and pq:
            inc, j = heapq.heappop(pq)
            if j in chosen_set:
                continue

            # recompute incremental distance term vs current set
            inc_dist = 0.0
            for i in chosen:
                a, b = (i, j) if i < j else (j, i)
                inc_dist += dmap.get((a, b), 0)
            inc_true = float(cand.set_index("Seat_ID").loc[j, "Noise"]) + lam_pair * inc_dist

            # Forward feasibility check for brightness
            if best_possible_brightness_sum(bsum + brightness_by_seat[j], len(chosen) + 1) < group_size * brightness_min:
                continue

            # accept j
            chosen.append(j)
            chosen_set.add(j)
            bsum += brightness_by_seat[j]

            # Update queue with a small lazy strategy: push updated costs for remaining seats
            new_entries = []
            for k in seat_list:
                if k in chosen_set:
                    continue
                # add distance to newly chosen j
                a, b = (k, j) if k < j else (j, k)
                extra = lam_pair * dmap.get((a, b), 0)
                # old inc for k is stale; push a fresh candidate
                base = float(cand.set_index("Seat_ID").loc[k, "Noise"])
                # approximate: distance to all chosen so far needs full recompute; do it when popped (lazy)
                heapq.heappush(pq, (base + extra, k))

        if len(chosen) == group_size and bsum >= group_size * brightness_min:
            return {"status": "ok", "seat_ids": chosen}

    return {"status": "infeasible", "reason": "no feasible greedy selection"}

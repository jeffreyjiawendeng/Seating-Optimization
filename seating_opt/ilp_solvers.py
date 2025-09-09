# seating_opt/ilp_solvers.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Optional, List, Tuple

try:
    import pulp as pl
    _HAS_PULP = True
except Exception:
    _HAS_PULP = False

from .distance import build_pair_set
from .utils import validate_seats_df, validate_groups_df


class ILPNotAvailableError(RuntimeError):
    pass


# ---------- Per-group (myopic) ILP with λ·pairwise distance  ----------
def solve_group_pair_ilp(
    seats_df: pd.DataFrame,
    group_size: int,
    brightness_min: float,
    lam_pair: float = 0.3,
    mu_penalty: float = 0.0,                           # 0 for myopic; >0 for SR-PQ
    per_seat_penalty: Optional[Dict[int, float]] = None,
    dmax_pairs: int = 3,
    time_limit_sec: Optional[int] = None,
) -> Dict:
    """
    Per-group ILP (minimize noise + λ*pairwise distance + μ*scarcity).
    Constraints:
      sum x_i = k
      sum Brightness_i * x_i >= k * brightness_min
      pairwise linearization: y_ij <= x_i, y_ij <= x_j, y_ij >= x_i + x_j - 1
    """
    if not _HAS_PULP:
        raise ILPNotAvailableError("PuLP not installed; cannot run ILP.")

    validate_seats_df(seats_df)
    cand = seats_df[seats_df["Seat_Available"]].copy()
    if len(cand) < group_size:
        return {"status": "infeasible", "reason": "insufficient capacity"}

    # Build pair set (only among currently available seats)
    pairs, dmap = build_pair_set(cand[["Seat_ID", "X", "Y"]], max_dist=dmax_pairs)

    # Costs
    base_cost = cand.set_index("Seat_ID")["Noise"].astype(float).to_dict()
    if per_seat_penalty is None:
        per_seat_penalty = {int(i): 0.0 for i in cand["Seat_ID"].astype(int)}
    # seat coefficient: noise + μ * scarcity_penalty
    seat_coeff = {
        int(i): float(base_cost[int(i)] + mu_penalty * per_seat_penalty.get(int(i), 0.0))
        for i in cand["Seat_ID"].astype(int)
    }

    # Model
    m = pl.LpProblem("SeatPairILP", pl.LpMinimize)
    x = {i: pl.LpVariable(f"x_{i}", 0, 1, pl.LpBinary) for i in seat_coeff.keys()}
    y = {(i, j): pl.LpVariable(f"y_{i}_{j}", 0, 1, pl.LpBinary) for (i, j) in pairs}

    # Objective: sum noise + μ*penalty + λ*Σ d_ij y_ij
    m += pl.lpSum(seat_coeff[i] * x[i] for i in x) + lam_pair * pl.lpSum(
        dmap[(i, j)] * y[(i, j)] for (i, j) in pairs
    )

    # Count
    m += pl.lpSum(x[i] for i in x) == group_size

    # Brightness average lower bound
    b_map = cand.set_index("Seat_ID")["Brightness"].astype(float).to_dict()
    m += pl.lpSum(b_map[i] * x[i] for i in x) >= group_size * brightness_min

    # Linearization
    for (i, j) in pairs:
        m += y[(i, j)] <= x[i]
        m += y[(i, j)] <= x[j]
        m += y[(i, j)] >= x[i] + x[j] - 1

    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    m.solve(solver)

    status = pl.LpStatus[m.status]
    chosen = [i for i, v in x.items() if v.value() and v.value() > 0.5]
    obj = pl.value(m.objective) if status in ("Optimal", "Not Solved") else None
    return {"status": status, "seat_ids": chosen, "objective": obj, "model": m}
# ---------------------------------------------------------------------


# ---------------------- Global (gold) ILP ----------------------------
def solve_global_pair_ilp(
    seats_df: pd.DataFrame,
    groups_df: pd.DataFrame,
    lam_pair: float = 0.3,
    dmax_pairs: int = 3,
    time_limit_sec: Optional[int] = None,
) -> Dict:
    """
    Global ILP (knows the whole sequence).
    Variables: x_{i,g} seat i to group g ; y_{ij,g} pair activation for group g
    Objective: Σ_g ( Σ_i Noise_i x_{i,g} + λ Σ_{(i,j)} d_ij y_{ij,g} )
    Constraints: per-group count & brightness; per-group linearization; no double-booking Σ_g x_{i,g} ≤ 1
    """
    if not _HAS_PULP:
        raise ILPNotAvailableError("PuLP not installed; cannot run ILP.")

    validate_seats_df(seats_df)
    validate_groups_df(groups_df)

    cand = seats_df[seats_df["Seat_Available"]].copy()
    seat_ids = list(cand["Seat_ID"].astype(int))
    # Build pair set once globally (among all available seats)
    pairs, dmap = build_pair_set(cand[["Seat_ID", "X", "Y"]], max_dist=dmax_pairs)

    groups = groups_df.sort_values("Group_ID").copy()
    G = list(groups["Group_ID"].astype(int))

    # Heuristic bound: keep small for tractability
    if len(seat_ids) * len(G) > 25000:
        raise ValueError("Global ILP too large; reduce problem size.")

    b_map = cand.set_index("Seat_ID")["Brightness"].astype(float).to_dict()
    n_map = cand.set_index("Seat_ID")["Noise"].astype(float).to_dict()

    m = pl.LpProblem("GlobalSeatPairILP", pl.LpMinimize)
    x = {(i, g): pl.LpVariable(f"x_{i}_{g}", 0, 1, pl.LpBinary) for i in seat_ids for g in G}
    y = {(i, j, g): pl.LpVariable(f"y_{i}_{j}_{g}", 0, 1, pl.LpBinary) for (i, j) in pairs for g in G}

    # Objective
    m += (
        pl.lpSum(n_map[i] * x[(i, g)] for i in seat_ids for g in G)
        + lam_pair * pl.lpSum(dmap[(i, j)] * y[(i, j, g)] for (i, j) in pairs for g in G)
    )

    # Per-group constraints
    for _, row in groups.iterrows():
        g = int(row["Group_ID"])
        k = int(row["Group_Size"])
        Bmin = float(row["Brightness_Min"])

        # Count
        m += pl.lpSum(x[(i, g)] for i in seat_ids) == k
        # Brightness
        m += pl.lpSum(b_map[i] * x[(i, g)] for i in seat_ids) >= k * Bmin
        # Linearization per group
        for (i, j) in pairs:
            m += y[(i, j, g)] <= x[(i, g)]
            m += y[(i, j, g)] <= x[(j, g)]
            m += y[(i, j, g)] >= x[(i, g)] + x[(j, g)] - 1

    # No double-booking
    for i in seat_ids:
        m += pl.lpSum(x[(i, g)] for g in G) <= 1

    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    m.solve(solver)

    status = pl.LpStatus[m.status]
    assign = {g: [] for g in G}
    if status == "Optimal":
        for g in G:
            for i in seat_ids:
                v = x[(i, g)].value()
                if v and v > 0.5:
                    assign[g].append(i)
    return {"status": status, "assignments": assign, "model": m}
# ---------------------------------------------------------------------


# --------- Helper: try to make the world globally feasible -----------
def ensure_global_feasible(
    seats_df: pd.DataFrame,
    groups_df: pd.DataFrame,
    lam_pair: float = 0.3,
    dmax_pairs: int = 3,
    start_Bmin_cap: float = 80.0,
    min_Bmin_cap: float = 35.0,
    step: float = 2.0,
    time_limit_sec: Optional[int] = 30,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Iteratively lower the groups' brightness minima (capped) until the global ILP is feasible.
    Does not change group sizes or order. Returns (adjusted_groups_df, last_solution_dict).
    """
    g_adj = groups_df.copy()
    cap = start_Bmin_cap
    while cap >= min_Bmin_cap:
        g_adj["Brightness_Min"] = g_adj["Brightness_Min"].clip(upper=cap)
        try:
            sol = solve_global_pair_ilp(
                seats_df, g_adj, lam_pair=lam_pair, dmax_pairs=dmax_pairs, time_limit_sec=time_limit_sec
            )
        except ILPNotAvailableError as e:
            raise
        if sol["status"] == "Optimal":
            return g_adj, sol
        cap -= step
    # If still infeasible, return last attempt status
    return g_adj, sol
# ---------------------------------------------------------------------
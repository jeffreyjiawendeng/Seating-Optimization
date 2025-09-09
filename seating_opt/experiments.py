# seating_opt/experiments.py
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from typing import Dict, List

# from .sr_runner import run_stochastic_rolling, SRConfig
from .ilp_solvers import (
    solve_group_pair_ilp,
    solve_global_pair_ilp,
    ensure_global_feasible,
    ILPNotAvailableError,
)
from .greedy import greedy_pairwise
from .sketchrefine import sketchrefine_solver
from .utils import validate_seats_df, validate_groups_df, avg_pairwise_distance


# ---------------------- Evaluation helpers ---------------------------
def objective_value_for_assignment(
    seats_df: pd.DataFrame, seat_ids: List[int], lam_pair: float = 0.3
) -> float:
    """Avg noise + λ * avg pairwise distance for a selection (k is fixed)."""
    if not seat_ids:
        return 0.0  # Return 0 instead of inf for failed assignments
    try:
        sel = seats_df.set_index("Seat_ID").loc[seat_ids]
        avg_noise = float(sel["Noise"].mean())
        # pairwise average distance
        coords = seats_df.set_index("Seat_ID")[["X", "Y"]]
        avg_d = avg_pairwise_distance(seat_ids, coords)
        return avg_noise + lam_pair * avg_d
    except Exception:
        return 0.0  # Return 0 if calculation fails
    avg_d = avg_pairwise_distance(seat_ids, coords)
    return avg_noise + lam_pair * avg_d
# --------------------------------------------------------------------


def run_greedy_online(seats_df: pd.DataFrame, groups_df: pd.DataFrame, lam_pair: float) -> Dict:
    s = seats_df.copy()
    results = []
    for _, g in groups_df.sort_values("Group_ID").iterrows():
        sol = greedy_pairwise(s, int(g.Group_Size), float(g.Brightness_Min), lam_pair=lam_pair)
        if sol["status"] == "ok":
            s.loc[s["Seat_ID"].isin(sol["seat_ids"]), "Seat_Available"] = False
        results.append({"Group_ID": int(g.Group_ID), "status": sol["status"], "Seat_IDs": sol.get("seat_ids", [])})
    return {"assignments": results, "final_seats": s}


def run_sketchrefine_online(
    seats_df: pd.DataFrame, groups_df: pd.DataFrame, lam_pair: float, dmax_pairs: int
) -> Dict:
    s = seats_df.copy()
    results = []
    for _, g in groups_df.sort_values("Group_ID").iterrows():
        sol = sketchrefine_solver(
            s, int(g.Group_Size), float(g.Brightness_Min), lam_pair=lam_pair, dmax_pairs=dmax_pairs
        )
        if sol["status"] == "ok":
            s.loc[s["Seat_ID"].isin(sol["seat_ids"]), "Seat_Available"] = False
        results.append({"Group_ID": int(g.Group_ID), "status": sol["status"], "Seat_IDs": sol.get("seat_ids", [])})
    return {"assignments": results, "final_seats": s}


def run_myopic_ilp_online(
    seats_df: pd.DataFrame, groups_df: pd.DataFrame, lam_pair: float, dmax_pairs: int
) -> Dict:
    s = seats_df.copy()
    results = []
    for _, g in groups_df.sort_values("Group_ID").iterrows():
        sol = solve_group_pair_ilp(
            s, int(g.Group_Size), float(g.Brightness_Min), lam_pair=lam_pair, dmax_pairs=dmax_pairs
        )
        if sol["status"] in ("Optimal", "Not Solved"):
            s.loc[s["Seat_ID"].isin(sol["seat_ids"]), "Seat_Available"] = False
        results.append({"Group_ID": int(g.Group_ID), "status": sol["status"], "Seat_IDs": sol.get("seat_ids", [])})
    return {"assignments": results, "final_seats": s}


def summarize_sequence(seats_df: pd.DataFrame, results: List[Dict], lam_pair: float) -> Dict[str, float]:
    noises, dists, brs, success = [], [], [], 0
    coords = seats_df.set_index("Seat_ID")[["X", "Y"]]
    total_obj = 0.0
    successful_groups = 0
    
    for r in results:
        ids = r.get("Seat_IDs", [])
        if r.get("status") in ("ok", "Optimal") and ids:
            success += 1
            successful_groups += 1
            sel = seats_df.set_index("Seat_ID").loc[ids]
            noises.append(float(sel["Noise"].mean()))
            brs.append(float(sel["Brightness"].mean()))
            dists.append(avg_pairwise_distance(ids, coords))
            
            # Calculate objective for successful groups only
            obj_val = objective_value_for_assignment(seats_df, ids, lam_pair=lam_pair)
            if obj_val != float("inf"):
                total_obj += obj_val
    
    # Return finite values or reasonable defaults
    return {
        "success_rate": success / max(1, len(results)),
        "avg_noise": float(np.mean(noises)) if noises else 0.0,
        "avg_brightness": float(np.mean(brs)) if brs else 0.0,
        "avg_pairwise_distance": float(np.mean(dists)) if dists else 0.0,
        "cumulative_objective": total_obj if successful_groups > 0 else 0.0,
    }


# --------------- Unified runner for the single fixed world -----------
def run_all(
    seats_df: pd.DataFrame,
    groups_df: pd.DataFrame,
    lam_pair: float = 0.3,
    dmax_pairs: int = 3,
) -> Dict:
    """Run Greedy, Myopic ILP, SR-PQ (passive & optimistic), and Global ILP on the SAME world."""
    validate_seats_df(seats_df)
    validate_groups_df(groups_df)

    # Ensure global feasibility by softly capping Brightness_Min
    g_adj, gold_sol = ensure_global_feasible(
        seats_df, groups_df, lam_pair=lam_pair, dmax_pairs=dmax_pairs
    )

    # Global ILP objective (for regret baseline)
    gold_assign = gold_sol["assignments"] if gold_sol["status"] == "Optimal" else {}
    gold_cum = 0.0
    for g, ids in gold_assign.items():
        gold_cum += objective_value_for_assignment(seats_df, ids, lam_pair=lam_pair)

    # Greedy
    t0 = time.time()
    greedy_res = run_greedy_online(seats_df, g_adj, lam_pair=lam_pair)
    t_greedy = (time.time() - t0) * 1000.0
    greedy_sum = summarize_sequence(seats_df, greedy_res["assignments"], lam_pair)
    greedy_sum["runtime_ms"] = t_greedy
    greedy_sum["regret_vs_gold"] = greedy_sum["cumulative_objective"] - gold_cum

    # Myopic ILP
    t0 = time.time()
    ilp_res = run_myopic_ilp_online(seats_df, g_adj, lam_pair=lam_pair, dmax_pairs=dmax_pairs)
    t_ilp = (time.time() - t0) * 1000.0
    ilp_sum = summarize_sequence(seats_df, ilp_res["assignments"], lam_pair)
    ilp_sum["runtime_ms"] = t_ilp
    ilp_sum["regret_vs_gold"] = ilp_sum["cumulative_objective"] - gold_cum

    # SketchRefine
    t0 = time.time()
    sketch_res = run_sketchrefine_online(seats_df, g_adj, lam_pair=lam_pair, dmax_pairs=dmax_pairs)
    t_sketch = (time.time() - t0) * 1000.0
    sketch_sum = summarize_sequence(seats_df, sketch_res["assignments"], lam_pair)
    sketch_sum["runtime_ms"] = t_sketch
    sketch_sum["regret_vs_gold"] = sketch_sum["cumulative_objective"] - gold_cum

    # # SR-PQ Passive
    # sr_cfg_p = SRConfig(lambda_pair=lam_pair, mu_penalty=0.7, dmax_pairs=dmax_pairs, weighting="passive")
    # t0 = time.time()
    # sr_p = run_stochastic_rolling(seats_df, g_adj, sr_cfg_p)
    # t_srp = (time.time() - t0) * 1000.0
    # srp_sum = summarize_sequence(seats_df, sr_p["assignments"], lam_pair)
    # srp_sum["runtime_ms"] = t_srp
    # srp_sum["regret_vs_gold"] = srp_sum["cumulative_objective"] - gold_cum

    # # SR-PQ Optimistic
    # sr_cfg_o = SRConfig(lambda_pair=lam_pair, mu_penalty=0.7, dmax_pairs=dmax_pairs, weighting="optimistic", optimistic_tau=0.7)
    # t0 = time.time()
    # sr_o = run_stochastic_rolling(seats_df, g_adj, sr_cfg_o)
    # t_sro = (time.time() - t0) * 1000.0
    # sro_sum = summarize_sequence(seats_df, sr_o["assignments"], lam_pair)
    # sro_sum["runtime_ms"] = t_sro
    # sro_sum["regret_vs_gold"] = sro_sum["cumulative_objective"] - gold_cum

    return {
        "world": {"groups_used": g_adj, "gold_status": gold_sol["status"], "gold_cum_obj": gold_cum},
        "greedy": {"results": greedy_res["assignments"], "summary": greedy_sum},
        "myopic_ilp": {"results": ilp_res["assignments"], "summary": ilp_sum},
        "sketchrefine": {"results": sketch_res["assignments"], "summary": sketch_sum},
        # "sr_pq_passive": {"results": sr_p["assignments"], "summary": srp_sum},
        # "sr_pq_optimistic": {"results": sr_o["assignments"], "summary": sro_sum},
    }
# --------------------------------------------------------------------
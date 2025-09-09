# seating_opt/sketchrefine.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import pulp as pl
    _HAS_PULP = True
except Exception:
    _HAS_PULP = False

from .distance import pairwise_distances, build_pair_set
from .utils import validate_seats_df
from .ilp_solvers import ILPNotAvailableError


def create_table_representatives(seats_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create representative seats for each table.
    
    Args:
        seats_df: DataFrame with seat data including Table_ID
    
    Returns:
        Tuple of (representatives_df, table_to_seats_mapping)
    """
    validate_seats_df(seats_df)
    available = seats_df[seats_df["Seat_Available"]].copy()
    
    representatives = []
    table_to_seats = {}
    
    for table_id in available["Table_ID"].unique():
        table_seats = available[available["Table_ID"] == table_id].copy()
        
        if len(table_seats) == 0:
            continue
            
        # Store actual seats for this table
        table_to_seats[table_id] = table_seats["Seat_ID"].tolist()
        
        # Calculate representative attributes (averages)
        avg_brightness = table_seats["Brightness"].mean()
        avg_noise = table_seats["Noise"].mean()
        avg_x = table_seats["X"].mean() 
        avg_y = table_seats["Y"].mean()
        
        representatives.append({
            "Table_ID": table_id,
            "Representative_Brightness": avg_brightness,
            "Representative_Noise": avg_noise,
            "Representative_X": avg_x,
            "Representative_Y": avg_y,
            "Seat_Count": len(table_seats)
        })
    
    reps_df = pd.DataFrame(representatives)
    return reps_df, table_to_seats


def solve_sketch_ilp(
    representatives_df: pd.DataFrame,
    group_size: int,
    brightness_min: float,
    lam_pair: float = 0.3,
    dmax_pairs: int = 3,
    time_limit_sec: Optional[int] = None,
) -> Dict:
    """
    Solve the sketch ILP over table representatives.
    
    Each table can be selected multiple times (up to group_size).
    """
    if not _HAS_PULP:
        raise ILPNotAvailableError("PuLP not installed; cannot run ILP.")
    
    if len(representatives_df) == 0:
        return {"status": "infeasible", "reason": "no available tables"}
    
    # Build distances between table representatives
    table_coords = representatives_df[["Table_ID", "Representative_X", "Representative_Y"]].copy()
    table_coords.columns = ["Seat_ID", "X", "Y"]  # Rename for compatibility
    
    pairs, dmap = build_pair_set(table_coords, max_dist=dmax_pairs)
    
    # Model
    m = pl.LpProblem("SketchILP", pl.LpMinimize)
    
    # Variables: x[table_id][count] = 1 if we select 'count' seats from table_id
    max_repeat = min(group_size, representatives_df["Seat_Count"].max())
    x = {}
    for _, row in representatives_df.iterrows():
        table_id = row["Table_ID"]
        max_from_table = min(group_size, row["Seat_Count"])
        for count in range(max_from_table + 1):  # 0 to max_from_table
            x[(table_id, count)] = pl.LpVariable(f"x_{table_id}_{count}", 0, 1, pl.LpBinary)
    
    # Pairwise variables for linearization
    y = {}
    for (i, j) in pairs:
        table_i, table_j = int(i), int(j)  # These are Table_IDs
        for count_i in range(1, max_repeat + 1):
            for count_j in range(1, max_repeat + 1):
                if (table_i, count_i) in x and (table_j, count_j) in x:
                    # y represents interaction between count_i seats from table_i and count_j seats from table_j
                    # The cost will be proportional to count_i * count_j * distance
                    y[(table_i, count_i, table_j, count_j)] = pl.LpVariable(
                        f"y_{table_i}_{count_i}_{table_j}_{count_j}", 0, 1, pl.LpBinary
                    )
    
    # Objective: minimize total noise + λ * pairwise distances
    noise_term = pl.lpSum(
        representatives_df.set_index("Table_ID").loc[table_id, "Representative_Noise"] * count * x[(table_id, count)]
        for (table_id, count) in x.keys() if count > 0
    )
    
    distance_term = pl.lpSum(
        dmap.get((min(table_i, table_j), max(table_i, table_j)), 0) * count_i * count_j * y[(table_i, count_i, table_j, count_j)]
        for (table_i, count_i, table_j, count_j) in y.keys()
    )
    
    m += noise_term + lam_pair * distance_term
    
    # Constraints:
    # 1. Each table can have at most one count selected
    for table_id in representatives_df["Table_ID"]:
        available_counts = [count for (tid, count) in x.keys() if tid == table_id]
        m += pl.lpSum(x[(table_id, count)] for count in available_counts) <= 1
    
    # 2. Total seats selected must equal group_size
    m += pl.lpSum(count * x[(table_id, count)] for (table_id, count) in x.keys()) == group_size
    
    # 3. Brightness constraint
    brightness_term = pl.lpSum(
        representatives_df.set_index("Table_ID").loc[table_id, "Representative_Brightness"] * count * x[(table_id, count)]
        for (table_id, count) in x.keys()
    )
    m += brightness_term >= group_size * brightness_min
    
    # 4. Linearization constraints for pairwise terms
    for (table_i, count_i, table_j, count_j) in y.keys():
        if (table_i, count_i) in x and (table_j, count_j) in x:
            m += y[(table_i, count_i, table_j, count_j)] <= x[(table_i, count_i)]
            m += y[(table_i, count_i, table_j, count_j)] <= x[(table_j, count_j)]
            m += y[(table_i, count_i, table_j, count_j)] >= x[(table_i, count_i)] + x[(table_j, count_j)] - 1
    
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    m.solve(solver)
    
    status = pl.LpStatus[m.status]
    if status not in ("Optimal", "Not Solved"):
        return {"status": "infeasible", "reason": f"ILP status: {status}"}
    
    # Extract solution: which tables and how many seats from each
    sketch_solution = []
    for (table_id, count), var in x.items():
        if var.value() and var.value() > 0.5 and count > 0:
            sketch_solution.append((int(table_id), int(count)))  # Convert to int
    
    obj = pl.value(m.objective) if status in ("Optimal", "Not Solved") else None
    return {
        "status": status,
        "sketch_solution": sketch_solution,
        "objective": obj,
        "model": m
    }


def refine_solution(
    seats_df: pd.DataFrame,
    table_to_seats: Dict,
    sketch_solution: List[Tuple[int, int]],  # (table_id, count)
    representatives_df: pd.DataFrame,
    group_size: int,
    brightness_min: float,
    lam_pair: float = 0.3,
    dmax_pairs: int = 3,
    time_limit_sec: Optional[int] = None,
) -> Dict:
    """
    Refine the sketch solution by replacing table representatives with actual seats.
    
    Iteratively refine each table selection in the sketch solution.
    """
    if not _HAS_PULP:
        raise ILPNotAvailableError("PuLP not installed; cannot run ILP.")
    
    # Start with the sketch solution
    current_solution = sketch_solution.copy()
    final_seat_ids = []
    
    # Process each table in the sketch solution
    for stage_idx, (target_table_id, target_count) in enumerate(sketch_solution):
        
        # Build the refinement ILP
        m = pl.LpProblem(f"RefineStage_{stage_idx}", pl.LpMinimize)
        
        # Get actual seats from the target table
        target_seats = seats_df[
            (seats_df["Table_ID"] == target_table_id) & 
            (seats_df["Seat_Available"] == True)
        ].copy()
        
        if len(target_seats) < target_count:
            return {
                "status": "infeasible", 
                "reason": f"Table {target_table_id} doesn't have {target_count} available seats"
            }
        
        # Collect all seats/representatives for this stage
        all_candidates = []
        
        # Add actual seats from target table
        for _, seat_row in target_seats.iterrows():
            all_candidates.append({
                "ID": f"seat_{seat_row['Seat_ID']}",
                "Type": "seat",
                "Seat_ID": seat_row["Seat_ID"],
                "Table_ID": seat_row["Table_ID"],
                "Brightness": seat_row["Brightness"],
                "Noise": seat_row["Noise"], 
                "X": seat_row["X"],
                "Y": seat_row["Y"],
                "Max_Select": 1
            })
        
        # Add table representatives for remaining tables in sketch
        for other_table_id, other_count in current_solution:
            if other_table_id != target_table_id:
                rep_row = representatives_df[representatives_df["Table_ID"] == other_table_id].iloc[0]
                all_candidates.append({
                    "ID": f"table_{other_table_id}",
                    "Type": "table",
                    "Seat_ID": None,
                    "Table_ID": other_table_id,
                    "Brightness": rep_row["Representative_Brightness"],
                    "Noise": rep_row["Representative_Noise"],
                    "X": rep_row["Representative_X"],
                    "Y": rep_row["Representative_Y"],
                    "Max_Select": other_count
                })
        
        candidates_df = pd.DataFrame(all_candidates)
        
        # Build pairwise distances
        coord_df = candidates_df[["ID", "X", "Y"]].copy()
        coord_df.columns = ["Seat_ID", "X", "Y"]
        pairs, dmap = build_pair_set(coord_df, max_dist=dmax_pairs)
        
        # Variables
        x = {}
        for _, cand in candidates_df.iterrows():
            cand_id = cand["ID"]
            max_sel = cand["Max_Select"]
            for count in range(max_sel + 1):
                x[(cand_id, count)] = pl.LpVariable(f"x_{cand_id}_{count}", 0, 1, pl.LpBinary)
        
        # Pairwise variables
        y = {}
        for (id_i, id_j) in pairs:
            cand_i = candidates_df[candidates_df["ID"] == id_i].iloc[0]
            cand_j = candidates_df[candidates_df["ID"] == id_j].iloc[0]
            for count_i in range(1, cand_i["Max_Select"] + 1):
                for count_j in range(1, cand_j["Max_Select"] + 1):
                    if (id_i, count_i) in x and (id_j, count_j) in x:
                        y[(id_i, count_i, id_j, count_j)] = pl.LpVariable(
                            f"y_{id_i}_{count_i}_{id_j}_{count_j}", 0, 1, pl.LpBinary
                        )
        
        # Objective
        noise_term = pl.lpSum(
            candidates_df.set_index("ID").loc[cand_id, "Noise"] * count * x[(cand_id, count)]
            for (cand_id, count) in x.keys() if count > 0
        )
        
        distance_term = pl.lpSum(
            dmap.get((min(id_i, id_j), max(id_i, id_j)), 0) * count_i * count_j * y[(id_i, count_i, id_j, count_j)]
            for (id_i, count_i, id_j, count_j) in y.keys()
        )
        
        m += noise_term + lam_pair * distance_term
        
        # Constraints
        # 1. Each candidate has at most one count selected
        for cand_id in candidates_df["ID"].unique():
            cand_row = candidates_df[candidates_df["ID"] == cand_id].iloc[0]
            available_counts = [count for (cid, count) in x.keys() if cid == cand_id]
            m += pl.lpSum(x[(cand_id, count)] for count in available_counts) <= 1
        
        # 2. Must select exactly target_count from target table
        target_candidates = candidates_df[candidates_df["Type"] == "seat"]["ID"].tolist()
        m += pl.lpSum(
            count * x[(cand_id, count)] 
            for cand_id in target_candidates 
            for count in range(1, 2) if (cand_id, count) in x  # seats have max_select=1
        ) == target_count
        
        # 3. Must select exact counts from other tables (from sketch)
        for other_table_id, other_count in current_solution:
            if other_table_id != target_table_id:
                table_cand_id = f"table_{other_table_id}"
                if table_cand_id in candidates_df["ID"].values:
                    m += x[(table_cand_id, other_count)] == 1
        
        # 4. Brightness constraint
        brightness_term = pl.lpSum(
            candidates_df.set_index("ID").loc[cand_id, "Brightness"] * count * x[(cand_id, count)]
            for (cand_id, count) in x.keys()
        )
        m += brightness_term >= group_size * brightness_min
        
        # 5. Linearization constraints
        for (id_i, count_i, id_j, count_j) in y.keys():
            if (id_i, count_i) in x and (id_j, count_j) in x:
                m += y[(id_i, count_i, id_j, count_j)] <= x[(id_i, count_i)]
                m += y[(id_i, count_i, id_j, count_j)] <= x[(id_j, count_j)]
                m += y[(id_i, count_i, id_j, count_j)] >= x[(id_i, count_i)] + x[(id_j, count_j)] - 1
        
        # Solve
        solver = pl.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
        m.solve(solver)
        
        status = pl.LpStatus[m.status]
        if status not in ("Optimal", "Not Solved"):
            return {"status": "infeasible", "reason": f"Refine stage {stage_idx} failed: {status}"}
        
        # Extract selected seats from target table for this stage
        stage_seats = []
        for (cand_id, count), var in x.items():
            if var.value() and var.value() > 0.5 and count > 0:
                if cand_id.startswith("seat_"):
                    seat_id = int(cand_id.replace("seat_", ""))
                    stage_seats.extend([seat_id] * count)
        
        final_seat_ids.extend(stage_seats)
        
        # Update current_solution by removing the refined table
        current_solution = [(tid, cnt) for tid, cnt in current_solution if tid != target_table_id]
    
    obj = None  # Could compute from final seats if needed
    return {
        "status": "ok",
        "seat_ids": final_seat_ids,
        "objective": obj
    }


def sketchrefine_solver(
    seats_df: pd.DataFrame,
    group_size: int,
    brightness_min: float,
    lam_pair: float = 0.3,
    dmax_pairs: int = 3,
    time_limit_sec: Optional[int] = None,
) -> Dict:
    """
    Main SketchRefine algorithm.
    
    Steps:
    1. Create table representatives
    2. Solve sketch ILP over representatives  
    3. Refine solution by replacing representatives with actual seats
    """
    try:
        validate_seats_df(seats_df)
        available = seats_df[seats_df["Seat_Available"]].copy()
        
        if len(available) < group_size:
            return {"status": "infeasible", "reason": "insufficient capacity"}
        
        # Step 1: Create table representatives
        representatives_df, table_to_seats = create_table_representatives(seats_df)
        
        if len(representatives_df) == 0:
            return {"status": "infeasible", "reason": "no available tables"}
        
        # Step 2: Solve sketch ILP
        sketch_result = solve_sketch_ilp(
            representatives_df, 
            group_size, 
            brightness_min,
            lam_pair=lam_pair,
            dmax_pairs=dmax_pairs,
            time_limit_sec=time_limit_sec
        )
        
        if sketch_result["status"] != "Optimal":
            return {
                "status": "infeasible", 
                "reason": f"Sketch phase failed: {sketch_result.get('reason', sketch_result['status'])}"
            }
        
        sketch_solution = sketch_result["sketch_solution"]
        
        # Step 3: Refine solution
        refine_result = refine_solution(
            seats_df,
            table_to_seats,
            sketch_solution,
            representatives_df,
            group_size,
            brightness_min,
            lam_pair=lam_pair,
            dmax_pairs=dmax_pairs,
            time_limit_sec=time_limit_sec
        )
        
        return refine_result
        
    except Exception as e:
        return {"status": "error", "reason": f"SketchRefine error: {str(e)}"}

import numpy as np
import pandas as pd
import pulp
from typing import List, Dict, Union, Tuple

def solve_ilp(
    sub: np.ndarray,                     # (N, D) data matrix
    size_: int,                          # exact total picks: Σ x_i == size_
    rep: int,                            # per-item max copies (1 for 0/1)
    obj: Dict[str, Union[int, str]],     # {'attr': j, 'pref': 'MIN'|'MAX'}
    cons: List[Dict[str, Union[int, float, str]]],  # [{'attr': a, 'pref': 'MIN'|'MAX', 'bound': b}, ...]
    verbose: bool = True
) -> Tuple[List[int], List[int]]:
    """
    Returns: (picked_indices, per_item_counts)
    Semantics:
      - Objective: 'MIN' => LpMinimize, 'MAX' => LpMaximize on attribute obj['attr'].
      - Constraint with 'MIN' => Σ attr[a]·x <= bound (upper bound).
      - Constraint with 'MAX' => Σ attr[a]·x >= bound (lower bound).
    """
    N, D = sub.shape

    # normalize objective fields
    j = int(obj['attr'])
    obj_pref = str(obj['pref']).upper()
    assert obj_pref in ('MIN', 'MAX'), "obj.pref must be 'MIN' or 'MAX'"
    assert 0 <= j < D, "obj.attr out of range"

    # normalize constraint fields
    norm_cons = []
    for c in cons:
        a = int(c['attr'])
        pref = str(c['pref']).upper()
        b = float(c['bound'])
        assert pref in ('MIN', 'MAX'), "cons.pref must be 'MIN' or 'MAX'"
        assert 0 <= a < D, "cons.attr out of range"
        norm_cons.append({'attr': a, 'pref': pref, 'bound': b})

    # decision variables: integers 0..rep
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=rep, cat=pulp.LpInteger) for i in range(N)]

    # model and cardinality
    sense = pulp.LpMinimize if obj_pref == 'MIN' else pulp.LpMaximize
    prob = pulp.LpProblem("package_selection", sense)
    prob += pulp.lpSum(x) == size_

    # constraints (MIN => <=, MAX => >=)
    cons_exprs = []
    for c in norm_cons:
        expr = pulp.lpSum(sub[i, c['attr']] * x[i] for i in range(N))
        cons_exprs.append(expr)
        if c['pref'] == 'MIN':
            prob += expr <= c['bound']
        else:
            prob += expr >= c['bound']

    # objective
    obj_expr = pulp.lpSum(sub[i, j] * x[i] for i in range(N))
    prob += obj_expr

    # solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_name = pulp.LpStatus[prob.status]

    vals = [int(v.value()) for v in x]
    picked = [i for i, v in enumerate(vals) if v > 0]

    if verbose:
        print(f"Status: {status_name}")
        if status_name != "Optimal":
            print("No optimal solution found (infeasible or solver stopped).")
        print(f"Objective value: {pulp.value(prob.objective):.6f}")
        print(f"Total picked: {sum(vals)}  (target size={size_})")
        for k, (c, e) in enumerate(zip(norm_cons, cons_exprs)):
            realized = float(e.value())
            arrow = "<=" if c['pref'] == 'MIN' else ">="
            print(f"Constraint {k}: Σ attr[{c['attr']}]·x {arrow} {c['bound']}  | realized={realized:.6f}")
        reps = {i: v for i, v in enumerate(vals) if v > 0}
        print(f"Picked indices (counts): {reps}")

    return picked, vals

def load_seat_data():
    """Load and prepare seat data for ILP."""
    seats_df = pd.read_csv("src/seats.csv")
    students_df = pd.read_csv("src/students.csv")
    
    # Filter only available seats
    available_seats = seats_df[seats_df['Seat_Available'] == True].copy()
    
    # Create data matrix with relevant attributes
    # Columns: [Brightness, Noise, Room_ID, Table_ID]
    data_matrix = available_seats[['Brightness', 'Noise', 'Room_ID', 'Table_ID']].values
    
    print(f"Loaded {len(available_seats)} available seats")
    print(f"Data matrix shape: {data_matrix.shape}")
    
    return data_matrix, available_seats, students_df

if __name__ == "__main__":
    # Load real seat data
    sub, seats_df, students_df = load_seat_data()
    
    # Example: Select 5 seats
    # Constraints: 
    # - Average brightness >= 70 (attr 0, MAX preference)
    # - Average noise <= 30 (attr 1, MIN preference)
    # Objective: Minimize average noise (attr 1, MIN preference)
    
    group_size = 5
    min_brightness = 70
    max_noise = 30
    
    # Convert to total bounds (multiply by group_size)
    cons = [
        {'attr': 0, 'pref': 'MAX', 'bound': min_brightness * group_size},  # Total brightness >= 70*5
        {'attr': 1, 'pref': 'MIN', 'bound': max_noise * group_size}        # Total noise <= 30*5
    ]
    
    obj = {'attr': 1, 'pref': 'MIN'}  # Minimize total noise
    
    picked, counts = solve_ilp(
        sub=sub, 
        size_=group_size, 
        rep=1,  # Each seat can only be selected once
        obj=obj, 
        cons=cons, 
        verbose=True
    )
    
    if picked:
        print("\nSelected seats:")
        selected_seats = seats_df.iloc[picked]
        for idx, (_, seat) in enumerate(selected_seats.iterrows()):
            print(f"  Seat {seat['Seat_ID']}: Table {seat['Table_ID']}, Room {seat['Room_ID']}, "
                  f"Brightness={seat['Brightness']}, Noise={seat['Noise']}")
        
        print(f"\nGroup statistics:")
        print(f"  Average brightness: {selected_seats['Brightness'].mean():.1f}")
        print(f"  Average noise: {selected_seats['Noise'].mean():.1f}")

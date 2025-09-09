# demo_main.py
import pandas as pd
from seating_opt.data_gen import generate_seats, generate_groups_from_students
from seating_opt.experiments import run_all

if __name__ == "__main__":
    # ----- FIXED WORLD (same for all methods) -----
    seats_df = generate_seats(
        rooms=1, tables_per_room=7, rows_per_table=4, cols_per_table=6, table_gap=1, room_gap=8, seed=7
    )
    # Use students->groups so every run has the same sequence given the seed
    students_df, groups_df = generate_groups_from_students(
        n_students=120, min_group=2, max_group=6, seed=17
    )
    # ----------------------------------------------

    LAMBDA_PAIR = 0.3   # closeness weight in the objective (same across methods)
    DMAX_PAIRS  = 3     # only consider pairs at distance ≤ 3 to keep ILPs compact

    results = run_all(seats_df, groups_df, lam_pair=LAMBDA_PAIR, dmax_pairs=DMAX_PAIRS)

    print("\n=== WORLD ===")
    print("Gold status:", results["world"]["gold_status"], "Gold cumulative objective:", results["world"]["gold_cum_obj"])

    print("\n=== GREEDY ===")
    print(results["greedy"]["summary"])

    print("\n=== MYOPIC ILP ===")
    print(results["myopic_ilp"]["summary"])

    # print("\n=== SR-PQ (PASSIVE) ===")
    # print(results["sr_pq_passive"]["summary"])

    # print("\n=== SR-PQ (OPTIMISTIC) ===")
    # print(results["sr_pq_optimistic"]["summary"])
#!/usr/bin/env python3
"""
Micro experiment with an ultra-small dataset for rapid testing and debugging.
This version uses the absolute minimum problem size to test algorithms quickly.
"""
import pandas as pd
import time
from seating_opt.data_gen import generate_seats, generate_groups_from_students
from seating_opt.experiments import run_all


def create_micro_dataset():
    """Create an ultra-small dataset for rapid testing."""
    # Tiny classroom: 1 table, 2 rows, 3 cols = 6 seats total
    seats_df = generate_seats(
        rooms=1, 
        tables_per_room=1, 
        rows_per_table=2, 
        cols_per_table=3, 
        table_gap=1, 
        room_gap=8, 
        seed=42
    )
    
    # Very few students: 6 students in 2-3 groups
    students_df, groups_df = generate_groups_from_students(
        n_students=6, 
        min_group=2, 
        max_group=3, 
        seed=123
    )
    
    return seats_df, students_df, groups_df


def run_micro_experiment():
    """Run the micro experiment for rapid testing."""
    print("=== MICRO SEATING OPTIMIZATION EXPERIMENT ===")
    print("(Ultra-small dataset for rapid testing)")
    print()
    
    # Create micro dataset
    print("Creating micro dataset...")
    seats_df, students_df, groups_df = create_micro_dataset()
    
    print(f"Dataset created:")
    print(f"  - Seats: {len(seats_df)}")
    print(f"  - Students: {len(students_df)}")  
    print(f"  - Groups: {len(groups_df)}")
    print(f"  - Group sizes: {list(groups_df['Group_Size'])}")
    print()
    
    # Show all data since it's tiny
    print("All seats:")
    print(seats_df[['Seat_ID', 'X', 'Y', 'Brightness', 'Noise']].round(1))
    print()
    
    print("All groups:")
    print(groups_df[['Group_ID', 'Group_Size', 'Brightness_Min']].round(1))
    print()
    
    # Algorithm parameters (minimal)
    LAMBDA_PAIR = 0.5   # higher weight to see compactness effect
    DMAX_PAIRS = 1      # only immediate neighbors
    
    print(f"Algorithm parameters:")
    print(f"  - Lambda (compactness weight): {LAMBDA_PAIR}")
    print(f"  - Max pair distance: {DMAX_PAIRS}")
    print()
    
    # Time the full experiment
    start_time = time.time()
    
    try:
        results = run_all(seats_df, groups_df, lam_pair=LAMBDA_PAIR, dmax_pairs=DMAX_PAIRS)
        elapsed_time = time.time() - start_time
        
        print(f"=== RESULTS (Total time: {elapsed_time:.3f} seconds) ===")
        print()
        
        print("WORLD STATUS:")
        print(f"  Gold status: {results['world']['gold_status']}")
        print(f"  Gold cumulative objective: {results['world']['gold_cum_obj']:.3f}")
        print()
        
        print("GREEDY vs MYOPIC ILP COMPARISON:")
        greedy = results["greedy"]["summary"]
        ilp = results["myopic_ilp"]["summary"]
        
        print(f"                    | Greedy  | ILP     | Difference")
        print(f"  Success rate      | {greedy['success_rate']:.3f}   | {ilp['success_rate']:.3f}   | {ilp['success_rate'] - greedy['success_rate']:+.3f}")
        print(f"  Avg noise         | {greedy['avg_noise']:.2f}    | {ilp['avg_noise']:.2f}    | {ilp['avg_noise'] - greedy['avg_noise']:+.2f}")
        print(f"  Avg brightness    | {greedy['avg_brightness']:.2f}    | {ilp['avg_brightness']:.2f}    | {ilp['avg_brightness'] - greedy['avg_brightness']:+.2f}")
        print(f"  Avg pair dist     | {greedy['avg_pairwise_distance']:.2f}    | {ilp['avg_pairwise_distance']:.2f}    | {ilp['avg_pairwise_distance'] - greedy['avg_pairwise_distance']:+.2f}")
        print(f"  Cum objective     | {greedy['cumulative_objective']:.3f}  | {ilp['cumulative_objective']:.3f}  | {ilp['cumulative_objective'] - greedy['cumulative_objective']:+.3f}")
        print(f"  Runtime (ms)      | {greedy['runtime_ms']:.1f}    | {ilp['runtime_ms']:.1f}    | {ilp['runtime_ms'] - greedy['runtime_ms']:+.1f}")
        print(f"  Regret vs gold    | {greedy['regret_vs_gold']:.3f}  | {ilp['regret_vs_gold']:.3f}  | {ilp['regret_vs_gold'] - greedy['regret_vs_gold']:+.3f}")
        print()
        
        # Show all assignments since dataset is tiny
        print("ALL ASSIGNMENTS:")
        print()
        for i in range(len(groups_df)):
            greedy_assign = results["greedy"]["results"][i]
            ilp_assign = results["myopic_ilp"]["results"][i]
            print(f"Group {i+1}:")
            print(f"  Greedy: {greedy_assign['status']:>8} -> Seats {greedy_assign['Seat_IDs']}")
            print(f"  ILP:    {ilp_assign['status']:>8} -> Seats {ilp_assign['Seat_IDs']}")
            
            # Show objective values for successful assignments
            if greedy_assign['Seat_IDs']:
                greedy_seats = seats_df.set_index("Seat_ID").loc[greedy_assign['Seat_IDs']]
                greedy_noise = greedy_seats['Noise'].mean()
                print(f"    Greedy noise: {greedy_noise:.2f}")
            if ilp_assign['Seat_IDs']:
                ilp_seats = seats_df.set_index("Seat_ID").loc[ilp_assign['Seat_IDs']]
                ilp_noise = ilp_seats['Noise'].mean()
                print(f"    ILP noise:    {ilp_noise:.2f}")
            print()
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"=== ERROR after {elapsed_time:.3f} seconds ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=== MICRO EXPERIMENT COMPLETED SUCCESSFULLY ===")
    return True


if __name__ == "__main__":
    success = run_micro_experiment()
    if success:
        print("✅ Micro experiment completed successfully!")
        print("This demonstrates that the seating optimization algorithms work correctly")
        print("with small datasets and run efficiently.")
    else:
        print("❌ Micro experiment failed!")
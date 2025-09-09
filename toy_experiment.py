#!/usr/bin/env python3
"""
Toy experiment with a small dataset to demonstrate the seating optimization algorithms.
This version uses a much smaller problem size to avoid performance issues.
"""
import pandas as pd
import time
from seating_opt.data_gen import generate_seats, generate_groups_from_students
from seating_opt.experiments import run_all


def create_toy_dataset():
    """Create a small toy dataset that runs quickly."""
    # Small classroom: 2 tables, 3 rows, 4 cols = 24 seats total
    seats_df = generate_seats(
        rooms=1, 
        tables_per_room=2, 
        rows_per_table=3, 
        cols_per_table=4, 
        table_gap=1, 
        room_gap=8, 
        seed=7
    )
    
    # Small number of students: 18 students in 4-6 groups
    students_df, groups_df = generate_groups_from_students(
        n_students=18, 
        min_group=2, 
        max_group=5, 
        seed=17
    )
    
    return seats_df, students_df, groups_df


def run_toy_experiment():
    """Run the toy experiment and time each component."""
    print("=== TOY SEATING OPTIMIZATION EXPERIMENT ===")
    print()
    
    # Create toy dataset
    print("Creating toy dataset...")
    seats_df, students_df, groups_df = create_toy_dataset()
    
    print(f"Dataset created:")
    print(f"  - Seats: {len(seats_df)}")
    print(f"  - Students: {len(students_df)}")  
    print(f"  - Groups: {len(groups_df)}")
    print(f"  - Group sizes: {list(groups_df['Group_Size'])}")
    print()
    
    # Show some sample data
    print("Sample seats (first 5):")
    print(seats_df[['Seat_ID', 'X', 'Y', 'Brightness', 'Noise']].head())
    print()
    
    print("Groups:")
    print(groups_df[['Group_ID', 'Group_Size', 'Brightness_Min']])
    print()
    
    # Algorithm parameters (same as demo but with smaller pair distance limit)
    LAMBDA_PAIR = 0.3   # closeness weight in the objective
    DMAX_PAIRS = 2      # reduced from 3 to keep problem smaller
    
    print(f"Algorithm parameters:")
    print(f"  - Lambda (compactness weight): {LAMBDA_PAIR}")
    print(f"  - Max pair distance: {DMAX_PAIRS}")
    print()
    
    # Time the full experiment
    start_time = time.time()
    
    try:
        results = run_all(seats_df, groups_df, lam_pair=LAMBDA_PAIR, dmax_pairs=DMAX_PAIRS)
        elapsed_time = time.time() - start_time
        
        print(f"=== RESULTS (Total time: {elapsed_time:.2f} seconds) ===")
        print()
        
        print("WORLD STATUS:")
        print(f"  Gold status: {results['world']['gold_status']}")
        print(f"  Gold cumulative objective: {results['world']['gold_cum_obj']:.3f}")
        print()
        
        print("GREEDY METHOD:")
        greedy_summary = results["greedy"]["summary"]
        print(f"  Success rate: {greedy_summary['success_rate']:.3f}")
        print(f"  Avg noise: {greedy_summary['avg_noise']:.2f}")
        print(f"  Avg brightness: {greedy_summary['avg_brightness']:.2f}")
        print(f"  Avg pairwise distance: {greedy_summary['avg_pairwise_distance']:.2f}")
        print(f"  Cumulative objective: {greedy_summary['cumulative_objective']:.3f}")
        print(f"  Runtime: {greedy_summary['runtime_ms']:.1f} ms")
        print(f"  Regret vs gold: {greedy_summary['regret_vs_gold']:.3f}")
        print()
        
        print("MYOPIC ILP METHOD:")
        ilp_summary = results["myopic_ilp"]["summary"]
        print(f"  Success rate: {ilp_summary['success_rate']:.3f}")
        print(f"  Avg noise: {ilp_summary['avg_noise']:.2f}")
        print(f"  Avg brightness: {ilp_summary['avg_brightness']:.2f}")
        print(f"  Avg pairwise distance: {ilp_summary['avg_pairwise_distance']:.2f}")
        print(f"  Cumulative objective: {ilp_summary['cumulative_objective']:.3f}")
        print(f"  Runtime: {ilp_summary['runtime_ms']:.1f} ms")
        print(f"  Regret vs gold: {ilp_summary['regret_vs_gold']:.3f}")
        print()
        
        # Detailed assignment results
        print("DETAILED ASSIGNMENTS:")
        print()
        print("Greedy assignments:")
        for assignment in results["greedy"]["results"][:3]:  # Show first 3
            print(f"  Group {assignment['Group_ID']}: {assignment['status']} -> Seats {assignment['Seat_IDs']}")
        if len(results["greedy"]["results"]) > 3:
            print(f"  ... and {len(results['greedy']['results']) - 3} more groups")
        print()
        
        print("Myopic ILP assignments:")
        for assignment in results["myopic_ilp"]["results"][:3]:  # Show first 3
            print(f"  Group {assignment['Group_ID']}: {assignment['status']} -> Seats {assignment['Seat_IDs']}")
        if len(results["myopic_ilp"]["results"]) > 3:
            print(f"  ... and {len(results['myopic_ilp']['results']) - 3} more groups")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"=== ERROR after {elapsed_time:.2f} seconds ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("=== EXPERIMENT COMPLETED SUCCESSFULLY ===")
    return True


if __name__ == "__main__":
    success = run_toy_experiment()
    if success:
        print("✅ Toy experiment completed successfully!")
    else:
        print("❌ Toy experiment failed!")
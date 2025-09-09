#!/usr/bin/env python3
"""
Performance comparison script to demonstrate the difference between
the original large dataset and the toy datasets.
"""
import pandas as pd
import time
import sys
from seating_opt.data_gen import generate_seats, generate_groups_from_students
from seating_opt.experiments import run_all


def create_original_dataset():
    """Create the original large dataset from demo_main.py"""
    seats_df = generate_seats(
        rooms=1, tables_per_room=7, rows_per_table=4, cols_per_table=6, table_gap=1, room_gap=8, seed=7
    )
    students_df, groups_df = generate_groups_from_students(
        n_students=120, min_group=2, max_group=6, seed=17
    )
    return seats_df, students_df, groups_df


def create_toy_dataset():
    """Create the toy dataset"""
    seats_df = generate_seats(
        rooms=1, tables_per_room=2, rows_per_table=3, cols_per_table=4, table_gap=1, room_gap=8, seed=7
    )
    students_df, groups_df = generate_groups_from_students(
        n_students=18, min_group=2, max_group=5, seed=17
    )
    return seats_df, students_df, groups_df


def create_micro_dataset():
    """Create the micro dataset"""
    seats_df = generate_seats(
        rooms=1, tables_per_room=1, rows_per_table=2, cols_per_table=3, table_gap=1, room_gap=8, seed=42
    )
    students_df, groups_df = generate_groups_from_students(
        n_students=6, min_group=2, max_group=3, seed=123
    )
    return seats_df, students_df, groups_df


def test_dataset(name, create_func, lambda_pair=0.3, dmax_pairs=3, timeout_sec=30):
    """Test a dataset with timeout"""
    print(f"\n=== Testing {name} Dataset ===")
    
    try:
        # Create dataset
        seats_df, students_df, groups_df = create_func()
        print(f"  Seats: {len(seats_df)}")
        print(f"  Students: {len(students_df)}")
        print(f"  Groups: {len(groups_df)}")
        print(f"  Parameters: λ={lambda_pair}, dmax={dmax_pairs}")
        
        # Estimate problem complexity
        num_pairs_approx = len(seats_df) * dmax_pairs * (dmax_pairs + 1) // 2
        num_variables_approx = len(seats_df) * len(groups_df) + num_pairs_approx * len(groups_df)
        print(f"  Estimated variables: ~{num_variables_approx}")
        
        # Try to run with timeout
        start_time = time.time()
        
        # Simple timeout mechanism using alarm (Unix only)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Timed out after {timeout_sec} seconds")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        
        try:
            results = run_all(seats_df, groups_df, lam_pair=lambda_pair, dmax_pairs=dmax_pairs)
            signal.alarm(0)  # Cancel alarm
            
            elapsed_time = time.time() - start_time
            print(f"  ✅ SUCCESS in {elapsed_time:.3f} seconds")
            
            # Show results summary
            print(f"  Gold status: {results['world']['gold_status']}")
            greedy = results["greedy"]["summary"]
            ilp = results["myopic_ilp"]["summary"]
            print(f"  Greedy:     {greedy['success_rate']:.1%} success, {greedy['runtime_ms']:.1f} ms")
            print(f"  Myopic ILP: {ilp['success_rate']:.1%} success, {ilp['runtime_ms']:.1f} ms")
            
            return True, elapsed_time
            
        except TimeoutError as e:
            signal.alarm(0)  # Cancel alarm
            elapsed_time = time.time() - start_time
            print(f"  ❌ TIMEOUT after {elapsed_time:.1f} seconds")
            return False, elapsed_time
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  ❌ ERROR after {elapsed_time:.1f} seconds: {e}")
        return False, elapsed_time


def main():
    """Run performance comparison"""
    print("SEATING OPTIMIZATION PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Test datasets in order of increasing size
    datasets = [
        ("Micro", create_micro_dataset, {"lambda_pair": 0.5, "dmax_pairs": 1, "timeout_sec": 10}),
        ("Toy", create_toy_dataset, {"lambda_pair": 0.3, "dmax_pairs": 2, "timeout_sec": 20}),
        ("Original", create_original_dataset, {"lambda_pair": 0.3, "dmax_pairs": 3, "timeout_sec": 30}),
    ]
    
    results = []
    
    for name, create_func, params in datasets:
        success, elapsed_time = test_dataset(name, create_func, **params)
        results.append((name, success, elapsed_time))
        
        # If original dataset fails, don't bother with larger ones
        if name == "Original" and not success:
            break
    
    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print("=" * 50)
    
    for name, success, elapsed_time in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name:8}: {status} ({elapsed_time:.3f}s)")
    
    print()
    working_datasets = [name for name, success, _ in results if success]
    if working_datasets:
        print(f"✅ Working datasets: {', '.join(working_datasets)}")
        print("Use the toy datasets for development and testing.")
        print("The micro dataset is perfect for rapid iteration and debugging.")
    else:
        print("❌ No datasets worked!")


if __name__ == "__main__":
    main()
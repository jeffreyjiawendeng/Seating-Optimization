# micro_experiment.py
"""
Micro experiment to verify all three algorithms work.
"""

from seating_opt.data_gen import generate_seats, generate_groups
from seating_opt.greedy import greedy_pairwise  
from seating_opt.ilp_solvers import solve_group_pair_ilp
from seating_opt.sketchrefine import sketchrefine_solver
import pandas as pd
import time

# Create tiny dataset
print("Creating dataset...")
seats_df = generate_seats(rooms=1, tables_per_room=2, rows_per_table=2, cols_per_table=3, seed=42)

# Create one simple group
groups_df = pd.DataFrame([{
    "Group_ID": 1,
    "Group_Size": 3,
    "Brightness_Min": 40.0,  # Very achievable
    "Objective": "Q1"
}])

print(f"Dataset: {len(seats_df)} seats, 1 group")
print(f"Group: size={groups_df.iloc[0]['Group_Size']}, brightness≥{groups_df.iloc[0]['Brightness_Min']}")

# Test each algorithm
algorithms = [
    ("Greedy", greedy_pairwise),
    ("ILP", solve_group_pair_ilp), 
    ("SketchRefine", sketchrefine_solver)
]

group = groups_df.iloc[0]

for name, solver in algorithms:
    print(f"\nTesting {name}...")
    start = time.time()
    
    try:
        result = solver(
            seats_df=seats_df.copy(),
            group_size=int(group['Group_Size']),
            brightness_min=float(group['Brightness_Min']),
            lam_pair=0.3
        )
        
        runtime = (time.time() - start) * 1000
        
        if result['status'] in ['ok', 'Optimal']:
            seats = seats_df[seats_df['Seat_ID'].isin(result['seat_ids'])]
            avg_brightness = seats['Brightness'].mean()
            print(f"  ✅ SUCCESS: {len(result['seat_ids'])} seats, brightness={avg_brightness:.1f}, {runtime:.1f}ms")
        else:
            print(f"  ❌ FAILED: {result.get('status', 'Unknown')} in {runtime:.1f}ms")
            
    except Exception as e:
        runtime = (time.time() - start) * 1000
        print(f"  💥 ERROR: {str(e)} in {runtime:.1f}ms")

print("\n🎉 Micro experiment completed!")
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
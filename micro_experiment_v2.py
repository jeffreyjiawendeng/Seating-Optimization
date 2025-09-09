# micro_experiment_v2.py
"""
Simple test of all three algorithms.
"""

from seating_opt.data_gen import generate_seats, generate_groups
from seating_opt.greedy import greedy_pairwise  
from seating_opt.ilp_solvers import solve_group_pair_ilp
from seating_opt.sketchrefine import sketchrefine_solver
import pandas as pd
import time

print("🔬 Testing All Three Algorithms")
print("=" * 35)

# Create tiny dataset
seats_df = generate_seats(rooms=1, tables_per_room=2, rows_per_table=2, cols_per_table=3, seed=42)

# Create one simple group
group = {
    "Group_ID": 1,
    "Group_Size": 3,
    "Brightness_Min": 40.0,  # Very achievable
}

print(f"Dataset: {len(seats_df)} seats")
print(f"Group: size={group['Group_Size']}, brightness≥{group['Brightness_Min']}")

# Test each algorithm
algorithms = [
    ("Greedy", greedy_pairwise),
    ("ILP", solve_group_pair_ilp), 
    ("SketchRefine", sketchrefine_solver)
]

for name, solver in algorithms:
    print(f"\n🧪 Testing {name}...")
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
            print(f"   ✅ SUCCESS: {len(result['seat_ids'])} seats selected")
            print(f"      Runtime: {runtime:.1f}ms")
            print(f"      Brightness: {avg_brightness:.1f} (required: ≥{group['Brightness_Min']})")
            print(f"      Seats: {result['seat_ids']}")
        else:
            print(f"   ❌ FAILED: {result.get('status', 'Unknown')}")
            print(f"      Runtime: {runtime:.1f}ms")
            if 'reason' in result:
                print(f"      Reason: {result['reason']}")
            
    except Exception as e:
        runtime = (time.time() - start) * 1000
        print(f"   💥 ERROR: {str(e)}")
        print(f"      Runtime: {runtime:.1f}ms")

print("\n🎉 All algorithms tested!")

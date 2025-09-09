# debug_sketchrefine.py
"""
Debug SketchRefine step by step.
"""

import pandas as pd
from seating_opt.data_gen import generate_seats, generate_groups
from seating_opt.sketchrefine import create_table_representatives, solve_sketch_ilp


def debug_sketch_phase():
    """Test just the sketch phase."""
    print("🔬 Debugging SketchRefine - Sketch Phase Only")
    print("=" * 50)
    
    # Create small dataset
    seats_df = generate_seats(rooms=1, tables_per_room=2, rows_per_table=2, cols_per_table=2, seed=42)
    
    # Create a group with reasonable brightness requirement
    groups_df = pd.DataFrame([{
        "Group_ID": 1,
        "Group_Size": 2,
        "Brightness_Min": 45.0,  # Lower requirement that can be satisfied
        "Objective": "Q1"
    }])
    
    group = groups_df.iloc[0]
    print(f"Dataset: {len(seats_df)} seats, Group size: {group['Group_Size']}, Brightness min: {group['Brightness_Min']:.1f}")
    
    # Step 1: Create representatives
    print("\n1️⃣ Creating table representatives...")
    try:
        representatives_df, table_to_seats = create_table_representatives(seats_df)
        print(f"   ✅ Created {len(representatives_df)} representatives")
        print(representatives_df[['Table_ID', 'Representative_Brightness', 'Representative_Noise', 'Seat_Count']])
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return
    
    # Step 2: Solve sketch ILP
    print("\n2️⃣ Solving sketch ILP...")
    try:
        sketch_result = solve_sketch_ilp(
            representatives_df=representatives_df,
            group_size=int(group['Group_Size']),
            brightness_min=float(group['Brightness_Min']),
            lam_pair=0.3,
            dmax_pairs=3,
            time_limit_sec=10  # Short timeout for testing
        )
        
        print(f"   Status: {sketch_result['status']}")
        if sketch_result['status'] == 'Optimal':
            print(f"   Solution: {sketch_result['sketch_solution']}")
            print(f"   Objective: {sketch_result.get('objective', 'N/A')}")
        else:
            print(f"   Reason: {sketch_result.get('reason', 'Unknown')}")
            
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_sketch_phase()

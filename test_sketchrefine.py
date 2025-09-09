# test_sketchrefine.py
"""
Quick test of the SketchRefine algorithm implementation.
"""

import pandas as pd
from seating_opt.data_gen import generate_seats, generate_groups
from seating_opt.sketchrefine import sketchrefine_solver


def test_sketchrefine():
    """Test SketchRefine on a small dataset."""
    print("🔬 Testing SketchRefine Algorithm")
    print("=" * 40)
    
    # Create small dataset
    seats_df = generate_seats(
        rooms=1,
        tables_per_room=4,
        rows_per_table=3,
        cols_per_table=3,
        seed=42
    )
    
    groups_df = generate_groups(
        n_groups=5,
        min_size=2,
        max_size=3,
        seed=42
    )
    
    print(f"📊 Dataset: {len(seats_df)} seats, {len(groups_df)} groups")
    print(f"   Tables: {seats_df['Table_ID'].nunique()}")
    
    # Test individual group
    group = groups_df.iloc[0]
    print(f"\n🎯 Testing Group {group['Group_ID']}: size={group['Group_Size']}, brightness_min={group['Brightness_Min']:.1f}")
    
    try:
        result = sketchrefine_solver(
            seats_df=seats_df,
            group_size=int(group['Group_Size']),
            brightness_min=float(group['Brightness_Min']),
            lam_pair=0.3,
            dmax_pairs=3
        )
        
        print(f"   Status: {result['status']}")
        if result['status'] == 'ok':
            print(f"   Selected seats: {result['seat_ids']}")
            
            # Verify solution
            selected_seats = seats_df[seats_df['Seat_ID'].isin(result['seat_ids'])]
            avg_brightness = selected_seats['Brightness'].mean()
            avg_noise = selected_seats['Noise'].mean()
            
            print(f"   Avg Brightness: {avg_brightness:.1f} (required: ≥{group['Brightness_Min']:.1f})")
            print(f"   Avg Noise: {avg_noise:.1f}")
            print(f"   Constraint satisfied: {avg_brightness >= group['Brightness_Min']}")
        else:
            print(f"   Reason: {result.get('reason', 'Unknown')}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ SketchRefine test completed!")


if __name__ == "__main__":
    test_sketchrefine()

# full_sketchrefine_test.py
"""
Test the complete SketchRefine algorithm.
"""

import pandas as pd
from seating_opt.data_gen import generate_seats
from seating_opt.sketchrefine import sketchrefine_solver


def test_full_sketchrefine():
    """Test complete SketchRefine algorithm."""
    print("🔬 Testing Full SketchRefine Algorithm")
    print("=" * 45)
    
    # Create small dataset
    seats_df = generate_seats(rooms=1, tables_per_room=3, rows_per_table=2, cols_per_table=3, seed=42)
    print(f"Dataset: {len(seats_df)} seats across {seats_df['Table_ID'].nunique()} tables")
    
    # Show brightness distribution
    table_brightness = seats_df.groupby('Table_ID')['Brightness'].mean()
    print("Table brightness averages:")
    for table_id, brightness in table_brightness.items():
        print(f"   Table {table_id}: {brightness:.1f}")
    
    # Test with reasonable brightness requirement
    group_size = 3
    brightness_min = 45.0  # Should be achievable
    
    print(f"\n🎯 Testing: Group size={group_size}, Brightness min={brightness_min}")
    
    try:
        result = sketchrefine_solver(
            seats_df=seats_df,
            group_size=group_size,
            brightness_min=brightness_min,
            lam_pair=0.3,
            dmax_pairs=3,
            time_limit_sec=30
        )
        
        print(f"   Status: {result['status']}")
        
        if result['status'] == 'ok':
            selected_seats = result['seat_ids']
            print(f"   Selected seats: {selected_seats}")
            
            # Verify solution
            seat_details = seats_df[seats_df['Seat_ID'].isin(selected_seats)]
            avg_brightness = seat_details['Brightness'].mean()
            avg_noise = seat_details['Noise'].mean()
            
            print(f"   ✅ Solution verification:")
            print(f"      Seats selected: {len(selected_seats)} (required: {group_size})")
            print(f"      Avg brightness: {avg_brightness:.1f} (required: ≥{brightness_min})")
            print(f"      Avg noise: {avg_noise:.1f}")
            print(f"      Constraint satisfied: {avg_brightness >= brightness_min}")
            print(f"      Tables used: {seat_details['Table_ID'].tolist()}")
            
        else:
            print(f"   ❌ Failed: {result.get('reason', 'Unknown error')}")
            
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_full_sketchrefine()

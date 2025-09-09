#!/usr/bin/env python3
"""
Ultra-simple toy example - Manual dataset creation
"""

import pandas as pd
import numpy as np

def create_simple_toy_data():
    """Create the simplest possible dataset."""
    # Create 6 seats manually
    seats_data = [
        {"Seat_ID": 1, "Room_ID": 1, "Table_ID": 1, "X": 0, "Y": 0, "Brightness": 80, "Noise": 20, "Seat_Available": True},
        {"Seat_ID": 2, "Room_ID": 1, "Table_ID": 1, "X": 1, "Y": 0, "Brightness": 75, "Noise": 25, "Seat_Available": True},
        {"Seat_ID": 3, "Room_ID": 1, "Table_ID": 1, "X": 0, "Y": 1, "Brightness": 70, "Noise": 30, "Seat_Available": True},
        {"Seat_ID": 4, "Room_ID": 1, "Table_ID": 2, "X": 3, "Y": 0, "Brightness": 85, "Noise": 15, "Seat_Available": True},
        {"Seat_ID": 5, "Room_ID": 1, "Table_ID": 2, "X": 4, "Y": 0, "Brightness": 80, "Noise": 20, "Seat_Available": True},
        {"Seat_ID": 6, "Room_ID": 1, "Table_ID": 2, "X": 3, "Y": 1, "Brightness": 75, "Noise": 25, "Seat_Available": True},
    ]
    
    # Create 2 simple groups
    groups_data = [
        {"Group_ID": 1, "Group_Size": 2, "Brightness_Min": 70.0},
        {"Group_ID": 2, "Group_Size": 2, "Brightness_Min": 75.0},
    ]
    
    seats_df = pd.DataFrame(seats_data)
    groups_df = pd.DataFrame(groups_data)
    
    return seats_df, groups_df

def test_greedy_only():
    """Test just the greedy algorithm on simple data."""
    print("🧪 ULTRA-SIMPLE TOY TEST")
    print("=" * 30)
    
    try:
        # Create simple data
        seats_df, groups_df = create_simple_toy_data()
        
        print(f"Created {len(seats_df)} seats and {len(groups_df)} groups")
        print("Seats:")
        print(seats_df[['Seat_ID', 'Table_ID', 'Brightness', 'Noise']])
        print("\nGroups:")
        print(groups_df)
        
        # Test greedy algorithm directly
        from seating_opt.greedy import greedy_pairwise
        
        print(f"\n🔬 Testing Greedy Algorithm...")
        
        for _, group in groups_df.iterrows():
            print(f"\nGroup {group['Group_ID']} (size={group['Group_Size']}, brightness_min={group['Brightness_Min']}):")
            
            result = greedy_pairwise(
                seats_df, 
                int(group['Group_Size']), 
                float(group['Brightness_Min']), 
                lam_pair=0.3
            )
            
            print(f"   Result: {result}")
            
            if result.get('status') == 'ok':
                selected_seats = result['seat_ids']
                # Mark seats as unavailable
                seats_df.loc[seats_df['Seat_ID'].isin(selected_seats), 'Seat_Available'] = False
                
                # Show selected seat details
                selected_data = seats_df[seats_df['Seat_ID'].isin(selected_seats)]
                print(f"   Selected seats details:")
                print(selected_data[['Seat_ID', 'Table_ID', 'Brightness', 'Noise']])
        
        print(f"\n✅ Simple test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_greedy_only()

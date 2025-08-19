#!/usr/bin/env python3
"""
Script to verify the groups.csv file structure.
"""

import pandas as pd

def verify_groups_file():
    """Verify that groups.csv has the correct structure."""
    
    print("Verifying groups.csv structure...")
    
    # Load the groups data
    groups_df = pd.read_csv('groups.csv')
    print(f"Loaded {len(groups_df)} students from groups.csv")
    
    # Check total count
    if len(groups_df) != 2000:
        print(f"ERROR: Expected 2000 students, but found {len(groups_df)}")
        return False
    
    # Check set structure
    print("\nSet verification:")
    for set_id in range(1, 5):
        set_data = groups_df[groups_df['Set_ID'] == set_id]
        set_count = len(set_data)
        unique_groups = set_data['Group_ID'].nunique()
        
        print(f"  Set {set_id}: {set_count} students, {unique_groups} groups")
        
        if set_count != 500:
            print(f"    ERROR: Set {set_id} should have 500 students, but has {set_count}")
            return False
    
    # Verify sequential loading pattern
    print("\nSequential loading verification:")
    
    # Trial 1: Set 1 (500 students)
    trial1 = groups_df[groups_df['Set_ID'] == 1]
    print(f"  Trial 1 (500 students): {len(trial1)} students ✓")
    
    # Trial 2: Sets 1 + 2 (1000 students)
    trial2 = groups_df[groups_df['Set_ID'].isin([1, 2])]
    print(f"  Trial 2 (1000 students): {len(trial2)} students ✓")
    
    # Trial 3: Sets 1 + 2 + 3 (1500 students)
    trial3 = groups_df[groups_df['Set_ID'].isin([1, 2, 3])]
    print(f"  Trial 3 (1500 students): {len(trial3)} students ✓")
    
    # Trial 4: Sets 1 + 2 + 3 + 4 (2000 students)
    trial4 = groups_df[groups_df['Set_ID'].isin([1, 2, 3, 4])]
    print(f"  Trial 4 (2000 students): {len(trial4)} students ✓")
    
    # Check that all students are in the correct order
    print("\nOrder verification:")
    first_student = groups_df.iloc[0]['Student_ID']
    last_student = groups_df.iloc[-1]['Student_ID']
    print(f"  First student: {first_student}")
    print(f"  Last student: {last_student}")
    
    # Check that Set_ID increases sequentially
    set_ids = groups_df['Set_ID'].values
    expected_transitions = [500, 1000, 1500]  # Indices where Set_ID should change
    
    for i, transition_idx in enumerate(expected_transitions):
        if set_ids[transition_idx] != i + 2:  # Should be Set_ID 2, 3, 4
            print(f"  ERROR: Set_ID should change to {i + 2} at index {transition_idx}, but found {set_ids[transition_idx]}")
            return False
    
    print("  Set_ID transitions are correct ✓")
    
    print("\nAll verifications passed! ✓")
    return True

if __name__ == "__main__":
    verify_groups_file() 
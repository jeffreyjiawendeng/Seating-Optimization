#!/usr/bin/env python3
"""
Test script for the new dataset generation and loading functions.
"""

import pandas as pd
import numpy as np
from experiments import generate_experiment_dataset, save_experiment_dataset, load_experiment_dataset

def test_dataset_generation():
    """Test the dataset generation and loading functions."""
    print("Testing dataset generation...")
    
    # Test 1: Generate new dataset
    print("\n1. Generating new dataset...")
    seats_df, students_df = generate_experiment_dataset()
    
    print(f"   Generated {len(seats_df)} seats")
    print(f"   Generated {len(students_df)} students")
    print(f"   Number of groups: {students_df['Group_ID'].nunique()}")
    
    # Test 2: Save dataset
    print("\n2. Saving dataset...")
    save_experiment_dataset(seats_df, students_df)
    
    # Test 3: Load dataset
    print("\n3. Loading dataset...")
    loaded_seats, loaded_students = load_experiment_dataset()
    
    print(f"   Loaded {len(loaded_seats)} seats")
    print(f"   Loaded {len(loaded_students)} students")
    print(f"   Number of groups: {loaded_students['Group_ID'].nunique()}")
    
    # Test 4: Verify data integrity
    print("\n4. Verifying data integrity...")
    seats_match = seats_df.equals(loaded_seats)
    students_match = students_df.equals(loaded_students)
    
    print(f"   Seats data matches: {seats_match}")
    print(f"   Students data matches: {students_match}")
    
    # Test 5: Check data structure
    print("\n5. Checking data structure...")
    print(f"   Seats columns: {list(seats_df.columns)}")
    print(f"   Students columns: {list(students_df.columns)}")
    
    # Test 6: Check data ranges
    print("\n6. Checking data ranges...")
    print(f"   Brightness range: {seats_df['Brightness'].min()} - {seats_df['Brightness'].max()}")
    print(f"   Noise range: {seats_df['Noise'].min()} - {seats_df['Noise'].max()}")
    print(f"   Room IDs: {sorted(seats_df['Room_ID'].unique())}")
    print(f"   Table IDs: {sorted(seats_df['Table_ID'].unique())}")
    
    # Test 7: Check group sizes
    print("\n7. Checking group sizes...")
    group_sizes = students_df.groupby('Group_ID').size()
    print(f"   Group size range: {group_sizes.min()} - {group_sizes.max()}")
    print(f"   Average group size: {group_sizes.mean():.2f}")
    
    print("\nDataset generation test completed successfully!")

if __name__ == "__main__":
    test_dataset_generation() 
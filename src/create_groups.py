#!/usr/bin/env python3
"""
Script to create groups.csv with 4 sets of 500 students each.
Each set will contain groups with varying sizes (1-10 students).
Each set will be loaded sequentially for experiments:
- Trial 1: 500 students (Set 1)
- Trial 2: 1000 students (Sets 1 + 2)
- Trial 3: 1500 students (Sets 1 + 2 + 3)
- Trial 4: 2000 students (Sets 1 + 2 + 3 + 4)
"""

import pandas as pd
import numpy as np

def create_groups_file():
    """Create groups.csv with 4 sets of 500 students each, with varying group sizes."""
    
    # Load the existing students data
    print("Loading existing students data...")
    students_df = pd.read_csv('students.csv')
    print(f"Loaded {len(students_df)} students")
    
    # Verify we have exactly 2000 students
    if len(students_df) != 2000:
        print(f"Warning: Expected 2000 students, but found {len(students_df)}")
        return
    
    # Create groups.csv structure
    # We'll create 4 sets, each with 500 students
    # Each set will contain groups with varying sizes (1-10 students)
    
    groups_data = []
    student_counter = 0
    group_counter = 1
    
    # Set 1: Students 1-500
    set1_students = students_df.iloc[0:500]
    current_group_size = 0
    current_group = []
    
    for _, student in set1_students.iterrows():
        if current_group_size == 0:
            # Start a new group with random size (1-10)
            current_group_size = np.random.randint(1, 11)
            current_group = []
        
        current_group.append({
            'Student_ID': student['Student_ID'],
            'Group_ID': group_counter,
            'Set_ID': 1,
            'Set_Size': 500,
            'Brightness': student['Brightness'],
            'Noise': student['Noise'],
            'Flexibility': student['Flexibility']
        })
        
        if len(current_group) == current_group_size:
            # Group is complete, add to data and start new group
            groups_data.extend(current_group)
            group_counter += 1
            current_group_size = 0
            current_group = []
    
    # Handle any remaining students in the last group
    if current_group:
        groups_data.extend(current_group)
        group_counter += 1
    
    # Set 2: Students 501-1000
    set2_students = students_df.iloc[500:1000]
    current_group_size = 0
    current_group = []
    
    for _, student in set2_students.iterrows():
        if current_group_size == 0:
            # Start a new group with random size (1-10)
            current_group_size = np.random.randint(1, 11)
            current_group = []
        
        current_group.append({
            'Student_ID': student['Student_ID'],
            'Group_ID': group_counter,
            'Set_ID': 2,
            'Set_Size': 500,
            'Brightness': student['Brightness'],
            'Noise': student['Noise'],
            'Flexibility': student['Flexibility']
        })
        
        if len(current_group) == current_group_size:
            # Group is complete, add to data and start new group
            groups_data.extend(current_group)
            group_counter += 1
            current_group_size = 0
            current_group = []
    
    # Handle any remaining students in the last group
    if current_group:
        groups_data.extend(current_group)
        group_counter += 1
    
    # Set 3: Students 1001-1500
    set3_students = students_df.iloc[1000:1500]
    current_group_size = 0
    current_group = []
    
    for _, student in set3_students.iterrows():
        if current_group_size == 0:
            # Start a new group with random size (1-10)
            current_group_size = np.random.randint(1, 11)
            current_group = []
        
        current_group.append({
            'Student_ID': student['Student_ID'],
            'Group_ID': group_counter,
            'Set_ID': 3,
            'Set_Size': 500,
            'Brightness': student['Brightness'],
            'Noise': student['Noise'],
            'Flexibility': student['Flexibility']
        })
        
        if len(current_group) == current_group_size:
            # Group is complete, add to data and start new group
            groups_data.extend(current_group)
            group_counter += 1
            current_group_size = 0
            current_group = []
    
    # Handle any remaining students in the last group
    if current_group:
        groups_data.extend(current_group)
        group_counter += 1
    
    # Set 4: Students 1501-2000
    set4_students = students_df.iloc[1500:2000]
    current_group_size = 0
    current_group = []
    
    for _, student in set4_students.iterrows():
        if current_group_size == 0:
            # Start a new group with random size (1-10)
            current_group_size = np.random.randint(1, 11)
            current_group = []
        
        current_group.append({
            'Student_ID': student['Student_ID'],
            'Group_ID': group_counter,
            'Set_ID': 4,
            'Set_Size': 500,
            'Brightness': student['Brightness'],
            'Noise': student['Noise'],
            'Flexibility': student['Flexibility']
        })
        
        if len(current_group) == current_group_size:
            # Group is complete, add to data and start new group
            groups_data.extend(current_group)
            group_counter += 1
            current_group_size = 0
            current_group = []
    
    # Handle any remaining students in the last group
    if current_group:
        groups_data.extend(current_group)
        group_counter += 1
    
    # Create DataFrame
    groups_df = pd.DataFrame(groups_data)
    
    # Save to CSV
    groups_df.to_csv('groups.csv', index=False)
    
    # Print summary
    print(f"\nCreated groups.csv with {len(groups_df)} students")
    print("\nSet breakdown:")
    for set_id in range(1, 5):
        set_data = groups_df[groups_df['Set_ID'] == set_id]
        unique_groups = set_data['Group_ID'].nunique()
        group_sizes = set_data.groupby('Group_ID').size()
        print(f"  Set {set_id}: {len(set_data)} students, {unique_groups} groups")
        print(f"    Group sizes: {group_sizes.min()}-{group_sizes.max()} students")
        print(f"    Average group size: {group_sizes.mean():.1f} students")
    
    # Verify the sequential loading pattern
    print("\nSequential loading pattern:")
    print("  Trial 1 (500 students): Set 1")
    print("  Trial 2 (1000 students): Sets 1 + 2")
    print("  Trial 3 (1500 students): Sets 1 + 2 + 3")
    print("  Trial 4 (2000 students): Sets 1 + 2 + 3 + 4")
    
    # Show sample of the data
    print(f"\nSample of groups.csv:")
    print(groups_df.head(15).to_string(index=False))
    
    return groups_df

if __name__ == "__main__":
    create_groups_file() 
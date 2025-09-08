import numpy as np
import pandas as pd
import pulp
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum
import math

def load_toy_data():
    """Load the toy dataset files."""
    try:
        seats_df = pd.read_csv("seats_toy.csv")
        students_df = pd.read_csv("students_toy.csv")
        tables_df = pd.read_csv("tables_toy.csv")
        return seats_df, students_df, tables_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run data_toy.py first to generate the toy datasets.")
        return None, None, None

def calculate_distance(seat1, seat2):
    """Calculate Euclidean distance between two seats."""
    return math.sqrt((seat1['seat_x'] - seat2['seat_x'])**2 + (seat1['seat_y'] - seat2['seat_y'])**2)

def solve_global_ilp(query_type='Q1'):
    """
    Solve global ILP optimization for all groups simultaneously.
    
    Args:
        query_type: 'Q1' or 'Q2' for different objectives
        
    Returns:
        dict: Results with assignments, objective value, and execution time
    """
    import time
    start_time = time.time()
    
    # Load data
    seats_df, students_df, tables_df = load_toy_data()
    if seats_df is None:
        return {'success': False, 'error': 'Failed to load data'}
    
    print(f"Solving Global ILP for {query_type}...")
    print(f"Seats: {len(seats_df)}, Students: {len(students_df)}, Groups: {students_df['group_id'].nunique()}")
    
    # Get all groups
    groups = students_df.groupby('group_id')
    group_list = sorted(groups, key=lambda x: x[0])
    
    # Create the optimization problem
    if query_type == 'Q1':
        prob = LpProblem("Global_Seating_Q1", LpMinimize)
    else:  # Q2
        prob = LpProblem("Global_Seating_Q2", LpMinimize)
    
    # Decision variables: x[student_id][seat_id] = 1 if student is assigned to seat
    student_ids = students_df['student_id'].tolist()
    seat_ids = seats_df['seat_id'].tolist()
    
    x = {}
    for student_id in student_ids:
        x[student_id] = {}
        for seat_id in seat_ids:
            x[student_id][seat_id] = LpVariable(f"x_{student_id}_{seat_id}", cat='Binary')
    
    # Objective function
    if query_type == 'Q1':
        # Q1: Minimize total noise
        prob += lpSum([
            x[student_id][seat_id] * seats_df[seats_df['seat_id'] == seat_id]['noise'].iloc[0]
            for student_id in student_ids
            for seat_id in seat_ids
        ])
        
    else:  # Q2
        # Q2: Minimize (noise - 0.3*brightness)
        prob += lpSum([
            x[student_id][seat_id] * (
                seats_df[seats_df['seat_id'] == seat_id]['noise'].iloc[0] - 
                0.3 * seats_df[seats_df['seat_id'] == seat_id]['brightness'].iloc[0]
            )
            for student_id in student_ids
            for seat_id in seat_ids
        ])
    
    # Constraints
    
    # 1. Each student gets exactly one seat
    for student_id in student_ids:
        prob += lpSum([x[student_id][seat_id] for seat_id in seat_ids]) == 1
    
    # 2. Each seat can be assigned to at most one student
    for seat_id in seat_ids:
        prob += lpSum([x[student_id][seat_id] for student_id in student_ids]) <= 1
    
    # 3. Brightness constraints: each student's seat must meet their brightness preference
    for _, student in students_df.iterrows():
        student_id = student['student_id']
        brightness_req = student['brightness_preference']
        prob += lpSum([
            x[student_id][seat_id] * seats_df[seats_df['seat_id'] == seat_id]['brightness'].iloc[0]
            for seat_id in seat_ids
        ]) >= brightness_req
    
    # Solve the problem
    print("Solving ILP...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    execution_time = (time.time() - start_time) * 1000
    
    # Check if solution was found
    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"ILP Status: {pulp.LpStatus[prob.status]}")
        return {
            'success': False,
            'status': pulp.LpStatus[prob.status],
            'execution_time_ms': execution_time
        }
    
    # Extract solution
    assignments = []
    total_noise = 0
    total_brightness = 0
    total_distance = 0
    
    for student_id in student_ids:
        for seat_id in seat_ids:
            if x[student_id][seat_id].varValue == 1:
                seat = seats_df[seats_df['seat_id'] == seat_id].iloc[0]
                student = students_df[students_df['student_id'] == student_id].iloc[0]
                
                assignments.append({
                    'student_id': student_id,
                    'group_id': student['group_id'],
                    'seat_id': seat_id,
                    'seat_x': seat['seat_x'],
                    'seat_y': seat['seat_y'],
                    'brightness': seat['brightness'],
                    'noise': seat['noise']
                })
                
                total_noise += seat['noise']
                total_brightness += seat['brightness']
                break
    
    # Calculate total distance between group members
    for group_id, group_students in group_list:
        group_assignments = [a for a in assignments if a['group_id'] == group_id]
        for i, assign1 in enumerate(group_assignments):
            for j, assign2 in enumerate(group_assignments):
                if i < j:
                    distance = calculate_distance(assign1, assign2)
                    total_distance += distance
    
    # Calculate objective value
    if query_type == 'Q1':
        objective_value = total_noise / len(students_df)
    else:  # Q2
        objective_value = (total_noise - 0.3 * total_brightness) / len(students_df)
    
    print(f"Solution found! Average objective value per student: {objective_value:.2f}")
    print(f"Total noise: {total_noise}, Total brightness: {total_brightness}")
    print(f"Total distance between group members: {total_distance:.2f}")
    print(f"Execution time: {execution_time:.2f}ms")
    
    return {
        'success': True,
        'assignments': assignments,
        'objective_value': objective_value,
        'total_noise': total_noise,
        'total_brightness': total_brightness,
        'total_distance': total_distance,
        'execution_time_ms': execution_time,
        'num_students_assigned': len(assignments)
    }

def print_assignments(result):
    """Print the assignment results in a readable format."""
    if not result['success']:
        print(f"Assignment failed: {result.get('status', 'Unknown error')}")
        return
    
    assignments = result['assignments']
    
    print("\n" + "="*80)
    print("GLOBAL ILP ASSIGNMENT RESULTS")
    print("="*80)
    
    # Group assignments by group
    groups = {}
    for assignment in assignments:
        group_id = assignment['group_id']
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(assignment)
    
    for group_id in sorted(groups.keys()):
        group_assignments = groups[group_id]
        print(f"\nGroup {group_id} ({len(group_assignments)} students):")
        
        # Calculate group statistics
        avg_brightness = np.mean([a['brightness'] for a in group_assignments])
        avg_noise = np.mean([a['noise'] for a in group_assignments])
        
        # Calculate max distance within group
        max_distance = 0
        for i, assign1 in enumerate(group_assignments):
            for j, assign2 in enumerate(group_assignments):
                if i < j:
                    distance = calculate_distance(assign1, assign2)
                    max_distance = max(max_distance, distance)
        
        print(f"  Avg Brightness: {avg_brightness:.1f}, Avg Noise: {avg_noise:.1f}")
        print(f"  Max Distance: {max_distance:.2f}")
        print("  Seat Assignments:")
        
        for assignment in group_assignments:
            print(f"    Student {assignment['student_id']} -> Seat {assignment['seat_id']} "
                  f"({assignment['seat_x']:.1f}, {assignment['seat_y']:.1f}) "
                  f"[B{assignment['brightness']}, N{assignment['noise']}]")
    
    print(f"\nOverall Statistics:")
    print(f"  Total Students Assigned: {result['num_students_assigned']}")
    print(f"  Objective Value: {result['objective_value']:.2f}")
    print(f"  Total Distance: {result['total_distance']:.2f}")
    print(f"  Execution Time: {result['execution_time_ms']:.2f}ms")

if __name__ == "__main__":
    # Test both Q1 and Q2
    print("Testing Global ILP with Toy Dataset")
    print("="*50)
    
    # Test Q1
    print("\n1. Testing Q1 (Minimize Noise)...")
    result_q1 = solve_global_ilp(query_type='Q1')
    print_assignments(result_q1)
    
    # Test Q2
    print("\n\n2. Testing Q2 (Minimize Noise-0.3*Brightness)...")
    result_q2 = solve_global_ilp(query_type='Q2')
    print_assignments(result_q2)
    
    print("\n" + "="*50)
    print("Global ILP Testing Complete!")

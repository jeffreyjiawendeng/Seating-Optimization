import pandas as pd
import numpy as np
from ilp import solve_ilp, solve_ilp_weighted, load_seat_data
from greedy import greedy_seat_selection, load_data, create_table_stats, create_adjacency_graph
import time
import os
import random
import matplotlib.pyplot as plt

def package_queries(model_type='greedy', group_size=5, brightness_threshold=70, query_type=None):
    """
    Execute package queries using either greedy or ILP algorithm.
    
    Args:
        model_type (str): 'greedy' or 'ilp'
        group_size (int): Size of the group to place
        brightness_threshold (float): Minimum brightness requirement
        query_type (str): 'Q1', 'Q2', or None for random selection
    
    Returns:
        dict: Results including selected seats, objective value, and execution time
    """
    # Randomly choose between Q1 and Q2 if not specified
    if query_type is None:
        query_type = random.choice(['Q1', 'Q2'])
    
    # Set weights based on query type
    if query_type == 'Q1':
        # Q1: minimize AVG(noise)
        w1 = 1.0
        w2 = 0.0
        objective_str = "Minimize AVG(noise)"
    else:
        # Q2: minimize AVG(noise) - 0.3*AVG(brightness)
        w1 = 1.0
        w2 = 0.3
        objective_str = "Minimize (AVG(noise) - 0.3*AVG(brightness))"
    
    start_time = time.time()
    
    if model_type == 'greedy':
        # Load data for greedy algorithm
        seats_df, students_df = load_experiment_dataset()
        table_stats = create_table_stats(seats_df)
        adjacency_graph = create_adjacency_graph(table_stats)
        
        # Run greedy algorithm
        result = greedy_seat_selection(
            group_size=group_size,
            brightness_threshold=brightness_threshold,
            table_stats=table_stats,
            adjacency_graph=adjacency_graph,
            w1=w1,
            w2=w2
        )
        
        if result:
            avg_brightness = np.mean([seat['Brightness'] for seat in result])
            avg_noise = np.mean([seat['Noise'] for seat in result])
            
            # Calculate objective value based on query type
            if query_type == 'Q1':
                objective_value = avg_noise
            else:
                objective_value = avg_noise - 0.3 * avg_brightness
        else:
            objective_value = None
            
    elif model_type == 'ilp':
        # Load data for ILP
        seats_df, students_df = load_experiment_dataset()
        # Prepare data matrix for ILP
        sub = seats_df[['Brightness', 'Noise', 'Room_ID', 'Table_ID']].values
        
        if query_type == 'Q1':
            # Q1: minimize total noise
            # Use standard ILP for Q1
            picked, counts = solve_ilp(
                sub=sub,
                size_=group_size,
                rep=1,
                obj={'attr': 1, 'pref': 'MIN'},  # Minimize noise
                cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
                verbose=False
            )
        else:
            # Q2: minimize (noise - 0.3*brightness)
            # Use weighted ILP for Q2 to match greedy objective
            # Weights: [brightness, noise, room_id, table_id] = [-0.3, 1.0, 0, 0]
            # This minimizes: -0.3*brightness + 1.0*noise = noise - 0.3*brightness
            picked, counts = solve_ilp_weighted(
                sub=sub,
                size_=group_size,
                rep=1,
                weights=[-0.3, 1.0, 0, 0],  # Negative weight for brightness (maximize), positive for noise (minimize)
                cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
                verbose=False
            )
        
        if picked:
            selected_seats = seats_df.iloc[picked]
            avg_brightness = selected_seats['Brightness'].mean()
            avg_noise = selected_seats['Noise'].mean()
            
            # Calculate objective value based on query type
            if query_type == 'Q1':
                objective_value = avg_noise
            else:
                objective_value = avg_noise - 0.3 * avg_brightness
                
            result = selected_seats.to_dict('records')
        else:
            result = None
            objective_value = None
    
    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return {
        'model_type': model_type,
        'query_type': query_type,
        'objective_str': objective_str,
        'result': result,
        'objective_value': objective_value,
        'execution_time_ms': execution_time,
        'success': result is not None
    }

def generate_experiment_dataset():
    """
    Generate a one-time dataset with 2000 seats and 2000 students for experiments.
    Returns the seats and students DataFrames.
    """
    print("Generating experiment dataset...")
    
    # Import data generation functions
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.data import generate_layout, generate_attribute_layout, attributes_to_csv, generate_students, students_to_csv
    
    # Generate layout for 2000 seats
    # We need: rooms * tr * tc * sr * sc = 2000
    # Let's use: 4 rooms * 5 tables per room * 4 rows per table * 5 seats per row = 400 seats per room
    # Total: 4 * 5 * 4 * 5 = 400 seats per room, so we need 5 rooms
    layout = generate_layout((5, 5, 4, 5, 4))  # 5 rooms, 5 tables, 4 rows, 5 seats, 4 columns
    attrs = generate_attribute_layout(
        layout,
        brightness_range=(80, 20),
        brightness_sd=3,
        noise_sd=10
    )
    
    # Convert to DataFrame
    rooms, tr, tc, sr, sc, _ = attrs.shape
    rows = []
    seat_id = 1
    table_counter = 1

    for room in range(rooms):
        room_id = room + 1
        for tr_i in range(tr):
            for tc_i in range(tc):
                table_id = table_counter
                table_counter += 1
                for sr_i in range(sr):
                    for sc_i in range(sc):
                        b, n = attrs[room, tr_i, tc_i, sr_i, sc_i]
                        rows.append({
                            "Seat_ID":         seat_id,
                            "Table_ID":        table_id,
                            "Room_ID":         room_id,
                            "Brightness":      b,
                            "Noise":           n,
                            "Seat_Available":  True,
                            "Table_Available": True,
                            "Room_Available":  True
                        })
                        seat_id += 1

    seats_df = pd.DataFrame(rows, columns=[
        "Seat_ID", 
        "Table_ID", 
        "Room_ID", 
        "Brightness", 
        "Noise", 
        "Seat_Available", 
        "Table_Available", 
        "Room_Available"
    ])
    
    # Generate 2000 students
    students_df = generate_students(
        num_groups=200,  # 200 groups
        group_size_range=(1, 20),  # Groups of 1-20 students
        brightness_mean=60,
        brightness_sd=20,
        noise_mean=40,
        noise_sd=10,
        flexibility_range=(1, 5)
    )
    
    # Ensure we have exactly 2000 students by adjusting group sizes if needed
    current_total = len(students_df)
    if current_total != 2000:
        print(f"Adjusting student count from {current_total} to 2000...")
        
        if current_total < 2000:
            # Add more students to existing groups
            needed = 2000 - current_total
            group_counts = students_df.groupby('Group_ID').size()
            
            for group_id in group_counts.index:
                if needed <= 0:
                    break
                # Add students to this group
                for i in range(min(needed, 5)):  # Add up to 5 students per group
                    if needed <= 0:
                        break
                    new_student = {
                        "Student_ID": f"S{len(students_df) + 1:03d}",
                        "Group_ID": group_id,
                        "Brightness": int(np.clip(np.random.normal(60, 20), 0, 100)),
                        "Noise": int(np.clip(np.random.normal(40, 10), 0, 100)),
                        "Flexibility": np.random.randint(1, 6)
                    }
                    students_df = pd.concat([students_df, pd.DataFrame([new_student])], ignore_index=True)
                    needed -= 1
        else:
            # Remove excess students (keep complete groups)
            excess = current_total - 2000
            group_counts = students_df.groupby('Group_ID').size()
            
            # Remove students from largest groups first
            for group_id in group_counts.sort_values(ascending=False).index:
                if excess <= 0:
                    break
                group_size = group_counts[group_id]
                if group_size > 1:  # Don't remove from single-student groups
                    remove_count = min(excess, group_size - 1)
                    # Remove the last students from this group
                    group_indices = students_df[students_df['Group_ID'] == group_id].index
                    if len(group_indices) > remove_count:
                        students_df = students_df.drop(group_indices[-remove_count:])
                        excess -= remove_count
    
    # Verify final counts
    print(f"Generated {len(seats_df)} seats and {len(students_df)} students")
    print(f"Number of groups: {students_df['Group_ID'].nunique()}")
    
    return seats_df, students_df

def save_experiment_dataset(seats_df, students_df, seats_path="seats.csv", students_path="students.csv"):
    """
    Save the generated experiment dataset to CSV files for reuse.
    """
    seats_df.to_csv(seats_path, index=False)
    students_df.to_csv(students_path, index=False)
    print(f"Saved experiment dataset to {seats_path} and {students_path}")

def load_experiment_dataset(seats_path="seats.csv", students_path="students.csv"):
    """
    Load the pre-generated experiment dataset from CSV files.
    If files don't exist, generate them first.
    """
    try:
        seats_df = pd.read_csv(seats_path)
        students_df = pd.read_csv(students_path)
        print(f"Loaded pre-generated dataset: {len(seats_df)} seats, {len(students_df)} students")
        return seats_df, students_df
    except FileNotFoundError:
        print("Pre-generated dataset not found. Generating new dataset...")
        seats_df, students_df = generate_experiment_dataset()
        save_experiment_dataset(seats_df, students_df, seats_path, students_path)
        return seats_df, students_df

def regenerate_experiment_dataset():
    """
    Regenerate the experiment dataset and save it to CSV files.
    This is useful for testing or if you want fresh data.
    """
    print("Regenerating experiment dataset...")
    seats_df, students_df = generate_experiment_dataset()
    save_experiment_dataset(seats_df, students_df)
    return seats_df, students_df

def run_experiments():
    """
    Run experiments comparing Greedy vs ILP performance across different dataset sizes.
    Creates plots showing objective value and execution time vs dataset size for both Q1 and Q2.
    Uses sequential loading from groups.csv: Set 1, Sets 1+2, Sets 1+2+3, Sets 1+2+3+4
    """
    # Dataset sizes to test (number of seats and students)
    seat_counts = [500, 1000, 1500, 2000]
    
    # Load the groups data
    print("Loading groups data...")
    groups_df = pd.read_csv('groups.csv')
    print(f"Loaded {len(groups_df)} students from groups.csv")
    
    # Run experiments for both query types
    for query_type in ['Q1', 'Q2']:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENTS FOR {query_type}")
        print(f"{'='*80}")
        
        # Results storage for this query
        greedy_results = {'sizes': [], 'objective_values': [], 'execution_times': [], 'success_rates': [], 'variances': []}
        ilp_results = {'sizes': [], 'objective_values': [], 'execution_times': [], 'success_rates': [], 'variances': []}
        
        print(f"Running experiments across different dataset sizes for {query_type}...")
        
        # Load the pre-generated dataset once
        seats_df, students_df = load_experiment_dataset()
        
        for seat_count in seat_counts:
            print(f"\nTesting with {seat_count} seats and {seat_count} students...")
            
            # Use the FIRST N seats (not random sampling) to ensure consistency
            seats_sample = seats_df.head(seat_count).copy()
            seats_sample['Seat_Available'] = True  # Reset availability
            
            # Load students based on the sequential set pattern
            if seat_count == 500:
                # Trial 1: Set 1 only
                students_sample = groups_df[groups_df['Set_ID'] == 1].copy()
            elif seat_count == 1000:
                # Trial 2: Sets 1 + 2
                students_sample = groups_df[groups_df['Set_ID'].isin([1, 2])].copy()
            elif seat_count == 1500:
                # Trial 3: Sets 1 + 2 + 3
                students_sample = groups_df[groups_df['Set_ID'].isin([1, 2, 3])].copy()
            elif seat_count == 2000:
                # Trial 4: Sets 1 + 2 + 3 + 4
                students_sample = groups_df[groups_df['Set_ID'].isin([1, 2, 3, 4])].copy()
            
            print(f"  Using seats 1-{seat_count} and groups from sets: {list(students_sample['Set_ID'].unique())}")
            print(f"  Seat range: {seats_sample['Seat_ID'].min()}-{seats_sample['Seat_ID'].max()}")
            print(f"  Group range: {students_sample['Group_ID'].min()}-{students_sample['Group_ID'].max()}")
            
            # Get all groups and their parameters from students data
            # Sort by Group_ID to ensure consistent order across algorithms
            groups = students_sample.groupby('Group_ID')
            # Convert to sorted list to ensure consistent processing order
            group_list = sorted(groups, key=lambda x: x[0])
            
            # Results storage for this dataset size
            greedy_results_for_size = []
            ilp_results_for_size = []
            
            # Set weights based on query type
            if query_type == 'Q1':
                w1, w2 = 1.0, 0.0
            else:  # Q2
                w1, w2 = 1.0, 0.3
            
            # Test each group separately
            for group_id, group_data in group_list:
                group_size = len(group_data)
                
                # Extract brightness_threshold from the group's data
                # Use the minimum brightness requirement from the group
                brightness_threshold = group_data['Brightness'].min()
                
                print(f"    Testing Group {group_id}: size={group_size}, brightness_threshold={brightness_threshold}")
                
                # Test Greedy Algorithm for this group
                greedy_success = False
                try:
                    table_stats = create_table_stats(seats_sample)
                    adjacency_graph = create_adjacency_graph(table_stats)
                    
                    start_time = time.time()
                    greedy_result = greedy_seat_selection(
                        group_size=group_size,
                        brightness_threshold=brightness_threshold,
                        table_stats=table_stats,
                        adjacency_graph=adjacency_graph,
                        w1=w1,
                        w2=w2
                    )
                    greedy_time = (time.time() - start_time) * 1000
                    
                    if greedy_result:
                        avg_brightness = np.mean([seat['Brightness'] for seat in greedy_result])
                        avg_noise = np.mean([seat['Noise'] for seat in greedy_result])
                        
                        # Calculate objective value based on query type
                        if query_type == 'Q1':
                            greedy_obj_value = avg_noise
                        else:  # Q2
                            greedy_obj_value = avg_noise - 0.3 * avg_brightness
                        
                        greedy_results_for_size.append({
                            'objective_value': greedy_obj_value,
                            'execution_time': greedy_time,
                            'success': True
                        })
                        greedy_success = True
                        
                        # Update seat availability for subsequent groups
                        for seat in greedy_result:
                            seat_mask = (seats_sample['Seat_ID'] == seat['Seat_ID'])
                            seats_sample.loc[seat_mask, 'Seat_Available'] = False
                    else:
                        greedy_results_for_size.append({
                            'objective_value': None,
                            'execution_time': greedy_time,
                            'success': False
                        })
                        
                except Exception as e:
                    print(f"      Greedy failed for group {group_id}: {e}")
                    greedy_results_for_size.append({
                        'objective_value': None,
                        'execution_time': None,
                        'success': False
                    })
                
                # Test ILP Algorithm for this group (only for smaller sizes)
                if seat_count <= 2000:
                    try:
                        # Prepare data matrix for ILP
                        available_seats = seats_sample[seats_sample['Seat_Available'] == True]
                        if len(available_seats) >= group_size:
                            data_matrix = available_seats[['Brightness', 'Noise', 'Room_ID', 'Table_ID']].values
                            
                            start_time = time.time()
                            
                            # Use appropriate ILP function based on query type
                            if query_type == 'Q1':
                                # Q1: minimize noise
                                picked, counts = solve_ilp(
                                    sub=data_matrix,
                                    size_=group_size,
                                    rep=1,
                                    obj={'attr': 1, 'pref': 'MIN'},  # Minimize noise
                                    cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
                                    verbose=False
                                )
                            else:
                                # Q2: minimize (noise - 0.3*brightness)
                                picked, counts = solve_ilp_weighted(
                                    sub=data_matrix,
                                    size_=group_size,
                                    rep=1,
                                    weights=[-0.3, 1.0, 0, 0],  # Match greedy objective
                                    cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
                                    verbose=False
                                )
                            ilp_time = (time.time() - start_time) * 1000
                            
                            if picked:
                                selected_seats = available_seats.iloc[picked]
                                avg_brightness = selected_seats['Brightness'].mean()
                                avg_noise = selected_seats['Noise'].mean()
                                
                                # Calculate objective value based on query type
                                if query_type == 'Q1':
                                    ilp_obj_value = avg_noise
                                else:  # Q2
                                    ilp_obj_value = avg_noise - 0.3 * avg_brightness
                                
                                ilp_results_for_size.append({
                                    'objective_value': ilp_obj_value,
                                    'execution_time': ilp_time,
                                    'success': True
                                })
                                
                                # Update seat availability for subsequent groups
                                for idx in picked:
                                    seat_id = available_seats.iloc[idx]['Seat_ID']
                                    seat_mask = (seats_sample['Seat_ID'] == seat_id)
                                    seats_sample.loc[seat_mask, 'Seat_Available'] = False
                            else:
                                ilp_results_for_size.append({
                                    'objective_value': None,
                                    'execution_time': ilp_time,
                                    'success': False
                                })
                        else:
                            ilp_results_for_size.append({
                                'objective_value': None,
                                'execution_time': None,
                                'success': False
                            })
                            
                    except Exception as e:
                        print(f"      ILP failed for group {group_id}: {e}")
                        ilp_results_for_size.append({
                            'objective_value': None,
                            'execution_time': None,
                            'success': False
                        })
            
            # Aggregate results for this dataset size
            if greedy_results_for_size:
                successful_greedy = [r for r in greedy_results_for_size if r['success']]
                greedy_success_rate = len(successful_greedy) / len(greedy_results_for_size)
                
                if successful_greedy:
                    avg_obj_greedy = np.mean([r['objective_value'] for r in successful_greedy])
                    avg_time_greedy = np.mean([r['execution_time'] for r in successful_greedy])
                    var_obj_greedy = np.var([r['objective_value'] for r in successful_greedy])
                    
                    greedy_results['sizes'].append(seat_count)
                    greedy_results['objective_values'].append(avg_obj_greedy)
                    greedy_results['execution_times'].append(avg_time_greedy)
                    greedy_results['success_rates'].append(greedy_success_rate)
                    greedy_results['variances'].append(var_obj_greedy)
                    
                    print(f"  Greedy Summary: {len(successful_greedy)}/{len(greedy_results_for_size)} groups successful ({greedy_success_rate*100:.1f}%)")
                    print(f"    Average objective: {avg_obj_greedy:.2f}, Variance: {var_obj_greedy:.2f}")
                else:
                    print(f"  Greedy Summary: 0/{len(greedy_results_for_size)} groups successful (0.0%)")
            
            if ilp_results_for_size and seat_count <= 2000:
                successful_ilp = [r for r in ilp_results_for_size if r['success']]
                ilp_success_rate = len(successful_ilp) / len(ilp_results_for_size)
                
                if successful_ilp:
                    avg_obj_ilp = np.mean([r['objective_value'] for r in successful_ilp])
                    avg_time_ilp = np.mean([r['execution_time'] for r in successful_ilp])
                    var_obj_ilp = np.var([r['objective_value'] for r in successful_ilp])
                    
                    ilp_results['sizes'].append(seat_count)
                    ilp_results['objective_values'].append(avg_obj_ilp)
                    ilp_results['execution_times'].append(avg_time_ilp)
                    ilp_results['success_rates'].append(ilp_success_rate)
                    ilp_results['variances'].append(var_obj_ilp)
                    
                    print(f"  ILP Summary: {len(successful_ilp)}/{len(ilp_results_for_size)} groups successful ({ilp_success_rate*100:.1f}%)")
                    print(f"    Average objective: {avg_obj_ilp:.2f}, Variance: {var_obj_ilp:.2f}")
                else:
                    print(f"  ILP Summary: 0/{len(ilp_results_for_size)} groups successful (0.0%)")
        
        # Print the data values for this query
        print(f"\n{'-'*60}")
        print(f"DATA VALUES FOR {query_type}")
        print(f"{'-'*60}")
        print(f"Greedy Results:")
        print(f"  Sizes: {greedy_results['sizes']}")
        print(f"  Objective Values: {[f'{val:.2f}' for val in greedy_results['objective_values']]}")
        print(f"  Execution Times (ms): {[f'{val:.2f}' for val in greedy_results['execution_times']]}")
        print(f"  Variances: {[f'{val:.2f}' for val in greedy_results['variances']]}")
        
        print(f"\nILP Results:")
        print(f"  Sizes: {ilp_results['sizes']}")
        print(f"  Objective Values: {[f'{val:.2f}' for val in ilp_results['objective_values']]}")
        print(f"  Execution Times (ms): {[f'{val:.2f}' for val in ilp_results['execution_times']]}")
        print(f"  Variances: {[f'{val:.2f}' for val in ilp_results['variances']]}")
        
        # Create plots for this query
        print(f"\nGenerating plots for {query_type}...")
        
        # Determine objective label based on query type
        if query_type == 'Q1':
            obj_label = 'Average Noise'
        else:
            obj_label = 'Average Noise - 0.3×Average Brightness'
        
        # Plot 1: Objective Value vs Dataset Size (with variance)
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        if greedy_results['sizes']:
            plt.plot(greedy_results['sizes'], greedy_results['objective_values'], 
                    'b-o', label='Greedy', linewidth=2, markersize=6)
            # Add variance bars
            for i, (size, obj_val, var) in enumerate(zip(greedy_results['sizes'], 
                                                          greedy_results['objective_values'], 
                                                          greedy_results['variances'])):
                plt.errorbar(size, obj_val, yerr=np.sqrt(var), fmt='none', color='blue', alpha=0.7)
        if ilp_results['sizes']:
            plt.plot(ilp_results['sizes'], ilp_results['objective_values'], 
                    'orange', marker='s', linestyle='-', label='Naive ILP', linewidth=2, markersize=6)
            # Add variance bars
            for i, (size, obj_val, var) in enumerate(zip(ilp_results['sizes'], 
                                                          ilp_results['objective_values'], 
                                                          ilp_results['variances'])):
                plt.errorbar(size, obj_val, yerr=np.sqrt(var), fmt='none', color='orange', alpha=0.7)
        
        plt.xlabel('Number of Seats/Students')
        plt.ylabel(f'Objective Value ({obj_label})')
        plt.title(f'{query_type}: Objective Value vs Dataset Size (with Variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Execution Time vs Dataset Size
        plt.subplot(2, 2, 2)
        if greedy_results['sizes']:
            plt.plot(greedy_results['sizes'], greedy_results['execution_times'], 
                    'b-o', label='Greedy', linewidth=2, markersize=6)
        if ilp_results['sizes']:
            plt.plot(ilp_results['sizes'], ilp_results['execution_times'], 
                    'orange', marker='s', linestyle='-', label='Naive ILP', linewidth=2, markersize=6)
        
        plt.xlabel('Number of Seats/Students')
        plt.ylabel('Execution Time (ms)')
        plt.title(f'{query_type}: Execution Time vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization of time differences
        
        # Plot 3: Variance vs Dataset Size
        plt.subplot(2, 2, 3)
        if greedy_results['sizes']:
            plt.plot(greedy_results['sizes'], greedy_results['variances'], 
                    'b-o', label='Greedy', linewidth=2, markersize=6)
        if ilp_results['sizes']:
            plt.plot(ilp_results['sizes'], ilp_results['variances'], 
                    'orange', marker='s', linestyle='-', label='Naive ILP', linewidth=2, markersize=6)
        
        plt.xlabel('Number of Seats/Students')
        plt.ylabel('Variance of Objective Values')
        plt.title(f'{query_type}: Variance vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Success Rate vs Dataset Size
        plt.subplot(2, 2, 4)
        if greedy_results['sizes']:
            plt.plot(greedy_results['sizes'], [r*100 for r in greedy_results['success_rates']], 
                    'b-o', label='Greedy', linewidth=2, markersize=6)
        if ilp_results['sizes']:
            plt.plot(ilp_results['sizes'], [r*100 for r in ilp_results['success_rates']], 
                    'orange', marker='s', linestyle='-', label='Naive ILP', linewidth=2, markersize=6)
        
        plt.xlabel('Number of Seats/Students')
        plt.ylabel('Success Rate (%)')
        plt.title(f'{query_type}: Success Rate vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'experiment_results_{query_type.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary for this query
        print(f"\n{'-'*60}")
        print(f"SUMMARY FOR {query_type}")
        print(f"{'-'*60}")
        print(f"Greedy Algorithm:")
        print(f"  - Tested on {len(greedy_results['sizes'])} dataset sizes")
        if greedy_results['execution_times']:
            print(f"  - Average execution time: {np.mean(greedy_results['execution_times']):.2f}ms")
        if greedy_results['objective_values']:
            print(f"  - Average objective value: {np.mean(greedy_results['objective_values']):.2f}")
        if greedy_results['variances']:
            print(f"  - Average variance: {np.mean(greedy_results['variances']):.2f}")
        
        print(f"\nILP Algorithm:")
        print(f"  - Tested on {len(ilp_results['sizes'])} dataset sizes")
        if ilp_results['execution_times']:
            print(f"  - Average execution time: {np.mean(ilp_results['execution_times']):.2f}ms")
        if ilp_results['objective_values']:
            print(f"  - Average objective value: {np.mean(ilp_results['objective_values']):.2f}")
        if ilp_results['variances']:
            print(f"  - Average variance: {np.mean(ilp_results['variances']):.2f}")
        
        # Calculate and print success rates
        print(f"\nSuccess Rates for {query_type}:")
        if greedy_results['sizes']:
            print(f"  Greedy Algorithm:")
            for i, size in enumerate(greedy_results['sizes']):
                if i < len(greedy_results['success_rates']):
                    success_rate = greedy_results['success_rates'][i] * 100
                    print(f"    {size} seats: {success_rate:.1f}% success rate")
        
        if ilp_results['sizes']:
            print(f"  ILP Algorithm:")
            for i, size in enumerate(ilp_results['sizes']):
                if i < len(ilp_results['success_rates']):
                    success_rate = ilp_results['success_rates'][i] * 100
                    print(f"    {size} seats: {success_rate:.1f}% success rate")
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print("Generated files: experiment_results_q1.png, experiment_results_q2.png")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Generate or load the experiment dataset first
    print("Setting up experiment dataset...")
    seats_df, students_df = load_experiment_dataset()
    
    # Example usage of package_queries function
    print("\nTesting package queries...")
    
    # Test greedy with Q1
    result_greedy_q1 = package_queries(model_type='greedy', query_type='Q1')
    obj_val_str = f"{result_greedy_q1['objective_value']:.2f}" if result_greedy_q1['objective_value'] is not None else 'N/A'
    print(f"Greedy Q1: Success={result_greedy_q1['success']}, "
          f"Objective={obj_val_str}, "
          f"Time={result_greedy_q1['execution_time_ms']:.2f}ms")
    
    # Test ILP with Q1
    result_ilp_q1 = package_queries(model_type='ilp', query_type='Q1')
    obj_val_str_ilp = f"{result_ilp_q1['objective_value']:.2f}" if result_ilp_q1['objective_value'] is not None else 'N/A'
    print(f"ILP Q1: Success={result_ilp_q1['success']}, "
          f"Objective={obj_val_str_ilp}, "
          f"Time={result_ilp_q1['execution_time_ms']:.2f}ms")
    
    # Run full experiments
    print("\nRunning full experiments...")
    run_experiments()
    
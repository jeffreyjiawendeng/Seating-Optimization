import pandas as pd
import numpy as np
from src.ilp import solve_ilp, solve_ilp_weighted, load_seat_data
from src.greedy import greedy_seat_selection, load_data, create_table_stats, create_adjacency_graph
from src.sketchrefine import sketchrefine_seat_selection as sequential_sketchrefine_seat_selection
import time
import matplotlib.pyplot as plt

def load_experiment_dataset():
    """Load the pre-generated experiment dataset from CSV files."""
    try:
        seats_df = pd.read_csv("seats.csv")
        students_df = pd.read_csv("students.csv")
        print(f"Loaded dataset: {len(seats_df)} seats, {len(students_df)} students")
        return seats_df, students_df
    except FileNotFoundError:
        print("Dataset files not found. Please ensure seats.csv and students.csv exist.")
        return None, None

def save_experiment_dataset(seats_df, students_df):
    """Save the generated dataset to CSV files."""
    seats_df.to_csv("seats.csv", index=False)
    students_df.to_csv("students.csv", index=False)
    print("Dataset saved to seats.csv and students.csv")

def generate_realistic_noise_layout(num_seats, num_rooms=10, num_tables_per_room=10):
    """Generate realistic noise layout with spatial correlation."""
    np.random.seed(42)  # For reproducibility
    
    seats = []
    seat_id = 1
    
    for room_id in range(1, num_rooms + 1):
        # Each room has different base noise levels
        room_base_noise = np.random.uniform(20, 80)
        
        for table_id in range(1, num_tables_per_room + 1):
            # Each table has slight variation from room base
            table_noise_offset = np.random.uniform(-10, 10)
            table_base_noise = room_base_noise + table_noise_offset
            
            # Generate seats for this table (4-8 seats per table)
            seats_per_table = np.random.randint(4, 9)
            
            for seat_num in range(1, seats_per_table + 1):
                # Each seat has small variation from table base
                seat_noise_offset = np.random.uniform(-5, 5)
                noise = max(0, min(100, table_base_noise + seat_noise_offset))
                
                # Brightness decreases with distance from windows (simulate room layout)
                brightness = max(0, min(100, 100 - (table_id * 3) + np.random.uniform(-10, 10)))
                
                seats.append({
                    'Seat_ID': seat_id,
                    'Table_ID': table_id + (room_id - 1) * num_tables_per_room,
                    'Room_ID': room_id,
                    'Brightness': brightness,
                    'Noise': noise,
                    'Seat_Available': True
                })
                seat_id += 1
                
                if seat_id > num_seats:
                    break
            if seat_id > num_seats:
                break
        if seat_id > num_seats:
            break
    
    return pd.DataFrame(seats)

def generate_experiment_dataset():
    """Generate a realistic dataset for experiments."""
    print("Generating experiment dataset...")
    
    # Generate seats with realistic noise layout
    seats_df = generate_realistic_noise_layout(2000, num_rooms=10, num_tables_per_room=20)
    
    # Generate students with realistic requirements
    np.random.seed(123)  # Different seed for students
    students = []
    student_id = 1
    
    # Create 200 groups with 10 students each (2000 total students)
    for group_id in range(1, 201):
        # Each group has similar but slightly varying requirements
        base_brightness = np.random.uniform(30, 70)
        base_noise = np.random.uniform(20, 60)
        
        for student_num in range(1, 11):  # 10 students per group
            # Individual students have slight variations from group requirements
            brightness_req = max(0, min(100, base_brightness + np.random.uniform(-5, 5)))
            noise_tolerance = max(0, min(100, base_noise + np.random.uniform(-5, 5)))
            flexibility = np.random.uniform(0.1, 0.3)
            
            students.append({
                'Student_ID': student_id,
                'Group_ID': group_id,
                'Brightness': brightness_req,
                'Noise': noise_tolerance,
                'Flexibility': flexibility,
                'Set_ID': ((group_id - 1) // 50) + 1  # Groups 1-50: Set 1, 51-100: Set 2, etc.
            })
            student_id += 1
    
    students_df = pd.DataFrame(students)
    
    print(f"Generated {len(seats_df)} seats and {len(students_df)} students")
    print(f"Seat range: {seats_df['Seat_ID'].min()}-{seats_df['Seat_ID'].max()}")
    print(f"Group range: {students_df['Group_ID'].min()}-{students_df['Group_ID'].max()}")
    print(f"Set distribution: {students_df['Set_ID'].value_counts().sort_index().to_dict()}")
    
    save_experiment_dataset(seats_df, students_df)
    return seats_df, students_df

def regenerate_experiment_dataset():
    """Regenerate the experiment dataset."""
    print("Regenerating experiment dataset...")
    seats_df, students_df = generate_experiment_dataset()
    return seats_df, students_df

def run_experiments():
    """
    Run experiments comparing Greedy vs ILP vs SketchRefine performance across different dataset sizes.
    Creates plots showing objective value and execution time vs dataset size for both Q1 and Q2.
    Uses sequential loading from groups.csv: Set 1, Sets 1+2, Sets 1+2+3, Sets 1+2+3+4
    
    MODIFIED: For quad graph mode, seats are consumed during each trial and reset between trials and modes.
    """
    # Dataset sizes to test (number of seats and students)
    seat_counts = [500, 1000, 1500, 2000]
    
    # Load the groups data
    print("Loading groups data...")
    groups_df = pd.read_csv('src/groups.csv')
    print(f"Loaded {len(groups_df)} students from groups.csv")
    
    # Run experiments for both query types
    for query_type in ['Q1', 'Q2']:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENTS FOR {query_type}")
        print(f"{'='*80}")
        
        # Results storage for this query
        greedy_results = {'sizes': [], 'objective_values': [], 'execution_times': [], 'success_rates': [], 'variances': []}
        ilp_results = {'sizes': [], 'objective_values': [], 'execution_times': [], 'success_rates': [], 'variances': []}
        sketchrefine_results = {'sizes': [], 'objective_values': [], 'execution_times': [], 'success_rates': [], 'variances': []}
        
        print(f"Running experiments across different dataset sizes for {query_type}...")
        
        # Load the pre-generated dataset once (seats are reset to available for each query type)
        seats_df, students_df = load_experiment_dataset()
        
        for seat_count in seat_counts:
            print(f"\nTesting with {seat_count} seats and {seat_count} students...")
            
            # Use the FIRST N seats (not random sampling) to ensure consistency
            # Different portions of the same dataset: 1-500, 1-1000, 1-1500, 1-2000
            seats_sample = seats_df.head(seat_count).copy()
            seats_sample['Seat_Available'] = True  # Reset availability for this trial
            
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
            # Sort by Group_ID to ensure consistent order across algorithms and trials
            # Groups are processed in the same order: 1, 2, 3, 4, 5, ..., N
            groups = students_sample.groupby('Group_ID')
            # Convert to sorted list to ensure consistent processing order
            group_list = sorted(groups, key=lambda x: x[0])
            
            # Results storage for this dataset size
            greedy_results_for_size = []
            ilp_results_for_size = []
            sketchrefine_results_for_size = []
            
            # Set weights based on query type
            if query_type == 'Q1':
                w1, w2 = 1.0, 0.0
            else:  # Q2
                w1, w2 = 1.0, 0.3
            
            # MODIFIED: Test each algorithm separately with seat consumption within each trial
            algorithms = ['greedy', 'ilp', 'sketchrefine']
            
            for algorithm in algorithms:
                print(f"    Testing {algorithm.upper()} algorithm...")
                
                # Reset seats for this algorithm (fresh start for each algorithm)
                algorithm_seats = seats_sample.copy()
                algorithm_seats['Seat_Available'] = True
                
                # Test each group for this algorithm
                for group_id, group_data in group_list:
                    group_size = len(group_data)
                    brightness_threshold = group_data['Brightness'].min()
                    
                    if algorithm == 'greedy':
                        try:
                            table_stats = create_table_stats(algorithm_seats)
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
                                
                                # MODIFIED: Mark assigned seats as unavailable for subsequent groups in this trial
                                for seat in greedy_result:
                                    seat_mask = (algorithm_seats['Seat_ID'] == seat['Seat_ID'])
                                    algorithm_seats.loc[seat_mask, 'Seat_Available'] = False
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
                    
                    elif algorithm == 'ilp' and seat_count <= 2000:
                        try:
                            # Prepare data matrix for ILP
                            available_seats = algorithm_seats[algorithm_seats['Seat_Available'] == True]
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
                                    
                                    # MODIFIED: Mark assigned seats as unavailable for subsequent groups in this trial
                                    for idx in picked:
                                        seat_id = available_seats.iloc[idx]['Seat_ID']
                                        seat_mask = (algorithm_seats['Seat_ID'] == seat_id)
                                        algorithm_seats.loc[seat_mask, 'Seat_Available'] = False
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
                    
                    elif algorithm == 'sketchrefine':
                        try:
                            start_time = time.time()
                            sketchrefine_result = sequential_sketchrefine_seat_selection(
                                group_size=group_size,
                                brightness_threshold=brightness_threshold,
                                seats_df=algorithm_seats,
                                query_type=query_type
                            )
                            
                            if sketchrefine_result and sketchrefine_result['success']:
                                avg_brightness = np.mean([seat['Brightness'] for seat in sketchrefine_result['seats']])
                                avg_noise = np.mean([seat['Noise'] for seat in sketchrefine_result['seats']])
                                
                                # Calculate objective value based on query type
                                if query_type == 'Q1':
                                    sketchrefine_obj_value = avg_noise
                                else:  # Q2
                                    sketchrefine_obj_value = avg_noise - 0.3 * avg_brightness
                                
                                sketchrefine_results_for_size.append({
                                    'objective_value': sketchrefine_obj_value,
                                    'execution_time': sketchrefine_result['execution_time_ms'],
                                    'success': True
                                })
                                
                                # MODIFIED: Mark assigned seats as unavailable for subsequent groups in this trial
                                for seat in sketchrefine_result['seats']:
                                    seat_mask = (algorithm_seats['Seat_ID'] == seat['Seat_ID'])
                                    algorithm_seats.loc[seat_mask, 'Seat_Available'] = False
                            else:
                                sketchrefine_results_for_size.append({
                                    'objective_value': None,
                                    'execution_time': sketchrefine_result['execution_time_ms'] if sketchrefine_result else None,
                                    'success': False
                                })
                                
                        except Exception as e:
                            print(f"      SketchRefine failed for group {group_id}: {e}")
                            sketchrefine_results_for_size.append({
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
            
            if sketchrefine_results_for_size:
                successful_sketchrefine = [r for r in sketchrefine_results_for_size if r['success']]
                sketchrefine_success_rate = len(successful_sketchrefine) / len(sketchrefine_results_for_size)
                
                if successful_sketchrefine:
                    avg_obj_sketchrefine = np.mean([r['objective_value'] for r in successful_sketchrefine])
                    avg_time_sketchrefine = np.mean([r['execution_time'] for r in successful_sketchrefine])
                    var_obj_sketchrefine = np.var([r['objective_value'] for r in successful_sketchrefine])
                    
                    sketchrefine_results['sizes'].append(seat_count)
                    sketchrefine_results['objective_values'].append(avg_obj_sketchrefine)
                    sketchrefine_results['execution_times'].append(avg_time_sketchrefine)
                    sketchrefine_results['success_rates'].append(sketchrefine_success_rate)
                    sketchrefine_results['variances'].append(var_obj_sketchrefine)
                    
                    print(f"  SketchRefine Summary: {len(successful_sketchrefine)}/{len(sketchrefine_results_for_size)} groups successful ({sketchrefine_success_rate*100:.1f}%)")
                    print(f"    Average objective: {avg_obj_sketchrefine:.2f}, Variance: {var_obj_sketchrefine:.2f}")
                else:
                    print(f"  SketchRefine Summary: 0/{len(sketchrefine_results_for_size)} groups successful (0.0%)")
        
        # Print summary results
        print(f"\n{query_type} Results Summary:")
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
        
        print(f"\nSketchRefine Results:")
        print(f"  Sizes: {sketchrefine_results['sizes']}")
        print(f"  Objective Values: {[f'{val:.2f}' for val in sketchrefine_results['objective_values']]}")
        print(f"  Execution Times (ms): {[f'{val:.2f}' for val in sketchrefine_results['execution_times']]}")
        print(f"  Variances: {[f'{val:.2f}' for val in sketchrefine_results['variances']]}")
        
        # Create plots for this query
        print(f"\nGenerating plots for {query_type}...")
        
        # Determine objective label based on query type
        if query_type == 'Q1':
            obj_label = 'Average Noise'
        else:
            obj_label = 'Average Noise - 0.3×Average Brightness'
        
        # Create 4-panel comparison plots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Objective Value vs Dataset Size
        plt.subplot(2, 2, 1)
        if greedy_results['sizes']:
            plt.plot(greedy_results['sizes'], greedy_results['objective_values'], 'b-o', label='Greedy', linewidth=2, markersize=6)
        if ilp_results['sizes']:
            plt.plot(ilp_results['sizes'], ilp_results['objective_values'], 'orange', marker='s', linestyle='-', label='ILP', linewidth=2, markersize=6)
        if sketchrefine_results['sizes']:
            plt.plot(sketchrefine_results['sizes'], sketchrefine_results['objective_values'], 'g-^', label='SketchRefine', linewidth=2, markersize=6)
        plt.xlabel('Dataset Size (Seats)')
        plt.ylabel(f'Objective Value ({obj_label})')
        plt.title(f'{query_type}: Objective Value vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Execution Time vs Dataset Size
        plt.subplot(2, 2, 2)
        if greedy_results['sizes']:
            plt.plot(greedy_results['sizes'], greedy_results['execution_times'], 'b-o', label='Greedy', linewidth=2, markersize=6)
        if ilp_results['sizes']:
            plt.plot(ilp_results['sizes'], ilp_results['execution_times'], 'orange', marker='s', linestyle='-', label='ILP', linewidth=2, markersize=6)
        if sketchrefine_results['sizes']:
            plt.plot(sketchrefine_results['sizes'], sketchrefine_results['execution_times'], 'g-^', label='SketchRefine', linewidth=2, markersize=6)
        plt.xlabel('Dataset Size (Seats)')
        plt.ylabel('Execution Time (ms)')
        plt.title(f'{query_type}: Execution Time vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Success Rate vs Dataset Size
        plt.subplot(2, 2, 3)
        if greedy_results['sizes']:
            plt.plot(greedy_results['sizes'], greedy_results['success_rates'], 'b-o', label='Greedy', linewidth=2, markersize=6)
        if ilp_results['sizes']:
            plt.plot(ilp_results['sizes'], ilp_results['success_rates'], 'orange', marker='s', linestyle='-', label='ILP', linewidth=2, markersize=6)
        if sketchrefine_results['sizes']:
            plt.plot(sketchrefine_results['sizes'], sketchrefine_results['success_rates'], 'g-^', label='SketchRefine', linewidth=2, markersize=6)
        plt.xlabel('Dataset Size (Seats)')
        plt.ylabel('Success Rate')
        plt.title(f'{query_type}: Success Rate vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        # Plot 4: Variance vs Dataset Size
        plt.subplot(2, 2, 4)
        if greedy_results['sizes']:
            plt.plot(greedy_results['sizes'], greedy_results['variances'], 'b-o', label='Greedy', linewidth=2, markersize=6)
        if ilp_results['sizes']:
            plt.plot(ilp_results['sizes'], ilp_results['variances'], 'orange', marker='s', linestyle='-', label='ILP', linewidth=2, markersize=6)
        if sketchrefine_results['sizes']:
            plt.plot(sketchrefine_results['sizes'], sketchrefine_results['variances'], 'g-^', label='SketchRefine', linewidth=2, markersize=6)
        plt.xlabel('Dataset Size (Seats)')
        plt.ylabel('Objective Value Variance')
        plt.title(f'{query_type}: Variance vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'experiment_results_{query_type.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Run per-group experiment and create additional plot
        print(f"\nRunning per-group experiment for {query_type}...")
        per_group_results = run_per_group_experiment(query_type)
        
        # Create per-group objective value plot
        plt.figure(figsize=(15, 8))
        
        # Plot per-group objective values
        group_numbers = range(1, len(per_group_results["group_ids"]) + 1)
        
        # Filter out None values for plotting
        greedy_valid = [(i+1, val) for i, val in enumerate(per_group_results["greedy_per_group"]) if val is not None]
        ilp_valid = [(i+1, val) for i, val in enumerate(per_group_results["ilp_per_group"]) if val is not None]
        sketchrefine_valid = [(i+1, val) for i, val in enumerate(per_group_results["sketchrefine_per_group"]) if val is not None]
        
        if greedy_valid:
            greedy_groups, greedy_vals = zip(*greedy_valid)
            plt.plot(greedy_groups, greedy_vals, 'b.-', label=f'Greedy ({len(greedy_valid)} groups)', alpha=0.8, markersize=3, linewidth=1)
        
        if ilp_valid:
            ilp_groups, ilp_vals = zip(*ilp_valid)
            plt.plot(ilp_groups, ilp_vals, 'orange', marker='.', linestyle='-', label=f'ILP ({len(ilp_valid)} groups)', alpha=0.8, markersize=3, linewidth=1)
        
        if sketchrefine_valid:
            sketchrefine_groups, sketchrefine_vals = zip(*sketchrefine_valid)
            plt.plot(sketchrefine_groups, sketchrefine_vals, 'g.-', label=f'SketchRefine ({len(sketchrefine_valid)} groups)', alpha=0.8, markersize=3, linewidth=1)
        
        plt.xlabel('Group Number')
        plt.ylabel(f'Objective Value ({obj_label})')
        plt.title(f'{query_type}: Per-Group Objective Values (All 2000 Students)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'\nStatistics:\n'
        if greedy_valid:
            stats_text += f'Greedy: Mean={np.mean(greedy_vals):.2f}, Std={np.std(greedy_vals):.2f}\n'
        if ilp_valid:
            stats_text += f'ILP: Mean={np.mean(ilp_vals):.2f}, Std={np.std(ilp_vals):.2f}\n'
        if sketchrefine_valid:
            stats_text += f'SketchRefine: Mean={np.mean(sketchrefine_vals):.2f}, Std={np.std(sketchrefine_vals):.2f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'per_group_objectives_{query_type.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()

def run_per_group_experiment(query_type):
    """
    Run experiments on all 2000 students, tracking objective value per group for all three algorithms.
    
    FIXED: All algorithms now process the same groups in the same order.
    Each algorithm gets fresh seats but processes the same group sequence.
    
    Args:
        query_type (str): 'Q1' or 'Q2' for different objectives
        
    Returns:
        dict: Results with per-group objective values for each algorithm
    """
    print(f"\n{'='*80}")
    print(f"RUNNING PER-GROUP EXPERIMENT FOR {query_type}")
    print(f"(All algorithms process same groups with fresh seats)")
    print(f"{'='*80}")
    
    # Load the full dataset
    seats_df, students_df = load_experiment_dataset()
    
    # Get all groups and their parameters from students data
    groups = students_df.groupby('Group_ID')
    group_list = sorted(groups, key=lambda x: x[0])
    
    # Set weights based on query type
    if query_type == 'Q1':
        w1, w2 = 1.0, 0.0
    else:  # Q2
        w1, w2 = 1.0, 0.3
    
    # Results storage
    greedy_per_group = []
    ilp_per_group = []
    sketchrefine_per_group = []
    
    print(f"Processing {len(group_list)} groups...")
    
    # FIXED: Test each algorithm separately with fresh seats for each algorithm
    algorithms = ['greedy', 'ilp', 'sketchrefine']
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm.upper()} algorithm with fresh seats...")
        
        # Reset seats for this algorithm (fresh start for each algorithm)
        algorithm_seats = seats_df.copy()
        algorithm_seats['Seat_Available'] = True
        
        # Process each group for this algorithm
        for group_idx, (group_id, group_data) in enumerate(group_list):
            if group_idx % 50 == 0:  # Progress indicator every 50 groups
                available_seats = algorithm_seats['Seat_Available'].sum()
                print(f"  Processing group {group_idx + 1}/{len(group_list)}... ({available_seats} seats remaining)")
            
            group_size = len(group_data)
            brightness_threshold = group_data['Brightness'].min()
            
            if algorithm == 'greedy':
                try:
                    table_stats = create_table_stats(algorithm_seats)
                    adjacency_graph = create_adjacency_graph(table_stats)
                    
                    greedy_result = greedy_seat_selection(
                        group_size=group_size,
                        brightness_threshold=brightness_threshold,
                        table_stats=table_stats,
                        adjacency_graph=adjacency_graph,
                        w1=w1,
                        w2=w2
                    )
                    
                    if greedy_result:
                        avg_brightness = np.mean([seat['Brightness'] for seat in greedy_result])
                        avg_noise = np.mean([seat['Noise'] for seat in greedy_result])
                        
                        if query_type == 'Q1':
                            greedy_obj_value = avg_noise
                        else:  # Q2
                            greedy_obj_value = avg_noise - 0.3 * avg_brightness
                        
                        greedy_per_group.append(greedy_obj_value)
                        
                        # Mark assigned seats as unavailable for subsequent groups
                        for seat in greedy_result:
                            seat_mask = (algorithm_seats['Seat_ID'] == seat['Seat_ID'])
                            algorithm_seats.loc[seat_mask, 'Seat_Available'] = False
                    else:
                        greedy_per_group.append(None)
                        
                except Exception as e:
                    greedy_per_group.append(None)
            
            elif algorithm == 'ilp':
                try:
                    available_seats = algorithm_seats[algorithm_seats['Seat_Available'] == True]
                    if len(available_seats) >= group_size:
                        data_matrix = available_seats[['Brightness', 'Noise', 'Room_ID', 'Table_ID']].values
                        
                        if query_type == 'Q1':
                            picked, counts = solve_ilp(
                                sub=data_matrix,
                                size_=group_size,
                                rep=1,
                                obj={'attr': 1, 'pref': 'MIN'},
                                cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
                                verbose=False
                            )
                        else:
                            picked, counts = solve_ilp_weighted(
                                sub=data_matrix,
                                size_=group_size,
                                rep=1,
                                weights=[-0.3, 1.0, 0, 0],
                                cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
                                verbose=False
                            )
                        
                        if picked:
                            selected_seats = available_seats.iloc[picked]
                            avg_brightness = selected_seats['Brightness'].mean()
                            avg_noise = selected_seats['Noise'].mean()
                            
                            if query_type == 'Q1':
                                ilp_obj_value = avg_noise
                            else:  # Q2
                                ilp_obj_value = avg_noise - 0.3 * avg_brightness
                            
                            ilp_per_group.append(ilp_obj_value)
                            
                            # Mark assigned seats as unavailable for subsequent groups
                            for idx in picked:
                                seat_id = available_seats.iloc[idx]['Seat_ID']
                                seat_mask = (algorithm_seats['Seat_ID'] == seat_id)
                                algorithm_seats.loc[seat_mask, 'Seat_Available'] = False
                        else:
                            ilp_per_group.append(None)
                    else:
                        ilp_per_group.append(None)
                        
                except Exception as e:
                    ilp_per_group.append(None)
            
            elif algorithm == 'sketchrefine':
                try:
                    sketchrefine_result = sequential_sketchrefine_seat_selection(
                        group_size=group_size,
                        brightness_threshold=brightness_threshold,
                        seats_df=algorithm_seats,
                        query_type=query_type
                    )
                    
                    if sketchrefine_result and sketchrefine_result['success']:
                        avg_brightness = np.mean([seat['Brightness'] for seat in sketchrefine_result['seats']])
                        avg_noise = np.mean([seat['Noise'] for seat in sketchrefine_result['seats']])
                        
                        if query_type == 'Q1':
                            sketchrefine_obj_value = avg_noise
                        else:  # Q2
                            sketchrefine_obj_value = avg_noise - 0.3 * avg_brightness
                        
                        sketchrefine_per_group.append(sketchrefine_obj_value)
                        
                        # Mark assigned seats as unavailable for subsequent groups
                        for seat in sketchrefine_result['seats']:
                            seat_mask = (algorithm_seats['Seat_ID'] == seat['Seat_ID'])
                            algorithm_seats.loc[seat_mask, 'Seat_Available'] = False
                    else:
                        sketchrefine_per_group.append(None)
                        
                except Exception as e:
                    sketchrefine_per_group.append(None)
        
        print(f"Completed {algorithm.upper()} algorithm")
    
    print(f"Completed per-group experiment for {query_type}")
    
    return {
        'greedy_per_group': greedy_per_group,
        'ilp_per_group': ilp_per_group,
        'sketchrefine_per_group': sketchrefine_per_group,
        'group_ids': [group_id for group_id, _ in group_list]
    }

if __name__ == "__main__":
    run_experiments()

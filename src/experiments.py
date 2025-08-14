import pandas as pd
import numpy as np
from ilp import solve_ilp, load_seat_data
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
        seats_df, students_df = load_data()
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
        sub, seats_df, students_df = load_seat_data()
        
        if query_type == 'Q1':
            # Q1: minimize total noise
            obj = {'attr': 1, 'pref': 'MIN'}  # attr 1 is Noise
            cons = [
                {'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}  # Total brightness >= threshold * size
            ]
        else:
            # Q2: minimize (noise - 0.3*brightness) - this needs custom objective
            # For ILP, we'll approximate by minimizing noise with brightness constraint
            obj = {'attr': 1, 'pref': 'MIN'}  # Minimize noise
            cons = [
                {'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}  # Total brightness >= threshold * size
            ]
        
        # Run ILP
        picked, counts = solve_ilp(
            sub=sub,
            size_=group_size,
            rep=1,
            obj=obj,
            cons=cons,
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

def run_experiments():
    """
    Run experiments comparing Greedy vs ILP performance across different dataset sizes.
    Creates plots showing objective value and execution time vs dataset size for both Q1 and Q2.
    """
    # Dataset sizes to test (number of seats)
    seat_counts = [100, 500, 1000, 2000, 5000, 10000]
    
    # Run experiments for both query types
    for query_type in ['Q1', 'Q2']:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENTS FOR {query_type}")
        print(f"{'='*80}")
        
        # Results storage for this query
        greedy_results = {'sizes': [], 'objective_values': [], 'execution_times': [], 'success_rates': []}
        ilp_results = {'sizes': [], 'objective_values': [], 'execution_times': [], 'success_rates': []}
        
        print(f"Running experiments across different dataset sizes for {query_type}...")
        
        for seat_count in seat_counts:
            print(f"\nTesting with {seat_count} seats...")
            
            # Generate or sample data for this size
            seats_df, students_df = load_data()
            
            # Generate seats data for this count
            if len(seats_df) < seat_count:
                # If we need more seats than available, replicate the dataset
                multiplier = (seat_count // len(seats_df)) + 1
                expanded_seats = []
                for i in range(multiplier):
                    temp_df = seats_df.copy()
                    temp_df['Seat_ID'] = temp_df['Seat_ID'] + i * len(seats_df)
                    temp_df['Table_ID'] = temp_df['Table_ID'] + i * seats_df['Table_ID'].max()
                    expanded_seats.append(temp_df)
                seats_df = pd.concat(expanded_seats, ignore_index=True)
            
            # Sample to exact size needed
            seats_sample = seats_df.sample(n=seat_count, random_state=42).copy()
            seats_sample['Seat_Available'] = True  # Reset availability
            
            # Generate students data with roughly equal count to seats
            student_count = seat_count  # Make student count equal to seat count
            if len(students_df) < student_count:
                # If we need more students than available, replicate the dataset
                multiplier = (student_count // len(students_df)) + 1
                expanded_students = []
                for i in range(multiplier):
                    temp_df = students_df.copy()
                    # Update Student_ID and Group_ID to avoid conflicts
                    # Handle string IDs by adding suffix
                    temp_df['Student_ID'] = temp_df['Student_ID'].astype(str) + f"_rep{i}"
                    temp_df['Group_ID'] = temp_df['Group_ID'] + i * students_df['Group_ID'].max()
                    expanded_students.append(temp_df)
                students_df = pd.concat(expanded_students, ignore_index=True)
            
            # Sample to exact size needed
            students_sample = students_df.sample(n=student_count, random_state=42).copy()
            
            print(f"  Generated {len(seats_sample)} seats and {len(students_sample)} students")
            
            # Get all groups and their parameters from students data
            groups = students_sample.groupby('Group_ID')
            
            # Results storage for this dataset size
            greedy_results_for_size = []
            ilp_results_for_size = []
            
            # Set weights based on query type
            if query_type == 'Q1':
                w1, w2 = 1.0, 0.0
            else:  # Q2
                w1, w2 = 1.0, 0.3
            
            # Test each group separately
            for group_id, group_data in groups:
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
                            picked, counts = solve_ilp(
                                sub=data_matrix,
                                size_=group_size,
                                rep=1,
                                obj={'attr': 1, 'pref': 'MIN'},  # Minimize noise
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
                    
                    greedy_results['sizes'].append(seat_count)
                    greedy_results['objective_values'].append(avg_obj_greedy)
                    greedy_results['execution_times'].append(avg_time_greedy)
                    greedy_results['success_rates'].append(greedy_success_rate)
                    
                    print(f"  Greedy Summary: {len(successful_greedy)}/{len(greedy_results_for_size)} groups successful ({greedy_success_rate*100:.1f}%)")
                else:
                    print(f"  Greedy Summary: 0/{len(greedy_results_for_size)} groups successful (0.0%)")
            
            if ilp_results_for_size and seat_count <= 2000:
                successful_ilp = [r for r in ilp_results_for_size if r['success']]
                ilp_success_rate = len(successful_ilp) / len(ilp_results_for_size)
                
                if successful_ilp:
                    avg_obj_ilp = np.mean([r['objective_value'] for r in successful_ilp])
                    avg_time_ilp = np.mean([r['execution_time'] for r in successful_ilp])
                    
                    ilp_results['sizes'].append(seat_count)
                    ilp_results['objective_values'].append(avg_obj_ilp)
                    ilp_results['execution_times'].append(avg_time_ilp)
                    ilp_results['success_rates'].append(ilp_success_rate)
                    
                    print(f"  ILP Summary: {len(successful_ilp)}/{len(ilp_results_for_size)} groups successful ({ilp_success_rate*100:.1f}%)")
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
        
        print(f"\nILP Results:")
        print(f"  Sizes: {ilp_results['sizes']}")
        print(f"  Objective Values: {[f'{val:.2f}' for val in ilp_results['objective_values']]}")
        print(f"  Execution Times (ms): {[f'{val:.2f}' for val in ilp_results['execution_times']]}")
        
        # Create plots for this query
        print(f"\nGenerating plots for {query_type}...")
        
        # Determine objective label based on query type
        if query_type == 'Q1':
            obj_label = 'Average Noise'
        else:
            obj_label = 'Average Noise - 0.3×Average Brightness'
        
        # Plot 1: Objective Value vs Dataset Size
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        if greedy_results['sizes']:
            plt.plot(greedy_results['sizes'], greedy_results['objective_values'], 
                    'b-o', label='Greedy', linewidth=2, markersize=6)
        if ilp_results['sizes']:
            plt.plot(ilp_results['sizes'], ilp_results['objective_values'], 
                    'orange', marker='s', linestyle='-', label='Naive ILP', linewidth=2, markersize=6)
        
        plt.xlabel('Number of Seats')
        plt.ylabel(f'Objective Value ({obj_label})')
        plt.title(f'{query_type}: Objective Value vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Execution Time vs Dataset Size
        plt.subplot(1, 2, 2)
        if greedy_results['sizes']:
            plt.plot(greedy_results['sizes'], greedy_results['execution_times'], 
                    'b-o', label='Greedy', linewidth=2, markersize=6)
        if ilp_results['sizes']:
            plt.plot(ilp_results['sizes'], ilp_results['execution_times'], 
                    'orange', marker='s', linestyle='-', label='Naive ILP', linewidth=2, markersize=6)
        
        plt.xlabel('Number of Seats')
        plt.ylabel('Execution Time (ms)')
        plt.title(f'{query_type}: Execution Time vs Dataset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization of time differences
        
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
        
        print(f"\nILP Algorithm:")
        print(f"  - Tested on {len(ilp_results['sizes'])} dataset sizes")
        if ilp_results['execution_times']:
            print(f"  - Average execution time: {np.mean(ilp_results['execution_times']):.2f}ms")
        if ilp_results['objective_values']:
            print(f"  - Average objective value: {np.mean(ilp_results['objective_values']):.2f}")
        
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
    # Example usage of package_queries function
    print("Testing package queries...")
    
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
    
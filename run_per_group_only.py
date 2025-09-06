import pandas as pd
import numpy as np
from src.ilp import solve_ilp, solve_ilp_weighted
from src.greedy import greedy_seat_selection, create_table_stats, create_adjacency_graph
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

def run_per_group_experiment(query_type):
    """
    Run experiments on all 2000 students, tracking objective value per group for all three algorithms.
    
    Args:
        query_type (str): 'Q1' or 'Q2' for different objectives
        
    Returns:
        dict: Results with per-group objective values for each algorithm
    """
    print(f"\n{'='*80}")
    print(f"RUNNING PER-GROUP EXPERIMENT FOR {query_type}")
    print(f"{'='*80}")
    
    # Load the full dataset
    seats_df, students_df = load_experiment_dataset()
    if seats_df is None:
        return None
    
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
    
    # Process each group
    for group_idx, (group_id, group_data) in enumerate(group_list):
        if group_idx % 10 == 0:  # Progress indicator every 10 groups
            successful_greedy = len([x for x in greedy_per_group if x is not None])
            successful_ilp = len([x for x in ilp_per_group if x is not None])
            successful_sketchrefine = len([x for x in sketchrefine_per_group if x is not None])
            print(f"  Processing group {group_idx + 1}/{len(group_list)}... (Greedy: {successful_greedy}, ILP: {successful_ilp}, SketchRefine: {successful_sketchrefine} successful)")
        
        group_size = len(group_data)
        brightness_threshold = group_data['Brightness'].min()
        
        # Test Greedy Algorithm
        try:
            greedy_seats = seats_df.copy()
            greedy_seats['Seat_Available'] = True
            
            table_stats = create_table_stats(greedy_seats)
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
            else:
                greedy_per_group.append(None)
                
        except Exception as e:
            greedy_per_group.append(None)
        
        # Test ILP Algorithm
        try:
            ilp_seats = seats_df.copy()
            ilp_seats['Seat_Available'] = True
            
            available_seats = ilp_seats[ilp_seats['Seat_Available'] == True]
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
                else:
                    ilp_per_group.append(None)
            else:
                ilp_per_group.append(None)
                
        except Exception as e:
            ilp_per_group.append(None)
        
        # Test SketchRefine Algorithm
        try:
            sketchrefine_seats = seats_df.copy()
            sketchrefine_seats['Seat_Available'] = True
            
            sketchrefine_result = sequential_sketchrefine_seat_selection(
                group_size=group_size,
                brightness_threshold=brightness_threshold,
                seats_df=sketchrefine_seats,
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
            else:
                sketchrefine_per_group.append(None)
                
        except Exception as e:
            sketchrefine_per_group.append(None)
    
    print(f"Completed per-group experiment for {query_type}")
    
    return {
        'greedy_per_group': greedy_per_group,
        'ilp_per_group': ilp_per_group,
        'sketchrefine_per_group': sketchrefine_per_group,
        'group_ids': [group_id for group_id, _ in group_list]
    }

def create_per_group_plot(query_type, per_group_results):
    """Create the per-group objective value plot with connected lines."""
    print(f"\nGenerating per-group plot for {query_type}...")
    
    # Create per-group objective value plot
    plt.figure(figsize=(15, 8))
    
    # Plot per-group objective values
    group_numbers = range(1, len(per_group_results['group_ids']) + 1)
    
    # Filter out None values for plotting
    greedy_valid = [(i+1, val) for i, val in enumerate(per_group_results['greedy_per_group']) if val is not None]
    ilp_valid = [(i+1, val) for i, val in enumerate(per_group_results['ilp_per_group']) if val is not None]
    sketchrefine_valid = [(i+1, val) for i, val in enumerate(per_group_results['sketchrefine_per_group']) if val is not None]
    
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
    if query_type == 'Q1':
        plt.ylabel('Objective Value (Average Noise)')
    else:
        plt.ylabel('Objective Value (Average Noise - 0.3×Average Brightness)')
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
    
    print(f"Generated per_group_objectives_{query_type.lower()}.png")

def main():
    """Run per-group experiments for both Q1 and Q2."""
    print("Starting per-group experiments for Q1 and Q2...")
    
    # Run Q1 experiment
    print("\n" + "="*80)
    print("RUNNING Q1 PER-GROUP EXPERIMENT")
    print("="*80)
    q1_results = run_per_group_experiment('Q1')
    if q1_results:
        create_per_group_plot('Q1', q1_results)
    
    # Run Q2 experiment
    print("\n" + "="*80)
    print("RUNNING Q2 PER-GROUP EXPERIMENT")
    print("="*80)
    q2_results = run_per_group_experiment('Q2')
    if q2_results:
        create_per_group_plot('Q2', q2_results)
    
    print("\n" + "="*80)
    print("ALL PER-GROUP EXPERIMENTS COMPLETED")
    print("Generated files: per_group_objectives_q1.png, per_group_objectives_q2.png")
    print("="*80)

if __name__ == "__main__":
    main()

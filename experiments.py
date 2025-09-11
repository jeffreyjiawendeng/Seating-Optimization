# Seating Optimization Algorithm Comparison Experiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from seating_opt.data_gen import generate_seats, generate_groups
from seating_opt.greedy import greedy_pairwise
from seating_opt.ilp_solvers import solve_group_pair_ilp, solve_global_pair_ilp
from seating_opt.sketchrefine import sketchrefine_solver
from seating_opt.distance import pairwise_distances


def calculate_objective(seat_ids, seats_df, lambda_pair=0.3):
    """Calculate objective value: Noise + λ * PairwiseDistance"""
    if not seat_ids:
        return None
    
    selected_seats = seats_df[seats_df['Seat_ID'].isin(seat_ids)]
    noise_sum = selected_seats['Noise'].sum()
    
    if len(seat_ids) <= 1:
        pair_distance_sum = 0
    else:
        coords_data = selected_seats[['Seat_ID', 'X', 'Y']]
        dmap = pairwise_distances(coords_data)
        pair_distance_sum = sum(dmap.values())
    
    return noise_sum + lambda_pair * pair_distance_sum


def run_incremental_algorithm(algorithm_name, seats_df, groups_df, brightness_reqs):
    """Run incremental algorithm (Greedy, Myopic ILP, SketchRefine) on all groups"""
    
    results = {
        'milestones': [],
        'objectives': [],
        'execution_times': [],
        'placement_rates': [],
        'groups_placed': []
    }
    
    available_seats = seats_df.copy()
    total_objective = 0
    total_time = 0
    groups_placed = 0
    
    # Test at regular intervals
    milestone_intervals = [10, 15, 20, 30, 40, 50, 60]
    
    print(f"\n>> Running {algorithm_name}")
    
    with tqdm(total=len(groups_df), desc=f"{algorithm_name}", unit="groups") as pbar:
        for i, (_, group) in enumerate(groups_df.iterrows()):
            group_id = group['Group_ID']
            group_size = group['Group_Size']
            brightness_min = brightness_reqs[group_id]
            
            start_time = time.time()
            
            # Run algorithm for this group
            try:
                if algorithm_name == 'Greedy':
                    result = greedy_pairwise(available_seats, group_size, brightness_min, lam_pair=0.3)
                    success = result.get('status') == 'ok'
                    seat_ids = result.get('seat_ids', [])
                    
                elif algorithm_name == 'Myopic ILP':
                    result = solve_group_pair_ilp(
                        available_seats, group_size, brightness_min, 
                        lam_pair=0.3, time_limit_sec=600
                    )
                    success = result.get('status') in ['Optimal', 'Feasible', 'Not Solved'] and result.get('seat_ids')
                    seat_ids = result.get('seat_ids', [])
                    
                elif algorithm_name == 'SketchRefine':
                    result = sketchrefine_solver(
                        available_seats, group_size, brightness_min, 
                        lam_pair=0.3, time_limit_sec=300
                    )
                    success = result.get('status') == 'ok' and result.get('seat_ids')
                    seat_ids = result.get('seat_ids', [])
                
                group_time = (time.time() - start_time) * 1000
                total_time += group_time
                
                if success and seat_ids and len(seat_ids) == group_size:
                    # Calculate objective
                    group_objective = calculate_objective(seat_ids, available_seats, 0.3)
                    total_objective += group_objective
                    groups_placed += 1
                    
                    # Remove assigned seats
                    available_seats = available_seats[~available_seats['Seat_ID'].isin(seat_ids)]
                
            except Exception as e:
                print(f"Error with group {group_id}: {e}")
            
            pbar.update(1)
            
            # Record milestone data
            current_groups = i + 1
            if current_groups in milestone_intervals:
                avg_objective = total_objective / groups_placed if groups_placed > 0 else None
                placement_rate = (groups_placed / current_groups) * 100
                
                results['milestones'].append(current_groups)
                results['objectives'].append(avg_objective)
                results['execution_times'].append(total_time)
                results['placement_rates'].append(placement_rate)
                results['groups_placed'].append(groups_placed)
    
    return results


def run_global_ilp(seats_df, groups_df, brightness_reqs):
    """Run Global ILP once for all groups"""
    
    print(f"\n>> Running Global ILP")
    
    # Convert brightness requirements
    groups_with_brightness = groups_df.copy()
    groups_with_brightness['Brightness_Min'] = groups_with_brightness['Group_ID'].map(brightness_reqs)
    
    # Estimate time needed (roughly 1-2 hours for 250 seats, 100 groups)
    estimated_time = 3600  # 1 hour
    print(f"   Estimated time: ~{estimated_time//60} minutes")
    
    start_time = time.time()
    
    try:
        result = solve_global_pair_ilp(
            seats_df, groups_with_brightness, 
            lam_pair=0.3, time_limit_sec=estimated_time
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        if result and result.get('status') in ['Optimal', 'Feasible']:
            assignments = result.get('assignments', {})
            total_objective = result.get('objective', None)
            
            # Count successful placements
            groups_placed = len([g for g, seats in assignments.items() if seats and len(seats) > 0])
            placement_rate = (groups_placed / len(groups_df)) * 100
            
            # Convert to average objective per group for comparison
            avg_objective = total_objective / len(groups_df) if total_objective else None
            
            print(f"   >> Success: {groups_placed}/{len(groups_df)} groups placed")
            print(f"   Average objective per group: {avg_objective:.4f}")
            print(f"   Execution time: {execution_time/1000:.1f} seconds")
            
            return {
                'success': True,
                'avg_objective': avg_objective,
                'execution_time': execution_time,
                'placement_rate': placement_rate,
                'groups_placed': groups_placed
            }
        else:
            print(f"   >> Failed: {result.get('status') if result else 'No result'}")
            return {'success': False, 'execution_time': execution_time}
            
    except Exception as e:
        print(f"   💥 Error: {str(e)}")
        return {'success': False, 'execution_time': estimated_time * 1000}


def create_comparison_plots(incremental_results, global_result):
    """Create 4-panel comparison plot"""
    
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Seating Optimization Algorithm Comparison\n250-Seat Layout, 100 Groups', 
                 fontsize=16, fontweight='bold')
    
    # Colors and styles for algorithms
    colors = {'Greedy': '#2E8B57', 'Myopic ILP': '#4169E1', 'SketchRefine': '#DC143C', 'Global ILP': '#FF8C00'}
    line_styles = {'Greedy': '-', 'Myopic ILP': '--', 'SketchRefine': '-.', 'Global ILP': ':'}
    markers = {'Greedy': 'o', 'Myopic ILP': 's', 'SketchRefine': '^', 'Global ILP': 'd'}
    
    algorithms = ['Greedy', 'Myopic ILP', 'SketchRefine']
    
    # Experiment 1: Objective Value Comparison
    ax1.set_title('Objective Value Comparison', fontweight='bold', fontsize=12)
    
    # Plot incremental algorithms
    for alg in algorithms:
        if alg in incremental_results and incremental_results[alg]['objectives']:
            milestones = incremental_results[alg]['milestones']
            objectives = [obj for obj in incremental_results[alg]['objectives'] if obj is not None]
            if objectives:
                ax1.plot(milestones[:len(objectives)], objectives, 
                        linestyle=line_styles[alg], marker=markers[alg], 
                        color=colors[alg], label=alg, linewidth=2.5, markersize=6)
    
    # Plot Global ILP as constant line (theoretical best)
    if global_result.get('success') and global_result.get('avg_objective'):
        global_obj = global_result['avg_objective']
        ax1.axhline(y=global_obj, color=colors['Global ILP'], 
                   linestyle=line_styles['Global ILP'], linewidth=2.5, 
                   label='Global ILP (Optimal)', alpha=0.9)
    
    ax1.set_xlabel('Number of Groups Placed')
    ax1.set_ylabel('Average Objective Value (lower is better)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Experiment 2: Execution Time (no Global ILP)
    ax2.set_title('Execution Time Comparison', fontweight='bold', fontsize=12)
    
    for alg in algorithms:  # Exclude Global ILP from time comparison
        if alg in incremental_results and incremental_results[alg]['execution_times']:
            milestones = incremental_results[alg]['milestones']
            times = [t/1000 for t in incremental_results[alg]['execution_times']]  # Convert to seconds
            ax2.plot(milestones, times, 
                    linestyle=line_styles[alg], marker=markers[alg], 
                    color=colors[alg], label=alg, linewidth=2.5, markersize=6)
    
    ax2.set_xlabel('Number of Groups Placed')
    ax2.set_ylabel('Cumulative Execution Time (seconds)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    # Experiment 3: Placement Success Rate
    ax3.set_title('Placement Success Rate', fontweight='bold', fontsize=12)
    
    for alg in algorithms:
        if alg in incremental_results and incremental_results[alg]['placement_rates']:
            milestones = incremental_results[alg]['milestones']
            rates = incremental_results[alg]['placement_rates']
            ax3.plot(milestones, rates, 
                    linestyle=line_styles[alg], marker=markers[alg], 
                    color=colors[alg], label=alg, linewidth=2.5, markersize=6)
    
    # Global ILP success rate as constant line
    if global_result.get('success') and global_result.get('placement_rate'):
        global_rate = global_result['placement_rate']
        ax3.axhline(y=global_rate, color=colors['Global ILP'], 
                   linestyle=line_styles['Global ILP'], linewidth=2.5, 
                   label='Global ILP', alpha=0.9)
    
    ax3.set_xlabel('Number of Groups Placed')
    ax3.set_ylabel('Placement Success Rate (%)')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Experiment 4: Optimality Gap vs Global ILP
    ax4.set_title('Optimality Gap vs Global ILP', fontweight='bold', fontsize=12)
    
    if global_result.get('success') and global_result.get('avg_objective'):
        global_obj = global_result['avg_objective']
        
        for alg in algorithms:
            if alg in incremental_results and incremental_results[alg]['objectives']:
                milestones = incremental_results[alg]['milestones']
                objectives = incremental_results[alg]['objectives']
                
                # Calculate optimality gaps
                gaps = []
                valid_milestones = []
                for i, obj in enumerate(objectives):
                    if obj is not None and global_obj > 0:
                        gap = ((obj - global_obj) / global_obj) * 100
                        gaps.append(gap)
                        valid_milestones.append(milestones[i])
                
                if gaps:
                    ax4.plot(valid_milestones, gaps, 
                            linestyle=line_styles[alg], marker=markers[alg], 
                            color=colors[alg], label=alg, linewidth=2.5, markersize=6)
        
        # Global ILP baseline at 0%
        ax4.axhline(y=0, color=colors['Global ILP'], 
                   linestyle=line_styles['Global ILP'], linewidth=2.5, 
                   label='Global ILP (0% gap)', alpha=0.9)
    
    ax4.set_xlabel('Number of Groups Placed')
    ax4.set_ylabel('Optimality Gap (%)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved to 'results.png'")
    
    return fig


def main():
    """Main experiment function"""
    print("SEATING OPTIMIZATION ALGORITHM COMPARISON")
    print("300-Seat Layout, 60 Groups")
    print("=" * 60)
    
    # Generate dataset (smaller for Global ILP feasibility)
    print("\nGenerating 300-seat dataset with 60 groups...")
    
    # Generate seats (12 tables × 5×5 = 300 seats)
    seats_df = generate_seats(
        rooms=1, 
        tables_per_room=12, 
        rows_per_table=5, 
        cols_per_table=5, 
        seed=42
    )
    
    # Generate groups (smaller for feasibility)
    groups_df = generate_groups(
        n_groups=60,
        min_size=2,
        max_size=4,  # Reduced max size for feasibility
        brightness_mean=40,  # Reduced brightness requirements
        brightness_std=10,   # Reduced variance
        seed=42
    )
    
    # Create brightness requirements dict
    brightness_reqs = dict(zip(groups_df['Group_ID'], groups_df['Brightness_Min']))
    
    # Calculate total seats needed
    total_people = groups_df['Group_Size'].sum()
    print(f"   Generated {len(seats_df)} seats, {len(groups_df)} groups")
    print(f"   Total people: {total_people} (seats/people ratio: {len(seats_df)/total_people:.2f})")
    
    # Run experiments
    incremental_results = {}
    
    # Run incremental algorithms
    algorithms = ['Greedy', 'Myopic ILP', 'SketchRefine']
    for alg in algorithms:
        incremental_results[alg] = run_incremental_algorithm(
            alg, seats_df.copy(), groups_df, brightness_reqs
        )
    
    # Run Global ILP
    global_result = run_global_ilp(seats_df.copy(), groups_df, brightness_reqs)
    
    # Create visualization
    print(f"\nCreating comparison plots...")
    fig = create_comparison_plots(incremental_results, global_result)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print(f"\nAlgorithm Performance (at 100 groups):")
    for alg in algorithms:
        if alg in incremental_results:
            results = incremental_results[alg]
            if results['objectives'] and len(results['objectives']) > 0:
                final_obj = results['objectives'][-1]
                final_rate = results['placement_rates'][-1]
                final_time = results['execution_times'][-1] / 1000
                
                # Handle None values
                if final_obj is not None and final_rate is not None and final_time is not None:
                    print(f"  {alg:<12}: Obj={final_obj:.3f}, Success={final_rate:.1f}%, Time={final_time:.1f}s")
                else:
                    print(f"  {alg:<12}: Failed or incomplete results")
            else:
                print(f"  {alg:<12}: No results available")
    
    if global_result.get('success'):
        global_obj = global_result['avg_objective']
        global_rate = global_result['placement_rate']
        global_time = global_result['execution_time'] / 1000
        print(f"  {'Global ILP':<12}: Obj={global_obj:.3f}, Success={global_rate:.1f}%, Time={global_time:.1f}s")
    else:
        print(f"  {'Global ILP':<12}: Failed (infeasible or timeout)")
    
    print(f"\nExpected Algorithm Ordering (objective values):")
    print("Global ILP <= Myopic ILP <= SketchRefine <= Greedy")
    
    print(f"\nExperiment completed successfully!")


if __name__ == "__main__":
    main()

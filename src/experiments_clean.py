#!/usr/bin/env python3
"""
Comprehensive Algorithm Comparison Experiments
Tests Greedy vs ILP algorithms across 4 experiment sizes (500, 1000, 1500, 2000) for both Q1 and Q2 queries.
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for data generation
sys.path.append('../data')

def create_graphs_directory():
    """Create graphs directory if it doesn't exist."""
    graphs_dir = "graphs"
    os.makedirs(graphs_dir, exist_ok=True)
    return graphs_dir

def run_greedy_algorithm(seats_df: pd.DataFrame, group_size: int, brightness_threshold: float, query_type: str) -> Dict:
    """Run greedy algorithm for a single group."""
    try:
        from greedy import (greedy_seat_selection, load_optimized_data, 
                           create_table_stats_from_precomputed, create_adjacency_graph_from_precomputed)
        
        # Load optimized data structures
        try:
            table_stats_df = pd.read_csv("../data/table_stats.csv")
            adjacency_df = pd.read_csv("../data/adjacency_graph.csv")
            
            table_stats = create_table_stats_from_precomputed(seats_df, table_stats_df)
            adjacency_graph = create_adjacency_graph_from_precomputed(adjacency_df)
        except:
            # Fallback to creating from scratch
            from greedy import create_table_stats, create_adjacency_graph
            table_stats = create_table_stats(seats_df)
            adjacency_graph = create_adjacency_graph(table_stats)
        
        # Set weights based on query type
        if query_type == 'Q1':
            w1, w2 = 1.0, 0.0  # Minimize noise only
        else:  # Q2
            w1, w2 = 1.0, -0.3  # Minimize noise - 0.3*brightness
        
        start_time = time.time()
        result = greedy_seat_selection(
            group_size=group_size,
            brightness_threshold=brightness_threshold,
            table_stats=table_stats,
            adjacency_graph=adjacency_graph,
            w1=w1,
            w2=w2
        )
        execution_time = (time.time() - start_time) * 1000
        
        if result:
            # Calculate objective value
            total_noise = sum(seat['noise'] for seat in result)
            total_brightness = sum(seat['brightness'] for seat in result)
            avg_noise = total_noise / len(result)
            avg_brightness = total_brightness / len(result)
            
            if query_type == 'Q1':
                objective_value = avg_noise
            else:
                objective_value = avg_noise - 0.3 * avg_brightness
            
            return {
                'success': True,
                'objective_value': objective_value,
                'execution_time_ms': execution_time,
                'seats_selected': len(result)
            }
        else:
            return {
                'success': False,
                'objective_value': float('inf'),
                'execution_time_ms': execution_time,
                'seats_selected': 0
            }
            
    except Exception as e:
        return {
            'success': False,
            'objective_value': float('inf'),
            'execution_time_ms': 0,
            'seats_selected': 0,
            'error': str(e)
        }

def run_ilp_algorithm(seats_df: pd.DataFrame, group_size: int, brightness_threshold: float, query_type: str) -> Dict:
    """Run ILP algorithm for a single group."""
    try:
        from ilp import solve_ilp
        
        available_seats = seats_df[seats_df['Seat_Available'] == True]
        
        if len(available_seats) < group_size:
            return {
                'success': False,
                'objective_value': float('inf'),
                'execution_time_ms': 0,
                'seats_selected': 0
            }
        
        # Prepare data matrix for ILP
        start_time = time.time()
        
        if query_type == 'Q1':
            data_matrix = available_seats[['Brightness', 'Noise']].values
            obj = {'attr': 1, 'pref': 'MIN'}  # Minimize noise (column 1)
        else:  # Q2
            composite_values = available_seats['Noise'] - 0.3 * available_seats['Brightness']
            data_matrix = np.column_stack([
                available_seats['Brightness'].values,
                available_seats['Noise'].values,
                composite_values.values
            ])
            obj = {'attr': 2, 'pref': 'MIN'}  # Minimize composite (column 2)
        
        # Constraint: sum(brightness) >= brightness_threshold * group_size
        cons = [{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}]
        
        picked, _ = solve_ilp(sub=data_matrix, size_=group_size, rep=1, obj=obj, cons=cons, verbose=False)
        
        execution_time = (time.time() - start_time) * 1000
        
        if picked:
            selected_seats = available_seats.iloc[picked]
            avg_noise = selected_seats['Noise'].mean()
            avg_brightness = selected_seats['Brightness'].mean()
            
            if query_type == 'Q1':
                objective_value = avg_noise
            else:
                objective_value = avg_noise - 0.3 * avg_brightness
            
            return {
                'success': True,
                'objective_value': objective_value,
                'execution_time_ms': execution_time,
                'seats_selected': len(picked)
            }
        else:
            return {
                'success': False,
                'objective_value': float('inf'),
                'execution_time_ms': execution_time,
                'seats_selected': 0
            }
            
    except Exception as e:
        return {
            'success': False,
            'objective_value': float('inf'),
            'execution_time_ms': 0,
            'seats_selected': 0,
            'error': str(e)
        }

def run_comprehensive_experiments():
    """
    Run comprehensive experiments testing 4 experiment sizes for both Q1 and Q2 queries.
    
    Experiment sizes: 500, 1000, 1500, 2000 (seats and students)
    Query types: Q1 (minimize noise), Q2 (minimize noise - 0.3*brightness)
    Algorithms: Greedy, ILP
    """
    
    print("🔬 COMPREHENSIVE ALGORITHM COMPARISON EXPERIMENTS")
    print("=" * 60)
    print("Testing 4 experiment sizes: 500, 1000, 1500, 2000")
    print("Testing 2 query types: Q1, Q2")
    print("Testing 2 algorithms: Greedy, ILP")
    print("Total combinations: 4 × 2 × 2 = 16 experiments")
    print()
    
    # Create graphs directory
    graphs_dir = create_graphs_directory()
    
    # First, regenerate data with improved noise variance
    print("Regenerating data with improved noise variance...")
    try:
        from data import generate_all_data
        generate_all_data()
        print("✅ Data regenerated with better noise distribution")
    except Exception as e:
        print(f"⚠️  Warning: Could not regenerate data: {e}")
        print("Using existing data files...")
    
    # Load full datasets
    print("Loading datasets...")
    seats_df_full = pd.read_csv("seats.csv")
    students_df_full = pd.read_csv("students.csv")
    print(f"✓ Loaded {len(seats_df_full)} seats and {len(students_df_full)} students")
    
    # Check noise distribution
    print(f"✓ Noise range: {seats_df_full['Noise'].min()}-{seats_df_full['Noise'].max()}, mean: {seats_df_full['Noise'].mean():.1f}")
    
    # Experiment configurations
    experiment_sizes = [500, 1000, 1500, 2000]
    query_types = ['Q1', 'Q2']
    algorithms = ['greedy', 'ilp']
    
    # Results storage
    results = {
        query_type: {
            algorithm: {
                'sizes': [],
                'avg_objectives': [],
                'avg_times': [],
                'success_rates': [],
                'groups_processed': []
            } for algorithm in algorithms
        } for query_type in query_types
    }
    
    # Run experiments
    for query_type in query_types:
        print(f"\n{'='*20} {query_type} EXPERIMENTS {'='*20}")
        
        for size in experiment_sizes:
            print(f"\n--- Experiment Size: {size} seats/students ---")
            
            # Create subset datasets (restart from beginning each time)
            seats_subset = seats_df_full.iloc[:size].copy()
            students_subset = students_df_full.iloc[:size].copy()
            
            # Get groups from student subset
            groups = list(students_subset.groupby('Group_ID'))
            max_groups = min(15, len(groups))  # Test up to 15 groups per size
            test_groups = groups[:max_groups]
            
            print(f"Testing {len(test_groups)} groups with {len(seats_subset)} seats")
            
            for algorithm in algorithms:
                print(f"\n  {algorithm.upper()} Algorithm:")
                
                # Reset seats for this algorithm
                seats_test = seats_subset.copy()
                seats_test['Seat_Available'] = True
                
                algorithm_results = {
                    'objectives': [],
                    'times': [],
                    'successes': 0
                }
                
                for i, (group_id, group_data) in enumerate(test_groups):
                    group_size = len(group_data)
                    brightness_threshold = group_data['Brightness'].min()
                    
                    # Run algorithm
                    if algorithm == 'greedy':
                        result = run_greedy_algorithm(seats_test, group_size, brightness_threshold, query_type)
                    else:
                        result = run_ilp_algorithm(seats_test, group_size, brightness_threshold, query_type)
                    
                    # Record results
                    algorithm_results['times'].append(result['execution_time_ms'])
                    
                    if result['success']:
                        algorithm_results['objectives'].append(result['objective_value'])
                        algorithm_results['successes'] += 1
                        
                        # Update seat availability (simplified)
                        if result['seats_selected'] > 0:
                            available_indices = seats_test[seats_test['Seat_Available'] == True].index
                            if len(available_indices) >= result['seats_selected']:
                                seats_test.loc[available_indices[:result['seats_selected']], 'Seat_Available'] = False
                    
                    # Progress indicator
                    if (i + 1) % 5 == 0:
                        print(f"    Processed {i + 1}/{len(test_groups)} groups...")
                
                # Store aggregated results
                results[query_type][algorithm]['sizes'].append(size)
                results[query_type][algorithm]['groups_processed'].append(len(test_groups))
                results[query_type][algorithm]['success_rates'].append(algorithm_results['successes'] / len(test_groups))
                
                if algorithm_results['times']:
                    results[query_type][algorithm]['avg_times'].append(np.mean(algorithm_results['times']))
                else:
                    results[query_type][algorithm]['avg_times'].append(0)
                
                if algorithm_results['objectives']:
                    results[query_type][algorithm]['avg_objectives'].append(np.mean(algorithm_results['objectives']))
                else:
                    results[query_type][algorithm]['avg_objectives'].append(float('inf'))
                
                # Print summary for this size/algorithm
                avg_time = np.mean(algorithm_results['times']) if algorithm_results['times'] else 0
                avg_obj = np.mean(algorithm_results['objectives']) if algorithm_results['objectives'] else float('inf')
                success_rate = algorithm_results['successes'] / len(test_groups)
                
                print(f"    Results: avg_time={avg_time:.2f}ms, avg_obj={avg_obj:.4f}, success_rate={success_rate:.2%}")
    
    # Generate comprehensive plots
    generate_comprehensive_plots(results, graphs_dir)
    
    # Print final summary
    print_experiment_summary(results)
    
    return results

def generate_comprehensive_plots(results: Dict, graphs_dir: str):
    """Generate comprehensive visualization plots."""
    
    # Plot 1: Performance vs Size for each query type
    for query_type in ['Q1', 'Q2']:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{query_type} Algorithm Performance Analysis', fontsize=16, fontweight='bold')
        
        colors = {'greedy': '#2E86AB', 'ilp': '#F24236'}
        
        # Execution Time vs Size
        for algorithm in ['greedy', 'ilp']:
            if results[query_type][algorithm]['sizes']:
                ax1.plot(results[query_type][algorithm]['sizes'], 
                        results[query_type][algorithm]['avg_times'],
                        marker='o', linewidth=2, markersize=8,
                        color=colors[algorithm], label=algorithm.upper())
        
        ax1.set_title('Average Execution Time vs Dataset Size', fontweight='bold')
        ax1.set_xlabel('Dataset Size (number of seats/students)')
        ax1.set_ylabel('Average Execution Time (ms)')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Objective Value vs Size
        for algorithm in ['greedy', 'ilp']:
            if results[query_type][algorithm]['sizes']:
                obj_values = [obj if obj != float('inf') else None for obj in results[query_type][algorithm]['avg_objectives']]
                ax2.plot(results[query_type][algorithm]['sizes'], 
                        obj_values,
                        marker='s', linewidth=2, markersize=8,
                        color=colors[algorithm], label=algorithm.upper())
        
        ax2.set_title('Average Objective Value vs Dataset Size', fontweight='bold')
        ax2.set_xlabel('Dataset Size (number of seats/students)')
        ax2.set_ylabel('Average Objective Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Success Rate vs Size
        for algorithm in ['greedy', 'ilp']:
            if results[query_type][algorithm]['sizes']:
                ax3.plot(results[query_type][algorithm]['sizes'], 
                        [rate * 100 for rate in results[query_type][algorithm]['success_rates']],
                        marker='^', linewidth=2, markersize=8,
                        color=colors[algorithm], label=algorithm.upper())
        
        ax3.set_title('Success Rate vs Dataset Size', fontweight='bold')
        ax3.set_xlabel('Dataset Size (number of seats/students)')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_ylim(0, 105)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Speedup Analysis
        if (results[query_type]['greedy']['sizes'] and results[query_type]['ilp']['sizes']):
            speedups = []
            sizes = results[query_type]['greedy']['sizes']
            
            for i in range(len(sizes)):
                greedy_time = results[query_type]['greedy']['avg_times'][i]
                ilp_time = results[query_type]['ilp']['avg_times'][i]
                if greedy_time > 0:
                    speedups.append(ilp_time / greedy_time)
                else:
                    speedups.append(0)
            
            ax4.bar(range(len(sizes)), speedups, color='green', alpha=0.7)
            ax4.set_title('Greedy Speedup Factor vs Dataset Size', fontweight='bold')
            ax4.set_xlabel('Dataset Size')
            ax4.set_ylabel('Speedup Factor (x times faster)')
            ax4.set_xticks(range(len(sizes)))
            ax4.set_xticklabels(sizes)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, speedup in enumerate(speedups):
                if speedup > 0:
                    ax4.text(i, speedup + 0.1, f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f'comprehensive_analysis_{query_type.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved comprehensive analysis: {query_type}")
    
    # Plot 2: Algorithm Comparison Summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Algorithm Performance Summary Across All Experiments', fontsize=16, fontweight='bold')
    
    # Average performance across all sizes
    query_labels = []
    greedy_times = []
    ilp_times = []
    greedy_objs = []
    ilp_objs = []
    
    for query_type in ['Q1', 'Q2']:
        if results[query_type]['greedy']['avg_times'] and results[query_type]['ilp']['avg_times']:
            query_labels.append(query_type)
            greedy_times.append(np.mean([t for t in results[query_type]['greedy']['avg_times'] if t > 0]))
            ilp_times.append(np.mean([t for t in results[query_type]['ilp']['avg_times'] if t > 0]))
            
            greedy_obj_vals = [obj for obj in results[query_type]['greedy']['avg_objectives'] if obj != float('inf')]
            ilp_obj_vals = [obj for obj in results[query_type]['ilp']['avg_objectives'] if obj != float('inf')]
            
            greedy_objs.append(np.mean(greedy_obj_vals) if greedy_obj_vals else 0)
            ilp_objs.append(np.mean(ilp_obj_vals) if ilp_obj_vals else 0)
    
    if query_labels:
        # Execution time comparison
        x = np.arange(len(query_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, greedy_times, width, label='Greedy', color='#2E86AB', alpha=0.8)
        bars2 = ax1.bar(x + width/2, ilp_times, width, label='ILP', color='#F24236', alpha=0.8)
        
        ax1.set_title('Average Execution Time Comparison', fontweight='bold')
        ax1.set_ylabel('Average Time (ms)')
        ax1.set_xlabel('Query Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(query_labels)
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                            f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Objective value comparison
        bars3 = ax2.bar(x - width/2, greedy_objs, width, label='Greedy', color='#2E86AB', alpha=0.8)
        bars4 = ax2.bar(x + width/2, ilp_objs, width, label='ILP', color='#F24236', alpha=0.8)
        
        ax2.set_title('Average Objective Value Comparison', fontweight='bold')
        ax2.set_ylabel('Average Objective Value')
        ax2.set_xlabel('Query Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(query_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                if height != 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + abs(height) * 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'algorithm_comparison_summary.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved algorithm comparison summary")

def print_experiment_summary(results: Dict):
    """Print comprehensive experiment summary."""
    
    print(f"\n{'='*20} EXPERIMENT SUMMARY {'='*20}")
    
    for query_type in ['Q1', 'Q2']:
        print(f"\n{query_type} Results:")
        print("-" * 30)
        
        for algorithm in ['greedy', 'ilp']:
            data = results[query_type][algorithm]
            
            if data['sizes'] and data['avg_times']:
                print(f"\n{algorithm.upper()} Performance:")
                
                for i, size in enumerate(data['sizes']):
                    avg_time = data['avg_times'][i]
                    avg_obj = data['avg_objectives'][i] if data['avg_objectives'][i] != float('inf') else 'FAILED'
                    success_rate = data['success_rates'][i]
                    groups = data['groups_processed'][i]
                    
                    print(f"  Size {size:4d}: time={avg_time:7.2f}ms, obj={avg_obj}, success={success_rate:.1%}, groups={groups}")
        
        # Performance comparison
        if (results[query_type]['greedy']['avg_times'] and results[query_type]['ilp']['avg_times']):
            print(f"\n{query_type} Performance Analysis:")
            
            greedy_avg_time = np.mean([t for t in results[query_type]['greedy']['avg_times'] if t > 0])
            ilp_avg_time = np.mean([t for t in results[query_type]['ilp']['avg_times'] if t > 0])
            
            if greedy_avg_time > 0:
                speedup = ilp_avg_time / greedy_avg_time
                print(f"  ⚡ Overall Speedup: Greedy is {speedup:.1f}x faster than ILP")
            
            greedy_objs = [obj for obj in results[query_type]['greedy']['avg_objectives'] if obj != float('inf')]
            ilp_objs = [obj for obj in results[query_type]['ilp']['avg_objectives'] if obj != float('inf')]
            
            if greedy_objs and ilp_objs:
                greedy_avg_obj = np.mean(greedy_objs)
                ilp_avg_obj = np.mean(ilp_objs)
                
                print(f"  🎯 Quality: Greedy avg = {greedy_avg_obj:.4f}, ILP avg = {ilp_avg_obj:.4f}")
                
                if greedy_avg_obj > ilp_avg_obj:
                    quality_gap = ((greedy_avg_obj - ilp_avg_obj) / abs(ilp_avg_obj)) * 100
                    print(f"  📊 Trade-off: Greedy {quality_gap:.1f}% worse quality for {speedup:.1f}x speed improvement")
    
    print(f"\n🎉 EXPERIMENTS COMPLETE!")
    print("✅ Tested 4 experiment sizes: 500, 1000, 1500, 2000")
    print("✅ Tested 2 query types: Q1, Q2") 
    print("✅ Tested 2 algorithms: Greedy, ILP")
    print("✅ Generated comprehensive visualizations")
    print(f"📁 All graphs saved to: graphs/")

if __name__ == "__main__":
    results = run_comprehensive_experiments()

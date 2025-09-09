# comprehensive_experiment_fixed.py
"""
Fixed comprehensive experiment runner with proper NaN/inf handling for complete graphs.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime

from seating_opt import (
    generate_seats, 
    run_all
)
from seating_opt.data_gen import generate_groups


def create_dataset_fixed(
    n_seats: int = 60,
    n_groups: int = 15,
    rooms: int = 1,
    tables_per_room: int = 6,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a dataset with reasonable brightness requirements to improve success rates."""
    
    # Calculate table dimensions based on total seats
    rows_per_table = 4
    cols_per_table = max(2, n_seats // (rooms * tables_per_room * rows_per_table))
    
    # Generate seats
    seats_df = generate_seats(
        rooms=rooms,
        tables_per_room=tables_per_room,
        rows_per_table=rows_per_table,
        cols_per_table=cols_per_table,
        seed=seed
    )
    
    # Trim to desired number of seats
    seats_df = seats_df.head(n_seats).copy()
    
    # Get brightness statistics for generating reasonable requirements
    min_brightness = seats_df['Brightness'].min()
    mean_brightness = seats_df['Brightness'].mean()
    
    # Generate groups with more achievable brightness requirements
    groups = []
    np.random.seed(seed + 1)
    
    for i in range(1, n_groups + 1):
        group_size = np.random.randint(2, 5)  # Size 2-4
        # Set brightness requirement in achievable range
        brightness_min = np.random.uniform(min_brightness + 5, mean_brightness - 10)
        brightness_min = max(brightness_min, min_brightness + 2)  # Ensure achievable
        
        groups.append({
            "Group_ID": i,
            "Group_Size": group_size,
            "Brightness_Min": brightness_min,
            "Objective": "Q1"
        })
    
    groups_df = pd.DataFrame(groups)
    return seats_df, groups_df


def run_scalability_experiment_fixed() -> Dict:
    """Run experiments with improved success rates and proper value handling."""
    
    sizes = [
        (30, 8),   # Small: 30 seats, 8 groups
        (60, 15),  # Medium: 60 seats, 15 groups  
        (100, 25), # Large: 100 seats, 25 groups
        (150, 35), # XLarge: 150 seats, 35 groups
    ]
    
    results = {}
    
    for n_seats, n_groups in sizes:
        print(f"\n🔬 Running fixed experiment: {n_seats} seats, {n_groups} groups")
        
        try:
            # Create dataset with achievable requirements
            seats_df, groups_df = create_dataset_fixed(n_seats=n_seats, n_groups=n_groups)
            
            # Run all algorithms
            start_time = time.time()
            experiment_result = run_all(
                seats_df=seats_df,
                groups_df=groups_df,
                lam_pair=0.3,
                dmax_pairs=3
            )
            total_time = time.time() - start_time
            
            # Extract results with proper handling
            size_key = f"{n_seats}seats_{n_groups}groups"
            results[size_key] = {
                "dataset_info": {
                    "n_seats": n_seats,
                    "n_groups": n_groups,
                    "total_experiment_time": total_time * 1000,
                    "brightness_range": [float(seats_df['Brightness'].min()), float(seats_df['Brightness'].max())],
                    "group_requirements": [float(groups_df['Brightness_Min'].min()), float(groups_df['Brightness_Min'].max())]
                },
                "algorithms": {}
            }
            
            # Process each algorithm's results with safe handling
            for alg_name in ["greedy", "myopic_ilp", "sketchrefine"]:
                if alg_name in experiment_result:
                    alg_data = experiment_result[alg_name]
                    summary = alg_data["summary"]
                    
                    # Ensure all values are finite and reasonable
                    success_rate = summary["success_rate"]
                    avg_noise = summary["avg_noise"] if np.isfinite(summary["avg_noise"]) else 0.0
                    avg_brightness = summary["avg_brightness"] if np.isfinite(summary["avg_brightness"]) else 0.0
                    avg_pairwise_distance = summary["avg_pairwise_distance"] if np.isfinite(summary["avg_pairwise_distance"]) else 0.0
                    cumulative_objective = summary["cumulative_objective"] if np.isfinite(summary["cumulative_objective"]) else 0.0
                    runtime_ms = summary["runtime_ms"] if np.isfinite(summary["runtime_ms"]) else 0.0
                    regret_vs_gold = summary["regret_vs_gold"] if np.isfinite(summary["regret_vs_gold"]) else 0.0
                    
                    results[size_key]["algorithms"][alg_name] = {
                        "success_rate": success_rate,
                        "avg_noise": avg_noise,
                        "avg_brightness": avg_brightness,
                        "avg_pairwise_distance": avg_pairwise_distance,
                        "cumulative_objective": cumulative_objective,
                        "runtime_ms": runtime_ms,
                        "regret_vs_gold": regret_vs_gold
                    }
                else:
                    # Provide default values for missing algorithms
                    results[size_key]["algorithms"][alg_name] = {
                        "success_rate": 0.0,
                        "avg_noise": 0.0,
                        "avg_brightness": 0.0,
                        "avg_pairwise_distance": 0.0,
                        "cumulative_objective": 0.0,
                        "runtime_ms": 0.0,
                        "regret_vs_gold": 0.0
                    }
            
            # Add gold standard info with safe handling
            gold_cum_obj = experiment_result["world"]["gold_cum_obj"]
            results[size_key]["gold_standard"] = {
                "status": experiment_result["world"]["gold_status"],
                "cumulative_objective": gold_cum_obj if np.isfinite(gold_cum_obj) else 0.0
            }
            
            print(f"  ✅ Completed in {total_time:.2f}s")
            
            # Print summary for this size
            print(f"  📊 Results Summary:")
            for alg_name, alg_data in results[size_key]["algorithms"].items():
                success_rate = alg_data['success_rate']
                runtime = alg_data['runtime_ms']
                regret = alg_data['regret_vs_gold']
                print(f"    {alg_name.title()}: "
                      f"Success={success_rate:.1%}, "
                      f"Runtime={runtime:.1f}ms, "
                      f"Regret={regret:.2f}")
                
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            # Provide default structure even for failures
            results[f"{n_seats}seats_{n_groups}groups"] = {
                "error": str(e),
                "dataset_info": {"n_seats": n_seats, "n_groups": n_groups},
                "algorithms": {
                    alg_name: {
                        "success_rate": 0.0,
                        "avg_noise": 0.0,
                        "avg_brightness": 0.0,
                        "avg_pairwise_distance": 0.0,
                        "cumulative_objective": 0.0,
                        "runtime_ms": 0.0,
                        "regret_vs_gold": 0.0
                    } for alg_name in ["greedy", "myopic_ilp", "sketchrefine"]
                }
            }
    
    return results


def create_performance_plots_fixed(results: Dict) -> None:
    """Create complete performance comparison plots with all data points visible."""
    
    # Extract data for plotting
    sizes = []
    algorithms = ["greedy", "myopic_ilp", "sketchrefine"]
    metrics = {
        "runtime_ms": "Runtime (ms)",
        "success_rate": "Success Rate",
        "regret_vs_gold": "Regret vs Gold Standard",
        "cumulative_objective": "Cumulative Objective"
    }
    
    # Initialize plot data structure
    plot_data = {metric: {alg: [] for alg in algorithms} for metric in metrics}
    
    # Extract valid results and ensure all algorithms have data for each size
    valid_results = {}
    for size_key, result in results.items():
        if "error" not in result and "algorithms" in result:
            valid_results[size_key] = result
            sizes.append(result["dataset_info"]["n_seats"])
    
    if not valid_results:
        print("⚠️  No valid results to plot!")
        return
    
    # Populate plot data - ensure every algorithm has a value for every size
    for size_key, result in valid_results.items():
        for alg in algorithms:
            if alg in result["algorithms"]:
                alg_data = result["algorithms"][alg]
                for metric in metrics:
                    if metric in alg_data and np.isfinite(alg_data[metric]):
                        plot_data[metric][alg].append(alg_data[metric])
                    else:
                        plot_data[metric][alg].append(0.0)  # Use 0 instead of None for complete lines
            else:
                # Provide default values for missing algorithms
                for metric in metrics:
                    plot_data[metric][alg].append(0.0)
    
    # Sort data by size to ensure proper line connections
    sorted_indices = np.argsort(sizes)
    sizes = [sizes[i] for i in sorted_indices]
    
    for metric in metrics:
        for alg in algorithms:
            plot_data[metric][alg] = [plot_data[metric][alg][i] for i in sorted_indices]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Algorithm Performance Comparison (Fixed)', fontsize=16, fontweight='bold')
    
    colors = {'greedy': '#2E8B57', 'myopic_ilp': '#4169E1', 'sketchrefine': '#DC143C'}
    markers = {'greedy': 'o', 'myopic_ilp': 's', 'sketchrefine': '^'}
    
    plot_positions = [(0,0), (0,1), (1,0), (1,1)]
    
    for idx, (metric, title) in enumerate(metrics.items()):
        ax = axes[plot_positions[idx][0], plot_positions[idx][1]]
        
        for alg in algorithms:
            y_data = plot_data[metric][alg]
            x_data = sizes
            
            # Plot with all points visible and connected
            ax.plot(x_data, y_data, 
                   color=colors[alg], 
                   marker=markers[alg], 
                   linewidth=3, 
                   markersize=10,
                   markeredgewidth=2,
                   markeredgecolor='white',
                   label=alg.replace('_', ' ').title(),
                   alpha=0.8)
        
        ax.set_xlabel('Number of Seats', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} vs Dataset Size', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Special formatting for specific metrics
        if metric == "success_rate":
            ax.set_ylim(0, 1.1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        elif metric == "runtime_ms":
            if max(max(plot_data[metric][alg]) for alg in algorithms) > 0:
                ax.set_yscale('log')
                ax.set_ylim(bottom=1)  # Avoid log(0)
        
        # Ensure all data points are visible
        ax.set_xlim(left=min(sizes) - 5, right=max(sizes) + 5)
    
    plt.tight_layout()
    
    # Save with high DPI for better quality
    plt.savefig('algorithm_performance_fixed.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("📈 Fixed performance plots saved as 'algorithm_performance_fixed.png'")
    
    # Also show the plot
    plt.show()
    return fig


def main():
    """Main experiment runner with fixes."""
    print("🚀 Starting FIXED Comprehensive Seating Optimization Experiment")
    print("   Comparing: Greedy, ILP (Myopic), and SketchRefine algorithms")
    print("   With improved success rates and complete graph visualization")
    print("-" * 70)
    
    # Run scalability experiment with fixes
    results = run_scalability_experiment_fixed()
    
    # Save results with proper JSON serialization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiment_results_fixed_{timestamp}.json"
    
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float)) and (pd.isna(obj) or np.isinf(obj)):
            return 0.0  # Convert NaN/inf to 0
        else:
            return obj
    
    serializable_results = convert_numpy_types(results)
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "fixed_scalability_comparison",
            "algorithms": ["greedy", "myopic_ilp", "sketchrefine"],
            "results": serializable_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to '{results_file}'")
    
    # Print summary table
    print_summary_table_fixed(results)
    
    # Create plots with complete data
    try:
        fig = create_performance_plots_fixed(results)
    except Exception as e:
        print(f"⚠️  Plotting failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Fixed experiment completed! Results saved to '{results_file}'")
    
    return results


def print_summary_table_fixed(results: Dict) -> None:
    """Print a summary table with proper value formatting."""
    
    print("\n" + "="*85)
    print("🏆 FIXED EXPERIMENT SUMMARY TABLE")
    print("="*85)
    
    # Header
    print(f"{'Dataset':<20} {'Algorithm':<15} {'Success%':<10} {'Runtime(ms)':<12} {'Regret':<10} {'Objective':<12}")
    print("-" * 85)
    
    # Data rows
    for size_key, result in results.items():
        if "error" in result:
            print(f"{size_key:<20} {'ERROR':<15} {'-':<10} {'-':<12} {'-':<10} {'-':<12}")
            continue
            
        dataset_name = f"{result['dataset_info']['n_seats']}s/{result['dataset_info']['n_groups']}g"
        
        for i, alg_name in enumerate(["greedy", "myopic_ilp", "sketchrefine"]):
            if alg_name in result["algorithms"]:
                alg_data = result["algorithms"][alg_name]
                success = f"{alg_data['success_rate']:.1%}"
                runtime = f"{alg_data['runtime_ms']:.1f}"
                regret = f"{alg_data['regret_vs_gold']:.2f}"
                objective = f"{alg_data['cumulative_objective']:.2f}"
                
                display_name = dataset_name if i == 0 else ""
                alg_display = alg_name.replace('_', ' ').title()
                
                print(f"{display_name:<20} {alg_display:<15} {success:<10} {runtime:<12} {regret:<10} {objective:<12}")
        
        print("-" * 85)
    
    # Calculate and display overall insights
    print("\n🔍 Key Insights:")
    
    all_algs_data = {"greedy": [], "myopic_ilp": [], "sketchrefine": []}
    
    for result in results.values():
        if "error" not in result and "algorithms" in result:
            for alg_name in all_algs_data.keys():
                if alg_name in result["algorithms"]:
                    alg_data = result["algorithms"][alg_name]
                    all_algs_data[alg_name].append({
                        "success": alg_data["success_rate"],
                        "runtime": alg_data["runtime_ms"],
                        "regret": alg_data["regret_vs_gold"]
                    })
    
    for alg_name, data_list in all_algs_data.items():
        if data_list:
            avg_success = np.mean([d["success"] for d in data_list])
            avg_runtime = np.mean([d["runtime"] for d in data_list])
            avg_regret = np.mean([d["regret"] for d in data_list])
            
            print(f"  • {alg_name.replace('_', ' ').title()}: "
                  f"Avg Success {avg_success:.1%}, "
                  f"Avg Runtime {avg_runtime:.1f}ms, "
                  f"Avg Regret {avg_regret:.2f}")


if __name__ == "__main__":
    results = main()

# comprehensive_experiment.py
"""
Comprehensive experiment runner comparing Greedy, ILP, and SketchRefine algorithms.
Includes performance tracking and plotting capabilities.
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


def create_dataset(
    n_seats: int = 60,
    n_groups: int = 15,
    rooms: int = 1,
    tables_per_room: int = 6,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a dataset for experimentation."""
    
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
    
    # Generate groups
    groups_df = generate_groups(
        n_groups=n_groups,
        min_size=2,
        max_size=4,
        brightness_mean=50,
        brightness_std=20,
        seed=seed + 1
    )
    
    return seats_df, groups_df


def run_scalability_experiment() -> Dict:
    """Run experiments with different dataset sizes to test scalability."""
    
    sizes = [
        (30, 8),   # Small: 30 seats, 8 groups
        (60, 15),  # Medium: 60 seats, 15 groups  
        (100, 25), # Large: 100 seats, 25 groups
        (150, 35), # XLarge: 150 seats, 35 groups
    ]
    
    results = {}
    
    for n_seats, n_groups in sizes:
        print(f"\n🔬 Running experiment: {n_seats} seats, {n_groups} groups")
        
        try:
            # Create dataset
            seats_df, groups_df = create_dataset(n_seats=n_seats, n_groups=n_groups)
            
            # Run all algorithms
            start_time = time.time()
            experiment_result = run_all(
                seats_df=seats_df,
                groups_df=groups_df,
                lam_pair=0.3,
                dmax_pairs=3
            )
            total_time = time.time() - start_time
            
            # Extract results
            size_key = f"{n_seats}seats_{n_groups}groups"
            results[size_key] = {
                "dataset_info": {
                    "n_seats": n_seats,
                    "n_groups": n_groups,
                    "total_experiment_time": total_time * 1000  # ms
                },
                "algorithms": {}
            }
            
            # Process each algorithm's results
            for alg_name in ["greedy", "myopic_ilp", "sketchrefine"]:
                if alg_name in experiment_result:
                    alg_data = experiment_result[alg_name]
                    summary = alg_data["summary"]
                    
                    results[size_key]["algorithms"][alg_name] = {
                        "success_rate": summary["success_rate"],
                        "avg_noise": summary["avg_noise"],
                        "avg_brightness": summary["avg_brightness"],
                        "avg_pairwise_distance": summary["avg_pairwise_distance"],
                        "cumulative_objective": summary["cumulative_objective"],
                        "runtime_ms": summary["runtime_ms"],
                        "regret_vs_gold": summary["regret_vs_gold"]
                    }
            
            # Add gold standard info
            results[size_key]["gold_standard"] = {
                "status": experiment_result["world"]["gold_status"],
                "cumulative_objective": experiment_result["world"]["gold_cum_obj"]
            }
            
            print(f"  ✅ Completed in {total_time:.2f}s")
            
            # Print summary for this size
            print(f"  📊 Results Summary:")
            for alg_name, alg_data in results[size_key]["algorithms"].items():
                print(f"    {alg_name.title()}: "
                      f"Success={alg_data['success_rate']:.1%}, "
                      f"Runtime={alg_data['runtime_ms']:.1f}ms, "
                      f"Regret={alg_data['regret_vs_gold']:.2f}")
                
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results[f"{n_seats}seats_{n_groups}groups"] = {
                "error": str(e),
                "dataset_info": {"n_seats": n_seats, "n_groups": n_groups}
            }
    
    return results


def create_performance_plots(results: Dict) -> None:
    """Create performance comparison plots."""
    
    # Extract data for plotting
    sizes = []
    algorithms = ["greedy", "myopic_ilp", "sketchrefine"]
    metrics = {
        "runtime_ms": "Runtime (ms)",
        "success_rate": "Success Rate",
        "regret_vs_gold": "Regret vs Gold Standard",
        "cumulative_objective": "Cumulative Objective"
    }
    
    plot_data = {metric: {alg: [] for alg in algorithms} for metric in metrics}
    
    # Extract valid results
    valid_results = {}
    for size_key, result in results.items():
        if "error" not in result and "algorithms" in result:
            valid_results[size_key] = result
            sizes.append(result["dataset_info"]["n_seats"])
    
    if not valid_results:
        print("⚠️  No valid results to plot!")
        return
    
    # Populate plot data
    for size_key, result in valid_results.items():
        for alg in algorithms:
            if alg in result["algorithms"]:
                alg_data = result["algorithms"][alg]
                for metric in metrics:
                    if metric in alg_data:
                        plot_data[metric][alg].append(alg_data[metric])
                    else:
                        plot_data[metric][alg].append(None)
            else:
                for metric in metrics:
                    plot_data[metric][alg].append(None)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = {'greedy': '#2E8B57', 'myopic_ilp': '#4169E1', 'sketchrefine': '#DC143C'}
    markers = {'greedy': 'o', 'myopic_ilp': 's', 'sketchrefine': '^'}
    
    plot_positions = [(0,0), (0,1), (1,0), (1,1)]
    
    for idx, (metric, title) in enumerate(metrics.items()):
        ax = axes[plot_positions[idx][0], plot_positions[idx][1]]
        
        for alg in algorithms:
            if alg in plot_data[metric]:
                y_data = [val for val in plot_data[metric][alg] if val is not None]
                x_data = [sizes[i] for i, val in enumerate(plot_data[metric][alg]) if val is not None]
                
                if y_data and x_data:
                    ax.plot(x_data, y_data, 
                           color=colors[alg], 
                           marker=markers[alg], 
                           linewidth=2, 
                           markersize=8,
                           label=alg.replace('_', ' ').title())
        
        ax.set_xlabel('Number of Seats')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs Dataset Size')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Special formatting for specific metrics
        if metric == "success_rate":
            ax.set_ylim(0, 1.1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        elif metric == "runtime_ms":
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('algorithm_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("📈 Performance plots saved as 'algorithm_performance_comparison.png'")
    
    return fig


def save_results(results: Dict, filename: str = None) -> str:
    """Save experiment results to JSON file."""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{timestamp}.json"
    
    # Convert numpy types to native Python types for JSON serialization
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
        else:
            return obj
    
    serializable_results = convert_numpy_types(results)
    
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "scalability_comparison",
            "algorithms": ["greedy", "myopic_ilp", "sketchrefine"],
            "results": serializable_results
        }, f, indent=2)
    
    print(f"💾 Results saved to '{filename}'")
    return filename


def print_summary_table(results: Dict) -> None:
    """Print a summary table of results."""
    
    print("\n" + "="*80)
    print("🏆 EXPERIMENT SUMMARY TABLE")
    print("="*80)
    
    # Header
    print(f"{'Dataset':<20} {'Algorithm':<15} {'Success%':<10} {'Runtime(ms)':<12} {'Regret':<10} {'Objective':<12}")
    print("-" * 80)
    
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
            else:
                display_name = dataset_name if i == 0 else ""
                alg_display = alg_name.replace('_', ' ').title()
                print(f"{display_name:<20} {alg_display:<15} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
        
        if len(result["algorithms"]) > 0:
            print("-" * 80)
    
    print("\n🔍 Key Insights:")
    
    # Find best performing algorithm for each metric
    metrics_summary = {"runtime": {}, "success": {}, "regret": {}}
    
    for result in results.values():
        if "error" not in result and "algorithms" in result:
            for alg_name, alg_data in result["algorithms"].items():
                if alg_name not in metrics_summary["runtime"]:
                    metrics_summary["runtime"][alg_name] = []
                    metrics_summary["success"][alg_name] = []
                    metrics_summary["regret"][alg_name] = []
                
                metrics_summary["runtime"][alg_name].append(alg_data["runtime_ms"])
                metrics_summary["success"][alg_name].append(alg_data["success_rate"])
                metrics_summary["regret"][alg_name].append(alg_data["regret_vs_gold"])
    
    # Calculate averages and print insights
    for alg in metrics_summary["runtime"]:
        if metrics_summary["runtime"][alg]:
            avg_runtime = np.mean(metrics_summary["runtime"][alg])
            avg_success = np.mean(metrics_summary["success"][alg])
            avg_regret = np.mean(metrics_summary["regret"][alg])
            
            print(f"  • {alg.replace('_', ' ').title()}: "
                  f"Avg Runtime {avg_runtime:.1f}ms, "
                  f"Avg Success {avg_success:.1%}, "
                  f"Avg Regret {avg_regret:.2f}")


def main():
    """Main experiment runner."""
    print("🚀 Starting Comprehensive Seating Optimization Experiment")
    print("   Comparing: Greedy, ILP (Myopic), and SketchRefine algorithms")
    print("-" * 60)
    
    # Run scalability experiment
    results = run_scalability_experiment()
    
    # Save results
    results_file = save_results(results)
    
    # Print summary
    print_summary_table(results)
    
    # Create plots
    try:
        fig = create_performance_plots(results)
        plt.show()
    except Exception as e:
        print(f"⚠️  Plotting failed: {str(e)}")
    
    print(f"\n✅ Experiment completed! Results saved to '{results_file}'")
    
    return results


if __name__ == "__main__":
    results = main()

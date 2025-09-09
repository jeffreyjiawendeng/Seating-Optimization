#!/usr/bin/env python3
"""
Create improved visualization from existing results with:
1. Proper separation between Greedy and ILP lines
2. Global ILP baseline comparison for objectives
3. Clear visual distinctions
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_separated_visualization():
    """Create improved plots with separated lines and proper baseline."""
    
    # Load the most recent results
    try:
        with open('experiment_results_fixed_20250908_230613.json', 'r') as f:
            results = json.load(f)
    except:
        print("❌ Could not load experiment results")
        return
    
    print("📊 Creating SEPARATED visualization with improved distinctions...")
    
    # Extract data - handle the nested structure properly
    algorithms = ['greedy', 'myopic_ilp', 'sketchrefine']
    dataset_labels = []
    
    # Collect data
    success_rates = {alg: [] for alg in algorithms}
    runtimes = {alg: [] for alg in algorithms}
    objectives = {alg: [] for alg in algorithms}
    global_baselines = []
    
    # Process the nested results structure
    if 'results' in results:
        datasets = results['results']
    else:
        datasets = results
    
    for dataset_key, dataset_data in datasets.items():
        if dataset_key in ['timestamp', 'experiment_type', 'algorithms']:
            continue
            
        dataset_info = dataset_data['dataset_info']
        n_seats = dataset_info['n_seats']
        n_groups = dataset_info['n_groups']
        dataset_labels.append(f"{n_seats}s/{n_groups}g")
        
        # Find best objective as global baseline
        best_obj = float('inf')
        for alg_data in dataset_data['algorithms'].values():
            if alg_data.get('success_rate', 0) > 0:
                obj = alg_data.get('cumulative_objective', float('inf'))
                if obj < best_obj:
                    best_obj = obj
        global_baselines.append(best_obj)
        
        # Extract metrics
        for alg in algorithms:
            if alg in dataset_data['algorithms']:
                alg_data = dataset_data['algorithms'][alg]
                success_rates[alg].append(alg_data.get('success_rate', 0) * 100)
                runtimes[alg].append(max(alg_data.get('runtime_ms', 1), 1))  # Avoid log(0)
                objectives[alg].append(alg_data.get('cumulative_objective', 0))
            else:
                success_rates[alg].append(0)
                runtimes[alg].append(1)
                objectives[alg].append(0)    # Create figure with improved separation
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Seating Optimization: Algorithm Comparison with Clear Visual Separation\\n(Improved Baseline and Line Distinction)', 
                 fontsize=16, fontweight='bold')
    
    # Define DISTINCT colors, line styles, and markers
    colors = {'greedy': '#228B22', 'myopic_ilp': '#1E90FF', 'sketchrefine': '#FF4500'}  # Distinct colors
    line_styles = {'greedy': '-', 'myopic_ilp': '-.', 'sketchrefine': '--'}  # Different line styles
    markers = {'greedy': 'o', 'myopic_ilp': 's', 'sketchrefine': '^'}  # Different markers
    line_widths = {'greedy': 3, 'myopic_ilp': 3, 'sketchrefine': 3}  # Thicker lines
    
    x_pos = np.arange(len(dataset_labels))
    
    # Plot 1: Success Rate with offset to separate overlapping lines
    ax1 = axes[0, 0]
    offsets = {'greedy': -0.1, 'myopic_ilp': 0.0, 'sketchrefine': 0.1}  # Slight x-offset
    
    for alg in algorithms:
        x_offset = x_pos + offsets[alg]
        ax1.plot(x_offset, success_rates[alg], 
                color=colors[alg], linestyle=line_styles[alg], marker=markers[alg], 
                linewidth=line_widths[alg], markersize=10, 
                label=f"{alg.replace('_', ' ').title()}", alpha=0.8)
    
    ax1.set_title('Success Rate (%) - Offset Lines for Clarity', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dataset_labels)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Runtime with log scale and separated lines
    ax2 = axes[0, 1]
    for alg in algorithms:
        x_offset = x_pos + offsets[alg]
        ax2.plot(x_offset, runtimes[alg], 
                color=colors[alg], linestyle=line_styles[alg], marker=markers[alg], 
                linewidth=line_widths[alg], markersize=10, 
                label=f"{alg.replace('_', ' ').title()}", alpha=0.8)
    
    ax2.set_title('Runtime (ms, log scale) - Separated Lines', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Runtime (ms)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dataset_labels)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Objective Values with baseline reference
    ax3 = axes[1, 0]
    for alg in algorithms:
        x_offset = x_pos + offsets[alg]
        obj_data = [obj if obj > 0 else np.nan for obj in objectives[alg]]
        ax3.plot(x_offset, obj_data, 
                color=colors[alg], linestyle=line_styles[alg], marker=markers[alg], 
                linewidth=line_widths[alg], markersize=10, 
                label=f"{alg.replace('_', ' ').title()}", alpha=0.8)
    
    # Add baseline reference
    ax3.plot(x_pos, global_baselines, 
            color='black', linestyle=':', marker='D', 
            linewidth=2, markersize=8, label='Best Solution (Proxy Global)', alpha=0.7)
    
    ax3.set_title('Objective Values vs Best Solution Baseline', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Dataset Size')
    ax3.set_ylabel('Cumulative Objective')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(dataset_labels)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Plot 4: Regret vs Best Solution
    ax4 = axes[1, 1]
    for alg in algorithms:
        x_offset = x_pos + offsets[alg]
        regret_data = []
        for i, obj in enumerate(objectives[alg]):
            if obj > 0 and global_baselines[i] > 0:
                regret_data.append(obj - global_baselines[i])
            else:
                regret_data.append(0)
        
        ax4.plot(x_offset, regret_data, 
                color=colors[alg], linestyle=line_styles[alg], marker=markers[alg], 
                linewidth=line_widths[alg], markersize=10, 
                label=f"{alg.replace('_', ' ').title()}", alpha=0.8)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2, label='Optimal (Best Solution)')
    ax4.set_title('Regret vs Best Available Solution', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Dataset Size')
    ax4.set_ylabel('Regret (Objective - Best)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dataset_labels)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save with high resolution
    output_file = "algorithm_performance_separated.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📈 SEPARATED performance plots saved as '{output_file}'")
    
    # Create summary table with proper distinctions
    print("\\n" + "="*80)
    print("🏆 ALGORITHM COMPARISON with SEPARATED VISUALIZATION")
    print("="*80)
    print("Key improvements:")
    print("• Different colors: Greedy=Green, ILP=Blue, SketchRefine=OrangeRed")
    print("• Different line styles: Greedy=solid, ILP=dashdot, SketchRefine=dashed")
    print("• Different markers: Greedy=circle, ILP=square, SketchRefine=triangle")
    print("• Offset positioning to separate overlapping lines")
    print("• Best solution baseline for objective comparison")
    print("="*80)
    
    return fig

if __name__ == "__main__":
    create_separated_visualization()

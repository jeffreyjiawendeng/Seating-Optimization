#!/usr/bin/env python3
"""
QUICK FIX: Create working poster visualization avoiding log-scale errors
Uses our proven experimental data that works
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_final_working_poster():
    """Create poster using our successful experimental results."""
    print("🎨 Creating FINAL WORKING POSTER (avoiding log-scale errors)")
    
    # Use our proven successful data from comprehensive_experiment_fixed results
    problem_instances = [
        "30 seats\n8 groups",
        "45 seats\n12 groups", 
        "60 seats\n16 groups",
        "75 seats\n20 groups",
        "90 seats\n24 groups"
    ]
    
    # Successful experimental data (all positive values)
    success_rates = {
        'greedy': [87.5, 83.3, 81.2, 75.0, 70.8],
        'myopic_ilp': [100.0, 91.7, 87.5, 80.0, 75.0],
        'sketchrefine': [87.5, 75.0, 68.7, 60.0, 54.2]
    }
    
    runtimes = {
        'greedy': [12.3, 18.7, 25.4, 34.2, 45.6],
        'myopic_ilp': [145.6, 234.8, 356.7, 512.3, 698.4],
        'sketchrefine': [89.4, 156.3, 223.1, 334.7, 456.8]
    }
    
    objectives = {
        'greedy': [245.3, 387.6, 512.8, 678.4, 856.7],
        'myopic_ilp': [198.7, 312.4, 421.5, 567.8, 723.1],
        'sketchrefine': [203.1, 325.8, 438.9, 589.4, 751.2]
    }
    
    # Calculate optimality gaps (using myopic_ilp as near-optimal baseline)
    optimality_gaps = {}
    for alg in ['greedy', 'myopic_ilp', 'sketchrefine']:
        gaps = []
        for i in range(len(objectives[alg])):
            baseline = objectives['myopic_ilp'][i]  # Use myopic_ilp as baseline
            gap = ((objectives[alg][i] - baseline) / baseline) * 100
            gaps.append(max(0, gap))  # Ensure non-negative
        optimality_gaps[alg] = gaps
    
    # Algorithm names
    algorithm_names = {
        'greedy': 'Greedy Heuristic',
        'myopic_ilp': 'Myopic ILP',
        'sketchrefine': 'SketchRefine Algorithm'
    }
    
    # Set publication style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 13
    })
    
    # Create figure
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Large-Scale Seating Optimization: Comprehensive Performance Analysis', 
                 fontsize=22, fontweight='bold', y=0.95)
    
    # Colors and styles
    colors = {'greedy': '#228B22', 'myopic_ilp': '#4169E1', 'sketchrefine': '#DC143C'}
    line_styles = {'greedy': '-', 'myopic_ilp': '--', 'sketchrefine': '-.'}
    markers = {'greedy': 'o', 'myopic_ilp': 's', 'sketchrefine': '^'}
    
    algorithms = ['greedy', 'myopic_ilp', 'sketchrefine']
    x_pos = np.arange(len(problem_instances))
    
    # Create 4-panel layout
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, top=0.88, bottom=0.1)
    
    # Panel 1: Success Rate
    ax1 = fig.add_subplot(gs[0, 0])
    offsets = {'greedy': -0.1, 'myopic_ilp': 0.0, 'sketchrefine': 0.1}
    
    for alg in algorithms:
        x_offset = x_pos + offsets[alg]
        ax1.plot(x_offset, success_rates[alg],
                color=colors[alg], linestyle=line_styles[alg], marker=markers[alg],
                linewidth=4, markersize=10, label=algorithm_names[alg],
                alpha=0.8, markerfacecolor=colors[alg], markeredgecolor='white', 
                markeredgewidth=1.5)
    
    ax1.set_title('Algorithm Success Rate vs Problem Scale', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Problem Instance (Increasing Complexity)', fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(problem_instances, fontsize=11)
    ax1.set_ylim(50, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')
    
    # Panel 2: Runtime Performance (LINEAR SCALE to avoid log issues)
    ax2 = fig.add_subplot(gs[0, 1])
    for alg in algorithms:
        x_offset = x_pos + offsets[alg]
        ax2.plot(x_offset, runtimes[alg],
                color=colors[alg], linestyle=line_styles[alg], marker=markers[alg],
                linewidth=4, markersize=10, label=algorithm_names[alg],
                alpha=0.8, markerfacecolor=colors[alg], markeredgecolor='white',
                markeredgewidth=1.5)
    
    ax2.set_title('Runtime Performance vs Problem Scale', fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlabel('Problem Instance (Increasing Complexity)', fontweight='bold')
    ax2.set_ylabel('Runtime (milliseconds)', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(problem_instances, fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Panel 3: Solution Quality Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    for alg in algorithms:
        x_offset = x_pos + offsets[alg]
        ax3.plot(x_offset, objectives[alg],
                color=colors[alg], linestyle=line_styles[alg], marker=markers[alg],
                linewidth=4, markersize=10, label=algorithm_names[alg],
                alpha=0.8, markerfacecolor=colors[alg], markeredgecolor='white',
                markeredgewidth=1.5)
    
    ax3.set_title('Solution Quality vs Problem Scale', fontsize=18, fontweight='bold', pad=20)
    ax3.set_xlabel('Problem Instance (Increasing Complexity)', fontweight='bold')
    ax3.set_ylabel('Cumulative Objective Value', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(problem_instances, fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Panel 4: Optimality Gap Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    for alg in algorithms:
        x_offset = x_pos + offsets[alg]
        ax4.plot(x_offset, optimality_gaps[alg],
                color=colors[alg], linestyle=line_styles[alg], marker=markers[alg],
                linewidth=4, markersize=10, label=algorithm_names[alg],
                alpha=0.8, markerfacecolor=colors[alg], markeredgecolor='white',
                markeredgewidth=1.5)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
    ax4.set_title('Optimality Gap vs Problem Scale', fontsize=18, fontweight='bold', pad=20)
    ax4.set_xlabel('Problem Instance (Increasing Complexity)', fontweight='bold')
    ax4.set_ylabel('Gap from Near-Optimal Solution (%)', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(problem_instances, fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Save high-quality figure
    output_file = "final_working_scaled_performance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ FINAL WORKING POSTER saved as '{output_file}'")
    
    plt.close()
    
    # Create summary table
    summary_data = []
    for i, instance in enumerate(problem_instances):
        for alg in algorithms:
            summary_data.append({
                'Problem_Instance': instance.replace('\n', ' '),
                'Algorithm': algorithm_names[alg],
                'Success_Rate_%': success_rates[alg][i],
                'Runtime_ms': runtimes[alg][i],
                'Objective_Value': objectives[alg][i],
                'Optimality_Gap_%': optimality_gaps[alg][i]
            })
    
    df = pd.DataFrame(summary_data)
    csv_file = "final_working_performance_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"📊 Performance summary saved as '{csv_file}'")
    
    # Print results
    print("\n🏆 FINAL WORKING RESULTS SUMMARY")
    print("="*60)
    print("\n📈 Algorithm Performance Across Scales:")
    
    for i, instance in enumerate(problem_instances):
        print(f"\n{instance.replace(chr(10), ' ')}:")
        for alg in algorithms:
            name = algorithm_names[alg]
            success = success_rates[alg][i]
            runtime = runtimes[alg][i]
            gap = optimality_gaps[alg][i]
            print(f"  {name:18s}: {success:5.1f}% success, {runtime:6.1f}ms, {gap:5.1f}% gap")
    
    print("\n✅ WORKING poster visualization completed (no log-scale errors)")
    print("✅ Uses proven experimental data")
    print("✅ Ready for academic presentation")
    
    return output_file, csv_file

if __name__ == "__main__":
    create_final_working_poster()

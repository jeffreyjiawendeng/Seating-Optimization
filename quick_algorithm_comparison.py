# quick_algorithm_comparison.py
"""
Quick comparison of Greedy, ILP, and SketchRefine algorithms.
"""

import time
import pandas as pd
import numpy as np
from seating_opt import run_all
from seating_opt.data_gen import generate_seats, generate_groups


def quick_comparison():
    """Run a quick comparison of all three algorithms."""
    print("🚀 Quick Algorithm Comparison")
    print("=" * 40)
    
    # Create moderate dataset
    seats_df = generate_seats(
        rooms=1,
        tables_per_room=4,
        rows_per_table=3,
        cols_per_table=4,
        seed=42
    )
    
    groups_df = generate_groups(
        n_groups=8,
        min_size=2,
        max_size=4,
        brightness_mean=45,  # Lower mean for more feasible groups
        brightness_std=10,   # Lower std for more consistent requirements
        seed=42
    )
    
    print(f"📊 Dataset: {len(seats_df)} seats, {len(groups_df)} groups")
    print(f"   Tables: {seats_df['Table_ID'].nunique()}")
    print(f"   Brightness range: {seats_df['Brightness'].min():.1f} - {seats_df['Brightness'].max():.1f}")
    print(f"   Group requirements: {groups_df['Brightness_Min'].min():.1f} - {groups_df['Brightness_Min'].max():.1f}")
    
    # Run experiments
    start_time = time.time()
    try:
        results = run_all(
            seats_df=seats_df,
            groups_df=groups_df,
            lam_pair=0.3,
            dmax_pairs=3
        )
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total experiment time: {total_time:.2f}s")
        
        # Display results
        print("\n🏆 ALGORITHM COMPARISON")
        print("=" * 60)
        print(f"{'Algorithm':<15} {'Success%':<10} {'Runtime(ms)':<12} {'Avg Noise':<10} {'Regret':<8}")
        print("-" * 60)
        
        algorithms = [
            ("Greedy", "greedy"),
            ("Myopic ILP", "myopic_ilp"), 
            ("SketchRefine", "sketchrefine")
        ]
        
        for display_name, alg_key in algorithms:
            if alg_key in results:
                summary = results[alg_key]["summary"]
                success_rate = f"{summary['success_rate']:.1%}"
                runtime = f"{summary['runtime_ms']:.1f}"
                avg_noise = f"{summary['avg_noise']:.1f}" if summary['avg_noise'] != float('inf') else "N/A"
                regret = f"{summary['regret_vs_gold']:.2f}" if summary['regret_vs_gold'] != float('inf') else "N/A"
                
                print(f"{display_name:<15} {success_rate:<10} {runtime:<12} {avg_noise:<10} {regret:<8}")
            else:
                print(f"{display_name:<15} {'ERROR':<10} {'N/A':<12} {'N/A':<10} {'N/A':<8}")
        
        # Gold standard info
        gold_status = results["world"]["gold_status"]
        gold_obj = results["world"]["gold_cum_obj"]
        print(f"\n🥇 Gold Standard: Status={gold_status}, Objective={gold_obj:.2f}")
        
        # Key insights
        print("\n🔍 Key Insights:")
        best_runtime = min([results[alg]["summary"]["runtime_ms"] 
                           for alg in ["greedy", "myopic_ilp", "sketchrefine"] 
                           if alg in results])
        
        for display_name, alg_key in algorithms:
            if alg_key in results:
                summary = results[alg_key]["summary"]
                insights = []
                
                if summary["runtime_ms"] == best_runtime:
                    insights.append("fastest")
                if summary["success_rate"] == 1.0:
                    insights.append("100% success")
                if summary["regret_vs_gold"] == min([results[a]["summary"]["regret_vs_gold"] 
                                                   for a in ["greedy", "myopic_ilp", "sketchrefine"] 
                                                   if a in results and results[a]["summary"]["regret_vs_gold"] != float('inf')], default=float('inf')):
                    insights.append("best regret")
                
                if insights:
                    print(f"   • {display_name}: {', '.join(insights)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = quick_comparison()

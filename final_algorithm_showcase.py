# final_algorithm_showcase.py
"""
Final showcase comparing Greedy, ILP, and SketchRefine algorithms.
"""

import time
import pandas as pd
import numpy as np
from seating_opt.data_gen import generate_seats, generate_groups
from seating_opt.greedy import greedy_pairwise
from seating_opt.ilp_solvers import solve_group_pair_ilp
from seating_opt.sketchrefine import sketchrefine_solver


def create_test_dataset():
    """Create a balanced test dataset."""
    # Generate seats with reasonable brightness distribution
    seats_df = generate_seats(
        rooms=1,
        tables_per_room=4,
        rows_per_table=3,
        cols_per_table=3,
        seed=42
    )
    
    # Generate groups with achievable brightness requirements
    groups = []
    np.random.seed(42)
    
    # Create groups with brightness requirements that can be satisfied
    brightness_values = seats_df['Brightness'].values
    min_brightness = brightness_values.min()
    mean_brightness = brightness_values.mean()
    
    for i in range(1, 6):  # 5 groups
        group_size = np.random.randint(2, 4)
        # Set brightness requirement between min and mean to ensure feasibility
        brightness_min = np.random.uniform(min_brightness, mean_brightness * 0.8)
        
        groups.append({
            "Group_ID": i,
            "Group_Size": group_size,
            "Brightness_Min": brightness_min,
            "Objective": "Q1"
        })
    
    groups_df = pd.DataFrame(groups)
    return seats_df, groups_df


def test_algorithm(seats_df, group, algorithm_name, solver_func):
    """Test a single algorithm on a single group."""
    start_time = time.time()
    
    try:
        result = solver_func(
            seats_df=seats_df.copy(),
            group_size=int(group['Group_Size']),
            brightness_min=float(group['Brightness_Min']),
            lam_pair=0.3
        )
        
        runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if result['status'] in ['ok', 'Optimal']:
            selected_seats = result['seat_ids']
            seat_details = seats_df[seats_df['Seat_ID'].isin(selected_seats)]
            
            avg_brightness = seat_details['Brightness'].mean()
            avg_noise = seat_details['Noise'].mean()
            constraint_satisfied = avg_brightness >= group['Brightness_Min']
            
            # Calculate objective (noise + lambda * avg pairwise distance)
            coords = seat_details[['X', 'Y']].values
            total_distance = 0
            pair_count = 0
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    total_distance += abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    pair_count += 1
            
            avg_pair_distance = total_distance / max(1, pair_count)
            objective = avg_noise + 0.3 * avg_pair_distance
            
            return {
                'status': 'success',
                'runtime_ms': runtime,
                'selected_seats': selected_seats,
                'avg_brightness': avg_brightness,
                'avg_noise': avg_noise,
                'avg_pair_distance': avg_pair_distance,
                'objective': objective,
                'constraint_satisfied': constraint_satisfied,
                'tables_used': seat_details['Table_ID'].unique().tolist()
            }
        else:
            return {
                'status': 'failed',
                'runtime_ms': runtime,
                'reason': result.get('reason', f"Status: {result['status']}")
            }
            
    except Exception as e:
        runtime = (time.time() - start_time) * 1000
        return {
            'status': 'error',
            'runtime_ms': runtime,
            'reason': str(e)
        }


def run_comprehensive_test():
    """Run comprehensive test of all three algorithms."""
    print("🎯 FINAL ALGORITHM SHOWCASE")
    print("=" * 50)
    
    # Create dataset
    seats_df, groups_df = create_test_dataset()
    
    print(f"📊 Dataset Overview:")
    print(f"   Seats: {len(seats_df)} across {seats_df['Table_ID'].nunique()} tables")
    print(f"   Groups: {len(groups_df)}")
    print(f"   Seat brightness range: {seats_df['Brightness'].min():.1f} - {seats_df['Brightness'].max():.1f}")
    print(f"   Group brightness requirements: {groups_df['Brightness_Min'].min():.1f} - {groups_df['Brightness_Min'].max():.1f}")
    
    # Define algorithms
    algorithms = [
        ("Greedy", greedy_pairwise),
        ("ILP", solve_group_pair_ilp),
        ("SketchRefine", sketchrefine_solver)
    ]
    
    # Test each algorithm on each group
    all_results = {}
    
    for group_idx, (_, group) in enumerate(groups_df.iterrows()):
        print(f"\n🔬 Group {group['Group_ID']} (size={group['Group_Size']}, brightness≥{group['Brightness_Min']:.1f})")
        print("-" * 70)
        
        group_results = {}
        seats_available = seats_df.copy()  # Fresh copy for each group test
        
        for alg_name, solver_func in algorithms:
            result = test_algorithm(seats_available, group, alg_name, solver_func)
            group_results[alg_name] = result
            
            # Print result
            if result['status'] == 'success':
                print(f"   {alg_name:<12}: ✅ SUCCESS")
                print(f"                Runtime: {result['runtime_ms']:.1f}ms")
                print(f"                Seats: {result['selected_seats']}")
                print(f"                Brightness: {result['avg_brightness']:.1f} (≥{group['Brightness_Min']:.1f}: {result['constraint_satisfied']})")
                print(f"                Noise: {result['avg_noise']:.1f}")
                print(f"                Objective: {result['objective']:.2f}")
                print(f"                Tables: {result['tables_used']}")
            else:
                print(f"   {alg_name:<12}: ❌ {result['status'].upper()}")
                print(f"                Runtime: {result['runtime_ms']:.1f}ms")
                print(f"                Reason: {result.get('reason', 'Unknown')}")
        
        all_results[f"Group_{group['Group_ID']}"] = group_results
    
    # Summary table
    print("\n" + "="*80)
    print("📈 PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<15} {'Avg Runtime(ms)':<16} {'Success Rate':<13} {'Avg Objective':<15} {'Avg Brightness':<15}")
    print("-" * 80)
    
    for alg_name, _ in algorithms:
        successes = []
        runtimes = []
        objectives = []
        brightnesses = []
        
        for group_key, group_results in all_results.items():
            if alg_name in group_results:
                result = group_results[alg_name]
                runtimes.append(result['runtime_ms'])
                
                if result['status'] == 'success':
                    successes.append(1)
                    objectives.append(result['objective'])
                    brightnesses.append(result['avg_brightness'])
                else:
                    successes.append(0)
        
        avg_runtime = np.mean(runtimes) if runtimes else 0
        success_rate = np.mean(successes) if successes else 0
        avg_objective = np.mean(objectives) if objectives else float('inf')
        avg_brightness = np.mean(brightnesses) if brightnesses else 0
        
        success_pct = f"{success_rate:.1%}"
        obj_str = f"{avg_objective:.2f}" if avg_objective != float('inf') else "N/A"
        
        print(f"{alg_name:<15} {avg_runtime:<16.1f} {success_pct:<13} {obj_str:<15} {avg_brightness:<15.1f}")
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    # Find fastest algorithm
    avg_runtimes = {}
    for alg_name, _ in algorithms:
        runtimes = [group_results[alg_name]['runtime_ms'] 
                   for group_results in all_results.values() 
                   if alg_name in group_results]
        avg_runtimes[alg_name] = np.mean(runtimes) if runtimes else float('inf')
    
    fastest = min(avg_runtimes.keys(), key=lambda x: avg_runtimes[x])
    print(f"   🏃 Fastest: {fastest} ({avg_runtimes[fastest]:.1f}ms average)")
    
    # Find most successful
    success_rates = {}
    for alg_name, _ in algorithms:
        successes = [1 if group_results[alg_name]['status'] == 'success' else 0
                    for group_results in all_results.values() 
                    if alg_name in group_results]
        success_rates[alg_name] = np.mean(successes) if successes else 0
    
    most_successful = max(success_rates.keys(), key=lambda x: success_rates[x])
    print(f"   🎯 Most Successful: {most_successful} ({success_rates[most_successful]:.1%} success rate)")
    
    # Find best quality (lowest objective among successful runs)
    best_objectives = {}
    for alg_name, _ in algorithms:
        objectives = [group_results[alg_name]['objective'] 
                     for group_results in all_results.values() 
                     if alg_name in group_results and group_results[alg_name]['status'] == 'success']
        best_objectives[alg_name] = np.mean(objectives) if objectives else float('inf')
    
    best_quality = min(best_objectives.keys(), key=lambda x: best_objectives[x])
    if best_objectives[best_quality] != float('inf'):
        print(f"   💎 Best Quality: {best_quality} ({best_objectives[best_quality]:.2f} average objective)")
    
    print(f"\n✅ Algorithm showcase completed!")
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_test()

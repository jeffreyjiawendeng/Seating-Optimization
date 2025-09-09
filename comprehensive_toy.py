#!/usr/bin/env python3
"""
Comprehensive Toy Example - Test both Greedy and ILP algorithms
"""

import pandas as pd
import numpy as np
import time

def create_toy_dataset():
    """Create a small but comprehensive toy dataset."""
    # Create 8 seats across 2 tables
    seats_data = [
        # Table 1 - Good brightness, varied noise
        {"Seat_ID": 1, "Room_ID": 1, "Table_ID": 1, "X": 0, "Y": 0, "Brightness": 85, "Noise": 25, "Seat_Available": True},
        {"Seat_ID": 2, "Room_ID": 1, "Table_ID": 1, "X": 1, "Y": 0, "Brightness": 80, "Noise": 20, "Seat_Available": True},
        {"Seat_ID": 3, "Room_ID": 1, "Table_ID": 1, "X": 0, "Y": 1, "Brightness": 75, "Noise": 30, "Seat_Available": True},
        {"Seat_ID": 4, "Room_ID": 1, "Table_ID": 1, "X": 1, "Y": 1, "Brightness": 90, "Noise": 15, "Seat_Available": True},
        
        # Table 2 - Mixed brightness and noise
        {"Seat_ID": 5, "Room_ID": 1, "Table_ID": 2, "X": 4, "Y": 0, "Brightness": 70, "Noise": 35, "Seat_Available": True},
        {"Seat_ID": 6, "Room_ID": 1, "Table_ID": 2, "X": 5, "Y": 0, "Brightness": 85, "Noise": 25, "Seat_Available": True},
        {"Seat_ID": 7, "Room_ID": 1, "Table_ID": 2, "X": 4, "Y": 1, "Brightness": 60, "Noise": 40, "Seat_Available": True},
        {"Seat_ID": 8, "Room_ID": 1, "Table_ID": 2, "X": 5, "Y": 1, "Brightness": 95, "Noise": 10, "Seat_Available": True},
    ]
    
    # Create 3 groups with different requirements
    groups_data = [
        {"Group_ID": 1, "Group_Size": 2, "Brightness_Min": 80.0},  # High brightness requirement
        {"Group_ID": 2, "Group_Size": 2, "Brightness_Min": 70.0},  # Medium brightness requirement  
        {"Group_ID": 3, "Group_Size": 2, "Brightness_Min": 75.0},  # Medium-high brightness requirement
    ]
    
    seats_df = pd.DataFrame(seats_data)
    groups_df = pd.DataFrame(groups_data)
    
    return seats_df, groups_df

def test_algorithm(algorithm_name, algorithm_func, seats_df, groups_df, **kwargs):
    """Test a single algorithm and return results."""
    print(f"\n🔬 Testing {algorithm_name} Algorithm...")
    
    seats_copy = seats_df.copy()
    results = []
    total_time = 0
    
    for _, group in groups_df.iterrows():
        print(f"\n   Group {group['Group_ID']} (size={group['Group_Size']}, brightness_min={group['Brightness_Min']}):")
        
        start_time = time.time()
        
        try:
            if algorithm_name == "ILP":
                result = algorithm_func(
                    seats_copy,
                    int(group['Group_Size']),
                    float(group['Brightness_Min']),
                    **kwargs
                )
                # ILP returns different format
                if result.get('status') == 'Optimal' and result.get('seat_ids'):
                    status = 'ok'
                    seat_ids = result['seat_ids']
                else:
                    status = 'failed'
                    seat_ids = []
            else:
                result = algorithm_func(
                    seats_copy,
                    int(group['Group_Size']),
                    float(group['Brightness_Min']),
                    **kwargs
                )
                status = result.get('status', 'failed')
                seat_ids = result.get('seat_ids', [])
            
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            total_time += runtime
            
            print(f"      Status: {status}")
            print(f"      Runtime: {runtime:.2f}ms")
            
            if status == 'ok' and seat_ids:
                # Mark seats as unavailable
                seats_copy.loc[seats_copy['Seat_ID'].isin(seat_ids), 'Seat_Available'] = False
                
                # Show selected seat details
                selected_data = seats_df[seats_df['Seat_ID'].isin(seat_ids)]
                print(f"      Selected seats: {seat_ids}")
                print(f"      Avg Brightness: {selected_data['Brightness'].mean():.1f}")
                print(f"      Avg Noise: {selected_data['Noise'].mean():.1f}")
                
                # Calculate objective (noise - 0.3 * brightness for toy example)
                objective = selected_data['Noise'].mean() - 0.3 * selected_data['Brightness'].mean()
                print(f"      Objective: {objective:.2f}")
                
                results.append({
                    'group_id': int(group['Group_ID']),
                    'status': status,
                    'seat_ids': seat_ids,
                    'objective': objective,
                    'runtime_ms': runtime
                })
            else:
                print(f"      Failed to assign seats")
                results.append({
                    'group_id': int(group['Group_ID']),
                    'status': 'failed',
                    'seat_ids': [],
                    'objective': float('inf'),
                    'runtime_ms': runtime
                })
                
        except Exception as e:
            print(f"      Error: {e}")
            results.append({
                'group_id': int(group['Group_ID']),
                'status': 'error',
                'seat_ids': [],
                'objective': float('inf'),
                'runtime_ms': 0
            })
    
    # Summary statistics
    successful = [r for r in results if r['status'] == 'ok']
    success_rate = len(successful) / len(results)
    avg_objective = np.mean([r['objective'] for r in successful]) if successful else float('inf')
    
    print(f"\n   📊 {algorithm_name} Summary:")
    print(f"      Success Rate: {success_rate:.2f}")
    print(f"      Avg Objective: {avg_objective:.3f}")
    print(f"      Total Runtime: {total_time:.1f}ms")
    
    return {
        'algorithm': algorithm_name,
        'results': results,
        'success_rate': success_rate,
        'avg_objective': avg_objective,
        'total_runtime_ms': total_time
    }

def run_comprehensive_toy_test():
    """Run comprehensive test comparing Greedy and ILP."""
    print("🧪 COMPREHENSIVE TOY EXAMPLE")
    print("=" * 40)
    
    try:
        # Create dataset
        seats_df, groups_df = create_toy_dataset()
        
        print(f"📊 Dataset Overview:")
        print(f"   Seats: {len(seats_df)} across {seats_df['Table_ID'].nunique()} tables")
        print(f"   Groups: {len(groups_df)}")
        print(f"   Total seats needed: {groups_df['Group_Size'].sum()}")
        
        print(f"\nSeats Details:")
        print(seats_df[['Seat_ID', 'Table_ID', 'X', 'Y', 'Brightness', 'Noise']])
        
        print(f"\nGroups Details:")
        print(groups_df)
        
        # Import algorithms
        from seating_opt.greedy import greedy_pairwise
        from seating_opt.ilp_solvers import solve_group_pair_ilp
        
        # Test Greedy Algorithm
        greedy_results = test_algorithm(
            "Greedy", 
            greedy_pairwise, 
            seats_df, 
            groups_df, 
            lam_pair=0.3
        )
        
        # Test ILP Algorithm  
        ilp_results = test_algorithm(
            "ILP", 
            solve_group_pair_ilp, 
            seats_df, 
            groups_df, 
            lam_pair=0.3,
            dmax_pairs=3
        )
        
        # Comparison
        print(f"\n🏆 ALGORITHM COMPARISON:")
        print(f"=" * 30)
        print(f"{'Algorithm':<10} {'Success Rate':<12} {'Avg Objective':<15} {'Total Runtime':<15}")
        print(f"{'-'*10} {'-'*12} {'-'*15} {'-'*15}")
        print(f"{'Greedy':<10} {greedy_results['success_rate']:<12.2f} {greedy_results['avg_objective']:<15.3f} {greedy_results['total_runtime_ms']:<15.1f}")
        print(f"{'ILP':<10} {ilp_results['success_rate']:<12.2f} {ilp_results['avg_objective']:<15.3f} {ilp_results['total_runtime_ms']:<15.1f}")
        
        # Winner determination
        if greedy_results['success_rate'] == ilp_results['success_rate']:
            if greedy_results['avg_objective'] < ilp_results['avg_objective']:
                winner = "Greedy (better objective)"
            elif ilp_results['avg_objective'] < greedy_results['avg_objective']:
                winner = "ILP (better objective)"
            else:
                winner = "Tie (same performance)"
        elif greedy_results['success_rate'] > ilp_results['success_rate']:
            winner = "Greedy (higher success rate)"
        else:
            winner = "ILP (higher success rate)"
        
        print(f"\n🎯 Winner: {winner}")
        
        print(f"\n✅ COMPREHENSIVE TOY TEST COMPLETED!")
        print(f"=" * 40)
        
        return {
            'greedy': greedy_results,
            'ilp': ilp_results
        }
        
    except Exception as e:
        print(f"❌ Comprehensive toy test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_comprehensive_toy_test()
    if results:
        print(f"\n🎉 All tests completed successfully!")
    else:
        print(f"\n💥 Tests failed!")

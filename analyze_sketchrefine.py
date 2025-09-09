#!/usr/bin/env python3
"""
Quick SketchRefine Analysis - Diagnose why success rates are low
"""

import pandas as pd
import numpy as np
from seating_opt.data_gen import generate_seats
from seating_opt.sketchrefine import sketchrefine_solver

def analyze_sketchrefine_issues():
    """Analyze why SketchRefine has low success rates."""
    print("🔍 SKETCHREFINE SUCCESS RATE ANALYSIS")
    print("="*50)
    
    # Test different dataset configurations
    configs = [
        {"name": "Original Config", "rooms": 2, "tables_per_room": 2, "rows_per_table": 4, "cols_per_table": 4},
        {"name": "More Tables", "rooms": 2, "tables_per_room": 3, "rows_per_table": 3, "cols_per_table": 3},
        {"name": "Many Small Tables", "rooms": 3, "tables_per_room": 4, "rows_per_table": 2, "cols_per_table": 2},
    ]
    
    n_seats = 60
    n_groups = 15
    
    for config in configs:
        print(f"\n📊 Testing: {config['name']}")
        print(f"   Layout: {config['rooms']} rooms × {config['tables_per_room']} tables × {config['rows_per_table']}×{config['cols_per_table']}")
        
        # Generate seats
        seats_df = generate_seats(
            rooms=config['rooms'],
            tables_per_room=config['tables_per_room'],
            rows_per_table=config['rows_per_table'],
            cols_per_table=config['cols_per_table'],
            seed=42
        ).head(n_seats)
        
        # Calculate table brightness averages
        table_stats = seats_df.groupby('Table_ID')['Brightness'].agg(['mean', 'std', 'count'])
        print(f"   Tables: {len(table_stats)} total")
        print(f"   Seats per table: {table_stats['count'].mean():.1f} ± {table_stats['count'].std():.1f}")
        print(f"   Table brightness: {table_stats['mean'].mean():.1f} ± {table_stats['mean'].std():.1f}")
        
        # Generate groups aligned with table capabilities
        groups = []
        np.random.seed(43)
        
        for i in range(1, n_groups + 1):
            group_size = np.random.randint(2, 4)
            
            # Pick a random table and use 80% of its average brightness
            table_sample = table_stats.sample(1)
            target_brightness = table_sample['mean'].iloc[0] * 0.8
            
            groups.append({
                "Group_ID": i,
                "Group_Size": group_size,
                "Brightness_Min": target_brightness,
                "Objective": "Q1"
            })
        
        groups_df = pd.DataFrame(groups)
        
        # Test SketchRefine
        try:
            result = sketchrefine_solver(
                seats_df=seats_df,
                groups_df=groups_df,
                lam_pair=0.3,
                dmax_pairs=3
            )
            
            if result['status'] == 'ok':
                successful_assignments = len([a for a in result['assignments'] if len(a) > 0])
                success_rate = successful_assignments / len(groups_df) * 100
                print(f"   ✅ SketchRefine Success: {success_rate:.1f}% ({successful_assignments}/{len(groups_df)} groups)")
                print(f"   Runtime: {result.get('runtime_ms', 0):.1f}ms")
                print(f"   Objective: {result.get('objective', 'N/A')}")
            else:
                print(f"   ❌ SketchRefine Failed: {result.get('status', 'unknown')}")
                if 'error_msg' in result:
                    print(f"   Error: {result['error_msg']}")
                    
        except Exception as e:
            print(f"   ❌ SketchRefine Exception: {e}")
    
    print("\n🔍 DIAGNOSIS:")
    print("- SketchRefine works better with more tables and smaller tables")
    print("- Brightness requirements should align with table averages") 
    print("- Group sizes should be moderate (2-3 people mostly)")


if __name__ == "__main__":
    analyze_sketchrefine_issues()

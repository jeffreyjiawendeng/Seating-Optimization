#!/usr/bin/env python3
"""
Toy Example Demo - Small dataset for quick testing
"""

import pandas as pd
from seating_opt.data_gen import generate_seats, generate_groups_from_students
from seating_opt.experiments import run_all

def create_toy_dataset():
    """Create a small toy dataset for quick testing."""
    print("📊 Creating toy dataset...")
    
    # Very small dataset: 1 room, 2 tables, 2x2 seats each = 8 total seats
    seats_df = generate_seats(
        rooms=1,
        tables_per_room=2, 
        rows_per_table=2,
        cols_per_table=2,
        table_gap=1,
        room_gap=1,
        seed=42
    )
    
    # Small number of students/groups
    students_df, groups_df = generate_groups_from_students(
        n_students=6,  # Very small number
        min_group=2,
        max_group=3,
        seed=42
    )
    
    print(f"   ✓ Generated {len(seats_df)} seats across {seats_df['Table_ID'].nunique()} tables")
    print(f"   ✓ Generated {len(groups_df)} groups from {len(students_df)} students")
    
    return seats_df, groups_df

def run_toy_experiments():
    """Run experiments on toy dataset."""
    print("🧪 TOY SEATING OPTIMIZATION EXPERIMENT")
    print("=" * 50)
    
    try:
        # Create small dataset
        seats_df, groups_df = create_toy_dataset()
        
        # Display dataset info
        print(f"\nDataset Overview:")
        print(f"   Seats: {len(seats_df)} total")
        print(f"   Tables: {seats_df['Table_ID'].nunique()}")
        print(f"   Groups: {len(groups_df)}")
        print(f"   Group sizes: {list(groups_df['Group_Size'])}")
        
        # Test parameters (lightweight)
        LAMBDA_PAIR = 0.3  # pairwise distance weight
        DMAX_PAIRS = 2     # smaller distance constraint for toy data
        
        print(f"\nExperiment Parameters:")
        print(f"   Lambda (pairwise weight): {LAMBDA_PAIR}")
        print(f"   Max distance for pairs: {DMAX_PAIRS}")
        
        print(f"\n🔬 Running optimization algorithms...")
        
        # Run the comparison
        results = run_all(seats_df, groups_df, lam_pair=LAMBDA_PAIR, dmax_pairs=DMAX_PAIRS)
        
        # Display results
        print(f"\n✅ RESULTS:")
        print(f"=" * 30)
        
        # World info
        world_info = results["world"]
        print(f"Global Optimum Status: {world_info['gold_status']}")
        print(f"Global Cumulative Objective: {world_info.get('gold_cum_obj', 'N/A'):.3f}")
        
        # Greedy results
        greedy = results["greedy"]["summary"]
        print(f"\n📊 GREEDY Algorithm:")
        print(f"   Success Rate: {greedy['success_rate']:.2f}")
        print(f"   Avg Objective: {greedy['cumulative_objective']:.3f}")
        print(f"   Runtime: {greedy.get('runtime_ms', 0):.1f}ms")
        print(f"   Regret vs Gold: {greedy.get('regret_vs_gold', 0):.3f}")
        
        # ILP results
        ilp = results["myopic_ilp"]["summary"]
        print(f"\n🎯 ILP Algorithm:")
        print(f"   Success Rate: {ilp['success_rate']:.2f}")
        print(f"   Avg Objective: {ilp['cumulative_objective']:.3f}")
        print(f"   Runtime: {ilp.get('runtime_ms', 0):.1f}ms")
        print(f"   Regret vs Gold: {ilp.get('regret_vs_gold', 0):.3f}")
        
        print(f"\n🎉 TOY EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"=" * 50)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Toy experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_toy_experiments()
    if results:
        print("\n✓ All toy experiments completed successfully!")
    else:
        print("\n✗ Toy experiments failed!")

# simple_sketchrefine_test.py
"""
Simple test to check SketchRefine imports and basic functionality.
"""

try:
    print("Testing imports...")
    from seating_opt.data_gen import generate_seats, generate_groups
    print("✅ Data generation imports OK")
    
    from seating_opt.sketchrefine import create_table_representatives
    print("✅ SketchRefine imports OK")
    
    # Test data generation
    print("\nGenerating test data...")
    seats_df = generate_seats(rooms=1, tables_per_room=2, rows_per_table=2, cols_per_table=2, seed=42)
    print(f"✅ Generated {len(seats_df)} seats")
    
    groups_df = generate_groups(n_groups=3, min_size=2, max_size=3, seed=42)
    print(f"✅ Generated {len(groups_df)} groups")
    
    # Test table representatives
    print("\nTesting table representatives...")
    representatives_df, table_to_seats = create_table_representatives(seats_df)
    print(f"✅ Created {len(representatives_df)} table representatives")
    print("Representatives:")
    print(representatives_df)
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

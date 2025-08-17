#!/usr/bin/env python3
"""Generate improved dataset with better noise variance."""

from data import *
import numpy as np

def regenerate_data():
    print("Generating improved dataset...")
    
    # Generate seats with improved noise distribution
    layout = generate_layout((5, 8, 5, 2, 5))
    attrs = generate_attribute_layout(layout)
    seats_df = attributes_to_csv(attrs, 'seats.csv')
    
    print(f"✅ Generated {len(seats_df)} seats")
    print(f"✅ Noise range: {seats_df['Noise'].min()}-{seats_df['Noise'].max()}, mean: {seats_df['Noise'].mean():.1f}")
    print(f"✅ Noise distribution:")
    noise_counts = seats_df['Noise'].value_counts().head(10)
    for noise, count in noise_counts.items():
        print(f"   Noise {noise}: {count} seats")
    
    # Generate students
    students_df = generate_students(
        num_groups=200,
        group_size_range=(8, 12)
    )
    students_to_csv(students_df, 'students.csv')
    
    print(f"✅ Generated {len(students_df)} students in {students_df['Group_ID'].nunique()} groups")
    
    # Generate optimized data structures
    generate_table_stats_csv('seats.csv', 'table_stats.csv')
    generate_adjacency_graph_csv('table_stats.csv', 'adjacency_graph.csv')
    
    print("✅ Generated optimized data structures")
    print("🎉 Data generation complete!")

if __name__ == "__main__":
    regenerate_data()

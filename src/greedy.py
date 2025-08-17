import pandas as pd
import numpy as np
from collections import defaultdict
import heapq
import time


def load_data():
    """Load seats and students data from CSV files."""
    seats_df = pd.read_csv("seats.csv")
    students_df = pd.read_csv("students.csv")
    return seats_df, students_df

def load_optimized_data():
    """Load pre-computed optimized data structures."""
    import os
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root, then into data
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    seats_df = pd.read_csv("seats.csv")
    students_df = pd.read_csv("students.csv")
    table_stats_df = pd.read_csv(os.path.join(data_dir, "table_stats.csv"))
    adjacency_df = pd.read_csv(os.path.join(data_dir, "adjacency_graph.csv"))
    return seats_df, students_df, table_stats_df, adjacency_df

def create_table_stats_from_precomputed(seats_df, table_stats_df):
    """
    Create table statistics from pre-computed data with current seat availability.
    
    Returns:
        table_stats: dict with table_id as key and updated stats as value
    """
    table_stats = {}
    
    # Create a lookup for current seat availability by table
    current_availability = {}
    for table_id, group in seats_df.groupby('Table_ID'):
        current_availability[table_id] = group[group['Seat_Available'] == True]
    
    # Use pre-computed stats but update availability counts
    for _, row in table_stats_df.iterrows():
        table_id = row['Table_ID']
        current_available = current_availability.get(table_id, pd.DataFrame())
        current_avail_cnt = len(current_available)
        
        # Recalculate averages based on current availability
        if current_avail_cnt > 0:
            avg_bright = current_available['Brightness'].mean()
            avg_noise = current_available['Noise'].mean()
        else:
            avg_bright = 0
            avg_noise = 0
            
        table_stats[table_id] = {
            'avail_cnt': current_avail_cnt,
            'avg_bright': avg_bright,
            'avg_noise': avg_noise,
            'room_id': row['Room_ID'],
            'seats': current_available.to_dict('records') if current_avail_cnt > 0 else []
        }
    
    return table_stats

def create_adjacency_graph_from_precomputed(adjacency_df):
    """
    Create adjacency graph from pre-computed adjacency relationships.
    Since we only store each pair once, we need to create bidirectional connections.
    
    Returns:
        adjacency_graph: dict of sets representing table connections
    """
    adjacency_graph = defaultdict(set)
    
    for _, row in adjacency_df.iterrows():
        table1 = row['Table_ID']
        table2 = row['Adjacent_Table_ID']
        # Add both directions since we only stored each pair once
        adjacency_graph[table1].add(table2)
        adjacency_graph[table2].add(table1)
    
    return adjacency_graph
    
    return adjacency_graph

def greedy_seat_selection(group_size, brightness_threshold, table_stats, adjacency_graph, w1=1.0, w2=1.0):
    """
    Greedy algorithm for seat selection with consistent constraint validation.
    
    Args:
        group_size (int): Number of seats needed for the group (N)
        brightness_threshold (float): Minimum average brightness threshold (B_min)
        table_stats (dict): Table statistics with avail_cnt, avg_bright, avg_noise
        adjacency_graph (dict): Adjacency relationships between tables
        w1 (float): Weight for noise in objective function
        w2 (float): Weight for brightness in objective function
    
    Returns:
        list: Selected seats (dictionaries) or empty list if infeasible
    """
    def calculate_score(table_id):
        """Calculate scoring function for table priority."""
        stats = table_stats[table_id]
        # Use same scoring as before: lower is better
        score = w1 * stats['avg_noise'] - w2 * stats['avg_bright']
        return score
    
    # Step 1-9: Find best starting table (removed brightness constraint pre-filtering)
    # Now consider all tables with available seats, constraint will be validated at the end
    valid_tables = [table_id for table_id, stats in table_stats.items() 
                   if stats['avail_cnt'] > 0]  # Only need available seats
    
    if not valid_tables:
        return []
    
    # Pick table with best score
    best_table = min(valid_tables, key=calculate_score)
    
    # Quick check: if one table has enough seats, use it
    if table_stats[best_table]['avail_cnt'] >= group_size:
        available_seats = [seat for seat in table_stats[best_table]['seats'] if seat['Seat_Available']]
        # Sort by noise to get quietest seats
        available_seats.sort(key=lambda x: x['Noise'])
        selected_seats = available_seats[:group_size]
        
        # Validate brightness constraint: sum(brightness) >= brightness_threshold * group_size
        total_brightness = sum(seat['Brightness'] for seat in selected_seats)
        if total_brightness >= brightness_threshold * group_size:
            return selected_seats
        # If constraint fails, continue with multi-table approach
    
    # Step 10-17: Use heap-based expansion for multiple tables
    P = []  # Selected seats
    visited_tables = set()
    
    # Initialize heap with best table
    heap = [(calculate_score(best_table), best_table)]
    
    while len(P) < group_size and heap:
        # Step 14: Pop table with minimum score
        _, current_table = heapq.heappop(heap)
        
        if current_table in visited_tables:
            continue
            
        visited_tables.add(current_table)
        
        # Step 15: Pick quietest seats from current table
        current_stats = table_stats[current_table]
        available_seats = [seat for seat in current_stats['seats'] 
                         if seat['Seat_Available'] and seat['Seat_ID'] not in [s['Seat_ID'] for s in P]]
        
        # Sort by noise and pick needed seats
        available_seats.sort(key=lambda x: x['Noise'])
        seats_needed = min(group_size - len(P), len(available_seats))
        P.extend(available_seats[:seats_needed])
        
        # Step 16: Add neighboring tables to heap
        for neighbor_table in adjacency_graph[current_table]:
            if neighbor_table not in visited_tables and neighbor_table in table_stats:
                neighbor_stats = table_stats[neighbor_table]
                if neighbor_stats['avail_cnt'] > 0:  # Only add tables with available seats
                    heapq.heappush(heap, (calculate_score(neighbor_table), neighbor_table))
    
    # Step 18-22: Validate brightness constraint and return result
    if len(P) == group_size:
        # CRITICAL: Use same constraint as ILP - sum(brightness) >= brightness_threshold * group_size
        total_brightness = sum(seat['Brightness'] for seat in P)
        if total_brightness >= brightness_threshold * group_size:
            return P
        else:
            return []  # Constraint violation
    else:
        return []  # Infeasible size

def pick_quietest_seats(table_stats, table_id, num_seats):
    """
    Helper function to pick the quietest available seats from a table.
    """
    available_seats = [seat for seat in table_stats[table_id]['seats'] if seat['Seat_Available']]
    available_seats.sort(key=lambda x: x['Noise'])
    return available_seats[:num_seats]

def demo_algorithm():
    """
    Demonstrate the greedy seat selection algorithm with multiple groups.
    """
    print("Loading data...")
    seats_df, students_df = load_data()
    
    print(f"Loaded {len(seats_df)} seats and {len(students_df)} students")
    
    successful_placements = 0
    failed_placements = 0
    
    import random
    g = 100
    for group_id in range(1, g+1):
        print(f"\n{'='*50}")
        print(f"PROCESSING GROUP {group_id}")
        print(f"{'='*50}")

        # Get current group
        current_group = students_df[students_df['Group_ID'] == group_id]
        if len(current_group) == 0:
            print(f"No students found in Group {group_id}")
            continue

        group_size = len(current_group)
        brightness_threshold = current_group['Brightness'].min()

        # Randomly choose between Q1 and Q2 for this group
        query_type = random.choice(['Q1', 'Q2'])
        if query_type == 'Q1':
            # Q1: minimize AVG(noise), w1=1.0, w2=0.0
            w1 = 1.0
            w2 = 0.0
            objective_str = "Minimize AVG(noise)"
        else:
            # Q2: minimize AVG(noise) - 0.3*AVG(brightness), w1=1.0, w2=0.3
            w1 = 1.0
            w2 = 0.3
            objective_str = "Minimize (AVG(noise) - 0.3*AVG(brightness))"

        print(f"Group {group_id} requirements:")
        print(f"- Group size: {group_size}")
        print(f"- Brightness threshold: {brightness_threshold}")
        print(f"- Query type: {query_type} ({objective_str})")

        # Load pre-computed data and update with current availability
        _, _, table_stats_df, adjacency_df = load_optimized_data()
        table_stats = create_table_stats_from_precomputed(seats_df, table_stats_df)
        adjacency_graph = create_adjacency_graph_from_precomputed(adjacency_df)

        # Show available seats before placement
        total_available = seats_df['Seat_Available'].sum()
        print(f"- Available seats: {total_available}")

        # Run greedy algorithm
        result = greedy_seat_selection(
            group_size=group_size,
            brightness_threshold=brightness_threshold,
            table_stats=table_stats,
            adjacency_graph=adjacency_graph,
            w1=w1,  # Weight for noise
            w2=w2   # Weight for brightness
        )

        if result:
            successful_placements += 1
            print(f"\n✓ Successfully placed Group {group_id}!")

            # Show seat assignments
            tables_used = set()
            for seat in result:
                tables_used.add(seat['Table_ID'])
                print(f"  - Seat {seat['Seat_ID']} (Table {seat['Table_ID']}, Room {seat['Room_ID']}): "
                      f"Brightness={seat['Brightness']}, Noise={seat['Noise']}")

            print(f"  - Tables used: {sorted(tables_used)}")

            # Calculate group satisfaction
            avg_brightness = np.mean([seat['Brightness'] for seat in result])
            avg_noise = np.mean([seat['Noise'] for seat in result])
            print(f"  - Average brightness: {avg_brightness:.1f}")
            print(f"  - Average noise: {avg_noise:.1f}")

            # UPDATE SEAT AVAILABILITY - Mark assigned seats as unavailable
            for seat in result:
                seat_mask = (seats_df['Seat_ID'] == seat['Seat_ID'])
                seats_df.loc[seat_mask, 'Seat_Available'] = False

            print(f"  - Marked {len(result)} seats as unavailable")

        else:
            failed_placements += 1
            print(f"\n✗ Failed to place Group {group_id}")
            print(f"  - No feasible seat assignment found")

        # Show remaining capacity
        remaining_available = seats_df['Seat_Available'].sum()
        print(f"  - Remaining available seats: {remaining_available}")
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Groups processed: {g}")
    print(f"Successful placements: {successful_placements}")
    print(f"Failed placements: {failed_placements}")
    print(f"Success rate: {successful_placements/(g)*100:.1f}%")
    
    # Show final seat utilization
    total_seats = len(seats_df)
    occupied_seats = total_seats - seats_df['Seat_Available'].sum()
    print(f"Total seats: {total_seats}")
    print(f"Occupied seats: {occupied_seats}")
    print(f"Available seats: {seats_df['Seat_Available'].sum()}")
    print(f"Utilization rate: {occupied_seats/total_seats*100:.1f}%")
    
    # Show table utilization statistics using pre-computed data
    _, _, table_stats_df, _ = load_optimized_data()
    final_table_stats = create_table_stats_from_precomputed(seats_df, table_stats_df)
    occupied_tables = sum(1 for stats in final_table_stats.values() if stats['avail_cnt'] < 10)  # Assuming 10 seats per table
    print(f"Tables with occupied seats: {occupied_tables}/{len(final_table_stats)}")

if __name__ == "__main__":
    demo_algorithm()

import pandas as pd
import numpy as np
from collections import defaultdict
import heapq
import time


def load_data():
    """Load seats and students data from CSV files."""
    seats_df = pd.read_csv("src/seats.csv")
    students_df = pd.read_csv("src/students.csv")
    return seats_df, students_df

def create_table_stats(seats_df):
    """
    Create table statistics: avail_cnt, avg_bright, avg_noise.
    
    Returns:
        table_stats: dict with table_id as key and stats as value
    """
    table_stats = {}
    
    # Group by table to calculate statistics
    for table_id, group in seats_df.groupby('Table_ID'):
        available_seats = group[group['Seat_Available'] == True] # filter where seat availability is true
        avail_cnt = len(available_seats)
        
        if avail_cnt > 0:
            avg_bright = available_seats['Brightness'].mean()
            avg_noise = available_seats['Noise'].mean()
        else:
            avg_bright = 0
            avg_noise = 0
            
        table_stats[table_id] = {
            'avail_cnt': avail_cnt,
            'avg_bright': avg_bright,
            'avg_noise': avg_noise,
            'room_id': group.iloc[0]['Room_ID'], # counts off using room id
            'seats': group.to_dict('records')
        }
    
    return table_stats

def create_adjacency_graph(table_stats):
    """
    Create adjacency graph G where tables in the same room are connected.
    
    Returns:
        adjacency_graph: dict of sets representing table connections
    """
    adjacency_graph = defaultdict(set)
    
    # Group tables by room
    room_tables = defaultdict(list)
    for table_id, stats in table_stats.items():
        room_tables[stats['room_id']].append(table_id)
    
    # Connect all tables within the same room
    for room_id, tables in room_tables.items():
        for i, table1 in enumerate(tables):
            for j, table2 in enumerate(tables):
                if i != j:
                    adjacency_graph[table1].add(table2)
    
    return adjacency_graph

def greedy_seat_selection(group_size, brightness_threshold, table_stats, adjacency_graph, w1=1.0, w2=1.0):
    """
    Greedy Table-First Seat Selection Algorithm
    
    Args:
        group_size (int): Size of the group (m)
        brightness_threshold (int): Minimum brightness threshold (B_min)
        table_stats (dict): Table statistics with avail_cnt, avg_bright, avg_noise
        adjacency_graph (dict): Adjacency graph G
        w1, w2 (float): Weights for scoring function (noise and brightness)
        
    Returns:
        list: Seat set P for the group, or empty list if infeasible
    """
    
    # Step 1: Find candidate tables C
    C = []
    for table_id, stats in table_stats.items():
        if stats['avail_cnt'] >= group_size and stats['avg_bright'] >= brightness_threshold:
            C.append(table_id)
    
    # Step 2: If no candidates, return empty
    if not C:
        return []
    
    # Step 3-6: Score tables and find best one
    def calculate_score(table_id):
        stats = table_stats[table_id]
        # Score based only on noise and brightness (remove distance component)
        score = w1 * stats['avg_noise'] - w2 * stats['avg_bright']
        return score
    
    # Find table with minimum score
    best_table = min(C, key=calculate_score)
    best_stats = table_stats[best_table]
    
    # Step 7-9: If best table has enough seats, pick quietest seats
    if best_stats['avail_cnt'] >= group_size:
        available_seats = [seat for seat in best_stats['seats'] if seat['Seat_Available']]
        # Sort by noise (ascending) to pick quietest seats
        available_seats.sort(key=lambda x: x['Noise'])
        selected_seats = available_seats[:group_size]
        return selected_seats
    
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
    
    # Step 18-22: Return result
    if len(P) == group_size:
        return P
    else:
        return []  # Infeasible

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
    
    g = 100
    # Process first 5 groups
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
        
        print(f"Group {group_id} requirements:")
        print(f"- Group size: {group_size}")
        print(f"- Brightness threshold: {brightness_threshold}")
        
        # Create fresh table statistics with current seat availability
        table_stats = create_table_stats(seats_df)
        adjacency_graph = create_adjacency_graph(table_stats)
        
        # Show available seats before placement
        total_available = seats_df['Seat_Available'].sum()
        print(f"- Available seats: {total_available}")
        
        # Run greedy algorithm
        result = greedy_seat_selection(
            group_size=group_size,
            brightness_threshold=brightness_threshold,
            table_stats=table_stats,
            adjacency_graph=adjacency_graph,
            w1=1.0,  # Weight for noise (higher noise = higher score = worse)
            w2=1.0   # Weight for brightness (higher brightness = lower score = better)
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
    
    # Show table utilization statistics
    final_table_stats = create_table_stats(seats_df)
    occupied_tables = sum(1 for stats in final_table_stats.values() if stats['avail_cnt'] < 10)  # Assuming 10 seats per table
    print(f"Tables with occupied seats: {occupied_tables}/{len(final_table_stats)}")

if __name__ == "__main__":
    demo_algorithm()

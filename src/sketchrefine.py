import pandas as pd
import numpy as np
from collections import defaultdict
import time
from ilp import solve_ilp, solve_ilp_weighted


def load_data():
    """Load seats and students data from CSV files."""
    seats_df = pd.read_csv("src/seats.csv")
    students_df = pd.read_csv("src/students.csv")
    return seats_df, students_df


def create_meta_tables(seats_df):
    """
    Create meta tables containing average attributes for each table.
    
    Returns:
        meta_tables_df: DataFrame with table-level aggregated statistics
        table_to_seats: dict mapping table_id to list of seat indices
    """
    meta_tables = []
    table_to_seats = {}
    
    # Group by table to calculate meta-table statistics
    for table_id, group in seats_df.groupby('Table_ID'):
        available_seats = group[group['Seat_Available'] == True]
        
        if len(available_seats) > 0:
            # Calculate meta-table attributes
            avg_brightness = available_seats['Brightness'].mean()
            avg_noise = available_seats['Noise'].mean()
            room_id = group.iloc[0]['Room_ID']
            total_seats = len(available_seats)
            
            meta_tables.append({
                'Table_ID': table_id,
                'Room_ID': room_id,
                'Avg_Brightness': avg_brightness,
                'Avg_Noise': avg_noise,
                'Available_Seats': total_seats
            })
            
            # Store mapping from table_id to seat indices
            table_to_seats[table_id] = available_seats.index.tolist()
    
    meta_tables_df = pd.DataFrame(meta_tables)
    return meta_tables_df, table_to_seats


def phase1_meta_table_query(meta_tables_df, group_size, brightness_threshold, query_type):
    """
    Phase 1: Package query on meta-tables to find suitable tables.
    
    Args:
        meta_tables_df: DataFrame with meta-table statistics
        group_size: Size of the group to place
        brightness_threshold: Minimum brightness requirement
        query_type: 'Q1' or 'Q2' for different objectives
        
    Returns:
        selected_tables: List of table IDs that passed the meta-table query
    """
    if len(meta_tables_df) == 0:
        return []
    
    # Get actual table capacities (number of available seats per table)
    table_capacities = meta_tables_df['Available_Seats'].values
    
    # Check if we have enough total capacity across all tables
    total_available_seats = sum(table_capacities)
    if total_available_seats < group_size:
        return []  # Not enough seats available across all tables
    
    # Use the maximum table capacity as the rep value for ILP
    max_capacity = max(table_capacities)
    
    # Prepare data matrix for ILP: [Avg_Brightness, Avg_Noise, Room_ID, Table_ID]
    data_matrix = meta_tables_df[['Avg_Brightness', 'Avg_Noise', 'Room_ID', 'Table_ID']].values
    
    # Set up constraints and objectives based on query type
    if query_type == 'Q1':
        # Q1: minimize AVG(noise)
        picked, counts = solve_ilp(
            sub=data_matrix,
            size_=group_size,  # We want to select exactly group_size seats
            rep=max_capacity,  # Use maximum capacity as repetition limit
            obj={'attr': 1, 'pref': 'MIN'},  # Minimize noise
            cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],  # Brightness constraint
            verbose=False
        )
    else:
        # Q2: minimize AVG(noise) - 0.3*AVG(brightness)
        picked, counts = solve_ilp_weighted(
            sub=data_matrix,
            size_=group_size,
            rep=max_capacity,  # Use maximum capacity as repetition limit
            weights=[-0.3, 1.0, 0, 0],  # Negative weight for brightness (maximize), positive for noise (minimize)
            cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
            verbose=False
        )
    
    if picked:
        # Extract selected table IDs with their selection counts
        selected_table_ids = []
        for i in picked:
            table_id = int(data_matrix[i][3])  # Table_ID is at index 3
            # Get the actual selection count, but cap it at the table's capacity
            selection_count = min(counts[i] if i < len(counts) else 1, table_capacities[i])
            selected_table_ids.extend([table_id] * selection_count)
        
        # Ensure we don't exceed the group size
        if len(selected_table_ids) > group_size:
            selected_table_ids = selected_table_ids[:group_size]
        
        return selected_table_ids
    else:
        return []


def phase2_seat_query(seats_df, table_to_seats, selected_tables, group_size, brightness_threshold, query_type):
    """
    Phase 2: Package query on actual seats from selected tables.
    
    Args:
        seats_df: DataFrame with all seats
        table_to_seats: dict mapping table_id to seat indices
        selected_tables: List of table IDs from phase 1
        group_size: Size of the group to place
        brightness_threshold: Minimum brightness requirement
        query_type: 'Q1' or 'Q2' for different objectives
        
    Returns:
        selected_seats: List of seat dictionaries or None if no solution
    """
    if not selected_tables:
        return None
    
    # Collect all available seats from selected tables
    available_seat_indices = []
    for table_id in selected_tables:
        if table_id in table_to_seats:
            available_seat_indices.extend(table_to_seats[table_id])
    
    if len(available_seat_indices) < group_size:
        return None
    
    # Get the actual seat data
    available_seats = seats_df.iloc[available_seat_indices]
    
    # Prepare data matrix for ILP: [Brightness, Noise, Room_ID, Table_ID]
    data_matrix = available_seats[['Brightness', 'Noise', 'Room_ID', 'Table_ID']].values
    
    # Set up constraints and objectives based on query type
    if query_type == 'Q1':
        # Q1: minimize AVG(noise)
        picked, counts = solve_ilp(
            sub=data_matrix,
            size_=group_size,
            rep=1,  # Each seat can only be selected once
            obj={'attr': 1, 'pref': 'MIN'},  # Minimize noise
            cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],  # Brightness constraint
            verbose=False
        )
    else:
        # Q2: minimize AVG(noise) - 0.3*AVG(brightness)
        picked, counts = solve_ilp_weighted(
            sub=data_matrix,
            size_=group_size,
            rep=1,
            weights=[-0.3, 1.0, 0, 0],  # Negative weight for brightness (maximize), positive for noise (minimize)
            cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
            verbose=False
        )
    
    if picked:
        # Extract selected seats
        selected_seats = available_seats.iloc[picked].to_dict('records')
        return selected_seats
    else:
        return None


def sketchrefine_seat_selection(group_size, brightness_threshold, seats_df, query_type):
    """
    SketchRefine Algorithm: Two-phase package query approach.
    
    Args:
        group_size (int): Size of the group (m)
        brightness_threshold (int): Minimum brightness threshold (B_min)
        seats_df (DataFrame): Seats data
        query_type (str): 'Q1' or 'Q2' for different objectives
        
    Returns:
        list: Seat set P for the group, or None if infeasible
    """
    start_time = time.time()
    
    # Phase 1: Create meta-tables and run package query
    meta_tables_df, table_to_seats = create_meta_tables(seats_df)
    
    if len(meta_tables_df) == 0:
        return None
    
    # Phase 1: Package query on meta-tables
    selected_tables = phase1_meta_table_query(
        meta_tables_df, 
        group_size, 
        brightness_threshold, 
        query_type
    )
    
    if not selected_tables:
        return None
    
    # Phase 2: Package query on actual seats from selected tables
    selected_seats = phase2_seat_query(
        seats_df,
        table_to_seats,
        selected_tables,
        group_size,
        brightness_threshold,
        query_type
    )
    
    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return {
        'seats': selected_seats,
        'execution_time_ms': execution_time,
        'selected_tables': selected_tables,
        'success': selected_seats is not None
    }


def demo_sketchrefine():
    """
    Demonstrate the SketchRefine algorithm with multiple groups.
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
            objective_str = "Minimize AVG(noise)"
        else:
            objective_str = "Minimize (AVG(noise) - 0.3*AVG(brightness))"

        print(f"Group {group_id} requirements:")
        print(f"- Group size: {group_size}")
        print(f"- Brightness threshold: {brightness_threshold}")
        print(f"- Query type: {query_type} ({objective_str})")

        # Show available seats before placement
        total_available = seats_df['Seat_Available'].sum()
        print(f"- Available seats: {total_available}")

        # Run SketchRefine algorithm
        result = sketchrefine_seat_selection(
            group_size=group_size,
            brightness_threshold=brightness_threshold,
            seats_df=seats_df,
            query_type=query_type
        )

        if result and result['success']:
            successful_placements += 1
            print(f"\n✓ Successfully placed Group {group_id}!")
            print(f"  - Execution time: {result['execution_time_ms']:.2f}ms")
            print(f"  - Tables selected in Phase 1: {result['selected_tables']}")

            # Show seat assignments
            tables_used = set()
            for seat in result['seats']:
                tables_used.add(seat['Table_ID'])
                print(f"  - Seat {seat['Seat_ID']} (Table {seat['Table_ID']}, Room {seat['Room_ID']}): "
                      f"Brightness={seat['Brightness']}, Noise={seat['Noise']}")

            print(f"  - Tables used: {sorted(tables_used)}")

            # Calculate group satisfaction
            avg_brightness = np.mean([seat['Brightness'] for seat in result['seats']])
            avg_noise = np.mean([seat['Noise'] for seat in result['seats']])
            print(f"  - Average brightness: {avg_brightness:.1f}")
            print(f"  - Average noise: {avg_noise:.1f}")

            # UPDATE SEAT AVAILABILITY - Mark assigned seats as unavailable
            for seat in result['seats']:
                seat_mask = (seats_df['Seat_ID'] == seat['Seat_ID'])
                seats_df.loc[seat_mask, 'Seat_Available'] = False

            print(f"  - Marked {len(result['seats'])} seats as unavailable")

        else:
            failed_placements += 1
            print(f"\n✗ Failed to place Group {group_id}")
            if result:
                print(f"  - Execution time: {result['execution_time_ms']:.2f}ms")
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


if __name__ == "__main__":
    demo_sketchrefine()

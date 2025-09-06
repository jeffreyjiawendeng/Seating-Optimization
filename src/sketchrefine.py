import pandas as pd
import numpy as np
from collections import defaultdict
import time
from ilp import solve_ilp, solve_ilp_weighted


def load_data():
    """Load seats and students data from CSV files."""
    seats_df = pd.read_csv("seats.csv")
    students_df = pd.read_csv("students.csv")
    return seats_df, students_df


def create_partitions(seats_df, max_partition_size=50):
    """
    Create partitions (tables) that serve as groups for SketchRefine.
    Each partition contains seats from the same table, with average attributes as representatives.
    
    Args:
        seats_df: DataFrame with all seats
        max_partition_size: Maximum size for each partition
        
    Returns:
        partitions: List of partition dictionaries
        partition_to_seats: dict mapping partition_id to list of seat indices
    """
    partitions = []
    partition_to_seats = {}
    partition_id = 0
    
    # Group by table to create partitions
    for table_id, group in seats_df.groupby('Table_ID'):
        available_seats = group[group['Seat_Available'] == True]
        
        if len(available_seats) > 0:
            # Calculate representative attributes for this partition
            avg_brightness = available_seats['Brightness'].mean()
            avg_noise = available_seats['Noise'].mean()
            room_id = group.iloc[0]['Room_ID']
            total_seats = len(available_seats)
            
            partitions.append({
                'partition_id': partition_id,
                'table_id': table_id,
                'room_id': room_id,
                'avg_brightness': avg_brightness,
                'avg_noise': avg_noise,
                'available_seats': total_seats,
                'refined': False  # Track if this partition has been refined
            })
            
            # Store mapping from partition_id to seat indices
            partition_to_seats[partition_id] = available_seats.index.tolist()
            partition_id += 1
    
    return partitions, partition_to_seats


def initial_sketch(partitions, group_size, brightness_threshold, query_type):
    """
    Initial SKETCH: Run package query on representative tuples (average table attributes).
    This gives us an initial approximate solution.
    
    Args:
        partitions: List of partition dictionaries
        group_size: Size of the group to place
        brightness_threshold: Minimum brightness requirement
        query_type: 'Q1' or 'Q2' for different objectives
        
    Returns:
        selected_partitions: List of (partition_id, count) tuples
    """
    if len(partitions) == 0:
        return []
    
    # Get partition capacities
    partition_capacities = [p['available_seats'] for p in partitions]
    
    # Check if we have enough total capacity
    total_available_seats = sum(partition_capacities)
    if total_available_seats < group_size:
        return []
    
    # Use maximum capacity as rep value for ILP
    max_capacity = max(partition_capacities)
    
    # Prepare data matrix: [avg_brightness, avg_noise, room_id, partition_id]
    data_matrix = np.array([
        [p['avg_brightness'], p['avg_noise'], p['room_id'], p['partition_id']]
        for p in partitions
    ])
    
    # Set up constraints and objectives
    if query_type == 'Q1':
        picked, counts = solve_ilp(
            sub=data_matrix,
            size_=group_size,
            rep=max_capacity,
            obj={'attr': 1, 'pref': 'MIN'},  # Minimize noise
            cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
            verbose=False
        )
    else:
        picked, counts = solve_ilp_weighted(
            sub=data_matrix,
            size_=group_size,
            rep=max_capacity,
            weights=[-0.3, 1.0, 0, 0],  # Negative weight for brightness (maximize), positive for noise (minimize)
            cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
            verbose=False
        )
    
    if picked:
        # Extract selected partitions with their counts
        selected_partitions = []
        for i in picked:
            partition_id = int(data_matrix[i][3])
            count = min(counts[i] if i < len(counts) else 1, partition_capacities[i])
            selected_partitions.append((partition_id, count))
        
        # Ensure we don't exceed group size
        total_selected = sum(count for _, count in selected_partitions)
        if total_selected > group_size:
            # Trim excess selections
            excess = total_selected - group_size
            for i in range(len(selected_partitions) - 1, -1, -1):
                if excess <= 0:
                    break
                partition_id, count = selected_partitions[i]
                reduction = min(excess, count)
                selected_partitions[i] = (partition_id, count - reduction)
                excess -= reduction
        
        return selected_partitions
    else:
        return []


def refine_partition(seats_df, partition_to_seats, partitions, selected_partitions, 
                    partition_id_to_refine, group_size, brightness_threshold, query_type):
    """
    Refine a specific partition by replacing its representative with actual seats.
    
    Args:
        seats_df: DataFrame with all seats
        partition_to_seats: dict mapping partition_id to seat indices
        partitions: List of partition dictionaries
        selected_partitions: Current selection of (partition_id, count) tuples
        partition_id_to_refine: ID of partition to refine
        group_size: Size of the group to place
        brightness_threshold: Minimum brightness requirement
        query_type: 'Q1' or 'Q2' for different objectives
        
    Returns:
        refined_selection: Updated selection with actual seats from refined partition
    """
    # Find the partition to refine
    partition_to_refine = None
    for p in partitions:
        if p['partition_id'] == partition_id_to_refine:
            partition_to_refine = p
            break
    
    if not partition_to_refine:
        return selected_partitions
    
    # Get seats from the partition to refine
    seat_indices = partition_to_seats[partition_id_to_refine]
    partition_seats = seats_df.iloc[seat_indices]
    
    # Create a mixed dataset: actual seats from partition_to_refine + representatives from others
    mixed_seats = []
    seat_to_partition_mapping = {}
    
    # Add actual seats from the partition being refined
    for idx, seat in partition_seats.iterrows():
        mixed_seats.append({
            'Brightness': seat['Brightness'],
            'Noise': seat['Noise'],
            'Room_ID': seat['Room_ID'],
            'Table_ID': seat['Table_ID'],
            'Seat_ID': seat['Seat_ID'],
            'is_representative': False,
            'partition_id': partition_id_to_refine
        })
        seat_to_partition_mapping[len(mixed_seats) - 1] = partition_id_to_refine
    
    # Add representative tuples from other partitions
    for partition_id, count in selected_partitions:
        if partition_id != partition_id_to_refine:
            partition = next(p for p in partitions if p['partition_id'] == partition_id)
            # Add representative tuple 'count' times
            for _ in range(count):
                mixed_seats.append({
                    'Brightness': partition['avg_brightness'],
                    'Noise': partition['avg_noise'],
                    'Room_ID': partition['room_id'],
                    'Table_ID': partition['table_id'],
                    'Seat_ID': f"REP_{partition_id}",
                    'is_representative': True,
                    'partition_id': partition_id
                })
                seat_to_partition_mapping[len(mixed_seats) - 1] = partition_id
    
    # Convert to DataFrame for ILP
    mixed_seats_df = pd.DataFrame(mixed_seats)
    data_matrix = mixed_seats_df[['Brightness', 'Noise', 'Room_ID', 'Table_ID']].values
    
    # Run ILP on mixed dataset
    if query_type == 'Q1':
        picked, counts = solve_ilp(
            sub=data_matrix,
            size_=group_size,
            rep=1,
            obj={'attr': 1, 'pref': 'MIN'},
            cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
            verbose=False
        )
    else:
        picked, counts = solve_ilp_weighted(
            sub=data_matrix,
            size_=group_size,
            rep=1,
            weights=[-0.3, 1.0, 0, 0],
            cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
            verbose=False
        )
    
    if picked:
        # Convert back to partition-based selection
        refined_selection = defaultdict(int)
        
        for i in picked:
            partition_id = seat_to_partition_mapping[i]
            refined_selection[partition_id] += 1
        
        # Convert to list format
        return [(pid, count) for pid, count in refined_selection.items()]
    else:
        return selected_partitions


def sequential_refinement(seats_df, partition_to_seats, partitions, initial_selection, 
                         group_size, brightness_threshold, query_type):
    """
    Sequentially refine each partition one by one.
    
    Args:
        seats_df: DataFrame with all seats
        partition_to_seats: dict mapping partition_id to seat indices
        partitions: List of partition dictionaries
        initial_selection: Initial selection from sketch stage
        group_size: Size of the group to place
        brightness_threshold: Minimum brightness requirement
        query_type: 'Q1' or 'Q2' for different objectives
        
    Returns:
        final_selection: Final refined selection
    """
    current_selection = initial_selection.copy()
    
    # Refine each partition sequentially
    for partition_id, count in initial_selection:
        if count > 0:  # Only refine partitions that were selected
            print(f"    Refining partition {partition_id} (Table {partitions[partition_id]['table_id']})...")
            
            # Refine this partition
            refined_selection = refine_partition(
                seats_df, partition_to_seats, partitions, current_selection,
                partition_id, group_size, brightness_threshold, query_type
            )
            
            # Update current selection
            current_selection = refined_selection
            
            # Mark partition as refined
            for p in partitions:
                if p['partition_id'] == partition_id:
                    p['refined'] = True
                    break
    
    return current_selection


def convert_to_actual_seats(seats_df, partition_to_seats, partitions, final_selection, 
                           group_size, brightness_threshold, query_type):
    """
    Convert final partition-based selection to actual seat assignments.
    
    Args:
        seats_df: DataFrame with all seats
        partition_to_seats: dict mapping partition_id to seat indices
        partitions: List of partition dictionaries
        final_selection: Final selection of (partition_id, count) tuples
        group_size: Size of the group to place
        brightness_threshold: Minimum brightness requirement
        query_type: 'Q1' or 'Q2' for different objectives
        
    Returns:
        selected_seats: List of actual seat dictionaries
    """
    # Collect all available seats from selected partitions
    available_seats_list = []
    seat_to_partition_mapping = {}
    
    for partition_id, count_needed in final_selection:
        if partition_id in partition_to_seats:
            seat_indices = partition_to_seats[partition_id]
            table_seats = seats_df.iloc[seat_indices]
            
            for idx, seat in table_seats.iterrows():
                available_seats_list.append(seat)
                seat_to_partition_mapping[len(available_seats_list) - 1] = partition_id
    
    if len(available_seats_list) < group_size:
        return None
    
    # Convert to DataFrame
    available_seats_df = pd.DataFrame(available_seats_list)
    data_matrix = available_seats_df[['Brightness', 'Noise', 'Room_ID', 'Table_ID']].values
    
    # Run final ILP to select actual seats
    if query_type == 'Q1':
        picked, counts = solve_ilp(
            sub=data_matrix,
            size_=group_size,
            rep=1,
            obj={'attr': 1, 'pref': 'MIN'},
            cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
            verbose=False
        )
    else:
        picked, counts = solve_ilp_weighted(
            sub=data_matrix,
            size_=group_size,
            rep=1,
            weights=[-0.3, 1.0, 0, 0],
            cons=[{'attr': 0, 'pref': 'MAX', 'bound': brightness_threshold * group_size}],
            verbose=False
        )
    
    if picked:
        selected_seats = available_seats_df.iloc[picked].to_dict('records')
        return selected_seats
    else:
        return None


def sketchrefine_seat_selection(group_size, brightness_threshold, seats_df, query_type):
    """
    SketchRefine Algorithm: Sequential refinement approach.
    
    1. Create partitions (tables) with representative tuples
    2. Initial SKETCH: Run query on representatives
    3. Sequential REFINE: Refine each partition one by one
    4. Final conversion: Convert to actual seat assignments
    
    Args:
        group_size (int): Size of the group (m)
        brightness_threshold (int): Minimum brightness threshold (B_min)
        seats_df (DataFrame): Seats data
        query_type (str): 'Q1' or 'Q2' for different objectives
        
    Returns:
        dict: Result containing seats, execution time, and success status
    """
    start_time = time.time()
    
    # Step 1: Create partitions with representative tuples
    partitions, partition_to_seats = create_partitions(seats_df)
    
    if len(partitions) == 0:
        return {
            'seats': None,
            'execution_time_ms': (time.time() - start_time) * 1000,
            'selected_tables': [],
            'success': False,
            'refinement_steps': 0
        }
    
    # Step 2: Initial SKETCH - run query on representatives
    print(f"    Initial SKETCH: Running query on {len(partitions)} representative tuples...")
    initial_selection = initial_sketch(partitions, group_size, brightness_threshold, query_type)
    
    if not initial_selection:
        return {
            'seats': None,
            'execution_time_ms': (time.time() - start_time) * 1000,
            'selected_tables': [],
            'success': False,
            'refinement_steps': 0
        }
    
    print(f"    Initial selection: {initial_selection}")
    
    # Step 3: Sequential REFINE - refine each partition one by one
    print(f"    Sequential REFINE: Refining {len(initial_selection)} partitions...")
    final_selection = sequential_refinement(
        seats_df, partition_to_seats, partitions, initial_selection,
        group_size, brightness_threshold, query_type
    )
    
    print(f"    Final selection: {final_selection}")
    
    # Step 4: Convert to actual seat assignments
    print(f"    Converting to actual seat assignments...")
    selected_seats = convert_to_actual_seats(
        seats_df, partition_to_seats, partitions, final_selection,
        group_size, brightness_threshold, query_type
    )
    
    execution_time = (time.time() - start_time) * 1000
    
    # Extract table IDs for compatibility
    selected_tables = []
    for partition_id, count in final_selection:
        table_id = next(p['table_id'] for p in partitions if p['partition_id'] == partition_id)
        selected_tables.extend([table_id] * count)
    
    return {
        'seats': selected_seats,
        'execution_time_ms': execution_time,
        'selected_tables': selected_tables,
        'success': selected_seats is not None,
        'refinement_steps': len(initial_selection)
    }


def demo_sketchrefine():
    """
    Demonstrate the SketchRefine algorithm with sequential refinement.
    """
    print("Loading data...")
    seats_df, students_df = load_data()
    
    print(f"Loaded {len(seats_df)} seats and {len(students_df)} students")
    
    successful_placements = 0
    failed_placements = 0
    
    import random
    g = 10  # Test with more groups to see refinement process
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

        # Run SketchRefine algorithm with sequential refinement
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
            print(f"  - Refinement steps: {result['refinement_steps']}")
            print(f"  - Tables selected: {result['selected_tables']}")

            # Show seat assignments
            tables_used = set()
            seat_ids_used = set()
            for seat in result['seats']:
                tables_used.add(seat['Table_ID'])
                seat_ids_used.add(seat['Seat_ID'])
                print(f"  - Seat {seat['Seat_ID']} (Table {seat['Table_ID']}, Room {seat['Room_ID']}): "
                      f"Brightness={seat['Brightness']}, Noise={seat['Noise']}")

            print(f"  - Tables used: {sorted(tables_used)}")
            print(f"  - Unique seats used: {len(seat_ids_used)}")

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
                print(f"  - Refinement steps attempted: {result['refinement_steps']}")
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

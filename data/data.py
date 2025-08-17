import numpy as np
import pandas as pd
from collections import defaultdict

def mirror_weights(length):
    """
    Create an array of weights that increases then decreases linearly at the midpoint of an array.
    
    Input: 
        length: number representing the length of the array.
       
    Output: 
        weights: List representing weights
     
    Ex:
        length=6 => [1, 2, 3, 3, 2, 1]
        length=7 => [1, 2, 3, 4, 3, 2, 1]
    """
    midpoint = length // 2
    
    if length % 2 == 0:
        # length is even: midpoint is the leftmost of the two middle points
        increasing = list(range(1, midpoint + 1)) # 1, 2, ..., midpoint
        decreasing = list(range(midpoint, 0, -1)) # midpoint, ..., 2, 1
    else:
        # length is odd: midpoint is the exact midpoint
        increasing = list(range(1, midpoint + 2)) # 1, 2, ..., midpoint
        decreasing = list(range(midpoint, 0, -1)) # midpoint - 1, ..., 1
    
    weights = increasing + decreasing
    
    return weights

def generate_layout(shape=(3, 2, 5, 2, 5)):
    """
    Create a layout tensor of ones with the given 5D shape.
    
    Input:
        shape: 5D shape representing the number of rooms, the shape of each room, and the shape of each table.
        
    Output:
        1's tensor of shape shape.
    """
    return np.ones(shape, dtype=int)

def generate_attribute_layout(layout, brightness_range=(80, 20), brightness_sd=3, noise_sd=10):
    """
    Returns a tensor of shape (rooms, tr, tc, sr, sc, 3) with attributes [brightness, noise, whiteboards] for each seat.

    Brightness: total_rows = table_rows * seat_rows => linear ramp.
    Noise: Improved distribution with better variance across the full range 1-100
    
    Input:
        layout: 5D tensor describing the layout of the library.
        brightness_range: range of integers for the brightness.
        brightness_sd: standard deviation for the brightness level, int.
        noise_sd: standard deviation for the noise level, int.
        
    Ouutput:
        attributes: 6D tensor of seats and their attributes.
    """ 
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Unpack and set dimension variables
    rooms, tr, tc, sr, sc = layout.shape
    total_rows = tr * sr
    total_cols = tc * sc

    # brightness base for each row 
    brightness_bases = np.linspace(brightness_range[0], brightness_range[1], total_rows)

    # Initialize zeros attribute tensor 
    attributes = np.zeros((rooms, tr, tc, sr, sc, 2), dtype=int)
    
    # Initialize attributes for each seat
    for room in range(rooms):
        for tr_i in range(tr):
            for tc_i in range(tc):
                for sr_i in range(sr):
                    for sc_i in range(sc):
                        # brightness with small noise
                        brightness = int(np.clip(np.random.normal(brightness_bases[tr_i * sr + sr_i], brightness_sd), 1, 100))
                        
                        # Improved noise distribution with better variance:
                        # 15% excellent (1-20), 25% good (21-40), 35% average (41-70), 20% poor (71-90), 5% terrible (91-100)
                        noise_category = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.25, 0.35, 0.20, 0.05])
                        if noise_category == 0:
                            noise = int(np.random.uniform(1, 21))    # Excellent
                        elif noise_category == 1:
                            noise = int(np.random.uniform(21, 41))   # Good  
                        elif noise_category == 2:
                            noise = int(np.random.uniform(41, 71))   # Average
                        elif noise_category == 3:
                            noise = int(np.random.uniform(71, 91))   # Poor
                        else:
                            noise = int(np.random.uniform(91, 101))  # Terrible
                        
                        # set attributes
                        attributes[room, tr_i, tc_i, sr_i, sc_i] = (brightness, noise)
                        
    # Return row of attributes 
    return attributes

def display_attribute_layout(attributes):
    """
    Prints each room's tables with seat attributes: (brightness, noise) for each seat.
    
    Input: 
        Attributes: 5D 1's tensor.
    """
    rooms, tr, tc, sr, sc, _ = attributes.shape

    for room in range(rooms):
        print(f"Room {room+1}")
        for tr_i in range(tr):
            for sr_i in range(sr):
                row_sections = []
                for tc_i in range(tc):
                    seats = []
                    for sc_i in range(sc):
                        b, n = attributes[room, tr_i, tc_i, sr_i, sc_i]
                        seats.append(f"({b:3d},{n:3d})")
                    row_sections.append(' '.join(seats))
                print(' | '.join(row_sections))
            if tr_i < tr - 1:
                print('-' * (12 * tc * sc + tc * (sc - 1) + (tc - 1) * 3))
        print()

def attributes_to_csv(seats, path="data/seats.csv"):
    """
    Turn the attrs tensor into a flat DataFrame with columns:
        Seat_ID, Table_ID, Room_ID, Brightness, Whiteboards, Noise, Seat_Available, Table_Available, Room_Available
    and save to CSV.
    
    Input:
        seats: 6D tensor of seats and their attributes (Brightness, Whiteboards, and Noise).
        path: CSV file path.
        
    Output: 
        df: dataframe containing the seats table.
    """
    rooms, tr, tc, sr, sc, _ = seats.shape
    rows = []
    seat_id = 1
    table_counter = 1

    for room in range(rooms):
        room_id = room + 1
        for tr_i in range(tr):
            for tc_i in range(tc):
                table_id = table_counter
                table_counter += 1
                for sr_i in range(sr):
                    for sc_i in range(sc):
                        b, n = seats[room, tr_i, tc_i, sr_i, sc_i]
                        rows.append({
                            "Seat_ID":         seat_id,
                            "Table_ID":        table_id,
                            "Room_ID":         room_id,
                            "Brightness":      b,
                            "Noise":           n,
                            "Seat_Available":  True,
                            "Table_Available": True,
                            "Room_Available":  True
                        })
                        seat_id += 1

    df = pd.DataFrame(rows, columns=[
        "Seat_ID", 
        "Table_ID", 
        "Room_ID", 
        "Brightness", 
        "Noise", 
        "Seat_Available", 
        "Table_Available", 
        "Room_Available"
    ])
    
    df.to_csv(path, index=False)    
    
    return df

def generate_students(num_groups,
                      group_size_range=(1, 10),
                      brightness_mean=50,
                      brightness_sd=15,
                      noise_mean=50,
                      noise_sd=15,
                      flexibility_range=(1, 5),
                      ):
    """
    Generates a students DataFrame with columns:
        Student_ID, Group_ID, Brightness, Noise, Flexibility
    
    Attributes:
        Brightness: Normal(mean, sd), clipped to [0,100], int.
        Noise: Normal(mean, sd), clipped to [0,100], int. 
        Whiteboards: random integer in whiteboards_range per student.
        Flexibility: random integer in flexibility_range per student.

    Input:
        num_groups: number of groups
        group_size_range: tuple (min_size, max_size) for each group's size, uniformly distributed.

    Output: 
        df: dataframe containing the students table.
    """
    students = []
    student_id = 1
    sizes = np.random.randint(group_size_range[0],
                              group_size_range[1] + 1,
                              size=num_groups)

    for group_id, group_size in enumerate(sizes, start=1):
        for _ in range(group_size):
            brightness = int(np.clip(np.random.normal(brightness_mean, brightness_sd), 1, 100))
            noise = int(np.clip(np.random.normal(noise_mean, noise_sd), 1, 100))
            flex = np.random.randint(flexibility_range[0], flexibility_range[1] + 1)
            students.append({
                "Student_ID":   student_id,  # Use numeric ID instead of S001 format
                "Group_ID":     group_id,
                "Brightness":   brightness,
                "Noise":        noise,
                "Flexibility":  flex
            })
            student_id += 1

    df = pd.DataFrame(students, columns=[
        "Student_ID", 
        "Group_ID",
        "Brightness", 
        "Noise",
        "Flexibility"
    ])
    
    return df

def students_to_csv(students, path="data/students.csv"):
    """
    Save the students DataFrame to CSV with columns:
        Student_ID, Group_ID, Brightness, Noise, Whiteboards, Flexibility
    """
    cols = [
        "Student_ID", 
        "Group_ID",
        "Brightness", 
        "Noise",
        "Flexibility"
    ]
    students.to_csv(path, columns=cols, index=False)
    print(f"Wrote {len(students)} students to {path}")
    
def generate_table_stats_csv(seats_df_path="data/seats.csv", output_path="data/table_stats.csv"):
    """
    Generate pre-computed table statistics to optimize greedy algorithm performance.
    
    Args:
        seats_df_path: Path to seats CSV file
        output_path: Path to save table statistics CSV
    
    Returns:
        DataFrame with table statistics
    """
    seats_df = pd.read_csv(seats_df_path)
    
    table_stats_list = []
    
    # Group by table to calculate statistics
    for table_id, group in seats_df.groupby('Table_ID'):
        total_seats = len(group)
        available_seats = group[group['Seat_Available'] == True]
        avail_cnt = len(available_seats)
        
        if avail_cnt > 0:
            avg_bright = available_seats['Brightness'].mean()
            avg_noise = available_seats['Noise'].mean()
            min_brightness = available_seats['Brightness'].min()
            max_brightness = available_seats['Brightness'].max()
            min_noise = available_seats['Noise'].min()
            max_noise = available_seats['Noise'].max()
        else:
            avg_bright = avg_noise = 0
            min_brightness = max_brightness = 0
            min_noise = max_noise = 0
            
        table_stats_list.append({
            'Table_ID': table_id,
            'Room_ID': group.iloc[0]['Room_ID'],
            'Total_Seats': total_seats,
            'Available_Seats': avail_cnt,
            'Avg_Brightness': avg_bright,
            'Avg_Noise': avg_noise,
            'Min_Brightness': min_brightness,
            'Max_Brightness': max_brightness,
            'Min_Noise': min_noise,
            'Max_Noise': max_noise
        })
    
    table_stats_df = pd.DataFrame(table_stats_list)
    table_stats_df.to_csv(output_path, index=False)
    print(f"Generated table statistics: {len(table_stats_df)} tables")
    return table_stats_df

def generate_adjacency_graph_csv(table_stats_df_path="data/table_stats.csv", output_path="data/adjacency_graph.csv"):
    """
    Generate pre-computed adjacency graph for tables (tables in same room are adjacent).
    Uses efficient representation - only stores unique pairs once.
    
    Args:
        table_stats_df_path: Path to table statistics CSV
        output_path: Path to save adjacency graph CSV
    
    Returns:
        DataFrame with adjacency relationships
    """
    table_stats_df = pd.read_csv(table_stats_df_path)
    
    adjacency_list = []
    
    # Group tables by room
    room_tables = defaultdict(list)
    for _, row in table_stats_df.iterrows():
        room_tables[row['Room_ID']].append(row['Table_ID'])
    
    # Connect all tables within the same room (only store each pair once)
    for room_id, tables in room_tables.items():
        for i, table1 in enumerate(tables):
            for j, table2 in enumerate(tables):
                if i < j:  # Only store each pair once (i < j avoids duplicates and self-loops)
                    adjacency_list.append({
                        'Table_ID': table1,
                        'Adjacent_Table_ID': table2,
                        'Room_ID': room_id
                    })
    
    adjacency_df = pd.DataFrame(adjacency_list)
    adjacency_df.to_csv(output_path, index=False)
    print(f"Generated adjacency graph: {len(adjacency_df)} connections")
    return adjacency_df

def generate_all_data():
    """
    Generate all data files including optimized pre-computed statistics.
    This ensures both algorithms work with the same optimized data structures.
    """
    print("Generating seats and students data...")
    
    # Generate exactly 2000 seats: (5, 8, 5, 2, 5) = 5*8*5*2*5 = 2000
    layout = generate_layout((5, 8, 5, 2, 5))
    attrs = generate_attribute_layout(
        layout,
        brightness_range=(80, 20),
        brightness_sd=3,
        noise_sd=10
    )
    print(f"Generated layout with {layout.size} seats")
    
    # Generate seats CSV
    seats_df = attributes_to_csv(attrs, path="data/seats.csv")
    print(f"Generated {len(seats_df)} seats")
    
    # Generate students with exactly 2000 students
    df_students = generate_students(
        num_groups=120,
        group_size_range=(8, 12),
        brightness_mean=60,
        brightness_sd=20,
        noise_mean=40,
        noise_sd=10,
        flexibility_range=(1, 5)
    )
    
    # If we need more students to reach 2000, generate additional
    if len(df_students) < 2000:
        remaining = 2000 - len(df_students)
        additional_groups = max(1, remaining // 10)  # Aim for ~10 students per additional group
        
        additional_students = generate_students(
            num_groups=additional_groups,
            group_size_range=(8, 12),
            brightness_mean=60,
            brightness_sd=20,
            noise_mean=40,
            noise_sd=10,
            flexibility_range=(1,5)
        )
        # Adjust Group_IDs to continue from existing ones
        max_group_id = df_students['Group_ID'].max()
        additional_students['Group_ID'] += max_group_id
        
        # Adjust Student_IDs to continue from existing ones
        max_student_id = df_students['Student_ID'].max()
        additional_students['Student_ID'] += max_student_id
        
        # Combine and take exactly 2000
        df_students = pd.concat([df_students, additional_students], ignore_index=True)
        df_students = df_students.iloc[:2000].copy()
    
    print(f"Generated {len(df_students)} students in {df_students['Group_ID'].nunique()} groups")
    students_to_csv(df_students, path="data/students.csv")
    print("Data generation complete!")
    
    print("Generating table statistics...")
    generate_table_stats_csv()
    
    print("Generating adjacency graph...")
    generate_adjacency_graph_csv()
    
    print("All optimized data generation complete!")

if __name__ == "__main__":
    generate_all_data()

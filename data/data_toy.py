import numpy as np
import pandas as pd
import math

def generate_seats(num_seats=100, seats_per_table=4, alpha=0.5, beta=50):
    """
    Generate seats with x,y coordinates in a grid pattern.
    
    Args:
        num_seats: Total number of seats (default: 100)
        seats_per_table: Number of seats per table (default: 4)
        alpha: Weight for distance-based brightness/noise calculation
        beta: Base noise level
        
    Returns:
        DataFrame with columns: [seat_id, table_id, brightness, noise, seat_x, seat_y, seat_available]
    """
    # Calculate grid dimensions
    num_tables = num_seats // seats_per_table  # Should be 25 tables
    tables_per_row = int(math.sqrt(num_tables))  # 5 tables per row
    tables_per_col = num_tables // tables_per_row  # 5 tables per column
    
    # Room center (assuming room spans from 0 to 20 in both dimensions)
    room_center_x = 10.0
    room_center_y = 10.0
    
    seats = []
    seat_id = 1
    
    # Generate seats in grid pattern (horizontal first, then vertical)
    for table_row in range(tables_per_col):
        for table_col in range(tables_per_row):
            table_id = table_row * tables_per_row + table_col + 1
            
            # Table center coordinates
            table_x = table_col * 4.0 + 2.0  # Tables spaced 4 units apart, centered at 2, 6, 10, 14, 18
            table_y = table_row * 4.0 + 2.0
            
            # Generate 4 seats per table in a 2x2 pattern
            seat_positions = [
                (table_x - 0.5, table_y - 0.5),  # Bottom-left
                (table_x + 0.5, table_y - 0.5),  # Bottom-right
                (table_x - 0.5, table_y + 0.5),  # Top-left
                (table_x + 0.5, table_y + 0.5),  # Top-right
            ]
            
            for seat_x, seat_y in seat_positions:
                # Calculate distance from room center
                distance = math.sqrt((seat_x - room_center_x)**2 + (seat_y - room_center_y)**2)
                
                # Brightness decreases with distance from center
                brightness_mean = alpha * distance
                brightness = int(np.clip(np.random.normal(brightness_mean, 15), 0, 100))
                
                # Noise increases with distance from center
                noise_mean = beta - alpha * distance
                noise = int(np.clip(np.random.normal(noise_mean, 15), 0, 100))
                
                seats.append({
                    "seat_id": seat_id,
                    "table_id": table_id,
                    "brightness": brightness,
                    "noise": noise,
                    "seat_x": seat_x,
                    "seat_y": seat_y,
                    "seat_available": True
                })
                seat_id += 1
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(seats)
    df.to_csv("seats_toy.csv", index=False)
    print(f"Generated {len(df)} seats in {num_tables} tables, saved to seats_toy.csv")
    
    return df

def generate_students(num_students=60, students_per_group=5):
    """
    Generate students with brightness and noise preferences.
    
    Args:
        num_students: Total number of students (default: 60)
        students_per_group: Number of students per group (default: 5)
        
    Returns:
        DataFrame with columns: [student_id, group_id, brightness_preference, noise_preference]
    """
    num_groups = num_students // students_per_group  # Should be 12 groups
    
    students = []
    student_id = 1
    
    for group_id in range(1, num_groups + 1):
        for _ in range(students_per_group):
            # Generate preferences with specified distributions
            brightness_preference = int(np.clip(np.random.normal(60, 15), 0, 100))
            noise_preference = int(np.clip(np.random.normal(40, 15), 0, 100))
            
            students.append({
                "student_id": student_id,
                "group_id": group_id,
                "brightness_preference": brightness_preference,
                "noise_preference": noise_preference
            })
            student_id += 1
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(students)
    df.to_csv("students_toy.csv", index=False)
    print(f"Generated {len(df)} students in {num_groups} groups, saved to students_toy.csv")
    
    return df

def calculate_table_statistics():
    """
    Calculate table statistics from the seats data.
    
    Returns:
        DataFrame with columns: [table_id, average_brightness, average_noise, table_x, table_y]
    """
    # Load seats data
    try:
        seats_df = pd.read_csv("seats_toy.csv")
    except FileNotFoundError:
        print("seats_toy.csv not found. Please run generate_seats() first.")
        return None
    
    # Calculate statistics for each table
    table_stats = []
    
    for table_id in sorted(seats_df['table_id'].unique()):
        table_seats = seats_df[seats_df['table_id'] == table_id]
        
        avg_brightness = table_seats['brightness'].mean()
        avg_noise = table_seats['noise'].mean()
        table_x = table_seats['seat_x'].mean()  # Average x coordinate
        table_y = table_seats['seat_y'].mean()   # Average y coordinate
        
        table_stats.append({
            "table_id": table_id,
            "average_brightness": round(avg_brightness, 2),
            "average_noise": round(avg_noise, 2),
            "table_x": round(table_x, 2),
            "table_y": round(table_y, 2)
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(table_stats)
    df.to_csv("tables_toy.csv", index=False)
    print(f"Generated statistics for {len(df)} tables, saved to tables_toy.csv")
    
    return df


def print_layout():
    """
    Print the layout of all tables and seats in a grid format.
    Shows the room layout with table IDs and seat coordinates.
    """
    try:
        seats_df = pd.read_csv("seats_toy.csv")
    except FileNotFoundError:
        print("seats_toy.csv not found. Please run generate_seats() first.")
        return
    
    # Get unique table positions
    table_positions = seats_df.groupby("table_id").agg({
        "table_id": "first",
        "seat_x": "mean",
        "seat_y": "mean"
    }).round(1)
    
    # Create a grid representation
    print("\n" + "="*60)
    print("ROOM LAYOUT - TABLES AND SEATS")
    print("="*60)
    
    # Find grid bounds
    min_x = seats_df["seat_x"].min()
    max_x = seats_df["seat_x"].max()
    min_y = seats_df["seat_y"].min()
    max_y = seats_df["seat_y"].max()
    
    print(f"Room bounds: X({min_x:.1f} to {max_x:.1f}), Y({min_y:.1f} to {max_y:.1f})")
    print(f"Room center: (10.0, 10.0)")
    print()
    
    # Print table layout
    print("TABLE LAYOUT (Table IDs):")
    print("-" * 40)
    
    # Create a grid for tables
    grid_size = 0.5
    x_steps = int((max_x - min_x) / grid_size) + 1
    y_steps = int((max_y - min_y) / grid_size) + 1
    
    # Initialize grid
    grid = [["  " for _ in range(x_steps)] for _ in range(y_steps)]
    
    # Fill grid with table IDs
    for _, table in table_positions.iterrows():
        x_idx = int((table["seat_x"] - min_x) / grid_size)
        y_idx = int((table["seat_y"] - min_y) / grid_size)
        if 0 <= x_idx < x_steps and 0 <= y_idx < y_steps:
            grid[y_idx][x_idx] = f"{int(table["table_id"]):2d}"
    
    # Print grid (flip Y axis so top is at top)
    for y_idx in reversed(range(y_steps)):
        row = " ".join(grid[y_idx])
        y_coord = min_y + y_idx * grid_size
        print(f"Y{y_coord:4.1f}: {row}")
    
    print("\nX coordinates:", end="")
    for x_idx in range(0, x_steps, 2):
        x_coord = min_x + x_idx * grid_size
        print(f"{x_coord:6.1f}", end="")
    print()
    
    # Print detailed seat layout for each table
    print("\n" + "="*60)
    print("DETAILED SEAT LAYOUT")
    print("="*60)
    
    for table_id in sorted(seats_df["table_id"].unique()):
        table_seats = seats_df[seats_df["table_id"] == table_id].sort_values(["seat_y", "seat_x"])
        
        print(f"\nTable {table_id} (Center: {table_seats["seat_x"].mean():.1f}, {table_seats["seat_y"].mean():.1f}):")
        print("  Seats:")
        
        # Group seats by row (same Y coordinate)
        for y_coord in sorted(table_seats["seat_y"].unique()):
            row_seats = table_seats[table_seats["seat_y"] == y_coord].sort_values("seat_x")
            seat_info = []
            for _, seat in row_seats.iterrows():
                seat_info.append(f"ID{seat["seat_id"]:2d}({seat["seat_x"]:.1f},{seat["seat_y"]:.1f})[B{seat["brightness"]:2d},N{seat["noise"]:2d}]")
            print(f"    Y{y_coord:.1f}: {" ".join(seat_info)}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("LAYOUT SUMMARY")
    print("="*60)
    print(f"Total tables: {seats_df["table_id"].nunique()}")
    print(f"Total seats: {len(seats_df)}")
    print(f"Seats per table: {len(seats_df) // seats_df["table_id"].nunique()}")
    print(f"Grid dimensions: {int(math.sqrt(seats_df["table_id"].nunique()))}x{int(math.sqrt(seats_df["table_id"].nunique()))}")
    
    # Brightness and noise statistics
    print(f"\nBrightness range: {seats_df["brightness"].min()}-{seats_df["brightness"].max()}")
    print(f"Noise range: {seats_df["noise"].min()}-{seats_df["noise"].max()}")
    print(f"Average brightness: {seats_df["brightness"].mean():.1f}")
    print(f"Average noise: {seats_df["noise"].mean():.1f}")
if __name__ == "__main__":
    # Generate all toy datasets
    print("Generating toy datasets...")
    
    # Generate seats (100 seats, 4 per table, 25 tables total)
    seats_df = generate_seats(num_seats=100, seats_per_table=4, alpha=0.5, beta=50)
    
    # Generate students (60 students, 5 per group, 12 groups total)
    students_df = generate_students(num_students=60, students_per_group=5)
    
    # Calculate table statistics
    tables_df = calculate_table_statistics()
    
    print("\nToy dataset generation complete!")
    print(f"Seats: {len(seats_df)} seats in {seats_df['table_id'].nunique()} tables")
    print(f"Students: {len(students_df)} students in {students_df['group_id'].nunique()} groups")
    print(f"Tables: {len(tables_df)} table statistics")
    
    
    # Print the layout
    print_layout()

def print_layout():
    """
    Print the layout of all tables and seats in a grid format.
    Shows the room layout with table IDs and seat coordinates.
    """
    try:
        seats_df = pd.read_csv("seats_toy.csv")
    except FileNotFoundError:
        print("seats_toy.csv not found. Please run generate_seats() first.")
        return
    
    # Get unique table positions
    table_positions = seats_df.groupby('table_id').agg({
        'table_id': 'first',
        'seat_x': 'mean',
        'seat_y': 'mean'
    }).round(1)
    
    # Create a grid representation
    print("\n" + "="*60)
    print("ROOM LAYOUT - TABLES AND SEATS")
    print("="*60)
    
    # Find grid bounds
    min_x = seats_df['seat_x'].min()
    max_x = seats_df['seat_x'].max()
    min_y = seats_df['seat_y'].min()
    max_y = seats_df['seat_y'].max()
    
    print(f"Room bounds: X({min_x:.1f} to {max_x:.1f}), Y({min_y:.1f} to {max_y:.1f})")
    print(f"Room center: (10.0, 10.0)")
    print()
    
    # Print table layout
    print("TABLE LAYOUT (Table IDs):")
    print("-" * 40)
    
    # Create a grid for tables
    grid_size = 0.5
    x_steps = int((max_x - min_x) / grid_size) + 1
    y_steps = int((max_y - min_y) / grid_size) + 1
    
    # Initialize grid
    grid = [['  ' for _ in range(x_steps)] for _ in range(y_steps)]
    
    # Fill grid with table IDs
    for _, table in table_positions.iterrows():
        x_idx = int((table['seat_x'] - min_x) / grid_size)
        y_idx = int((table['seat_y'] - min_y) / grid_size)
        if 0 <= x_idx < x_steps and 0 <= y_idx < y_steps:
            grid[y_idx][x_idx] = f"{int(table['table_id']):2d}"
    
    # Print grid (flip Y axis so top is at top)
    for y_idx in reversed(range(y_steps)):
        row = ' '.join(grid[y_idx])
        y_coord = min_y + y_idx * grid_size
        print(f"Y{y_coord:4.1f}: {row}")
    
    print("\nX coordinates:", end="")
    for x_idx in range(0, x_steps, 2):
        x_coord = min_x + x_idx * grid_size
        print(f"{x_coord:6.1f}", end="")
    print()
    
    # Print detailed seat layout for each table
    print("\n" + "="*60)
    print("DETAILED SEAT LAYOUT")
    print("="*60)
    
    for table_id in sorted(seats_df['table_id'].unique()):
        table_seats = seats_df[seats_df['table_id'] == table_id].sort_values(['seat_y', 'seat_x'])
        
        print(f"\nTable {table_id} (Center: {table_seats['seat_x'].mean():.1f}, {table_seats['seat_y'].mean():.1f}):")
        print("  Seats:")
        
        # Group seats by row (same Y coordinate)
        for y_coord in sorted(table_seats['seat_y'].unique()):
            row_seats = table_seats[table_seats['seat_y'] == y_coord].sort_values('seat_x')
            seat_info = []
            for _, seat in row_seats.iterrows():
                seat_info.append(f"ID{seat['seat_id']:2d}({seat['seat_x']:.1f},{seat['seat_y']:.1f})[B{seat['brightness']:2d},N{seat['noise']:2d}]")
            print(f"    Y{y_coord:.1f}: {' '.join(seat_info)}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("LAYOUT SUMMARY")
    print("="*60)
    print(f"Total tables: {seats_df['table_id'].nunique()}")
    print(f"Total seats: {len(seats_df)}")
    print(f"Seats per table: {len(seats_df) // seats_df['table_id'].nunique()}")
    print(f"Grid dimensions: {int(math.sqrt(seats_df['table_id'].nunique()))}x{int(math.sqrt(seats_df['table_id'].nunique()))}")
    
    # Brightness and noise statistics
    print(f"\nBrightness range: {seats_df['brightness'].min()}-{seats_df['brightness'].max()}")
    print(f"Noise range: {seats_df['noise'].min()}-{seats_df['noise'].max()}")
    print(f"Average brightness: {seats_df['brightness'].mean():.1f}")
    print(f"Average noise: {seats_df['noise'].mean():.1f}")


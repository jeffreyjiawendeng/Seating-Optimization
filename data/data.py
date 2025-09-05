import numpy as np
import pandas as pd


def generate_seats(num_seats=2000,
                   num_tables=None,
                   num_rooms=5,
                   cleanliness_min=1,
                   cleanliness_max=100,
                   noise_min=1,
                   noise_max=100):
    """
    Generate seats with uniformly distributed random attributes.
    
    Args:
        num_seats: Total number of seats to generate
        num_tables: Number of tables (if None, calculated as num_seats // 10)
        num_rooms: Number of rooms to distribute tables across
        cleanliness_min: Minimum value for cleanliness uniform distribution
        cleanliness_max: Maximum value for cleanliness uniform distribution
        noise_min: Minimum value for noise uniform distribution
        noise_max: Maximum value for noise uniform distribution
        
    Returns:
        DataFrame with seat data
    """
    if num_tables is None:
        num_tables = max(1, num_seats // 10)  # Default: ~10 seats per table
    
    seats = []
    seat_id = 1
    
    # Calculate seats per table
    seats_per_table = max(1, num_seats // num_tables)
    
    for table_id in range(1, num_tables + 1):
        # Assign room (distribute tables evenly across rooms)
        room_id = ((table_id - 1) % num_rooms) + 1
        
        # Determine how many seats for this table
        remaining_seats = num_seats - (seat_id - 1)
        remaining_tables = num_tables - table_id + 1
        current_table_seats = min(seats_per_table, remaining_seats // remaining_tables) if remaining_tables > 1 else remaining_seats
        
        for _ in range(current_table_seats):
            if seat_id > num_seats:
                break
                
            # Generate uniformly distributed attributes
            cleanliness = np.random.randint(cleanliness_min, cleanliness_max + 1)
            noise = np.random.randint(noise_min, noise_max + 1)
            
            seats.append({
                "Seat_ID": seat_id,
                "Table_ID": table_id,
                "Room_ID": room_id,
                "Cleanliness": cleanliness,
                "Noise": noise,
                "Seat_Available": True,
                "Table_Available": True,
                "Room_Available": True
            })
            seat_id += 1
    
    df = pd.DataFrame(seats, columns=[
        "Seat_ID", 
        "Table_ID", 
        "Room_ID", 
        "Cleanliness", 
        "Noise", 
        "Seat_Available", 
        "Table_Available", 
        "Room_Available"
    ])
    
    return df


def generate_students(num_students=2000,
                      num_groups=None,
                      group_size_range=(1, 10),
                      cleanliness_mean=50,
                      cleanliness_sd=15,
                      noise_mean=50,
                      noise_sd=15):
    """
    Generate students with normally distributed random attributes.
    
    Args:
        num_students: Total number of students to generate
        num_groups: Number of groups (if None, calculated based on average group size)
        group_size_range: Range for group sizes
        cleanliness_mean: Mean for cleanliness preference
        cleanliness_sd: Standard deviation for cleanliness
        noise_mean: Mean for noise tolerance
        noise_sd: Standard deviation for noise
        
    Returns:
        DataFrame with student data
    """
    if num_groups is None:
        avg_group_size = (group_size_range[0] + group_size_range[1]) / 2
        num_groups = max(1, int(num_students / avg_group_size))
    
    students = []
    student_id = 1
    
    # Generate group sizes
    group_sizes = np.random.randint(group_size_range[0], group_size_range[1] + 1, size=num_groups)
    
    # Adjust group sizes to match total students
    total_assigned = sum(group_sizes)
    if total_assigned != num_students:
        diff = num_students - total_assigned
        # Distribute the difference across groups
        for i in range(abs(diff)):
            group_idx = i % num_groups
            if diff > 0:
                group_sizes[group_idx] += 1
            else:
                group_sizes[group_idx] = max(1, group_sizes[group_idx] - 1)
    
    for group_id, group_size in enumerate(group_sizes, start=1):
        for _ in range(group_size):
            if student_id > num_students:
                break
                
            # Generate normally distributed attributes
            cleanliness = int(np.clip(np.random.normal(cleanliness_mean, cleanliness_sd), 1, 100))
            noise = int(np.clip(np.random.normal(noise_mean, noise_sd), 1, 100))
            
            students.append({
                "Student_ID": student_id,
                "Group_ID": group_id,
                "Cleanliness": cleanliness,
                "Noise": noise
            })
            student_id += 1
    
    df = pd.DataFrame(students, columns=[
        "Student_ID", 
        "Group_ID",
        "Cleanliness", 
        "Noise"
    ])
    
    return df


def save_to_csv(seats_df=None, students_df=None, 
                seats_path="seats.csv", 
                students_path="students.csv"):
    """
    Save DataFrames to CSV files.
    
    Args:
        seats_df: DataFrame containing seat data
        students_df: DataFrame containing student data
        seats_path: Path for seats CSV file
        students_path: Path for students CSV file
    """
    if seats_df is not None:
        seats_df.to_csv(seats_path, index=False)
        print(f"Generated {len(seats_df)} seats and saved to {seats_path}")
        print(f"  Tables: {seats_df['Table_ID'].nunique()}, Rooms: {seats_df['Room_ID'].nunique()}")
        print(f"  Cleanliness: mean={seats_df['Cleanliness'].mean():.1f}, std={seats_df['Cleanliness'].std():.1f}")
        print(f"  Noise: mean={seats_df['Noise'].mean():.1f}, std={seats_df['Noise'].std():.1f}")
    
    if students_df is not None:
        students_df.to_csv(students_path, index=False)
        print(f"Generated {len(students_df)} students and saved to {students_path}")
        print(f"  Groups: {students_df['Group_ID'].nunique()}")
        print(f"  Cleanliness: mean={students_df['Cleanliness'].mean():.1f}, std={students_df['Cleanliness'].std():.1f}")
        print(f"  Noise: mean={students_df['Noise'].mean():.1f}, std={students_df['Noise'].std():.1f}")
        group_sizes = students_df.groupby('Group_ID').size()
        print(f"  Group sizes: min={group_sizes.min()}, max={group_sizes.max()}, avg={group_sizes.mean():.1f}")


if __name__ == "__main__":
    # Set random seed for reproducible results (optional)
    np.random.seed(42)
    
    # Get user input for number of rows
    try:
        num_seats = int(input("Enter number of seats to generate (default 2000): ") or 2000)
        num_students = int(input("Enter number of students to generate (default 2000): ") or 2000)
    except ValueError:
        print("Using default values: 2000 seats, 2000 students")
        num_seats = 2000
        num_students = 2000
    
    print(f"\nGenerating {num_seats} seats and {num_students} students...")
    
    # Generate seats with uniformly distributed attributes
    seats_df = generate_seats(
        num_seats=num_seats,
        cleanliness_min=1,
        cleanliness_max=100,
        noise_min=1,
        noise_max=100
    )
    
    # Generate students with normally distributed attributes
    students_df = generate_students(
        num_students=num_students,
        cleanliness_mean=60,  # Students prefer cleaner seats
        cleanliness_sd=15,
        noise_mean=40,        # Students prefer quieter environments
        noise_sd=15
    )
    
    # Save to CSV files
    save_to_csv(seats_df, students_df)
    
    print("\nData generation complete!")

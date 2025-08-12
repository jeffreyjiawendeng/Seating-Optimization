import numpy as np
import pandas as pd

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
    Noise: base = row_weight * col_weight where row_weight and col_weight are mirrored ramps => scaled max = 100 => add a Gaussian
    
    Input:
        layout: 5D tensor describing the layout of the library.
        brightness_range: range of integers for the brightness.
        brightness_sd: standard deviation for the brightness level, int.
        noise_sd: standard deviation for the noise level, int.
        
    Ouutput:
        attributes: 6D tensor of seats and their attributes.
    """ 
    # Unpack and set dimension variables
    rooms, tr, tc, sr, sc = layout.shape
    total_rows = tr * sr
    total_cols = tc * sc

    # brightness base for each row 
    brightness_bases = np.linspace(brightness_range[0], brightness_range[1], total_rows)

    # linear ramps
    row_weights = mirror_weights(total_rows) 
    col_weights = mirror_weights(total_cols) 

    # Scale maps max to 100
    max_base = (total_rows//2) * (total_cols//2)
    scale = 100 / max_base

    # Initialize zeros attribute tensor 
    attributes = np.zeros((rooms, tr, tc, sr, sc, 2), dtype=int)
    
    # Initialize attributes for each seat
    for room in range(rooms):
        for tr_i in range(tr):
            for tc_i in range(tc):
                for sr_i in range(sr):
                    for sc_i in range(sc):
                        # brightness with small noise
                        brightness = int(np.clip(np.random.normal(brightness_bases[tr_i * sr + sr_i], brightness_sd), 0, 100))
                        
                        # noise base
                        noise_base = row_weights[tr_i * sr + sr_i] * col_weights[tc_i * sc + sc_i] * scale
                        
                        # add gaussian noise and clip
                        noise = int(np.clip(np.random.normal(noise_base, noise_sd), 0, 100))                
                        
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
                        b, n = attrs[room, tr_i, tc_i, sr_i, sc_i]
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
            brightness = int(np.clip(np.random.normal(brightness_mean, brightness_sd), 0, 100))
            noise = int(np.clip(np.random.normal(noise_mean, noise_sd), 0, 100))
            flex = np.random.randint(flexibility_range[0], flexibility_range[1] + 1)
            students.append({
                "Student_ID":   f"S{student_id:03d}",
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
    
if __name__ == "__main__":
    layout = generate_layout((3, 2, 5, 2, 5))
    attrs = generate_attribute_layout(
        layout,
        brightness_range=(80, 20),
        brightness_sd=3,
        noise_sd=10
    )
    display_attribute_layout(attrs)
    attributes_to_csv(attrs)
    
    df_students = generate_students(
        num_groups=200,
        brightness_mean=60,
        brightness_sd=20,
        noise_mean=40,
        noise_sd=10,
        flexibility_range=(1,5)
    )
    print(df_students.head())
    students_to_csv(df_students)

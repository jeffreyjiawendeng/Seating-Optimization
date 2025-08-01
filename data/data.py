import numpy as np
import pandas as pd

def mirror_weights(length):
    """
    Create an array of weights that increases then decreases linearly at the midpoint of an array.
    
    length: number representing the length of the array.
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
        
    return increasing + decreasing


def generate_layout(shape=(3, 2, 2, 2, 3)):
    """
    Create a layout tensor of ones with the given 5D shape.
    
    shape: 5D shape representing the number of rooms, the shape of each room, and the shape of each table.
    """
    return np.ones(shape, dtype=int)

def generate_attribute_layout(layout, brightness_range=(80, 20), whiteboard_range=(0, 1), brightness_sd=3, noise_sd=10):
    """
    Returns a tensor of shape (rooms, tr, tc, sr, sc, 3) with attributes [brightness, noise, whiteboards] for each seat.

    Brightness: total_rows = table_rows * seat_rows => linear ramp.

    Noise: base = row_weight * col_weight where row_weight and col_weight are mirrored ramps => scaled max = 100 => add a Gaussian.

    Whiteboards: random integer in whiteboard_range per seat.
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
    attributes = np.zeros((rooms, tr, tc, sr, sc, 3), dtype=int)
    
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
                        
                        # whiteboards => uniform distribution 
                        whiteboards = np.random.randint(whiteboard_range[0], whiteboard_range[1] + 1)
                        
                        # set attributes
                        attributes[room, tr_i, tc_i, sr_i, sc_i] = (brightness, noise, whiteboards)
                        
    # Return row of attributes 
    return attributes

def display_attribute_layout(attributes):
    """
    Prints each room's tables with seat attributes: (brightness, noise, whiteboards) for each seat.
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
                        b, n, w = attributes[room, tr_i, tc_i, sr_i, sc_i]
                        seats.append(f"({b:3d},{n:3d},{w:2d})")
                    row_sections.append(' '.join(seats))
                print(' | '.join(row_sections))
            if tr_i < tr - 1:
                print('-' * (12 * tc * sc + tc * (sc - 1) + (tc - 1) * 3))
        print()

def attributes_to_csv(attributes, path="data/seats.csv"):
    """
    Turn the attrs tensor into a flat DataFrame with columns:
        Seat_ID, Table_ID, Room_ID, Brightness, Whiteboards, Noise, Seat_Available, Table_Available, Room_Available
    and save to CSV.
    """
    rooms, tr, tc, sr, sc, _ = attributes.shape
    rows = []
    seat_id = 1

    for room in range(rooms):
        room_id = f"R{room+1}"
        for tr_i in range(tr):
            for tc_i in range(tc):
                table_id = f"{room_id}_T{tr_i+1}_{tc_i+1}"
                for sr_i in range(sr):
                    for sc_i in range(sc):
                        b, n, wb = attributes[room, tr_i, tc_i, sr_i, sc_i]
                        rows.append({
                            "Seat_ID":         seat_id,
                            "Table_ID":        table_id,
                            "Room_ID":         room_id,
                            "Brightness":      b,
                            "Whiteboards":     wb,
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
        "Whiteboards", 
        "Noise", 
        "Seat_Available", 
        "Table_Available", 
        "Room_Available"
    ])
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} seats to {path}")

if __name__ == "__main__":
    layout = generate_layout((3, 2, 2, 2, 3))
    attrs = generate_attribute_layout(layout,
                                brightness_range=(80, 20),
                                whiteboard_range=(0, 1),
                                brightness_sd=3,
                                noise_sd=10)
    display_attribute_layout(attrs)
    attributes_to_csv(attrs)

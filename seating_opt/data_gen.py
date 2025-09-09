# seating_opt/data_gen.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple


def generate_seats(
    rooms: int = 1,
    tables_per_room: int = 6,
    rows_per_table: int = 4,
    cols_per_table: int = 5,
    table_gap: int = 2,
    room_gap: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a stitched grid with (X,Y):
      - Brightness: linear ramp front->back + small noise (80->20 approx)
      - Noise: hill shape (louder center) + Gaussian noise
    """
    rng = np.random.default_rng(seed)
    seats = []
    sid = 1
    # Lay out rooms in X with room_gap, tables in X within a room with table_gap
    for r in range(rooms):
        room_x0 = r * (tables_per_room * (cols_per_table + table_gap) + room_gap)
        for t in range(tables_per_room):
            table_x0 = room_x0 + t * (cols_per_table + table_gap)
            for rr in range(rows_per_table):
                for cc in range(cols_per_table):
                    X = table_x0 + cc
                    Y = rr  # stitch tables row-aligned
                    # Brightness: higher in front (smaller Y), 0..100-ish
                    base_b = 80 - (Y * (60 / max(1, rows_per_table - 1)))  # ~80..20
                    b = np.clip(base_b + rng.normal(0, 3), 0, 100)
                    # Noise: hill (center rows/cols are louder)
                    row_profile = np.array([1, 2, 3, 2, 1][:rows_per_table])
                    col_profile = np.array([1, 2, 3, 2, 1][:cols_per_table])
                    rp = row_profile[min(rr, len(row_profile) - 1)]
                    cp = col_profile[min(cc, len(col_profile) - 1)]
                    base_n = 15 + 4 * (rp + cp)  # ~min 23.. max ~43
                    n = np.clip(base_n + rng.normal(0, 2), 0, 100)
                    seats.append(
                        {
                            "Seat_ID": sid,
                            "Room_ID": r + 1,
                            "Table_ID": t + 1 + r * tables_per_room,
                            "X": int(X),
                            "Y": int(Y),
                            "Brightness": float(b),
                            "Noise": float(n),
                            "Seat_Available": True,
                        }
                    )
                    sid += 1
    return pd.DataFrame(seats)


def generate_groups_from_students(
    n_students: int = 200,
    min_group: int = 1,
    max_group: int = 8,
    brightness_mu: float = 60,
    brightness_sigma: float = 20,
    noise_mu: float = 40,
    noise_sigma: float = 10,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a 'students' table (preferences) and a 'groups' table used by solvers.
    Brightness/Noise preferences are per student; groups are random partitions.
    For this prototype we reduce to per-group fields:
      - Group_Size
      - Brightness_Min := min student brightness preference in group
      - Objective = "Q1" (minimize noise)
    """
    rng = np.random.default_rng(seed)
    # Students
    sids = np.arange(1, n_students + 1)
    br = np.clip(rng.normal(brightness_mu, brightness_sigma, size=n_students), 0, 100)
    no = np.clip(rng.normal(noise_mu, noise_sigma, size=n_students), 0, 100)

    # Random group sizes that sum to n_students
    sizes = []
    remaining = n_students
    while remaining > 0:
        g = int(np.clip(rng.integers(min_group, max_group + 1), min_group, remaining))
        sizes.append(g)
        remaining -= g
    rng.shuffle(sizes)

    groups_rows = []
    students_rows = []
    gid = 1
    offset = 0
    for sz in sizes:
        idx = np.arange(offset, offset + sz)
        offset += sz
        bmin = float(np.min(br[idx]))
        # record students
        for s in sids[idx]:
            students_rows.append(
                {
                    "Student_ID": int(s),
                    "Group_ID": gid,
                    "Brightness_Preference": float(br[s - 1]),
                    "Noise_Preference": float(no[s - 1]),
                }
            )
        groups_rows.append(
            {
                "Group_ID": gid,
                "Group_Size": int(sz),
                "Brightness_Min": float(bmin),
                "Objective": "Q1",
            }
        )
        gid += 1

    students_df = pd.DataFrame(students_rows)
    groups_df = pd.DataFrame(groups_rows)
    return students_df, groups_df


def generate_groups(
    n_groups: int = 15,
    min_size: int = 2,
    max_size: int = 6,
    brightness_mean: float = 60,
    brightness_std: float = 20,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generate a simple groups DataFrame for experiments.
    
    Args:
        n_groups: Number of groups to generate
        min_size: Minimum group size
        max_size: Maximum group size  
        brightness_mean: Mean brightness requirement
        brightness_std: Std dev of brightness requirements
        seed: Random seed
        
    Returns:
        DataFrame with columns: Group_ID, Group_Size, Brightness_Min, Objective
    """
    rng = np.random.default_rng(seed)
    
    groups = []
    for i in range(1, n_groups + 1):
        group_size = int(rng.integers(min_size, max_size + 1))
        brightness_min = float(np.clip(rng.normal(brightness_mean, brightness_std), 10, 95))
        
        groups.append({
            "Group_ID": i,
            "Group_Size": group_size,
            "Brightness_Min": brightness_min,
            "Objective": "Q1"
        })
    
    return pd.DataFrame(groups)
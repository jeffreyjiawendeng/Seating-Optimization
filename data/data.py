import random
import numpy as np
import pandas as pd
from pathlib import Path

def make_students(n: int = 100, seed: int = 42, out_path: str = "data/students.csv") -> pd.DataFrame:
    """
    Generate a dataset of student seating preferences.

    Columns:
    - Student_ID         : Unique ID from 1 to n
    - Group_ID           : 0 for solo; same non-zero ID for group members
    - Noise              : Preference from 1 (low) to 5 (high)
    - Brightness         : Preference from 1 (low) to 5 (high)
    - Flexibility        : 'hard', 'medium', or 'not_strict'
    - Flexibility_Score  : 1 = hard, 2 = medium, 3 = not_strict
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Assign group sizes (~60% solo, 30% pairs, 10% small groups)
    group_ids, i, gid = [], 0, 1
    while i < n:
        r = rng.random()
        size = 1 if r < 0.60 else (2 if r < 0.90 else rng.randint(3, 4))
        for _ in range(min(size, n - i)):
            group_ids.append(gid if size > 1 else 0)
            i += 1
        if size > 1:
            gid += 1

    # Generate noise and brightness preferences
    def pref_scale(mean=3.0, std_dev=1.1, count=n):
        return np.clip(np.rint(np.random.normal(mean, std_dev, count)), 1, 5).astype(int)

    noise = pref_scale()
    brightness = pref_scale()

    # Assign flexibility levels
    FLEX_LEVELS = ["hard", "medium", "not_strict"]
    idx = np.random.choice([0, 1, 2], size=n, p=[0.25, 0.50, 0.25])
    flexibility = [FLEX_LEVELS[i] for i in idx]
    flexibility_score = (idx + 1).astype(int)

    # Assemble final DataFrame
    df = pd.DataFrame({
        "Student_ID": np.arange(1, n + 1),
        "Group_ID": group_ids,
        "Noise": noise,
        "Brightness": brightness,
        "Flexibility": flexibility,
        "Flexibility_Score": flexibility_score
    })

    # Save to CSV
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df

if __name__ == "__main__":
    df = make_students(n=100, seed=42, out_path="data/students.csv")
    print(" Student dataset saved to data/students.csv")
    print("\n First 10 rows:\n")
    print(df.head(10))
    print("\n Flexibility breakdown:")
    print(df["Flexibility"].value_counts())

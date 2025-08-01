import random
import numpy as np
import pandas as pd
from pathlib import Path

def make_students(n: int = 100, seed: int = 42, out_path: str = "data/students.csv") -> pd.DataFrame:
    """
    Build the students dataset.

    Columns
    -------
    - Student_ID : 1..n (unique)
    - Group_ID   : 0 for solo; same non-zero ID means same group
    - Noise      : 1..5 (preference/sensitivity scale; 1=low, 5=high)
    - Brightness : 1..5 (preference/sensitivity scale; 1=low, 5=high)
    - Flexibility: {'hard','medium','not_strict'} meaning seat flexibility
    - Flexibility_Score: 1=hard, 2=medium, 3=not_strict
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # ---- Group assignment: ~60% solo, 30% pairs, 10% groups of 3–4
    group_ids, i, gid = [], 0, 1
    while i < n:
        r = rng.random()
        size = 1 if r < 0.60 else (2 if r < 0.90 else rng.randint(3, 4))
        for _ in range(min(size, n - i)):
            group_ids.append(gid if size > 1 else 0)
            i += 1
        if size > 1:
            gid += 1

    # ---- Preferences (1..5), centered around 3
    def pref_scale(mu=3.0, sigma=1.1, count=n):
        return np.clip(np.rint(np.random.normal(mu, sigma, count)), 1, 5).astype(int)

    noise = pref_scale()
    brightness = pref_scale()

    # ---- Flexibility categories + numeric score
    FLEX_LEVELS = ["hard", "medium", "not_strict"]
    idx = np.random.choice([0, 1, 2], size=n, p=[0.25, 0.50, 0.25])
    flexibility = [FLEX_LEVELS[i] for i in idx]
    flexibility_score = (idx + 1).astype(int)

    # ---- Final DataFrame
    df = pd.DataFrame({
        "Student_ID": np.arange(1, n + 1, dtype=int),
        "Group_ID": np.array(group_ids, dtype=int),
        "Noise": noise,
        "Brightness": brightness,
        "Flexibility": flexibility,
        "Flexibility_Score": flexibility_score,
    })

    # ---- Basic checks
    assert df["Student_ID"].is_unique
    assert df["Noise"].between(1, 5).all()
    assert df["Brightness"].between(1, 5).all()
    assert df["Flexibility_Score"].between(1, 3).all()

    # ---- Save to CSV
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    return df

if __name__ == "__main__":
    df = make_students(n=100, seed=42, out_path="data/students.csv")
    print("Wrote data/students.csv")
    print(df.head(10))
    print("\nFlexibility counts:\n", df["Flexibility"].value_counts())

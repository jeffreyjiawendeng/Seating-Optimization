# Experiment Fairness Verification

## **Data Consistency Across Trials**

### **Seat Selection (Fixed)**
- **Before**: Random sampling with `random_state=42` (different seats each trial)
- **After**: Sequential selection using `seats_df.head(seat_count)` (same seats each trial)

### **Trial Structure**
| Trial | Seats Used | Groups Used | Students |
|-------|------------|-------------|----------|
| 1     | 1-500      | Set 1       | 500      |
| 2     | 1-1000     | Sets 1+2    | 1000     |
| 3     | 1-1500     | Sets 1+2+3  | 1500     |
| 4     | 1-2000     | Sets 1+2+3+4| 2000     |

### **Data Consistency Guarantees**

#### **1. Seats**
- **Trial 1**: Seats 1, 2, 3, ..., 500
- **Trial 2**: Seats 1, 2, 3, ..., 1000 (includes all from Trial 1)
- **Trial 3**: Seats 1, 2, 3, ..., 1500 (includes all from Trial 2)
- **Trial 4**: Seats 1, 2, 3, ..., 2000 (includes all from Trial 3)

#### **2. Groups**
- **Trial 1**: Groups from Set 1 only
- **Trial 2**: Groups from Sets 1 + 2 (includes all from Trial 1)
- **Trial 3**: Groups from Sets 1 + 2 + 3 (includes all from Trial 2)
- **Trial 4**: Groups from Sets 1 + 2 + 3 + 4 (includes all from Trial 3)

#### **3. Processing Order**
- Groups are processed in **sorted order** by Group_ID
- Both algorithms (Greedy and ILP) receive groups in the **exact same order**
- Seat availability is updated consistently between algorithms

### **Algorithm Fairness**

#### **Same Input Data**
- ✅ **Seats**: Identical seat sets for each trial
- ✅ **Groups**: Identical group sets for each trial
- ✅ **Order**: Identical processing order for each trial
- ✅ **Constraints**: Identical brightness thresholds and group sizes

#### **Same Objective Functions**
- **Q1**: Both minimize `noise`
- **Q2**: Both minimize `(noise - 0.3 × brightness)`

### **Verification Commands**

```bash
# Test seat consistency
python3 -c "from experiments import load_experiment_dataset; seats_df, _ = load_experiment_dataset(); print('Trial 1 seats:', seats_df.head(500)['Seat_ID'].tolist()[:5]); print('Trial 2 seats:', seats_df.head(1000)['Seat_ID'].tolist()[:5])"

# Test group consistency
python3 -c "import pandas as pd; df = pd.read_csv('groups.csv'); print('Set 1:', len(df[df['Set_ID'] == 1])); print('Sets 1+2:', len(df[df['Set_ID'].isin([1,2])]))"
```

### **Benefits of This Approach**

1. **Fair Comparison**: Both algorithms solve identical problems
2. **Reproducible Results**: Same data every time
3. **Scalability Testing**: True performance comparison across dataset sizes
4. **Consistent Constraints**: Same seat availability patterns
5. **Order Independence**: Results don't depend on random sampling

### **What This Fixes**

- ❌ **Before**: Random seat sampling → Different problems each trial
- ❌ **Before**: Potential group ordering differences → Inconsistent processing
- ✅ **After**: Sequential seat selection → Identical problems each trial
- ✅ **After**: Sorted group processing → Consistent processing order

The experiments now provide a **truly fair comparison** between Greedy and ILP algorithms on **identical problem instances** across all trials. 
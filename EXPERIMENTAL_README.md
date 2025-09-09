# Experimental Files for Seating Optimization

This directory contains experimental files with toy datasets to address the performance issues with the original large dataset.

## Problem

The original `demo_main.py` creates a large dataset (168 seats, 120 students in ~32 groups) that causes the ILP solver to time out due to the exponential growth in problem complexity.

## Solution: Toy Datasets

We've created smaller datasets that demonstrate the same algorithms but run quickly:

### Files

1. **`micro_experiment.py`** - Ultra-small dataset (6 seats, 6 students, 3 groups)
   - Runs in ~0.2 seconds
   - Perfect for rapid development and debugging
   - Shows detailed algorithm comparison

2. **`toy_experiment.py`** - Small realistic dataset (24 seats, 18 students, 5 groups)  
   - Runs in ~0.4 seconds
   - Good balance of realism and performance
   - Demonstrates all algorithms working properly

3. **`performance_comparison.py`** - Compares all dataset sizes
   - Shows the performance difference between dataset sizes
   - Demonstrates why the original dataset times out

## Usage

```bash
# Run the micro experiment (fastest)
python micro_experiment.py

# Run the toy experiment (realistic but small)
python toy_experiment.py

# Compare performance of all datasets
python performance_comparison.py
```

## Algorithm Results

Both toy datasets successfully demonstrate:

- **Greedy Algorithm**: Fast heuristic that builds seats incrementally
- **Myopic ILP**: Optimal per-group assignments using Integer Linear Programming
- **Global ILP**: Oracle solution knowing all groups in advance (for comparison)

The toy datasets show that:
- ILP methods find better solutions than greedy (lower regret)
- Greedy is faster but sometimes suboptimal
- Both methods handle brightness constraints and compactness objectives correctly

## Technical Details

### Problem Complexity
The original dataset creates ~37,000 ILP variables, while:
- Micro dataset: ~36 variables (1000x smaller)
- Toy dataset: ~480 variables (78x smaller)

### Why the Original Times Out
1. **Exponential pair variables**: For each group, the ILP creates variables for all seat pairs within distance `dmax_pairs`
2. **Complex constraints**: Each pair requires 3 linearization constraints
3. **Global feasibility search**: Multiple ILP solves to find feasible brightness caps
4. **CBC solver limits**: The open-source solver struggles with large mixed-integer problems

### Benefits of Toy Datasets
- ✅ Fast iteration during development
- ✅ Easy debugging and verification
- ✅ Same algorithmic behavior as large datasets
- ✅ Clear visualization of results
- ✅ Suitable for demonstrations and testing
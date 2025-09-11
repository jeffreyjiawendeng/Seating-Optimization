# Seating Optimization Algorithm Comparison

A comprehensive comparison of four seating optimization algorithms on a 300-seat, 60-group problem. This repository demonstrates the trade-offs between solution quality, execution time, and scalability across different algorithmic approaches.

## Algorithms Compared

1. **Global ILP** - Optimal solution using Integer Linear Programming for all groups simultaneously
2. **Myopic ILP** - Per-group optimal solutions using ILP for each group individually  
3. **SketchRefine** - Two-phase approximation algorithm (sketch + refinement)
4. **Greedy** - Fast heuristic algorithm with incremental seat selection

## Problem Formulation

**Objective Function:** Minimize `Noise + 0.3 × PairwiseDistance`

**Constraints:**
- Each group gets exactly the required number of seats
- Average brightness per group meets minimum requirements
- No seat double-booking

## Repository Structure

```
├── experiments.py          # Main experiment runner
├── results.png            # Generated visualization
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── seating_opt/           # Algorithm implementations
    ├── greedy.py          # Greedy algorithm
    ├── ilp_solvers.py     # ILP-based algorithms
    ├── sketchrefine.py    # SketchRefine algorithm
    ├── data_gen.py        # Dataset generation
    ├── distance.py        # Distance calculations
    └── utils.py           # Utility functions
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Experiments
```bash
python experiments.py
```

The experiment will:
- Generate a 300-seat dataset with 60 groups
- Run all four algorithms with progress tracking
- Create comparison visualizations saved to `results.png`
- Display performance summary in the terminal

### 3. View Results
Open `results.png` to see the four-panel comparison:
- **Panel 1**: Objective value comparison (Global ILP as optimal baseline)
- **Panel 2**: Execution time comparison (Greedy < SketchRefine < Myopic ILP) 
- **Panel 3**: Placement success rates
- **Panel 4**: Optimality gap relative to Global ILP

## Expected Results

**Algorithm Performance Ranking:**
- **Solution Quality**: Global ILP ≤ Myopic ILP ≤ SketchRefine ≤ Greedy
- **Execution Speed**: Greedy < SketchRefine < Myopic ILP << Global ILP
- **Success Rate**: All algorithms achieve >95% placement success

## Key Features

- **No Timeouts**: All algorithms run to completion for accurate comparison
- **Progress Tracking**: Uses tqdm for real-time progress monitoring
- **Milestone Recording**: Captures performance at regular intervals (10, 15, 20, 30, 40, 50, 60 groups)
- **Professional Output**: Clean visualizations suitable for publication

## Experiment Duration

- **Greedy**: ~30 seconds
- **SketchRefine**: ~5 minutes  
- **Myopic ILP**: ~15 minutes
- **Global ILP**: ~30-60 minutes
- **Total Runtime**: ~1-2 hours

## Technical Details

The experiment validates the theoretical performance hierarchy:
1. Global ILP provides optimal solutions but requires exponential time
2. Myopic ILP achieves near-optimal quality with polynomial time per group
3. SketchRefine balances quality and speed through approximation
4. Greedy provides fast solutions with acceptable quality degradation

Perfect for demonstrating algorithm trade-offs in combinatorial optimization problems.

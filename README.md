# Seating-Optimization

A Python project for optimizing seating arrangements in library spaces using both greedy and ILP (Integer Linear Programming) algorithms.

## Overview

This project implements two approaches to solve seating optimization problems:
1. **Greedy Algorithm**: Fast heuristic approach for quick solutions
2. **ILP Algorithm**: Optimal solution using Integer Linear Programming

The system handles two types of queries:
- **Q1**: Minimize average noise
- **Q2**: Minimize (average noise - 0.3 × average brightness)

## Dataset Generation

The project now uses a **one-time dataset generation approach** instead of regenerating data for each experiment:

- **2000 seats** generated using the layout conventions from `data.py`
- **2000 students** organized into groups of varying sizes (1-20 students per group)
- Dataset is saved to CSV files (`src/seats.csv`, `src/students.csv`) for reuse
- Subsequent runs load the pre-generated dataset instead of regenerating

### Benefits of the New Approach

1. **Consistency**: Same dataset across all experiments ensures fair comparison
2. **Performance**: No need to regenerate data for each seat count
3. **Reproducibility**: Results are consistent across multiple runs
4. **Efficiency**: Faster experiment execution

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Seating-Optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

To run the full experiment suite comparing greedy vs ILP performance:

```bash
cd src
python experiments.py
```

This will:
1. Generate or load the 2000-seat/2000-student dataset
2. Run experiments across different dataset sizes (100, 500, 1000, 2000, 5000, 10000)
3. Generate comparison plots for both Q1 and Q2
4. Save results as PNG files

### Testing Dataset Generation

To test the dataset generation functions:

```bash
cd src
python test_dataset.py
```

### Regenerating Dataset

If you want to generate a fresh dataset:

```python
from experiments import regenerate_experiment_dataset
seats_df, students_df = regenerate_experiment_dataset()
```

## File Structure

```
Seating-Optimization/
├── data/
│   ├── data.py              # Data generation utilities
│   ├── seats.csv            # Generated seats data
│   └── students.csv         # Generated students data
├── src/
│   ├── experiments.py       # Main experiment runner
│   ├── greedy.py            # Greedy algorithm implementation
│   ├── ilp.py               # ILP algorithm implementation
│   ├── test_dataset.py      # Dataset testing script
│   ├── seats.csv            # Experiment dataset (seats)
│   └── students.csv         # Experiment dataset (students)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Algorithm Details

### Greedy Algorithm
- **Complexity**: O(n log n) where n is the number of seats
- **Approach**: Iteratively selects the best available seat based on weighted criteria
- **Use case**: Fast solutions for large datasets

### ILP Algorithm
- **Complexity**: Exponential in worst case
- **Approach**: Formulates the problem as an integer linear program and solves optimally
- **Use case**: Optimal solutions for smaller datasets (≤2000 seats)

## Experiment Results

The experiments generate two types of plots:
1. **Objective Value vs Dataset Size**: Shows solution quality comparison
2. **Execution Time vs Dataset Size**: Shows performance comparison

Results are saved as:
- `experiment_results_q1.png` - Results for Q1 (minimize noise)
- `experiment_results_q2.png` - Results for Q2 (minimize noise - 0.3×brightness)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

See LICENSE file for details.
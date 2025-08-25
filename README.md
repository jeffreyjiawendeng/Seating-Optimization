## Data Generation

### Seats
- **Layout**: Configurable room/table/seat structure (default: 5 rooms × 5 tables × 4 rows × 5 seats = 2000 total seats)
- **Brightness**: Linear ramp from front (80) to back (20) with small random variation
- **Noise**: Realistic pattern where center seats are loudest, corner seats are quietest

### Students
- **Preferences**: Normal distribution around mean values
  - Brightness: mean=60, std=20 (0-100 scale)
  - Noise: mean=40, std=10 (0-100 scale, lower=quieter)
- **Groups**: 1-10 students per group, randomly sized
- **Dataset**: 2000 students organized into 368 groups across 4 sequential sets

## Algorithms

### Greedy
- Processes groups sequentially by ID
- For each group, finds available seats meeting brightness threshold
- Selects seats to minimize average noise (Q1) or noise-0.3×brightness (Q2)
- Marks assigned seats as unavailable for subsequent groups

### ILP
- Formulates seat selection as integer linear program
- Objective: minimize noise (Q1) or noise-0.3×brightness (Q2)
- Constraint: average brightness ≥ group's minimum threshold
- Solves using mathematical optimization for optimal solution

## Experiments

### Test Setup
- **Dataset sizes**: 500, 1000, 1500, 2000 students
- **Sequential loading**: Each trial builds on previous (Set 1, Sets 1+2, etc.)
- **Query types**: Q1 (minimize noise only), Q2 (minimize noise-0.3×brightness)
- **Fair comparison**: Seat availability reset between algorithms and trials

### Metrics
- **Objective value**: Average noise (Q1) or noise-0.3×brightness (Q2)
- **Execution time**: Milliseconds per group placement
- **Success rate**: Percentage of groups successfully seated
- **Variance**: Variation in objective values across groups

## Usage

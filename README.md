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



9/7/2025
TODO:
1. Global ILP
2. Pairwise Distance as part of the objective function (tables and seats)
2.5 Pairwise distance for table centroids first, then seats
2.6 Distance between seats and table centroids for refine
2.7 Seat seat distance after refine
3. Keep greedy algorithm 
4. SKETCHREFINE implement 

1. Create 100 seats (table orientation with coordinates)
2. Put 20 groups of 3 people (60 total)
3. Global ILP
4. Greedy (1 by 1)
5. SF 


# Toy Example Psudeocode
data.py 
generate_seats(num_seats, seats_per_table) -> csv file with row[seat_id, table_id, brightness, noise, seat_x, seat_y, seat_available]
generate_students(num_students, students_per_group) -> csc file with row[student_id, group_id, brightness_preference, noise_preference]

generate_seats() should enumerate seat_id and table_id, generate seat_x and seat_y according to table layout, 
  and generate brightness on ~N(\alpha * dist(seat_x, seat_y, room_x, room_y), 15) and generate noise on ~N(\beta - \alpha * (seat_x, seat_y, room_x, room_y), 15)

brightness increases from center, noise decreases from center


generate_students() should enumerate student_id and group_id, generate brightness preferences on ~N(60, 15) and noise preferences on ~N(40, 15)

calculate_table_statistics() -> csv file with row[table_id, average_brightness, average_noise, table_x, table_y]


ilp.py 
calculate_package_query() 

global ilp calculates the best possible solution 

-not online algorithm
-takes a long time
+optimal

greedy.py
calculate_package_query()

greedy calculates the best local solution
-not optimal
+online
+takes less time

sketchrefine.py
calculate_package_query()

+online
+takes less time
+approximately optimal

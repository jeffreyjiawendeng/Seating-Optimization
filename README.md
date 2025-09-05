TODO:
a. Naive ILP
  1. Filter constraints over seats
  2. Order seats by objective
  3. Dynamic programming to find the table with the best seats
b. Greedy
  1. Order table averages by objective
  2. Greedily select the table with the best seat
  3. If infeasible, select the table with the next best seat
c. SKETCHREFINE
  1. Offline partition of seats into tables
  2. Calculate representative seats for each table
  3. Query over tables with repeat=k
  4. Use ILP over representatives to get a sketch solution
  5. For each tuple of the sketch solution we are going to replace it with their actual seats in a Refine style. 



Q1 Minimize avg Noise (goal is to create an unbroken chain )
  avg brightness > 15
  k = 4 -- four people

brightness | noise
Top block: 

T1 (left) (? | 20)                 T2 (right) (? | 11.75)
r0: 70|11.5  70|11.5     ||       20|12  20|10  10|11  
r1: 58|26  58|21         ||              15|14 
--------------------------------------------------------------------------------
Bottom block: 

T3 (left)  (? | 11.7)                T4 (right) ( ? | 11.5)
r0: 70|12   62|11.5    ||           
r1:         62|11.5    ||   62|11.5  62|11.5  


11.75

sketch solution 2 x T3 + 2 x T4

PQ1 
  minimize avg Noise (goal is to create an unbroken chain) 
  avg brightness > 15
  avg cleaniness > 10
  k = 4 -- four people
  (not for now) closeness contraint 

meta table ILP

t1  (? | 20)    REPEAT k = 4
t2  (? | 11.75) REPEAT k = 4
t3  (? | 11.7)  REPEAT k = 4      => ILP (t1,t2,t3,t4) => solution 2 x T3 + 2 x T4   (SKETCH SOLUTION) =>   REPEAT 1 over the seats now
t4  (? | 11.5)  REPEAT k = 4

REFINE stage 1: We are going to replace the first tuple of the sketch solution with the actual data + the representatives of the rest tuples of the sketch solution

T3 (left)                  T4 (right) ( ? | 11.5)
r0: 70|12   62|11.5        
r1:         62|11.5     

s1 70|12        REPEAT 1
s2 62|11.5      REPEAT 1          => ILP (s1,s2,s3,t4)  => solution 1 x s2 + 1 xs3 + 2 x t4
s3 62|11.5      REPEAT 1
t4 ( ? | 11.5)  REPEAT 4


REFINE stage 2: (you carry the solution from the previous refine stage)


s2 62|11.5  REPEAT 1  
s3 62|11.5  REPEAT 1 
                           => ILP (s2,s3,s4,s5)  => solution 1 x s2 + 1 xs3 + 2 x t4
s4 62|11.5  REPEAT 1  
s5 62|11.5  REPEAT 1  

for now, don't worry about being connected
later will implement connection constraint

original ILP

s1     REPEAT 1
s2     REPEAT 1 
s3     REPEAT 1
s4     REPEAT 1 
..

s100   REPEAT 1
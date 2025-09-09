# Incremental Stochastic Package Query for Group Seating

---

## 0) Framing the task precisely

We must seat arriving **groups** of size \(k_g\) into **available seats** \(i \in \mathcal{S}\). Every seat carries attributes—**brightness** \(b_i\), **noise** \(n_i\), and integer coordinates \((x_i,y_i)\)—and every group brings a **hard** requirement on the package average brightness:
\[
\frac{1}{k_g}\sum_{i\in P_g} b_i \;\ge\; B_{\min,g}.
\]
The planner should also keep the package spatially compact and acoustically comfortable. The compactness notion must be machine-checkable and scale to thousands of seats; the comfort must be comparable across runs. Those requirements push toward **Manhattan distance** for geometry and a **linear, pairwise** compactness term; they also push brightness into a **constraint** and noise into the **objective**, so the solver can trade comfort against compactness while never violating a group’s hard need.

All methods must evaluate on the **same world** to compare fairly: identical layout, identical seat attributes, identical arriving groups. That constraint shapes every design choice below.

---

## 1) The world that produces the right pressures

The seat map is a stitched grid; coordinates are integers \((x_i, y_i)\) and **Manhattan distance** governs proximity:
\[
d_{ij} \;=\; |x_i - x_j| + |y_i - y_j|.
\]
This metric matches aisles/rows, keeps triangle inequalities tight in a grid, and—crucially—keeps pair neighborhoods finite and countable so we can bound ILP size analytically.

Brightness is a **front-to-back ramp** with small jitter:
\[
b_i \;=\; \theta_b^{\text{front}} - \gamma_b\,y_i + \varepsilon_i^{(b)},\quad \varepsilon_i^{(b)}\sim \mathcal{N}(0,\sigma_b^2),\quad b_i\in[0,100].
\]
This induces **structured scarcity** at the front. Without that scarcity the planner never faces a meaningful intertemporal trade-off and future-aware methods collapse to myopic ones.

Noise is a **central hill** with jitter:
\[
n_i \;=\; \theta_n + \alpha_n\,\phi_{\text{row}}(y_i)+\beta_n\,\phi_{\text{col}}(x_i)+\varepsilon_i^{(n)},\quad \varepsilon_i^{(n)}\sim \mathcal{N}(0,\sigma_n^2).
\]
The hill reflects busier central lanes and creates a differentiated acoustic field. A flat field would trivialize the comfort term; an overly erratic field would wash out structure and increase solver variance. A smooth hill with small noise preserves signal while avoiding degeneracies.

Students draw preferences independently,
\[
B_s \sim \text{clip}\big(\mathcal{N}(\mu_B,\sigma_B^2),0,100\big),\qquad
N_s \sim \text{clip}\big(\mathcal{N}(\mu_N,\sigma_N^2),0,100\big),
\]
and are partitioned into groups \(g\) with sizes \(k_g\in\{2,\dots,6\}\). For each group we compute the **binding** brightness:
\[
B_{\min,g} \;=\; \min_{s\in g} B_s.
\]
Using the minimum encodes the non-negotiable member’s need; moving it into the objective would allow the solver to trade it away, which violates the premise that some constraints are not for sale.

All randomness uses fixed seeds, split by source (layout jitter, preferences, grouping, scenarios), so reseeding one component does not silently alter the others. That separation makes ablations and bug hunts tractable.

---

## 2) Compactness as an objective and the math that keeps it linear

Compactness is a **soft priority**, not a hard law—groups often accept a little extra span for significantly better comfort. Hence it belongs in the objective. The natural compactness score is the sum of pairwise distances in the package,
\[
\sum_{\{i,j\}\subset P_g} d_{ij}.
\]
To keep linearity we lift the product \(x_i x_j\) into binary activations \(y_{ij}\) with the standard hull:
\[
y_{ij}\le x_i,\quad y_{ij}\le x_j,\quad y_{ij}\ge x_i+x_j-1,\quad x_i,y_{ij}\in\{0,1\}.
\]
Then \(\sum d_{ij}y_{ij}\) equals \(\sum d_{ij}x_i x_j\) at integral solutions and upper bounds it at fractional ones; optimal integer solutions respect the intended semantics without big-\(M\) constants.

We cap the pair neighborhood at radius \(D_{\max}\),
\[
\mathcal{P} \;=\; \{(i,j)\,:\,i<j,\ d_{ij}\le D_{\max}\},
\]
because marginal signal beyond 2–3 grid steps is weak while variable count grows. On an \(L^1\) grid the number of neighbors within distance \(D_{\max}\) is \(2D_{\max}(D_{\max}{+}1)\); thus
\[
|\mathcal{P}| \;\le\; M\,D_{\max}(D_{\max}{+}1)
\]
for \(M\) available seats. This bound matters when sizing the ILP; it converts hand-wavy “should be fine” into a concrete growth rate.

---

## 3) The deterministic per-group ILP (myopic baseline)

For an arriving group with size \(k\) and floor \(B_{\min}\), the per-group problem is
\[
\begin{aligned}
\min_{x,y}\quad & \sum_i n_i x_i \;+\; \lambda \sum_{(i,j)\in\mathcal{P}} d_{ij} y_{ij} \\
\text{s.t.}\quad & \sum_i x_i = k, \\
& \sum_i b_i x_i \ge k\,B_{\min}, \\
& y_{ij}\le x_i,\ y_{ij}\le x_j,\ y_{ij}\ge x_i+x_j-1,\ \forall(i,j)\in\mathcal{P}, \\
& x_i\in\{0,1\},\ y_{ij}\in\{0,1\}.
\end{aligned}\tag{PG}
\]
Averages are equivalent to sums when \(k\) is fixed, so we keep sums and absorb scaling into \(\lambda\). The problem is a small binary ILP with a knapsack-like brightness constraint and a modest quadratic-to-linear compactness term. This model is the clean **optimal-now** baseline. Any future-aware policy should only beat it when scarcity is real.

---

## 4) The greedy heuristic that mirrors the objective

Greedy clarifies local structure and produces warm starts. It seeds at the lowest-noise seat and adds one seat at a time minimizing
\[
\Delta(j) \;=\; n_j + \lambda \sum_{i\in P} d_{ij}.
\]
Before accepting \(j\), it checks **forward brightness feasibility**:
\[
\left(\sum_{i\in P} b_i\right) + \sum_{\ell=1}^{k-|P|} b_\ell^{\downarrow} \;\ge\; k\,B_{\min},
\]
where \(b_\ell^{\downarrow}\) are the top brightness values among remaining seats. This guard is a necessary condition for feasibility; without it, greedy can trap itself by spending budget on dim seats early. The check is fast (partial sums of a heap) and mirrors the classic feasibility test in cardinality-constrained knapsacks.

Greedy will miss globally better **clusters** when a single exceptionally quiet seat is far from the rest; that failure mode is expected and diagnostic rather than a bug.

---

## 5) The offline gold standard (global ILP) and making it always evaluable

To benchmark online policies we solve a **single** ILP for all groups:
\[
\begin{aligned}
\min\quad & \sum_{g=1}^G\Big(\sum_i n_i x_{i,g} + \lambda \sum_{(i,j)\in\mathcal{P}} d_{ij} y_{ij,g}\Big) \\
\text{s.t.}\quad & \sum_i x_{i,g} = k_g,\quad \sum_i b_i x_{i,g} \ge k_g B_{\min,g},\ \forall g,\\
& \sum_{g=1}^G x_{i,g} \le 1,\ \forall i, \\
& y_{ij,g}\le x_{i,g},\ y_{ij,g}\le x_{j,g},\ y_{ij,g}\ge x_{i,g}+x_{j,g}-1,\ \forall g,(i,j)\in\mathcal{P},\\
& x_{i,g},y_{ij,g}\in\{0,1\}.
\end{aligned}\tag{GLOBAL}
\]
This “oracle” knows the full sequence and enforces **no double-booking** exactly. In realistic instances the raw data may be infeasible (too many bright-needy groups versus bright capacity). We introduce a **brightness cap** \(C\) and monotonically relax each group’s requirement to \(B_{\min,g}^{\text{cap}}=\min(B_{\min,g},C)\) until (GLOBAL) is feasible; we never go below a floor \(C_{\text{floor}}\) chosen so that
\[
\sum_{g=1}^G k_g\,C_{\text{floor}} \;\le\; \sum_{i} b_i.
\]
Feasibility is monotone in \(C\); the search terminates; the cap is logged; and the **same** capped sequence is used for all online methods. This guarantees a gold baseline \(\mathcal{J}^\star\) in every run and keeps comparisons honest.

---

## 6) Future awareness without losing linearity: SR-PQ

The myopic planner can inadvertently consume resources the near future probably needs—here, **bright seats**. Full multistage stochastic control is intractable at our scale; chance constraints complicate the solver stack. The practical alternative is to encode **expected scarcity** as a linear surcharge on using scarce items now.

We model the next \(h\) arrivals by a **scenario bank**. A scenario \(\omega\) is a sequence \(\big((t^{(1)},k^{(1)}),\dots,(t^{(h)},k^{(h)})\big)\), where each type \(t\in\{\text{bright},\text{quiet},\text{balanced}\}\) has a bright fraction \(f_t\). Types are drawn from a categorical mixture \(\pi\) with a Dirichlet prior \(\alpha\); after observing data, the posterior mean \(\hat{\pi}\) yields **passive** scenario weights
\[
w_\omega^{\text{pass}} \propto \prod_{r=1}^h \hat{\pi}_{t^{(r)}}.
\]
We optionally **tilt** toward scenarios that do not overload bright demand relative to current bright supply \(V_{\text{bright}}\):
\[
w_\omega^{\text{opt}} \propto w_\omega^{\text{pass}} \cdot \exp\!\big(-\tau\cdot \text{Shortage}(\omega)\big),\quad
\text{Shortage}(\omega)=\max\{0, D_{\text{bright}}(\omega)-V_{\text{bright}}\},
\]
with \(D_{\text{bright}}(\omega)=\sum_{r=1}^{h} f_{t^{(r)}} k^{(r)}\), and \(\tau\ge 0\). We normalize weights to sum to one and compute expected horizon demand,
\[
\mathbb{E}[D_{\text{bright}}] = \sum_\omega w_\omega\,D_{\text{bright}}(\omega).
\]
A dimensionless **scarcity ratio** follows,
\[
\sigma \;=\; \max\!\left(0,\ \frac{\mathbb{E}[D_{\text{bright}}] - V_{\text{bright}}}{V_{\text{bright}}+\epsilon}\right),
\]
which increases smoothly from zero as expected demand outstrips supply. We define a per-seat penalty
\[
p_i \;=\; \begin{cases}
\sigma,& b_i \ge C_{\text{bright}},\\
0,& \text{otherwise},
\end{cases}
\]
and replace \(n_i\) by \(c_i=n_i+\mu p_i\) in (PG). The **SR-PQ** objective is therefore
\[
\min \sum_i (n_i+\mu p_i) x_i + \lambda \sum_{(i,j)\in\mathcal{P}} d_{ij} y_{ij}
\]
with the same constraints. The parameter \(\mu\) sets the *marginal price* of consuming a bright seat under scarcity. This **keeps linearity**, **remains interpretable**, and **reacts online** as availability and observations change.

> Economic intuition: \(\mu p_i\) acts like a shadow price for a scarce capacity; it approximates the costate you would obtain from a dynamic program, but without solving one.

---

## 7) Units and calibration so the knobs are meaningful

Noise is in points; distance is in grid steps. To make \(\lambda\) legible, scale distances by a typical neighborhood mean
\[
\bar{d} \;=\; \mathbb{E}[d_{ij}\mid(i,j)\in\mathcal{P}],
\]
or, equivalently, choose \(\lambda\) such that increasing average pair distance by one step costs roughly \(\delta_n\) noise units (e.g., 1–2). That mapping makes plots interpretable.

The scarcity ratio \(\sigma\) lives in \([0,1]\) in typical regimes. Choose \(\mu\) so \(\mu\sigma\) is on the same numerical scale as \(n_i\); values in \([0.3,2]\) cover most rooms. If SR-PQ behaves like Myopic, either scarcity is weak in that world or \(\mu\) is too small.

The pair radius \(D_{\max}\) controls a quadratic factor in ILP size; values \(2\) or \(3\) capture compactness well on a grid while keeping \(|\mathcal{P}|\) manageable.

The horizon \(h\) and scenario count \(M\) smooth \(\sigma\); small \(h\) (e.g., 5) with a few hundred scenarios stabilizes behavior without heavy runtime.

---

## 8) Complexity facts that drove engineering choices

Per group with \(M\) available seats, the ILP has \(M\) binary \(x_i\) variables and \(\mathcal{O}(M D_{\max}^2)\) binary \(y_{ij}\). Constraints scale similarly. That count fits comfortably in modern MILP solvers for \(M\) in the low thousands when \(D_{\max}\in\{2,3\}\). The global ILP multiplies by \(G\) and is intentionally reserved for small instances to compute \(\mathcal{J}^\star\) reliably.

We pre-filter \(\mathcal{P}\) by availability so pairs attached to occupied seats never enter the model. That reduces re-solve size as the room fills.

---

## 9) Experimental protocol that preserves comparability

One world is generated and **frozen** (layout, attributes, groups). If the global ILP needs a brightness cap to be feasible, that cap is applied to **all** online methods. All policies use the same \(\lambda\), the same \(D_{\max}\), and the same pair set \(\mathcal{P}\). Online methods process groups in order and update availability between groups. This protocol isolates **policy quality** from world idiosyncrasies.

Metrics are defined in closed form and reported consistently:

- Success rate:
  \[
  S \;=\; \frac{1}{G}\sum_{g=1}^G \mathbb{1}\{P_g\ \text{found}\}.
  \]
- Average noise among seated groups \(\mathcal{G}_{\text{ok}}\):
  \[
  \overline{N} \;=\; \frac{1}{|\mathcal{G}_{\text{ok}}|}\sum_{g\in\mathcal{G}_{\text{ok}}} \frac{1}{k_g}\sum_{i\in P_g} n_i.
  \]
- Average pair distance:
  \[
  \overline{D} \;=\; \frac{1}{|\mathcal{G}_{\text{ok}}|}\sum_{g\in\mathcal{G}_{\text{ok}}} \frac{2}{k_g(k_g-1)}\sum_{\{i,j\}\subset P_g} d_{ij}.
  \]
- Cumulative objective:
  \[
  \mathcal{J} \;=\; \sum_{g\in\mathcal{G}_{\text{ok}}}\left(\frac{1}{k_g}\sum_{i\in P_g} n_i + \lambda \frac{2}{k_g(k_g-1)}\sum_{\{i,j\}\subset P_g} d_{ij}\right).
  \]
- Regret vs. gold:
  \[
  \mathrm{Regret} \;=\; \mathcal{J} - \mathcal{J}^\star.
  \]

Plotting the partial sums \(\sum_{g\le t} J_g\) against \(\mathcal{J}^\star\) reveals exactly when scarcity matters: SR-PQ should diverge favorably from Myopic near those epochs.

---

## 10) Edge cases handled up front

If a group’s \(k_g\) exceeds remaining capacity, or the brightness constraint is infeasible even with best remaining seats, the policy must decline that group; the success indicator records this. If \(D_{\max}=0\), compactness is disabled and we recover pure noise minimization—useful as a sanity check. Ties in costs are broken deterministically to ensure reproducibility. All random number generators are seeded and logged.

---

## 11) Alternatives we considered and the reason they’re not in v1

A **hard radius** compactness constraint forbids near-optimal solutions sitting just outside the circle; encoding compactness in the objective lets the solver trade distance for comfort when beneficial. A **noise ceiling** as a hard constraint often creates brittle infeasibility where a tiny increase in noise would have yielded a compact, bright solution; keeping noise in the objective avoids that cliff. **Euclidean distance** breaks the clean integer geometry and adds little on a rectilinear room; \(L^1\) keeps pair neighborhoods analyzable. **Chance constraints** on future brightness preservation are elegant but would complicate the solver and the data needs; the expectation-based surcharge captures the primary effect in a linear form and is easy to tune.

---

## 12) A concrete counterexample that demonstrates Greedy’s limitation

Seats \(\{1,2,3,4\}\), with noise \(n_1=1\), \(n_2=n_3=n_4=5\). Distances satisfy \(d_{23}=d_{24}=d_{34}=1\) and \(d_{1j}=5\) for \(j\in\{2,3,4\}\). Let \(k=3\), \(\lambda=1\), and assume brightness is slack.

- Greedy picks seat 1, then pays \(5\) distance twice to attach two more seats: total \(1 + (5+5) + (5+5)=21\).
- The myopic ILP picks the tight cluster \(\{2,3,4\}\): total \(5+5+5 + (1+1+1)=18\).

This example demonstrates that a locally outstanding seat can mislead a purely incremental policy when pairwise structure dominates.

---

## 13) The two PaQL-style queries that the code implements

**Deterministic (Greedy & Myopic)**


SELECT  PACKAGE(s) AS P
FROM    seats AS s
WHERE   s.Seat_Available = TRUE
SUCH THAT
COUNT(P) = :k
AND AVG(P.Brightness) >= :B_min
MINIMIZE
AVG(P.Noise) + λ * AVGPAIR(Distance(P))


**Stochastic Rolling (SR-PQ)**

SELECT  PACKAGE(s) AS P
FROM    seats AS s
WHERE   s.Seat_Available = TRUE
SUCH THAT
COUNT(P) = :k
AND AVG(P.Brightness) >= :B_min
MINIMIZE
AVG(P.Noise) + λ * AVGPAIR(Distance(P))
+ μ * EXPECTED_ω [ SUM_{s∈P} ScarcityPenalty(s | ω) ]


They differ by a single expectation term. That design choice keeps the surface area small and the behavior easy to reason about.

---

## 14) Summary of the reasoning chain

- Seat geometry on an integer grid with \(L^1\) distance makes compactness computable and controllable.
- Brightness scarcity must exist in the world; a front-to-back ramp with jitter creates it.
- The **minimum** group brightness belongs in a **constraint** to protect the least flexible member.
- Compactness belongs in the **objective** so the solver can trade it against noise when that yields better packages.
- A per-group ILP is the right deterministic baseline; Greedy provides intuition and warm starts but cannot see across clusters.
- A global ILP with a monotone brightness cap always yields a gold baseline and keeps comparisons honest.
- Expected scarcity, encoded as a **linear surcharge** on bright seats under a scenario bank, adds future awareness without breaking linearity or interpretability.
- Knobs \(\lambda\) and \(\mu\) are calibrated to human-legible units so plots are read correctly and decisions can be debated quantitatively.
- The experiment harness fixes the world and metrics, converting anecdotes into measurable differences.

The result is a compact system that makes the same decisions you would make if you tracked capacity in your head: sit people comfortably, keep them together, and don’t burn the few bright seats today when you have good reason to expect someone will truly need them tomorrow.

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
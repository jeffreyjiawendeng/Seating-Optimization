# seating_opt/__init__.py

# Import key functions for easy access
from .experiments import run_all
from .data_gen import generate_seats
from .greedy import greedy_pairwise
from .ilp_solvers import solve_global_pair_ilp, solve_group_pair_ilp
from .sketchrefine import sketchrefine_solver

__all__ = [
    "data_gen",
    "distance", 
    "experiments",
    "greedy",
    "ilp_solvers",
    "sketchrefine",
    "utils",
    # Key functions
    "run_all",
    "generate_seats",
    "greedy_pairwise",
    "solve_global_pair_ilp",
    "solve_group_pair_ilp",
    "sketchrefine_solver"
]
# Job Shop Problem Solvers: Experiment Results Guide

## Overview


1. **CP-SAT Solver**: A constraint programming solver using Google's OR-Tools CP-SAT
2. **Hybrid Solver with Limiter**: A hybrid approach combining genetic algorithm and constraint programming with limiting mechanisms
3. **GA Solver**: A genetic algorithm-only approach 
4. **Hybrid Solver**: A hybrid approach combining genetic algorithm and constraint programming without limiting mechanisms

## CSV File Structure

Each row in the CSV file represents the results of running all four solvers on a single JSP instance. The columns provide performance metrics for each solver.

### Column Descriptions

| Column Name | Description |
|-------------|-------------|
| `instance` | Path to the JSP instance file that was processed |
| `cp_opt_make` | Makespan (total completion time) found by the CP-SAT solver |
| `cp_opt_time` | Execution time (in seconds) for the CP-SAT solver |
| `cp_opt_memory` | Memory usage (in bytes) for the CP-SAT solver |
| `cp_opt_status` | Status code returned by the CP-SAT solver (4 = OPTIMAL) |
| `hy_lim_make` | Makespan found by the Hybrid solver with limiter |
| `hy_lim_tot_time` | Total execution time (in seconds) for the Hybrid solver with limiter |
| `hy_lim_tot_memory` | Memory usage (in bytes) for the Hybrid solver with limiter |
| `diff_hy_lim_cp_opt` | Relative difference between Hybrid limiter makespan and CP-SAT optimal makespan ((hy_lim_make - cp_opt_make) / cp_opt_make) |
| `ga_make` | Makespan found by the GA-only solver |
| `ga_time` | Execution time (in seconds) for the GA-only solver |
| `ga_memory` | Memory usage (in bytes) for the GA-only solver |
| `hy_make` | Makespan found by the Hybrid solver (without limiter) |
| `hy_tot_time` | Total execution time (in seconds) for the Hybrid solver |
| `hy_tot_memory` | Memory usage (in bytes) for the Hybrid solver |
| `diff_hy_cp_opt` | Relative difference between Hybrid makespan and CP-SAT optimal makespan ((hy_make - cp_opt_make) / cp_opt_make) |


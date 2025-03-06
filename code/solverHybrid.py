import collections
import numpy as np
from typing import List, Dict, Tuple 
import time             # Time tracking
import tracemalloc      # Memory tracking
import argparse
import random

# from solverICPlimit import *    # JSP solver using OR-Tools CP-SAT 
from solverICP import *          # JSP solver using OR-Tools CP-SAT
from solverGA import *          # JSP solver using Genetic Algorithm
from utils import *

'''
Hybrid JSP solver using OR-Tools CP-SAT and Genetic Algorithm with machine-based representation

The idea is to take the a valid (not necessarily optimal) solution found by the CP-SAT solver after a certain time limit,
and use it as the initial population for the Genetic Algorithm solver.
This way, the GA solver can start from a good solution and improve it further.
'''

class HybridSolver:
    def __init__(self, instance, time_budget = 2000):
        self.instance = instance
        
        self.cp_solver = ICPSolver(instance)    # CP-SAT solver
        self.cp_solver.solver.parameters.max_time_in_seconds = time_budget * 0.3 # 30% of time budget
        self.cp_solver.solver.parameters.random_seed = 10

        self.ga_solver = GASolver(instance, seed=123)     # GA solver
        self.ga_solver.max_time = time_budget * 0.7 # 70% of time budget
        
    def create_initial_population(self, base_chromosome, pop_size=50, num_copies=10):
        '''Create initial population for GA solver
        
        args:
            base_chromosome : list of (job_id, task_id) tuples (from cp-sat solution)
            pop_size : int, total population size
            num_copies : int, number of exact copies of the base chromosome to add to the population
        '''
        initial_population = []
        
        # Add exact copies of the original solution
        for _ in range(num_copies):
            initial_population.append(base_chromosome.copy())
            
        # Add gradually mutated versions for the rest of the population
        for i in range(pop_size - num_copies):
            mutated = base_chromosome.copy()
            
            # Use a progressive mutation rate
            mutation_rate = min(0.05, 0.01 + (i / pop_size * 0.04))  # 1% to 5% mutation
            mutation_count = max(1, int(len(mutated) * mutation_rate))
            
            # Apply mutations that maintain precedence constraints
            for _ in range(mutation_count):
                max_attempts = 10
                for attempt in range(max_attempts):
                    # Select two random positions to swap
                    pos1, pos2 = random.sample(range(len(mutated)), 2)
                    
                    # Get the tasks at these positions
                    task1 = mutated[pos1]
                    task2 = mutated[pos2]
                    
                    # Skip if they're from the same job (maintain precedence)
                    if task1[0] == task2[0]:
                        continue
                    
                    # Try the swap
                    mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
                    
                    # Check if still valid
                    if self.ga_solver.is_valid_chromosome(mutated):
                        break
                    else:
                        # Undo the swap and try again
                        mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
            
            # Add to population even if no valid mutation was found
            # (in that case, it's just another copy of the original)
            initial_population.append(mutated)
        
        
    def solve(self):
        '''Solve JSP instance using a hybrid approach'''
        
        tracemalloc.start() # Start memory tracking
        
        # ----------------- Step 1 : CP-SAT solver -----------------
        print(f"\nSolving using CP-SAT solver...")
        cp_start_time = time.time() # Start time
        snapshot1 = tracemalloc.take_snapshot() # Memory snapshot
        
        # Solve using CP-SAT solver
        schedule_icp, makespan_icp, solver_icp, status_icp = self.cp_solver.solve() # || CP-SAT SOLVER ||
        
        snapshot2 = tracemalloc.take_snapshot() # Memory snapshot
        cp_end_time = time.time() # End time
        
        cp_time = cp_end_time - cp_start_time # Time taken  
        cp_stats = snapshot2.compare_to(snapshot1, 'lineno')
        cp_memory = sum(stat.size_diff for stat in cp_stats) # Memory usage
        
        print(f"CP-SAT solver found a solution with makespan {makespan_icp} in {cp_time:.2f} seconds")
        print(f"Memory usage: {cp_memory / 1024 / 1024:.2f} MB")
        
        if status_icp == cp_model.OPTIMAL:
            print("\nOptimal solution found by CP-SAT solver!")
            # print("\n Exiting...")
            # return None, 0, makespan_icp, cp_time, cp_memory
        
        if status_icp == cp_model.UNKNOWN:
            print("\nCP-SAT solver could not find a solution. Exiting...")
            return None, 0, 0, 0, 0
        
        # ----------------- Step 2 : GA solver -----------------
        # Use the solution found by CP-SAT solver as initial population for GA solver
        print(f"\nSolving using GA solver...")
        
        # Convert schedule to (job_id, task_id) tuple chromosome format
        
        # First, collect all tasks from the schedule
        all_tasks = []
        for machine in schedule_icp.values():
            all_tasks.extend(machine)
        
        # Sort tasks by start time to get the execution sequence
        all_tasks.sort(key=lambda x: x.start_time)
        
        # Create chromosome with (job_id, task_id) tuples
        base_chromosome = [(task.job_id, task.task_id) for task in all_tasks]
        
        # Verify that the base chromosome is valid
        if not self.ga_solver.is_valid_chromosome(base_chromosome):
            print("Warning: Base chromosome from CP-SAT solution is not valid. Repairing...")
            base_chromosome = self.ga_solver.repair_chromosome(base_chromosome)
        
        # Create initial population for GA solver
        # note if you want to clone all the chromosomes in the initial population, you can use set num_copies = pop_size
        initial_population = self.create_initial_population(base_chromosome, pop_size=50, num_copies=10)
        args = {'initial_population': initial_population}

        ga_start_time = time.time() # Start time
        snapshot3 = tracemalloc.take_snapshot() # Memory snapshot
        
        # Solve using GA solver
        schedule_ga, makespan_ga = self.ga_solver.solve(args) # || GA SOLVER ||
        
        snapshot4 = tracemalloc.take_snapshot() # Memory snapshot
        ga_end_time = time.time() # End time
        
        ga_time = ga_end_time - ga_start_time # Time taken
        ga_stats = snapshot4.compare_to(snapshot3, 'lineno')
        ga_memory = sum(stat.size_diff for stat in ga_stats) # Memory usage
        
        print(f"GA solver found a solution with makespan {makespan_ga} in {ga_time:.2f} seconds")
        print(f"Memory usage: {ga_memory / 1024 / 1024:.2f} MB")
        
        # ----------------- Results ----------------
        
        total_time = cp_time + ga_time
        total_memory = (cp_memory + ga_memory) / 1024 / 1024
        
        return schedule_ga, makespan_ga, makespan_icp, total_time, total_memory
    
# ----------------- Main -----------------

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Job Shop Problem Solver using CP-SAT')
    parser.add_argument('instance_file', type=str, help='Path to the instance file')
    parser.add_argument('--time_limit', type=int, default=60, 
                        help='Time limit in seconds (default: 60)')
    parser.add_argument('--output', type=str, default='scheduleHybrid',
                        help='Base name for output files (default: scheduleHybrid)')
    
    args = parser.parse_args()
    
    # Load and validate instance
    print(f"Loading instance from {args.instance_file}...")
    instance = load_instance(args.instance_file)
    print(f"Instance loaded: {instance.num_jobs} jobs, {instance.num_machines} machines")
    
    # Initialize and run solver
    solver = HybridSolver(instance, time_budget=args.time_limit)
    schedule, makespan_ga, makespan_icp, tot_time, tot_memory = solver.solve()
    
    print(f"----------------------------------")
    print(f"\nMakespan: {makespan_ga} in {tot_time:.2f} seconds")
    print(f"----------------------------------")
    
    if makespan_icp > 0 and makespan_ga > 0:
        improvement = (makespan_icp - makespan_ga) / makespan_icp * 100
        print(f"Improvement: {improvement:.2f}%")
    
    print(f"Total time: {tot_time:.2f} seconds")
    print(f"Total memory: {tot_memory:.2f} MB")
    
    # Log schedule to file
    # log_schedule(schedule, makespan_ga, f'{args.output}.txt')
    
    # Visualize and save schedule
    # visualize_schedule(schedule, makespan_ga, instance, f'{args.output}.png')
    
if __name__ == '__main__':
    main()
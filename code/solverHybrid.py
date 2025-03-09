import collections
import numpy as np
from typing import List, Dict, Tuple 
import argparse
import random

from solverICP import *             # JSP solver using OR-Tools CP-SAT
from solverICP_limiter import *     # JSP solver using OR-Tools CP-SAT with solution limit
from solverGA import *              # JSP solver using Genetic Algorithm
from utils import *

'''
Hybrid JSP solver using OR-Tools CP-SAT and Genetic Algorithm with machine-based representation

The idea is to take the a valid (not necessarily optimal) solution found by the CP-SAT solver after a certain time limit,
and use it as the initial population for the Genetic Algorithm solver.
This way, the GA solver can start from a good solution and improve it further.
'''

class HybridSolver:
    def __init__(self, instance, seed = 10, use_limiter = False, time_budget = 2000, limit=1):
        self.instance = instance
        
        if use_limiter:
            self.cp_solver = ICPSolverLimiter(instance, limit)   # CP-SAT solver with solution limit
        else:
            self.cp_solver = ICPSolver(instance)   # CP-SAT solver with time limit
            self.cp_solver.solver.parameters.max_time_in_seconds = time_budget * 0.3 # 30% of time budget
        self.cp_solver.solver.parameters.random_seed = seed

        ga_time_budget = time_budget if use_limiter else time_budget * 0.7
        self.ga_solver = GASolver(instance, seed=seed, hybrid=True)     # GA solver
        self.ga_solver.max_time = ga_time_budget # 70% of time budget
        
    def create_initial_population(self, base_chromosome, pop_size=50, 
                                  num_copies=1,  num_random=1):
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
            
        # Add random chromosomes using generator
        for _ in range(num_random):
            initial_population.append(self.ga_solver.generator_chromosome(self.ga_solver.prng, None))
            
        # Add remaining population applying mutation to the base chromosome
        for i in range(pop_size - num_copies - num_random):
            mutated = base_chromosome.copy()
            
            # Mutation attempts for each chromosome (1 to 5)
            attempts = random.randint(1, 5)
            for _ in range(attempts):
                # Choose two random positions to swap
                pos1 = random.randint(0, len(mutated) - 1)
                pos2 = random.randint(0, len(mutated) - 1)
                
                # Swap
                mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
            
                # Check if still valid
                if self.ga_solver.is_valid_chromosome(mutated):
                    break
                else:
                    # Undo the swap and try again
                    mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
    
            # Add to population even if no valid mutation was found
            initial_population.append(mutated)

        return initial_population
        
    def solve(self):
        '''Solve JSP instance using a hybrid approach'''
        
        # ----------------- Step 1 : CP-SAT solver -----------------
        # print(f"\nSolving using CP-SAT solver...")
        
        # Solve using CP-SAT solver
        schedule_icp, makespan_icp, solver_icp, status_icp, time_icp, memory_icp = self.cp_solver.solve() # || CP-SAT SOLVER ||

        # print(f"CP-SAT solver found a solution with makespan {makespan_icp} in {time_icp:.2f} seconds")
        # print(f"Memory usage: {memory_icp / 1024 / 1024:.2f} MB")
        
        #TODO: check if the solution found by CP-SAT is optimal
        if status_icp == cp_model.OPTIMAL:
            # print("\nOptimal solution found by CP-SAT solver!")
            # print("\n Exiting...")
            return schedule_icp, 0, makespan_icp, time_icp, memory_icp
        
        if status_icp == cp_model.UNKNOWN:
            # print("\nCP-SAT solver could not find a solution. Exiting...")
            return None, 0, 0, 0, 0, 

        # ----------------- Step 2 : GA solver -----------------
        # Use the solution found by CP-SAT solver as initial population for GA solver
        # print(f"\nSolving using GA solver...")
        
        # First, collect all tasks from the schedule
        all_tasks = []
        for machine in schedule_icp.values():
            all_tasks.extend(machine)
        
        # Sort tasks by start time to get the execution sequence
        all_tasks.sort(key=lambda x: x.start_time)
        
        # Create chromosome as list of job_id representing the execution sequence
        base_chromosome = [task.job_id for task in all_tasks]    
        
        # Create initial population for GA solver
        initial_population = self.create_initial_population(base_chromosome, pop_size=100, num_copies=1, num_random=30)
        args = {'initial_population': initial_population}
        
        # Solve using GA solver
        self.ga_solver.max_time = self.ga_solver.max_time - time_icp # Remaining time budget
        schedule_ga, makespan_ga, time_ga, memory_ga = self.ga_solver.solve(args) # || GA SOLVER ||

        
        # print(f"GA solver found a solution with makespan {makespan_ga} in {time_ga:.2f} seconds")
        # print(f"Memory usage: {memory_ga / 1024 / 1024:.2f} MB")
        
        # ----------------- Results ----------------
        
        total_time = time_icp + time_ga
        total_memory = memory_icp + memory_ga
        
        return schedule_ga, makespan_ga, makespan_icp, total_time, total_memory
    
# ----------------- Main -----------------

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Job Shop Problem Solver using CP-SAT')
    parser.add_argument('--instance_file', type=str, help='Path to the instance file', default='../instances/ClassicBenchmark/jobshop_abz5')
    parser.add_argument('--time_limit', type=int, help='Time limit in seconds (default: 60)', default=60)
    parser.add_argument('--limit', type=int, help='Solution limit for CP-SAT solver', default=0)
    parser.add_argument('--seed', type=int, help='Random seed', default=10)
    parser.add_argument('--output', type=str, default='scheduleHybrid',
                        help='Base name for output files (default: scheduleHybrid)')
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # if the solution limit is provided, use ICP solver with solution limit
    # otherwise, use ICP solver with time limit
    use_limiter = args.limit > 0 # use solution limit
    
    # Load and validate instance
    print(f"Loading instance from {args.instance_file}...")
    instance = load_instance(args.instance_file)
    print(f"Instance loaded: {instance.num_jobs} jobs, {instance.num_machines} machines")
    
    # Initialize and run solver
    solver = HybridSolver(instance, seed = args.seed, use_limiter = use_limiter, time_budget=args.time_limit, limit=args.limit)
    schedule, makespan_ga, makespan_icp, tot_time, tot_memory = solver.solve()
    
    print(f"----------------------------------")
    print(f"\nMakespan: {makespan_ga} in {tot_time:.2f} seconds")
    print(f"----------------------------------")
    
    if makespan_icp > 0 and makespan_ga > 0:
        improvement = (makespan_icp - makespan_ga) / makespan_icp * 100
        print(f"Improvement: {improvement:.2f}%")
    
    print(f"Total time: {tot_time:.2f} seconds")
    print(f"Total memory: {(tot_memory / 1024 / 1024) :.2f} MB")
    
    # Log schedule to file
    # log_schedule(schedule, makespan_ga, f'output/{args.output}.txt')
    
    # Visualize and save schedule
    # visualize_schedule(schedule, makespan_ga, instance, f'output/{args.output}.png')
    
if __name__ == '__main__':
    main()
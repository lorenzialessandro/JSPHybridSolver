import collections
import numpy as np
from typing import List, Dict, Tuple 
import argparse
import random

from solverICP import *             # JSP solver using OR-Tools CP-SAT
from solverICP_limiter import *     # JSP solver using OR-Tools CP-SAT with solution limit
from solverICP_collect import *     # JSP solver using OR-Tools CP-SAT with solution collection
from solverGA import *              # JSP solver using Genetic Algorithm
from utils import *

'''
Hybrid JSP solver using OR-Tools CP-SAT and Genetic Algorithm with machine-based representation

The idea is to take the a valid (not necessarily optimal) solution found by the CP-SAT solver after a certain time limit,
and use it as the initial population for the Genetic Algorithm solver.
This way, the GA solver can start from a good solution and improve it further.
'''

class HybridSolver:
    def __init__(self, instance, seed = 10, use_limiter = False, use_collector = False, time_budget = 2000, limit=1):
        self.instance = instance
        self.seed = seed
        self.use_limiter = use_limiter
        self.use_collector = use_collector
        
        if self.use_limiter:
            self.cp_solver = ICPSolverLimiter(instance, limit)   # CP-SAT solver with solution limit
        elif self.use_collector:
            self.cp_solver = ICPSolverCollectorLimiter(instance, time_budget * 0.3)  # CP-SAT solver with solution collection
            self.cp_solver.solver.parameters.max_time_in_seconds = time_budget * 0.3 # 30% of time budget
        else:
            self.cp_solver = ICPSolver(instance)   # CP-SAT solver with time limit
            self.cp_solver.solver.parameters.max_time_in_seconds = time_budget * 0.3 # 30% of time budget
        self.cp_solver.solver.parameters.random_seed = seed

        ga_time_budget = time_budget if use_limiter else time_budget * 0.7
        self.ga_solver = GASolver(instance, seed=seed, hybrid=True)     # GA solver
        self.ga_solver.max_time = ga_time_budget # 70% of time budget
        
    def create_initial_population(self, base_chromosomes, pop_size=50, 
                                  num_copies=1,  num_random=1):
        '''Create initial population for GA solver
        
        args:
            base_chromosomes : list, list of the base chromosomes to copy in the initial population from CP-SAT solver solutions
            pop_size : int, total population size
            num_copies : int, number of exact copies of the base chromosome to add to the population
        '''
        initial_population = []
        
        # Add exact copies of the original solution
        for _ in range(num_copies):
            for base_chromosome in base_chromosomes:
                initial_population.append(base_chromosome.copy())
            
        # Add random chromosomes using generator
        for _ in range(num_random):
            initial_population.append(self.ga_solver.generator_chromosome(self.ga_solver.prng, None))
            
        # Add remaining population applying mutation to the base chromosome
        remaining = int((pop_size - len(base_chromosomes) - num_random)  / len(base_chromosomes))

        for i in range(remaining):
            for base_chromosome in base_chromosomes:
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

        # check if the population size is less than the required size
        if len(initial_population) < pop_size:
            remaining = pop_size - len(initial_population)
            for _ in range(remaining):
                initial_population.append(self.ga_solver.generator_chromosome(self.ga_solver.prng, None))
        # resize the population to the required size
        initial_population = initial_population[:pop_size]
        return initial_population
        
    def solve(self):
        '''Solve JSP instance using a hybrid approach'''
        
        # ----------------- Step 1 : CP-SAT solver -----------------

        if self.use_collector: 
            # Use solution collector for CP-SAT solver : collect all solutions found by CP-SAT solver and use to build initial population for GA solver
            schedule_icp, makespan_icp, solver_icp, status_icp, time_icp, memory_icp, schedules = self.cp_solver.solve() # || CP-SAT SOLVER with collector ||
        else:
            # Use CP-SAT solver with time limit
            schedule_icp, makespan_icp, solver_icp, status_icp, time_icp, memory_icp = self.cp_solver.solve() # || CP-SAT SOLVER ||
            
        if status_icp == cp_model.OPTIMAL:  # If the solution found by CP-SAT solver is optimal, return it (no need to use GA solver)
            return schedule_icp, makespan_icp, makespan_icp, time_icp, memory_icp
        
        if status_icp == cp_model.UNKNOWN: # If CP-SAT solver could not find a solution, return None
            return None, 0, 0, 0, 0
            
        # Collect chromosome(s) from the solution(s) found by CP-SAT solver
        chromosomes = []
        if self.use_collector:
            # For each solution found by CP-SAT solver, create a chromosome
            for schedule in schedules:
                all_tasks = []
                for machine in schedule.values():
                    all_tasks.extend(machine)
                all_tasks.sort(key=lambda x: x.start_time)
                chromosome = [task.job_id for task in all_tasks]
                chromosomes.append(chromosome)
        else: 
            # Use the schedule found by CP-SAT solver to create one chromosome
            all_tasks = []
            for machine in schedule_icp.values():
                all_tasks.extend(machine)
            all_tasks.sort(key=lambda x: x.start_time)
            base_chromosome = [task.job_id for task in all_tasks]    
            chromosomes.append(base_chromosome)
        
        # Create initial population for GA solver
        #TODO: move the pop_size, num_copies, num_random to the arguments of the function
        initial_population = self.create_initial_population(chromosomes, pop_size=100, num_copies=1, num_random=30)
        args = {'initial_population': initial_population}

        # ----------------- Step 2 : GA solver -----------------  
    
        # Solve using GA solver
        self.ga_solver.max_time = self.ga_solver.max_time - time_icp # Set remaining time budget for GA solver (TODO: add overhead for creating initial population)
        schedule_ga, makespan_ga, time_ga, memory_ga = self.ga_solver.solve(args) # || GA SOLVER ||
        
        # Results
        total_time = time_icp + time_ga
        total_memory = memory_icp + memory_ga
        
        return schedule_ga, makespan_ga, makespan_icp, total_time, total_memory
    

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Job Shop Problem Solver using CP-SAT')
    parser.add_argument('--instance_file', type=str, help='Path to the instance file', default='../instances/ClassicBenchmark/jobshop_abz5')
    parser.add_argument('--time_limit', type=int, help='Time limit in seconds (default: 60)', default=60)
    parser.add_argument('--limit', type=int, help='Solution limit for CP-SAT solver', default=0)
    parser.add_argument('--collector', action='store_true', help='Use solution collector for CP-SAT solver', default=False)
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
    solver = HybridSolver(instance, seed = args.seed, use_limiter = use_limiter, use_collector = args.collector, time_budget=args.time_limit, limit=args.limit)
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
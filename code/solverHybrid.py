import collections
import numpy as np
from typing import List, Dict, Tuple 

from solverICPlimit import *    # JSP solver using OR-Tools CP-SAT 
from solverGA import *          # JSP solver using Genetic Algorithm
from utils import *

'''
Hybrid JSP solver using OR-Tools CP-SAT and Genetic Algorithm

The idea is to take the first valid (not necessarily optimal) solution found by the CP-SAT solver and use it as the initial population for the Genetic Algorithm solver.
This way, the GA solver can start from a good solution and improve it further.
'''

class HybridSolver:
    def __init__(self, instance):
        self.instance = instance
        self.cp_solver = ICPSolver(instance)    # CP-SAT solver
        self.ga_solver = GASolver(instance, seed=1)     # GA solver
        
    def solve(self):
        '''Solve JSP instance using a hybrid approach'''
        
        # Step 1 : Solve using CP-SAT solver with a limit of first solution found
        schedule_icp, makespan_icp, solver_icp, status_icp = self.cp_solver.solve()
        print(f"CP-SAT solver found a solution with makespan {makespan_icp}")
        
        
        # Step 2 : Use the solution found by CP-SAT solver as initial population for GA solver
        # convert schedule to chromosome format
        all_tasks = []
        for machine in schedule_icp.values():
            all_tasks.extend(machine)
        all_tasks.sort(key=lambda x: x.start_time)
        # Create chromosome by job order
        chromosome = [task.job_id for task in all_tasks]
        
        # Solve using GA solver with initial population
        args = {'initial_population': [chromosome]}
        schedule_ga, makespan_ga = self.ga_solver.solve(args)
        print(f"GA solver found a solution with makespan {makespan_ga}")
        
        return schedule_ga, makespan_ga
    
# ----------------- Main -----------------

INSTANCE = '../instances/long-js-600000-100-10000-1.data'
INSTANCE = '../instances/abz5.data'
INSTANCE = '../instances/test_s15'

def main():
    # Load and validate instance
    print(f"Loading instance from {INSTANCE}...")
    instance = load_instance(INSTANCE)
    print(f"Instance loaded: {instance.num_jobs} jobs, {instance.num_machines} machines")
    
    # Initialize and run solver
    solver = HybridSolver(instance)
    schedule, makespan = solver.solve()
    
    print(f"\nMakespan: {makespan}")
    
    
if __name__ == '__main__':
    main() 
import collections
import numpy as np
from typing import List, Dict, Tuple 
from ortools.sat.python import cp_model
import time             # Time tracking
import tracemalloc      # Memory tracking
import argparse

from utils import *

class ICPSolver:
    def __init__(self, instance):
        self.instance = instance
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
    def solve(self):
        '''Solve JSP instance using OR-Tools (CP-SAT)'''
        
        # Track time and memory usage
        start_time_t = time.time()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        
        max_time_in_seconds = self.solver.parameters.max_time_in_seconds # Time limit
        
        # Calculate reasonable horizon
        horizon = sum(sum(task[1] for task in job) for job in self.instance.tasks)
        
        # Create variables
        starts = {}         # (job_id, task_id) -> start_time_var
        ends = {}           # (job_id, task_id) -> end_time_var
        intervals = {}      # (job_id, task_id) -> interval_var
        
        # Create job intervals and add to the corresponding machine lists
        for job_id, job in enumerate(self.instance.tasks):
            for task_id, task in enumerate(job):
                machine, duration = task
                suffix = f"_{job_id}_{task_id}"
                
                # Create start time variable
                start = self.model.NewIntVar(0, horizon, f"start{suffix}")
                end = self.model.NewIntVar(0, horizon, f"end{suffix}")
                interval = self.model.NewIntervalVar(start, duration, end, f"interval{suffix}")
                
                starts[job_id, task_id] = start
                ends[job_id, task_id] = end
                intervals[job_id, task_id] = interval
                
        # Add precedence constraints within each job
        for job_id, job in enumerate(self.instance.tasks):
            for task_id in range(len(job) - 1):
                self.model.Add(ends[job_id, task_id] <= starts[job_id, task_id + 1])
                
        # Add no-overlap constraints for machines
        machine_to_intervals = collections.defaultdict(list)
        for job_id, job in enumerate(self.instance.tasks):
            for task_id, task in enumerate(job):
                machine, _ = task
                machine_to_intervals[machine].append(intervals[job_id, task_id])
        
        for machine in range(self.instance.num_machines):
            if machine_to_intervals[machine]:  # Only add constraint if machine has tasks
                self.model.AddNoOverlap(machine_to_intervals[machine])
            
        # Makespan objective
        makespan = self.model.NewIntVar(0, horizon, 'makespan')
        self.model.AddMaxEquality(
            makespan, 
            [ends[job_id, len(job)-1] for job_id, job in enumerate(self.instance.tasks)]
        )
        self.model.Minimize(makespan)
        
        # Solve
        # print("Solving JSP instance using OR-Tools (CP-SAT)...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time_in_seconds # Set Time Limit
        status = solver.Solve(self.model)
        
        # print(f"Status: {solver.StatusName(status)}")
        
        # Check if solution found
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print(f"No solution found. Status: {solver.StatusName(status)}")
            return None, None, solver, status, 0, 0
            
        # Extract Solution
        schedule = {}
        makespan_value = solver.Value(makespan)
        
        for job_id, job in enumerate(self.instance.tasks):
            for task_id, (machine, duration) in enumerate(job):
                start_time = solver.Value(starts[job_id, task_id])
                task = Task(start_time, job_id, task_id, duration, machine)
                
                if machine not in schedule:
                    schedule[machine] = []
                schedule[machine].append(task)
                
        # Sort tasks on each machine by start time
        for machine in schedule:
            schedule[machine].sort(key=lambda x: x.start_time)
         
        # Track time and memory usage
        snapshot2 = tracemalloc.take_snapshot()
        end_time_t = time.time()
        
        cp_time = end_time_t - start_time_t
        cp_stats = snapshot2.compare_to(snapshot1, 'lineno')
        cp_memory = sum(stat.size_diff for stat in cp_stats)
            
        return schedule, makespan_value, solver, status, cp_time, cp_memory

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Job Shop Problem Solver using CP-SAT')
    parser.add_argument('--instance_file', type=str, help='Path to the instance file')
    parser.add_argument('--time_limit', type=int, default=0, 
                        help='Time limit in seconds (default: 0). 0 means no time limit')
    parser.add_argument('--output', type=str, default='scheduleICP',
                        help='Base name for output files (default: scheduleICP)')
    
    args = parser.parse_args()
    
    # Load and validate instance
    print(f"Loading instance from {args.instance_file}...")
    instance = load_instance(args.instance_file)
    print(f"Instance loaded: {instance.num_jobs} jobs, {instance.num_machines} machines")
    
    # Initialize and run solver
    solver = ICPSolver(instance)
    if args.time_limit != 0: # if time limit 0 then no time limit is set so it will run until optimal solution is found
        solver.solver.parameters.max_time_in_seconds = args.time_limit
    solver.solver.parameters.random_seed = 10 # for reproducibility
    tracemalloc.start() # Start memory tracking
    
    schedule, makespan, solver, status, cp_time, cp_memory = solver.solve()
    
    if status == cp_model.OPTIMAL:
        print("\nOptimal solution found!")
    elif status == cp_model.FEASIBLE:
        print("\nFeasible solution found (may not be optimal)")
    else:
        print(f"\nNo solution found. Status: {solver.StatusName(status)}")
        return
        
    print(f"Makespan: {makespan}")
    print("\nSolver Statistics:")
    print(f"  - conflicts : {solver.NumConflicts()}")
    print(f"  - branches  : {solver.NumBranches()}")
    print(f"  - wall time : {solver.WallTime():.2f} seconds")
    print(f"  - time      : {cp_time:.2f} seconds")
    print(f"  - memory    : {cp_memory / 1024 / 1024:.2f} MB")
    
    # log schedule to file
    # log_schedule(schedule, makespan, f'{args.output}.txt')
    
    # visualize and save schedule
    visualize_schedule(schedule, makespan, instance, f'output/{args.output}.png')
    
if __name__ == '__main__':
    main()
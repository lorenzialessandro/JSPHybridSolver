import collections
import numpy as np
from typing import List, Dict, Tuple 
from ortools.sat.python import cp_model

from utils import *

class ICPSolver:
    def __init__(self, instance):
        self.instance = instance
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
    def solve(self):
        '''Solve JSP instance using OR-Tools (CP-SAT)'''
        
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
        print("Solving JSP instance using OR-Tools (CP-SAT)...")
        solver = cp_model.CpSolver()
        # solver.parameters.max_time_in_seconds = 600  # 10 minutes time limit
        status = solver.Solve(self.model)
        
        print(f"Status: {solver.StatusName(status)}")
        
        # Check if solution found
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print(f"No solution found. Status: {solver.StatusName(status)}")
            return None, None, solver, status
            
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
            
        return schedule, makespan_value, solver, status

# ----------------- Main -----------------

INSTANCE = '../instances/long-js-600000-100-10000-1'
INSTANCE = '../instances/abz5'

def main():
    # Load and validate instance
    print(f"Loading instance from {INSTANCE}...")
    instance = load_instance(INSTANCE)
    print(f"Instance loaded: {instance.num_jobs} jobs, {instance.num_machines} machines")
    
    # Initialize and run solver
    solver = ICPSolver(instance)
    schedule, makespan, solver, status = solver.solve()
    
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
    
    # log schedule to file
    log_schedule(schedule, makespan, 'scheduleICP.txt')
    
    # visualize and save schedule
    visualize_schedule(schedule, makespan, instance, 'scheduleICP.png')
    
if __name__ == '__main__':
    main()
import collections
import numpy as np
import random
from typing import List, Dict, Tuple 
from ortools.sat.python import cp_model

from utils import *

class ICPSolver():
    def __init__(self, instance):
        self.instance = instance
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
    def solve(self):
        '''Solve JSP instance using OR-Tools (CP-SAT)'''
        horizon = sum([sum([task[1] for task in job]) for job in self.instance.tasks])
        
        starts = {}
        ends = {}
        intervals = {}
        
        # Create job intervals and add to the corresponding machine lists.
        for job_id, job in enumerate(self.instance.tasks):
            for task_id, task in enumerate(job):
                machine, duration = task
                suffix = f"_{job_id}_{task_id}"
                start = self.model.NewIntVar(0, horizon, f"start{suffix}")
                end = self.model.NewIntVar(0, horizon, f"end{suffix}")
                interval = self.model.NewIntervalVar(start, duration, end, f"interval{suffix}")
                starts[job_id, task_id] = start
                ends[job_id, task_id] = end
                intervals[job_id, task_id] = interval
                
        # Precedences inside a job
        for job_id, job in enumerate(self.instance.tasks):
            for task_id in range(len(job) - 1):
                self.model.Add(ends[job_id, task_id] <= starts[job_id, task_id + 1])
                
        # No overlap constraints
        machine_to_intervals = collections.defaultdict(list)
        for job_id, job in enumerate(self.instance.tasks):
            for task_id, task in enumerate(job):
                machine, _ = task
                machine_to_intervals[machine].append(intervals[job_id, task_id])
        
        for machine in range(self.instance.num_machines):
            self.model.AddNoOverlap(machine_to_intervals[machine])
            
        # Makespan objective
        makespan = self.model.NewIntVar(0, horizon, 'makespan')
        self.model.AddMaxEquality(makespan, [ends[job_id, len(job) - 1] for job_id, job in enumerate(self.instance.tasks)])
        self.model.Minimize(makespan)
        
        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)
        
        # ----------------- Extract Solution -----------------
        makespan = solver.ObjectiveValue()
        schedule = {}
        
        for job_id, job in enumerate(self.instance.tasks):
            for task_id, (machine, duration) in enumerate(job):
                start_time = solver.Value(starts[job_id, task_id])
                task = Task(start_time, job_id, task_id, duration, machine)
                
                if machine not in schedule:
                    schedule[machine] = []
                schedule[machine].append(task)
                
        return schedule, makespan, solver, status
   
# ----------------- Main -----------------
     
INSTANCE = '../instances/abz5'

def main():
    # Load instance
    instance = load_instance(INSTANCE)
    
    # Initialize and run solver
    solver = ICPSolver(instance)
    schedule, makespan, solver, status = solver.solve()
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Solution: (Optimal: {status == cp_model.OPTIMAL})")
        
    print(f"Optimal makespan: {makespan}")
    # Print statistics
    print("\nStatistics")
    print(f"  - conflicts: {solver.num_conflicts}")
    print(f"  - branches : {solver.num_branches}")
    print(f"  - wall time: {solver.wall_time}s")
    
    # log schedule to file
    log_schedule(schedule, makespan, 'scheduleICP.txt')
    
    # visualize and save schedule
    visualize_schedule(schedule, makespan, instance, 'scheduleICP.png')
    
    
if __name__ == '__main__':
    main()
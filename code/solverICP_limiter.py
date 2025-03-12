import collections
import numpy as np
from typing import List, Dict, Tuple 
from ortools.sat.python import cp_model
import time             # Time tracking
import tracemalloc      # Memory tracking
import argparse

from utils import *

class Limiter(cp_model.CpSolverSolutionCallback):
    def __init__(self, limit: int):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__solution_limit = limit

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        if self.__solution_count >= self.__solution_limit:
            # print(f"Limit of {self.__solution_limit} solutions reached. Stopping search.")
            self.stop_search()

class ICPSolverLimiter:
    def __init__(self, instance, limit: int = 1):
        self.instance = instance
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.limit = limit
        
    def solve(self):
        '''Solve JSP instance using OR-Tools (CP-SAT)'''
        
        # Track time and memory usage
        start_time_t = time.time()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        
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
        limiter = Limiter(self.limit) # Set solution limit
        status = solver.Solve(self.model, limiter)
        
        # print(f"Status: {solver.StatusName(status)}")
        
        # Check if solution found
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # print(f"No solution found. Status: {solver.StatusName(status)}")
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
            
        # Track time and memory usage
        end_time_t = time.time()
        snapshot2 = tracemalloc.take_snapshot()
        
        cp_time = end_time_t - start_time_t
        cp_stats = snapshot2.compare_to(snapshot1, 'lineno')
        cp_memory = sum(stat.size_diff for stat in cp_stats)
            
        return schedule, makespan_value, solver, status, cp_time, cp_memory
import collections
import numpy as np
import random
from typing import List, Dict, Tuple
from inspyred import ec
import time             # Time tracking
import tracemalloc      # Memory tracking
import argparse

from utils import *

class GASolver():
    def __init__(self, instance, seed):
        self.instance = instance
        self.prng = random.Random(seed)
        
        self.lengths_jobs = [len(job) for job in self.instance.tasks] # number of tasks in each job
        self.num_tasks = sum(self.lengths_jobs) # total number of tasks
        
        # Create a mapping of tasks to (job_id, task_id)
        self.machine_tasks = collections.defaultdict(list)
        for job_id, job in enumerate(self.instance.tasks):
            for task_id, (machine, _) in enumerate(job):
                self.machine_tasks[machine].append((job_id, task_id))
        
        self.best_schedule = None
        self.best_makespan = float('inf')
        
        self.max_time = 60 # default time limit for GA solver
        
    def is_valid_chromosome(self, chromosome):
        """Validate if a chromosome represents a valid machine-task assignment schedule"""
        # A chromosome is now a list of (job_id, task_id) tuples
        
        # Check if chromosome length is correct
        if len(chromosome) != self.num_tasks:
            return False
            
        # Check if each job's tasks appear in the correct number
        job_task_counts = collections.defaultdict(int)
        for job_id, task_id in chromosome:
            job_task_counts[job_id] += 1
            
        for job_id, expected_count in enumerate(self.lengths_jobs):
            if job_task_counts[job_id] != expected_count:
                return False
                
        # Check if machine assignments are valid
        machine_assignment_valid = True
        for i, (job_id, task_id) in enumerate(chromosome):
            # Verify job and task indices are valid
            if job_id >= self.instance.num_jobs or task_id >= self.lengths_jobs[job_id]:
                return False
                
        # Check if task precedence within jobs is maintained
        task_positions = {}
        for pos, (job_id, task_id) in enumerate(chromosome):
            if job_id not in task_positions:
                task_positions[job_id] = {}
            task_positions[job_id][task_id] = pos
            
        for job_id in range(self.instance.num_jobs):
            for task_id in range(1, self.lengths_jobs[job_id]):
                if task_positions[job_id][task_id - 1] > task_positions[job_id][task_id]:
                    return False
                    
        return True

    def find_earliest_start(self, machine: int, job_id: int, task_id: int, 
                          job_ends: List[int], machine_tasks: List[Task], duration: int) -> int:
        
        # TODO: check if this function is correct
        """Find the earliest possible start time for a task considering both job and machine constraints"""
        # Earliest possible start based on job precedence
        start = job_ends[job_id] if task_id > 0 else 0
        
        if not machine_tasks:
            return start
            
        # Sort machine tasks by start time
        sorted_tasks = sorted(machine_tasks, key=lambda x: x.start_time)
        
        # Try to fit the task in gaps between existing tasks
        for i in range(len(sorted_tasks) + 1):
            current_start = start
            
            # Check if we can place before first task
            if i == 0:
                if current_start + duration <= sorted_tasks[0].start_time:
                    return current_start
                continue
                
            # Check if we can place after last task
            if i == len(sorted_tasks):
                return max(current_start, sorted_tasks[-1].start_time + sorted_tasks[-1].duration)
                
            # Check if we can place between tasks
            prev_task = sorted_tasks[i-1]
            next_task = sorted_tasks[i]
            earliest_possible = max(current_start, prev_task.start_time + prev_task.duration)
            
            if earliest_possible + duration <= next_task.start_time:
                return earliest_possible
                
        # If we couldn't find a gap, place after the last task
        return sorted_tasks[-1].start_time + sorted_tasks[-1].duration
    
    def decoder(self, chromosome, args):
        '''Decode chromosome: convert chromosome with (job_id, task_id) tuples to schedule and calculate makespan'''
        if not self.is_valid_chromosome(chromosome):
            chromosome = self.repair_chromosome(chromosome)
            if not self.is_valid_chromosome(chromosome):
                return None, float('inf')
            
        job_task_tracker = [0] * self.instance.num_jobs  # Track which task is next for each job
        job_ends = [0] * self.instance.num_jobs          # End time of last scheduled task for each job
        schedule = collections.defaultdict(list)         # Schedule of tasks on each machine
        
        # Process tasks in chromosome order
        for job_id, task_id in chromosome:
            # Get machine and duration for this task
            machine, duration = self.instance.tasks[job_id][task_id]
            
            # Find earliest possible start time
            start_time = self.find_earliest_start(
                machine, job_id, task_id,
                job_ends, schedule[machine], duration
            )
            
            # Create task object
            task = Task(
                start_time=start_time,
                job_id=job_id,
                task_id=task_id,
                duration=duration,
                machine=machine
            )
            
            schedule[machine].append(task)
            job_task_tracker[job_id] += 1
            job_ends[job_id] = start_time + duration
            
        # Validate the schedule
        if not self.validate_schedule(schedule):
            return None, float('inf')
            
        makespan = max(job_ends)
        return schedule, makespan
        
    def validate_schedule(self, schedule):
        """Validate that the schedule respects all constraints"""
        # Check for machine conflicts
        for machine, tasks in schedule.items():
            sorted_tasks = sorted(tasks, key=lambda x: x.start_time)
            for i in range(len(sorted_tasks) - 1):
                current = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]
                if current.start_time + current.duration > next_task.start_time:
                    return False
                    
        # Check job precedence
        job_task_times = collections.defaultdict(list)
        for machine, tasks in schedule.items():
            for task in tasks:
                job_task_times[task.job_id].append((task.task_id, task.start_time, task.duration))
                
        for job_id, task_times in job_task_times.items():
            sorted_tasks = sorted(task_times, key=lambda x: x[0])  # Sort by task index
            for i in range(len(sorted_tasks) - 1):
                current = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]
                if current[1] + current[2] > next_task[1]:  # start + duration > next_start
                    return False
                    
        return True
        
    def repair_chromosome(self, chromosome):
        """Repair invalid chromosomes to make them valid"""
        # Check if all job tasks are present and in the right order
        job_positions = collections.defaultdict(list)
        
        # Build a valid chromosome template
        valid_chromosome = []
        
        # Create a list of all tasks that need to be scheduled
        all_tasks = []
        for job_id in range(self.instance.num_jobs):
            for task_id in range(self.lengths_jobs[job_id]):
                all_tasks.append((job_id, task_id))
        
        # Try to preserve valid parts of the chromosome
        used_tasks = set()
        for job_id, task_id in chromosome:
            # Skip if this task is already used or is invalid
            if (job_id, task_id) in used_tasks or job_id >= self.instance.num_jobs or task_id >= self.lengths_jobs[job_id]:
                continue
                
            # Check if previous tasks from the same job are already included
            valid_to_include = True
            for prev_task_id in range(task_id):
                if (job_id, prev_task_id) not in used_tasks and (job_id, prev_task_id) not in valid_chromosome:
                    valid_to_include = False
                    break
                    
            if valid_to_include:
                valid_chromosome.append((job_id, task_id))
                used_tasks.add((job_id, task_id))
        
        # Add missing tasks in the correct order
        remaining_tasks = [task for task in all_tasks if task not in used_tasks]
        remaining_tasks.sort()  # Sort by job_id, then task_id to preserve precedence
        
        valid_chromosome.extend(remaining_tasks)
        
        # Double-check validity (debug)
        if not self.is_valid_chromosome(valid_chromosome):
            # Fall back to generating a new valid chromosome
            return self.generator(self.prng, None)
            
        return valid_chromosome
        
    def generator(self, random, args):
        '''Generate valid chromosome with (job_id, task_id) tuples'''
        # Start with ordered tasks for each job
        all_tasks = []
        for job_id in range(self.instance.num_jobs):
            for task_id in range(self.lengths_jobs[job_id]):
                all_tasks.append((job_id, task_id))
                
        # Shuffle maintaining precedence constraints
        chromosome = []
        available_tasks = []
        
        # Initialize available tasks with first task of each job
        for job_id in range(self.instance.num_jobs):
            available_tasks.append((job_id, 0))
            
        # Build chromosome by selecting random available tasks
        job_progress = [0] * self.instance.num_jobs
        
        while available_tasks:
            # Select random available task
            idx = random.randint(0, len(available_tasks) - 1)
            job_id, task_id = available_tasks.pop(idx)
            
            # Add to chromosome
            chromosome.append((job_id, task_id))
            
            # Update job progress
            job_progress[job_id] += 1
            
            # Add next task from this job if available
            if job_progress[job_id] < self.lengths_jobs[job_id]:
                available_tasks.append((job_id, job_progress[job_id]))
                
        return chromosome
        
    def evaluator(self, candidates, args):
        '''Evaluate chromosome'''
        fitness = []
        for chromosome in candidates:
            # Repair invalid chromosomes
            if not self.is_valid_chromosome(chromosome):
                chromosome = self.repair_chromosome(chromosome)
                
            _, makespan = self.decoder(chromosome, args)
            fitness.append(makespan)
            
        return fitness
        
    def observer(self, population, num_generations, num_evaluations, args):
        '''Observer function to track best solution'''
        best = min(population)
        schedule, makespan = self.decoder(best.candidate, args)
        
        if makespan < self.best_makespan:
            self.best_makespan = makespan
            self.best_schedule = schedule
            
        if num_generations % 100 == 0:
            print(f"Generation {num_generations}: Best makespan = {self.best_makespan}")
    
    def solve(self, args):
        '''Solve JSP instance using GA'''
        ga = ec.GA(random=self.prng)
        ga.observer = self.observer
        ga.terminator = self.time_and_generation_terminator
        ga.replacer = ec.replacers.truncation_replacement
        ga.variator = [self.custom_crossover, self.custom_mutation]
        ga.selector = ec.selectors.tournament_selection
        
        # Use initial population if provided
        initial_population = None
        if args is not None and 'initial_population' in args:
            print("Using initial population from args")
            initial_population = args['initial_population']
        
        # Run GA solver
        final_pop = ga.evolve(
            generator=self.generator,
            evaluator=self.evaluator,
            pop_size=50,
            maximize=False,
            bounder=None,  # Custom bounds handling in generator
            max_generations=1000,
            mutation_rate=0.1,
            crossover_rate=0.9,
            num_selected=50,
            initial_population=initial_population,
            max_time = self.max_time,
            start_time = time.time()
        )
        
        return self.best_schedule, self.best_makespan
    
    def custom_crossover(self, random, candidates, args):
        """Custom crossover operator: Order Crossover (OX)""" 
        # Select a point in the job sequence and copy the jobs between two parents
        children = []
        # the candidates are the chromosomes to be crossed (chromosomes are the schedules)
        for i in range(0, len(candidates) - 1, 2):
            # Select parents
            mom = candidates[i]
            dad = candidates[i + 1]
            
            # Perform crossover
            crossover_point = random.randint(0, len(mom) - 1) # select crossover point
            child1 = mom[:crossover_point] + dad[crossover_point:]
            child2 = dad[:crossover_point] + mom[crossover_point:]
            
            # Repair children if they're invalid
            child1 = self.repair_chromosome(child1)
            child2 = self.repair_chromosome(child2)
            
            # Add children to the list
            children.extend([child1, child2])
            
        # Add the last candidate if the population size is odd
        if len(candidates) % 2 == 1:
            children.append(candidates[-1])
            
        return children
        
    def custom_mutation(self, random, candidates, args):
        """Custom mutation operator: Swap Mutation"""
        # Swap two random tasks in the chromosome
        mutants = []
        for candidate in candidates:
            mutant = candidate[:]
            if random.random() < args["mutation_rate"]:
                idx1, idx2 = random.sample(range(len(mutant)), 2) # select two random indices
                mutant[idx1], mutant[idx2] = mutant[idx2], mutant[idx1] # swap jobs
                
                # Repair if invalid
                if not self.is_valid_chromosome(mutant):
                    mutant = self.repair_chromosome(mutant)
                    
            mutants.append(mutant) # add mutant to the list
            
        return mutants
    
    # Custom terminator that combines generation limit and time limit
    def time_and_generation_terminator(self, population, num_generations, num_evaluations, args):
        """Terminate when either max generations or time limit is reached"""
        time_elapsed = time.time() - args['start_time']
        if time_elapsed >= args['max_time']:
            return True
        
        # Check for generation termination
        return num_generations >= args.get('max_generations', 0)
    
# ----------------- Main -----------------

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Job Shop Problem Solver using CP-SAT')
    parser.add_argument('instance_file', type=str, help='Path to the instance file')
    parser.add_argument('--max_time', type=int, default=60, 
                        help='Time limit in seconds (default: 60)')
    parser.add_argument('--output', type=str, default='scheduleICP',
                        help='Base name for output files (default: scheduleGA)')
    
    args = parser.parse_args()
    
    # Load and validate instance
    print(f"Loading instance from {args.instance_file}...")
    instance = load_instance(args.instance_file)
    print(f"Instance loaded: {instance.num_jobs} jobs, {instance.num_machines} machines")
    
    # Initialize and run solver
    solver = GASolver(instance, seed=10)
    
    start_time = time.time()    # track time
    tracemalloc.start()         # track memory
    
    snapshot1 = tracemalloc.take_snapshot() # memory snapshot
    
    schedule, makespan = solver.solve(None) # GA solver
    
    snapshot2 = tracemalloc.take_snapshot() # memory snapshot
    end_time = time.time() # end time
    
    ga_time = end_time - start_time # time taken
    ga_stats = snapshot2.compare_to(snapshot1, 'lineno')
    ga_memory = sum(stat.size_diff for stat in ga_stats) # memory usage
    
    print(f"\nMakespan: {makespan}")
    print(f"    - time: {ga_time:.2f} seconds")
    print(f"    - memory: {ga_memory / 1024 / 1024:.2f} MB")
    
    # Validate final schedule
    is_valid = solver.validate_schedule(schedule)
    print(f"Schedule is valid: {is_valid}")
    
    # Log schedule to file
    # log_schedule(schedule, makespan, f'{args.output}.txt')
    
    # Visualize and save schedule
    # visualize_schedule(schedule, makespan, instance, f'{args.output}.png')

if __name__ == '__main__':
    main()
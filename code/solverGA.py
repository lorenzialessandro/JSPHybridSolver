import collections
import numpy as np
import random
from typing import List, Dict, Tuple
from inspyred import ec

from utils import *

class GASolver():
    def __init__(self, instance, seed):
        self.instance = instance
        self.prng = random.Random(seed)
        
        self.lengths_jobs = [len(job) for job in self.instance.tasks] # number of tasks in each job
        self.num_tasks = sum(self.lengths_jobs) 
        
        self.best_schedule = None
        self.best_makespan = float('inf')
        
    def is_valid_chromosome(self, chromosome):
        """Validate if a chromosome represents a valid schedule"""
        if len(chromosome) != self.num_tasks:
            return False
            
        # Check if each job appears the correct number of times
        job_counts = collections.Counter(chromosome) # count of each job
        for job_id, expected_count in enumerate(self.lengths_jobs):
            if job_counts[job_id] != expected_count: # if job appears more or less than expected
                return False
                
        return True

    def find_earliest_start(self, machine: int, job_id: int, task_id: int, 
                          job_ends: List[int], machine_tasks: List[Task], duration: int) -> int:
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
        return sorted_tasks[-1].start + sorted_tasks[-1].duration
        
    def decoder(self, chromosome, args):
        '''Decode chromosome: convert chromosome to schedule and calculate makespan'''
        if not self.is_valid_chromosome(chromosome):
            return None, float('inf')
            
        job_counts = [0] * self.instance.num_jobs
        job_ends = [0] * self.instance.num_jobs
        schedule = collections.defaultdict(list)
        
        # Process tasks in chromosome order
        for job_id in chromosome:
            task_id = job_counts[job_id]
            machine, duration = self.instance.tasks[job_id][task_id]
            
            # Find earliest possible start time considering both job precedence and machine availability
            start_time = self.find_earliest_start(
                machine, job_id, task_id, 
                job_ends, schedule[machine], duration
            )
            
            task = Task(
                start_time=start_time,
                job_id=job_id,
                task_id=task_id,
                duration=duration,
                machine=machine
            )
            
            schedule[machine].append(task)
            job_counts[job_id] += 1
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
        if len(chromosome) != self.num_tasks:
            return self.generator(self.prng, None)
            
        job_counts = collections.Counter(chromosome)
        fixed_chromosome = []
        remaining = self.lengths_jobs.copy()
        
        # Keep valid job assignments where possible
        for job_id in chromosome:
            if remaining[job_id] > 0:
                fixed_chromosome.append(job_id)
                remaining[job_id] -= 1
                
        # Fill in missing jobs
        while len(fixed_chromosome) < self.num_tasks:
            available_jobs = [j for j, count in enumerate(remaining) if count > 0]
            job = self.prng.choice(available_jobs)
            fixed_chromosome.append(job)
            remaining[job] -= 1
            
        return fixed_chromosome
        
    def generator(self, random, args):
        '''Generate valid chromosome'''
        # Use initial population if provided
        if args['initial_population'] is not None:
            return args['initial_population'][0]
        
        # Generate random chromosome as initial population
        chromosome = []
        remaining = self.lengths_jobs.copy()
        
        while len(chromosome) < self.num_tasks:
            available_jobs = [job_id for job_id, length in enumerate(remaining) if length > 0]
            if not available_jobs:
                break
            job = random.choice(available_jobs)
            chromosome.append(job)
            remaining[job] -= 1
            
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
        ga.terminator = ec.terminators.generation_termination
        ga.replacer = ec.replacers.steady_state_replacement
        ga.variator = [self.custom_crossover, self.custom_mutation]
        ga.selector = ec.selectors.tournament_selection
        
        if args is not None and 'initial_population' in args:
            print("Using initial population from args")
            initial_population = args['initial_population']
        
        
        final_pop = ga.evolve(
            generator=self.generator,
            evaluator=self.evaluator,
            pop_size=200,
            maximize=False,
            bounder=ec.Bounder(0, self.instance.num_jobs - 1),
            max_generations=1000,
            mutation_rate=0.1,
            crossover_rate=0.9,
            num_selected=50,
            initial_population=initial_population
        )
        
        return self.best_schedule, self.best_makespan

    def custom_crossover(self, random, candidates, args):
        """Custom crossover operator: Order Crossover (OX)""" 
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
    
# ----------------- Main -----------------

INSTANCE = '../instances/long-js-600000-100-10000-1'
# INSTANCE = '../instances/abz5'

def main():
    # Load and validate instance
    print(f"Loading instance from {INSTANCE}...")
    instance = load_instance(INSTANCE)
    print(f"Instance loaded: {instance.num_jobs} jobs, {instance.num_machines} machines")
    
    # Initialize and run solver
    solver = GASolver(instance, seed=10)
    schedule, makespan = solver.solve(None)
    
    print(f"\Makespan: {makespan}")
    
    # Validate final schedule
    is_valid = solver.validate_schedule(schedule)
    print(f"Schedule is valid: {is_valid}")
    
    # Log schedule to file
    log_schedule(schedule, makespan, filename='scheduleGA.txt')
    
    # Visualize and save schedule
    visualize_schedule(schedule, makespan, instance, filename='scheduleGA.png')

if __name__ == '__main__':
    main()
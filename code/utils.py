from matplotlib import pyplot as plt
import numpy as np
from typing import NamedTuple
import tracemalloc

class Task():
    """Represents a job task with timing and resource information"""
    def __init__(self, start_time: int, job_id: int, task_id: int, duration: int, machine: int):
        self.start_time = start_time
        self.job_id = job_id
        self.task_id = task_id
        self.duration = duration
        self.machine = machine
        self.end_time = start_time + duration
    
    def __repr__(self):
        return f"({self.start_time}, {self.job_id}, {self.task_id}, {self.duration})"


class Instance():
    '''Instance of Job Shop Scheduling Problem'''
    def __init__(self, name, num_jobs, num_machines, tasks):
        self.name = name
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.tasks = tasks # task = (machine_id, processing_time)
        
    def __str__(self):
        return f"{self.name}: {self.num_jobs} jobs, {self.num_machines} machines"
    
    def __repr__(self):
        return f"{self.name}: {self.num_jobs} jobs, {self.num_machines} machines"
    
def load_instance(filename):
    """Load and parse a JSP instance file"""
    tasks = []
    num_machines = None
    
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Parse first line for dimensions of the instance : num_jobs, num_machines    
    num_jobs, num_machines = map(int, lines[0].split())
    
    # Parse each job's tasks
    for line in lines[1:]:
        if len(tasks) >= num_jobs:
            break
            
        numbers = list(map(int, line.split()))
        job_tasks = []
        i = 0
        
        while i < len(numbers) and numbers[i] >= 0:
            machine, duration = numbers[i:i+2]
            job_tasks.append((machine, duration))
            i += 2
            
        if job_tasks:
            tasks.append(job_tasks)

    # Create instance object
    instance = Instance(filename, num_jobs, num_machines, tasks)
        
    return instance

# Measure Time
def measure_memory(func):
    tracemalloc.start()
    func() # Run the function
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # return peak
    return peak / 1024 / 1024 # Convert to MB

# Visualize Schedule
     
def visualize_schedule(schedule, makespan, instance, filename='schedule.png'):
    """Visualize the schedule using matplotlib"""
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, instance.num_jobs))
    
    for machine, tasks in schedule.items():
        for task in tasks:
            plt.barh(y=machine, 
                    width=task.duration,
                    left=task.start_time,
                    color=colors[task.job_id],
                    edgecolor='black')
            
            # Add task labels
            plt.text(task.start_time + task.duration/2, machine,
                    f'J{task.job_id}',
                    ha='center', va='center')
            
    plt.title(f'Job Shop Schedule (Makespan: {makespan})')
    plt.xlabel('Time')
    plt.ylabel('Machine')
    plt.yticks(range(instance.num_machines), [f'M{i}' for i in range(instance.num_machines)])
    plt.grid(True, alpha=0.2)
    plt.savefig(filename)
    plt.close()        

# Log results
def log_schedule(schedule, makespan, filename='schedule.txt'):
    with open(filename, 'w') as f:
        f.write(f"Best makespan: {makespan}\n")
        f.write("Schedule:\n")
        for machine, tasks in schedule.items():
            f.write(f"Machine {machine}: {tasks}\n")
        f.write("\n")
        f.close()


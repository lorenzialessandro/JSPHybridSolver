from matplotlib import pyplot as plt
import numpy as np
from typing import NamedTuple

class Task(NamedTuple):
    start: int
    job: int
    index: int
    duration: int
    machine: int  # track which machine processes this task
    
    def __repr__(self):
        return f"({self.start}, {self.job}, {self.index}, {self.duration})"


class Instance():
    '''Instance of Job Shop Scheduling Problem'''
    def __init__(self, name, num_jobs, num_machines):
        self.name = name
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.tasks = [] # task = (machine_id, processing_time)
        
    def add_job_tasks(self, tasks):
        # tasks = [(machine_id, processing_time), ...]
        self.tasks.append(tasks) 
        
    def __str__(self):
        return f"{self.name}: {self.num_jobs} jobs, {self.num_machines} machines"
    
    def __repr__(self):
        return f"{self.name}: {self.num_jobs} jobs, {self.num_machines} machines"
    
def load_instance(filename):
    '''Load instance from file'''
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    line_idx = 1 # skip first line
    # read instance name
    name = lines[line_idx].split()[2] # '# instance name'
    line_idx = 4 # skip second line
    # read number of jobs and machines
    num_jobs, num_machines = map(int, lines[line_idx].split()) # 'num_jobs num_machines'
    
    # create instance
    instance = Instance(name, num_jobs, num_machines)
    
    # read tasks
    for job_id in range(num_jobs):
        line_idx += 1
        tasks = list(map(int, lines[line_idx].split()))
        tasks = [(tasks[i], tasks[i+1]) for i in range(0, len(tasks), 2)]
        instance.add_job_tasks(tasks)
        
    return instance

# ----------------- Visualize Schedule -----------------
     
def visualize_schedule(schedule, makespan, instance, filename='schedule.png'):
    """Visualize the schedule using matplotlib"""
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, instance.num_jobs))
    
    for machine, tasks in schedule.items():
        for task in tasks:
            plt.barh(y=machine, 
                    width=task.duration,
                    left=task.start,
                    color=colors[task.job],
                    edgecolor='black')
            
            # Add task labels
            plt.text(task.start + task.duration/2, machine,
                    f'J{task.job}',
                    ha='center', va='center')
            
    plt.title(f'Job Shop Schedule (Makespan: {makespan})')
    plt.xlabel('Time')
    plt.ylabel('Machine')
    plt.yticks(range(instance.num_machines), [f'M{i}' for i in range(instance.num_machines)])
    plt.grid(True, alpha=0.2)
    plt.savefig(filename)
    plt.close()        

# ----------------- Log results -----------------
def log_schedule(schedule, makespan, filename='schedule.txt'):
    with open(filename, 'w') as f:
        f.write(f"Best makespan: {makespan}\n")
        f.write("Schedule:\n")
        for machine, tasks in schedule.items():
            tasks.sort()
            f.write(f"Machine {machine}: {tasks}\n")
        f.write("\n")
        f.close()

# Example usage
def main():
    instance = load_instance('../instances/abz5')
    print(instance)
    # print tasks
    for job_id, tasks in enumerate(instance.tasks):
        print(f"Job {job_id}: {tasks}")
    
if __name__ == '__main__':
    main()
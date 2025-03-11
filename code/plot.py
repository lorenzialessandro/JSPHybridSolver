import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from IPython.display import display

from utils import *
import os

def get_instances_sizes(instances):
    '''Get the sizes (num_jobs, num_machines) of the instances in the data directory and return array of sizes'''
    n_jobs = []
    n_machines = []
    for instance_file in instances:
        instance = load_instance(instance_file)
        n_jobs.append(instance.num_jobs)
        n_machines.append(instance.num_machines)
    return n_jobs, n_machines

def load_data(filename):
    '''Load data from the csv file and return as pandas dataframe'''
    dataframe = pd.read_csv(filename)
    # Convert pandas Series to numpy arrays
    instances = np.array(dataframe['instance'])
    # cp-sat
    cp_makespans = np.array(dataframe['cp_opt_make'])
    cp_times = np.array(dataframe['cp_opt_time'])
    cp_memories = np.array(dataframe['cp_opt_memory'])
    # hybrid
    hy_makespans = np.array(dataframe['hy_make'])
    hy_times = np.array(dataframe['hy_tot_time'])
    hy_memories = np.array(dataframe['hy_tot_memory'])
    # hybrid with collector
    hy_col_makespans = np.array(dataframe['hy_col_make'])
    hy_col_times = np.array(dataframe['hy_col_tot_time'])
    hy_col_memories = np.array(dataframe['hy_col_tot_memory'])
    # hybrid with limit
    hy_lim_makespans = np.array(dataframe['hy_lim_make'])
    hy_lim_times = np.array(dataframe['hy_lim_tot_time'])
    hy_lim_memories = np.array(dataframe['hy_lim_tot_memory'])
    return instances, cp_makespans, cp_times, cp_memories, hy_makespans, hy_times, hy_memories, hy_col_makespans, hy_col_times, hy_col_memories, hy_lim_makespans, hy_lim_times, hy_lim_memories
    
    
def main():
    parser = argparse.ArgumentParser(description='Read data from csv file and plot the results')
    parser.add_argument('--csv_file', type=str, help='Path to the csv file', default='results.csv')
    args = parser.parse_args()
    
    # Load data
    instances, cp_makespans, cp_times, cp_memories, hy_makespans, hy_times, hy_memories, hy_col_makespans, hy_col_times, hy_col_memories, hy_lim_makespans, hy_lim_times, hy_lim_memories = load_data(args.csv_file)
    
    n_jobs, n_machines = get_instances_sizes(instances) # get the sizes of the instances
    
    # convert instances long name to just the last char before last "_"
    instances = [instance.split('_')[-1] for instance in instances]
    # remove negative values from memory usage
    cp_memories = [None if memory < 0 else memory for memory in cp_memories]
    hy_memories = [None if memory < 0 else memory for memory in hy_memories]
    hy_col_memories = [None if memory < 0 else memory for memory in hy_col_memories]
    hy_lim_memories = [None if memory < 0 else memory for memory in hy_lim_memories]
    # remove "0" values from makespan
    cp_makespans = [None if makespan == 0 else makespan for makespan in cp_makespans]
    hy_makespans = [None if makespan == 0 else makespan for makespan in hy_makespans]
    hy_col_makespans = [None if makespan == 0 else makespan for makespan in hy_col_makespans]
    hy_lim_makespans = [None if makespan == 0 else makespan for makespan in hy_lim_makespans]
    
    # Plotting ...

    # Plot 0: Instance sizes bubble plot in the space of number of jobs and number of machines
    plt.figure(figsize=(12, 6))
    plt.scatter(n_jobs, n_machines, s=100, c='r', alpha=0.5)
    plt.xlabel('Number of Jobs')
    plt.ylabel('Number of Machines')
    plt.title('Instance Sizes')
    plt.grid(True, linestyle='--', alpha=0.7)
    #plt.savefig('plot/instance_sizes.png')
    plt.savefig("filepath.svg", format = 'svg', dpi=300)
    plt.show()

    # Plot 1: Makespan comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_makespans, 'o-', label='CP-SAT')
    plt.plot(instances, hy_makespans, 's-', label='Hybrid')
    plt.plot(instances, hy_col_makespans, 's-', label='Hybrid with collector')
    plt.plot(instances, hy_lim_makespans, 's-', label='Hybrid with limit', alpha=0.5)
    plt.xlabel('Instance')
    plt.ylabel('Makespan')
    plt.title('Makespan Comparison')
    plt.legend()
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plot/makespan_comparison.png')
    plt.show()
    
    # Plot 1.1: Makespan comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, hy_makespans, 's-', label='Hybrid')
    plt.plot(instances, hy_col_makespans, 's-', label='Hybrid with collector')
    plt.xlabel('Instance')
    plt.ylabel('Makespan')
    plt.title('Makespan Comparison')
    plt.legend()
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plot/makespan_comparison_hybrid.png')
    plt.show()

    # Plot 2: Computation time comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_times, 'o', label='CP-SAT')
    plt.plot(instances, hy_times, 's', label='Hybrid')
    plt.plot(instances, hy_col_times, 's', label='Hybrid with collector')
    plt.plot(instances, hy_lim_times, 's', label='Hybrid with limit', alpha=0.5)
    # plot the area between all the points
    plt.fill_between(instances, cp_times, hy_times, color='b', alpha=0.1)
    plt.fill_between(instances, cp_times, hy_col_times, color='g', alpha=0.1)
    plt.fill_between(instances, cp_times, hy_lim_times, color='y', alpha=0.1)
    
    plt.xlabel('Instance')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Comparison')
    plt.legend()
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plot/time_comparison.png')
    plt.show()
    
    # Plot 2.1: Computation time comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, hy_times, 's', label='Hybrid')
    plt.plot(instances, hy_col_times, 's', label='Hybrid with collector')
    # plot the area between all the points
    plt.fill_between(instances, hy_times, hy_col_times, color='g', alpha=0.1)
    
    plt.xlabel('Instance')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Comparison')
    plt.legend()
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plot/time_comparison_hybrid.png')
    plt.show()

    # Plot 3 : Memory usage comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_memories, 'o-', label='CP-SAT')
    plt.plot(instances, hy_memories, 's-', label='Hybrid')
    #plt.plot(instances, hy_col_memories, 's-', label='Hybrid with collector')
    plt.plot(instances, hy_lim_memories, 's-', label='Hybrid with limit')
    plt.xlabel('Instance')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plot/memory_comparison.png')
    plt.show()
    
    # For each instance plot bar of makespan of all approaches
    plt.figure(figsize=(12, 6))
    for i, instance in enumerate(instances):
        plt.bar(instance, 0 if cp_makespans[i] is None else cp_makespans[i], color='r', alpha=0.5)
        plt.bar(instance, 0 if hy_makespans[i] is None else hy_makespans[i], color='b', alpha=0.5)
        plt.bar(instance, 0 if hy_col_makespans[i] is None else hy_col_makespans[i], color='g', alpha=0.5)
        plt.bar(instance, 0 if hy_lim_makespans[i] is None else hy_lim_makespans[i], color='y', alpha=0.5)
    plt.xlabel('Instance')
    plt.ylabel('Makespan')
    plt.title('Makespan Comparison')
    plt.legend(['CP-SAT', 'Hybrid', 'Hybrid with collector', 'Hybrid with limit'])
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    plt.savefig('plot/makespan_comparison_bar.png')
    plt.show()
    
    # For each instance plot bar of computation time of all approaches
    plt.figure(figsize=(12, 6))
    for i, instance in enumerate(instances):
        plt.bar(instance, 0 if cp_times[i] is None else cp_times[i], color='r', alpha=0.5)
        plt.bar(instance, 0 if hy_times[i] is None else hy_times[i], color='b', alpha=0.5)
        plt.bar(instance, 0 if hy_col_times[i] is None else hy_col_times[i], color='g', alpha=0.5)
        plt.bar(instance, 0 if hy_lim_times[i] is None else hy_lim_times[i], color='y', alpha=0.5)
    plt.xlabel('Instance')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Comparison')
    plt.legend(['CP-SAT', 'Hybrid', 'Hybrid with collector', 'Hybrid with limit'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    plt.savefig('plot/time_comparison_bar.png')
    plt.show()

    
if __name__ == '__main__':
    main()
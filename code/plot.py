import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

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
    cp_makespans = np.array(dataframe['cp_opt_make'])
    cp_times = np.array(dataframe['cp_opt_time'])
    cp_memories = np.array(dataframe['cp_opt_memory'])
    hy_makespans = np.array(dataframe['hy_make'])
    hy_times = np.array(dataframe['hy_tot_time'])
    hy_memories = np.array(dataframe['hy_tot_memory'])
    # hylim_makespans = np.array(dataframe['hy_lim_make'])
    # hy_lim_times = np.array(dataframe['hy_lim_tot_time'])
    # hy_lim_memories = np.array(dataframe['hy_lim_tot_memory'])
    
    return instances, cp_makespans, cp_times, cp_memories, hy_makespans, hy_times, hy_memories
    
    
def main():
    parser = argparse.ArgumentParser(description='Read data from csv file and plot the results')
    parser.add_argument('--csv_file', type=str, help='Path to the csv file', default='results.csv')
    args = parser.parse_args()
    
    # Load data
    instances, cp_makespans, cp_times, cp_memories, hy_makespans, hy_times, hy_memories = load_data(args.csv_file)
    
    n_jobs, n_machines = get_instances_sizes(instances) # get the sizes of the instances
    
    # convert instances long name to just the last char before last "_"
    instances = [instance.split('_')[-1] for instance in instances]
    # remove negative values from memory usage
    cp_memories = [0 if memory < 0 else memory for memory in cp_memories]
    hy_memories = [0 if memory < 0 else memory for memory in hy_memories]
    # hy_lim_memories = [0 if memory < 0 else memory / 1024 / 1024 for memory in hy_lim_memories]

    # Plotting ...

    # Plot 0: Instance sizes bubble plot in the space of number of jobs and number of machines
    plt.figure(figsize=(12, 6))
    plt.scatter(n_jobs, n_machines, s=100, c='r', alpha=0.5)
    plt.xlabel('Number of Jobs')
    plt.ylabel('Number of Machines')
    plt.title('Instance Sizes')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plot/instance_sizes.png')
    plt.show()

    # Plot 1: Makespan comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_makespans, 'o-', label='CP-SAT')
    plt.plot(instances, hy_makespans, 's-', label='Hybrid')
    # plt.plot(instances, hylim_makespans, 's-', label='Hybrid with limit')
    plt.xlabel('Instance')
    plt.ylabel('Makespan')
    plt.title('Makespan Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plot/makespan_comparison.png')
    plt.show()

    # Plot 2: Computation time comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_times, 'o-', label='CP-SAT')
    plt.plot(instances, hy_times, 's-', label='Hybrid')
    # plt.plot(instances, hy_lim_times, 's-', label='Hybrid with limit')
    # plot the area between the two lines
    plt.fill_between(instances, cp_times, hy_times, color='gray', alpha=0.5)
    # plt.fill_between(instances, cp_times, hy_lim_times, color='gray', alpha=0.5)
    # plt.fill_between(instances, hy_times, hy_lim_times, color='gray', alpha=0.5)
    plt.xlabel('Instance')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plot/time_comparison.png')
    plt.show()

    # Plot 3 : Memory usage comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_memories, 'o-', label='CP-SAT')
    plt.plot(instances, hy_memories, 's-', label='Hybrid')
    # plt.plot(instances, hy_lim_memories, 's-', label='Hybrid with limit')
    plt.xlabel('Instance')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plot/memory_comparison.png')
    plt.show()

    # Plot 4 : plot both makespan and computation time in one figure as points
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_makespans, 'o', label='CP-SAT Makespan')
    plt.plot(instances, cp_times, 's', label='CP-SAT Time')
    plt.plot(instances, hy_makespans, 'o', label='Hybrid Makespan')
    plt.plot(instances, hy_times, 's', label='Hybrid Time')
    plt.xlabel('Instance')
    plt.ylabel('Value')
    plt.title('Makespan and Computation Time Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plot/makespan_time_comparison.png')
    plt.show()

    # Plot 5: Time comparison and percentage difference
    barWidth = 0.35
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: Bar plot comparing times
    ax1.bar(instances, cp_times, color='b', width=barWidth, label='CP-SAT')
    ax1.bar([i + barWidth for i in range(len(instances))], hy_times, color='r', width=barWidth, label='Hybrid')
    ax1.set_xlabel('Instance')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Computation Time Comparison')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Bottom panel: Percentage difference
    percentage_diff = [(cp_times[i] - hy_times[i]) / hy_times[i] * 100 if hy_times[i] > 0 else 0 for i in range(len(instances))]
    ax2.bar(instances, percentage_diff, color='g', width=barWidth)
    ax2.set_xlabel('Instance')
    ax2.set_ylabel('Percentage Difference (%)')
    ax2.set_title('Percentage Difference: (CP-SAT - Hybrid) / Hybrid')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=1)


    plt.tight_layout()
    plt.savefig('plot/time_comparison_two_panel.png')
    plt.show()
    

if __name__ == '__main__':
    main()
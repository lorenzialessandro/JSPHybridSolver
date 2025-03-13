import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from utils import *

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
    '''Load data from the csv file, compute means for each instance, and return as numpy arrays'''
    # Read the dataframe
    dataframe = pd.read_csv(filename)
    
    # Filter out zeros and inf from makespan of 4 approaches
    dataframe = dataframe[(dataframe['cp_opt_make'] > 0) & np.isfinite(dataframe['cp_opt_make'])]
    dataframe = dataframe[(dataframe['hy_make'] > 0) & np.isfinite(dataframe['hy_make'])]
    dataframe = dataframe[(dataframe['hy_col_make'] > 0) & np.isfinite(dataframe['hy_col_make'])]
    dataframe = dataframe[(dataframe['ga_make'] > 0) & np.isfinite(dataframe['ga_make'])]
    # Filter out zeros and inf from time of 4 approaches
    dataframe = dataframe[(dataframe['cp_opt_time'] > 0) & np.isfinite(dataframe['cp_opt_time'])]
    dataframe = dataframe[(dataframe['hy_tot_time'] > 0) & np.isfinite(dataframe['hy_tot_time'])]
    dataframe = dataframe[(dataframe['hy_col_tot_time'] > 0) & np.isfinite(dataframe['hy_col_tot_time'])]
    dataframe = dataframe[(dataframe['ga_time'] > 0) & np.isfinite(dataframe['ga_time'])]
    # Filter out zeros and inf from memory of 4 approaches
    dataframe = dataframe[(dataframe['cp_opt_memory'] > 0) & np.isfinite(dataframe['cp_opt_memory'])]
    dataframe = dataframe[(dataframe['hy_tot_memory'] > 0) & np.isfinite(dataframe['hy_tot_memory'])]
    dataframe = dataframe[(dataframe['hy_col_tot_memory'] > 0) & np.isfinite(dataframe['hy_col_tot_memory'])]
    dataframe = dataframe[np.isfinite(dataframe['ga_memory'])]
    
    # Group by instance and compute means for numerical columns
    mean_data = dataframe.groupby('instance').mean().reset_index()
    
    # Convert pandas DataFrame to numpy arrays
    instances = np.array(mean_data['instance'])
    
    # CP-SAT
    cp_makespans = np.array(mean_data['cp_opt_make'])
    cp_times = np.array(mean_data['cp_opt_time'])
    cp_memories = np.array(mean_data['cp_opt_memory'])
    
    # Hybrid
    hy_makespans = np.array(mean_data['hy_make'])
    hy_times = np.array(mean_data['hy_tot_time'])
    hy_memories = np.array(mean_data['hy_tot_memory'])
    
    # Hybrid with collector
    hy_col_makespans = np.array(mean_data['hy_col_make'])
    hy_col_times = np.array(mean_data['hy_col_tot_time'])
    hy_col_memories = np.array(mean_data['hy_col_tot_memory'])
    
    # GA 
    ga_makespans = np.array(mean_data['ga_make'])
    ga_times = np.array(mean_data['ga_time'])
    ga_memories = np.array(mean_data['ga_memory'])
    
    return instances, cp_makespans, cp_times, cp_memories, hy_makespans, hy_times, hy_memories, hy_col_makespans, hy_col_times, hy_col_memories, ga_makespans, ga_times, ga_memories

def main():
    parser = argparse.ArgumentParser(description='Read data from csv file, compute means, and plot the results')
    parser.add_argument('--csv_file', type=str, help='Path to the csv file', default='results.csv')
    parser.add_argument('--output', type=str, help='Path to the output directory', default='plot')
    parser.add_argument('--save', action='store_true', help='Save plots to output directory', default=False)
    parser.add_argument('--latex', action='store_true', help='Save plots in PGF format for LaTeX compatibility', default=False)
    args = parser.parse_args()
    args.output = os.path.join(args.output, args.csv_file.split('/')[-1].split('.')[0])
    
    if args.save:
        os.makedirs(args.output, exist_ok=True)
    
    # Load data and compute means
    instances, cp_makespans, cp_times, cp_memories, hy_makespans, hy_times, hy_memories, hy_col_makespans, hy_col_times, hy_col_memories, ga_makespans, ga_times, ga_memories = load_data(args.csv_file)
    print(f'Loaded data from {args.csv_file}')
    # Get instance sizes
    n_jobs, n_machines = get_instances_sizes(instances)
    # Convert instances long name to just the last part after last "/"
    instances = [instance.split('_')[-1] for instance in instances]
    
    # Remove "0" values from memory and convert to MB
    cp_memories = [0 if memory == 0 else (memory / 1024 / 1024) for memory in cp_memories]
    hy_memories = [0 if memory == 0 else (memory / 1024 / 1024) for memory in hy_memories]
    hy_col_memories = [0 if memory == 0 else (memory / 1024 / 1024) for memory in hy_col_memories]
    ga_memories = [0 if memory == 0 else (memory / 1024 / 1024) for memory in ga_memories]
    
    # Remove "0" values from makespan
    cp_makespans = [None if makespan == 0 else makespan for makespan in cp_makespans]
    hy_makespans = [None if makespan == 0 else makespan for makespan in hy_makespans]
    hy_col_makespans = [None if makespan == 0 else makespan for makespan in hy_col_makespans]
    ga_makespans = [None if makespan == 0 else makespan for makespan in ga_makespans]
    
    # Function to save figures based on arguments
    def save_figure(filename):
        if args.save:
            if args.latex:
                plt.savefig(f'{args.output}/{filename}.eps', format='eps')
            else:
                plt.savefig(f'{args.output}/{filename}.png')
        plt.show()
    
    # Plot 0: Instance sizes bubble plot in the space of number of jobs and number of machines
    plt.figure(figsize=(12, 6))
    plt.scatter(n_jobs, n_machines, s=100, c='r', alpha=0.5)
    plt.xlabel('Number of Jobs')
    plt.ylabel('Number of Machines')
    plt.title('Instance Sizes')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure('instance_sizes')
    
    # Plot 1: Makespan comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_makespans, 'o', markerfacecolor='none', label='CP-SAT')
    plt.plot(instances, hy_makespans, 'o', markerfacecolor='none', label='Hybrid')
    plt.plot(instances, hy_col_makespans, 'o', markerfacecolor='none', label='Hybrid with collector')
    plt.plot(instances, ga_makespans, 'o', markerfacecolor='none', label='GA')
    plt.xlabel('Instance')
    plt.ylabel('Makespan (mean)')
    plt.title('Makespan Comparison (Average of Multiple Runs)')
    plt.legend()
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure('makespan_comparison')

    # Plot 2: Computation time comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_times, 'o', markerfacecolor='none', label='CP-SAT')
    plt.plot(instances, hy_times, 'o', markerfacecolor='none', label='Hybrid')
    plt.plot(instances, hy_col_times, 'o', markerfacecolor='none', label='Hybrid with collector')
    plt.plot(instances, ga_times, 'o', markerfacecolor='none', label='GA')
    # plot the area between all the points
    plt.fill_between(instances, cp_times, hy_times, color='grey', alpha=0.3)
    plt.fill_between(instances, cp_times, hy_col_times, color='grey', alpha=0.3)
    plt.fill_between(instances, cp_times, ga_times, color='grey', alpha=0.3)
    plt.xlabel('Instance')
    plt.ylabel('Computation Time (s) (mean)')
    plt.title('Computation Time Comparison (Average of Multiple Runs)')
    plt.legend()
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure('time_comparison')

    # Plot 3: Memory usage comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_memories, 'o', markerfacecolor='none', label='CP-SAT')
    plt.plot(instances, hy_memories, 'o', markerfacecolor='none', label='Hybrid')
    plt.plot(instances, hy_col_memories, 'o', markerfacecolor='none', label='Hybrid with collector')
    plt.plot(instances, ga_memories, 'o', markerfacecolor='none', label='GA')
    # plot the area between all the points
    plt.fill_between(instances, cp_memories, hy_memories, color='grey', alpha=0.3)
    plt.fill_between(instances, cp_memories, hy_col_memories, color='grey', alpha=0.3)
    plt.fill_between(instances, cp_memories, ga_memories, color='grey', alpha=0.3)
    plt.xlabel('Instance')
    plt.ylabel('Memory Usage (MB) (mean)')
    plt.title('Memory Usage Comparison (Average of Multiple Runs)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    save_figure('memory_comparison')
    
    # Plot 4: Stacked bar plot for makespan
    plt.figure(figsize=(12, 6))
    # Replace None with 0 (or another appropriate value)
    cp_makespans_no_none = [0 if x is None else x for x in cp_makespans]
    hy_makespans_no_none = [0 if x is None else x for x in hy_makespans]
    hy_col_makespans_no_none = [0 if x is None else x for x in hy_col_makespans]
    ga_makespans_no_none = [0 if x is None else x for x in ga_makespans]

    # Create positions for the bars
    bar_positions = np.arange(len(instances))

    # Choose one as the baseline (e.g., CP-SAT)
    # Then calculate differences
    hy_diff = [h - c for h, c in zip(hy_makespans_no_none, cp_makespans_no_none)]
    hy_col_diff = [hc - c for hc, c in zip(hy_col_makespans_no_none, cp_makespans_no_none)]
    ga_diff = [g - c for g, c in zip(ga_makespans_no_none, cp_makespans_no_none)]

    # Create the bar chart with CP-SAT as baseline
    plt.bar(bar_positions, cp_makespans_no_none, color='b', edgecolor='grey', label='CP-SAT')

    # Add bars for the differences
    plt.bar(bar_positions, hy_diff, color='g', edgecolor='grey', label='Hybrid (difference)', 
            bottom=cp_makespans_no_none)
    plt.bar(bar_positions, hy_col_diff, color='r', edgecolor='grey', label='Hybrid with collector (difference)', 
            bottom=[max(c, h) for c, h in zip(cp_makespans_no_none, hy_makespans_no_none)])
    plt.bar(bar_positions, ga_diff, color='y', edgecolor='grey', label='GA (difference)', 
            bottom=[max(c, hc) for c, hc in zip(cp_makespans_no_none, hy_col_makespans_no_none)])

    plt.xlabel('Instance')
    plt.ylabel('Makespan (mean)')
    plt.title('Makespan Comparison with Differences (Average of Multiple Runs)')
    plt.xticks(bar_positions, instances, rotation=45, fontsize=8)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure('makespan_comparison_differences')
    
if __name__ == '__main__':
    main()
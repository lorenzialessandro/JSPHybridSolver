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

    return instances, cp_makespans, cp_times, cp_memories, hy_makespans, hy_times, hy_memories, hy_col_makespans, hy_col_times, hy_col_memories
    
    
def main():
    parser = argparse.ArgumentParser(description='Read data from csv file and plot the results')
    parser.add_argument('--csv_file', type=str, help='Path to the csv file', default='results.csv')
    parser.add_argument('--output', type=str, help='Path to the output directory', default='plot')
    parser.add_argument('--save', action='store_true', help='Save plots to output directory', default=False)
    parser.add_argument('--latex', action='store_true', help='Save plots in PGF format for LaTeX compatibility', default=False)
    args = parser.parse_args()
    args.output = os.path.join(args.output, args.csv_file.split('/')[-1].split('.')[0])
    
    if args.save:
        os.makedirs(args.output, exist_ok=True)
    
    # Configure matplotlib for LaTeX output if requested
    # if args.latex:
    #     plt.rcParams.update({
    #         "pgf.texsystem": "pdflatex",
    #         "text.usetex": True,
    #         "font.family": "serif",
    #         "font.serif": [],
    #         "font.sans-serif": [],
    #         "font.monospace": [],
    #         "figure.figsize": (12, 6)
    #     })
    
    # Load data
    instances, cp_makespans, cp_times, cp_memories, hy_makespans, hy_times, hy_memories, hy_col_makespans, hy_col_times, hy_col_memories = load_data(args.csv_file)
    
    n_jobs, n_machines = get_instances_sizes(instances) # get the sizes of the instances
    
    # convert instances long name to just the last char before last "_"
    instances = [instance.split('_')[-1] for instance in instances]
    # remove "0" values from memory and convert to MB
    cp_memories = [0 if memory == 0 else (memory / 1024 / 1024) for memory in cp_memories]
    hy_memories = [0 if memory == 0  else (memory / 1024 / 1024) for memory in hy_memories]
    hy_col_memories = [0 if memory == 0 else (memory / 1024 / 1024) for memory in hy_col_memories]
    # remove "0" values from makespan
    cp_makespans = [None if makespan == 0 else makespan for makespan in cp_makespans]
    hy_makespans = [None if makespan == 0 else makespan for makespan in hy_makespans]
    hy_col_makespans = [None if makespan == 0 else makespan for makespan in hy_col_makespans]
    
    # Function to save figures based on arguments
    def save_figure(filename):
        if args.save:
            if args.latex:
                plt.savefig(f'{args.output}/{filename}.pgf', backend='pgf')
            else:
                plt.savefig(f'{args.output}/{filename}.png')
        plt.show()
    
    # Plotting ...

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
    plt.xlabel('Instance')
    plt.ylabel('Makespan')
    plt.title('Makespan Comparison')
    plt.legend()
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure('makespan_comparison')

    # Plot 2: Computation time comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_times, 'o', markerfacecolor='none', label='CP-SAT')
    plt.plot(instances, hy_times, 'o', markerfacecolor='none', label='Hybrid')
    plt.plot(instances, hy_col_times, 'o', markerfacecolor='none', label='Hybrid with collector')
    # plot the area between all the points
    plt.fill_between(instances, cp_times, hy_times, color='grey', alpha=0.3)
    plt.fill_between(instances, cp_times, hy_col_times, color='grey', alpha=0.3)
    plt.xlabel('Instance')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Comparison')
    plt.legend()
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure('time_comparison')

    # Plot 3 : Memory usage comparison
    plt.figure(figsize=(12, 6))
    plt.plot(instances, cp_memories, 'o', markerfacecolor='none', label='CP-SAT')
    plt.plot(instances, hy_memories, 'o', markerfacecolor='none', label='Hybrid')
    plt.plot(instances, hy_col_memories, 'o', markerfacecolor='none', label='Hybrid with collector')
    # plot the area between all the points
    plt.fill_between(instances, cp_memories, hy_memories, color='grey', alpha=0.3)
    plt.fill_between(instances, cp_memories, hy_col_memories, color='grey', alpha=0.3)
    plt.xlabel('Instance')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='x', rotation=45, labelsize=8)
    save_figure('memory_comparison')
    
    # Plot 4: Stacked bar plot for makespan
    plt.figure(figsize=(12, 6))
    # Replace None with 0 (or another appropriate value)
    cp_makespans = [0 if x is None else x for x in cp_makespans]
    hy_makespans = [0 if x is None else x for x in hy_makespans]
    hy_col_makespans = [0 if x is None else x for x in hy_col_makespans]

    # Create positions for the bars
    bar_positions = np.arange(len(instances))

    # Choose one as the baseline (e.g., CP-SAT)
    # Then calculate differences
    hy_diff = [h - c for h, c in zip(hy_makespans, cp_makespans)]
    hy_col_diff = [hc - c for hc, c in zip(hy_col_makespans, cp_makespans)]

    # Create the bar chart with CP-SAT as baseline
    plt.bar(bar_positions, cp_makespans, color='b', edgecolor='grey', label='CP-SAT')

    # Add bars for the differences
    plt.bar(bar_positions, hy_diff, color='g', edgecolor='grey', label='Hybrid (difference)', 
            bottom=cp_makespans)
    plt.bar(bar_positions, hy_col_diff, color='r', edgecolor='grey', label='Hybrid with collector (difference)', 
            bottom=[max(c, h) for c, h in zip(cp_makespans, hy_makespans)])

    plt.xlabel('Instance')
    plt.ylabel('Makespan')
    plt.title('Makespan Comparison with Differences')
    plt.xticks(bar_positions, instances, rotation=45, fontsize=8)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure('makespan_comparison_differences')
    
    
if __name__ == '__main__':
    main()
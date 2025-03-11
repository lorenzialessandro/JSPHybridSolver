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
    

def plot_pareto_front():
    """
    Plot the Pareto front of CP-SAT and Hybrid approaches considering makespan and computation time.
    A solution is on the Pareto front if no other solution is better in both metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load data from the load_data function (reusing the existing code)
    instances, cp_makespans, cp_times, cp_memories, hy_makespans, hy_times, hy_memories, hy_col_makespans, hy_col_times, hy_col_memories, hy_lim_makespans, hy_lim_times, hy_lim_memories = load_data(args.csv_file)
    
    # Remove None values or pairs where one value is None
    cp_data = []
    hy_data = []
    
    for i in range(len(instances)):
        if cp_makespans[i] is not None and cp_times[i] is not None:
            cp_data.append((cp_makespans[i], cp_times[i], instances[i]))
        if hy_makespans[i] is not None and hy_times[i] is not None:
            hy_data.append((hy_makespans[i], hy_times[i], instances[i]))
    
    # Function to identify Pareto-optimal points
    def get_pareto_points(data):
        points = np.array([(d[0], d[1]) for d in data])
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, point in enumerate(points):
            # If already marked as not efficient, skip
            if not is_efficient[i]:
                continue
            # Find all points not worse than current
            is_efficient[is_efficient] = np.any(points[is_efficient] < point, axis=1) | (points[is_efficient] == point).all(axis=1)
            # Keep self as efficient
            is_efficient[i] = True
        return [data[i] for i in range(len(data)) if is_efficient[i]]
    
    # Get Pareto points
    cp_pareto = get_pareto_points(cp_data)
    hy_pareto = get_pareto_points(hy_data)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    cp_x = [d[0] for d in cp_data]
    cp_y = [d[1] for d in cp_data]
    hy_x = [d[0] for d in hy_data]
    hy_y = [d[1] for d in hy_data]
    
    plt.scatter(cp_x, cp_y, color='blue', alpha=0.5, label='CP-SAT Solutions')
    plt.scatter(hy_x, hy_y, color='green', alpha=0.5, label='Hybrid Solutions')
    
    # Plot Pareto fronts
    cp_pareto_x = [d[0] for d in cp_pareto]
    cp_pareto_y = [d[1] for d in cp_pareto]
    hy_pareto_x = [d[0] for d in hy_pareto]
    hy_pareto_y = [d[1] for d in hy_pareto]
    
    # Sort the Pareto points by x-coordinate (makespan) for connecting lines
    cp_pareto_sorted = sorted(zip(cp_pareto_x, cp_pareto_y))
    hy_pareto_sorted = sorted(zip(hy_pareto_x, hy_pareto_y))
    
    cp_pareto_x_sorted = [p[0] for p in cp_pareto_sorted]
    cp_pareto_y_sorted = [p[1] for p in cp_pareto_sorted]
    hy_pareto_x_sorted = [p[0] for p in hy_pareto_sorted]
    hy_pareto_y_sorted = [p[1] for p in hy_pareto_sorted]
    
    # Plot Pareto front lines with more prominent styling
    plt.plot(cp_pareto_x_sorted, cp_pareto_y_sorted, 'b-', linewidth=3, label='CP-SAT Pareto Front')
    plt.plot(hy_pareto_x_sorted, hy_pareto_y_sorted, 'g-', linewidth=3, label='Hybrid Pareto Front')
    
    # Add shaded areas to highlight Pareto dominance regions
    # For CP-SAT: Add points at max y-value for each x to create proper boundary
    cp_area_x = cp_pareto_x_sorted.copy()
    cp_area_y = cp_pareto_y_sorted.copy()
    if len(cp_area_x) > 0:
        max_y = max(cp_y) * 1.1  # Slightly above the maximum y value
        cp_area_x.append(cp_area_x[-1])
        cp_area_y.append(max_y)
        cp_area_x.insert(0, cp_area_x[0])
        cp_area_y.insert(0, max_y)
        plt.fill(cp_area_x, cp_area_y, color='blue', alpha=0.1, label='CP-SAT Dominated Area')
    
    # For Hybrid: Add points at max y-value for each x to create proper boundary
    hy_area_x = hy_pareto_x_sorted.copy()
    hy_area_y = hy_pareto_y_sorted.copy()
    if len(hy_area_x) > 0:
        max_y = max(hy_y) * 1.1  # Slightly above the maximum y value
        hy_area_x.append(hy_area_x[-1])
        hy_area_y.append(max_y)
        hy_area_x.insert(0, hy_area_x[0])
        hy_area_y.insert(0, max_y)
        plt.fill(hy_area_x, hy_area_y, color='green', alpha=0.1, label='Hybrid Dominated Area')
    
    # Add cleaner instance labels for Pareto points
    # Extract just the instance name (last part after the last slash or underscore)
    for point in cp_pareto:
        instance_name = point[2].split('/')[-1] if '/' in point[2] else point[2].split('_')[-1]
        plt.annotate(instance_name, (point[0], point[1]), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=8, color='blue', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7))
    
    for point in hy_pareto:
        instance_name = point[2].split('/')[-1] if '/' in point[2] else point[2].split('_')[-1]
        plt.annotate(instance_name, (point[0], point[1]), textcoords="offset points", 
                     xytext=(0,-10), ha='center', fontsize=8, color='green',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.7))
    
    # Set labels and title
    plt.xlabel('Makespan (lower is better)')
    plt.ylabel('Computation Time (s) (lower is better)')
    plt.title('Pareto Front of CP-SAT vs Hybrid Approach')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save and show plot
    plt.savefig('plot/pareto_front_comparison.png')
    plt.show()

# To include this in the main function, add the following line just before the end of main():
# plot_pareto_front()

# Alternatively, you can call it directly like this:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read data from csv file and plot the results')
    parser.add_argument('--csv_file', type=str, help='Path to the csv file', default='results.csv')
    args = parser.parse_args()
    
    plot_pareto_front()
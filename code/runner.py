import time             
import argparse
import random
import csv
import os
import logging
from datetime import datetime
from contextlib import contextmanager
import concurrent.futures
import threading

from solverHybrid import *
from utils import *

# Set up logging with thread safety
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)-13s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'log/solver_log_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Thread-safe CSV writer
csv_lock = threading.Lock()

@contextmanager
def timer():
    """Context manager for timing code execution."""
    start = time.time()
    yield
    end = time.time()
    return end - start

def run_and_log_experiment(instance, csv_file, seed, first_run=False):
    """Run all solvers on an instance and log results to CSV file."""
    try:
        logger.info(f"Running experiment for instance {instance.name}")
        
        # 1. Run CP-SAT solver to find the optimal solution
        logger.info(f"[{instance.name}] Running CP-SAT to find optimal solution...")
        cp_opt_make, cp_opt_time, cp_opt_memory, cp_opt_status = run_cp_sat_find_optimal(instance, seed)
        logger.info(f"[{instance.name}] CP-SAT completed: makespan={cp_opt_make}, time={cp_opt_time:.2f}s")

        # 2. Run Hybrid solver with limiter
        logger.info(f"[{instance.name}] Running Hybrid solver with limiter...")
        hy_lim_make, _, hy_lim_tot_time, hy_lim_tot_memory = run_hybrid_limiter(instance, seed, cp_opt_time, 1)
        diff_hy_lim_cp_opt = (hy_lim_make - cp_opt_make) / cp_opt_make if cp_opt_make else float('inf')
        logger.info(f"[{instance.name}] Hybrid limiter completed: makespan={hy_lim_make}, time={hy_lim_tot_time:.2f}s")

        # 3. Run GA only solver
        logger.info(f"[{instance.name}] Running GA solver...")
        ga_make, ga_time, ga_memory = run_ga(instance, seed, cp_opt_time)
        logger.info(f"[{instance.name}] GA completed: makespan={ga_make}, time={ga_time:.2f}s")
        
        # 4. Run Hybrid solver
        logger.info(f"[{instance.name}] Running Hybrid solver...")
        hy_make, _, hy_tot_time, hy_tot_memory = run_hybrid(instance, seed, cp_opt_time)
        diff_hy_cp_opt = (hy_make - cp_opt_make) / cp_opt_make if cp_opt_make else float('inf')
        logger.info(f"[{instance.name}] Hybrid completed: makespan={hy_make}, time={hy_tot_time:.2f}s")
        
        # Thread-safe CSV writing
        with csv_lock:
            with open(csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write results
                writer.writerow([
                    instance.name, cp_opt_make, cp_opt_time, cp_opt_memory, cp_opt_status,
                    hy_lim_make, hy_lim_tot_time, hy_lim_tot_memory, diff_hy_lim_cp_opt,
                    ga_make, ga_time, ga_memory,
                    hy_make, hy_tot_time, hy_tot_memory, diff_hy_cp_opt
                ])
        
        logger.info(f"[{instance.name}] Experiment completed and results saved")
        return True
    
    except Exception as e:
        logger.error(f"Error running experiment for {instance.name}: {str(e)}")
        
        # Log error to CSV with thread safety
        with csv_lock:
            with open(csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([instance.name, "ERROR", str(e), "", "", "", "", "", "", "", "", "", "", "", "", ""])
        
        return False

# Runner functions 

def run_hybrid_limiter(instance, seed, time_budget, limit):
    try:
        hybrid_solver = HybridSolver(instance, seed=seed, use_limiter=True, time_budget=time_budget, limit=limit)
        schedule, makespan_ga, makespan_icp, tot_time, tot_memory = hybrid_solver.solve()
        return makespan_ga, makespan_icp, tot_time, tot_memory
    except Exception as e:
        logger.error(f"Error in hybrid limiter for {instance.name}: {str(e)}")
        return float('inf'), float('inf'), 0, 0
        
def run_hybrid(instance, seed, time_budget):
    try:
        hybrid_solver = HybridSolver(instance, seed=seed, use_limiter=False, time_budget=time_budget, limit=0)
        schedule, makespan_ga, makespan_icp, tot_time, tot_memory = hybrid_solver.solve()
        return makespan_ga, makespan_icp, tot_time, tot_memory
    except Exception as e:
        logger.error(f"Error in hybrid solver for {instance.name}: {str(e)}")
        return float('inf'), float('inf'), 0, 0
        
def run_ga(instance, seed, time_budget):
    try:
        ga_solver = GASolver(instance, seed=seed, hybrid=False)
        ga_solver.max_time = time_budget
        schedule, makespan, ga_time, ga_memory = ga_solver.solve(args=None)
        return makespan, ga_time, ga_memory
    except Exception as e:
        logger.error(f"Error in GA solver for {instance.name}: {str(e)}")
        return float('inf'), 0, 0

def run_cp_sat_find_optimal(instance, seed):
    try:
        cp_solver = ICPSolver(instance)
        cp_solver.solver.parameters.random_seed = seed
        schedule, makespan, solver, status, cp_time, cp_memory = cp_solver.solve()
        return makespan, cp_time, cp_memory, status
    except Exception as e:
        logger.error(f"Error in CP-SAT solver for {instance.name}: {str(e)}")
        return float('inf'), 0, 0, "ERROR"
    
def create_csv_file(csv_path):
    """Create CSV file with headers if it doesn't exist."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'instance', 'cp_opt_make', 'cp_opt_time', 'cp_opt_memory', 'cp_opt_status',
                'hy_lim_make', 'hy_lim_tot_time', 'hy_lim_tot_memory', 'diff_hy_lim_cp_opt',
                'ga_make', 'ga_time', 'ga_memory',
                'hy_make', 'hy_tot_time', 'hy_tot_memory', 'diff_hy_cp_opt'
            ])
        return True
    return False

def process_instance(file_path, csv_file, seed, is_first=False):
    """Process a single instance file. Used as a worker function for parallel execution."""
    random.seed(seed)  # Each process gets its own random seed
    instance = load_instance(file_path)
    return run_and_log_experiment(instance, csv_file, seed, first_run=is_first)

def main():
    parser = argparse.ArgumentParser(description='Log Job Shop Problem Solver using CP-SAT')
    parser.add_argument('--folder', type=str, help='Path to the folder containing the instances', default='../instances/ClassicBenchmark')
    parser.add_argument('--instance', type=str, help='Path to a specific instance file')
    parser.add_argument('--csv_file', type=str, default=f'csv/results_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv', help='CSV file to log the results')
    parser.add_argument('--seed', type=int, help='Random seed', default=10)
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: number of CPU cores)', default=None)
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.csv_file), exist_ok=True)
    os.makedirs('log', exist_ok=True)
    
    # Create CSV file with headers
    first_run = create_csv_file(args.csv_file)
    
    # Set number of workers
    max_workers = args.workers if args.workers else None  # None will use CPU count
    
    # Run for a specific instance if provided
    if args.instance is not None:
        random.seed(args.seed)
        instance = load_instance(args.instance)
        run_and_log_experiment(instance, args.csv_file, args.seed, first_run=True)
    
    # Run for all instances in the folder with parallelization
    elif args.folder is not None:
        instance_files = sorted([os.path.join(args.folder, f) 
                               for f in os.listdir(args.folder) 
                               if os.path.isfile(os.path.join(args.folder, f))])
        
        if not instance_files:
            logger.error(f"No instance files found in folder: {args.folder}")
            return
            
        logger.info(f"Starting parallel processing of {len(instance_files)} instances with {max_workers or 'default'} workers")
        
        # Initialize CSV with headers before parallel execution
        create_csv_file(args.csv_file)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all instances to the executor
            # Only mark the first one as first_run=True to write headers
            future_to_instance = {}
            for i, file_path in enumerate(instance_files):
                future = executor.submit(process_instance, file_path, args.csv_file, args.seed, i==0)
                future_to_instance[future] = file_path
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_instance):
                instance_path = future_to_instance[future]
                instance_name = os.path.basename(instance_path)
                try:
                    success = future.result()
                    completed += 1
                    logger.info(f"Progress: {completed}/{len(instance_files)} instances completed. {instance_name} {'succeeded' if success else 'failed'}")
                except Exception as e:
                    logger.error(f"Instance {instance_name} generated an exception: {str(e)}")
            
    logger.info(f"All experiments completed. Results saved to {args.csv_file}")
    
if __name__ == '__main__':
    main()
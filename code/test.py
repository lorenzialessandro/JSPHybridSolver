from runner import run_ga
from utils import *
import resource
import tracemalloc

def get_peak_memory():
    """Returns peak memory usage in MB."""
    peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_total = peak_self + peak_children
    # return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # Convert Byte to MB
    return peak_total / 1024 / 1024  # Convert Byte to MB


def test1():
    instance_path = '../instances/ClassicBenchmark/jobshop_swv02'
    instance = load_instance(instance_path)
    args = {
        'instance': instance,
        'seed': 0,
        'time_budget': 10,
    }
    mem_before = get_peak_memory()
    run_ga(**args)
    mem_after = get_peak_memory()
    peak = max(mem_before, mem_after)
    return peak

def test2():
    instance_path = '../instances/ClassicBenchmark/jobshop_swv02'
    instance = load_instance(instance_path)
    args = {
        'instance': instance,
        'seed': 0,
        'time_budget': 10,
    }
    tracemalloc.start()
    run_ga(**args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024

def main():
    
    peak1 = test1()
    peak2 = test2()
    
    print(f"Memory peak using resource: {peak1:.2f} MB")
    print(f"Memory peak using tracemalloc: {peak2:.2f} MB")
    
    
if __name__ == '__main__':
    main()
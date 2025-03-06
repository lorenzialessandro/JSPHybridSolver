import re
import argparse
import os

def parse_minizinc_jsp(file_path):
    """
    Parse a MiniZinc JSP instance file.
    
    Parameters:
    file_path (str): Path to the MiniZinc file
    
    Returns:
    tuple: (n_jobs, n_machines, job_task_machine, job_task_duration)
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract basic information
    n_jobs_match = re.search(r'n_jobs\s*=\s*(\d+)', content)
    n_machines_match = re.search(r'n_machines\s*=\s*(\d+)', content)
    
    if not n_jobs_match or not n_machines_match:
        raise ValueError("Could not find n_jobs or n_machines in the file")
    
    n_jobs = int(n_jobs_match.group(1))
    n_machines = int(n_machines_match.group(1))
    
    # Extract job_task_machine data
    machine_pattern = r'job_task_machine\s*=\s*array2d\(jobs,\s*tasks,\s*\[(.*?)\]\)'
    machine_match = re.search(machine_pattern, content, re.DOTALL)
    
    if not machine_match:
        raise ValueError("Could not find job_task_machine in the file")
    
    machine_data = machine_match.group(1).replace('\n', ' ').replace('\t', ' ')
    machine_values = [int(x) for x in re.findall(r'\d+', machine_data)]
    
    # Extract job_task_duration data
    duration_pattern = r'job_task_duration\s*=\s*array2d\(jobs,\s*tasks,\s*\[(.*?)\]\)'
    duration_match = re.search(duration_pattern, content, re.DOTALL)
    
    if not duration_match:
        raise ValueError("Could not find job_task_duration in the file")
    
    duration_data = duration_match.group(1).replace('\n', ' ').replace('\t', ' ')
    duration_values = [int(x) for x in re.findall(r'\d+', duration_data)]
    
    # Convert to 2D arrays
    job_task_machine = []
    job_task_duration = []
    
    for j in range(n_jobs):
        job_machines = []
        job_durations = []
        for t in range(n_machines):
            index = j * n_machines + t
            job_machines.append(machine_values[index])
            job_durations.append(duration_values[index])
        job_task_machine.append(job_machines)
        job_task_duration.append(job_durations)
    
    return n_jobs, n_machines, job_task_machine, job_task_duration

def convert_to_jsp_format(n_jobs, n_machines, job_task_machine, job_task_duration, output_file):
    """
    Convert the parsed data to JSP format and save to file.
    """
    with open(output_file, 'w') as f:
        # Write header
        instance = output_file.split('/')[-1]
        f.write("#+++++++++++++++++++++++++++++\n")
        f.write(f"# instance {instance}\n")
        f.write("#+++++++++++++++++++++++++++++\n")
        f.write("# MiniZinc\n")
        # Write basic information
        f.write(f"{n_jobs} {n_machines}\n")
        
        # Write job data
        for j in range(n_jobs):
            line = []
            for t in range(n_machines):
                machine = job_task_machine[j][t]
                duration = job_task_duration[j][t]
                line.append(f"{machine} {duration}")
            f.write(" ".join(line) + "\n")
    
    print(f"Conversion complete. File saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert MiniZinc JSP instance to simple text format')
    parser.add_argument('input_folder', help='Directory path to the MiniZinc JSP instance files')
    parser.add_argument('output_folder', help='Directory path to store output files')
    
    args = parser.parse_args()
    
    # Load and validate instance
    print(f"Loading instances from {args.input_folder}...")
    instance_files = [f for f in os.listdir(args.input_folder) if f.endswith('.dzn')]
    
    for instance_file in instance_files:
        input_file = os.path.join(args.input_folder, instance_file)
        n_jobs, n_machines, job_task_machine, job_task_duration = parse_minizinc_jsp(input_file)
        
        output_file = os.path.join(args.output_folder, instance_file.replace('.dzn', ''))
        convert_to_jsp_format(n_jobs, n_machines, job_task_machine, job_task_duration, output_file)
        
    print(f"Conversion complete")
    
if __name__ == "__main__":
    main()
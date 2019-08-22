import subprocess, re, os, sys

def run_command(cmd):
    """Run command, return output as string."""
    
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

    
def set_gpu(num_gpu,gpu_list = [0,1,2,3,4,5,6,7]):
    tot_mem = 11172.0
    memory_gpu_map =sorted([(memory/tot_mem, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()if gpu_id in gpu_list])
    available_gpus = []
    for item in memory_gpu_map:
        mem_usage, gpu_id = item
        if mem_usage < 0.9:
            available_gpus.append(gpu_id)
    print ("Found %d GPU(s)"%len(memory_gpu_map),"only %d gpu below threshold"%len(available_gpus))
    if len(available_gpus)>=num_gpu:
        gpus = ','.join(str(e) for e in available_gpus[:num_gpu])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus   
        print ("Using GPU %s"%gpus)
    else:
        raise ValueError("No GPUs available, current number of available GPU is %d, requested for %d GPU(s)"%
                         (len(available_gpus),num_gpu))

import os
import yaml
import hostlist

os.system("nvidia-smi")
hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
print(os.environ["SLURM_JOB_NODELIST"])
print(hostnames)
master_addr = os.environ["SLURMD_NODENAME"]
master_port = 12346
num_machines = int(os.environ["SLURM_NNODES"])
nb_gpus = len(os.environ["SLURM_JOB_GPUS"].split(","))
num_processes = num_machines * nb_gpus


for machine_rank in range(num_machines):
    config_accelerate = {
        "compute_environment": "LOCAL_MACHINE",
        "deepspeed_config": {},
        "distributed_type": "MULTI_GPU",
        "downcast_bf16": "no",
        #'dynamo_backend': 'NO',
        "fsdp_config": {},
        "gpu_ids": "all",
        "machine_rank": machine_rank,
        "main_process_ip": hostnames[0],
        "main_process_port": master_port,
        "main_training_function": "main",
        "megatron_lm_config": {},
        "mixed_precision": "no",
        "num_machines": num_machines,
        "num_processes": num_processes,
        "rdzv_backend": "static",
        "same_network": True,
        #'tpu_env': [],
        #'tpu_use_cluster': False,
        #'tpu_use_sudo': False,
        "use_cpu": False,
    }

    with open(f"./config_accelerate_rank{machine_rank}.yaml", "w") as file:
        yaml.dump(config_accelerate, file)

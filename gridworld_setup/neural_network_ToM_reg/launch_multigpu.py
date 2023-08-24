import os
from time import sleep
import time

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

nb_nodes = 1

job_directory = "%s/results/" % os.getcwd()

# Make top level directories
mkdir_p(job_directory)

batch_size = [2]
nb_epoch = [100] 


for epoch in nb_epoch:
    for bs in batch_size:
    
        config = f'_nbepoch_{epoch}_bs_{bs}'
        run_directory = job_directory + str(time.ctime()) + config + '_multigpu/'
        run_directory = run_directory.replace(' ', '_').replace('.', '-').replace(':', '__')
        mkdir_p(run_directory)

        job_file = os.path.join(
            run_directory, "expe.slurm")

        with open(job_file, 'w') as fh:

            fh.writelines("#!/bin/bash\n")
            fh.writelines(
                "#SBATCH --job-name=expe\n")
            fh.writelines(
                "#SBATCH --output=" + run_directory + "expe_%j.out\n")
            fh.writelines(
                "#SBATCH --error=" + run_directory + "expe_%j.out\n")
            fh.writelines(
                "#SBATCH --ntasks-per-node=1\n")
            fh.writelines(
                "#SBATCH --cpus-per-task=20\n")
            fh.writelines(
                "#SBATCH --nodes="+str(nb_nodes)+"\n")
            fh.writelines(
                "#SBATCH --hint=nomultithread\n")

            fh.writelines(
                "#SBATCH --account=kcr@v100\n")
            fh.writelines(
                "#SBATCH --qos=qos_gpu-dev\n")
            fh.writelines(
                "#SBATCH -C v100-32g\n")
            fh.writelines(
                "#SBATCH --time=00:59:59\n")
            fh.writelines(
                "#SBATCH --gres=gpu:4\n")
                
            fh.writelines(
                "module load pytorch-gpu/py3/2.0.0\n")

            fh.writelines(
                "set -x\n")

            fh.writelines(
                "python idr_accelerate.py\n")

            fh.writelines(
                f"srun bash -c 'accelerate launch --config_file ./config_accelerate_rank${{SLURM_PROCID}}.yaml train_multigpu.py -e {epoch} -b {bs}'")

        os.system("sbatch %s" % job_file)
        sleep(1)
    print(job_file)

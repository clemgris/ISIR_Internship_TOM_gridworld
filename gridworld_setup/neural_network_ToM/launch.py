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

batch_size = [3, 5]
nb_epoch = [100] 


for epoch in nb_epoch:
    for bs in batch_size:
    
        config = '_nbepoch_' + str(epoch) + '_bs_' + str(bs) 
        run_directory = job_directory + str(time.ctime()) + config + '/'
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
                "#SBATCH --gres=gpu:1\n")
                
            fh.writelines(
                "module load pytorch-gpu/py3/2.0.0\n")

            fh.writelines(
                "set -x\n")

            fh.writelines(
                "srun bash -c 'python train.py -e {} -b {}'".format(epoch, bs))

        os.system("sbatch %s" % job_file)
        sleep(1)
    print(job_file)

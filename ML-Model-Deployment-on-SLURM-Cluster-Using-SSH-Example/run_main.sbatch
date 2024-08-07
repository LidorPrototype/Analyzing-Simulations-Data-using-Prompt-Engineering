#!/bin/bash

#SBATCH --partition=main                ### Specify partition name where to run a job.
#SBATCH --time=0-02:00:00               ### Job running time limit.
#SBATCH --job-name='ml_model_job'       ### Name of the job.
#SBATCH --output=job-%J.out             ### Output log for running job - %J is the job number variable
#SBATCH --mail-user=your_email@domain.com        ### User's email for sending job status
#SBATCH --mail-type=ALL                 ### Conditions when to send the email.

#SBATCH --gpus=1                        ### Number of GPUs
##SBATCH --mem=32G                      ### Amount of RAM memory
##SBATCH --cpus-per-task=6             ### Number of CPU cores

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ###
module load anaconda                    ### Load anaconda module
source activate env             ### Activate environment

# Execute the Python script with an argument
python main.py "$1"

#!/bin/bash

#SBATCH --partition main                                    ### Specify partition name where to run a job.
#SBATCH --time 0-04:00:00                                   ### Job running time limit. Make sure it is not exceeding the partition time li>
#SBATCH --job-name 'orgenize_dataset'                       ### Name of the job. replace my_job with your desired job name
#SBATCH --output=outputs/job-%J-orgenize-dataset.out        ### Output log for running job - %J is the job number variable
##SBATCH --error=error_job-%J.out
#SBATCH --mail-user=lidorel@jce.ac.il                       ### User's email for sending job status
#SBATCH --mail-type=ALL                                     ### Conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

#SBATCH --gpus=1                                            ### number of GPUs, ask for more than 1 only if you can parallelize your code for multi>
##SBATCH --mem=32G                                          ### ammount of RAM memory
##SBATCH --cpus-per-task=6                                  ### number of CPU cores

NEW_JOB_NAME="${USER}_JOB_ID:${SLURM_JOB_ID}_NODE:${SLURM_JOB_NODELIST}"  ### Creates new job name
### Print some data to output file ###
scontrol update JobId=$SLURM_JOB_ID JobName=$NEW_JOB_NAME                 ### Update Slurm Job Info
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo "Job $NEW_JOB_NAME Finished"
echo ""

### Start you code below ####
module load anaconda                                        ### load anaconda module (must present when working with conda environments)
source activate env_lora_test                               ### activating environment, environment must be configured before running the job                           
python orgenize_dataset.py
#!/bin/bash

# Define the server credentials
USER="user_name"
SERVER_IP="192.168.200.2"
SCRIPT_PATH="/home/sbatch_user/ML-Model-Deployment-on-SLURM-Cluster-Using-SSH-Example" # Update the path to the SLURM job script

# Define the string argument you want to pass
argument="Hello, SLURM!"

# SSH into the server and execute the SLURM job
ssh $USER@$SERVER_IP << ENDSSH
# Navigate to the directory containing the SLURM job script
cd $SCRIPT_PATH

# Submit the SLURM job script with the argument
job_id=\$(sbatch run_main.sbatch "$argument" | awk '{print \$4}')

# Wait for the job to complete
while squeue -u $USER | grep -q \$job_id; do sleep 5; done

# Display the results
cat job-\$job_id.out
ENDSSH

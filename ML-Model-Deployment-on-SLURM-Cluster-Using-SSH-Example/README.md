
# ML-Model Deployment on SLURM Cluster Using SSH Example

This project provides an example of deploying a machine learning model on a SLURM cluster using SSH. It demonstrates how to submit a job to a SLURM cluster and retrieve the results via SSH.

## Project Structure

```
ML-Model-Deployment-on-SLURM-Cluster-Using-SSH-Example/
├── main.py
├── run_main.sbatch
├── run_job/
│   ├── run_job.sh
│   ├── run_job.py
│   └── run_job.md
└── README.md
```

- **main.py**: A Python script that serves as a placeholder for your ML model. It takes a string argument and converts it to uppercase.
- **run_main.sbatch**: A SLURM job script for submitting the job to the cluster.
- **run_job/run_job.sh**: A Bash script to automate the SSH connection, job submission, and result retrieval.
- **run_job/run_job.py**: A Python script that demonstrates how to submit a job to a SLURM cluster using SSH.
- **README.md**: This file.

## Prerequisites

- Access to a SLURM cluster.
- SSH access to the server where the SLURM cluster is set up.
- Anaconda environment with Python installed on the server.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https:--/ML-Model-Deployment-on-SLURM-Cluster-Using-SSH-Example.git
   cd ML-Model-Deployment-on-SLURM-Cluster-Using-SSH-Example
   ```

2. **Update the SLURM Script (`run_main.sbatch`)**:
   - Replace `your_email@domain.com` with your actual email to receive job status notifications.

3. **Configure SSH Script (`run_job.sh`) or (`run_job.py`)**:
   - Update the `USER`, `SERVER_IP`, and `SCRIPT_PATH` variables with your server details and script path.

4. **Copy Files to SLURM Cluster**:
   
   To run the SLURM job, you'll need to transfer your `.sbatch` and Python files to the master node of your SLURM cluster. Here's how you can do it:

   ```bash
   scp main.py run_main.sbatch your_username@your_master_node_ip:/path/to/destination/
   ```

   - Replace `your_username` with your username on the SLURM cluster.
   - Replace `your_master_node_ip` with the IP address of your master node.
   - Replace `/path/to/destination/` with the directory path where you want to place the files on the cluster.

5. **Run the SSH Script**:
   ```bash
   chmod +x run_job.sh
   ./run_job.sh
   ```
   or
   ```python run_job.py```

6. **Check Output**:
   - The output will be displayed in the terminal, showing the converted uppercase string.

## Customization

- **ML Model**: Replace the code in `main.py` with your ML model code.
- **SLURM Configuration**: Adjust the SLURM parameters in `run_main.sbatch` according to your job requirements.
- **Arguments**: Modify `run_job.sh` to pass different arguments to your script.

## Expected Output

After running the SSH script, the expected output should be displayed in the terminal as follows:

```
Uppercase String: HELLO, SLURM!
```

This output indicates that the Python script successfully executed on the SLURM cluster, processed the input string, and returned the uppercase version.

## Troubleshooting

- Check the SLURM job logs (`job-<job_id>.out`) for any errors.


## Authors

- [Ephi Cohen- Software Lab Engineer](mailto:ephraimco@jce.ac.il) - [Azrieli College of Engineering Jerusalem](https://www.jce.ac.il/) @ephi052

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
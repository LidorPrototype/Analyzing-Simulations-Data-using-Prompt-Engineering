# This script demonstrates how to submit a job to a SLURM cluster using SSH.
# It prompts the user for their password and then submits a job to the SLURM cluster.
# The job script is assumed to be located on the remote server.
# The script waits for the job to complete and then prints the output of the job.
# The script uses the paramiko library to establish an SSH connection to the server.
# To install paramiko, run: pip install paramiko
# The user is prompted to enter their password securely using the getpass library.
# The script can be modified to accept command-line arguments for the username, server IP, script path, and argument.
# The script can be run as a standalone script or imported as a module in another script.

# @Author: Ephi Cohen
# @Date: 08/08/2024
# @Version: 1.0

import paramiko
import getpass

# Define parameters here for easy editing
USER = "your_username"   # Username for SSH connection
SERVER_IP = "your_server_ip"  # IP address of the HPC cluster master node
SCRIPT_PATH = "/path/to/your/slurm_script" # Path to the SLURM script on the remote server
ARGUMENT = "your_argument" # Argument to pass to the SLURM script

def main():    # Prompt the user for their password
    password = getpass.getpass(prompt=f"Enter password for {USER}@{SERVER_IP}: ")

    # SSH connection setup
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(SERVER_IP, username=USER, password=password)

        # Construct the command to submit the SLURM job
        submit_job_cmd = f"""
        cd {SCRIPT_PATH}
        job_id=$(sbatch run_main.sbatch "{ARGUMENT}" | awk '{{print $4}}')
        while squeue -u {USER} | grep -q $job_id; do sleep 5; done
        cat job-$job_id.out
        """

        # Execute the command on the remote server
        stdin, stdout, stderr = ssh.exec_command(submit_job_cmd)

        # Wait for the command to complete
        exit_status = stdout.channel.recv_exit_status()

        # Print the job output
        if exit_status == 0:
            print("Job output:")
            print(stdout.read().decode())
        else:
            print("Error:")
            print(stderr.read().decode())

    except paramiko.AuthenticationException:
        print("Authentication failed, please verify your credentials.")
    except paramiko.SSHException as sshException:
        print(f"Unable to establish SSH connection: {sshException}")
    except Exception as e:
        print(f"Exception in connecting to the server: {e}")
    finally:
        # Close the SSH connection
        ssh.close()

if __name__ == "__main__":
    main()

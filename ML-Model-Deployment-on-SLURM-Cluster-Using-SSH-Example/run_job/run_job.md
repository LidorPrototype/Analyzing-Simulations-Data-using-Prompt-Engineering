# Run Job Script for Submitting SLURM Jobs via SSH

This script demonstrates how to submit a job to a SLURM cluster using SSH. It prompts the user for their password and then submits a job to the SLURM cluster. The job script is assumed to be located on the remote server. The script waits for the job to complete and then prints the output of the job.

## Features

- Securely prompts the user for their SSH password.
- Submits a job script to a remote SLURM cluster.
- Waits for the job to complete and prints the job output.
- Uses the `paramiko` library for SSH connections.
- Easy to configure with user-editable parameters.

## Requirements

- Python 3.x
- `paramiko` library

## Installation

1. **Install the `paramiko` library**:

   ```
   pip install paramiko
   ```

## Configuration

Update the following parameters in the script to match your setup:

- \`USER\`: Your username for SSH.
- \`SERVER_IP\`: The IP address of the HPC cluster master node.
- \`SCRIPT_PATH\`: The path to the SLURM script on the remote server.
- \`ARGUMENT\`: The argument to pass to the SLURM script.

```python
# Define parameters here for easy editing
USER = "your_username"   # Username for SSH connection
SERVER_IP = "your_server_ip"  # IP address of the HPC cluster master node
SCRIPT_PATH = "/path/to/your/slurm_script" # Path to the SLURM script on the remote server
ARGUMENT = "your_argument" # Argument to pass to the SLURM script
```

## Usage

1. **Run the script**:

   ```
   python run_job.py
   ```

2. **Enter your SSH password** when prompted.

3. **The script will handle the rest**, including connecting to the server, submitting the job, waiting for its completion, and printing the job output.

## Example

If your username is \`john_doe\`, your server IP is \`192.168.1.100\`, your SLURM script is located at \`/home/john_doe/slurm_scripts\`, and you want to pass the argument \`test_argument\`:

1. Update the script parameters:

   ```python
   USER = "john_doe"
   SERVER_IP = "192.168.1.100"
   SCRIPT_PATH = "/home/john_doe/slurm_scripts"
   ARGUMENT = "test_argument"
   ```

2. Run the script:

   ```bash
   python run_job.py
   ```

3. Enter your SSH password when prompted.

## Error Handling

The script includes error handling for:

- Authentication failures
- SSH connection issues
- General exceptions

If any of these errors occur, a relevant message will be printed to help diagnose the issue.

## License

This project is licensed under the MIT License.

## Author

Ephi Cohen

## Date

08/08/2024

## Version

1.0

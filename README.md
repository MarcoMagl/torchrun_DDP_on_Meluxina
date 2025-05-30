# Distributed Training Example with PyTorch on Meluxina

This repository demonstrates how to run distributed training with PyTorch, using either the PyTorch installation from the system's software stack or within an Apptainer (Singularity) container.

## Repository Contents

*   `README.md`: This file.
*   `resnet50ScriptDDP_for_torchrun.py`: A PyTorch training script for a ResNet50 model, configured for DistributedDataParallel (DDP) using `torchrun`.
*   `usingApptainer/`: Contains files for building and running the training script within an Apptainer container.
*   `usingPyTorchFromTheStack/`: Contains files for running the training script using the PyTorch installation from the system's software stack.

## Running the Example

There are two ways to run the distributed training example:

1.  **Using PyTorch from the Software Stack**
2.  **Using an Apptainer Container**

Choose the method that best suits you.

### 1. Using PyTorch from the Software Stack

This method utilizes the PyTorch installation available in the system's software stack. 

**Steps:**

1.  **Navigate to the `usingPyTorchFromTheStack/` directory:**

    ```bash
    cd usingPyTorchFromTheStack/
    ```

2.  **Submit the SLURM job using `sbatch`:**

    ```bash
    sbatch launcher_DDP_torchrun.sh
    ```

    This will submit the `launcher_DDP_torchrun.sh` script to the SLURM queue.  The script will handle the distributed training setup and execution.

3.  **Monitor the job:**

    Use `squeue` or other SLURM commands to monitor the job's progress.  Check the output files generated by the job for any errors or progress updates.


**Important Notes:**
*   **`--rdzv_endpoint`:** The `MASTER_NODE` and `PORT` need to be defined.  A simple way to do this is to set `MASTER_NODE=$(hostname)` and `PORT=29500` before the `torchrun` command.  If you are running on multiple nodes, you may need to use the IP address of the master node.
*   **Adapt the SBATCH parameters:** Adjust the `#SBATCH` parameters (e.g., `--nodes`, `--ntasks-per-node`, `--gres`, `--time`) to match your resource requirements and the available resources for your script 
*   **Output Logs:**  Check the `output_%j.log` file for any errors or progress updates.

### 2. Using an Apptainer Container

This method encapsulates the PyTorch environment within an Apptainer container, providing a consistent and reproducible environment across different systems.

**Prerequisites:**

*   Apptainer (Singularity) installed in the Meluxina software stack.
*   A Docker Hub account (or access to a container registry) to pull the base image.
*   A SLURM workload manager environment.

**Steps:**

1.  **Build the Apptainer image (if you haven't already):**

    *   **Reserve a node in interactive mode:**

        ```bash
        salloc -p gpu -N1 -A p200xxx -t 60 -q default
        ```

        This command requests a GPU node for 60 minutes.  Adjust the parameters as needed. Change the `p200xxx` with your proper project number 

    *   **Navigate to the `usingApptainer/` directory:**

        ```bash
        cd usingApptainer/
        ```

    *   **Source the `build_sif_image_from_docker.sh` script:**

        ```bash
        source build_sif_image_from_docker.sh
        ```

        This script will build the Apptainer image (`pytorch_22.08.sif` or similar) from a Docker image.  The script contains commands like `apptainer build pytorch_22.08.sif docker://nvcr.io/nvidia/pytorch:22.08-py3`.  **You can of course adapt the Docker image name to your desired PyTorch version and CUDA version.**

    *   **Exit the interactive node:**

        ```bash
        exit
        ```

2.  **Run the distributed training:**

    *   **Submit the SLURM job using `sbatch`:**

        ```bash
        sbatch run_with_apptainer.sh
        ```

        This will submit the `run_with_apptainer.sh` script to the SLURM queue.

3.  **Monitor the job:**

    Use `squeue` or other SLURM commands to monitor the job's progress.  Check the output files generated by the job for any errors or progress updates.

**Important Notes:**

*   **`DOCKER_IMAGE`:**  **Crucially, adapt the `DOCKER_IMAGE` variable to the correct Docker image name.**  The `nvcr.io/nvidia/pytorch` images are a good starting point, but you need to choose the version that matches your desired PyTorch, CUDA, and Python versions.  For example, `nvcr.io/nvidia/pytorch:22.08-py3` uses PyTorch 22.08 and Python 3.  Check the NVIDIA NGC catalog for available images.
*   **Image Size:** Apptainer images can be large. Ensure you have sufficient storage space.


**Important Notes:**

*   **Adapt the script:** You may need to adapt the script to your specific training task, dataset, and model.
*   **Dependencies:** Ensure that all necessary PyTorch dependencies are installed in your environment (either in the system's software stack or within the Apptainer container).
*   **Command-line arguments:** The script expects command-line arguments for configuration.  Make sure to provide the correct values when launching the script.

## Troubleshooting

*   **"ModuleNotFoundError: No module named 'torch'":**  This indicates that PyTorch is not installed or not accessible in your environment.  Double-check that you have loaded the correct PyTorch module (if using the software stack) or that PyTorch is installed correctly within your Apptainer container.
*   **"CUDA error: out of memory":**  This indicates that your GPU is running out of memory.  Try reducing the batch size, model size, or using mixed-precision training.
*   **"RuntimeError: Address already in use":**  This indicates that the `MASTER_PORT` is already in use.  Try using a different port number.
*   **"Connection refused":** This indicates that the processes are unable to communicate with each other.  Double-check that the `MASTER_ADDR` is correct and that the processes are able to reach each other over the network.
*   **Training not starting:** Double-check the `world_size`, `rank`, and `local_rank` are being correctly passed to and used by `torch.distributed.init_process_group`.

## Contributing

Contributions to this repository are welcome!  Please submit a pull request with your changes.

This README provides a comprehensive guide to running the distributed training example. Remember to adapt the scripts and instructions to your specific environment and requirements. Good luck!

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from contextlib import nullcontext
from torch.profiler import profile, ProfilerActivity
import os
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


def ddp_setup_for_torchrun():
    
    return world_size, rank, local_rank



def cleanup():
    dist.destroy_process_group()


def main(save_every, total_epochs, batch_size, num_workers):

    try:
        # first thing to do is setup the distr env
        # setup(rank, world_size)
        # world_size, rank, local_rank = ddp_setup_for_torchrun()

        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        transform = transforms.Compose([
            transforms.Resize(224),  # ResNet-50 expects 224x224 images
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

        batch_size_total = int(batch_size)

        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_total, sampler=train_sampler, shuffle=False , num_workers=num_workers, pin_memory=True,)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, )

        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_total, sampler=test_sampler, shuffle=False, num_workers=num_workers, pin_memory=True,)

        model = torchvision.models.resnet50(pretrained=True)
        # !!!!!!!!!!!!!!!!!! TRICKY
        #     This causes 
        #     -- Process 1 terminated with the following error:
        # Traceback (most recent call last):
        #   File "/apps/USE/easybuild/release/2024.1/software/PyTorch/2.3.0-foss-2024a-CUDA-12.6.0/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 75, in _wrap
        #     fn(i, *args)
        #   File "/mnt/tier2/project/lxp/mmagliulo/Resnet50FineTuningNativePytorch/resnet50ScriptDDP.py", line 63, in main
        #     device = torch.device(f'cuda:{rank}')
        #              ^^^^^
        # model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 100)  # CIFAR-100 has 100 classes
        print(f"Running basic DDP example on rank {rank}.")

        device = torch.device(f'cuda:{local_rank}')
        model.to(device)
        model = DDP(model, device_ids=[local_rank])

        useCudnnBenchmark = False 
        if useCudnnBenchmark:
            torch.backends.cudnn.benchmark = True

        # 03/11 I moved the loss to the rank 
        loss = nn.CrossEntropyLoss().to(local_rank)
        lr_corrected = 0.01/(world_size)
        optimizer = optim.SGD(model.parameters(), lr=lr_corrected, momentum=0.9)
        num_epochs = total_epochs 

        DoProfile = False 
        jobid= os.environ['SLURM_JOB_ID'] 


        useAMP = True 

        for epoch in range(total_epochs):

            print(f"Starting the epoch {epoch}")

            train_sampler.set_epoch(epoch)
            epoch_compute_time = 0.0  # Local compute time accumulator
            total_images = 0 
            global_start_time = time.time()

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                batch_start_time = time.time()
                
                # Use non_blocking=True for async transfers
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                if useAMP:
                    #Enable Automatic Mixed Precision (AMP) if GPU supports it
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(inputs)
                        running_loss = loss(outputs, labels)
                else:
                    outputs = model(inputs)
                    running_loss = loss(outputs, labels)

                running_loss.backward()
                optimizer.step()
                
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                
                epoch_compute_time += batch_time
                total_images += inputs.size(0)

                if batch_idx % 10 == 0:
                    images_per_second = inputs.size(0) / batch_time
                    pathfile = os.path.join(os.getcwd(), f"output_{rank}.txt")
                    with open(pathfile, "a") as f:
                        print(f"Batch {batch_idx + 1}: {images_per_second:.2f} images/second, processed {total_images} in tot", file=f)

                # prof.step causes a severe bug to not uncomment 

                if DoProfile:
                    # if rank == 0:
                    #     prof.step()  # Only call profiler step on the main process
                    prof.step()

            # Synchronize all processes at the end of the epoch.
            dist.barrier()

            # Aggregate the total images and compute time across processes.
            total_images_tensor = torch.tensor(total_images, device=device, dtype=torch.float32)
            total_compute_time_tensor = torch.tensor(epoch_compute_time, device=device, dtype=torch.float32)

            dist.all_reduce(total_images_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_compute_time_tensor, op=dist.ReduceOp.SUM)

            # Calculate accurate average throughput.
            if total_compute_time_tensor.item() > 0:
                avg_images_per_sec = total_images_tensor.item() / (total_compute_time_tensor.item() / dist.get_world_size())
            else:
                avg_images_per_sec = float('nan')

            global_end_time = time.time()
            elapsed_time = global_end_time - global_start_time

            if rank == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
                print(f"Total images processed: {total_images_tensor.item()} in {elapsed_time:.2f} seconds (wall-clock time)")
                print(f"Average throughput (compute-only, no synchronisation) per GPU: {avg_images_per_sec:.2f} images/second")
                print(f"Epoch {epoch+1} Start time: {global_start_time}")
                print(f"Epoch {epoch+1} End time: {global_end_time}")

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Test Accuracy: {100 * correct / total:.2f}%")

            # one must use the test loader for the model eval
            compute_throughput(model, test_loader , local_rank, world_size)

        # if DoProfile:
        #     # prof.export_chrome_trace("trace.json")
        #     writer.close()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

def set_random_seeds(random_seed=0):
    import random
    import numpy as np

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



def compute_throughput(model, data_loader, rank, world_size):
    model.eval()  # Set model to evaluation mode
    total_images = 0
    start_time = time.time()  # Start the timer

    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch  # Assuming inputs are the images, and _ are the labels
            inputs = inputs.cuda(rank, non_blocking=True)  # Move the input to the correct device

            # Forward pass
            outputs = model(inputs)

            # Update the count of processed images for this process (rank)
            total_images += inputs.size(0)  # inputs.size(0) gives batch size

    # Synchronize the total number of images processed across all processes
    total_images_tensor = torch.tensor(total_images).cuda(rank)
    dist.all_reduce(total_images_tensor, op=dist.ReduceOp.SUM)

    # Only rank 0 will print the throughput, others can skip this step
    if rank == 0:
        total_time = time.time() - start_time  # Total time taken across all processes
        total_images = total_images_tensor.item()  # Get the aggregated total
        throughput = total_images / total_time  # Images per second
        print(f"Throughput: {throughput:.2f} images per second (total across {world_size} GPUs)")



# def run_demo(main_fn, world_size, args):
#     mp.spawn(main_fn,
#              args=(world_size,args.save_every, args.total_epochs, args.batch_size, args.num_workers,),
#              nprocs=world_size,
#              join=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', default=1, type=int, help='How often to save a snapshot')
    parser.add_argument('batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('num_workers', default=2, type=int, help='Number of workers')
    random_seed_default = 0
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    args = parser.parse_args()

    world_size=4

    set_random_seeds(args.random_seed)

    main(args.save_every, args.total_epochs, args.batch_size, args.num_workers,)


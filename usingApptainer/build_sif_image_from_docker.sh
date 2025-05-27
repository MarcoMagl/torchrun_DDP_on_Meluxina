#!/bin/bash
module load Apptainer
export TMPDIR=/project/scratch/lxp/app-reframe/tmp_build
apptainer build pytorch_22.08.sif docker://nvcr.io/nvidia/pytorch:22.08-py3

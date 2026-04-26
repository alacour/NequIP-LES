#!/bin/bash
# Run once on a NERSC login node to create the NequIP-LES conda environment.
# Usage: bash setup_nequip_nersc.sh

set -e

export CONDA_PKGS_DIRS=$SCRATCH/conda/pkgs
mkdir -p $SCRATCH/conda/pkgs

module load conda
source $(conda info --base)/etc/profile.d/conda.sh

conda create --prefix $SCRATCH/code/les/nnpackages/nequip python=3.11 -y
conda activate $SCRATCH/code/les/nnpackages/nequip

pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install nequip==0.16.0
pip install les@git+https://github.com/ChengUCB/les
pip install -e $SCRATCH/code/les/NequIP-LES

echo "Environment setup complete. Activate with: conda activate $SCRATCH/code/les/nnpackages/nequip"

#!/bin/bash
#SBATCH -A m2834_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH -J nequip_electrolyte_les
#SBATCH -o logs/nequip_%j.out
#SBATCH -e logs/nequip_%j.err

mkdir -p logs

cd $SCRATCH/code/les/NequIP-LES/scripts

$SCRATCH/code/les/nnpackages/nequip/bin/nequip-train config_electrolyte_les.yaml

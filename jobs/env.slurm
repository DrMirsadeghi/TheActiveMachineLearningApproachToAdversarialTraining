#!/bin/bash
#SBATCH --job-name=env
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=0-100:00:00
#SBATCH --mem=10GB
#SBATCH --output=env.log

module load rocky8-spack/master > /dev/null
module load cuda/11.3.1-gcc-10.3.0-ehi3 > /dev/null
module load py-pip/23.1.2-gcc-10.3.0-python-3.9.7-pmrk > /dev/null

python -m venv my_venv .
source ./my_venv/bin/activate

python -m pip install torch
python -m pip install torchvision
python -m pip install numpy
python -m pip install matplotlib pandas
python -m pip install git+https://github.com/Harry24k/MAIR.git
python -m pip list

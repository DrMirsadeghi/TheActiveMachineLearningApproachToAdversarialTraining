#!/bin/bash
#SBATCH --job-name=pre_cf_m2
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=0-100:00:00
#SBATCH --mem=10GB
#SBATCH --output=pre_cf_m2.log

module load rocky8-spack/master > /dev/null
module load cuda/11.3.1-gcc-10.3.0-ehi3 > /dev/null
module load py-pip/23.1.2-gcc-10.3.0-python-3.9.7-pmrk > /dev/null

source ./my_venv/bin/activate

python -m pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torchvision
python -m pip install --user torchvision
#python -m pip install  torch torchvision
python -m pip install --quiet --user numpy==1.23 matplotlib pandas > /dev/null
python -m pip install --quiet -U numpy > /dev/null
python -m pip install --quiet --user torch torchvision git+https://github.com/Harry24k/MAIR.git > /dev/null

python -m pip list

python ../pretrain.py 0 1

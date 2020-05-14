#!/bin/bash
#SBATCH --job-name=DKBAT_wn18_paths
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl


conda activate dgat
python main.py ---use_2hop 1 --get_2hop 1 --partial_2hop 0 --data ./data/WN18RR/ --output_folder ./checkpoints/wn/out/
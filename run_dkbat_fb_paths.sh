#!/bin/bash
#SBATCH --job-name=DKBAT_fb15k_paths
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python main.py --data ./data/FB15k-237/ --epochs_gat 3000 --epochs_conv 150 --weight_decay_gat 0.00001 --use_2hop 1 --get_2hop 1 --partial_2hop 1 --batch_size_gat 272115 --margin 1 --out_channels 50 --drop_conv 0.3 --output_folder ./checkpoints/fb/out/
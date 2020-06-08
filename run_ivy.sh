#!/bin/bash
#SBATCH --job-name=DKBAT_ivy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl


conda activate dgat
python main.py --use_2hop 0 --get_2hop 0 --partial_2hop 0  --data ./data/ivy141-all/ --output_folder ./checkpoints/ivy/out/ --batch_size_conv 128 --out_channels 500 --valid_invalid_ratio_conv 40 --valid_invalid_ratio_gat 2 --epochs_gat 3000 --batch_size_gat 32185 --weight_decay_gat 5e-6 --alpha 0.2 --margin 1 --drop_GAT 0.3
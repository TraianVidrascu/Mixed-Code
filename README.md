# Bidirectional Neighborhood Model

The Bidirectional Neighborhood Model is an adaptation of the KBAT model from the paper: [Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](https://arxiv.org/abs/1906.01195)

The neighborhood model was written over the source code of the [KBAT model](https://github.com/deepakn97/relationPrediction).

# Installation
Use the [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) package manager to install the environment of this project.

```bash
conda env create -f envname.yml
source activate base
``` 

# Modifications
- usage of the inbound and outbound node neighborhoods 
- Topological Relation Layer

# Datasets
- FB15k-237
- WN18RR
- Kinship
- Ivy v1.4.1 software graph (Private company dataset)

# Reproduce Results
- FB15k-237 
```bash
python main.py --data ./data/FB15k-237/  --valid_invalid_ratio_conv 40  --out_channels 50 --drop_conv 0.3  --batch_size_conv 128  --epochs_gat 3000 --epochs_conv 150 --weight_decay_gat 0.00001 --use_2hop 0 --get_2hop 0 --partial_2hop 0 --batch_size_gat 272115 --margin 1 --out_channels 50 --drop_conv 0.3 --output_folder ./checkpoints/fb/out/
``` 

- WN18RR
```bash
python main.py --use_2hop 0 --get_2hop 0 --partial_2hop 0 --batch_size_conv 128 --out_channels 500 --valid_invalid_ratio_conv 40 --valid_invalid_ratio_gat 2 --epochs_gat 3000 --batch_size_gat 86835 --weight_decay_gat 5e-6 --alpha 0.2 --margin 5 --drop_GAT 0.3 --data ./data/WN18RR/ --output_folder ./checkpoints/wn/out/
``` 

- Kinship
```bash
python main.py --data ./data/kinship/ --epochs_gat 3000 --epochs_conv 400 --weight_decay_gat 0.00001 --use_2hop 1 --get_2hop 1 --partial_2hop 0 --batch_size_gat 8544 --margin 1 --out_channels 50 neg_s_conv 10 --drop_conv 0.3 --output_folder ./checkpoints/fb/out/
``` 

# License
[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/) 

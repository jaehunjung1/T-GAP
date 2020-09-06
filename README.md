# T-GAP

This is an official implementation of **T-GAP: Learning to Walk across Time for Temporal Knowledge Graph Completion** (submitted to AAAI2020).

## Requirements
* python >= 3.7
* torch >= 1.4.0
* numpy == 1.17.2
* dgl-cu101 == 0.4.2
* tqdm == 4.36.1
* ipdb == 0.12.3

You can install the requirements using `pip install -r requirements.txt`.

## Datasets, Pre-processing, and Pretrained Model
We provide a shell script to download the datasets and pre-process them. The datasets used in the paper are **ICEWS14**, **ICEWS05-15**, and **Wikidata11k**.
Go to the `data` folder, and type `sh preprocess.sh`.
The script downloads, and pre-processes the official distribution of the 3 public datasets provided by [Garcia-Duran et al](https://github.com/nle-ml/mmkb).

We also provide the version of T-GAP trained on **ICEWS14** dataset. The model file can be found at `results/checkpoint/icews14.ckpt`.

## Simple Demo
You can run a demo script to evaluate the trained model on ICEWS14, by running `sh scripts/demo.sh`.
The experimental results of the trained model are as follows:

|**MRR**|**Hits@1**|**Hits@3**|**Hits@10**|
|:------:|:--------:|:--------:|:---------:|
|0.610|50.7|67.7|79.0|

Note that we have used 1 Tesla V100 GPU for all experiments.

## Training
We provide additional scripts for each dataset that trains T-GAP with the same hyperparameter configurations as used in the paper.
The scripts can be run as follows: 
- **ICEWS14**: `sh scripts/icews14.sh` 
- **ICEWS05-15**: `sh scripts/icews05-15.sh`
- **Wikidata11k**: `sh sh scripts/wikidata11k.sh`

Running these files will automatically create checkpoint files and tensorboard log in `results/checkpoint/${DATASET}`, and `results/log/${DATASET}` respectively. For a tutorial on how to use tensorboard, refer to [Pytorch Tensorboard Tutorial](https://pytorch.org/docs/stable/tensorboard.html).

To train with custom hyperparameters, run `main.py` in the project root folder as follows:
```shell
python main.py \
  --run=${DIR_NAME} \
  --dataset=${DATASET_NAME} \
  --device=${cuda/cpu} \
  --seed=... \
  --epoch=... \
  --lr=... \
  --grad_clip=... \
  --node_dim=... \
  --num_in_heads=... \
  --num_out_heads=... \
  --num_step=... \
  --num_sample_from=... \
  --max_num_neighbor=... \
```

| **Option** | **Description** | **Default** |
|:--- | :--- | :---: |
|`run`| Output dir name (automatically created under `results/checkpoint`, `results/log`) | Current Time |
|`dataset`| Dataset to train (among `data/icews14_aug`, `data/icews05-15_aug`, `data/wikidata11k_aug`) | `data/icews14_aug`|
|`device`| Torch device to use (`cuda` or `cpu`) | `cuda`|
|`seed`| Random seed | 999 |
|`epoch`| Number of training epochs | 20 |
|`lr` | Learning Rate | 5e-4 |
|`grad_clip`| Gradient clipping norm | 3 |
|`node_dim`| Embedding dimension | 100 |
|`num_in_heads`| Number of PGNN, SGNN attention heads | 5 |
|`num_out_heads`| Number of Attention Flow attention heads | 5 |
|`num_step`| Number of propagation steps | 3 |
|`num_sample_from`| Number of core nodes | 10 |
|`max_num_neighbor`| Number of sampled / added edges | 100 |

## Evaluation
You can evaluate a trained model, using Mean Reciprocal Rank (MRR), Hits@1/3/10 on the test set.

For evaluation, you need to locate the checkpoint file generated during training.
To the same command used in the training, add `--test` and `--ckpt=${CKPT_DIR}` to evaluate.

```shell
python main.py
  --test \
  --ckpt=${CKPT_DIR} \
  --other_options (Same configuration with training)
```


## References
Garcia-Duran. A, Dumancic. S, and Niepert. M, Learning Sequence Encoders for Temporal Knowledge Graph Completion. EMNLP 2018

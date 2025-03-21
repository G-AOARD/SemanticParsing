# StructSP: Efficient Fine-tuning of Task-Oriented Dialog System by Using Structure-aware Boosting and Grammar Constraints
(ACL 2023)

## Create a conda environment and install the required packages

```bash
conda create -p structsp_env python=3.8
conda activate ./structsp_env
pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Train model
Scripts for running our experiment can be found in the ./*/scripts/ folder. Note that you need to change DATA_DIR, BERT_DIR, OUTPUT_DIR to your own dataset path, bert model path and log path, respectively.

```bash
cd top
./scripts/train_*.sh
```

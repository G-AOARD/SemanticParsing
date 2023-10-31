#!/usr/bin/bash
#
#         Job Script for VPCC , JAIST
#                                    2018.2.25 

#PBS -N eval
#PBS -j oe -l select=1
#PBS -q GPU-1A
#PBS -o pbs_infer-sp.log
#PBS -e infer-sp.err.log
#PBS -M s2110418@jaist.ac.jp 
#PBS -m e 

source ~/.bashrc

REPO_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_with_grammars

conda activate /home/s2110418/anaconda3/envs/IDLT && cd $REPO_PATH && export PYTHONPATH="$PYTHONPATH:$REPO_PATH" && \

# seed = 0
MODEL_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_with_grammars/outputs/top/trainer_SimSemRoberta_logical_focus_10_epochs_seed_0/epoch=9.ckpt
HPARAM_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_with_grammars/outputs/top/trainer_SimSemRoberta_logical_focus_10_epochs_seed_0/lightning_logs/version_0/hparams.yaml
python evaluate/evaluate.py $MODEL_PATH $HPARAM_PATH
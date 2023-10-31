#!/usr/bin/bash
#
#         Job Script for VPCC , JAIST
#                                    2018.2.25 

#PBS -N eval
#PBS -j oe -l select=1
#PBS -q GPU-1
#PBS -o pbs_infer-sp.log
#PBS -e infer-sp.err.log
#PBS -M s2110418@jaist.ac.jp 
#PBS -m e 

source ~/.bashrc

REPO_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars

conda activate /home/s2110418/anaconda3/envs/IDLT && cd $REPO_PATH && export PYTHONPATH="$PYTHONPATH:$REPO_PATH" && \

# seed = 0
MODEL_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars/outputs/trainer_SimSemRoBerta_10_epochs_weather_500spis_seed_0/epoch=13.ckpt
HPARAM_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars/outputs/trainer_SimSemRoBerta_10_epochs_weather_500spis_seed_0/lightning_logs/version_0/hparams.yaml
python evaluate/evaluate.py $MODEL_PATH $HPARAM_PATH

# seed = 1
MODEL_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars/outputs/trainer_SimSemRoBerta_10_epochs_weather_500spis_seed_1/epoch=21.ckpt
HPARAM_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars/outputs/trainer_SimSemRoBerta_10_epochs_weather_500spis_seed_1/lightning_logs/version_0/hparams.yaml
python evaluate/evaluate.py $MODEL_PATH $HPARAM_PATH

# seed = 2
MODEL_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars/outputs/trainer_SimSemRoBerta_10_epochs_weather_500spis_seed_2/epoch=25.ckpt
HPARAM_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars/outputs/trainer_SimSemRoBerta_10_epochs_weather_500spis_seed_2/lightning_logs/version_0/hparams.yaml
python evaluate/evaluate.py $MODEL_PATH $HPARAM_PATH

# seed = 3
MODEL_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars/outputs/trainer_SimSemRoBerta_10_epochs_weather_500spis_seed_3/epoch=24.ckpt
HPARAM_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars/outputs/trainer_SimSemRoBerta_10_epochs_weather_500spis_seed_3/lightning_logs/version_0/hparams.yaml
python evaluate/evaluate.py $MODEL_PATH $HPARAM_PATH

# seed = 4
MODEL_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars/outputs/trainer_SimSemRoBerta_10_epochs_weather_500spis_seed_4/epoch=24.ckpt
HPARAM_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_topv2_with_grammars/outputs/trainer_SimSemRoBerta_10_epochs_weather_500spis_seed_4/lightning_logs/version_0/hparams.yaml
python evaluate/evaluate.py $MODEL_PATH $HPARAM_PATH
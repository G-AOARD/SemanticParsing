REPO_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_with_grammars
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

# seed = 0
MODEL_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_with_grammars/outputs/top/trainer_SimSemRoberta_logical_focus_10_epochs_test_seed_0/epoch=0.ckpt
HPARAM_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams_v2_with_grammars/outputs/top/trainer_SimSemRoberta_logical_focus_10_epochs_test_seed_0/lightning_logs/version_0/hparams.yaml
CUDA_VISIBLE_DEVICES=0 python evaluate/evaluate.py $MODEL_PATH $HPARAM_PATH
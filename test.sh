set -euxo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8
#export CUDA_VISIBLE_DEVICES=$gpu

cd src
. ./path.sh

python train.py config/run.yaml --dataset SynAudioMNIST --language digits --n_phonemes 12 --model_class CRDNN_CTC --model_name test_model --model !include:../models/CRDNN_CTC/model.yaml --extra_overrides "{model: {n_epochs: 1}}" --debug --debug_batches 2

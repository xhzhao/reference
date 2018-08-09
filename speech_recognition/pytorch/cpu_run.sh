# Script to train and time DeepSpeech 2 implementation

export KMP_AFFINITY=compact,1,0,granularity=fine
#export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=40
#export MKL_VERBOSE=1
#export KMP_AFFINITY=verbose
which python

RANDOM_SEED=1
TARGET_ACC=23

python train.py --model_path models/deepspeech_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED --acc $TARGET_ACC

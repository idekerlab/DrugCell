
echo DrugCell SETTINGS
echo SETTINGS
PROCS=3
echo PROCS $PROCS
export CANDLE_CUDA_OFFSET=1
# General Settings
export PROCS=4
export PPN=2
export WALLTIME=06:00:00
export NUM_ITERATIONS=2
export POPULATION_SIZE=4
# GA Settings
export STRATEGY='mu_plus_lambda'
export OFF_PROP=0.5
export MUT_PROB=0.8
export CX_PROB=0.2
export MUT_INDPB=0.5
export CX_INDPB=0.5
export TOURNAMENT_SIZE=4
# Lambda Settings
export CANDLE_CUDA_OFFSET=1
export CANDLE_DATA_DIR=/tmp/rgnanaolivu
# Polaris Settings
# export QUEUE="debug"


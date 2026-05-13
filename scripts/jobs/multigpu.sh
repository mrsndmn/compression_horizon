set -x
set -e

ENV_PREFIX=/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin
WORKDIR=/workspace-SR004.nfs2/d.tarasov/compression_horizon


# Добавим в PATH префикс окружения
PATH=$ENV_PREFIX:$PATH

# Джобы запускаются под управлением MPI
# Поэтому нам нужно получить имя мастер-ноды
# Чтобы сохранить его в переменные окружения,
# с которыми работает DDP
MASTER_HOST_PREFIX=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$//; print \$x ")
MASTER_HOST=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$/-mpimaster-0/; print \$x ")

MASTER_HOST_FULL="$MASTER_HOST.$MASTER_HOST_PREFIX"
echo "MASTER_HOST_FULL $MASTER_HOST_FULL"

JOBS_TMP_PROC="/workspace-SR004.nfs2/d.tarasov/jobs_tmp_proc/$MASTER_HOST"
echo "JOBS_TMP_PROC $JOBS_TMP_PROC"
mkdir -p $JOBS_TMP_PROC

# Сохраняем в переменные окружения,
# с которыми работает DDP

if [ -z "$LOCAL_MASTER_ADDR" ]; then
    export MASTER_ADDR=${MASTER_HOST_FULL:-"127.0.0.1"}
else
    export MASTER_ADDR=${LOCAL_MASTER_ADDR}
fi

export MASTER_PORT=12345
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-"1"}
export RANK=${OMPI_COMM_WORLD_RANK:-"0"}

NUM_GPUS=$(nvidia-smi -L | nvidia-smi -L | grep -c "GPU")

echo "NUM_GPUS $NUM_GPUS"
echo "WORLD_SIZE $WORLD_SIZE"
echo "RANK $RANK"
echo "MASTER_ADDR $MASTER_ADDR"

cd $WORKDIR


${ENV_PREFIX}/python ${ENV_PREFIX}/accelerate launch \
    --config_file ./configs/accelerate.yaml \
    --machine_rank $RANK \
    --num_processes $(($NUM_GPUS * $WORLD_SIZE)) \
    --num_machines $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    $@

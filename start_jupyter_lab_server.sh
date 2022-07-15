#!/bin/bash

source vars.sh

$CONTAINER_CMD run --gpus all -it --rm \
  --env HOME=$WORKDIR \
  --env SHELL="/bin/bash" \
  --publish $JUPYTER_PORT:$JUPYTER_PORT \
  --workdir $WORKDIR \
  --volume $WORKDIR:$WORKDIR \
  --ipc=host \
  $IMAGE python -m jupyterlab --allow-root --ServerApp.port $JUPYTER_PORT \
    --no-browser --ServerApp.ip='0.0.0.0' --certfile=$CERTFILE --keyfile=$KEYFILE

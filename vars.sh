CONTAINER_CMD=docker
WORKDIR=$(realpath $(pwd))
IMAGE=wisard-image
JUPYTER_PORT=10089
CERT_DIR=.ssl_cert
CERTFILE=$CERT_DIR/certificate.crt
KEYFILE=$CERT_DIR/certificate.key

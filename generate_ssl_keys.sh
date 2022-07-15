#!/bin/bash

source vars.sh
mkdir -p $CERT_DIR

COUNTRY_CODE="BR"
STATE="Sao paulo"
LOCATION="Campinas"
ORGANIZATION="UNICAMP"
UNIT="H.IAAC"
WEB="hiaac.unicamp.br"

openssl req -newkey rsa:4096 \
            -x509 \
            -sha256 \
            -days 3650 \
            -nodes \
            -out $CERTFILE \
            -keyout $KEYFILE \
            -subj "/C=$COUNTRY_CODE/ST=$STATE/L=$LOCATION/O=$ORGANIZATION/OU=$UNIT/CN=$WEB"

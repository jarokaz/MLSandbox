FROM horovod/horovod:0.16.1-tf1.12.0-torch1.0.0-mxnet1.4.0-py3.5

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        curl \
        wget \
        jq

COPY ./docker-entrypoint.sh /

ENTRYPOINT ["/docker-entrypoint.sh"]


FROM tensorflow/tensorflow:latest-gpu

WORKDIR /train

COPY train/train.py /train/train.py

ENTRYPOINT ["python", "train.py"]


FROM tensorflow/tensorflow:1.12.0-gpu

WORKDIR /train

COPY train/train.py /train/train.py

ENTRYPOINT ["python", "train.py"]



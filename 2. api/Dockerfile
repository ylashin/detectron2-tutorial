FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

# Detectron2 prerequisites
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html

# Development packages
RUN pip install flask flask-cors requests opencv-python

WORKDIR /app

COPY web_api.py web_api.py

ENTRYPOINT ["python", "/app/web_api.py"]
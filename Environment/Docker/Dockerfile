FROM ubuntu:20.04
FROM tensorflow/tensorflow:2.4.1-gpu

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update --fix-missing && \
    apt-get install -y curl ffmpeg git graphviz libbz2-dev liblzma-dev libssl-dev libffi-dev nano python-pydot wget python-tk python3-tk python3.7-tk tk-dev

ENV PATH="/python3.7.7/bin:${PATH}"
RUN mkdir python3.7.7 && cd python3.7.7 && \
    wget https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tgz && tar xzf Python-3.7.7.tgz && cd Python-3.7.7 && \
    ./configure --prefix=/python3.7.7 --enable-optimizations && make install && \
    cd .. && rm -rf Python-3.7.7 Python-3.7.7.tgz && \
    update-alternatives --install /usr/local/bin/python python /python3.7.7/bin/python3.7 20 && \
    source /etc/environment && export PATH

COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

WORKDIR workspace/NeuralNetwork/FieldBoundary

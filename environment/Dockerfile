# hash:sha256:d7215f1bf06cb95b87a72be6ebbeac41d5152b297930bed5d5eaec2625296b1e
FROM registry.codeocean.com/codeocean/miniconda3:4.3.30-python3.6.3-ubuntu16.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyter==1.0.0 \
        keras==2.2.4 \
        numpy==1.18.1 \
        pandas==1.0.3 \
        scikit-learn==0.22.1 \
    && conda clean -ya

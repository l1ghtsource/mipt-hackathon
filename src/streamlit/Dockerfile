FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN sudo apt install build-essential python3-dev libopenblas-dev -y

RUN pip install -U git+https://github.com/richlukich/MinkowskiEngine -v --no-deps \
    --config-settings="--install-option=--force_cuda" \
    --config-settings="--install-option=--blas=openblas"

RUN pip install git+https://github.com/alexmelekhin/open_place_recognition.git@feat/itlp_outdoor

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501
COPY . .
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
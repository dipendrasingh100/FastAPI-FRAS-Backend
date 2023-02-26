FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libgtk-3-dev \
        libboost-all-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
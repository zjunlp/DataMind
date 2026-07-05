FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libhdf5-dev \
    bedtools \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the application code
COPY . /app
WORKDIR /app

# Set up environment
ENV PORT=8432
ENV CODE_FOLDER=""
ENV PYTHONWARNINGS="ignore::FutureWarning"

# Run the application
CMD ["python", "main.py"] 
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*
    
# Install dependencies once during build (not at runtime)
COPY requirements.txt /tmp/requirements.txt
# RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir nltk 
    # python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
COPY nltk_data /usr/local/share/nltk_data
ENV NLTK_DATA=/usr/local/share/nltk_data

# Copy the application code
COPY . /app
WORKDIR /app

# Set up environment
ENV PORT=8432
ENV CODE_FOLDER=""
ENV PYTHONWARNINGS="ignore::FutureWarning"

# Run the application
CMD ["python", "main.py"] 
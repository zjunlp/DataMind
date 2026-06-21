# MLE Bench + DSGym kernel image
#
# Based on our DSGym Kaggle image; extended to also run the MLE Bench grading server
# (listens on :5000) while keeping the persistent Jupyter kernel API (on :8432).
#
# Build context should be the repo root, so we can COPY files from executors/ and src/.

FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++
# -----------------------------
# Copy DSGym kernel server files
# -----------------------------
# Copy DSGym files
# We reuse the FastAPI + Jupyter kernel executor implementation from the Kaggle image
COPY executors/container_images/kaggle_image/main.py /app/main.py
COPY executors/container_images/kaggle_image/output_cleaning.py /app/output_cleaning.py
COPY executors/container_images/kaggle_image/kernel_executor.py /app/kernel_executor.py
COPY executors/container_images/kaggle_image/requirements.kaggle.txt /tmp/requirements.kaggle.txt


# Copy MLE Files
# Make private directory (root) owner-only. Grading server will be added here, later in the build
# The test set answers will be added here separately via a mounted docker volume
RUN mkdir /private && chmod 700 /private
# Copy over relevant files
COPY executors/container_images/mle_image/grading_server.py /private/grading_server.py
COPY executors/container_images/mle_image/instructions.txt /app/instructions.txt
COPY executors/container_images/mle_image/instructions_obfuscated.txt /app/instructions_obfuscated.txt
COPY executors/container_images/mle_image/validate_submission.sh /app/validate_submission.sh
COPY executors/container_images/mle_image/entrypoint.sh /app/entrypoint.sh
COPY executors/container_images/mle_image/requirements.txt /tmp/requirements.txt
ARG MLE_PATH=examples/MLE_Bench_Eval/mle-bench/
COPY ${MLE_PATH} /mlebench

# -----------------------------
# Environment and ports
# -----------------------------
ENV PORT=8432 \
    CODE_FOLDER="/code" \
    SUBMISSION_DIR=/home/submission \
    LOGS_DIR=/home/logs \
    AGENT_DIR=/home/agent

EXPOSE 8432
EXPOSE 5000

# -----------------------------
# Dependencies
# -----------------------------

# Add minimal extra deps for the grading server and MLE Bench utils
RUN pip install --no-cache-dir -r /tmp/requirements.kaggle.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# Install MLE Bench if provided in build context under ./mle-bench
RUN pip install flask && \
    pip install -e /mlebench


# -----------------------------
# Runtime dirs
# -----------------------------

# -----------------------------
# System user and base layout
# -----------------------------
RUN groupadd -g 243000848 appuser && \
    useradd -r -u 243000848 -g appuser appuser
RUN chmod +x /app/entrypoint.sh

WORKDIR /app


# Create submission directory and set ownership
RUN mkdir -p /submission && chown -R appuser:appuser /submission /app
# RUN chmod -R 777 /submission /app


# Switch to non-root user for execution
USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]


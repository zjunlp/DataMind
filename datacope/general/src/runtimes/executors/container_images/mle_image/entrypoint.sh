#!/usr/bin/env bash
# Print commands and their arguments as they are executed
set -x


# Start the MLE Bench grading server (Flask) on :5000
# log into /home/logs
LOGS_DIR=/app/logs
mkdir -p $LOGS_DIR

ls -l /app

echo "Starting grading server..."
python /private/grading_server.py >> "$LOGS_DIR/grading_server.log" 2>&1 &


# Start the Jupyter kernel FastAPI server (our standard DSGym executor) on :8432
python /app/main.py




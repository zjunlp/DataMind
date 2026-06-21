#!/bin/bash

set -e

echo "🔥 GPU Integration Test"
echo "======================"
echo "Tests GPU access through the executor system API"
echo ""

# Configuration
MANAGER_IMAGE="${MANAGER_IMAGE:-manager-prebuilt}"
EXECUTOR_IMAGE="${EXECUTOR_IMAGE:-executor-prebuilt}"
NUM_CONTAINERS=2
GPU_IDS="0,1"

# Change to project root
cd "$(dirname "$0")/.."

echo "🔧 Configuration:"
echo "  Manager image: $MANAGER_IMAGE"
echo "  Executor image: $EXECUTOR_IMAGE"
echo "  Containers: $NUM_CONTAINERS"
echo "  GPU IDs: $GPU_IDS"
echo ""

# Check for NVIDIA Docker support
echo "🔍 Checking NVIDIA Docker support..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found! GPU testing requires NVIDIA drivers"
    exit 1
fi

# Generate GPU-enabled compose file
echo "🔧 Generating GPU-enabled Docker Compose configuration..."
python generate_compose.py \
    --num-containers $NUM_CONTAINERS \
    --types $EXECUTOR_IMAGE:$NUM_CONTAINERS \
    --gpu-ids $GPU_IDS \
    --output docker-compose-gpu-test.yml

if [ ! -f "docker-compose-gpu-test.yml" ]; then
    echo "❌ Failed to generate docker-compose-gpu-test.yml"
    exit 1
fi

echo "✅ Generated docker-compose-gpu-test.yml"

# Build images if they don't exist
echo "🔍 Checking if images exist..."
images_need_build=false

if ! docker image inspect "$MANAGER_IMAGE" &> /dev/null; then
    echo "⚠️  Manager image $MANAGER_IMAGE not found"
    images_need_build=true
fi

if ! docker image inspect "$EXECUTOR_IMAGE" &> /dev/null; then
    echo "⚠️  Executor image $EXECUTOR_IMAGE not found"  
    images_need_build=true
fi

if [ "$images_need_build" = true ]; then
    echo "🏗️  Building required images..."
    
    # Build manager image
    echo "📦 Building manager image..."
    cd manager
    docker build -t "$MANAGER_IMAGE" .
    cd ..
    
    # Build executor image  
    echo "📦 Building executor image..."
    cd container_images/instance
    docker build -t "$EXECUTOR_IMAGE" .
    cd ../..
    
    echo "✅ Images built successfully"
else
    echo "✅ All images exist"
fi

# Clean up any existing test containers
echo "🧹 Cleaning up existing test containers..."
docker compose -f docker-compose-gpu-test.yml down 2>/dev/null || true

# Start the GPU test environment
echo "🚀 Starting GPU test environment..."
docker compose -f docker-compose-gpu-test.yml up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Wait for services to be ready
echo "🔍 Waiting for services to become ready..."
for attempt in {1..12}; do
    echo "  Attempt $attempt/12..."
    all_ready=true
    
    # Check manager health
    if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo "    Manager not ready yet"
        all_ready=false
    fi
    
    # Check container readiness via manager
    for container_id in $(seq 0 $((NUM_CONTAINERS-1))); do
        if ! curl -s http://localhost:5000/session/$container_id/ready | grep -q '"ready":true'; then
            echo "    Container $container_id not ready yet"
            all_ready=false
        fi
    done
    
    if [ "$all_ready" = true ]; then
        echo "  ✅ All services are ready!"
        break
    fi
    
    if [ $attempt -lt 12 ]; then
        echo "    Waiting 5 seconds before next attempt..."
        sleep 5
    fi
done

if [ "$all_ready" != true ]; then
    echo "❌ Services failed to become ready!"
    echo "📊 Docker logs:"
    docker compose -f docker-compose-gpu-test.yml logs --tail=20
    docker compose -f docker-compose-gpu-test.yml down
    rm -f docker-compose-gpu-test.yml container_config.json
    exit 1
fi

# Run the GPU test
echo ""
echo "🎯 Running GPU access tests through API..."
python tests/test_gpu_execution.py

test_exit_code=$?

echo ""
echo "📋 GPU TEST COMPLETE"
echo "===================="

# Clean up
echo "🧹 Cleaning up test environment..."
docker compose -f docker-compose-gpu-test.yml down
rm -f docker-compose-gpu-test.yml container_config.json

if [ $test_exit_code -eq 0 ]; then
    echo "✅ GPU integration test PASSED"
    echo "   All containers can access their assigned GPUs through the API"
else
    echo "❌ GPU integration test FAILED"
    echo "   Check the output above for details"
fi

exit $test_exit_code

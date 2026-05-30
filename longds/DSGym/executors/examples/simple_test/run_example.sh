#!/bin/bash

set -e

echo "🚀 Starting Simple Test Example"
echo "================================"

# Docker-compose will build images automatically
echo "⚙️  Using existing docker-compose configuration..."

# Container config already exists as container_config.json
echo "📝 Using existing container configuration..."

# Start all services (manager + 4 containers)
echo "🐳 Starting manager and 4 containers..."
sudo docker-compose -f docker-compose-simple.yml up -d --build

echo "⏳ Waiting for services to start..."
sleep 25

# Wait for all services to be ready
echo "🔍 Waiting for all services to become ready..."
for attempt in {1..10}; do
    echo "  Attempt $attempt/10..."
    all_ready=true
    
    # Check manager health
    if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo "    Manager not ready yet"
        all_ready=false
    fi
    
    # Check container readiness via manager
    for container_id in {1..4}; do
        if ! curl -s http://localhost:5000/session/$container_id/ready | grep -q '"ready":true'; then
            echo "    Container $container_id not ready yet"
            all_ready=false
        fi
    done
    
    if [ "$all_ready" = true ]; then
        echo "  ✅ All services are ready!"
        break
    fi
    
    if [ $attempt -lt 10 ]; then
        echo "    Waiting 10 seconds before next attempt..."
        sleep 10
    fi
done

if [ "$all_ready" != true ]; then
    echo "⚠️  Some services may not be fully ready, but continuing with demo..."
fi

echo ""
echo "🚀 RUNNING CODE EXECUTION DEMONSTRATION"
echo "========================================"

# Install requests if not available
if ! python3 -c "import requests" 2>/dev/null; then
    echo "📦 Installing requests library..."
    pip3 install requests
fi

# Run the comprehensive test
echo "🎯 Starting comprehensive code execution test..."
python3 test_simple.py

test_exit_code=$?

echo ""
echo "📋 EXAMPLE COMPLETE"
echo "==================="
if [ $test_exit_code -eq 0 ]; then
    echo "✅ Code execution demonstration completed successfully!"
    echo ""
    echo "🔗 System endpoints:"
    echo "   Manager: http://localhost:5000"
    echo "   Containers: ports 60000-60003"
    echo ""
    echo "💡 You can now:"
    echo "   • Execute code: curl -X POST http://localhost:5000/session/1/execute -H 'Content-Type: application/json' -d '{\"code\":\"print(\\\"Hello World\\\")\"}'"
    echo "   • Check status: curl http://localhost:5000/session/1/ready"
    echo "   • Restart kernel: curl -X POST http://localhost:5000/session/1/restart"
else
    echo "❌ Code execution demonstration failed"
    echo "   Check the output above for details"
fi

echo ""
echo "🧹 Cleanup: sudo docker-compose -f docker-compose-simple.yml down" 
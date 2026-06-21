# Executor Instance - Reference Implementation

This directory contains the **reference implementation** of an executor container using Jupyter kernels with FastAPI. This is just one example - you can implement executor containers in any language/framework as long as they follow the required HTTP API contract.

## Features

- **Jupyter Kernel Integration**: Runs Python code in isolated Jupyter kernel
- **Mountable Volumes**: Load custom Python code into the execution environment
- **Kernel Lifecycle Management**: Start, restart, and monitor kernel health
- **FastAPI Interface**: HTTP API that matches manager contract requirements
- **Error Handling**: Proper timeout and error reporting
- **Variable Persistence**: State maintained across executions within same container

## Required API Endpoints

This implementation provides all required endpoints for manager compatibility:

- `GET /health` - Return `{"status": "ok"}` if container is healthy
- `GET /ready` - Return `{"ready": true/false}` with kernel readiness
- `POST /execute` - Execute code and return outputs 
- `POST /restart` - Restart Jupyter kernel (clears all state)

## Mountable Volumes

The key feature of this implementation is the ability to load custom Python code into the execution environment.

### How It Works

When you mount a directory to `/code:ro` in the container:

1. **Container starts** and initializes Jupyter kernel
2. **Code loading**: All `.py` files in `/code` are automatically executed during startup
3. **Global availability**: Functions, classes, and variables become available in the kernel
4. **Persistence**: Loaded code remains available for all subsequent executions

### Basic Usage

```bash
# Generate compose with your code directory
python generate_compose.py \
  --mountable-volumes /path/to/your/python/code \
  --num-containers 4 \
  --types python:4 \
  --output with-code.yml

# Start containers
docker compose -f with-code.yml up -d

# Your Python files are now loaded and available
curl -X POST http://localhost:5000/session/0/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "your_function_from_mounted_code()"}'
```

### Example with Custom Code

Create your Python utilities:

```bash
mkdir my_utilities/

cat > my_utilities/data_processing.py << 'EOF'
def process_list(data):
    return [x * 2 for x in data if x > 0]

def calculate_stats(numbers):
    return {
        'sum': sum(numbers),
        'avg': sum(numbers) / len(numbers),
        'count': len(numbers)
    }

print("Data processing utilities loaded")
EOF

cat > my_utilities/formatters.py << 'EOF'
def format_currency(amount):
    return f"${amount:,.2f}"

def format_percentage(value):
    return f"{value:.1f}%"

print("Formatters loaded")
EOF
```

Deploy and use:

```bash
# Deploy with your utilities
python generate_compose.py \
  --mountable-volumes $(pwd)/my_utilities \
  --num-containers 2 \
  --types python:2 \
  --output utilities-demo.yml

docker compose -f utilities-demo.yml up -d

# Use your functions
curl -X POST http://localhost:5000/session/0/execute \
  -H "Content-Type: application/json" \
  -d '{
    "code": "
data = [1, -2, 3, 4, -1, 5]
processed = process_list(data)
stats = calculate_stats(processed)
print(f\"Processed: {processed}\")
print(f\"Stats: {stats}\")
print(f\"Total: {format_currency(stats[\"sum\"])}\")
"}'
```

Expected output:
```
Processed: [2, 6, 8, 10]
Stats: {'sum': 26, 'avg': 6.5, 'count': 4}
Total: $26.00
```

### What Gets Loaded

- **All `.py` files** in the mounted directory
- **Executed during startup** - before the container accepts requests
- **Global scope** - functions and variables available everywhere
- **Import statements** - can import other modules
- **Print statements** - shown in container logs during startup

### Use Cases

- **Utility functions**: Data processing, formatting, calculations
- **ML models**: Load trained models and inference functions
- **Business logic**: Domain-specific functions and workflows  
- **Configuration**: Constants, settings, connection strings
- **Libraries**: Custom modules and packages

### Testing Your Code

Test with the provided examples:

```bash
# Use the included test code
python generate_compose.py \
  --mountable-volumes $(pwd)/../tests/test_code \
  --num-containers 2 \
  --types python:2 \
  --output test-workspace.yml

docker compose -f test-workspace.yml up -d

# Run the test suite
python ../tests/test_code_folder.py
```

The test code includes:
- `calculate_circle_area(radius)` - Mathematical calculations
- `greet(name)` - String formatting
- `factorial(n)` - Recursive functions
- `add_numbers(a, b)` - Basic operations

### Environment Variables

The instance recognizes these environment variables:

- `WORKSPACE_FOLDER` - Override code loading directory (default: `/code`)
- `PORT` - HTTP server port (default: `8432`)
- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `PYTHONDONTWRITEBYTECODE=1` - Skip `.pyc` file generation

### Kernel Management

The Jupyter kernel can be restarted to clear all state:

```bash
# Restart kernel (clears all variables and imported modules)
curl -X POST http://localhost:5000/session/0/restart

# Check if restart is complete
curl http://localhost:5000/session/0/ready

# Mounted code will be reloaded automatically
```

### Implementation Details

- **Timeout**: Code execution times out after 3 seconds (configurable)
- **Memory**: Containers use 256MB RAM by default
- **CPU**: 0.25 CPU cores per container
- **Threading**: Thread-safe kernel management with locks
- **Error handling**: Proper exception capture and reporting

### Building the Image

```bash
# Build the instance image
docker build -t python ./

# Or build with custom tag
docker build -t my-executor-instance ./
```

### Custom Implementation Guide

To create your own executor implementation based on this:

1. **Choose your runtime** (Node.js, Go, Rust, etc.)
2. **Implement the 4 required HTTP endpoints**
3. **Add your own code loading mechanism** (if desired)
4. **Build Docker image**
5. **Use in compose configurations**

The manager doesn't care what's inside your executor containers as long as they implement the HTTP contract.

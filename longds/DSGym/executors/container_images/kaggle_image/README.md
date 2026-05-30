# Kaggle Image

## 🎯 Two Build Approaches

### 1. Local Build (Recommended for Development)
**✅ No Docker Hub login required**  

```bash
# Basic local build (uses 'local' as tag prefix)
./build-local.sh
```

**Advanced usage with custom tags:**
```bash
# Use your own username as tag prefix
LOCAL_USERNAME=your-name ./build-local.sh
```

### 2. Pull Pre-built Images
**⚡ Fastest option**  
**📦 No building required**

```bash
# Download pre-built images from Docker Hub
./pull.sh
```

### (Optional) Push Your Local Image

```bash
# First docker login set your local username
# Then push your local task-specific image to your docker hub
./push-local.sh
```

## 🔧 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BASE_DOCKER_USERNAME` | `fannie437` | Where to pull the base image from |
| `LOCAL_USERNAME` | `local` | Local tag prefix for built images |

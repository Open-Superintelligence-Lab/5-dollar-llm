# Docker Setup

This document provides instructions for running the 5-dollar-llm project using Docker containers with GPU support.

## Prerequisites

### NVIDIA Docker Support
Ensure you have NVIDIA Container Toolkit installed:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Docker Compose
Ensure you have Docker Compose installed (version 3.8+).

## Quick Start

### 1. Clone and Build
```bash
git clone <repository-url>
cd 5-dollar-llm
docker-compose -f docker/docker-compose.yml build
```

### 2. Run Training Container
```bash
docker-compose -f docker/docker-compose.yml up llm-training
```

This will:
- Download the 40M token subset dataset
- Start the training process automatically
- Mount volumes for data, checkpoints, and logs

### 3. Run with Jupyter Lab (Optional)
For interactive development and experimentation:
```bash
docker-compose -f docker/docker-compose.yml up jupyter
```
Access Jupyter Lab at `http://localhost:8888` (no password required).

## Docker Configuration

**Note**: The `docker-compose.yml` file uses modern Docker Compose syntax without the obsolete `version` field. If you see warnings about the version attribute, you can safely ignore them.

### Base Image
- **PyTorch**: `pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel`
- CUDA 13.0 with cuDNN 9 support
- Optimized for GPU training workloads

### Services

#### llm-training
- **Container Name**: `five-dollar-llm`
- **GPU Support**: All available NVIDIA GPUs
- **Volumes**:
  - `./:/app` - Source code (development)
  - `./data:/app/data` - Dataset storage
  - `./checkpoints:/app/checkpoints` - Model checkpoints
  - `./logs:/app/logs` - Training logs
- **Entry Point**: Downloads 40M token subset and starts training

#### jupyter
- **Container Name**: `five-dollar-llm-jupyter`
- **Port**: `8888` (mapped to host)
- **GPU Support**: All available NVIDIA GPUs
- **Volumes**: Same as training + `./notebooks:/app/notebooks`
- **Access**: No authentication token or password

## Volumes and Data Persistence

### Essential Directories
- **`./data/`**: Datasets and processed data
- **`./checkpoints/`**: Model checkpoints (saved during training)
- **`./logs/`**: Training logs and metrics
- **`./notebooks/`**: Jupyter notebooks (if using Jupyter service)

### Automatic Directory Creation
The Dockerfile automatically creates these directories inside the container:
```dockerfile
RUN mkdir -p /app/data /app/models /app/logs /app/checkpoints
```

## Environment Variables

### Default Environment
- `PYTHONPATH=/app:$PYTHONPATH` - Ensures proper module imports
- `TOKENIZERS_PARALLELISM=false` - Prevents tokenizer warnings
- `NVIDIA_VISIBLE_DEVICES=all` - Exposes all GPUs to container
- `CUDA_VISIBLE_DEVICES=0` - Default GPU selection

### Customization
You can override environment variables in `docker-compose.yml` or via command line:
```bash
CUDA_VISIBLE_DEVICES=0,1 docker-compose up llm-training
```

## Development Workflow

### 1. Making Changes
Since the source code is mounted as a volume, changes on your host are reflected immediately in the container.

### 2. Installing New Dependencies
Add dependencies to `requirements.txt`, then rebuild:
```bash
docker-compose build --no-cache
```

### 3. Running Specific Commands
Execute commands inside the running container:
```bash
docker-compose -f docker/docker-compose.yml exec llm-training python train_llm.py --help
docker-compose -f docker/docker-compose.yml exec llm-training bash
```

### 4. Monitoring Training
View logs in real-time:
```bash
docker-compose -f docker/docker-compose.yml logs -f llm-training
```

## GPU Configuration

### Multi-GPU Training
The container supports multi-GPU setups. To use multiple GPUs:
```bash
docker-compose -f docker/docker-compose.yml up llm-training  # Uses all available GPUs
# Or specify specific GPUs
CUDA_VISIBLE_DEVICES=0,1 docker-compose -f docker/docker-compose.yml up llm-training
```

### GPU Memory Management
Monitor GPU usage:
```bash
docker-compose -f docker/docker-compose.yml exec llm-training nvidia-smi
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check NVIDIA Docker installation
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### 2. Permission Issues
Ensure proper ownership of mounted volumes:
```bash
sudo chown -R $USER:$USER ./data ./checkpoints ./logs
```

#### 3. Build Issues
Clear Docker cache and rebuild:
```bash
docker-compose -f docker/docker-compose.yml down
docker system prune -f
docker-compose -f docker/docker-compose.yml build --no-cache
```

#### 4. Port Conflicts
If port 8888 is occupied, modify `docker-compose.yml`:
```yaml
ports:
  - "8889:8888"  # Use different host port
```

### Debug Mode
Run container with interactive shell:
```bash
docker-compose -f docker/docker-compose.yml run --rm llm-training bash
```

## Production Considerations

### 1. Resource Limits
Add resource constraints to `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 32G
      cpus: '8'
```

### 2. Security
- Remove development volume mounts in production
- Add proper authentication to Jupyter service
- Use specific image tags instead of `latest`

### 3. Monitoring
Consider adding monitoring services (Prometheus, Grafana) for production deployments.

## Advanced Usage

### Custom Training Scripts
Override the default command:
```bash
docker-compose -f docker/docker-compose.yml run --rm llm-training python your_custom_script.py
```

### Distributed Training
For multi-node training, modify the `docker-compose.yml` to include networking and service discovery.

### Integration with CI/CD
The Docker setup can be integrated into CI/CD pipelines:
```yaml
# Example GitHub Actions
- name: Run training tests
  run: |
    docker-compose -f docker/docker-compose.yml up --abort-on-container-exit
```

## File Structure
```
.
├── docker/
│   ├── Dockerfile              # Main container definition
│   ├── docker-compose.yml      # Service orchestration
│   ├── entrypoint.sh          # Container startup script
│   └── .dockerignore          # Docker ignore patterns
├── requirements.txt           # Python dependencies
├── data/                      # Dataset storage (mounted)
├── checkpoints/               # Model checkpoints (mounted)
├── logs/                      # Training logs (mounted)
└── notebooks/                 # Jupyter notebooks (optional)
```

## Additional Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [PyTorch Docker Hub](https://hub.docker.com/r/pytorch/pytorch)
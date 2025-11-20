# Docker Build Troubleshooting

Common Docker build issues and their solutions.

## No Space Left on Device

**Error**: `no space left on device` during build

### Quick Fix

```bash
# Run cleanup script
./scripts/prebuild_cleanup.sh

# Or manually:
docker system prune -a -f --volumes
docker builder prune -a -f
```

### Check Available Space

```bash
# Check Docker disk usage
docker system df

# Check host disk space
df -h
```

### Increase Docker Disk Space

#### macOS (Docker Desktop)

1. Open Docker Desktop
2. Settings → Resources → Disk image size
3. Increase to **100 GB**
4. Apply & Restart
5. Retry build

#### Windows (Docker Desktop + WSL2)

1. Stop WSL:
   ```powershell
   wsl --shutdown
   ```

2. Locate WSL disk image:
   ```
   %USERPROFILE%\AppData\Local\Docker\wsl\data\ext4.vhdx
   ```

3. Increase size using `diskpart` or:
   ```powershell
   wsl --manage <distro> --set-sparse true
   ```

4. Restart Docker Desktop

#### Linux

Docker uses host filesystem directly. Free up space:

```bash
# Clean package cache
sudo apt-get clean  # Ubuntu/Debian
sudo yum clean all  # RHEL/CentOS

# Remove old logs
sudo journalctl --vacuum-time=7d

# Check large files
sudo du -h / | sort -rh | head -n 20
```

### Build on EC2 (Recommended)

If local space is limited, build on EC2:

```bash
# Launch t3.xlarge with 200GB EBS
aws ec2 run-instances \
  --image-id ami-0c9c942bd7bf113a2 \
  --instance-type t3.xlarge \
  --block-device-mappings '[{
    "DeviceName": "/dev/xvda",
    "Ebs": {"VolumeSize": 200, "VolumeType": "gp3"}
  }]' \
  --region ap-northeast-2
```

See [Build on EC2](./BUILD.md#build-on-ec2-recommended-for-large-builds) for details.

## Build Timeout or Hanging

**Symptoms**: Build runs for hours without progress

### Check Docker Resources

#### Docker Desktop

1. Settings → Resources
2. Increase:
   - **CPUs**: 4-8 cores
   - **Memory**: 8-16 GB
   - **Swap**: 2-4 GB

#### Linux

```bash
# Check system resources
htop
free -h
df -h
```

### Kill Stalled Build

```bash
# Stop all builds
docker ps -q | xargs docker stop

# Remove build containers
docker container prune -f

# Clear buildx cache
docker buildx prune -a -f
```

### Build Without Cache

```bash
docker buildx build \
  --no-cache \
  --platform linux/amd64 \
  -t your-image:latest \
  --push \
  .
```

## Cannot Pull Base Image

**Error**: `failed to pull nvidia/cuda:12.1.1-devel-ubuntu22.04`

### Solutions

```bash
# 1. Check internet connectivity
ping hub.docker.com

# 2. Try pulling manually
docker pull nvidia/cuda:12.1.1-devel-ubuntu22.04

# 3. Use different mirror (if in China)
docker pull dockerproxy.com/nvidia/cuda:12.1.1-devel-ubuntu22.04

# 4. Check Docker Hub rate limits
# Login to Docker Hub
docker login
```

## ECR Login Issues

**Error**: `error getting credentials`, `no basic auth credentials`

### Solutions

```bash
# 1. Check AWS credentials
aws sts get-caller-identity

# 2. Reconfigure AWS CLI
aws configure

# 3. Login to ECR again
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  236357498583.dkr.ecr.ap-northeast-2.amazonaws.com

# 4. Check ECR repository exists
aws ecr describe-repositories \
  --repository-names 3dgs/builder \
  --region ap-northeast-2

# 5. Create repository if missing
aws ecr create-repository \
  --repository-name 3dgs/builder \
  --region ap-northeast-2
```

## Permission Denied

**Error**: `permission denied while trying to connect to Docker daemon`

### Linux

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or:
newgrp docker

# Verify
docker ps
```

### macOS/Windows

- Restart Docker Desktop
- Check Docker Desktop is running

## Build Fails During Python Package Installation

**Error**: Package installation fails or times out

### Common Issues

#### 1. PyTorch Installation Fails

```bash
# Test manually
docker run --rm -it nvidia/cuda:12.1.1-devel-ubuntu22.04 bash
pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

**Fix**: Check network, use local PyPI mirror if needed

#### 2. CUDA Extension Compilation Fails

**Error**: `error: command 'gcc' failed`

**Fix**: Ensure CUDA toolkit is properly installed (already in Dockerfile)

#### 3. Out of Memory During Build

```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 8-16GB
```

## buildx Not Found

**Error**: `docker: 'buildx' is not a docker command`

### Solutions

```bash
# 1. Update Docker (buildx included in 19.03+)
docker --version

# 2. Install buildx manually
mkdir -p ~/.docker/cli-plugins
curl -Lo ~/.docker/cli-plugins/docker-buildx \
  https://github.com/docker/buildx/releases/latest/download/buildx-linux-amd64
chmod +x ~/.docker/cli-plugins/docker-buildx

# 3. Verify
docker buildx version
```

## Platform Mismatch Warning

**Warning**: `The requested image's platform (linux/amd64) does not match`

This is expected on Apple Silicon (M1/M2) Macs. The warning is safe to ignore when building for AWS (which uses linux/amd64).

To suppress:
```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

## CUDA/GPU Issues in Built Image

**Error**: `CUDA not available` when running container

### Verify NVIDIA Runtime

```bash
# Check nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi

# If fails, install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Image Too Large

**Issue**: Final image larger than expected (>10GB)

### Investigate

```bash
# Check layer sizes
docker history your-image:latest

# Use dive tool
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  wagoodman/dive:latest your-image:latest
```

### Optimize

Already optimized in Dockerfile:
- ✅ Multi-stage build
- ✅ Cleaned Python cache
- ✅ Stripped binaries
- ✅ Removed build dependencies

## Network Connectivity Issues

**Error**: Timeouts downloading packages

### Use Proxy

```bash
docker buildx build \
  --build-arg HTTP_PROXY=http://proxy:port \
  --build-arg HTTPS_PROXY=http://proxy:port \
  ...
```

### Use Mirror (China)

Edit Dockerfile to use mirrors:
```dockerfile
# Add before pip install
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## Common Error Messages

### "failed to solve with frontend dockerfile.v0"

**Fix**: Update Docker, use newer BuildKit

### "error checking context: 'can't stat'"

**Fix**: Check .dockerignore, exclude large files

### "executor failed running"

**Fix**: Check previous step's logs for actual error

### "layer does not exist"

**Fix**: Clear build cache, rebuild

## Debug Build Process

### Verbose Logging

```bash
docker buildx build \
  --progress=plain \
  -t your-image:latest \
  .
```

### Build Specific Stage

```bash
# Build only builder stage
docker buildx build \
  --target builder \
  -t test-builder:latest \
  .

# Test builder stage
docker run --rm -it test-builder:latest bash
```

### Interactive Debugging

```bash
# Build up to failing step, commit, debug
docker build -t debug:latest .  # Let it fail
docker run --rm -it $(docker ps -lq) bash  # Debug last container
```

## Performance Optimization

### Enable BuildKit

```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

### Use Build Cache

```bash
# Save cache
docker buildx build \
  --cache-to type=local,dest=/tmp/cache \
  ...

# Use cache
docker buildx build \
  --cache-from type=local,src=/tmp/cache \
  ...
```

## Getting Help

If issues persist:

1. **Check Docker logs**:
   ```bash
   # Linux
   journalctl -u docker.service
   
   # macOS
   # Docker Desktop → Troubleshoot → Show logs
   ```

2. **Check disk space** (most common issue):
   ```bash
   docker system df
   df -h
   ```

3. **Clean everything and retry**:
   ```bash
   ./scripts/prebuild_cleanup.sh
   ./scripts/build_and_push.sh
   ```

4. **Build on EC2** (most reliable):
   - See [BUILD.md - Build on EC2](./BUILD.md#build-on-ec2-recommended-for-large-builds)

## Related Documentation

- [Build Guide](./BUILD.md)
- [Dockerfile Reference](./DOCKERFILE_REFERENCE.md)
- [AWS Batch Setup](../aws/SETUP.md)

# Docker Build Guide for 3DGS Pipeline

Complete guide for building and pushing the Docker image to AWS ECR.

## 📚 Quick Links

- **[Build Instructions](#build-instructions)** - Step-by-step build process
- **[Troubleshooting](./TROUBLESHOOTING.md)** - Common build issues
- **[Dockerfile Reference](./DOCKERFILE_REFERENCE.md)** - Explanation of Dockerfile stages

## Prerequisites

- Docker Desktop or Docker Engine installed
- Docker buildx plugin (included in Docker Desktop)
- AWS CLI configured with ECR access
- **60GB+ free disk space** (recommended: 100GB)

## Quick Start

### Automated Build (Recommended)

```bash
# 1. Check disk space and clean if needed
./scripts/prebuild_cleanup.sh

# 2. Build and push to ECR
./scripts/build_and_push.sh
```

### Manual Build

```bash
# 1. Login to ECR
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  236357498583.dkr.ecr.ap-northeast-2.amazonaws.com

# 2. Build and push
docker buildx build \
  --platform linux/amd64 \
  -t 236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:latest \
  --push \
  .
```

## Build Instructions

### Step 1: Prerequisites Check

```bash
# Check Docker version
docker --version  # Should be 20.10+ 

# Check buildx
docker buildx version

# Check disk space
df -h .  # Should have 60GB+ free

# Check AWS credentials
aws sts get-caller-identity
```

### Step 2: Pre-Build Cleanup (Optional)

```bash
# Run cleanup script
chmod +x scripts/prebuild_cleanup.sh
./scripts/prebuild_cleanup.sh

# Or manually clean Docker
docker system prune -a -f
docker builder prune -a -f
```

### Step 3: ECR Login

```bash
# Login to ECR repository
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  236357498583.dkr.ecr.ap-northeast-2.amazonaws.com
```

### Step 4: Build Image

```bash
# Option A: Using build script (recommended)
chmod +x scripts/build_and_push.sh
./scripts/build_and_push.sh

# Option B: Manual build with custom tag
docker buildx build \
  --platform linux/amd64 \
  --build-arg ARCHS="8.0;8.6;8.9" \
  -t 236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:v1.0 \
  --push \
  .
```

### Step 5: Verify

```bash
# Check image in ECR
aws ecr describe-images \
  --repository-name 3dgs/builder \
  --region ap-northeast-2

# Test image locally
docker run --rm --gpus all \
  -e PROJECT=test \
  236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:latest \
  --help
```

## Build Configuration

### CUDA Architectures

The Dockerfile supports multiple CUDA architectures:

```dockerfile
ARG ARCHS="7.0;7.5;8.0;8.6;8.9"
```

**Supported architectures:**
- 7.0: V100 (p3 instances)
- 7.5: T4 (g4dn instances)
- 8.0: A100 (p4d instances)
- 8.6: A10G (g5 instances)
- 8.9: H100 (future support)

**Custom build for specific GPU:**
```bash
docker buildx build \
  --build-arg ARCHS="8.6" \
  -t my-image:latest \
  .
```

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `ARCHS` | 7.0;7.5;8.0;8.6;8.9 | CUDA compute capabilities |
| `MAMBA_ROOT_PREFIX` | /opt/conda | Micromamba install path |
| `MAMBA_DOCKERFILE_ACTIVATE` | 1 | Auto-activate conda env |

## Multi-Stage Build

The Dockerfile uses a two-stage build:

### Stage 1: Builder
- CUDA development environment
- Compiles CUDA extensions
- Builds Python packages
- ~20GB in size

### Stage 2: Runtime
- CUDA runtime environment (smaller)
- Copies built artifacts from builder
- Final image ~8GB

This reduces final image size by 60%.

## Build Time

Expected build times:

| Hardware | Time |
|----------|------|
| Local (MacBook M1/M2) | 30-45 min |
| Local (Intel/AMD) | 25-35 min |
| EC2 t3.xlarge | 20-30 min |
| EC2 c5.2xlarge | 15-20 min |

## Disk Space Requirements

| Stage | Space Required |
|-------|----------------|
| Source code | ~500MB |
| Build cache | ~20GB |
| Builder stage | ~20GB |
| Runtime stage | ~8GB |
| **Total during build** | **~40-50GB** |
| **Final image** | **~8GB** |

## Build on EC2 (Recommended for Large Builds)

### Launch EC2 Instance

```bash
# Launch instance
aws ec2 run-instances \
  --image-id ami-0c9c942bd7bf113a2 \
  --instance-type t3.xlarge \
  --key-name your-key \
  --block-device-mappings '[
    {
      "DeviceName": "/dev/xvda",
      "Ebs": {
        "VolumeSize": 200,
        "VolumeType": "gp3"
      }
    }
  ]' \
  --region ap-northeast-2
```

### Build on EC2

```bash
# SSH to instance
ssh -i your-key.pem ec2-user@instance-ip

# Install Docker
sudo yum install -y docker git
sudo systemctl start docker
sudo usermod -a -G docker ec2-user

# Log out and back in, then:
git clone https://github.com/ShubhangiSRG24/3DGS_Pipeline.git
cd 3DGS_Pipeline

# Configure AWS credentials
aws configure

# Build
./scripts/build_and_push.sh
```

## Optimization Tips

### Speed Up Builds

1. **Use BuildKit cache**:
   ```bash
   export DOCKER_BUILDKIT=1
   ```

2. **Parallel builds** (if building multiple versions):
   ```bash
   docker buildx create --use --name multi-builder
   docker buildx build --builder multi-builder ...
   ```

3. **Reduce CUDA architectures**:
   ```bash
   # Only build for your target GPU
   docker buildx build --build-arg ARCHS="8.6" ...
   ```

### Reduce Image Size

Already optimized, but you can:
- Remove unused Python packages
- Strip binaries (already done)
- Use multi-stage build (already done)

### Build Cache

```bash
# Clear cache if needed
docker builder prune -a -f

# Or keep cache and build with --no-cache only when needed
docker buildx build --no-cache ...
```

## Troubleshooting

For common build issues, see:
- **[Docker Troubleshooting Guide](./TROUBLESHOOTING.md)**

Quick fixes:

```bash
# No space left on device
./scripts/prebuild_cleanup.sh

# Can't pull base image
docker pull nvidia/cuda:12.1.1-devel-ubuntu22.04

# Build hanging
# Check Docker Desktop resource allocation
# Increase CPU/Memory in Docker Desktop settings
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-northeast-2
      
      - name: Login to ECR
        run: |
          aws ecr get-login-password --region ap-northeast-2 | \
            docker login --username AWS --password-stdin \
            236357498583.dkr.ecr.ap-northeast-2.amazonaws.com
      
      - name: Build and push
        run: ./scripts/build_and_push.sh
```

## Image Management

### List Images

```bash
aws ecr describe-images \
  --repository-name 3dgs/builder \
  --region ap-northeast-2
```

### Tag Image

```bash
# Pull existing image
docker pull 236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:latest

# Tag with version
docker tag \
  236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:latest \
  236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:v1.0

# Push new tag
docker push 236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:v1.0
```

### Delete Old Images

```bash
# List all images
aws ecr list-images \
  --repository-name 3dgs/builder \
  --region ap-northeast-2

# Delete specific image
aws ecr batch-delete-image \
  --repository-name 3dgs/builder \
  --image-ids imageDigest=sha256:xxx \
  --region ap-northeast-2
```

## Next Steps

After successful build:

1. ✅ Image is in ECR
2. 📋 [Register AWS Batch job definition](../aws/SETUP.md)
3. 🚀 [Submit test job](../aws/QUICK_REFERENCE.md)

## Related Documentation

- [AWS Batch Setup](../aws/SETUP.md)
- [Dockerfile Reference](./DOCKERFILE_REFERENCE.md)
- [Troubleshooting](./TROUBLESHOOTING.md)

# Scripts Directory

Utility scripts for building, deploying, and running the 3DGS Pipeline.

## Build & Deploy Scripts

### `build_and_push.sh`

Builds Docker image and pushes to AWS ECR.

**Usage:**
```bash
# Build and push with 'latest' tag
./scripts/build_and_push.sh

# Build and push with custom tag
./scripts/build_and_push.sh v1.0
```

**What it does:**
1. Logs in to AWS ECR
2. Creates repository if it doesn't exist
3. Builds Docker image for linux/amd64
4. Pushes to ECR
5. Displays next steps

**Configuration:**
- ECR Registry: `236357498583.dkr.ecr.ap-northeast-2.amazonaws.com`
- Image Name: `3dgs/builder`
- Region: `ap-northeast-2`
- Platform: `linux/amd64`
- CUDA Archs: `8.0;8.6;8.9`

**Requirements:**
- AWS CLI configured
- Docker with buildx
- ECR permissions

### `prebuild_cleanup.sh`

Checks disk space and cleans Docker resources before building.

**Usage:**
```bash
./scripts/prebuild_cleanup.sh
```

**What it does:**
1. Checks Docker disk usage
2. Checks host disk space
3. Warns if insufficient space (<60GB)
4. Offers to clean Docker resources
5. Verifies space after cleanup

**Cleanup actions:**
- Stops containers
- Removes stopped containers
- Removes unused images
- Clears build cache
- Removes unused volumes

**Interactive:**
- Prompts before cleaning
- Shows space before/after
- Provides recommendations if still insufficient

## Runtime Scripts

### `entrypoint.sh`

Main entrypoint for Docker container. Orchestrates the complete 3DGS pipeline.

**Usage:**
```bash
# Run via Docker
docker run --rm -it --gpus all \
  -e PROJECT=my-scene \
  -e S3_INPUT_BUCKET=s3://bucket/input \
  -e S3_OUTPUT_BUCKET=s3://bucket/output \
  your-image:latest

# Run on AWS Batch (automatic)
```

**Pipeline Stages:**
1. **Extract**: Extract frames from video
2. **SfM**: Structure from Motion (COLMAP)
3. **Depth**: Depth estimation (Depth-Anything-V2)
4. **Train**: 3D Gaussian Splatting training

**Environment Variables:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PROJECT` | No | auto | Project/scene name |
| `VIDEO_URL` | No | - | URL of input video |
| `NUM_IMAGES` | No | 120 | Number of frames to extract |
| `SKIP_EXTRACT` | No | 0 | Skip frame extraction (1=yes) |
| `SKIP_SFM` | No | 0 | Skip Structure from Motion |
| `SKIP_DEPTH` | No | 0 | Skip depth estimation |
| `SKIP_TRAIN` | No | 0 | Skip training |
| `S3_INPUT_BUCKET` | No | - | S3 path for input images |
| `S3_OUTPUT_BUCKET` | No | - | S3 path for outputs |
| `AWS_REGION` | No | us-east-1 | AWS region |
| `DEPTH_CKPT_URL` | No | - | URL for depth model checkpoint |

**Features:**
- ✅ Timing for each stage
- ✅ CSV/TXT timing reports
- ✅ S3 sync (input/output)
- ✅ Automatic error handling
- ✅ Progress logging

**Outputs:**
- `data/<PROJECT>/images/` - Extracted frames
- `data/<PROJECT>/colmap/` - COLMAP model
- `data/<PROJECT>/depthmap/` - Depth maps
- `output/` - Trained Gaussian model
- `data/<PROJECT>/pipeline_times.csv` - Timing data
- `data/<PROJECT>/pipeline_times.txt` - Human-readable timing

## Usage Examples

### Local Development

```bash
# Clean before build
./scripts/prebuild_cleanup.sh

# Build and push
./scripts/build_and_push.sh

# Test locally
docker run --rm -it --gpus all \
  -e PROJECT=test \
  -e SKIP_EXTRACT=1 \
  -v "$PWD/data/test/images:/workspace/app/data/test/images:ro" \
  -v "$PWD/output:/workspace/app/output" \
  236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:latest
```

### AWS Batch

```python
# Submit job (uses entrypoint.sh automatically)
from submit_batch_job import BatchJobSubmitter

submitter = BatchJobSubmitter(region="ap-northeast-2")
submitter.submit_job(
    project_name="my-scene",
    s3_input_bucket="s3://bucket/input",
    s3_output_bucket="s3://bucket/output"
)
```

### CI/CD Pipeline

```bash
# In GitHub Actions or similar
- name: Build and push
  run: |
    aws configure set aws_access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws configure set aws_secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws configure set region ap-northeast-2
    ./scripts/build_and_push.sh ${{ github.sha }}
```

## Maintenance

### Update ECR Registry

Edit `build_and_push.sh`:
```bash
ECR_REGISTRY="your-account.dkr.ecr.your-region.amazonaws.com"
IMAGE_NAME="your-image-name"
REGION="your-region"
```

### Update CUDA Architectures

Edit `build_and_push.sh`:
```bash
ARCHS="8.0;8.6;8.9;9.0"  # Add/remove compute capabilities
```

### Modify Pipeline Stages

Edit `entrypoint.sh` to:
- Add new stages
- Modify stage logic
- Change timing format
- Add validation steps

## Troubleshooting

### build_and_push.sh fails

```bash
# Check AWS credentials
aws sts get-caller-identity

# Check Docker
docker version
docker buildx version

# Check disk space
df -h
```

### prebuild_cleanup.sh warnings

```bash
# Force cleanup without prompts
docker system prune -a -f --volumes

# Check what's using space
docker system df -v
```

### entrypoint.sh errors

```bash
# Run with debug
docker run --rm -it your-image:latest bash
# Then manually run: bash -x /workspace/app/scripts/entrypoint.sh
```

## Related Documentation

- [Docker Build Guide](../docs/docker/BUILD.md)
- [AWS Batch Setup](../docs/aws/SETUP.md)
- [Main README](../README.md)

#!/bin/bash
# Build and push Docker image to ECR

set -e

# Configuration
ECR_REGISTRY="236357498583.dkr.ecr.ap-northeast-2.amazonaws.com"
IMAGE_NAME="3dgs/builder"
TAG="${1:-latest}"
REGION="ap-northeast-2"
PLATFORM="linux/amd64"
ARCHS="8.0;8.6;8.9"  # CUDA architectures

echo "=========================================="
echo "Building and Pushing 3DGS Docker Image"
echo "=========================================="
echo "Registry: ${ECR_REGISTRY}"
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "Platform: ${PLATFORM}"
echo "Region: ${REGION}"
echo "CUDA Archs: ${ARCHS}"
echo ""

# Step 1: Login to ECR
echo "[1/3] Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin ${ECR_REGISTRY}

echo "✓ Logged in to ECR"
echo ""

# Step 2: Create repository if it doesn't exist
echo "[2/3] Checking ECR repository..."
aws ecr describe-repositories \
  --repository-names ${IMAGE_NAME} \
  --region ${REGION} 2>/dev/null || \
aws ecr create-repository \
  --repository-name ${IMAGE_NAME} \
  --region ${REGION} \
  --image-scanning-configuration scanOnPush=true

echo "✓ Repository ready"
echo ""

# Step 3: Build and push
echo "[3/3] Building and pushing image..."
echo "This may take 20-30 minutes..."
echo ""

docker buildx build \
  --platform ${PLATFORM} \
  --build-arg ARCHS="${ARCHS}" \
  -t ${ECR_REGISTRY}/${IMAGE_NAME}:${TAG} \
  --push \
  .

echo ""
echo "=========================================="
echo "✓ Build and push completed!"
echo "=========================================="
echo "Image: ${ECR_REGISTRY}/${IMAGE_NAME}:${TAG}"
echo ""
echo "Next steps:"
echo "1. Register job definition:"
echo "   aws batch register-job-definition \\"
echo "     --cli-input-json file://aws-batch-job-definition.json \\"
echo "     --region ${REGION}"
echo ""
echo "2. Test image locally:"
echo "   docker run --rm --gpus all \\"
echo "     -e PROJECT=test \\"
echo "     ${ECR_REGISTRY}/${IMAGE_NAME}:${TAG}"
echo ""
echo "3. Debug S3 access:"
echo "   docker run --rm -it \\"
echo "     -e AWS_ACCESS_KEY_ID=\$AWS_ACCESS_KEY_ID \\"
echo "     -e AWS_SECRET_ACCESS_KEY=\$AWS_SECRET_ACCESS_KEY \\"
echo "     ${ECR_REGISTRY}/${IMAGE_NAME}:${TAG} \\"
echo "     aws s3 ls s3://your-bucket/"

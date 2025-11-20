#!/bin/bash
# Pre-build cleanup and disk space check

set -e

echo "=========================================="
echo "Docker Pre-Build Cleanup & Check"
echo "=========================================="
echo ""

# Check current disk usage
echo "[1/4] Checking Docker disk usage..."
docker system df
echo ""

# Check host disk space
echo "[2/4] Checking host disk space..."
df -h | grep -E "Filesystem|/System/Volumes/Data|/$" || df -h
echo ""

# Calculate required space
REQUIRED_GB=60
AVAILABLE=$(df -k . | tail -1 | awk '{print $4}')
AVAILABLE_GB=$((AVAILABLE / 1024 / 1024))

echo "Available disk space: ${AVAILABLE_GB} GB"
echo "Required disk space: ${REQUIRED_GB} GB"
echo ""

if [ $AVAILABLE_GB -lt $REQUIRED_GB ]; then
    echo "⚠️  WARNING: Insufficient disk space!"
    echo "   You have ${AVAILABLE_GB} GB but need at least ${REQUIRED_GB} GB"
    echo ""
    echo "Would you like to clean up Docker resources? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo ""
        echo "[3/4] Cleaning up Docker resources..."
        
        # Stop all containers
        echo "  → Stopping containers..."
        docker ps -q | xargs -r docker stop || true
        
        # Remove stopped containers
        echo "  → Removing stopped containers..."
        docker container prune -f
        
        # Remove unused images
        echo "  → Removing unused images..."
        docker image prune -a -f
        
        # Remove build cache
        echo "  → Removing build cache..."
        docker builder prune -a -f
        
        # Remove unused volumes
        echo "  → Removing unused volumes..."
        docker volume prune -f
        
        echo "✓ Cleanup complete!"
        echo ""
        
        # Check again
        echo "[4/4] Disk space after cleanup..."
        docker system df
        echo ""
        df -h | grep -E "Filesystem|/System/Volumes/Data|/$" || df -h
        echo ""
        
        AVAILABLE=$(df -k . | tail -1 | awk '{print $4}')
        AVAILABLE_GB=$((AVAILABLE / 1024 / 1024))
        
        if [ $AVAILABLE_GB -lt $REQUIRED_GB ]; then
            echo "⚠️  Still insufficient space (${AVAILABLE_GB} GB available)"
            echo ""
            echo "Recommendations:"
            echo "1. Increase Docker Desktop disk allocation (macOS/Windows)"
            echo "2. Free up disk space on your system"
            echo "3. Build on an EC2 instance with more storage"
            echo ""
            echo "See docs/docker/TROUBLESHOOTING.md for more options."
            exit 1
        fi
    else
        echo "Skipping cleanup. Build may fail due to insufficient space."
        echo "See docs/docker/TROUBLESHOOTING.md for solutions."
        exit 1
    fi
else
    echo "✓ Sufficient disk space available"
    echo ""
    echo "Would you like to clean up Docker resources anyway? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo ""
        echo "[3/4] Cleaning up Docker resources..."
        docker system prune -a -f
        docker builder prune -a -f
        echo "✓ Cleanup complete!"
        echo ""
    else
        echo "Skipping cleanup."
    fi
fi

echo "=========================================="
echo "✓ Pre-build check complete!"
echo "=========================================="
echo ""
echo "Ready to build. Run:"
echo "  ./scripts/build_and_push.sh"
echo ""
echo "Or manually:"
echo "  docker buildx build \\"
echo "    --platform linux/amd64 \\"
echo "    -t 236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:latest \\"
echo "    --push ."

# AWS Batch Integration for 3DGS Pipeline

Complete guide for running the 3D Gaussian Splatting pipeline on AWS Batch with GPU instances.

## 📚 Documentation Index

1. **[Setup Guide](./SETUP.md)** - Complete setup instructions for AWS Batch
2. **[Quick Reference](./QUICK_REFERENCE.md)** - Common commands and examples
3. **[Cost Guide](./COSTS.md)** - Pricing information and optimization tips
4. **[Troubleshooting](./TROUBLESHOOTING.md)** - Common issues and solutions

## 🚀 Quick Start

### Prerequisites
- AWS Account with appropriate permissions
- Docker image built and pushed to ECR
- S3 buckets for input/output data

### Basic Workflow

```bash
# 1. Register job definition (one-time)
aws batch register-job-definition \
  --cli-input-json file://aws-batch-job-definition.json \
  --region ap-northeast-2

# 2. Create log group (one-time)
aws logs create-log-group \
  --log-group-name /aws/batch/3dgs-pipeline \
  --region ap-northeast-2

# 3. Submit a job (Python)
python -c "
from submit_batch_job import BatchJobSubmitter
submitter = BatchJobSubmitter(region='ap-northeast-2')
submitter.submit_job(
    project_name='my-scene',
    s3_input_bucket='s3://my-bucket/inputs/my-scene/images',
    s3_output_bucket='s3://my-bucket/outputs'
)
"
```

## 📦 Your AWS Resources

### Compute Environment
- **Name**: GSJobs
- **Type**: EC2 Managed
- **Instance Types**: g4dn, g5, p3, p4d
- **Max vCPUs**: 256
- **Region**: ap-northeast-2

### Job Queue
- **Name**: GSJobsQueue
- **State**: ENABLED
- **Priority**: 1

### Container Image
- **Repository**: `236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:latest`
- **Platform**: linux/amd64

## 🔧 Job Configuration

Default settings:
- **vCPUs**: 8
- **Memory**: 32GB
- **GPUs**: 1
- **Timeout**: 12 hours
- **Retry Attempts**: 2

Override these when submitting jobs based on your scene complexity.

## 📊 Monitoring

- **Console**: https://console.aws.amazon.com/batch/home?region=ap-northeast-2#jobs
- **Logs**: CloudWatch Logs → `/aws/batch/3dgs-pipeline`

## 💰 Estimated Costs

| Instance Type | GPU | Hourly Cost | 12-Hour Job |
|--------------|-----|-------------|-------------|
| g4dn.2xlarge | T4 | ~$0.75 | ~$9.00 |
| g4dn.4xlarge | T4 | ~$1.50 | ~$18.00 |
| g5.2xlarge | A10G | ~$1.21 | ~$14.52 |
| p3.2xlarge | V100 | ~$3.67 | ~$44.04 |

*Prices for ap-northeast-2 (Seoul) region. Use Spot instances for 50-70% savings.*

## 🔗 Related Documentation

- [Docker Build Guide](../docker/BUILD.md)
- [S3 Usage](./S3_USAGE.md)
- [Main Pipeline README](../../README.md)

## ✅ Features

- ✅ S3 integration for input/output
- ✅ GPU-accelerated processing
- ✅ Automated job submission via boto3
- ✅ CloudWatch logging
- ✅ Automatic retry on failure
- ✅ Resource optimization

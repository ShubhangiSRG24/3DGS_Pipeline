# AWS Batch Setup Guide

Complete setup instructions for running the 3DGS Pipeline on AWS Batch.

## Prerequisites

1. **AWS Batch Environment** ✓
   - Compute Environment: `GSJobs`
   - Job Queue: `GSJobsQueue`
   - Region: `ap-northeast-2`

2. **ECR Image** ✓
   - Image: `236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:latest`
   - See [Docker Build Guide](../docker/BUILD.md) for building

3. **IAM Roles** (Create if not exists)
   - `BatchJobRole` - Container S3 access
   - `ecsTaskExecutionRole` - ECS image pulling and logging

## Step 1: Create IAM Roles

### A. BatchJobRole (Container S3 Access)

```bash
# Create the role
aws iam create-role \
  --role-name BatchJobRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' \
  --region ap-northeast-2

# Attach S3 access policy
aws iam put-role-policy \
  --role-name BatchJobRole \
  --policy-name S3AccessPolicy \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket/*",
        "arn:aws:s3:::your-bucket"
      ]
    }]
  }' \
  --region ap-northeast-2
```

**Important**: Replace `your-bucket` with your actual bucket name(s).

### B. ecsTaskExecutionRole (ECS Operations)

```bash
# Create the role
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' \
  --region ap-northeast-2

# Attach AWS managed policy
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
  --region ap-northeast-2
```

## Step 2: Create CloudWatch Log Group

```bash
aws logs create-log-group \
  --log-group-name /aws/batch/3dgs-pipeline \
  --region ap-northeast-2
```

## Step 3: Register Job Definition

### Option A: Using AWS CLI

```bash
aws batch register-job-definition \
  --cli-input-json file://aws-batch-job-definition.json \
  --region ap-northeast-2
```

### Option B: Using Python (boto3)

```python
import boto3
import json

batch_client = boto3.client('batch', region_name='ap-northeast-2')

with open('aws-batch-job-definition.json', 'r') as f:
    job_def = json.load(f)

response = batch_client.register_job_definition(**job_def)
print(f"Registered: {response['jobDefinitionName']}:{response['revision']}")
```

## Step 4: Test Job Submission

### Python Script

```python
from submit_batch_job import BatchJobSubmitter

submitter = BatchJobSubmitter(region="ap-northeast-2")

response = submitter.submit_job(
    project_name="test-scene",
    s3_input_bucket="s3://my-bucket/inputs/test-scene/images",
    s3_output_bucket="s3://my-bucket/outputs",
    skip_extract=True
)

print(f"Job ID: {response['jobId']}")

# Wait for completion (optional)
status = submitter.wait_for_job(response['jobId'])
print(f"Final status: {status}")
```

### Command Line

```bash
python submit_batch_job.py test-scene \
  s3://my-bucket/inputs/test-scene/images \
  s3://my-bucket/outputs \
  --wait
```

## Step 5: Verify Setup

### Check Job Status

```bash
# List running jobs
aws batch list-jobs \
  --job-queue GSJobsQueue \
  --job-status RUNNING \
  --region ap-northeast-2

# Describe specific job
aws batch describe-jobs \
  --jobs <job-id> \
  --region ap-northeast-2
```

### View Logs

```bash
# Get log stream
aws batch describe-jobs \
  --jobs <job-id> \
  --region ap-northeast-2 \
  --query 'jobs[0].container.logStreamName' \
  --output text

# View logs
aws logs tail /aws/batch/3dgs-pipeline \
  --follow \
  --region ap-northeast-2
```

## Job Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_name` | str | required | Project/scene name |
| `s3_input_bucket` | str | required | S3 path for input images |
| `s3_output_bucket` | str | required | S3 path for outputs |
| `vcpus` | int | 8 | Number of vCPUs (1-32) |
| `memory` | int | 32768 | Memory in MiB |
| `gpu_count` | int | 1 | Number of GPUs (1-8) |
| `timeout_hours` | int | 12 | Job timeout |
| `skip_extract` | bool | True | Skip video extraction |
| `skip_sfm` | bool | False | Skip SfM |
| `skip_depth` | bool | False | Skip depth |
| `skip_train` | bool | False | Skip training |

## Instance Type Selection

Your compute environment supports:

| Family | GPU | Best For | Cost Level |
|--------|-----|----------|-----------|
| g4dn | NVIDIA T4 | Budget-friendly | $ |
| g5 | NVIDIA A10G | Balanced | $$ |
| p3 | NVIDIA V100 | High performance | $$$ |
| p4d | NVIDIA A100 | Maximum performance | $$$$ |

AWS Batch automatically selects the best available instance based on your resource requirements.

## Best Practices

1. **Start Small**: Test with a small scene first
2. **Monitor Costs**: Set up AWS Budgets and alerts
3. **Use Spot Instances**: Configure for 50-70% savings
4. **Tag Resources**: Add tags for cost tracking
5. **Set Timeouts**: Prevent runaway jobs
6. **Clean Up**: Delete old S3 data regularly
7. **Right-Size**: Don't over-allocate resources

## Next Steps

1. ✅ Complete this setup
2. 📖 Read [Quick Reference](./QUICK_REFERENCE.md) for examples
3. 💰 Review [Cost Guide](./COSTS.md) for optimization
4. 🚀 Start processing scenes!

## Support Resources

- [AWS Batch Console](https://console.aws.amazon.com/batch)
- [CloudWatch Logs](https://console.aws.amazon.com/cloudwatch/)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)

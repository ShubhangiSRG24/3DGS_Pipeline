# AWS Batch Troubleshooting Guide

Common issues and solutions when running the 3DGS Pipeline on AWS Batch.

## Job Status Issues

### Job Stuck in RUNNABLE

**Symptoms**: Job stays in RUNNABLE state and never starts.

**Causes**:
1. Insufficient compute capacity
2. Instance type unavailable
3. Service limits reached
4. Subnet/networking issues

**Solutions**:

```bash
# Check compute environment status
aws batch describe-compute-environments \
  --compute-environments GSJobs \
  --region ap-northeast-2

# Check if instances are launching
aws ec2 describe-instances \
  --filters "Name=tag:aws:batch:job-queue,Values=GSJobsQueue" \
  --region ap-northeast-2

# Check service limits
aws service-quotas list-service-quotas \
  --service-code batch \
  --region ap-northeast-2
```

**Fix**:
1. Wait - capacity may become available
2. Try different instance type
3. Request limit increase
4. Check VPC subnet has available IPs

### Job Stuck in STARTING

**Symptoms**: Job moves to STARTING but doesn't progress.

**Causes**:
1. Container image pull failure
2. IAM role issues
3. Network connectivity problems

**Solutions**:

```bash
# Check job details
aws batch describe-jobs \
  --jobs job-id-here \
  --region ap-northeast-2

# Look for statusReason
aws batch describe-jobs \
  --jobs job-id-here \
  --query 'jobs[0].statusReason' \
  --region ap-northeast-2
```

**Fix**:
- Verify ECR image exists and is accessible
- Check ecsTaskExecutionRole permissions
- Ensure compute environment has internet access

### Job Fails Immediately

**Symptoms**: Job starts but fails within seconds.

**Causes**:
1. Container entrypoint error
2. Missing environment variables
3. S3 permissions issue

**Solutions**:

```bash
# Get logs
aws batch describe-jobs \
  --jobs job-id-here \
  --query 'jobs[0].container.logStreamName' \
  --region ap-northeast-2 \
  --output text

# View logs
aws logs tail /aws/batch/3dgs-pipeline \
  --log-stream-names <stream-name> \
  --region ap-northeast-2
```

**Common Fixes**:
```python
# Ensure S3 paths are correct
submitter.submit_job(
    s3_input_bucket="s3://bucket/path",  # Must start with s3://
    s3_output_bucket="s3://bucket/output"
)

# Check environment variables
environment={
    'PROJECT': 'scene-name',
    'AWS_REGION': 'ap-northeast-2'
}
```

## Container Issues

### CannotPullContainerError

**Error**: `CannotPullContainerError: Error response from daemon`

**Causes**:
1. Image doesn't exist in ECR
2. Wrong image URI
3. ecsTaskExecutionRole missing permissions
4. Network connectivity

**Solutions**:

```bash
# Verify image exists
aws ecr describe-images \
  --repository-name 3dgs/builder \
  --region ap-northeast-2

# Check ECR login
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  236357498583.dkr.ecr.ap-northeast-2.amazonaws.com

# Test pull manually
docker pull 236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:latest
```

**Fix ecsTaskExecutionRole**:
```bash
# Attach required policy
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
  --region ap-northeast-2
```

### Essential Container Exited

**Error**: `Essential container in task exited`

**Causes**:
1. Application error
2. Out of memory
3. Missing dependencies

**Solutions**:

```bash
# Check exit code in logs
aws batch describe-jobs \
  --jobs job-id-here \
  --query 'jobs[0].container.exitCode' \
  --region ap-northeast-2

# Common exit codes:
# 0 = Success
# 1 = General error
# 137 = Out of memory (OOM killed)
# 139 = Segmentation fault
```

**Fix OOM (exit code 137)**:
```python
# Increase memory allocation
submitter.submit_job(
    memory=65536,  # Increase to 64GB
    vcpus=16
)
```

## S3 Access Issues

### Access Denied (S3)

**Error**: `An error occurred (AccessDenied) when calling the GetObject operation`

**Causes**:
1. BatchJobRole missing S3 permissions
2. Bucket policy blocking access
3. Wrong bucket name/region

**Solutions**:

```bash
# Test S3 access from job
aws s3 ls s3://my-bucket/inputs/ --region ap-northeast-2

# Check BatchJobRole policy
aws iam get-role-policy \
  --role-name BatchJobRole \
  --policy-name S3AccessPolicy
```

**Fix BatchJobRole permissions**:
```bash
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
        "arn:aws:s3:::my-bucket/*",
        "arn:aws:s3:::my-bucket"
      ]
    }]
  }'
```

### No Such Bucket

**Error**: `The specified bucket does not exist`

**Solutions**:

```bash
# List all buckets
aws s3 ls

# Check bucket region
aws s3api get-bucket-location --bucket my-bucket

# Create bucket if needed
aws s3 mb s3://my-bucket --region ap-northeast-2
```

### Slow S3 Transfer

**Issue**: S3 sync taking too long

**Solutions**:

```bash
# Use parallel uploads
aws s3 sync /local/path s3://bucket/path \
  --no-progress \
  --region ap-northeast-2

# Check transfer acceleration (if enabled)
aws s3api get-bucket-accelerate-configuration \
  --bucket my-bucket
```

**Enable transfer acceleration**:
```bash
aws s3api put-bucket-accelerate-configuration \
  --bucket my-bucket \
  --accelerate-configuration Status=Enabled
```

## Resource Issues

### Out of Memory

**Symptoms**: Job fails with exit code 137 or OOM messages in logs

**Solutions**:

```python
# Increase memory
submitter.submit_job(
    memory=65536,  # 64GB instead of 32GB
    vcpus=16
)

# Or process fewer images
environment={'NUM_IMAGES': '100'}  # Instead of 200
```

### CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:

```python
# Use larger GPU instance
submitter.submit_job(
    vcpus=16,
    memory=65536,
    gpu_count=1,  # Or 2 GPUs
    # Will select g5.4xlarge or similar
)

# Or reduce batch size in training
environment={
    'NUM_IMAGES': '100',
    'BATCH_SIZE': '4'  # If supported by your code
}
```

### Timeout Exceeded

**Error**: Job terminated due to timeout

**Solutions**:

```python
# Increase timeout
submitter.submit_job(
    timeout_hours=24  # Instead of 12
)

# Or optimize processing
environment={
    'SKIP_DEPTH': '1',  # Skip expensive stages if testing
}
```

## Networking Issues

### No Internet Access

**Symptoms**: Cannot download models, cannot reach S3

**Causes**:
1. Compute environment in private subnet without NAT
2. Security group blocking egress

**Solutions**:

```bash
# Check compute environment subnets
aws batch describe-compute-environments \
  --compute-environments GSJobs \
  --query 'computeEnvironments[0].computeResources.subnets' \
  --region ap-northeast-2

# Check NAT gateway exists
aws ec2 describe-nat-gateways --region ap-northeast-2

# Check security group allows outbound
aws ec2 describe-security-groups \
  --group-ids sg-xxxxx \
  --region ap-northeast-2
```

**Fix**: Ensure compute environment uses:
- Public subnet with Internet Gateway, OR
- Private subnet with NAT Gateway

## Logging Issues

### No Logs Appearing

**Causes**:
1. Log group doesn't exist
2. ecsTaskExecutionRole missing CloudWatch permissions
3. Container crashes before logging starts

**Solutions**:

```bash
# Create log group
aws logs create-log-group \
  --log-group-name /aws/batch/3dgs-pipeline \
  --region ap-northeast-2

# Check ecsTaskExecutionRole has CloudWatch permissions
aws iam get-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-name CloudWatchLogsPolicy
```

### Log Stream Not Found

**Error**: Cannot find log stream for job

**Solution**:
```bash
# List all log streams
aws logs describe-log-streams \
  --log-group-name /aws/batch/3dgs-pipeline \
  --region ap-northeast-2

# Get job's log stream from job details
aws batch describe-jobs \
  --jobs job-id-here \
  --query 'jobs[0].container.logStreamName' \
  --region ap-northeast-2
```

## Performance Issues

### Job Taking Too Long

**Symptoms**: Job runs but takes much longer than expected

**Diagnostics**:

```python
# Check CloudWatch logs for bottlenecks
import boto3

logs = boto3.client('logs', region_name='ap-northeast-2')
response = logs.get_log_events(
    logGroupName='/aws/batch/3dgs-pipeline',
    logStreamName='stream-name'
)

# Look for timing information
for event in response['events']:
    if 'TIMING' in event['message']:
        print(event['message'])
```

**Solutions**:
1. Use faster GPU instance (g5 instead of g4dn)
2. Increase vCPUs for parallel processing
3. Check if I/O bound (S3 transfer slow)
4. Profile code to find bottleneck

### Insufficient GPU Performance

**Issue**: Training slower than expected

**Solutions**:

```python
# Upgrade to better GPU
submitter.submit_job(
    # Use A10G instead of T4
    vcpus=16,
    memory=65536,
    gpu_count=1,  # Will select g5.4xlarge (A10G)
)

# Or V100
submitter.submit_job(
    vcpus=8,
    memory=61440,
    gpu_count=1,  # Will select p3.2xlarge (V100)
)
```

## Common Error Messages

### "Service is unhealthy"

**Fix**: Check compute environment status, may need to recreate

### "Host EC2 instance terminated"

**Fix**: Likely Spot interruption, job will retry automatically

### "ResourceInitializationError"

**Fix**: Usually IAM/permissions issue, check roles

### "InvalidParameterException"

**Fix**: Check job definition parameters, may need update

## Debugging Workflow

1. **Check job status**:
   ```bash
   aws batch describe-jobs --jobs job-id --region ap-northeast-2
   ```

2. **Get status reason**:
   ```bash
   aws batch describe-jobs --jobs job-id \
     --query 'jobs[0].statusReason' \
     --region ap-northeast-2
   ```

3. **View logs**:
   ```bash
   aws logs tail /aws/batch/3dgs-pipeline --follow --region ap-northeast-2
   ```

4. **Check container exit code**:
   ```bash
   aws batch describe-jobs --jobs job-id \
     --query 'jobs[0].container.exitCode' \
     --region ap-northeast-2
   ```

5. **Test locally**:
   ```bash
   docker run --rm -it --gpus all \
     -e PROJECT=test \
     -e S3_INPUT_BUCKET=s3://bucket/input \
     -e S3_OUTPUT_BUCKET=s3://bucket/output \
     236357498583.dkr.ecr.ap-northeast-2.amazonaws.com/3dgs/builder:latest
   ```

## Getting Help

1. Check [AWS Batch Console](https://console.aws.amazon.com/batch) for visual status
2. Review [CloudWatch Logs](https://console.aws.amazon.com/cloudwatch/)
3. Check AWS Service Health Dashboard
4. Review [Setup Guide](./SETUP.md) and [Quick Reference](./QUICK_REFERENCE.md)

## Support Commands

```bash
# Export job details for support
aws batch describe-jobs --jobs job-id --region ap-northeast-2 > job-details.json

# Export logs
aws logs get-log-events \
  --log-group-name /aws/batch/3dgs-pipeline \
  --log-stream-name stream-name \
  --region ap-northeast-2 > job-logs.txt

# Check account limits
aws service-quotas list-service-quotas \
  --service-code batch \
  --region ap-northeast-2
```

# Quick Reference: AWS Batch Commands

Common commands and code examples for AWS Batch job submission and monitoring.

## Job Submission

### Simple Job
```python
from submit_batch_job import BatchJobSubmitter

submitter = BatchJobSubmitter(region="ap-northeast-2")

response = submitter.submit_job(
    project_name="my-scene",
    s3_input_bucket="s3://my-bucket/inputs/my-scene/images",
    s3_output_bucket="s3://my-bucket/outputs"
)

print(f"Job ID: {response['jobId']}")
```

### Command Line
```bash
python submit_batch_job.py my-scene \
  s3://my-bucket/inputs/my-scene/images \
  s3://my-bucket/outputs \
  --wait
```

### High-Performance Job (Multiple GPUs)
```python
response = submitter.submit_job(
    project_name="large-scene",
    s3_input_bucket="s3://my-bucket/inputs/large-scene/images",
    s3_output_bucket="s3://my-bucket/outputs",
    vcpus=16,
    memory=65536,
    gpu_count=2,
    timeout_hours=24
)
```

### Batch Process Multiple Scenes
```python
scenes = ["scene01", "scene02", "scene03"]
job_ids = []

for scene in scenes:
    response = submitter.submit_job(
        project_name=scene,
        s3_input_bucket=f"s3://my-bucket/inputs/{scene}/images",
        s3_output_bucket="s3://my-bucket/outputs"
    )
    job_ids.append(response['jobId'])
    print(f"Submitted {scene}: {response['jobId']}")

print(f"Total: {len(job_ids)} jobs")
```

## Job Monitoring

### List Jobs by Status
```python
# Running jobs
running = submitter.list_jobs(status="RUNNING")
print(f"Running: {len(running)}")
for job in running:
    print(f"  {job['jobName']}: {job['jobId']}")

# Pending jobs
pending = submitter.list_jobs(status="PENDING")
print(f"Pending: {len(pending)}")

# Failed jobs
failed = submitter.list_jobs(status="FAILED")
print(f"Failed: {len(failed)}")
```

### Check Job Status
```python
job = submitter.get_job_status("job-id-here")
print(f"Status: {job['status']}")
print(f"Started: {job.get('startedAt', 'N/A')}")
print(f"Stopped: {job.get('stoppedAt', 'N/A')}")
```

### Wait for Job Completion
```python
status = submitter.wait_for_job("job-id-here")
print(f"Final status: {status}")
```

### Cancel Job
```python
submitter.cancel_job(
    job_id="job-id-here",
    reason="User cancelled"
)
```

## AWS CLI Commands

### List Jobs
```bash
# Running jobs
aws batch list-jobs \
  --job-queue GSJobsQueue \
  --job-status RUNNING \
  --region ap-northeast-2

# All jobs
aws batch list-jobs \
  --job-queue GSJobsQueue \
  --region ap-northeast-2
```

### Describe Job
```bash
aws batch describe-jobs \
  --jobs job-id-here \
  --region ap-northeast-2
```

### Cancel Job
```bash
aws batch cancel-job \
  --job-id job-id-here \
  --reason "User cancelled" \
  --region ap-northeast-2
```

### View Logs
```bash
# Get log stream name
aws batch describe-jobs \
  --jobs job-id-here \
  --region ap-northeast-2 \
  --query 'jobs[0].container.logStreamName' \
  --output text

# Tail logs (follow mode)
aws logs tail /aws/batch/3dgs-pipeline \
  --follow \
  --region ap-northeast-2
```

## S3 Operations

### Upload Images
```bash
# Sync local directory to S3
aws s3 sync /local/images/ s3://my-bucket/inputs/my-scene/images/

# Upload single file
aws s3 cp /local/image.jpg s3://my-bucket/inputs/my-scene/images/
```

### Download Results
```bash
# Download all outputs for a scene
aws s3 sync s3://my-bucket/outputs/my-scene/ /local/output/

# Download specific file
aws s3 cp s3://my-bucket/outputs/my-scene/output.ply /local/
```

### List Files
```bash
# List all outputs
aws s3 ls s3://my-bucket/outputs/my-scene/ --recursive

# List with sizes
aws s3 ls s3://my-bucket/outputs/my-scene/ --recursive --human-readable
```

## Job Configuration Examples

### Small Scene (Fast Processing)
```python
submitter.submit_job(
    project_name="small-scene",
    s3_input_bucket="s3://bucket/input",
    s3_output_bucket="s3://bucket/output",
    vcpus=4,
    memory=16384,
    gpu_count=1,
    timeout_hours=6
)
```

### Standard Scene (Default Settings)
```python
submitter.submit_job(
    project_name="standard-scene",
    s3_input_bucket="s3://bucket/input",
    s3_output_bucket="s3://bucket/output",
    vcpus=8,
    memory=32768,
    gpu_count=1,
    timeout_hours=12
)
```

### Large Scene (High Resources)
```python
submitter.submit_job(
    project_name="large-scene",
    s3_input_bucket="s3://bucket/input",
    s3_output_bucket="s3://bucket/output",
    vcpus=16,
    memory=65536,
    gpu_count=2,
    timeout_hours=24
)
```

### Skip Specific Stages
```python
# Skip video extraction (images already in S3)
submitter.submit_job(
    project_name="scene",
    s3_input_bucket="s3://bucket/input",
    s3_output_bucket="s3://bucket/output",
    skip_extract=True
)

# Skip training (just run preprocessing)
submitter.submit_job(
    project_name="scene",
    s3_input_bucket="s3://bucket/input",
    s3_output_bucket="s3://bucket/output",
    skip_train=True
)
```

## Troubleshooting Quick Fixes

### Job Stuck in RUNNABLE
```bash
# Check compute environment
aws batch describe-compute-environments \
  --compute-environments GSJobs \
  --region ap-northeast-2
```

### View Job Logs
```python
import boto3

logs_client = boto3.client('logs', region_name='ap-northeast-2')
batch_client = boto3.client('batch', region_name='ap-northeast-2')

# Get job details
job = batch_client.describe_jobs(jobs=['job-id'])['jobs'][0]
log_stream = job['container']['logStreamName']

# Get logs
response = logs_client.get_log_events(
    logGroupName='/aws/batch/3dgs-pipeline',
    logStreamName=log_stream
)

for event in response['events']:
    print(event['message'])
```

### Check S3 Access
```bash
# Test S3 access
aws s3 ls s3://my-bucket/ --region ap-northeast-2

# Check bucket permissions
aws s3api get-bucket-policy --bucket my-bucket
```

## Useful One-Liners

```bash
# Count running jobs
aws batch list-jobs --job-queue GSJobsQueue --job-status RUNNING --region ap-northeast-2 | jq '.jobSummaryList | length'

# Get all job IDs
aws batch list-jobs --job-queue GSJobsQueue --region ap-northeast-2 | jq -r '.jobSummaryList[].jobId'

# Cancel all running jobs (use with caution!)
aws batch list-jobs --job-queue GSJobsQueue --job-status RUNNING --region ap-northeast-2 | \
  jq -r '.jobSummaryList[].jobId' | \
  xargs -I {} aws batch cancel-job --job-id {} --reason "Bulk cancel" --region ap-northeast-2
```

## Related Documentation

- [Setup Guide](./SETUP.md) - Initial configuration
- [Cost Guide](./COSTS.md) - Pricing and optimization
- [Troubleshooting](./TROUBLESHOOTING.md) - Common issues

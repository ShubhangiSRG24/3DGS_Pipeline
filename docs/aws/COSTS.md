# AWS Batch Cost Guide

Pricing information, cost estimation, and optimization strategies for running 3DGS Pipeline on AWS Batch.

## Instance Pricing (ap-northeast-2 Seoul)

### On-Demand Pricing

| Instance Type | GPU | vCPU | RAM | Hourly Cost | 12-Hour Job |
|--------------|-----|------|-----|-------------|-------------|
| g4dn.xlarge | T4 | 4 | 16GB | $0.63 | $7.56 |
| g4dn.2xlarge | T4 | 8 | 32GB | $0.75 | $9.00 |
| g4dn.4xlarge | T4 | 16 | 64GB | $1.50 | $18.00 |
| g4dn.8xlarge | T4 | 32 | 128GB | $2.72 | $32.64 |
| g5.2xlarge | A10G | 8 | 32GB | $1.21 | $14.52 |
| g5.4xlarge | A10G | 16 | 64GB | $1.62 | $19.44 |
| p3.2xlarge | V100 | 8 | 61GB | $3.67 | $44.04 |
| p3.8xlarge | V100×4 | 32 | 244GB | $14.69 | $176.28 |

### Spot Instance Savings

Spot instances offer 50-70% discount over On-Demand:

| Instance Type | On-Demand | Spot (avg) | Savings |
|--------------|-----------|------------|---------|
| g4dn.2xlarge | $0.75/hr | ~$0.30/hr | 60% |
| g4dn.4xlarge | $1.50/hr | ~$0.60/hr | 60% |
| g5.2xlarge | $1.21/hr | ~$0.48/hr | 60% |
| p3.2xlarge | $3.67/hr | ~$1.47/hr | 60% |

*Spot prices vary based on demand. Check current prices in AWS Console.*

## Additional AWS Costs

### S3 Storage
- **Standard Storage**: $0.025/GB/month
- **PUT/POST requests**: $0.005 per 1,000 requests
- **GET requests**: $0.0004 per 1,000 requests
- **Data transfer out**: $0.126/GB (first 10TB)

### CloudWatch Logs
- **Ingestion**: $0.76 per GB
- **Storage**: $0.033 per GB/month
- **Typical log size**: ~100MB per job

### Data Transfer
- **S3 to EC2 (same region)**: Free
- **S3 to Internet**: $0.126/GB (first 10TB)
- **Between regions**: Varies by region pair

## Cost Examples

### Small Scene (100 images, 6 hours)
```
Instance: g4dn.2xlarge
Duration: 6 hours
Compute: $0.75/hr × 6 = $4.50

S3 Input: 2GB × $0.025/30 = $0.002
S3 Output: 1GB × $0.025/30 = $0.001
S3 Transfer: Minimal (same region)
CloudWatch Logs: ~50MB × $0.76/GB = $0.04

Total: ~$4.55
```

### Standard Scene (200 images, 12 hours)
```
Instance: g4dn.4xlarge
Duration: 12 hours
Compute: $1.50/hr × 12 = $18.00

S3 Input: 4GB × $0.025/30 = $0.003
S3 Output: 2GB × $0.025/30 = $0.002
CloudWatch Logs: ~100MB × $0.76/GB = $0.08

Total: ~$18.09
```

### Large Scene (500 images, 24 hours)
```
Instance: g5.4xlarge
Duration: 24 hours
Compute: $1.62/hr × 24 = $38.88

S3 Input: 10GB × $0.025/30 = $0.008
S3 Output: 5GB × $0.025/30 = $0.004
CloudWatch Logs: ~200MB × $0.76/GB = $0.15

Total: ~$39.04
```

### Spot Instance Savings (Standard Scene)
```
On-Demand: $18.09
Spot Instance: $0.60/hr × 12 = $7.20 + overhead = $7.29
Savings: $10.80 (60% reduction)
```

## Cost Optimization Strategies

### 1. Use Spot Instances

**Savings: 50-70%**

Configure compute environment for SPOT allocation:
```bash
aws batch update-compute-environment \
  --compute-environment GSJobs \
  --compute-resources allocationStrategy=SPOT_CAPACITY_OPTIMIZED
```

**Pros:**
- Massive cost savings
- Good for non-urgent workloads

**Cons:**
- May be interrupted
- Slightly less predictable timing

### 2. Right-Size Resources

**Savings: 20-50%**

Don't over-allocate:
- Small scenes: 4 vCPU, 16GB RAM
- Medium scenes: 8 vCPU, 32GB RAM
- Large scenes: 16 vCPU, 64GB RAM

```python
# Don't do this for small scenes
submitter.submit_job(
    vcpus=32,  # Too many!
    memory=128000,  # Way too much!
    gpu_count=4  # Overkill!
)

# Do this instead
submitter.submit_job(
    vcpus=8,
    memory=32768,
    gpu_count=1
)
```

### 3. Set Appropriate Timeouts

**Savings: Prevents runaway costs**

```python
# Set realistic timeout
submitter.submit_job(
    timeout_hours=12  # Kill job if exceeds 12 hours
)
```

### 4. Clean Up S3 Data

**Savings: Reduce storage costs**

```bash
# Delete data older than 30 days
aws s3 rm s3://my-bucket/outputs/ \
  --recursive \
  --exclude "*" \
  --include "*" \
  --older-than 30d
```

Set up S3 Lifecycle policies:
```json
{
  "Rules": [{
    "Id": "DeleteOldOutputs",
    "Status": "Enabled",
    "Prefix": "outputs/",
    "Expiration": {
      "Days": 30
    }
  }]
}
```

### 5. Batch Multiple Scenes

**Savings: Better resource utilization**

Submit multiple jobs to keep GPU busy:
```python
for scene in scenes:
    submitter.submit_job(project_name=scene, ...)
```

### 6. Use S3 Intelligent-Tiering

**Savings: Automatic cost optimization**

```bash
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket my-bucket \
  --id optimize-storage \
  --intelligent-tiering-configuration '{
    "Id": "optimize-storage",
    "Status": "Enabled",
    "Tierings": [{
      "Days": 90,
      "AccessTier": "ARCHIVE_ACCESS"
    }]
  }'
```

### 7. Monitor and Alert

**Savings: Catch cost overruns early**

Set up AWS Budgets:
```bash
aws budgets create-budget \
  --account-id YOUR_ACCOUNT_ID \
  --budget '{
    "BudgetName": "3DGS-Monthly",
    "BudgetLimit": {
      "Amount": "1000",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }'
```

## Cost Monitoring

### View Current Costs
```bash
# Cost Explorer (requires CLI v2)
aws ce get-cost-and-usage \
  --time-period Start=2025-11-01,End=2025-11-30 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --filter file://filter.json
```

### Tag Resources for Tracking
```python
submitter.submit_job(
    project_name="scene",
    tags={
        'Project': '3DGS',
        'Environment': 'Production',
        'CostCenter': 'Research'
    }
)
```

### CloudWatch Cost Anomaly Detection

Enable automatic cost anomaly detection in AWS Console:
1. Go to AWS Cost Explorer
2. Enable Cost Anomaly Detection
3. Set threshold ($100 unexpected cost)
4. Configure SNS notifications

## Estimated Monthly Costs

### Light Usage (10 scenes/month)
```
Compute: 10 × $9.00 = $90.00
S3 Storage: 50GB × $0.025 = $1.25
S3 Requests: ~$0.50
CloudWatch: ~$1.00

Total: ~$93/month
```

### Medium Usage (50 scenes/month)
```
Compute: 50 × $9.00 = $450.00
S3 Storage: 250GB × $0.025 = $6.25
S3 Requests: ~$2.50
CloudWatch: ~$5.00

Total: ~$464/month
```

### Heavy Usage (200 scenes/month)
```
Compute: 200 × $9.00 = $1,800.00
S3 Storage: 1TB × $0.025 = $25.60
S3 Requests: ~$10.00
CloudWatch: ~$20.00

Total: ~$1,856/month

With Spot: ~$742/month (60% savings)
```

## Free Tier

AWS Free Tier includes (first 12 months):
- **S3**: 5GB storage
- **CloudWatch**: 5GB logs ingestion
- **Data Transfer**: 1GB/month

*Note: EC2/Batch GPU instances are not included in free tier.*

## Cost Comparison: Local vs AWS

### Local Development GPU Workstation
```
Hardware: $3,000-$5,000 (one-time)
Electricity: ~$50/month
Maintenance: ~$100/year
Depreciation: ~$1,000/year

Effective Monthly: ~$180-250/month
```

### AWS Batch (Medium Usage)
```
50 scenes/month: ~$464/month
With Spot: ~$186/month
```

**Break-even**: ~1 year with Spot instances

**AWS Advantages:**
- No upfront cost
- Scalability
- No maintenance
- Latest GPUs
- Pay only for usage

## Related Documentation

- [Setup Guide](./SETUP.md)
- [Quick Reference](./QUICK_REFERENCE.md)
- [Troubleshooting](./TROUBLESHOOTING.md)

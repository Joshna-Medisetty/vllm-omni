# Quick Start Guide - Claude Code in Docker

## One-Time Setup

### 1. Your token is already configured ✅
The script has your AWS Bedrock credentials.

### 2. Rebuild your Docker image
```bash
cd /workspace/vllm-omni
docker build -f docker/Dockerfile.xpu -t vllm-xpu:latest .
```

## Daily Usage

### Start containers (run in background)
```bash
bash docker/claude-credentials.sh joshna-container-1
bash docker/claude-credentials.sh joshna-container-2
bash docker/claude-credentials.sh joshna-container-3
```

### Attach to any container
```bash
docker exec -it joshna-container-1 bash
```

### Use Claude Code inside container
```bash
claude
```

### Exit container (it keeps running)
```bash
exit
```

### Re-attach later
```bash
docker exec -it joshna-container-1 bash
```

## Managing Containers

```bash
# List your running containers
docker ps | grep joshna

# Stop a container
docker stop joshna-container-1

# Start it again (credentials preserved)
docker start joshna-container-1
docker exec -it joshna-container-1 bash

# Remove a container
docker rm -f joshna-container-1
```

## What You Get

✅ Claude Code pre-installed  
✅ AWS Bedrock credentials configured  
✅ All your GPU/device settings  
✅ All your proxy settings  
✅ All your volume mounts  
✅ Runs in background (detached mode)  
✅ **No credentials on shared server filesystem**

## Need Help?

Inside Claude Code:
```bash
claude
/help
```

View full documentation: `docker/CLAUDE_SETUP.md`

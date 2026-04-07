# Claude Code Setup for joshna-omni Docker Containers

This guide explains how to use Claude Code in your Docker containers with AWS Bedrock credentials securely on a shared server.

## What Was Changed

### 1. Dockerfile.xpu Modified
- Added `ca-certificates` and `ripgrep` packages (required for Claude Code)
- Installed Claude Code CLI (pre-installed in all containers built from your image)
- Claude Code is now available at `/root/.local/bin/claude` in all containers

### 2. Created Credential Script
- `docker/claude-credentials.sh` - Securely passes credentials to containers at runtime
- **Your credentials are NEVER stored on the shared server filesystem**
- Only exists in running container memory

## Setup Instructions

### Step 1: Update Your Credentials

Edit `docker/claude-credentials.sh` and replace the placeholder with your full token:

```bash
export AWS_BEARER_TOKEN_BEDROCK="YOUR_COMPLETE_TOKEN_HERE"
```

Save the file. **Keep this file in your personal directory or secure location.**

### Step 2: Rebuild Your Docker Image

Rebuild your image to include Claude Code:

```bash
cd /workspace/vllm-omni
docker build -f docker/Dockerfile.xpu -t vllm-xpu:latest .
```

This only needs to be done once (or when you update the Dockerfile).

### Step 3: Start Containers with Credentials

Your containers run in **detached mode** (background) with all your existing settings.

#### Option A: Quick Start (Recommended)
```bash
# Start container in background
bash docker/claude-credentials.sh joshna-container-1

# Attach to the running container
docker exec -it joshna-container-1 bash
```

#### Option B: Use Helper Function
```bash
# Source the script to load the function
source docker/claude-credentials.sh

# Start container in background
start_claude_container joshna-container-2

# Attach to it
docker exec -it joshna-container-2 bash
```

#### Option C: Add to Your Existing Docker Command
```bash
# Source credentials
source docker/claude-credentials.sh

# Add these three lines to your existing docker run command:
# -e AWS_REGION="$AWS_REGION" \
# -e AWS_BEARER_TOKEN_BEDROCK="$AWS_BEARER_TOKEN_BEDROCK" \
# -e CLAUDE_CODE_USE_BEDROCK="$CLAUDE_CODE_USE_BEDROCK" \
```

## Using Claude Code Inside Container

Once inside your container, Claude Code is ready to use:

```bash
# Check Claude Code is installed
claude --version

# Verify credentials are set
echo $CLAUDE_CODE_USE_BEDROCK  # Should print: 1

# Start using Claude Code
claude
```

## What's Included in the Container

The script automatically includes **all your existing docker settings**:
- ✅ Intel GPU device mounting (`/dev/dri`)
- ✅ Proxy settings (HTTP_PROXY, HTTPS_PROXY)
- ✅ HuggingFace token (HF_TOKEN)
- ✅ All your volume mounts (vllm, vllm-omni, vllm-ci, vllm-omni-ai-crew)
- ✅ Port mapping (8001:8001)
- ✅ Shared memory (32GB)
- ✅ IPC and device settings
- ✅ Auto-restart unless stopped
- ✅ **Plus Claude Code credentials** (AWS Bedrock)

## Security Features

✅ **Credentials never stored on shared server filesystem**
- Credentials only exist as environment variables in your running container
- When container stops, credentials disappear
- Other employees cannot access your credentials

✅ **Pre-installed Claude Code**
- No need to install in each container manually
- Ready to use immediately after container starts

✅ **Isolated per container**
- Each of your 3-4 containers gets its own isolated environment
- Credentials passed independently to each container

## Example Workflow

```bash
# On shared server (as user joshna)
cd /workspace/vllm-omni

# Start containers in background (they keep running)
bash docker/claude-credentials.sh joshna-work-1
bash docker/claude-credentials.sh joshna-work-2
bash docker/claude-credentials.sh joshna-work-3

# Check they're running
docker ps | grep joshna

# Attach to any container when you need it
docker exec -it joshna-work-1 bash
```

Inside the container:
```bash
# Use Claude Code
claude

# Do your work...
cd /workspace/vllm-omni
python your_script.py

# Exit (container keeps running in background)
exit

# Later, re-attach to the same container
docker exec -it joshna-work-1 bash
```

## Troubleshooting

### Issue: Claude Code not found
```bash
# Verify PATH includes claude
echo $PATH  # Should show /root/.local/bin

# Check installation
ls -la /root/.local/bin/claude

# If missing, rebuild the Docker image
```

### Issue: Authentication failed
```bash
# Verify credentials are set
env | grep AWS
env | grep CLAUDE

# Should see:
# AWS_REGION=us-east-2
# AWS_BEARER_TOKEN_BEDROCK=<your-token>
# CLAUDE_CODE_USE_BEDROCK=1

# If missing, restart container with credentials script
```

### Issue: Other employees can see my credentials
This should NOT happen if you follow the instructions:
- Credentials are only in your terminal environment
- Not stored in any shared file
- Only passed to YOUR containers at runtime

## Important Notes

1. **Never commit credentials to git**
   - Add `claude-credentials.sh` to `.gitignore` if needed
   - Or keep it outside the git repository

2. **Rebuild image when Dockerfile changes**
   - If you update Dockerfile.xpu, rebuild the image
   - Existing containers need to be recreated from new image

3. **Container lifecycle**
   - Credentials exist only while container is running
   - When you stop/remove container, credentials are gone
   - Start new container with script each time

4. **Multiple containers**
   - You can run 3-4 containers simultaneously
   - Each gets its own isolated environment
   - All share the same credentials from your script

## Questions?

If you need help:
- Inside Claude Code, type `/help`
- Check authentication: `claude /status`
- View documentation: https://code.claude.com/docs

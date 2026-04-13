#!/bin/bash
# Claude Code AWS Bedrock Credentials Script for joshna-omni containers
# This script securely passes your AWS Bedrock credentials to Docker containers
# without storing them on the shared server filesystem.

# =============================================================================
# CONFIGURATION (set in your environment — do not put real keys in this file)
# =============================================================================
# For Bedrock / Claude Code inside a container you typically pass through:
#   AWS_REGION
#   AWS_BEARER_TOKEN_BEDROCK   (if your org uses a Bedrock bearer token)
#   CLAUDE_CODE_USE_BEDROCK=1
# Or rely on IAM/instance role and omit the bearer token.
# See docker/CLAUDE_SETUP.md. Never commit access keys or tokens to git.

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================
# This script provides THREE ways to use Claude Code with your credentials:
#
# METHOD 1 (RECOMMENDED): Run this script with container name argument
#   $ bash docker/claude-credentials.sh joshna-container-1
#   $ docker exec -it joshna-container-1 bash
#   (Container runs in background, attach when needed)
#
# METHOD 2: Use the helper function after sourcing
#   $ source docker/claude-credentials.sh
#   $ start_claude_container joshna-container-2
#   $ docker exec -it joshna-container-2 bash
#
# METHOD 3: Add credentials to your existing docker run command
#   $ source docker/claude-credentials.sh
#   $ docker run -d ... \
#       -e AWS_REGION="$AWS_REGION" \
#       -e AWS_BEARER_TOKEN_BEDROCK="$AWS_BEARER_TOKEN_BEDROCK" \
#       -e CLAUDE_CODE_USE_BEDROCK="$CLAUDE_CODE_USE_BEDROCK" \
#       ... rest of your command
#
# STARTING MULTIPLE CONTAINERS:
#   $ bash docker/claude-credentials.sh joshna-container-1
#   $ bash docker/claude-credentials.sh joshna-container-2
#   $ bash docker/claude-credentials.sh joshna-container-3
#   $ docker exec -it joshna-container-1 bash  # Attach to any container
#
# =============================================================================

# Helper function to start a container with Claude Code credentials
start_claude_container() {
    local CONTAINER_NAME=${1:-joshna-omni-$(date +%s)}
    local IMAGE_NAME=${2:-vllm-xpu:latest}

    echo "Starting container: $CONTAINER_NAME"
    echo "Using image: $IMAGE_NAME"
    echo ""

    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        --ipc=host \
        --device /dev/dri \
        -v /dev/dri:/dev/dri \
        -p 8002:8002 \
        --shm-size 32g \
        -e HTTP_PROXY="${HTTP_PROXY}" \
        -e HTTPS_PROXY="${HTTPS_PROXY}" \
        -e http_proxy="${http_proxy}" \
        -e https_proxy="${https_proxy}" \
        -e NO_PROXY="${NO_PROXY}" \
        -e no_proxy="${no_proxy}" \
        -e HF_TOKEN="${HF_TOKEN}" \
        -e HF_HOME=/root/.cache/huggingface \
        -v /home/sdp/.cache/huggingface:/root/.cache/huggingface \
        -v /home/sdp/joshna/.claude:/root/.claude \
        -v /home/sdp/joshna/vllm:/workspace/vllm \
        -v /home/sdp/joshna/vllm-omni:/workspace/vllm-omni \
        -v /home/sdp/joshna/vllm-ci:/workspace/vllm-ci \
        -v /home/sdp/joshna/vllm-omni-ai-crew:/workspace/vllm-omni-ai-crew \
        --entrypoint /bin/bash \
        "$IMAGE_NAME" \
        -c "sleep infinity"

    echo "Container $CONTAINER_NAME started in detached mode"
    echo "To attach: docker exec -it $CONTAINER_NAME bash"
}

# If script is executed (not sourced) with an argument, start container
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ -z "$1" ]; then
        echo "Usage: $0 <container-name> [image-name]"
        echo ""
        echo "Examples:"
        echo "  $0 joshna-container-1"
        echo "  $0 joshna-container-2 vllm-xpu:latest"
        echo ""
        echo "After starting, attach to the container:"
        echo "  docker exec -it joshna-container-1 bash"
        echo ""
        echo "Or source this script and use the function:"
        echo "  source $0"
        echo "  start_claude_container joshna-container-1"
        exit 1
    fi

    start_claude_container "$1" "$2"
fi

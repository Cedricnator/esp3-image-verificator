#!/bin/bash
set -e

# Ensure cache directory exists with correct permissions
mkdir -p /app/.cache/torch/checkpoints 2>/dev/null || true

# Execute the main command
exec "$@"

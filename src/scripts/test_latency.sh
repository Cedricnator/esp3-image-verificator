#!/usr/bin/env bash
# Navigate to repo root (2 levels up from src/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# Configuration
URL='http://localhost:5000/verify'
IMAGE_PATH='data/cropped/me/5_crop.jpg'
TIMES="${1:-50}"  # Default 50 iterations, can override with arg

# Check if image exists
if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "❌ Error: Image not found at ${IMAGE_PATH}"
  exit 1
fi

echo "Benchmarking API latency..."
echo "   URL: ${URL}"
echo "   Image: ${IMAGE_PATH}"
echo "   Iterations: ${TIMES}"
echo ""

# Temporary file to store latencies
TEMP_FILE=$(mktemp)

for i in $(seq 1 $TIMES); do
  # Get time_total and append to file
  curl -s -w "%{time_total}\n" -o /dev/null -X POST -F "image=@${IMAGE_PATH}" "${URL}" >> "${TEMP_FILE}"
  
  # Print individual latency
  latency=$(tail -n 1 "${TEMP_FILE}")
  echo "Request ${i}/${TIMES}: ${latency}s"
done

echo ""
echo "📊 Results:"
echo "   Total requests: ${TIMES}"




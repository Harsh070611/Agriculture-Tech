#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p static/uploads
mkdir -p saved_models

# Set environment variables if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
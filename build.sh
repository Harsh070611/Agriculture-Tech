#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip first
pip install --upgrade pip

# Install build tools first
pip install setuptools==65.5.1 wheel==0.38.4 build==0.10.0

# Install other dependencies
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p static/uploads
mkdir -p saved_models

# Set environment variables if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
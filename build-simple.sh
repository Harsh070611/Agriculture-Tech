#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip first
pip install --upgrade pip

# Install dependencies from simplified requirements
pip install -r requirements-simple.txt

# Create necessary directories if they don't exist
mkdir -p static/uploads
mkdir -p saved_models

# Set environment variables if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
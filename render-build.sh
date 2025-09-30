#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Starting build process..."

# Print Python version
python --version

# Print ffmpeg version (should be pre-installed in Render's environment)
echo "Checking for ffmpeg..."
which ffmpeg || echo "ffmpeg not found in PATH"
ffmpeg -version || echo "Could not get ffmpeg version"

# Upgrade pip and install dependencies
echo "Upgrading pip..."
pip install --upgrade pip

echo "-----> Installing Python dependencies from pyproject.toml..."
# Add -vvv for verbose output from pip
pip install . --no-cache-dir -vvv
echo "-----> Python dependencies installation attempt complete."

echo "-----> Checking google-genai installation..."
pip show google-genai

echo "-----> Attempting to import google.genai in build script..."
python -c "from google import genai; print('SUCCESS: google.genai imported correctly in build script')" || echo "FAILURE: google.genai import failed in build script"

echo "-----> Build process finished."

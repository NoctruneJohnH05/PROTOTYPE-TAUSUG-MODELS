#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit  # Exit on error

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Collecting static files..."
python manage.py collectstatic --no-input

echo "Running database migrations..."
python manage.py migrate

echo "Build completed successfully!"

# Optimized for 512MB RAM
# - Single worker (multiple workers = multiple model copies)
# - Increased timeout for first model load
# - Max requests forces worker restart to clear memory leaks
gunicorn mysite.wsgi:application \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --timeout 120 \
    --max-requests 100 \
    --max-requests-jitter 10 \
    --worker-class sync \
    --log-level info
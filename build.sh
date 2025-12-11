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

# Start with increased timeout for model loading
gunicorn mysite.wsgi:application --timeout 120 --workers 1 --bind 0.0.0.0:$PORT
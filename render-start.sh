python manage.py migrate
gunicorn mysite.wsgi:application --timeout 120 --workers 1
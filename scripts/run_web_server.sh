#!/bin/sh

# wait for PSQL server to start
sleep 3

cd /app || cd .

python src/manage.py makemigrations
python src/manage.py migrate
python src/manage.py runserver 0.0.0.0:8000

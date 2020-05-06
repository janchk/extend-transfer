#!/bin/sh

# wait for RabbitMQ server to start
sleep 5

cd /app || cd .

export PYTHONPATH=/app/src/ && celery worker -A src.ExtendTransfer_app.tasks -l info
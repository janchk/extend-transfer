#!/bin/bash
redis-server &&  sudo rabbitmq-server

export PYTHONPATH=src/
celery worker -A src.ExtendTransfer_app.tasks -l info
# Dockerfile
# Pull base image
FROM python:3.7

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
#RUN mkdir /app
WORKDIR /
ADD requirements.txt /

# Install dependencies
RUN pip install -r requirements.txt

RUN adduser --disabled-password --gecos '' default_user
# Copy project
COPY . /
FROM python:3.9-slim-buster

# Tell pipenv where the shell is. 
# This allows us to use "pipenv shell" as a container entry point.
# run docker build -t training -f Dockerfile .
ENV PYENV_SHELL=/bin/bash

ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/ac215-tbd-1.json

# Ensure we have an up to date baseline, install dependencies 
RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    mkdir -p /app

WORKDIR /app

# Add Pipfile, Pipfile.lock
ADD requirements.txt /app/
ADD *.py /app/
RUN pip install -r requirements.txt && \
apt-get install -y libcublas-12-0 

# Entry point
ENTRYPOINT ["/bin/bash","./docker-entrypoint.sh"]


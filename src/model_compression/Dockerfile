FROM python:3.9-slim-bookworm

# Tell pipenv where the shell is. 
# This allows us to use "pipenv shell" as a container entry point.
# run docker build -t training -f Dockerfile .
ENV PYENV_SHELL=/bin/bash

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
ADD requirements.txt *.py /app/
ADD *.sh /app/
RUN pip install -r requirements.txt


# Entry point
ENTRYPOINT ["/bin/bash","docker-entrypoint.sh"]


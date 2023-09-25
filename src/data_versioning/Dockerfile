FROM python:3.9-slim-bookworm

ARG DEBIAN_PACKAGES="build-essential git curl"

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/ac215-tbd-1.json
# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell pipenv where the shell is. This allows us to use "pipenv shell" as a
# container entry point.
ENV PYENV_SHELL=/bin/bash

# Tell Python to disable buffering so we don't lose any logs.
ENV PYTHONUNBUFFERED=1

# Ensure we have an up-to-date baseline and install dependencies
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    mkdir -p /app

# Set the user to root
USER root

WORKDIR /app

# Install python packages
ADD Pipfile Pipfile.lock /app/

RUN pipenv sync

# Entry point
#ENTRYPOINT ["/bin/bash","./docker-entrypoint.sh"]
ENTRYPOINT ["pipenv","shell"]
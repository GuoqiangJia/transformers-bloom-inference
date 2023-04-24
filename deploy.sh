#!/bin/bash

# Git pull
git pull

# Docker compose stop
docker compose stop

# Docker compose build
docker compose build

# Docker compose up -d
docker compose up -d
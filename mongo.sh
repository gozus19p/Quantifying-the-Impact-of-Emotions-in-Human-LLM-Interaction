#!/bin/bash

docker run -d --platform linux/arm64 -p 27017:27017 -v /Users/manuel/PersonalProjects/thesis/data:/data/db --name mongodb-thesis mongo:6.0.20

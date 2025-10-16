#!/bin/bash
# Redis Startup Script for Linux/Mac
# Starts Redis server via Docker (recommended method)

set -e

echo "========================================"
echo "Redis Server Startup Script (Linux/Mac)"
echo "========================================"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    echo "Please install Docker from: https://www.docker.com/get-docker"
    exit 1
fi

# Check if Redis container exists
if docker ps -a --filter "name=voice-cloning-redis" --format "{{.Names}}" | grep -q "voice-cloning-redis"; then
    echo "Redis container already exists. Starting..."
    docker start voice-cloning-redis
else
    echo "Creating new Redis container..."
    docker run -d \
        --name voice-cloning-redis \
        -p 6379:6379 \
        --restart unless-stopped \
        redis:7-alpine redis-server --appendonly yes
fi

echo ""
echo "Verifying Redis connection..."
sleep 2
if docker exec voice-cloning-redis redis-cli ping | grep -q "PONG"; then
    echo ""
    echo "========================================"
    echo "SUCCESS: Redis is running on localhost:6379"
    echo "========================================"
    echo ""
    echo "To stop Redis:  docker stop voice-cloning-redis"
    echo "To view logs:   docker logs -f voice-cloning-redis"
    echo "To remove:      docker rm -f voice-cloning-redis"
    echo ""
else
    echo "ERROR: Redis is not responding"
    exit 1
fi

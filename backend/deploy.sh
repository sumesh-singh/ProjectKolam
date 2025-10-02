#!/bin/bash

# Kolam AI System Deployment Script
# This script helps deploy the Kolam AI system with Docker

set -e

echo "🚀 Kolam AI System Deployment"
echo "============================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p backend/models
mkdir -p backend/generated_images
mkdir -p backend/dataset_analysis
mkdir -p backend/temp

# Check if models exist
if [ ! -f "backend/models/kolam_cnn_final.h5" ] || [ ! -f "backend/models/kolam_generator.h5" ]; then
    echo "⚠️  Models not found. Please run training first:"
    echo "   python kolam_ai_pipeline.py"
    echo ""
    echo "   Or use the quick test to verify models work:"
    echo "   python quick_test.py"
    echo ""
    read -p "Do you want to continue without models? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build and start the application
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting Kolam AI API..."
docker-compose up -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is healthy
echo "🏥 Checking application health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Application is healthy and running!"
else
    echo "⚠️  Application might still be starting. Checking logs..."
    docker-compose logs kolam-ai-api
fi

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📋 Available endpoints:"
echo "   Health Check: http://localhost:8000/health"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Pattern Recognition: POST http://localhost:8000/ai/recognize"
echo "   Pattern Generation: POST http://localhost:8000/ai/generate"
echo "   Complete Workflow: POST http://localhost:8000/ai/analyze_and_generate"
echo ""
echo "🛠️  Useful commands:"
echo "   View logs: docker-compose logs -f kolam-ai-api"
echo "   Stop application: docker-compose down"
echo "   Restart application: docker-compose restart"
echo "   Rebuild: docker-compose build --no-cache"
echo ""
echo "📝 To test the API:"
echo "   curl -X POST 'http://localhost:8000/ai/generate' \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"num_patterns\": 3}'"
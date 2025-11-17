#!/bin/bash
# SafeSpeak Deployment Script

set -e

echo "🚀 SafeSpeak Deployment Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    print_status "Docker and Docker Compose are installed"
}

# Build the application
build_app() {
    print_status "Building SafeSpeak API..."

    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi

    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found"
        exit 1
    fi

    docker build -t safespeak-api:latest .
    print_status "API built successfully"
}

# Start services
start_services() {
    print_status "Starting SafeSpeak services..."

    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found"
        exit 1
    fi

    docker-compose up -d
    print_status "Services started successfully"
}

# Wait for services to be healthy
wait_for_health() {
    print_status "Waiting for services to be healthy..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health &>/dev/null; then
            print_status "API is healthy!"
            return 0
        fi

        print_warning "Attempt $attempt/$max_attempts: API not ready yet..."
        sleep 10
        ((attempt++))
    done

    print_error "API failed to become healthy within expected time"
    return 1
}

# Test the API
test_api() {
    print_status "Testing API endpoints..."

    # Test health endpoint
    if ! curl -f http://localhost/health &>/dev/null; then
        print_error "Health check failed"
        return 1
    fi

    # Test prediction endpoint
    local test_response=$(curl -s -X POST http://localhost/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "This is a test message"}')

    if [[ $test_response == *"success"* ]]; then
        print_status "API test successful"
    else
        print_error "API test failed"
        return 1
    fi
}

# Show usage information
show_info() {
    echo ""
    print_status "SafeSpeak is now running!"
    echo ""
    echo "🌐 API Endpoints:"
    echo "   Health Check: http://localhost/health"
    echo "   Single Prediction: http://localhost/predict"
    echo "   Batch Prediction: http://localhost/predict/batch"
    echo "   API Documentation: http://localhost/docs"
    echo "   Usage Statistics: http://localhost/stats"
    echo ""
    echo "📊 Monitoring:"
    echo "   MLflow UI: http://localhost:5000"
    echo ""
    echo "🛑 To stop: docker-compose down"
    echo "🔄 To restart: docker-compose restart"
    echo "📝 To view logs: docker-compose logs -f"
}

# Main deployment function
deploy() {
    check_docker
    build_app
    start_services

    if wait_for_health; then
        if test_api; then
            show_info
            print_status "🎉 Deployment completed successfully!"
        else
            print_error "API tests failed"
            exit 1
        fi
    else
        print_error "Services failed to start properly"
        docker-compose logs
        exit 1
    fi
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker-compose down -v 2>/dev/null || true
    docker rmi safespeak-api:latest 2>/dev/null || true
    print_status "Cleanup completed"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "stop")
        print_status "Stopping services..."
        docker-compose down
        print_status "Services stopped"
        ;;
    "restart")
        print_status "Restarting services..."
        docker-compose restart
        wait_for_health && print_status "Services restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "cleanup")
        cleanup
        ;;
    "test")
        test_api
        ;;
    *)
        echo "Usage: $0 [deploy|stop|restart|logs|cleanup|test]"
        echo ""
        echo "Commands:"
        echo "  deploy   - Build and start all services (default)"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - Show service logs"
        echo "  cleanup  - Remove containers and images"
        echo "  test     - Test API endpoints"
        exit 1
        ;;
esac
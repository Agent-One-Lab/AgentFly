

    fgrt
    
    #!/bin/bash
set -e

# ALFWorld HTTP Environment Docker Build Script
# This script builds the ALFWorld HTTP server Docker image

# Configuration
IMAGE_NAME="alfworld-http-env"
IMAGE_TAG="latest"
BUILD_CONTEXT="."
DOCKERFILE="Dockerfile"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or not accessible"
        log_info "Try running: sudo systemctl start docker"
        exit 1
    fi
    
    log_success "Docker is available and running"
}

# Function to clean up old images
cleanup_old_images() {
    log_info "Cleaning up old images..."
    
    # Remove existing image if it exists
    if docker images | grep -q "$IMAGE_NAME"; then
        log_info "Removing existing image: $IMAGE_NAME:$IMAGE_TAG"
        docker rmi "$IMAGE_NAME:$IMAGE_TAG" 2>/dev/null || true
    fi
    
    # Clean up dangling images
    if docker images -f "dangling=true" -q | grep -q .; then
        log_info "Removing dangling images..."
        docker image prune -f
    fi
}

# Function to build the Docker image
build_image() {
    log_info "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
    log_info "Build context: $BUILD_CONTEXT"
    log_info "Dockerfile: $DOCKERFILE"
    
    # Build the image
    if docker build \
        --tag "$IMAGE_NAME:$IMAGE_TAG" \
        --file "$DOCKERFILE" \
        --no-cache \
        "$BUILD_CONTEXT"; then
        log_success "Docker image built successfully: $IMAGE_NAME:$IMAGE_TAG"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to verify the built image
verify_image() {
    log_info "Verifying built image..."
    
    # Check if image exists
    if ! docker images | grep -q "$IMAGE_NAME.*$IMAGE_TAG"; then
        log_error "Built image not found in docker images list"
        exit 1
    fi
    
    # Get image size
    IMAGE_SIZE=$(docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep "$IMAGE_NAME" | grep "$IMAGE_TAG" | awk '{print $3}')
    log_success "Image built successfully - Size: $IMAGE_SIZE"
    
    # Show image details
    log_info "Image details:"
    docker images | grep "$IMAGE_NAME" | grep "$IMAGE_TAG"
}

# Function to run a quick test
test_image() {
    log_info "Running quick health check test..."
    
    # Run container briefly to test health endpoint
    CONTAINER_NAME="alfworld-test-$(date +%s)"
    
    log_info "Starting test container: $CONTAINER_NAME"
    if docker run -d \
        --name "$CONTAINER_NAME" \
        --publish 8000:8000 \
        "$IMAGE_NAME:$IMAGE_TAG"; then
        
        # Wait for container to start
        log_info "Waiting for container to initialize..."
        sleep 15
        
        # Test health endpoint
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "Health check passed!"
        else
            log_warning "Health check failed - container may still be initializing"
        fi
        
        # Cleanup test container
        log_info "Cleaning up test container..."
        docker stop "$CONTAINER_NAME" &> /dev/null || true
        docker rm "$CONTAINER_NAME" &> /dev/null || true
        
    else
        log_error "Failed to start test container"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --no-cleanup    Skip cleanup of old images"
    echo "  --no-test       Skip post-build testing"
    echo "  --tag TAG       Use custom tag (default: latest)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Build with default settings"
    echo "  $0 --tag v1.0              # Build with custom tag"
    echo "  $0 --no-cleanup --no-test  # Quick build without cleanup or testing"
}

# Parse command line arguments
CLEANUP=true
TEST=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        --no-test)
            TEST=false
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "=== ALFWorld HTTP Environment Docker Build ==="
    log_info "Image: $IMAGE_NAME:$IMAGE_TAG"
    echo ""
    
    # Check prerequisites
    check_docker
    
    # Cleanup old images if requested
    if [ "$CLEANUP" = true ]; then
        cleanup_old_images
    fi
    
    # Build the image
    build_image
    
    # Verify the build
    verify_image
    
    # Test the image if requested
    if [ "$TEST" = true ]; then
        test_image
    fi
    
    echo ""
    log_success "Build completed successfully!"
    log_info "To run the container:"
    echo "  docker run -p 8000:8000 $IMAGE_NAME:$IMAGE_TAG"
    echo ""
    log_info "To push to registry:"
    echo "  docker tag $IMAGE_NAME:$IMAGE_TAG <registry>/$IMAGE_NAME:$IMAGE_TAG"
    echo "  docker push <registry>/$IMAGE_NAME:$IMAGE_TAG"
}

# Run main function
main "$@" 
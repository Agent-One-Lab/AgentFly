#!/bin/bash

# AgentFly Installation Script
# This script handles the complete installation of AgentFly and its dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if user has sudo access
check_sudo() {
    if sudo -n true 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to install enroot
install_enroot() {
    print_status "Installing enroot..."

    # Check if we're on a supported system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Debian
        if command_exists apt-get; then
            print_status "Detected Ubuntu/Debian system, installing enroot via deb packages..."

            # Get architecture
            arch=$(dpkg --print-architecture)
            if [ $? -eq 0 ]; then
                print_status "Detected architecture: $arch"
                INSTALLATION_STATUS+=("architecture detection: SUCCESS")
            else
                print_error "Failed to detect architecture"
                INSTALLATION_STATUS+=("architecture detection: FAILED")
                return 1
            fi

            # Download enroot packages
            print_status "Downloading enroot packages..."
            curl -fSsL -O "https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot-hardened_3.5.0-1_${arch}.deb"
            if [ $? -eq 0 ]; then
                print_success "Downloaded enroot-hardened package"
                INSTALLATION_STATUS+=("enroot-hardened download: SUCCESS")
            else
                print_error "Failed to download enroot-hardened package"
                INSTALLATION_STATUS+=("enroot-hardened download: FAILED")
                return 1
            fi

            curl -fSsL -O "https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot-hardened+caps_3.5.0-1_${arch}.deb"
            if [ $? -eq 0 ]; then
                print_success "Downloaded enroot-hardened+caps package"
                INSTALLATION_STATUS+=("enroot-hardened+caps download: SUCCESS")
            else
                print_error "Failed to download enroot-hardened+caps package"
                INSTALLATION_STATUS+=("enroot-hardened+caps download: FAILED")
                return 1
            fi

            # Install packages
            print_status "Installing enroot packages..."
            sudo apt install -y ./*.deb
            if [ $? -eq 0 ]; then
                print_success "enroot packages installed successfully!"
                INSTALLATION_STATUS+=("enroot package installation: SUCCESS")
            else
                print_error "Failed to install enroot packages"
                INSTALLATION_STATUS+=("enroot package installation: FAILED")
                return 1
            fi

            # Clean up downloaded packages
            rm -f ./*.deb
            print_status "Cleaned up downloaded packages"
            INSTALLATION_STATUS+=("package cleanup: SUCCESS")

        else
            print_warning "Unsupported package manager. Please install enroot manually from: https://github.com/NVIDIA/enroot/blob/master/doc/installation.md"
            return 1
        fi
    else
        print_warning "Unsupported operating system. Please install enroot manually from: https://github.com/NVIDIA/enroot/blob/master/doc/installation.md"
        return 1
    fi

    if command_exists enroot; then
        print_success "enroot installed successfully!"
        return 0
    else
        print_error "Failed to install enroot. Please install manually."
        return 1
    fi
}

# Function to check conda and create agentfly environment
setup_conda_environment() {
    if ! command_exists conda; then
        print_error "conda not found. Please install conda first."
        exit 1
    fi

    print_success "conda found"

    # Check if agentfly environment already exists
    if conda env list | grep -q "agentfly"; then
        print_status "agentfly environment already exists, activating it..."
        conda activate agentfly
        if [ $? -eq 0 ]; then
            print_success "agentfly environment activated"
            INSTALLATION_STATUS+=("conda environment activation: SUCCESS")
        else
            print_error "Failed to activate existing agentfly environment"
            INSTALLATION_STATUS+=("conda environment activation: FAILED")
            return 1
        fi
    else
        print_status "Creating new conda environment 'agentfly' with Python 3.12..."
        conda create -n agentfly python=3.12 -y
        if [ $? -eq 0 ]; then
            print_success "agentfly environment created successfully!"
            INSTALLATION_STATUS+=("conda environment creation: SUCCESS")
        else
            print_error "Failed to create agentfly environment"
            INSTALLATION_STATUS+=("conda environment creation: FAILED")
            return 1
        fi

        print_status "Activating agentfly environment..."
        conda activate agentfly
        if [ $? -eq 0 ]; then
            print_success "agentfly environment activated"
            INSTALLATION_STATUS+=("conda environment activation: SUCCESS")
        else
            print_error "Failed to activate agentfly environment"
            INSTALLATION_STATUS+=("conda environment activation: FAILED")
            return 1
        fi
    fi
}

# Function to install redis-server via conda
install_redis() {
    print_status "Installing redis-server via conda..."

    # Ensure conda is in PATH
    if command_exists conda; then
        conda install -y conda-forge::redis-server==7.4.0
        if [ $? -eq 0 ]; then
            print_success "redis-server installed successfully!"
            return 0
        else
            print_error "Failed to install redis-server"
            return 1
        fi
    else
        print_error "conda not found. Please install conda first or install redis-server manually."
        return 1
    fi
}

# Main installation function
main() {
    echo "=========================================="
    echo "    AgentFly Installation Script"
    echo "=========================================="
    echo ""

    # Check Python version (will be checked again after conda environment setup)
    print_status "Checking Python version..."
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_status "Found Python version: $PYTHON_VERSION"
        INSTALLATION_STATUS+=("Python availability: SUCCESS")
    else
        print_error "Python 3 not found. Please install Python 3.12.x first."
        INSTALLATION_STATUS+=("Python availability: FAILED")
        exit 1
    fi

    # Check pip
    print_status "Checking pip..."
    if command_exists pip3; then
        print_success "pip3 found"
        INSTALLATION_STATUS+=("pip availability: SUCCESS")
    elif command_exists pip; then
        print_success "pip found"
        INSTALLATION_STATUS+=("pip availability: SUCCESS")
    else
        print_error "pip not found. Please install pip first."
        INSTALLATION_STATUS+=("pip availability: FAILED")
        exit 1
    fi

    # Check git
    print_status "Checking git..."
    if command_exists git; then
        print_success "git found"
        INSTALLATION_STATUS+=("git availability: SUCCESS")
    else
        print_error "git not found. Please install git first."
        INSTALLATION_STATUS+=("git availability: FAILED")
        exit 1
    fi

    # Initialize git submodules
    print_status "Initializing git submodules..."
    if [ -d ".git" ]; then
        git submodule init
        if [ $? -eq 0 ]; then
            git submodule update
            if [ $? -eq 0 ]; then
                print_success "Git submodules initialized successfully!"
                INSTALLATION_STATUS+=("Git submodules: SUCCESS")
            else
                print_error "Failed to update git submodules"
                INSTALLATION_STATUS+=("Git submodules: FAILED")
            fi
        else
            print_error "Failed to init git submodules"
            INSTALLATION_STATUS+=("Git submodules: FAILED")
        fi
    else
        print_warning "Not in a git repository. Skipping submodule initialization."
        INSTALLATION_STATUS+=("Git submodules: SKIPPED (not git repo)")
    fi

    # Install Python dependencies
    print_status "Installing basic Python dependencies, this may take a while..."
    pip install -e . > /dev/null
    if [ $? -eq 0 ]; then
        print_success "Basic dependencies installed successfully!"
        INSTALLATION_STATUS+=("Basic Python dependencies: SUCCESS")
    else
        print_error "Failed to install basic dependencies"
        INSTALLATION_STATUS+=("Basic Python dependencies: FAILED")
    fi

    print_status "Installing VERL dependencies..."
    pip install -e '.[verl]' --no-build-isolation > /dev/null
    if [ $? -eq 0 ]; then
        print_success "VERL dependencies installed successfully!"
        INSTALLATION_STATUS+=("VERL dependencies: SUCCESS")
    else
        print_error "Failed to install VERL dependencies"
        INSTALLATION_STATUS+=("VERL dependencies: FAILED")
    fi

    # Check and install enroot if needed
    print_status "Checking enroot installation..."
    if command_exists enroot; then
        print_success "enroot is already installed"
    else
        print_warning "enroot not found. Some tools require it for container management."

        if check_sudo; then
            print_status "Sudo access detected. Attempting to install enroot..."
            INSTALLATION_STATUS+=("sudo access: SUCCESS")
            if install_enroot; then
                print_success "enroot installation completed!"
                INSTALLATION_STATUS+=("enroot installation: SUCCESS")
            else
                print_warning "enroot installation failed. Some tools may not work properly."
                INSTALLATION_STATUS+=("enroot installation: FAILED")
            fi
        else
            print_warning "No sudo access. Please install enroot manually from: https://github.com/NVIDIA/enroot/blob/master/doc/installation.md"
            INSTALLATION_STATUS+=("sudo access: FAILED")
            INSTALLATION_STATUS+=("enroot installation: SKIPPED (no sudo)")
        fi
    fi

    # Check conda availability
    print_status "Checking conda availability..."
    if command_exists conda; then
        print_success "conda found"
        INSTALLATION_STATUS+=("conda availability: SUCCESS")
    else
        print_error "conda not found. Please install conda first."
        INSTALLATION_STATUS+=("conda availability: FAILED")
        exit 1
    fi

    # Install redis-server (assuming we're already in a conda environment)
    print_status "Installing redis-server via conda..."
    if install_redis; then
        INSTALLATION_STATUS+=("redis-server installation: SUCCESS")
    else
        INSTALLATION_STATUS+=("redis-server installation: FAILED")
    fi

    # Final checks and summary
    echo ""
    echo "=========================================="
    echo "    Installation Summary"
    echo "=========================================="

    print_status "Checking installed components..."

    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        if [[ "$PYTHON_VERSION" =~ ^3\.12\. ]]; then
            print_success "✓ Python 3.12.x ($PYTHON_VERSION)"
            INSTALLATION_STATUS+=("Python 3.12.x verification: SUCCESS")
        else
            print_error "✗ Python version $PYTHON_VERSION does not meet requirements (need 3.10.x)"
            INSTALLATION_STATUS+=("Python 3.12.x verification: FAILED")
        fi
    else
        print_error "✗ Python 3 not found"
        INSTALLATION_STATUS+=("Python 3.12.x verification: FAILED")
    fi

    if [ -d "AgentFly.egg-info" ] || [ -d "agents" ]; then
        print_success "✓ AgentFly package"
        INSTALLATION_STATUS+=("AgentFly package verification: SUCCESS")
    else
        print_error "✗ AgentFly package not found"
        INSTALLATION_STATUS+=("AgentFly package verification: FAILED")
    fi

    if command_exists enroot; then
        print_success "✓ enroot"
        INSTALLATION_STATUS+=("enroot verification: SUCCESS")
    else
        print_warning "✗ enroot (not installed - some tools may not work)"
        INSTALLATION_STATUS+=("enroot verification: FAILED")
    fi

    if command_exists conda; then
        print_success "✓ conda"
        INSTALLATION_STATUS+=("conda verification: SUCCESS")
    else
        print_error "✗ conda not found"
        INSTALLATION_STATUS+=("conda verification: FAILED")
    fi

    # Skip agentfly environment check - assuming we're already in a conda environment
    print_success "✓ conda environment (assuming active)"
    INSTALLATION_STATUS+=("conda environment verification: SKIPPED (assumed active)")

    if command_exists redis-server; then
        print_success "✓ redis-server"
        INSTALLATION_STATUS+=("redis-server verification: SUCCESS")
    else
        print_error "✗ redis-server not found"
        INSTALLATION_STATUS+=("redis-server verification: FAILED")
    fi

    echo ""
    echo "=========================================="
    echo "    Step-by-Step Status Report"
    echo "=========================================="

    # Count successes, failures, and skips
    SUCCESS_COUNT=0
    FAILED_COUNT=0
    SKIPPED_COUNT=0

    for status in "${INSTALLATION_STATUS[@]}"; do
        if [[ $status == *"SUCCESS"* ]]; then
            echo -e "${GREEN}✓${NC} $status"
            ((SUCCESS_COUNT++))
        elif [[ $status == *"FAILED"* ]]; then
            echo -e "${RED}✗${NC} $status"
            ((FAILED_COUNT++))
        else
            echo "  $status"
            ((SKIPPED_COUNT++))
        fi
    done

    echo ""
    echo "=========================================="
    echo "    Summary Statistics"
    echo "=========================================="
    echo -e "${GREEN}Successful steps: $SUCCESS_COUNT${NC}"
    echo -e "${RED}Failed steps: $FAILED_COUNT${NC}"
    if [ $SKIPPED_COUNT -gt 0 ]; then
        echo -e "${YELLOW}Skipped steps: $SKIPPED_COUNT${NC}"
    fi

    echo ""
    if [ $FAILED_COUNT -eq 0 ]; then
        print_success "AgentFly installation completed successfully!"
    elif [ $FAILED_COUNT -le 2 ]; then
        print_warning "AgentFly installation completed with minor issues. Some features may not work properly."
    else
        print_error "AgentFly installation completed with significant issues. Please review the failed steps above."
    fi

    echo ""
    print_status "Next steps:"
    echo "  1. If you just installed enroot, you may need to restart your terminal"
    echo "  2. Check the documentation at: https://agentfly.readthedocs.io/"
    echo "  3. Try running an example: cd verl && bash examples/run_agents/run_code_agent.sh"
    echo ""
}

# Run main function
main "$@"

#!/bin/bash

# AgentFly Installation Script
# This script handles the complete installation of AgentFly and its dependencies
# using uv (https://docs.astral.sh/uv/). uv creates and manages a project-local
# virtual environment (.venv), pins Python 3.12, and installs from the committed
# uv.lock for a reproducible environment.

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

# Function to install uv
install_uv() {
    print_status "Installing uv (https://astral.sh/uv)..."
    if command_exists curl; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command_exists wget; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        print_error "Neither curl nor wget found. Install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
        return 1
    fi
    # The official installer drops uv in ~/.local/bin (or $XDG_BIN_HOME).
    export PATH="$HOME/.local/bin:$PATH"
    command_exists uv
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

# Main installation function
main() {
    echo "=========================================="
    echo "    AgentFly Installation Script"
    echo "=========================================="
    echo ""

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

    # Check out the verl submodule (imported through the src/agentfly/verl
    # symlink) ONLY if it has not been checked out yet. If it is already present,
    # leave it untouched so local changes in verl/ are never clobbered by a
    # re-run of this script. A leading '-' in `git submodule status` marks an
    # uninitialized submodule.
    print_status "Checking verl submodule..."
    if [ -d ".git" ]; then
        if git submodule status verl 2>/dev/null | grep -q '^-'; then
            print_status "verl submodule not initialized; checking it out..."
            if git submodule update --init verl; then
                print_success "verl submodule checked out!"
                INSTALLATION_STATUS+=("verl submodule: SUCCESS")
            else
                print_error "Failed to check out verl submodule"
                INSTALLATION_STATUS+=("verl submodule: FAILED")
            fi
        else
            print_success "verl submodule already present; leaving it untouched"
            INSTALLATION_STATUS+=("verl submodule: SKIPPED (already present)")
        fi
    else
        print_warning "Not in a git repository. Skipping submodule checkout."
        INSTALLATION_STATUS+=("verl submodule: SKIPPED (not git repo)")
    fi

    # Check uv (installs Python 3.12 + all dependencies from uv.lock)
    print_status "Checking uv..."
    if command_exists uv; then
        print_success "uv found ($(uv --version))"
        INSTALLATION_STATUS+=("uv availability: SUCCESS")
    else
        print_warning "uv not found. Installing it..."
        if install_uv; then
            print_success "uv installed ($(uv --version))"
            INSTALLATION_STATUS+=("uv installation: SUCCESS")
        else
            print_error "Failed to install uv. Install it manually: https://docs.astral.sh/uv/getting-started/installation/"
            INSTALLATION_STATUS+=("uv installation: FAILED")
            exit 1
        fi
    fi

    # Install AgentFly + training (verl) dependencies into a project-local .venv.
    # uv provisions Python 3.12, resolves from uv.lock, and builds liger-kernel
    # without build isolation (configured in pyproject.toml [tool.uv]).
    print_status "Installing AgentFly and training dependencies (this may take a while)..."
    if uv sync --extra verl; then
        print_success "AgentFly (with verl extras) installed successfully!"
        INSTALLATION_STATUS+=("AgentFly dependencies: SUCCESS")
    else
        print_error "Failed to install AgentFly dependencies"
        INSTALLATION_STATUS+=("AgentFly dependencies: FAILED")
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

    # Final checks and summary
    echo ""
    echo "=========================================="
    echo "    Installation Summary"
    echo "=========================================="

    print_status "Checking installed components..."

    if command_exists uv; then
        print_success "✓ uv ($(uv --version))"
        INSTALLATION_STATUS+=("uv verification: SUCCESS")
    else
        print_error "✗ uv not found"
        INSTALLATION_STATUS+=("uv verification: FAILED")
    fi

    if [ -x ".venv/bin/python" ]; then
        VENV_PYTHON_VERSION=$(.venv/bin/python --version 2>&1 | awk '{print $2}')
        if [[ "$VENV_PYTHON_VERSION" =~ ^3\.12\. ]]; then
            print_success "✓ .venv Python 3.12.x ($VENV_PYTHON_VERSION)"
            INSTALLATION_STATUS+=(".venv Python 3.12.x verification: SUCCESS")
        else
            print_warning "✗ .venv Python is $VENV_PYTHON_VERSION (expected 3.12.x)"
            INSTALLATION_STATUS+=(".venv Python 3.12.x verification: FAILED")
        fi
    else
        print_error "✗ .venv not found"
        INSTALLATION_STATUS+=(".venv verification: FAILED")
    fi

    if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import agentfly" >/dev/null 2>&1; then
        print_success "✓ agentfly importable"
        INSTALLATION_STATUS+=("AgentFly package verification: SUCCESS")
    else
        print_error "✗ agentfly not importable"
        INSTALLATION_STATUS+=("AgentFly package verification: FAILED")
    fi

    if command_exists enroot; then
        print_success "✓ enroot"
        INSTALLATION_STATUS+=("enroot verification: SUCCESS")
    else
        print_warning "✗ enroot (not installed - some tools may not work)"
        INSTALLATION_STATUS+=("enroot verification: FAILED")
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
    echo "  1. Activate the environment:  source .venv/bin/activate   (or prefix commands with 'uv run')"
    echo "  2. If you just installed enroot, you may need to restart your terminal"
    echo "  3. Redis-backed tools (e.g. search) need a redis-server on your PATH; install it separately if you use them"
    echo "  4. Check the documentation at: https://agent-one-lab.github.io/AgentFly"
    echo ""
}

# Run main function
main "$@"

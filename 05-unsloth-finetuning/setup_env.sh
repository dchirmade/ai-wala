#!/bin/bash

# ==========================================
#  Unsloth AI Setup for RTX 3050 (Ubuntu)
# ==========================================

# Determine if the script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    SOURCED=1
else
    SOURCED=0
fi

set -e # Stop on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Define Environment Name
ENV_NAME="unsloth_env"

echo -e "${GREEN}>>> 1. Checking Environment...${NC}"

if [ -d "$ENV_NAME" ]; then
    echo -e "${YELLOW}    Found existing environment: $ENV_NAME${NC}"
    echo "    Skipping creation."
else
    echo "    Creating new virtual environment: $ENV_NAME"
    python3 -m venv "$ENV_NAME"
fi

# Activate the environment
# shellcheck source=/dev/null
if [ -f "$ENV_NAME/bin/activate" ]; then
    source "$ENV_NAME/bin/activate"
    echo -e "${GREEN}    Environment active in script context.${NC}"
else
    echo "Error: Cannot find activation script."
    exit 1
fi

echo -e "${GREEN}>>> 2. Verifying Dependencies...${NC}"

# Check if unsloth is already installed to avoid redundant pip operations
if python3 -c "import unsloth" 2>/dev/null; then
    echo -e "${YELLOW}    Unsloth is already installed. Skipping Heavy Installation to save time.${NC}"
    echo "    (If you need to force update, delete '$ENV_NAME' and rerun)"
else
    echo "    Installing/Updating Dependencies..."
    
    echo "    > Upgrading pip..."
    pip install --upgrade pip

    echo "    > Installing PyTorch (CUDA 12.4)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    echo "    > Installing Unsloth and Core Dependencies..."
    pip install unsloth
    pip install --no-deps trl peft accelerate bitsandbytes
    pip install datasets scipy packaging
fi

echo "=========================================="
echo -e "${GREEN} Setup Complete!${NC}"
echo "=========================================="

if [ $SOURCED -eq 1 ]; then
    echo -e "${GREEN}✅ Environment is now ACTIVE in your shell.${NC}"
    echo "   You can proceed to run: python train_unsloth.py"
else
    echo -e "${YELLOW}⚠️  You ran this as a script (./setup_env.sh).${NC}"
    echo "   To activate the environment in your terminal, run:"
    echo -e "   ${GREEN}source $ENV_NAME/bin/activate${NC}"
    echo ""
    echo "   Then run:"
    echo "   python train_unsloth.py"
fi
echo "=========================================="

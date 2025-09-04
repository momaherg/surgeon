#!/bin/bash

# Script to test NPT permanent update functionality

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}NPT Permanent Update Testing${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check if checkpoint path is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No checkpoint path provided${NC}"
    echo "Usage: $0 <checkpoint_path> [--demo|--interactive]"
    echo "Example: $0 ./outputs/npt-improved-1B/checkpoint-500 --demo"
    exit 1
fi

CHECKPOINT_PATH=$1
MODE=${2:-"--interactive"}  # Default to interactive mode

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}Error: Checkpoint directory does not exist: $CHECKPOINT_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}Checkpoint: $CHECKPOINT_PATH${NC}"
echo -e "${BLUE}Mode: $MODE${NC}\n"

# Check for Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Run the appropriate script based on mode
if [ "$MODE" == "--demo" ]; then
    echo -e "${GREEN}Running demonstration script...${NC}\n"
    python demo_permanent_update.py "$CHECKPOINT_PATH"
elif [ "$MODE" == "--interactive" ]; then
    echo -e "${GREEN}Starting interactive mode...${NC}\n"
    python interactive_permanent_update.py "$CHECKPOINT_PATH"
else
    echo -e "${RED}Error: Unknown mode $MODE${NC}"
    echo "Use --demo or --interactive"
    exit 1
fi

echo -e "\n${GREEN}Testing completed!${NC}"

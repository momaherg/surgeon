#!/bin/bash

# Enhanced NPT Permanent Update Testing Script
# Now with stronger default settings for better fact retention

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}NPT Enhanced Permanent Update Testing${NC}"
echo -e "${CYAN}========================================${NC}\n"

# Check if checkpoint path is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No checkpoint path provided${NC}"
    echo "Usage: $0 <checkpoint_path> [mode] [modulation_scale]"
    echo ""
    echo "Modes:"
    echo "  --interactive   Interactive mode with enhanced settings (default)"
    echo "  --experiment    Run experiments to find best configuration"
    echo "  --demo          Run demonstration with predefined facts"
    echo ""
    echo "Examples:"
    echo "  $0 ./outputs/npt-improved-1B/checkpoint-500"
    echo "  $0 ./outputs/npt-improved-1B/checkpoint-500 --experiment"
    echo "  $0 ./outputs/npt-improved-1B/checkpoint-500 --interactive 0.7"
    exit 1
fi

CHECKPOINT_PATH=$1
MODE=${2:-"--interactive"}
MODULATION_SCALE=${3:-"0.5"}

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}Error: Checkpoint directory does not exist: $CHECKPOINT_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Checkpoint: $CHECKPOINT_PATH"
echo -e "  Mode: $MODE"
if [ "$MODE" == "--interactive" ]; then
    echo -e "  Modulation scale: $MODULATION_SCALE (original: 0.1)"
fi
echo ""

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

# Show enhancement notice
echo -e "${GREEN}This enhanced version uses stronger settings:${NC}"
echo -e "  ${CYAN}• Modulation scale: 5x stronger (0.5 vs 0.1)${NC}"
echo -e "  ${CYAN}• Update strength: 10x stronger (α=1.0 vs 0.1)${NC}"
echo -e "  ${CYAN}• Multiple iterations for better retention${NC}"
echo -e "  ${CYAN}• Configurable per-fact settings${NC}"
echo ""

# Run the appropriate script based on mode
case "$MODE" in
    "--experiment")
        echo -e "${MAGENTA}Running experiments to find optimal configuration...${NC}\n"
        python improved_permanent_update.py "$CHECKPOINT_PATH"
        ;;
    "--demo")
        echo -e "${MAGENTA}Running demonstration with original settings...${NC}\n"
        python demo_permanent_update.py "$CHECKPOINT_PATH"
        echo -e "\n${YELLOW}Note: The demo uses original weak settings.${NC}"
        echo -e "${YELLOW}For better results, use --interactive or --experiment mode.${NC}"
        ;;
    "--interactive"|*)
        echo -e "${MAGENTA}Starting enhanced interactive mode...${NC}\n"
        echo -e "${GREEN}New commands available:${NC}"
        echo -e "  ${CYAN}inject+${NC} - Inject with custom settings"
        echo -e "  ${CYAN}config${NC}  - View/change default configuration"
        echo ""
        python interactive_permanent_update_enhanced.py "$CHECKPOINT_PATH" "$MODULATION_SCALE"
        ;;
esac

echo -e "\n${GREEN}Testing completed!${NC}"

# Show tips
if [ "$MODE" != "--demo" ]; then
    echo -e "\n${YELLOW}Tips for better results:${NC}"
    echo -e "  • Start with α=1.0, iterations=2"
    echo -e "  • If facts aren't retained, try α=1.5, iterations=3"
    echo -e "  • Target upper layers (16-20) for fact storage"
    echo -e "  • Test recall immediately after injection"
fi

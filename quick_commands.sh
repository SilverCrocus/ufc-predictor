#!/bin/bash
# UFC Predictor - Quick Commands
# Simple shortcuts for common operations

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

case "$1" in
    train)
        echo -e "${GREEN}🏋️ Training with walk-forward validation...${NC}"
        uv run main.py pipeline --tune --optimize
        ;;
    
    backtest)
        echo -e "${BLUE}📊 Running ROI backtest with your test set odds...${NC}"
        uv run proper_roi_backtest.py
        ;;
    
    predict)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${YELLOW}Usage: ./quick_commands.sh predict 'Fighter1' 'Fighter2'${NC}"
        else
            uv run main.py predict --fighter1 "$2" --fighter2 "$3"
        fi
        ;;
    
    bet)
        echo -e "${GREEN}💰 Analyzing betting opportunities...${NC}"
        uv run main.py betting --bankroll 1000
        ;;
    
    validate)
        echo -e "${BLUE}🔍 Checking model overfitting...${NC}"
        uv run run_walkforward_with_optimized.py
        ;;
    
    roi)
        echo -e "${GREEN}💵 Calculating realistic ROI...${NC}"
        uv run recalculate_roi_walkforward.py
        ;;
    
    help|*)
        echo -e "${GREEN}UFC Predictor - Quick Commands${NC}"
        echo ""
        echo "Usage: ./quick_commands.sh [command]"
        echo ""
        echo "Commands:"
        echo "  train     - Train models with walk-forward validation"
        echo "  backtest  - Run realistic walk-forward backtest"
        echo "  predict   - Predict a fight outcome"
        echo "  bet       - Analyze current betting opportunities"
        echo "  validate  - Check model overfitting"
        echo "  roi       - Calculate realistic ROI potential"
        echo ""
        echo "Examples:"
        echo "  ./quick_commands.sh train"
        echo "  ./quick_commands.sh predict 'Max Holloway' 'Ilia Topuria'"
        echo "  ./quick_commands.sh backtest"
        ;;
esac
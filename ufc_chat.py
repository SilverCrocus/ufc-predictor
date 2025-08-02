#!/usr/bin/env python3
"""
UFC Chat - Conversational Interface Entry Point
==============================================

Command-line entry point for the conversational UFC betting analysis system.
Provides natural language interaction with the sophisticated ML prediction 
and profitability analysis capabilities.

Usage:
    python3 ufc_chat.py                    # Start interactive chat
    python3 ufc_chat.py --demo             # Demo mode (no live data)
    python3 ufc_chat.py --help             # Show help
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.conversational_ufc_interface import ConversationalUFCInterface
    INTERFACE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Conversational interface not available: {e}")
    INTERFACE_AVAILABLE = False


def create_banner():
    """Create welcome banner for the chat interface"""
    return """
╔══════════════════════════════════════════════════════════════╗
║                    🥊 UFC BETTING ASSISTANT 🥊                 ║
║                  Conversational AI Interface                  ║
╠══════════════════════════════════════════════════════════════╣
║  Powered by 72.88% accurate ML models trained on 21K+ fights  ║
║  Features: Predictions • Betting Analysis • Risk Management   ║
╚══════════════════════════════════════════════════════════════╝
"""


def show_quick_start():
    """Show quick start guide"""
    return """
🚀 **QUICK START GUIDE**

1️⃣ **Set Your Bankroll**: "I have $500 to bet with"
2️⃣ **Get Predictions**: "Who wins Jones vs Miocic?"  
3️⃣ **Betting Analysis**: "Should I bet on this fight?"
4️⃣ **Get Help**: Type "help" anytime for more commands

**Example Session**:
👤 You: "Set my bankroll to $1000"
🤖 Bot: *Sets up risk management for $1000 bankroll*
👤 You: "Predict Israel Adesanya vs Alex Pereira"  
🤖 Bot: *ML prediction with confidence levels*
👤 You: "Should I bet on Adesanya?"
🤖 Bot: *Expected value analysis with bet sizing*

Ready to start? Ask me anything about UFC betting! 🥊
"""


def run_demo_mode():
    """Run a demo conversation to show capabilities"""
    print("🎬 **DEMO MODE** - Showing conversational capabilities\n")
    
    # Simulate a conversation
    demo_conversation = [
        ("Hello!", "🥊 Hey there! I'm your UFC betting assistant..."),
        ("Set my bankroll to $500", "💰 Bankroll Set: $500\nRisk Tier: MEDIUM..."),
        ("Who wins Jon Jones vs Tom Aspinall?", "🥊 Jon Jones vs Tom Aspinall\n🏆 Predicted Winner: Jon Jones (67.3%)..."),
        ("Should I bet on Jones?", "💰 Betting Analysis (Bankroll: $500)\n🎯 Found profitable opportunity..."),
        ("How accurate are your models?", "🤖 Model Performance Status\nCurrent Accuracy: 72.88%...")
    ]
    
    for user_msg, bot_response in demo_conversation:
        print(f"👤 You: {user_msg}")
        print(f"🤖 Bot: {bot_response[:80]}...")
        print()
        input("Press Enter to continue...")
        print()
    
    print("🎬 **Demo Complete!** Run without --demo flag to use the real system.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="UFC Betting Assistant - Conversational Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ufc_chat.py                    Start interactive chat
  python3 ufc_chat.py --demo             Run demo mode
  python3 ufc_chat.py --quick-start      Show quick start guide
  
Features:
  • Natural language fight predictions
  • Betting profitability analysis  
  • Risk management with Kelly criterion
  • Model performance insights
  • Live odds integration (TAB Australia)
        """
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run in demo mode (simulated conversation)'
    )
    
    parser.add_argument(
        '--quick-start',
        action='store_true', 
        help='Show quick start guide and exit'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='UFC Chat v1.0 - Conversational Betting Assistant'
    )
    
    args = parser.parse_args()
    
    # Show banner
    print(create_banner())
    
    # Handle quick start
    if args.quick_start:
        print(show_quick_start())
        return
    
    # Handle demo mode
    if args.demo:
        run_demo_mode()
        return
    
    # Check if interface is available
    if not INTERFACE_AVAILABLE:
        print("⚠️  **System Error**: Conversational interface not available")
        print("\nThis might be due to:")
        print("• Missing dependencies")
        print("• Python path issues")
        print("• Module import problems")
        print("\nTry running from the project root directory:")
        print("cd /path/to/ufc-predictor && python3 ufc_chat.py")
        sys.exit(1)
    
    # Show quick start guide
    print(show_quick_start())
    
    # Initialize and start chat interface
    try:
        print("🔄 Initializing UFC betting assistant...")
        interface = ConversationalUFCInterface()
        print("✅ System ready! Starting chat session...\n")
        
        # Start interactive chat
        interface.start_chat_session()
        
    except KeyboardInterrupt:
        print("\n👋 Thanks for using UFC Chat! Good luck with your bets!")
    except Exception as e:
        print(f"\n❌ **Error**: {str(e)}")
        print("\nIf this persists, try:")
        print("• Checking your Python environment")
        print("• Running from the project root directory") 
        print("• Using --demo mode to test the interface")
        sys.exit(1)


if __name__ == "__main__":
    main()
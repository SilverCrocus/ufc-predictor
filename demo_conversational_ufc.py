#!/usr/bin/env python3
"""
UFC Conversational Interface Demo
================================

Interactive demonstration of the conversational UFC betting assistant
for analyzing upcoming cards with natural language interaction.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def show_demo_banner():
    """Show the demo banner"""
    return """
╔══════════════════════════════════════════════════════════════════════╗
║                 🥊 UFC CONVERSATIONAL ASSISTANT DEMO 🥊                ║
║                        Natural Language Betting Analysis              ║
╠══════════════════════════════════════════════════════════════════════╣
║  🤖 AI-Powered: 72.88% accurate ML models (21,074 fights trained)     ║
║  💬 Conversational: Natural language interface for complex analysis   ║
║  💰 Smart Betting: Kelly criterion + Expected value calculations      ║
║  🎯 Real-time: Live odds integration with risk management             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

def demo_conversation_flow():
    """Show a sample conversation demonstrating capabilities"""
    print("🎬 **DEMO CONVERSATION** - See how natural language works with sophisticated UFC analysis")
    print("=" * 80)
    
    conversation = [
        {
            "user": "Hello! I want to analyze the upcoming UFC card",
            "response": """🥊 **Hey there! I'm your UFC betting assistant.**

I can help you with:
• 🔮 **Fight predictions**: "Who wins Jones vs Miocic?"
• 📊 **Card analysis**: "Analyze the upcoming UFC card"  
• 💰 **Betting advice**: "What should I bet on?"
• ⚙️ **Model insights**: "How accurate are your predictions?"

Ready to start? Let's set up your analysis preferences!"""
        },
        {
            "user": "Set my bankroll to $1000",
            "response": """💰 **Bankroll Set: $1000**

**Risk Tier**: MEDIUM
**Max Single Bet**: $50 (5% rule)
**Max Multi-bet**: $20 (2% rule)

**Betting Strategy**:
• Conservative: 1-2% per bet
• Moderate: 3-5% for good opportunities  
• Aggressive: Max 5% for high confidence

Ready to analyze some fights! Ask me for predictions or betting advice."""
        },
        {
            "user": "Who wins Jon Jones vs Tom Aspinall?",
            "response": """🥊 **Jon Jones vs Tom Aspinall**

🏆 **Predicted Winner**: Jon Jones (55.6% confidence)
⚔️ **Predicted Method**: KO/TKO

📊 **Confidence Level**: Modest confidence - leaning this way

🥊 **Method Breakdown**:
• Decision: 5.1%
• KO/TKO: 68.9%
• Submission: 26.0%

🤖 **Model Info**: Based on 72.88% accurate ensemble (21K+ fights)

💡 Want betting analysis? I can check live odds and calculate expected value with your $1000 bankroll."""
        },
        {
            "user": "Should I bet on Jones?",
            "response": """💰 **Betting Analysis** (Bankroll: $1000)

🎯 **Found 1 profitable opportunity**
💵 **Expected profit**: $42.50

**Jon Jones vs Tom Aspinall**
   💰 Bet: $35.00 → Expected profit: $42.50
   📈 Edge: 12.1% expected value
   🎲 TAB Odds: $1.85

🛡️ **Risk Management**: Balanced approach: Max 5% per bet, consider correlation for multi-bets

📊 **Data**: Live TAB Australia odds (real-time)"""
        },
        {
            "user": "How accurate are your models?",
            "response": """🤖 **Model Performance Status**

**Current Accuracy**: 72.88% (winner prediction)
**Training Data**: 21,074 fights with 70 features
**Model Type**: Random Forest + XGBoost ensemble

**Recent Performance**:
• Winner Model: 72.88% accuracy (tuned)
• Method Model: 74.50% accuracy 
• Training Session: 2025-07-23 (latest)

**Confidence Levels**:
• 🟢 High (70%+): Very reliable predictions
• 🟡 Moderate (55-70%): Decent edge, use with caution
• 🔴 Low (<55%): Avoid betting, too close to call

The models are retrained regularly with new fight data to maintain accuracy."""
        }
    ]
    
    for i, turn in enumerate(conversation, 1):
        print(f"\n**Turn {i}**")
        print(f"👤 **You**: {turn['user']}")
        print(f"🤖 **UFC Assistant**: {turn['response']}")
        print("\n" + "-" * 80)
        
        if i < len(conversation):
            input("Press Enter to continue to next turn...")

def show_features_summary():
    """Show summary of conversational features"""
    return """
🌟 **CONVERSATIONAL FEATURES IMPLEMENTED**

**🗣️ Natural Language Understanding**:
• Intent recognition for fight predictions, betting analysis, system queries
• Fighter name extraction from various formats ("Jones vs Aspinall", "who wins X vs Y")  
• Monetary amount parsing ($500, 1000, etc.)
• Context-aware follow-up questions

**🧠 Intelligent Context Management**:
• Remembers your bankroll across conversation
• Tracks risk tolerance and technical level preferences
• Maintains conversation history for follow-ups
• Session state management

**⚡ Real-Time Integration**:
• Calls actual UFC ML prediction system (72.88% accuracy)
• Integrates with live TAB Australia odds scraping
• Expected value calculations with Kelly criterion
• Risk-adjusted bet sizing recommendations

**🎯 User Experience**:
• Progressive disclosure (basic → detailed on request)
• Error handling with helpful suggestions
• Fallback modes when systems unavailable
• Technical level adaptation (beginner/intermediate/expert)

**💡 Smart Features**:
• Bankroll tier determination (MICRO/SMALL/MEDIUM/LARGE)
• Risk management advice based on user preferences
• Model confidence communication in natural language
• System status monitoring and reporting
"""

def show_usage_examples():
    """Show practical usage examples"""
    return """
🎯 **HOW TO USE THE CONVERSATIONAL INTERFACE**

**🚀 Quick Start**:
```bash
python3 ufc_chat.py                 # Start interactive chat
python3 ufc_chat.py --demo          # See demo conversation
python3 ufc_chat.py --quick-start   # Show quick start guide
```

**💬 Natural Language Examples**:

**Fight Predictions**:
• "Who wins Jon Jones vs Tom Aspinall?"
• "Predict Israel Adesanya vs Alex Pereira"
• "Analyze the main event tonight"

**Bankroll Management**:
• "Set my bankroll to $500"
• "I have $1000 to bet with"
• "My budget is $200"

**Betting Analysis**:
• "Should I bet on this fight?"
• "What's profitable this weekend?"
• "Give me betting recommendations"

**System Information**:
• "How accurate are your models?"
• "Show me system status"
• "Help me understand your predictions"

**📱 Integration Ready**:
• Command-line interface (ready now)
• Web interface integration (future)
• Discord/Slack bot potential
• API endpoints for mobile apps
"""

def main():
    """Main demo function"""
    print(show_demo_banner())
    
    print("\n🎬 Choose your demo experience:")
    print("1. 🗣️  Interactive conversation demo")
    print("2. 📚 Features and capabilities overview")  
    print("3. 💻 Usage examples and getting started")
    print("4. 🚀 Launch live conversational interface")
    print("5. ❌ Exit")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            demo_conversation_flow()
        elif choice == '2':
            print(show_features_summary())
        elif choice == '3':
            print(show_usage_examples())
        elif choice == '4':
            print("\n🚀 Launching conversational interface...")
            os.system("python3 ufc_chat.py")
        elif choice == '5':
            print("👋 Thanks for checking out the UFC Conversational Assistant!")
            return
        else:
            print("❌ Invalid option. Please choose 1-5.")
            
    except KeyboardInterrupt:
        print("\n👋 Demo ended. Thanks for your interest!")
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    main()
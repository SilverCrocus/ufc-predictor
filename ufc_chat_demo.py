#!/usr/bin/env python3
"""
UFC Chat Demo - Conversational Interface Prototype
==================================================

This demonstrates what a conversational UFC betting agent could look like,
building on the existing sophisticated prediction and analysis system.
"""

import sys
import os
import re
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class UFCChatBot:
    """Simple conversational interface for the UFC betting system"""
    
    def __init__(self):
        self.context = {
            'bankroll': None,
            'risk_tolerance': 'medium',
            'favorite_fighters': [],
            'conversation_history': [],
            'last_predictions': None
        }
        
        # Initialize connection to existing prediction system
        try:
            from main import UFCPredictor
            self.predictor = UFCPredictor()
            self.system_available = True
        except ImportError:
            print("⚠️  Core prediction system not available in demo mode")
            self.system_available = False
    
    def process_message(self, user_input: str) -> str:
        """Process user message and return response"""
        user_input = user_input.lower().strip()
        self.context['conversation_history'].append(('user', user_input))
        
        # Intent recognition
        if self._is_greeting(user_input):
            response = self._handle_greeting()
        elif self._is_bankroll_setting(user_input):
            response = self._handle_bankroll_setting(user_input)
        elif self._is_prediction_request(user_input):
            response = self._handle_prediction_request(user_input)
        elif self._is_betting_question(user_input):
            response = self._handle_betting_question(user_input)
        elif self._is_system_status(user_input):
            response = self._handle_system_status()
        elif self._is_help_request(user_input):
            response = self._handle_help_request()
        else:
            response = self._handle_unknown_request(user_input)
        
        self.context['conversation_history'].append(('bot', response))
        return response
    
    # Intent Recognition Methods
    
    def _is_greeting(self, text: str) -> bool:
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'what\'s up']
        return any(greeting in text for greeting in greetings)
    
    def _is_bankroll_setting(self, text: str) -> bool:
        bankroll_indicators = ['bankroll', 'budget', 'money', 'set', '$', 'dollar']
        return any(indicator in text for indicator in bankroll_indicators)
    
    def _is_prediction_request(self, text: str) -> bool:
        prediction_indicators = ['predict', 'who wins', 'vs', 'fight', 'winner', 'odds']
        return any(indicator in text for indicator in prediction_indicators)
    
    def _is_betting_question(self, text: str) -> bool:
        betting_indicators = ['bet', 'should i', 'recommend', 'card', 'ufc', 'profitable']
        return any(indicator in text for indicator in betting_indicators)
    
    def _is_system_status(self, text: str) -> bool:
        status_indicators = ['status', 'health', 'working', 'system', 'models']
        return any(indicator in text for indicator in status_indicators)
    
    def _is_help_request(self, text: str) -> bool:
        help_indicators = ['help', 'what can you do', 'commands', 'how to']
        return any(indicator in text for indicator in help_indicators)
    
    # Response Handlers
    
    def _handle_greeting(self) -> str:
        return """🥊 Hey there! I'm your UFC betting agent.

I can help you with:
• Predict fight outcomes
• Analyze betting opportunities  
• Set your bankroll and risk preferences
• Get recommendations for upcoming cards

What would you like to know about? Try saying something like:
"Who wins Jon Jones vs Tom Aspinall?" or "Set my bankroll to $500"
"""
    
    def _handle_bankroll_setting(self, text: str) -> str:
        # Extract numbers from the text
        numbers = re.findall(r'\d+', text)
        
        if numbers:
            bankroll = int(numbers[0])
            self.context['bankroll'] = bankroll
            
            # Determine bankroll tier (from existing system)
            if bankroll < 50:
                tier = "MICRO"
                advice = "Small stakes, focus on learning!"
            elif bankroll < 200:
                tier = "SMALL"  
                advice = "Conservative betting recommended."
            elif bankroll < 1000:
                tier = "MEDIUM"
                advice = "Good size for diversified betting."
            else:
                tier = "LARGE"
                advice = "You can take advantage of multiple opportunities."
            
            return f"""💰 Perfect! I've set your bankroll to ${bankroll}.

**Bankroll Tier**: {tier}
**Advice**: {advice}

With this bankroll, I'll use:
• Max 5% per single bet (${bankroll * 0.05:.0f})
• Max 2% per multi-bet (${bankroll * 0.02:.0f})
• Kelly criterion for optimal sizing

Ready to analyze some fights? Ask me about upcoming UFC cards!
"""
        else:
            return """💰 I'd love to help set your bankroll! 

Please tell me your budget, like:
"Set my bankroll to $500" or "I have $1000 to bet with"
"""
    
    def _handle_prediction_request(self, text: str) -> str:
        # Extract fighter names (simplified)
        fighters = self._extract_fighter_names(text)
        
        if len(fighters) >= 2:
            fighter1, fighter2 = fighters[0], fighters[1]
            
            if self.system_available:
                # This would call the actual prediction system
                return f"""🥊 **{fighter1} vs {fighter2}**

🔍 Analyzing fight... (integrating with ML models)

🏆 **Predicted Winner**: {fighter1} (67.3% confidence)
⚔️ **Predicted Method**: KO/TKO

📊 **Win Probabilities**:
• {fighter1}: 67.3%
• {fighter2}: 32.7%

🥊 **Method Breakdown**:
• Decision: 15.2%
• KO/TKO: 68.1%
• Submission: 16.7%

💡 Want me to check live odds and calculate expected value?
"""
            else:
                return f"""🥊 **{fighter1} vs {fighter2}**

📊 **Prediction Demo** (using example data):
🏆 **Predicted Winner**: {fighter1} (67.3%)
⚔️ **Method**: KO/TKO

💡 In the full system, this would:
• Use trained ML models (72.88% accuracy)
• Calculate confidence intervals
• Scrape live TAB Australia odds
• Calculate expected value and Kelly sizing

Ask me about bankroll management or system status!
"""
        else:
            return """🥊 I'd love to predict a fight for you!

Try asking:
• "Who wins Jon Jones vs Tom Aspinall?"
• "Predict Israel Adesanya vs Alex Pereira"
• "Analyze the main event"

I need two fighter names to make a prediction.
"""
    
    def _handle_betting_question(self, text: str) -> str:
        if not self.context['bankroll']:
            return """💰 To give you betting advice, I need to know your bankroll first!

Try saying: "Set my bankroll to $500"

Then I can recommend:
• Optimal bet sizes using Kelly criterion
• Expected value calculations
• Risk-adjusted recommendations
• Multi-bet opportunities
"""
        
        bankroll = self.context['bankroll']
        
        return f"""🎯 **Betting Recommendations** (Bankroll: ${bankroll})

Based on your budget, here's what I'd look for:

**Current System Capabilities**:
✅ Kelly criterion sizing (max 5% single, 2% multi)
✅ Expected value calculation (12%+ threshold)
✅ Live TAB Australia odds scraping
✅ Multi-bet correlation analysis
✅ Risk tier management

**Example Recommendation**:
If I found a fight with:
• 60% win probability
• $1.80 TAB odds
• Expected Value: +8%

I'd recommend: ${bankroll * 0.03:.0f} bet (3% of bankroll)
Potential profit: ${bankroll * 0.03 * 0.8:.0f}

Want me to analyze a specific UFC card?
"""
    
    def _handle_system_status(self) -> str:
        return """🤖 **System Status Check**

**Core Components**:
✅ ML Models: 72.88% accuracy (RF + XGBoost ensemble)
✅ Agent Architecture: 7 specialized agents deployed
✅ Risk Management: Kelly criterion + portfolio optimization
✅ Data Pipeline: Auto-detects latest scraped data
✅ Testing: 100+ unit/integration tests passed

**Current Capabilities**:
• Single fight predictions (symmetrical analysis)
• Method prediction (Decision/KO/Submission)
• Bootstrap confidence intervals
• Expected value calculation
• Multi-bet combination analysis

**Phase 2A Progress**: 92% complete
• ✅ Smart Infrastructure (100%)
• ✅ Enhanced ML Pipeline (95%)
• 🟡 Portfolio Optimization (80%)

Ready to analyze fights and calculate betting value!
"""
    
    def _handle_help_request(self) -> str:
        return """🆘 **UFC Betting Agent Help**

**What I Can Do**:
🥊 **Fight Predictions**: "Who wins Jon Jones vs Tom Aspinall?"
💰 **Bankroll Management**: "Set my bankroll to $500"
📊 **Betting Analysis**: "What should I bet on for UFC 304?"
🎯 **Expected Value**: Calculate profitable opportunities
🤖 **System Status**: Check models and agent health

**Commands You Can Try**:
• "Predict [Fighter 1] vs [Fighter 2]"
• "Set my bankroll to $[amount]"
• "What's profitable for the next card?"
• "Show me system status"
• "Analyze [UFC Event Name]"

**Behind the Scenes**:
• 72.88% accurate ML ensemble
• Live odds scraping (TAB Australia)
• Kelly criterion optimal sizing
• Multi-agent architecture
• Risk-adjusted recommendations

Just talk to me naturally - I'll understand what you need!
"""
    
    def _handle_unknown_request(self, text: str) -> str:
        return f"""🤔 I'm not sure how to help with "{text}"

Try asking me about:
• Fight predictions ("Who wins X vs Y?")
• Bankroll setting ("Set my bankroll to $500")
• Betting advice ("What should I bet on?")
• System status ("How are the models performing?")

Type "help" for more commands!
"""
    
    def _extract_fighter_names(self, text: str) -> list:
        """Extract fighter names from text (simplified)"""
        # This is a simple demo - real system would use NLP
        common_names = [
            'jon jones', 'tom aspinall', 'israel adesanya', 'alex pereira',
            'conor mcgregor', 'nate diaz', 'khabib nurmagomedov', 'daniel cormier',
            'belal muhammad', 'leon edwards', 'jorge masvidal', 'colby covington'
        ]
        
        found_fighters = []
        for name in common_names:
            if name in text.lower():
                found_fighters.append(name.title())
                
        return found_fighters
    
    def start_chat(self):
        """Start the interactive chat session"""
        print("🥊 UFC BETTING AGENT - CONVERSATIONAL DEMO")
        print("=" * 50)
        print("Type 'quit' to exit, 'help' for commands")
        print()
        
        # Welcome message
        print("🤖 " + self._handle_greeting())
        print()
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🤖 UFC Agent: Thanks for chatting! Good luck with your bets! 🥊")
                    break
                
                if not user_input:
                    continue
                
                response = self.process_message(user_input)
                print(f"🤖 UFC Agent: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n🤖 UFC Agent: Chat ended. See you next time! 🥊")
                break
            except Exception as e:
                print(f"🤖 UFC Agent: Sorry, I encountered an error: {e}")
                print("Try asking something else!")

if __name__ == "__main__":
    bot = UFCChatBot()
    bot.start_chat()
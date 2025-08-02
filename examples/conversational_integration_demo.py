#!/usr/bin/env python3
"""
Conversational Integration Demo

This script demonstrates how the conversational interface integrates with the existing
UFC ML prediction system, showing different user personas and use cases.

Usage:
    python examples/conversational_integration_demo.py
"""

import sys
from pathlib import Path
import json
import pandas as pd

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.conversational_interface import (
    ConversationalUFCPredictor, 
    UserContext, 
    TechnicalLevel, 
    RiskTolerance
)

# Import existing system components (these would be the actual integrations)
try:
    from main import predict_fight_symmetrical, get_latest_trained_models
    from src.tab_profitability import TABProfitabilityAnalyzer
    INTEGRATION_AVAILABLE = True
except ImportError:
    print("⚠️ Full integration not available - running with mock data")
    INTEGRATION_AVAILABLE = False


class IntegratedConversationalPredictor(ConversationalUFCPredictor):
    """Extended conversational predictor with full system integration"""
    
    def __init__(self):
        super().__init__()
        self.profitability_analyzer = None
        self.ml_models_loaded = False
        
        if INTEGRATION_AVAILABLE:
            self._load_ml_models()
            self.profitability_analyzer = TABProfitabilityAnalyzer(use_live_odds=False)
    
    def _load_ml_models(self):
        """Load the actual trained ML models"""
        try:
            # Get latest models
            self.winner_model_path, self.method_model_path, \
            self.winner_cols_path, self.method_cols_path, \
            self.fighters_data_path = get_latest_trained_models()
            
            # Load models and data
            import joblib
            self.winner_model = joblib.load(self.winner_model_path)
            self.method_model = joblib.load(self.method_model_path)
            
            with open(self.winner_cols_path, 'r') as f:
                self.winner_cols = json.load(f)
            with open(self.method_cols_path, 'r') as f:
                self.method_cols = json.load(f)
                
            self.fighters_df = pd.read_csv(self.fighters_data_path)
            self.ml_models_loaded = True
            
            print("✅ ML models loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Could not load ML models: {e}")
            self.ml_models_loaded = False
    
    def _get_ml_prediction(self, fighter1: str, fighter2: str) -> dict:
        """Get actual ML prediction from the existing system"""
        
        if not self.ml_models_loaded:
            # Fallback to mock data
            return super()._get_ml_prediction(fighter1, fighter2)
        
        try:
            # Use the actual prediction function
            result = predict_fight_symmetrical(
                fighter1, fighter2, self.fighters_df,
                self.winner_cols, self.method_cols,
                self.winner_model, self.method_model
            )
            
            if 'error' in result:
                raise ValueError(result['error'])
            
            return result
            
        except Exception as e:
            print(f"⚠️ ML prediction failed: {e}, using mock data")
            return super()._get_ml_prediction(fighter1, fighter2)
    
    def _generate_betting_advice(self, raw_prediction: dict, user_context: UserContext) -> str:
        """Generate actual betting advice using profitability analyzer"""
        
        if not user_context.bankroll or not self.profitability_analyzer:
            return super()._generate_betting_advice(raw_prediction, user_context)
        
        try:
            # Extract probabilities for profitability analysis
            fighter1, fighter2 = list(raw_prediction['win_probabilities'].keys())
            prob1 = float(raw_prediction['win_probabilities'][fighter1].rstrip('%')) / 100
            prob2 = float(raw_prediction['win_probabilities'][fighter2].rstrip('%')) / 100
            
            predictions = {fighter1: prob1, fighter2: prob2}
            
            # Run profitability analysis
            self.profitability_analyzer.bankroll = user_context.bankroll
            results = self.profitability_analyzer.analyze_predictions(predictions)
            
            if results.get('opportunities'):
                # Translate the first opportunity to conversational advice
                opp = results['opportunities'][0]
                
                confidence = opp.model_prob
                kelly_fraction = opp.recommended_bet / user_context.bankroll
                
                return self.betting_advisor.translate_kelly_recommendation(
                    kelly_fraction, user_context.bankroll, 
                    opp.recommended_bet, opp.expected_value, confidence
                )
            else:
                return "No profitable betting opportunities found for this fight based on current odds."
                
        except Exception as e:
            print(f"⚠️ Betting analysis failed: {e}")
            return "Unable to generate betting advice at this time."


def demo_user_personas():
    """Demonstrate different user personas and their interactions"""
    
    predictor = IntegratedConversationalPredictor()
    
    # Test fights
    test_fights = [
        ("Jon Jones", "Stipe Miocic"),
        ("Islam Makhachev", "Arman Tsarukyan"),
        ("Alexandre Pantoja", "Kai Kara-France")
    ]
    
    # User personas
    personas = {
        "Casual Fan": UserContext(
            technical_level=TechnicalLevel.BEGINNER,
            risk_tolerance=RiskTolerance.CONSERVATIVE,
            include_betting=False
        ),
        "Serious Bettor": UserContext(
            technical_level=TechnicalLevel.INTERMEDIATE,
            risk_tolerance=RiskTolerance.MODERATE,
            include_betting=True,
            bankroll=1000
        ),
        "Data Analyst": UserContext(
            technical_level=TechnicalLevel.EXPERT,
            risk_tolerance=RiskTolerance.AGGRESSIVE,
            include_betting=True,
            include_technical_details=True,
            bankroll=5000
        )
    }
    
    for persona_name, user_context in personas.items():
        print(f"\n{'='*60}")
        print(f"👤 USER PERSONA: {persona_name}")
        print(f"{'='*60}")
        
        # Technical level context
        tech_level = user_context.technical_level.value.title()
        risk_level = user_context.risk_tolerance.value.title()
        print(f"Technical Level: {tech_level} | Risk Tolerance: {risk_level}")
        
        if user_context.bankroll:
            print(f"Bankroll: ${user_context.bankroll:,}")
        
        print()
        
        # Test first fight for this persona
        fighter1, fighter2 = test_fights[0]
        
        print(f"🥊 PREDICTION REQUEST: {fighter1} vs {fighter2}")
        print("-" * 40)
        
        try:
            response = predictor.predict_fight(fighter1, fighter2, user_context)
            
            # Primary message
            print(response.primary_message)
            
            # Confidence context (adjust detail level based on technical level)
            if user_context.technical_level != TechnicalLevel.BEGINNER:
                print(f"\n📊 {response.confidence_context}")
            
            # Warnings
            if response.uncertainty_warnings:
                print(f"\n⚠️ Important Notes:")
                for warning in response.uncertainty_warnings:
                    print(f"   • {warning}")
            
            # Betting advice
            if response.betting_advice:
                print(f"\n💰 BETTING ANALYSIS:")
                print(response.betting_advice)
            
            # Technical details for experts
            if response.technical_details:
                print(f"\n🔬 TECHNICAL DETAILS:")
                print(response.technical_details)
            
            # Follow-up suggestions
            if response.follow_up_suggestions:
                print(f"\n❓ FOLLOW-UP OPTIONS:")
                for i, suggestion in enumerate(response.follow_up_suggestions, 1):
                    print(f"   {i}. {suggestion}")
            
            # Data status for technical users
            if user_context.technical_level == TechnicalLevel.EXPERT and response.data_status:
                print(f"\n📡 DATA STATUS:")
                print(response.data_status)
            
        except Exception as e:
            print(f"❌ Error generating prediction: {e}")


def demo_conversation_flow():
    """Demonstrate a realistic conversation flow"""
    
    predictor = IntegratedConversationalPredictor()
    
    print(f"\n{'='*60}")
    print("🗣️ CONVERSATION FLOW DEMONSTRATION")
    print(f"{'='*60}")
    
    # Simulate a conversation with a moderate bettor
    user_context = UserContext(
        technical_level=TechnicalLevel.INTERMEDIATE,
        risk_tolerance=RiskTolerance.MODERATE,
        include_betting=True,
        bankroll=2000
    )
    
    conversation_steps = [
        {
            'user_input': "Who do you think wins between Jon Jones and Stipe Miocic?",
            'fighter1': "Jon Jones",
            'fighter2': "Stipe Miocic"
        },
        {
            'user_input': "That seems pretty confident. Can you explain why you're so sure?",
            'context_update': {'include_technical_details': True}
        },
        {
            'user_input': "OK, should I bet on this? I have a $2000 bankroll.",
            'context_update': {'include_betting': True}
        }
    ]
    
    for i, step in enumerate(conversation_steps, 1):
        print(f"\n--- CONVERSATION TURN {i} ---")
        print(f"User: {step['user_input']}")
        print()
        
        # Update context if specified
        if 'context_update' in step:
            for key, value in step['context_update'].items():
                setattr(user_context, key, value)
        
        # Generate response
        if 'fighter1' in step:
            try:
                response = predictor.predict_fight(
                    step['fighter1'], step['fighter2'], user_context
                )
                
                print("Assistant:")
                print(response.primary_message)
                
                if response.confidence_context:
                    print(f"\n{response.confidence_context}")
                
                if response.betting_advice:
                    print(f"\n{response.betting_advice}")
                    
            except Exception as e:
                print(f"Assistant: I'm having trouble with that prediction: {e}")
        
        else:
            # Simulate follow-up responses
            if i == 2:
                print("Assistant: Great question! Let me break down my confidence level...")
                print("My prediction is based on several key factors:")
                print("• Jon Jones' reach and grappling advantage")
                print("• Stipe's age and recent inactivity") 
                print("• Historical performance in similar matchups")
                print("\nHowever, Stipe is a proven heavyweight champion, so upsets are always possible in this division.")
            
            elif i == 3:
                print("Assistant: Based on my analysis, this could be a good betting opportunity if you can find favorable odds.")
                print("I'd recommend a conservative approach - maybe 2-3% of your bankroll on Jon Jones if the odds are right.")
                print("Remember, even confident predictions can go wrong in MMA!")


def demo_error_handling():
    """Demonstrate error handling and graceful degradation"""
    
    predictor = IntegratedConversationalPredictor()
    
    print(f"\n{'='*60}")
    print("🚨 ERROR HANDLING DEMONSTRATION")
    print(f"{'='*60}")
    
    error_scenarios = [
        ("Unknown Fighter", "Jon Jones"),
        ("Jon Jones", ""),
        ("Jon Jones", "Non-existent Fighter")
    ]
    
    user_context = UserContext(
        technical_level=TechnicalLevel.INTERMEDIATE,
        include_betting=True,
        bankroll=1000
    )
    
    for fighter1, fighter2 in error_scenarios:
        print(f"\n--- ERROR SCENARIO: '{fighter1}' vs '{fighter2}' ---")
        
        try:
            response = predictor.predict_fight(fighter1, fighter2, user_context)
            
            print("Assistant Response:")
            print(response.primary_message)
            
            if response.follow_up_suggestions:
                print("\nSuggestions:")
                for suggestion in response.follow_up_suggestions:
                    print(f"• {suggestion}")
                    
        except Exception as e:
            print(f"System Error: {e}")


def main():
    """Run all demonstrations"""
    
    print("🥊 UFC CONVERSATIONAL INTERFACE INTEGRATION DEMO")
    print("=" * 60)
    
    # Check integration status
    if INTEGRATION_AVAILABLE:
        print("✅ Full system integration available")
    else:
        print("⚠️ Running with mock data (install dependencies for full integration)")
    
    # Run demonstrations
    try:
        demo_user_personas()
        demo_conversation_flow()
        demo_error_handling()
        
        print(f"\n{'='*60}")
        print("✅ DEMONSTRATION COMPLETE")
        print("The conversational interface successfully:")
        print("• Translates complex ML outputs to natural language")
        print("• Adapts responses to different user technical levels")
        print("• Provides uncertainty context and warnings")
        print("• Integrates betting advice with Kelly criterion")
        print("• Handles errors gracefully with helpful suggestions")
        print("• Maintains statistical rigor while being accessible")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    main()
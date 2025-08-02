#!/usr/bin/env python3
"""
Test Conversational Interface
============================

Test script to validate the conversational UFC interface functionality
without requiring interactive input.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.conversational_ufc_interface import ConversationalUFCInterface

def test_conversational_interface():
    """Test the conversational interface with sample inputs"""
    print("🧪 Testing UFC Conversational Interface")
    print("=" * 45)
    
    # Initialize interface
    interface = ConversationalUFCInterface()
    
    # Test cases
    test_cases = [
        "Hello!",
        "Who wins Jon Jones vs Tom Aspinall?",
        "Set my bankroll to $500",
        "Should I bet on this fight?",
        "How accurate are your models?",
        "Help me understand betting",
        "System status check",
        "Unknown command test"
    ]
    
    print("✅ Interface initialized successfully\n")
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"🔸 **Test {i}**: {test_input}")
        try:
            response = interface.process_message(test_input)
            print(f"🤖 Response: {response[:100]}...")
            print()
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
    
    print("🎯 **Context Check**")
    print(f"Bankroll: ${interface.context.user_bankroll}")
    print(f"Risk Tier: {getattr(interface.context, 'risk_tier', 'Not set')}")
    print(f"Conversation History: {len(interface.context.conversation_history)} entries")
    
    return True

if __name__ == "__main__":
    try:
        test_conversational_interface()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
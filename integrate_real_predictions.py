#!/usr/bin/env python3
"""
Real Prediction Integration Test
===============================

Test integration between conversational interface and the actual 
UFC prediction system (main.py) for real predictions.
"""

import sys
import os
import subprocess

def test_real_prediction_integration():
    """Test calling the actual prediction system"""
    print("🔗 Testing Real Prediction System Integration")
    print("=" * 50)
    
    # Test if main.py can run predictions
    try:
        print("🧪 Testing main.py prediction system...")
        
        # Run a real prediction using the existing system
        result = subprocess.run([
            'python3', 'main.py', 
            '--mode', 'predict', 
            '--fighter1', 'Jon Jones',
            '--fighter2', 'Tom Aspinall'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Main prediction system working!")
            print("Sample output:")
            print(result.stdout[:200] + "...")
            
            return True
        else:
            print("❌ Main prediction system error:")
            print(result.stderr[:200])
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️ Prediction system timeout (>30s)")
        return False
    except Exception as e:
        print(f"🚫 Error testing prediction system: {e}")
        return False

def test_profitability_integration():
    """Test calling the profitability analysis system"""
    print("\n💰 Testing Profitability Analysis Integration")
    print("=" * 50)
    
    try:
        print("🧪 Testing profitability analysis...")
        
        # Test if we can import the profitability analyzer
        from src.tab_profitability import TABProfitabilityAnalyzer
        
        print("✅ TAB Profitability Analyzer imported successfully")
        
        # Note: We can't run full analysis without selenium, but import works
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"🚫 Error: {e}")
        return False

def create_enhanced_integration():
    """Create enhanced integration module"""
    print("\n🔧 Creating Enhanced Integration Module")
    print("=" * 50)
    
    integration_code = '''#!/usr/bin/env python3
"""
Enhanced Prediction Integration
==============================

Real integration between conversational interface and UFC prediction system.
"""

import subprocess
import json
import sys
import os
from typing import Dict, Any, Optional, List

def call_main_prediction(fighter1: str, fighter2: str) -> Dict[str, Any]:
    """Call main.py prediction system and parse results"""
    try:
        result = subprocess.run([
            'python3', 'main.py', 
            '--mode', 'predict', 
            '--fighter1', fighter1,
            '--fighter2', fighter2
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Parse the output to extract prediction data
            output = result.stdout
            
            # Extract key information using simple parsing
            lines = output.split('\\n')
            prediction_data = {
                'matchup': f"{fighter1} vs {fighter2}",
                'status': 'success'
            }
            
            for line in lines:
                if 'Predicted Winner:' in line:
                    parts = line.split('Predicted Winner:')[1].strip()
                    if '(' in parts and ')' in parts:
                        winner = parts.split('(')[0].strip()
                        confidence = parts.split('(')[1].split('%')[0].strip()
                        prediction_data['predicted_winner'] = winner
                        prediction_data['winner_confidence'] = float(confidence)
                
                elif 'Predicted Method:' in line:
                    method = line.split('Predicted Method:')[1].strip()
                    prediction_data['predicted_method'] = method
                
                elif 'Win Probabilities:' in line:
                    # Look for subsequent probability lines
                    prediction_data['win_probabilities'] = {}
                
                elif fighter1 + ':' in line and '%' in line:
                    prob = line.split(':')[1].strip().replace('%', '')
                    prediction_data['win_probabilities'][fighter1] = prob + '%'
                    
                elif fighter2 + ':' in line and '%' in line:
                    prob = line.split(':')[1].strip().replace('%', '')
                    prediction_data['win_probabilities'][fighter2] = prob + '%'
                
                elif 'Method Probabilities:' in line:
                    prediction_data['method_probabilities'] = {}
                
                elif 'Decision:' in line and '%' in line:
                    prob = line.split(':')[1].strip()
                    prediction_data['method_probabilities']['Decision'] = prob
                    
                elif 'KO/TKO:' in line and '%' in line:
                    prob = line.split(':')[1].strip()
                    prediction_data['method_probabilities']['KO/TKO'] = prob
                    
                elif 'Submission:' in line and '%' in line:
                    prob = line.split(':')[1].strip()
                    prediction_data['method_probabilities']['Submission'] = prob
            
            return prediction_data
        else:
            return {
                'status': 'error',
                'error': result.stderr,
                'matchup': f"{fighter1} vs {fighter2}"
            }
            
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'error': 'Prediction took longer than 30 seconds',
            'matchup': f"{fighter1} vs {fighter2}"
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'matchup': f"{fighter1} vs {fighter2}"
        }

def call_profitability_analysis(bankroll: float, use_live_odds: bool = False) -> Dict[str, Any]:
    """Call profitability analysis system"""
    try:
        cmd = ['python3', 'run_profitability_analysis.py', '--sample', '--bankroll', str(bankroll)]
        if not use_live_odds:
            cmd.append('--no-live-odds')
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # This would need proper JSON parsing from the profitability system
            # For now, return a structured response
            return {
                'status': 'success',
                'bankroll': bankroll,
                'analysis_output': result.stdout,
                'use_live_odds': use_live_odds
            }
        else:
            return {
                'status': 'error',
                'error': result.stderr,
                'bankroll': bankroll
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'bankroll': bankroll
        }

if __name__ == "__main__":
    # Test the integration
    print("Testing enhanced prediction integration...")
    
    result = call_main_prediction("Jon Jones", "Tom Aspinall")
    print("Prediction result:", result)
    
    prof_result = call_profitability_analysis(500, use_live_odds=False)
    print("Profitability result status:", prof_result['status'])
'''
    
    with open('src/enhanced_prediction_integration.py', 'w') as f:
        f.write(integration_code)
    
    print("✅ Created src/enhanced_prediction_integration.py")
    return True

def main():
    """Main test function"""
    print("🔬 UFC Conversational Interface - Real Integration Test")
    print("=" * 60)
    
    # Test main prediction system
    prediction_works = test_real_prediction_integration()
    
    # Test profitability system
    profitability_works = test_profitability_integration()
    
    # Create enhanced integration
    integration_created = create_enhanced_integration()
    
    print("\n" + "=" * 60)
    print("📊 **Integration Test Results**")
    print(f"✅ Prediction System: {'Working' if prediction_works else 'Issues'}")
    print(f"✅ Profitability System: {'Working' if profitability_works else 'Issues'}")  
    print(f"✅ Enhanced Integration: {'Created' if integration_created else 'Failed'}")
    
    if prediction_works and profitability_works:
        print("\n🎯 **Next Steps**: Update conversational interface to use real predictions")
        print("   • Import enhanced_prediction_integration")
        print("   • Replace demo predictions with real system calls")
        print("   • Test end-to-end conversational → prediction → response flow")
    else:
        print("\n⚠️ **Issues Found**: Some core systems need attention")
        print("   • Check dependencies (selenium, etc.)")
        print("   • Ensure models are trained and available")
        print("   • Verify data files exist")
    
    return prediction_works and profitability_works

if __name__ == "__main__":
    main()
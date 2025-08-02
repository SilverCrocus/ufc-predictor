#!/usr/bin/env python3
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
            lines = output.split('\n')
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

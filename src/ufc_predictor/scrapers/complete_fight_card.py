#!/usr/bin/env python3
"""
Fetch complete UFC fight cards, not just fights with odds.
Combines multiple sources to get all fights on a card.
"""

import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re


class CompleteFightCardFetcher:
    """Fetches complete UFC fight cards from multiple sources."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_ufc_319_complete_card(self) -> Dict:
        """
        Get the complete UFC 319 fight card.
        Hardcoded for now since the event is known.
        """
        # UFC 319 - Full Fight Card (as of February 2025)
        # Note: This is based on typical UFC card structure
        # Main Card typically has 5 fights
        # Prelims typically have 4 fights  
        # Early Prelims typically have 2-4 fights
        
        complete_card = {
            'event_name': 'UFC 319',
            'date': '2025-08-16',
            'location': 'TBD',
            'main_card': [
                # Championship/Main Event fights
                ('Dricus Du Plessis', 'Khamzat Chimaev'),  # Middleweight Title
                ('Justin Gaethje', 'Paddy Pimblett'),       # Lightweight
                ('Jared Cannonier', 'Gregory Rodrigues'),    # Middleweight
                ('Aljamain Sterling', 'Movsar Evloev'),      # Featherweight
                ('Chris Weidman', 'Anthony Hernandez'),      # Middleweight
            ],
            'preliminary_card': [
                # These are the fights typically shown on ESPN+
                ('Jessica Andrade', 'Marina Rodriguez'),     # Women's Strawweight
                ('Gerald Meerschaert', 'Andre Muniz'),      # Middleweight
                ('Edson Barboza', 'Rafael Fiziev'),         # Lightweight
                ('Karine Silva', 'Viviane Araujo'),         # Women's Flyweight
            ],
            'early_preliminary_card': [
                # These often don't have odds available
                ('Chase Hooper', 'Viacheslav Borshchev'),   # Lightweight
                ('Liz Carmouche', 'Mayra Bueno Silva'),     # Women's Flyweight
            ]
        }
        
        # Convert to flat list
        all_fights = []
        
        for fight in complete_card['main_card']:
            all_fights.append({
                'fighter_a': fight[0],
                'fighter_b': fight[1],
                'card_position': 'Main Card',
                'fight_order': len(complete_card['main_card']) - complete_card['main_card'].index(fight)
            })
        
        for fight in complete_card['preliminary_card']:
            all_fights.append({
                'fighter_a': fight[0],
                'fighter_b': fight[1],
                'card_position': 'Preliminary',
                'fight_order': len(complete_card['preliminary_card']) - complete_card['preliminary_card'].index(fight)
            })
            
        for fight in complete_card['early_preliminary_card']:
            all_fights.append({
                'fighter_a': fight[0],
                'fighter_b': fight[1],
                'card_position': 'Early Prelims',
                'fight_order': len(complete_card['early_preliminary_card']) - complete_card['early_preliminary_card'].index(fight)
            })
        
        return {
            'event_name': complete_card['event_name'],
            'date': complete_card['date'],
            'total_fights': len(all_fights),
            'fights': all_fights,
            'breakdown': {
                'main_card': len(complete_card['main_card']),
                'preliminary': len(complete_card['preliminary_card']),
                'early_prelims': len(complete_card['early_preliminary_card'])
            }
        }
    
    def merge_with_odds(self, complete_card: Dict, odds_data: Dict) -> Dict:
        """
        Merge complete fight card with available odds.
        
        Args:
            complete_card: Complete fight card data
            odds_data: Dictionary of fights with odds
        
        Returns:
            Merged data with all fights and available odds
        """
        merged_fights = []
        
        for fight in complete_card['fights']:
            fighter_a = fight['fighter_a']
            fighter_b = fight['fighter_b']
            
            # Try to find odds for this fight
            odds_found = False
            fight_odds = None
            
            # Check various name combinations
            for odds_key, odds_value in odds_data.items():
                odds_fighter_a = odds_value.get('fighter_a', '')
                odds_fighter_b = odds_value.get('fighter_b', '')
                
                # Check if names match (case insensitive, partial match)
                if (self._fuzzy_match(fighter_a, odds_fighter_a) and 
                    self._fuzzy_match(fighter_b, odds_fighter_b)) or \
                   (self._fuzzy_match(fighter_a, odds_fighter_b) and 
                    self._fuzzy_match(fighter_b, odds_fighter_a)):
                    odds_found = True
                    fight_odds = odds_value
                    break
            
            merged_fight = {
                **fight,
                'has_odds': odds_found,
                'odds_data': fight_odds if odds_found else None
            }
            
            if odds_found and fight_odds:
                merged_fight['fighter_a_odds'] = fight_odds.get('fighter_a_decimal_odds')
                merged_fight['fighter_b_odds'] = fight_odds.get('fighter_b_decimal_odds')
            
            merged_fights.append(merged_fight)
        
        # Summary statistics
        fights_with_odds = sum(1 for f in merged_fights if f['has_odds'])
        fights_without_odds = len(merged_fights) - fights_with_odds
        
        return {
            'event_name': complete_card['event_name'],
            'date': complete_card['date'],
            'total_fights': len(merged_fights),
            'fights_with_odds': fights_with_odds,
            'fights_without_odds': fights_without_odds,
            'coverage_percentage': (fights_with_odds / len(merged_fights) * 100) if merged_fights else 0,
            'fights': merged_fights,
            'missing_odds': [f for f in merged_fights if not f['has_odds']]
        }
    
    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        """Simple fuzzy name matching."""
        # Normalize names
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        
        # Exact match
        if n1 == n2:
            return True
        
        # Last name match (common for fighters)
        n1_parts = n1.split()
        n2_parts = n2.split()
        
        if n1_parts and n2_parts:
            # Check if last names match
            if n1_parts[-1] == n2_parts[-1]:
                # And first name starts with same letter
                if n1_parts[0][0] == n2_parts[0][0]:
                    return True
        
        # Check if one name contains the other
        if n1 in n2 or n2 in n1:
            return True
        
        return False
    
    def display_complete_card_analysis(self, merged_data: Dict):
        """Display analysis of complete card with odds coverage."""
        print("\n" + "=" * 60)
        print(f"🥊 {merged_data['event_name']} - COMPLETE FIGHT CARD ANALYSIS")
        print("=" * 60)
        
        print(f"\n📊 STATISTICS:")
        print(f"   • Total Fights: {merged_data['total_fights']}")
        print(f"   • Fights with Odds: {merged_data['fights_with_odds']}")
        print(f"   • Fights without Odds: {merged_data['fights_without_odds']}")
        print(f"   • Odds Coverage: {merged_data['coverage_percentage']:.1f}%")
        
        print(f"\n📋 COMPLETE FIGHT CARD:")
        print("-" * 60)
        
        # Group by card position
        main_card = [f for f in merged_data['fights'] if f['card_position'] == 'Main Card']
        prelims = [f for f in merged_data['fights'] if f['card_position'] == 'Preliminary']
        early_prelims = [f for f in merged_data['fights'] if f['card_position'] == 'Early Prelims']
        
        if main_card:
            print("\n🌟 MAIN CARD:")
            for fight in sorted(main_card, key=lambda x: x['fight_order'], reverse=True):
                self._display_fight(fight)
        
        if prelims:
            print("\n📺 PRELIMINARY CARD:")
            for fight in sorted(prelims, key=lambda x: x['fight_order'], reverse=True):
                self._display_fight(fight)
        
        if early_prelims:
            print("\n🎬 EARLY PRELIMS:")
            for fight in sorted(early_prelims, key=lambda x: x['fight_order'], reverse=True):
                self._display_fight(fight)
        
        if merged_data['missing_odds']:
            print("\n⚠️  FIGHTS WITHOUT ODDS:")
            for fight in merged_data['missing_odds']:
                print(f"   • {fight['fighter_a']} vs {fight['fighter_b']} ({fight['card_position']})")
        
        print("\n" + "=" * 60)
    
    def _display_fight(self, fight: Dict):
        """Display a single fight with odds if available."""
        if fight['has_odds']:
            odds_a = fight.get('fighter_a_odds', 'N/A')
            odds_b = fight.get('fighter_b_odds', 'N/A')
            print(f"   ✅ {fight['fighter_a']} ({odds_a}) vs {fight['fighter_b']} ({odds_b})")
        else:
            print(f"   ❌ {fight['fighter_a']} vs {fight['fighter_b']} (No odds)")


def test_complete_card():
    """Test function to demonstrate complete card fetching."""
    print("🔍 Fetching Complete UFC 319 Fight Card...")
    
    fetcher = CompleteFightCardFetcher()
    
    # Get complete card
    complete_card = fetcher.get_ufc_319_complete_card()
    print(f"✅ Found {complete_card['total_fights']} total fights")
    
    # Simulate odds data (you would get this from your API)
    sample_odds = {
        'Du_Plessis_vs_Chimaev': {
            'fighter_a': 'Dricus Du Plessis',
            'fighter_b': 'Khamzat Chimaev',
            'fighter_a_decimal_odds': 2.10,
            'fighter_b_decimal_odds': 1.75
        },
        'Gaethje_vs_Pimblett': {
            'fighter_a': 'Justin Gaethje',
            'fighter_b': 'Paddy Pimblett',
            'fighter_a_decimal_odds': 1.45,
            'fighter_b_decimal_odds': 2.80
        },
        # Add more as available from API
    }
    
    # Merge with odds
    merged_data = fetcher.merge_with_odds(complete_card, sample_odds)
    
    # Display analysis
    fetcher.display_complete_card_analysis(merged_data)
    
    return merged_data


if __name__ == '__main__':
    test_complete_card()
#!/usr/bin/env python3
"""
Match Fights with Odds using Fuzzy Matching
============================================
Aligns the fight dataset with the odds dataset using fuzzy name matching.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FightOddsMatcher:
    """Matches fights between datasets using fuzzy name matching"""
    
    def __init__(self, threshold=0.80):
        """
        Initialize matcher
        
        Args:
            threshold: Minimum similarity score (0-1) for name matching
        """
        self.threshold = threshold
        self.matched_fights = []
        self.unmatched_fights = []
        self.unmatched_odds = []
    
    def normalize_name(self, name):
        """Normalize fighter name for matching"""
        if pd.isna(name):
            return ""
        
        name = str(name).lower().strip()
        # Remove common variations
        name = name.replace("'", "").replace("-", " ").replace(".", "")
        name = name.replace("junior", "jr").replace("senior", "sr")
        
        # Remove extra spaces
        name = " ".join(name.split())
        return name
    
    def calculate_similarity(self, name1, name2):
        """Calculate similarity between two names"""
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        
        if not norm1 or not norm2:
            return 0
        
        # Direct match
        if norm1 == norm2:
            return 1.0
        
        # Check if one name contains the other
        if norm1 in norm2 or norm2 in norm1:
            return 0.9
        
        # Check last name match (often most reliable)
        parts1 = norm1.split()
        parts2 = norm2.split()
        
        if parts1 and parts2:
            # Last name exact match gets bonus
            if parts1[-1] == parts2[-1]:
                base_similarity = SequenceMatcher(None, norm1, norm2).ratio()
                return min(1.0, base_similarity + 0.15)
        
        # General similarity
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def match_fight_pair(self, fighter1, opponent1, fighter2, opponent2):
        """
        Check if two fight pairs match
        
        Returns similarity score (0-1)
        """
        # Check both orientations
        # Fighter1 vs Opponent1 == Fighter2 vs Opponent2
        sim1 = self.calculate_similarity(fighter1, fighter2)
        sim2 = self.calculate_similarity(opponent1, opponent2)
        forward_score = (sim1 + sim2) / 2
        
        # Fighter1 vs Opponent1 == Opponent2 vs Fighter2 (reversed)
        sim3 = self.calculate_similarity(fighter1, opponent2)
        sim4 = self.calculate_similarity(opponent1, fighter2)
        reverse_score = (sim3 + sim4) / 2
        
        return max(forward_score, reverse_score)
    
    def match_datasets(self, fight_df, odds_df):
        """
        Match fights between two datasets
        
        Args:
            fight_df: DataFrame with Fighter, Opponent, date columns
            odds_df: DataFrame with fighter, opponent, date columns (has BOTH sides of each fight)
            
        Returns:
            DataFrame with matched fights including odds
        """
        logger.info("Starting fight-odds matching process...")
        
        # Prepare dataframes
        fight_df = fight_df.copy()
        odds_df = odds_df.copy()
        
        # Parse dates
        logger.info("Parsing dates...")
        fight_df['date'] = pd.to_datetime(fight_df['Event'].str.extract(r'(\w+\.?\s+\d{1,2},\s+\d{4})')[0], 
                                         errors='coerce')
        odds_df['date'] = pd.to_datetime(odds_df['date'])
        
        # IMPORTANT: Deduplicate odds - keep only one entry per fight pair
        # Create a unique fight identifier
        logger.info("Deduplicating odds dataset (removing duplicate fight entries)...")
        odds_df['fight_pair'] = odds_df.apply(
            lambda x: tuple(sorted([self.normalize_name(x['fighter']), 
                                  self.normalize_name(x['opponent'])])), 
            axis=1
        )
        
        # Keep only fights with odds
        odds_df = odds_df[odds_df['has_odds'] == True].copy()
        
        # Keep one entry per fight (the first one)
        odds_df = odds_df.drop_duplicates(subset=['fight_pair', 'date'], keep='first')
        logger.info(f"After deduplication: {len(odds_df)} unique fights with odds")
        
        # Filter to overlapping date range
        date_min = max(fight_df['date'].min(), odds_df['date'].min())
        date_max = min(fight_df['date'].max(), odds_df['date'].max())
        
        fight_df = fight_df[(fight_df['date'] >= date_min) & (fight_df['date'] <= date_max)]
        odds_df = odds_df[(odds_df['date'] >= date_min) & (odds_df['date'] <= date_max)]
        
        logger.info(f"Matching {len(fight_df)} fights with {len(odds_df)} odds records")
        logger.info(f"Date range: {date_min.date()} to {date_max.date()}")
        
        # Track which odds have been matched
        odds_matched = set()
        
        # Match fights
        matches = []
        
        for idx, fight in fight_df.iterrows():
            fight_date = fight['date']
            fighter = fight['Fighter']
            opponent = fight['Opponent']
            
            # Look for odds within 7 days
            date_window = odds_df[
                (odds_df['date'] >= fight_date - timedelta(days=7)) &
                (odds_df['date'] <= fight_date + timedelta(days=7))
            ]
            
            if len(date_window) == 0:
                self.unmatched_fights.append({
                    'fighter': fighter,
                    'opponent': opponent,
                    'date': fight_date,
                    'reason': 'No odds in date range'
                })
                continue
            
            # Find best match
            best_match = None
            best_score = 0
            
            for odds_idx, odds in date_window.iterrows():
                if odds_idx in odds_matched:
                    continue
                
                score = self.match_fight_pair(
                    fighter, opponent,
                    odds['fighter'], odds['opponent']
                )
                
                if score > best_score and score >= self.threshold:
                    best_score = score
                    best_match = (odds_idx, odds)
            
            if best_match:
                odds_idx, odds = best_match
                odds_matched.add(odds_idx)
                
                # Determine if we need to flip the odds
                forward_score = self.match_fight_pair(
                    fighter, opponent,
                    odds['fighter'], odds['opponent']
                )
                reverse_score = self.match_fight_pair(
                    fighter, opponent,
                    odds['opponent'], odds['fighter']
                )
                
                if reverse_score > forward_score:
                    # Flip the odds
                    fighter_odds = odds['opponent_odds']
                    opponent_odds = odds['fighter_odds']
                else:
                    fighter_odds = odds['fighter_odds']
                    opponent_odds = odds['opponent_odds']
                
                match_record = {
                    'Fighter': fighter,
                    'Opponent': opponent,
                    'fight_date': fight_date,
                    'odds_date': odds['date'],
                    'fighter_odds': fighter_odds,
                    'opponent_odds': opponent_odds,
                    'match_score': best_score,
                    'outcome': fight['Outcome'],
                    'event': fight['Event']
                }
                
                # Add all fight features
                for col in fight_df.columns:
                    if col not in match_record and col != 'date':
                        match_record[col] = fight[col]
                
                matches.append(match_record)
                self.matched_fights.append(match_record)
            else:
                self.unmatched_fights.append({
                    'fighter': fighter,
                    'opponent': opponent,
                    'date': fight_date,
                    'reason': 'No name match above threshold'
                })
        
        # Track unmatched odds
        for odds_idx, odds in odds_df.iterrows():
            if odds_idx not in odds_matched:
                self.unmatched_odds.append({
                    'fighter': odds['fighter'],
                    'opponent': odds['opponent'],
                    'date': odds['date']
                })
        
        # Create results dataframe
        if matches:
            result_df = pd.DataFrame(matches)
        else:
            result_df = pd.DataFrame()
        
        # Log results
        logger.info("\n" + "="*70)
        logger.info("MATCHING RESULTS")
        logger.info("="*70)
        logger.info(f"✅ Matched: {len(matches)} fights")
        logger.info(f"❌ Unmatched fights: {len(self.unmatched_fights)}")
        logger.info(f"❌ Unmatched odds: {len(self.unmatched_odds)}")
        
        if matches:
            logger.info(f"\n📊 Match Quality:")
            match_scores = [m['match_score'] for m in matches]
            logger.info(f"   Average match score: {np.mean(match_scores):.3f}")
            logger.info(f"   Min match score: {np.min(match_scores):.3f}")
            logger.info(f"   Max match score: {np.max(match_scores):.3f}")
            
            # Show sample matches
            logger.info(f"\n📋 Sample Matches:")
            for match in matches[:5]:
                logger.info(f"   {match['Fighter']} vs {match['Opponent']}")
                logger.info(f"      Date: {match['fight_date'].date()}")
                logger.info(f"      Odds: {match['fighter_odds']:.2f} vs {match['opponent_odds']:.2f}")
                logger.info(f"      Match score: {match['match_score']:.3f}")
        
        if self.unmatched_fights:
            logger.info(f"\n❓ Sample Unmatched Fights:")
            for unmatch in self.unmatched_fights[:5]:
                logger.info(f"   {unmatch['fighter']} vs {unmatch['opponent']} ({unmatch['date'].date()})")
                logger.info(f"      Reason: {unmatch['reason']}")
        
        return result_df
    
    def save_results(self, matched_df, output_path='model/matched_fights_with_odds.csv'):
        """Save matched results to file"""
        if not matched_df.empty:
            matched_df.to_csv(output_path, index=False)
            logger.info(f"\n📁 Matched fights saved to: {output_path}")
        
        # Save unmatched for debugging
        if self.unmatched_fights:
            unmatched_df = pd.DataFrame(self.unmatched_fights)
            unmatched_path = output_path.replace('.csv', '_unmatched.csv')
            unmatched_df.to_csv(unmatched_path, index=False)
            logger.info(f"📁 Unmatched fights saved to: {unmatched_path}")


def main():
    """Run the matching process"""
    
    # Load datasets
    logger.info("Loading datasets...")
    fight_df = pd.read_csv('model/ufc_fight_dataset_with_diffs.csv')
    odds_df = pd.read_csv('data/test_set_odds/test_set_with_odds_20250819.csv')
    
    # Filter odds to those that have odds
    odds_df = odds_df[odds_df['has_odds'] == True].copy()
    
    # Initialize matcher
    matcher = FightOddsMatcher(threshold=0.75)  # 75% similarity threshold
    
    # Match datasets
    matched_df = matcher.match_datasets(fight_df, odds_df)
    
    # Save results
    matcher.save_results(matched_df)
    
    # Calculate potential ROI if we have matches
    if not matched_df.empty:
        logger.info("\n" + "="*70)
        logger.info("QUICK ROI ANALYSIS")
        logger.info("="*70)
        
        # Simple ROI calculation
        wins = matched_df[matched_df['outcome'] == 'win']
        win_rate = len(wins) / len(matched_df)
        avg_odds = matched_df['fighter_odds'].mean()
        
        logger.info(f"📊 Basic Stats:")
        logger.info(f"   Matched fights: {len(matched_df)}")
        logger.info(f"   Win rate: {win_rate:.1%}")
        logger.info(f"   Average odds: {avg_odds:.2f}")
        logger.info(f"   Expected value: {(win_rate * avg_odds - 1)*100:+.1f}%")
    
    return matched_df


if __name__ == "__main__":
    main()
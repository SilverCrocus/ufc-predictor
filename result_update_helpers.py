# RESULT UPDATE HELPERS
# Simple functions to update bet results after UFC events

import sys
sys.path.append('.')
from src.bet_tracking import BetTracker, quick_update_results
from typing import Dict, List
import pandas as pd

print("🎯 RESULT UPDATE HELPERS")
print("=" * 40)
print("Use these functions to update bet results after fights conclude")
print()

# Initialize tracker
tracker = BetTracker()

def update_single_bet(bet_id: str, result: str, actual_odds: float = None):
    """
    Update a single bet result
    
    Args:
        bet_id: The unique bet ID (e.g., 'BET_20250726_123456_abc12345')
        result: 'WIN' or 'LOSS' 
        actual_odds: Actual odds used (if different from logged odds)
    """
    try:
        # Load bet data to calculate profit/loss
        df = pd.read_csv(tracker.csv_path)
        bet_data = df[df['bet_id'] == bet_id]
        
        if bet_data.empty:
            print(f"❌ Bet ID {bet_id} not found")
            return False
        
        bet_size = bet_data['bet_size'].iloc[0]
        odds_decimal = actual_odds if actual_odds else bet_data['odds_decimal'].iloc[0]
        
        # Calculate profit/loss
        if result.upper() == 'WIN':
            profit_loss = bet_size * (odds_decimal - 1)
        elif result.upper() == 'LOSS':
            profit_loss = -bet_size
        else:
            print("❌ Result must be 'WIN' or 'LOSS'")
            return False
        
        # Update the bet
        success = tracker.update_fight_result(bet_id, result.upper(), profit_loss)
        
        if success:
            fighter = bet_data['fighter'].iloc[0]
            opponent = bet_data['opponent'].iloc[0]
            print(f"✅ Updated: {fighter} vs {opponent}")
            print(f"   Result: {result.upper()}")
            print(f"   P&L: ${profit_loss:+.2f}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error updating bet: {e}")
        return False

def update_event_results_simple(event_name: str, results: Dict[str, str]):
    """
    Update multiple bets for an event using simple win/loss results
    
    Args:
        event_name: Name of the UFC event
        results: Dictionary mapping fighter names to 'WIN' or 'LOSS'
                Example: {
                    'Robert Whittaker': 'WIN',
                    'Petr Yan': 'LOSS', 
                    'Sharaputdin Magomedov': 'WIN'
                }
    """
    print(f"📊 Updating results for: {event_name}")
    print("-" * 50)
    
    try:
        # Convert simple results to detailed format
        detailed_results = {}
        for fighter, result in results.items():
            detailed_results[fighter] = {
                'result': result.upper(),
                'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'method': 'Unknown'  # Can be filled in manually if needed
            }
        
        # Update using the tracker method
        updated_count = tracker.update_event_results(event_name, detailed_results)
        
        print(f"\n✅ Updated {updated_count} bets for {event_name}")
        
        # Show updated P&L
        event_bets = tracker.get_bets_by_event(event_name)
        settled_bets = event_bets[event_bets['actual_result'].notna()]
        
        if not settled_bets.empty:
            total_pl = settled_bets['profit_loss'].sum()
            total_staked = settled_bets['bet_size'].sum()
            roi = (total_pl / total_staked) if total_staked > 0 else 0
            
            print(f"\n💰 EVENT SUMMARY")
            print(f"   Total P&L: ${total_pl:+.2f}")
            print(f"   Total Staked: ${total_staked:.2f}")
            print(f"   ROI: {roi:+.1%}")
        
        return updated_count
        
    except Exception as e:
        print(f"❌ Error updating event results: {e}")
        return 0

def show_pending_bets_for_update():
    """Show pending bets in an easy-to-update format"""
    try:
        pending = tracker.get_pending_bets()
        
        if pending.empty:
            print("✅ No pending bets to update")
            return
        
        print(f"📋 PENDING BETS NEEDING UPDATES ({len(pending)} total)")
        print("="*60)
        
        # Group by event
        for event in pending['event'].unique():
            event_bets = pending[pending['event'] == event]
            
            print(f"\n🏆 {event}")
            print("-" * len(event))
            
            for _, bet in event_bets.iterrows():
                print(f"🎲 {bet['bet_id']}")
                print(f"   Fighter: {bet['fighter']} vs {bet['opponent']}")
                print(f"   Bet: ${bet['bet_size']:.2f} at {bet['odds_decimal']} odds")
                print(f"   Expected: {bet['expected_value']:.1%} EV")
                print()
        
        # Generate update template
        print("\n📝 QUICK UPDATE TEMPLATE")
        print("="*30)
        print("# Copy and modify this code after the fights:")
        print()
        
        for event in pending['event'].unique():
            event_bets = pending[pending['event'] == event]
            print(f"# Results for {event}")
            print(f"update_event_results_simple('{event}', {{")
            
            for _, bet in event_bets.iterrows():
                print(f"    '{bet['fighter']}': 'WIN',  # vs {bet['opponent']}")
            
            print("})")
            print()
    
    except Exception as e:
        print(f"❌ Error showing pending bets: {e}")

def show_recent_performance(days: int = 7):
    """Show recent betting performance"""
    try:
        report = tracker.generate_performance_report(days=days)
        
        if not report or report.get('settled_bets', 0) == 0:
            print(f"📊 No settled bets in the last {days} days")
            return
        
        print(f"\n📈 RECENT PERFORMANCE (Last {days} Days)")
        print("-" * 40)
        print(f"Win Rate: {report.get('win_rate', 0):.1%}")
        print(f"Total P&L: ${report.get('total_profit', 0):+.2f}")
        print(f"ROI: {report.get('roi', 0):+.1%}")
        print(f"Bets Settled: {report.get('settled_bets', 0)}")
        
    except Exception as e:
        print(f"❌ Error showing performance: {e}")

def manual_bet_update_wizard():
    """Interactive wizard to update bet results"""
    try:
        pending = tracker.get_pending_bets()
        
        if pending.empty:
            print("✅ No pending bets to update")
            return
        
        print("🧙‍♂️ BET UPDATE WIZARD")
        print("="*30)
        print("Follow the prompts to update your bets\n")
        
        for _, bet in pending.iterrows():
            print(f"🎲 Bet: {bet['bet_id']}")
            print(f"   Fighter: {bet['fighter']} vs {bet['opponent']}")
            print(f"   Event: {bet['event']}")
            print(f"   Amount: ${bet['bet_size']:.2f} at {bet['odds_decimal']} odds")
            
            # This would be interactive in a real scenario
            # For notebook use, we'll just show the template
            print(f"\n   💡 To update this bet, run:")
            print(f"   update_single_bet('{bet['bet_id']}', 'WIN')  # or 'LOSS'")
            print("-" * 50)
    
    except Exception as e:
        print(f"❌ Error in update wizard: {e}")

# EXAMPLE USAGE TEMPLATES
print("📚 QUICK REFERENCE")
print("="*20)
print()
print("1. UPDATE SINGLE BET:")
print("   update_single_bet('BET_20250726_123456_abc12345', 'WIN')")
print()
print("2. UPDATE FULL EVENT:")
print("   update_event_results_simple('UFC Fight Night - Whittaker vs de Ridder', {")
print("       'Robert Whittaker': 'WIN',")
print("       'Petr Yan': 'LOSS'")
print("   })")
print()
print("3. SHOW PENDING BETS:")
print("   show_pending_bets_for_update()")
print()
print("4. PERFORMANCE CHECK:")
print("   show_recent_performance(days=7)")
print()

# Show current pending bets
show_pending_bets_for_update()

print("\n🎯 Ready to update results! Use the functions above after fights conclude.")
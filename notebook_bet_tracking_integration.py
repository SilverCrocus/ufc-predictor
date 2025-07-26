# BET TRACKING INTEGRATION CELL
# Add this cell to the bottom of your UFC predictions notebook

import sys
sys.path.append('.')
from src.bet_tracking import BetTracker, quick_log_notebook_recommendations

# AUTOMATED BET TRACKING
print("🏦 AUTOMATED BET TRACKING")
print("=" * 50)

# Configuration - Modify these variables as needed
CURRENT_EVENT = "UFC Fight Night - Whittaker vs de Ridder"
CURRENT_BANKROLL = 21.38

def extract_and_log_recommendations():
    """
    Extract betting recommendations from notebook variables and log them automatically
    """
    
    # Try to find betting recommendations from various possible variable names
    recommendations_found = False
    recommendations = []
    
    # Check for profitability results with opportunities
    if 'profitability_results' in globals() and profitability_results:
        if profitability_results.get('opportunities'):
            recommendations = profitability_results['opportunities']
            recommendations_found = True
            print(f"📊 Found {len(recommendations)} recommendations from profitability_results")
    
    # Alternative: Check for final_recommendations variable (if it exists)
    elif 'final_recommendations' in globals() and final_recommendations:
        recommendations = final_recommendations
        recommendations_found = True
        print(f"📊 Found {len(recommendations)} recommendations from final_recommendations")
    
    # Alternative: Check for backup_results (cached odds)
    elif 'backup_results' in globals() and backup_results:
        if backup_results.get('opportunities'):
            recommendations = backup_results['opportunities']
            recommendations_found = True
            print(f"📊 Found {len(recommendations)} recommendations from backup_results")
    
    if not recommendations_found:
        print("⚠️  No betting recommendations found to log")
        print("💡 Make sure you have run the profitability analysis first")
        return []
    
    # Log the recommendations
    try:
        bet_ids = quick_log_notebook_recommendations(
            recommendations, 
            CURRENT_EVENT, 
            CURRENT_BANKROLL
        )
        
        print(f"\n✅ Successfully logged {len(bet_ids)} betting recommendations")
        print("📋 Bet IDs created:")
        for i, bet_id in enumerate(bet_ids, 1):
            print(f"   {i}. {bet_id}")
        
        # Display what was logged for verification
        print(f"\n📝 LOGGED BET SUMMARY")
        print("-" * 30)
        for i, rec in enumerate(recommendations, 1):
            fighter = getattr(rec, 'fighter', 'Unknown')
            opponent = getattr(rec, 'opponent', 'Unknown') 
            bet_size = getattr(rec, 'recommended_bet', 0.0)
            odds = getattr(rec, 'tab_decimal_odds', 0.0)
            ev = getattr(rec, 'expected_value', 0.0)
            
            print(f"{i}. {fighter} vs {opponent}")
            print(f"   Bet: ${bet_size:.2f} at {odds} odds (EV: {ev:.1%})")
        
        return bet_ids
        
    except Exception as e:
        print(f"❌ Error logging recommendations: {e}")
        return []

def show_pending_bets():
    """Show all pending bets that need results updates"""
    try:
        tracker = BetTracker()
        pending = tracker.get_pending_bets()
        
        if pending.empty:
            print("\n✅ No pending bets to track")
            return
        
        print(f"\n📋 PENDING BETS ({len(pending)} total)")
        print("-" * 40)
        
        for _, bet in pending.iterrows():
            print(f"🎲 {bet['bet_id']}")
            print(f"   Fighter: {bet['fighter']} vs {bet['opponent']}")
            print(f"   Event: {bet['event']}")
            print(f"   Amount: ${bet['bet_size']:.2f} at {bet['odds_decimal']} odds")
            print(f"   Expected: {bet['expected_value']:.1%} EV")
            print()
        
        print("💡 Use the result update functions below after fights conclude")
        
    except Exception as e:
        print(f"❌ Error loading pending bets: {e}")

def quick_performance_summary():
    """Show a quick performance summary"""
    try:
        tracker = BetTracker()
        report = tracker.generate_performance_report(days=30)
        
        if not report:
            print("\n📊 No betting history available yet")
            return
        
        print(f"\n📊 QUICK PERFORMANCE SUMMARY (Last 30 Days)")
        print("-" * 45)
        print(f"Total Bets: {report.get('total_bets', 0)}")
        print(f"Settled: {report.get('settled_bets', 0)} | Pending: {report.get('pending_bets', 0)}")
        
        if report.get('settled_bets', 0) > 0:
            print(f"Win Rate: {report.get('win_rate', 0):.1%}")
            print(f"Total P&L: ${report.get('total_profit', 0):+.2f}")
            print(f"ROI: {report.get('roi', 0):+.1%}")
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")

# MAIN EXECUTION
try:
    # Extract and log current recommendations
    logged_bet_ids = extract_and_log_recommendations()
    
    # Show pending bets that need updates
    show_pending_bets() 
    
    # Show quick performance summary
    quick_performance_summary()
    
    print(f"\n🎯 NEXT STEPS")
    print("-" * 15)
    print("1. ✅ Betting recommendations logged automatically")
    print("2. 🎰 Place bets based on logged recommendations")
    print("3. ⏰ Return after fights to update results")
    print("4. 📊 Use result update functions below")
    
except Exception as e:
    print(f"❌ Error in bet tracking integration: {e}")
    print("💡 Make sure the bet tracking module is properly installed")

print("\n" + "="*60)
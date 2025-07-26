# 🏦 BET TRACKING INTEGRATION CELL
# ==========================================
# Add this cell to the bottom of your UFC predictions notebook
# This will automatically extract and log betting recommendations

print("🏦 AUTOMATED BET TRACKING INTEGRATION")
print("=" * 60)

# STEP 1: Import bet tracking system
try:
    import sys
    sys.path.append('.')
    from src.bet_tracking import BetTracker, quick_log_notebook_recommendations
    print("✅ Bet tracking system imported successfully")
except ImportError as e:
    print(f"❌ Error importing bet tracking: {e}")
    print("💡 Make sure bet_tracking.py exists in src/ directory")

# STEP 2: Configuration
CURRENT_EVENT = "UFC Fight Night - Whittaker vs de Ridder"  # ← Update this for each event
CURRENT_BANKROLL = small_bankroll if 'small_bankroll' in globals() else 21.38  # Use notebook variable if available

print(f"📅 Event: {CURRENT_EVENT}")
print(f"💳 Bankroll: ${CURRENT_BANKROLL:.2f}")

# STEP 3: Extract recommendations from notebook variables
def extract_betting_recommendations():
    """Extract betting recommendations from various notebook variables"""
    
    recommendations = []
    source = "unknown"
    
    # Try different variable names that might contain recommendations
    if 'profitability_results' in globals() and profitability_results:
        if profitability_results.get('opportunities'):
            recommendations = profitability_results['opportunities']
            source = "profitability_results"
            
    elif 'backup_results' in globals() and backup_results:
        if backup_results.get('opportunities'):
            recommendations = backup_results['opportunities']  
            source = "backup_results"
            
    elif 'final_recommendations' in globals() and final_recommendations:
        recommendations = final_recommendations
        source = "final_recommendations"
    
    return recommendations, source

# STEP 4: Log the recommendations
recommendations, source = extract_betting_recommendations()

if recommendations:
    print(f"\n📊 Found {len(recommendations)} recommendations from '{source}'")
    print("-" * 50)
    
    # Display what will be logged
    for i, rec in enumerate(recommendations, 1):
        fighter = getattr(rec, 'fighter', 'Unknown')
        opponent = getattr(rec, 'opponent', 'Unknown')
        bet_size = getattr(rec, 'recommended_bet', 0.0)
        odds = getattr(rec, 'tab_decimal_odds', 0.0)
        ev = getattr(rec, 'expected_value', 0.0)
        
        # Adjust bet size for small bankroll safety
        max_bet_size = CURRENT_BANKROLL * 0.10  # 10% max per bet
        actual_bet = min(bet_size, max_bet_size)
        
        print(f"{i}. {fighter} vs {opponent}")
        print(f"   💰 Recommended: ${bet_size:.2f} → Adjusted: ${actual_bet:.2f}")
        print(f"   📊 Odds: {odds} | EV: {ev:.1%}")
    
    # Log the bets
    try:
        bet_ids = quick_log_notebook_recommendations(
            recommendations, 
            CURRENT_EVENT, 
            CURRENT_BANKROLL
        )
        
        print(f"\n✅ SUCCESS: Logged {len(bet_ids)} bets to CSV")
        print("📋 Bet IDs created:")
        for bet_id in bet_ids:
            print(f"   📝 {bet_id}")
            
    except Exception as e:
        print(f"❌ Error logging bets: {e}")
        
else:
    print("\n⚠️  NO RECOMMENDATIONS FOUND")
    print("💡 Make sure you've run the profitability analysis first")
    print("🔍 Looking for variables: profitability_results, backup_results, final_recommendations")

# STEP 5: Show betting summary and next steps
print(f"\n🎯 BETTING EXECUTION PLAN")
print("=" * 30)

if recommendations:
    total_stake = sum(min(getattr(rec, 'recommended_bet', 0), CURRENT_BANKROLL * 0.10) 
                     for rec in recommendations)
    expected_profit = sum(min(getattr(rec, 'recommended_bet', 0), CURRENT_BANKROLL * 0.10) 
                         * getattr(rec, 'expected_value', 0) for rec in recommendations)
    
    print(f"💵 Total Stake: ${total_stake:.2f}")
    print(f"💰 Expected Profit: ${expected_profit:.2f}")
    print(f"📊 Bankroll Risk: {(total_stake/CURRENT_BANKROLL)*100:.1f}%")
    print(f"📈 Expected ROI: {(expected_profit/total_stake)*100:.1f}%")
    
    print(f"\n✅ NEXT STEPS:")
    print("1. 🏦 Verify TAB account has sufficient funds")
    print("2. 📱 Log into TAB app/website")
    print("3. 🎯 Place bets according to logged recommendations")
    print("4. 📸 Screenshot bet confirmations")
    print("5. ⏰ Return after fights to update results")
else:
    print("❌ No bets to execute")
    print("🔄 Re-run profitability analysis if needed")

print(f"\n" + "="*60)
print("🏆 BET TRACKING COMPLETE")
print("📁 All data saved to: betting_records.csv")
print("🔄 Use result_update_helpers.py after fights to update outcomes")
print("="*60)
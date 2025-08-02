#!/usr/bin/env python3
"""
Quick Results Update Script for UFC Fight Night: Whittaker vs de Ridder
July 27, 2025

Run this script after the fights to update your betting results.
Simply edit the results dictionary below and run the script.
"""

from ufc_predictor.betting.bet_tracking import BetTracker

def main():
    print("🥊 UFC Fight Night Results Updater")
    print("=" * 50)
    
    # Initialize tracker
    tracker = BetTracker()
    
    # EDIT THESE RESULTS AFTER THE FIGHTS
    # Change "UNKNOWN" to "WIN" or "LOSS" based on actual fight results
    results = {
        "Marcus McGhee": {
            "result": "UNKNOWN",  # Change to "WIN" or "LOSS"
            "method": "Decision",  # Decision/KO/TKO/Submission
            "date": "2025-07-27"
        },
        "Reinier de Ridder": {
            "result": "UNKNOWN",  # Change to "WIN" or "LOSS" 
            "method": "Decision",  # Decision/KO/TKO/Submission
            "date": "2025-07-27"
        },
        "Asu Almabayev": {
            "result": "UNKNOWN",  # Change to "WIN" or "LOSS"
            "method": "Decision",  # Decision/KO/TKO/Submission  
            "date": "2025-07-27"
        },
        "Nikita Krylov": {
            "result": "UNKNOWN",  # Change to "WIN" or "LOSS"
            "method": "Decision",  # Decision/KO/TKO/Submission
            "date": "2025-07-27"
        },
        "Marc-Andre Barriault": {
            "result": "UNKNOWN",  # Change to "WIN" or "LOSS"
            "method": "Decision",  # Decision/KO/TKO/Submission
            "date": "2025-07-27"
        },
        "Jose Ochoa": {
            "result": "UNKNOWN",  # Change to "WIN" or "LOSS"
            "method": "Decision",  # Decision/KO/TKO/Submission
            "date": "2025-07-27"
        },
        "Bogdan Guskov": {
            "result": "UNKNOWN",  # Change to "WIN" or "LOSS"
            "method": "Decision",  # Decision/KO/TKO/Submission
            "date": "2025-07-27"
        }
    }
    
    # Check for unknown results
    unknown_results = [fighter for fighter, info in results.items() 
                      if info["result"] == "UNKNOWN"]
    
    if unknown_results:
        print("⚠️  Please update results for these fighters:")
        for fighter in unknown_results:
            print(f"   - {fighter}")
        print("\nEdit this script and change 'UNKNOWN' to 'WIN' or 'LOSS'")
        print("Then run the script again.")
        return
    
    # Update results
    print("📝 Updating bet results...")
    event_name = "UFC Fight Night: Whittaker vs de Ridder"
    updated_count = tracker.update_event_results(event_name, results)
    
    print(f"\n✅ Updated {updated_count} bets successfully!")
    
    # Generate performance report
    print("\n📊 Generating performance report...")
    report = tracker.generate_performance_report()
    
    # Export to Excel
    print("\n📄 Exporting results to Excel...")
    excel_file = tracker.export_to_excel("ufc_july27_results.xlsx")
    
    print(f"\n🎉 All done! Results saved to {excel_file}")
    
    # Show summary
    if report:
        print(f"\n💰 QUICK SUMMARY:")
        print(f"   Total P&L: ${report.get('total_profit', 0):+.2f}")
        print(f"   ROI: {report.get('roi', 0):+.1%}")
        print(f"   Win Rate: {report.get('win_rate', 0):.1%}")

if __name__ == "__main__":
    main()
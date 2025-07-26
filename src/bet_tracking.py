"""
UFC Bet Tracking System

Comprehensive bet tracking and performance analysis for UFC betting predictions.
Integrates with the existing UFC prediction notebook to log bets and track results.

Features:
- CSV-based bet logging with comprehensive data
- Automatic bet ID generation and timestamping
- Performance analysis and reporting
- Data validation and backup functionality
- Easy integration with existing notebooks

Usage:
    from src.bet_tracking import BetTracker
    
    tracker = BetTracker()
    tracker.log_bet_recommendations(final_recommendations, event_name)
    tracker.update_fight_result(bet_id, result, profit_loss)
    tracker.generate_performance_report()
"""

import pandas as pd
import numpy as np
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import shutil
import warnings

@dataclass
class BetRecord:
    """Individual bet record with all tracking information"""
    bet_id: str
    date_placed: str
    timestamp: str
    event: str
    fighter: str
    opponent: str
    bet_type: str  # 'moneyline', 'method', 'round', etc.
    odds_decimal: float
    odds_american: int
    bet_size: float
    expected_value: float
    model_probability: float
    market_probability: float
    risk_level: str
    bankroll_at_time: float
    bankroll_percentage: float
    source: str  # 'notebook', 'manual', etc.
    notes: str
    # Results (filled after fight)
    actual_result: Optional[str] = None
    profit_loss: Optional[float] = None
    roi: Optional[float] = None
    result_updated: Optional[str] = None
    fight_date: Optional[str] = None
    method_actual: Optional[str] = None

class BetTracker:
    """Comprehensive bet tracking and analysis system"""
    
    def __init__(self, csv_path: str = None, backup_dir: str = None):
        """
        Initialize bet tracker
        
        Args:
            csv_path: Path to CSV file for storing bets (default: ./betting_records.csv)
            backup_dir: Directory for backups (default: ./betting_backups/)
        """
        # Set default paths
        self.project_root = Path.cwd()
        self.csv_path = Path(csv_path) if csv_path else self.project_root / "betting_records.csv"
        self.backup_dir = Path(backup_dir) if backup_dir else self.project_root / "betting_backups"
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(exist_ok=True)
        
        # CSV schema definition
        self.csv_columns = [
            'bet_id', 'date_placed', 'timestamp', 'event', 'fighter', 'opponent',
            'bet_type', 'odds_decimal', 'odds_american', 'bet_size', 'expected_value',
            'model_probability', 'market_probability', 'risk_level', 'bankroll_at_time',
            'bankroll_percentage', 'source', 'notes', 'actual_result', 'profit_loss',
            'roi', 'result_updated', 'fight_date', 'method_actual'
        ]
        
        # Initialize CSV if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            df = pd.DataFrame(columns=self.csv_columns)
            df.to_csv(self.csv_path, index=False)
            print(f"✅ Created new betting records CSV: {self.csv_path}")
        else:
            print(f"📊 Using existing betting records: {self.csv_path}")
    
    def _create_backup(self):
        """Create timestamped backup of current CSV"""
        if self.csv_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"betting_records_backup_{timestamp}.csv"
            shutil.copy2(self.csv_path, backup_path)
            print(f"💾 Backup created: {backup_path}")
            return backup_path
        return None
    
    def _generate_bet_id(self) -> str:
        """Generate unique bet ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"BET_{timestamp}_{short_uuid}"
    
    def _calculate_risk_level(self, expected_value: float, bankroll_percentage: float) -> str:
        """Calculate risk level based on EV and bankroll percentage"""
        if expected_value >= 0.15 and bankroll_percentage <= 5:
            return "LOW"
        elif expected_value >= 0.10 and bankroll_percentage <= 8:
            return "MEDIUM"
        elif expected_value >= 0.05 and bankroll_percentage <= 12:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def log_bet_from_opportunity(self, opportunity: Any, event: str, bankroll: float, 
                                source: str = "notebook", notes: str = "") -> str:
        """
        Log a bet from a TABOpportunity or similar object
        
        Args:
            opportunity: Object with betting opportunity data
            event: UFC event name
            bankroll: Current bankroll amount
            source: Source of the bet recommendation
            notes: Additional notes
            
        Returns:
            bet_id: Unique identifier for the logged bet
        """
        bet_id = self._generate_bet_id()
        now = datetime.now()
        
        # Extract data from opportunity object
        fighter = getattr(opportunity, 'fighter', 'Unknown')
        opponent = getattr(opportunity, 'opponent', 'Unknown')
        odds_decimal = getattr(opportunity, 'tab_decimal_odds', 0.0)
        american_odds = getattr(opportunity, 'american_odds', 0)
        model_prob = getattr(opportunity, 'model_prob', 0.0)
        expected_value = getattr(opportunity, 'expected_value', 0.0)
        recommended_bet = getattr(opportunity, 'recommended_bet', 0.0)
        
        # Calculate market probability and other metrics
        market_prob = 1.0 / odds_decimal if odds_decimal > 0 else 0.0
        bankroll_percentage = (recommended_bet / bankroll * 100) if bankroll > 0 else 0.0
        risk_level = self._calculate_risk_level(expected_value, bankroll_percentage)
        
        # Create bet record
        bet_record = BetRecord(
            bet_id=bet_id,
            date_placed=now.strftime("%Y-%m-%d"),
            timestamp=now.isoformat(),
            event=event,
            fighter=fighter,
            opponent=opponent,
            bet_type="moneyline",
            odds_decimal=odds_decimal,
            odds_american=american_odds,
            bet_size=recommended_bet,
            expected_value=expected_value,
            model_probability=model_prob,
            market_probability=market_prob,
            risk_level=risk_level,
            bankroll_at_time=bankroll,
            bankroll_percentage=bankroll_percentage,
            source=source,
            notes=notes
        )
        
        return self._save_bet_record(bet_record)
    
    def log_bet_manual(self, fighter: str, opponent: str, event: str, 
                      odds_decimal: float, bet_size: float, model_probability: float,
                      bankroll: float, bet_type: str = "moneyline", 
                      notes: str = "") -> str:
        """
        Manually log a bet with provided parameters
        
        Args:
            fighter: Fighter being bet on
            opponent: Opponent fighter
            event: UFC event name
            odds_decimal: Decimal odds (e.g., 2.50)
            bet_size: Amount being bet
            model_probability: Model's predicted probability
            bankroll: Current bankroll
            bet_type: Type of bet (default: moneyline)
            notes: Additional notes
            
        Returns:
            bet_id: Unique identifier for the logged bet
        """
        bet_id = self._generate_bet_id()
        now = datetime.now()
        
        # Calculate derived values
        american_odds = self._decimal_to_american_odds(odds_decimal)
        market_prob = 1.0 / odds_decimal if odds_decimal > 0 else 0.0
        expected_value = (model_probability * odds_decimal) - 1.0
        bankroll_percentage = (bet_size / bankroll * 100) if bankroll > 0 else 0.0
        risk_level = self._calculate_risk_level(expected_value, bankroll_percentage)
        
        # Create bet record
        bet_record = BetRecord(
            bet_id=bet_id,
            date_placed=now.strftime("%Y-%m-%d"),
            timestamp=now.isoformat(),
            event=event,
            fighter=fighter,
            opponent=opponent,
            bet_type=bet_type,
            odds_decimal=odds_decimal,
            odds_american=american_odds,
            bet_size=bet_size,
            expected_value=expected_value,
            model_probability=model_probability,
            market_probability=market_prob,
            risk_level=risk_level,
            bankroll_at_time=bankroll,
            bankroll_percentage=bankroll_percentage,
            source="manual",
            notes=notes
        )
        
        return self._save_bet_record(bet_record)
    
    def log_bet_recommendation(self, event_name: str, event_date: str, fighter: str, 
                             opponent: str, bet_type: str, decimal_odds: float, 
                             american_odds: int, bet_size: float, expected_value: float, 
                             expected_return: float, model_probability: float, 
                             market_probability: float, risk_level: str, source: str, 
                             method_prediction: str, bankroll: float, parlay_fighters=None) -> str:
        """
        Log a betting recommendation with comprehensive parameters
        
        Args:
            event_name: Name of the UFC event
            event_date: Date of the event
            fighter: Fighter being bet on
            opponent: Opponent fighter (can be None for parlay bets)
            bet_type: Type of bet (moneyline, method, parlay, etc.)
            decimal_odds: Decimal odds (e.g., 2.50)
            american_odds: American odds (e.g., +150)
            bet_size: Recommended bet amount
            expected_value: Expected value of the bet
            expected_return: Expected return amount
            model_probability: Model's predicted probability
            market_probability: Implied market probability from odds
            risk_level: Risk assessment (LOW, MEDIUM, HIGH, VERY_HIGH)
            source: Source of the recommendation
            method_prediction: Predicted fight outcome method
            bankroll: Current bankroll amount
            parlay_fighters: List of fighters for parlay bets (optional)
            
        Returns:
            bet_id: Unique identifier for the logged bet
        """
        bet_id = self._generate_bet_id()
        now = datetime.now()
        
        # Handle edge cases
        if opponent is None or opponent == "":
            opponent = "Multiple" if parlay_fighters else "Unknown"
        
        # Calculate bankroll percentage
        bankroll_percentage = (bet_size / bankroll * 100) if bankroll > 0 else 0.0
        
        # Override risk level calculation if not provided or validate provided risk level
        valid_risk_levels = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
        if risk_level not in valid_risk_levels:
            risk_level = self._calculate_risk_level(expected_value, bankroll_percentage)
        
        # Create comprehensive notes
        notes_parts = []
        if method_prediction:
            notes_parts.append(f"Method prediction: {method_prediction}")
        if parlay_fighters:
            notes_parts.append(f"Parlay fighters: {', '.join(parlay_fighters)}")
        if expected_return:
            notes_parts.append(f"Expected return: ${expected_return:.2f}")
        
        notes = " | ".join(notes_parts) if notes_parts else f"Recommendation from {source}"
        
        # Create bet record
        bet_record = BetRecord(
            bet_id=bet_id,
            date_placed=now.strftime("%Y-%m-%d"),
            timestamp=now.isoformat(),
            event=event_name,
            fighter=fighter,
            opponent=opponent,
            bet_type=bet_type,
            odds_decimal=decimal_odds,
            odds_american=american_odds,
            bet_size=bet_size,
            expected_value=expected_value,
            model_probability=model_probability,
            market_probability=market_probability,
            risk_level=risk_level,
            bankroll_at_time=bankroll,
            bankroll_percentage=bankroll_percentage,
            source=source,
            notes=notes,
            fight_date=event_date if event_date else None
        )
        
        return self._save_bet_record(bet_record)
    
    def _save_bet_record(self, bet_record: BetRecord) -> str:
        """Save bet record to CSV"""
        try:
            # Create backup before making changes
            self._create_backup()
            
            # Load existing data
            df = pd.read_csv(self.csv_path)
            
            # Convert bet record to dict and add to dataframe
            bet_dict = asdict(bet_record)
            new_row = pd.DataFrame([bet_dict])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Save to CSV
            df.to_csv(self.csv_path, index=False)
            
            print(f"✅ Bet logged: {bet_record.bet_id}")
            print(f"   Fighter: {bet_record.fighter} vs {bet_record.opponent}")
            print(f"   Amount: ${bet_record.bet_size:.2f} at {bet_record.odds_decimal} odds")
            print(f"   Expected Value: {bet_record.expected_value:.1%}")
            
            return bet_record.bet_id
            
        except Exception as e:
            print(f"❌ Error saving bet record: {e}")
            return ""
    
    def log_bet_recommendations(self, recommendations: List[Any], event: str, 
                               bankroll: float, source: str = "notebook") -> List[str]:
        """
        Log multiple betting recommendations from notebook output
        
        Args:
            recommendations: List of betting opportunity objects
            event: UFC event name
            bankroll: Current bankroll
            source: Source identifier
            
        Returns:
            List of bet IDs that were created
        """
        bet_ids = []
        
        print(f"📝 Logging {len(recommendations)} betting recommendations for {event}")
        print("-" * 60)
        
        for i, recommendation in enumerate(recommendations, 1):
            try:
                bet_id = self.log_bet_from_opportunity(
                    recommendation, event, bankroll, source, 
                    f"Recommendation {i} from {source}"
                )
                bet_ids.append(bet_id)
                
            except Exception as e:
                print(f"❌ Error logging recommendation {i}: {e}")
        
        print(f"\n✅ Successfully logged {len(bet_ids)} bets")
        return bet_ids
    
    def update_fight_result(self, bet_id: str, actual_result: str, 
                           profit_loss: float, fight_date: str = None,
                           method_actual: str = None) -> bool:
        """
        Update bet with actual fight result
        
        Args:
            bet_id: Unique bet identifier
            actual_result: 'WIN' or 'LOSS'
            profit_loss: Actual profit/loss amount
            fight_date: Date of the fight
            method_actual: How the fight ended
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Create backup before making changes
            self._create_backup()
            
            # Load data
            df = pd.read_csv(self.csv_path)
            
            # Find the bet
            bet_mask = df['bet_id'] == bet_id
            if not bet_mask.any():
                print(f"❌ Bet ID {bet_id} not found")
                return False
            
            # Update the result
            now = datetime.now()
            df.loc[bet_mask, 'actual_result'] = actual_result
            df.loc[bet_mask, 'profit_loss'] = profit_loss
            df.loc[bet_mask, 'result_updated'] = now.isoformat()
            
            if fight_date:
                df.loc[bet_mask, 'fight_date'] = fight_date
            if method_actual:
                df.loc[bet_mask, 'method_actual'] = method_actual
            
            # Calculate ROI
            bet_size = df.loc[bet_mask, 'bet_size'].iloc[0]
            roi = (profit_loss / bet_size) if bet_size > 0 else 0.0
            df.loc[bet_mask, 'roi'] = roi
            
            # Save changes
            df.to_csv(self.csv_path, index=False)
            
            print(f"✅ Updated bet {bet_id}")
            print(f"   Result: {actual_result}")
            print(f"   P&L: ${profit_loss:+.2f}")
            print(f"   ROI: {roi:+.1%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating bet result: {e}")
            return False
    
    def update_event_results(self, event: str, results: Dict[str, Dict]) -> int:
        """
        Update multiple bet results for an entire event
        
        Args:
            event: Event name to update
            results: Dict mapping fighter names to result info
                    {'Fighter Name': {'result': 'WIN/LOSS', 'method': 'KO/TKO', 'date': '2025-01-26'}}
        
        Returns:
            Number of bets updated
        """
        try:
            df = pd.read_csv(self.csv_path)
            event_bets = df[df['event'] == event]
            
            if event_bets.empty:
                print(f"❌ No bets found for event: {event}")
                return 0
            
            updated_count = 0
            
            for _, bet in event_bets.iterrows():
                fighter = bet['fighter']
                
                if fighter in results:
                    result_info = results[fighter]
                    actual_result = result_info.get('result', 'UNKNOWN')
                    fight_date = result_info.get('date', '')
                    method_actual = result_info.get('method', '')
                    
                    # Calculate profit/loss
                    if actual_result == 'WIN':
                        profit_loss = bet['bet_size'] * (bet['odds_decimal'] - 1)
                    elif actual_result == 'LOSS':
                        profit_loss = -bet['bet_size']
                    else:
                        profit_loss = 0.0  # Push/No contest
                    
                    # Update the bet
                    if self.update_fight_result(bet['bet_id'], actual_result, 
                                              profit_loss, fight_date, method_actual):
                        updated_count += 1
            
            print(f"✅ Updated {updated_count} bets for {event}")
            return updated_count
            
        except Exception as e:
            print(f"❌ Error updating event results: {e}")
            return 0
    
    def generate_performance_report(self, days: int = 30) -> Dict:
        """
        Generate comprehensive performance report
        
        Args:
            days: Number of days to include in report (0 = all time)
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                print("📊 No betting data available")
                return {}
            
            # Filter by date if specified
            if days > 0:
                cutoff_date = datetime.now() - pd.Timedelta(days=days)
                df['date_placed'] = pd.to_datetime(df['date_placed'])
                df = df[df['date_placed'] >= cutoff_date]
            
            # Calculate metrics
            total_bets = len(df)
            settled_bets = df[df['actual_result'].notna()]
            pending_bets = df[df['actual_result'].isna()]
            
            if settled_bets.empty:
                print("📊 No settled bets available for analysis")
                return {'total_bets': total_bets, 'settled_bets': 0, 'pending_bets': len(pending_bets)}
            
            # Basic metrics
            total_staked = settled_bets['bet_size'].sum()
            total_profit = settled_bets['profit_loss'].sum()
            wins = settled_bets[settled_bets['actual_result'] == 'WIN']
            losses = settled_bets[settled_bets['actual_result'] == 'LOSS']
            
            win_count = len(wins)
            loss_count = len(losses)
            win_rate = win_count / len(settled_bets) if len(settled_bets) > 0 else 0
            
            # ROI calculations
            roi = total_profit / total_staked if total_staked > 0 else 0
            avg_roi_per_bet = settled_bets['roi'].mean()
            
            # Risk analysis
            risk_distribution = df['risk_level'].value_counts()
            
            # Expected vs actual performance
            expected_profit = (settled_bets['bet_size'] * settled_bets['expected_value']).sum()
            ev_accuracy = (total_profit / expected_profit) if expected_profit != 0 else 0
            
            report = {
                'period': f"Last {days} days" if days > 0 else "All time",
                'total_bets': total_bets,
                'settled_bets': len(settled_bets),
                'pending_bets': len(pending_bets),
                'total_staked': total_staked,
                'total_profit': total_profit,
                'roi': roi,
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_rate,
                'avg_roi_per_bet': avg_roi_per_bet,
                'expected_profit': expected_profit,
                'ev_accuracy': ev_accuracy,
                'risk_distribution': dict(risk_distribution),
                'best_bet': wins.loc[wins['profit_loss'].idxmax()] if not wins.empty else None,
                'worst_bet': losses.loc[losses['profit_loss'].idxmin()] if not losses.empty else None
            }
            
            # Print formatted report
            self._print_performance_report(report)
            
            return report
            
        except Exception as e:
            print(f"❌ Error generating performance report: {e}")
            return {}
    
    def _print_performance_report(self, report: Dict):
        """Print formatted performance report"""
        print("📊 BETTING PERFORMANCE REPORT")
        print("=" * 50)
        print(f"Period: {report.get('period', 'Unknown')}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("📈 OVERVIEW")
        print("-" * 20)
        print(f"Total Bets Placed: {report.get('total_bets', 0)}")
        print(f"Settled Bets: {report.get('settled_bets', 0)}")
        print(f"Pending Bets: {report.get('pending_bets', 0)}")
        print()
        
        if report.get('settled_bets', 0) > 0:
            print("💰 FINANCIAL PERFORMANCE")
            print("-" * 25)
            print(f"Total Staked: ${report.get('total_staked', 0):.2f}")
            print(f"Total Profit: ${report.get('total_profit', 0):+.2f}")
            print(f"ROI: {report.get('roi', 0):+.1%}")
            print(f"Average ROI per Bet: {report.get('avg_roi_per_bet', 0):+.1%}")
            print()
            
            print("🎯 WIN/LOSS RECORD")
            print("-" * 20)
            print(f"Wins: {report.get('win_count', 0)}")
            print(f"Losses: {report.get('loss_count', 0)}")
            print(f"Win Rate: {report.get('win_rate', 0):.1%}")
            print()
            
            print("📊 EXPECTED VALUE ANALYSIS")
            print("-" * 30)
            print(f"Expected Profit: ${report.get('expected_profit', 0):.2f}")
            print(f"Actual Profit: ${report.get('total_profit', 0):+.2f}")
            print(f"EV Accuracy: {report.get('ev_accuracy', 0):.1%}")
            print()
            
            if report.get('risk_distribution'):
                print("⚠️  RISK DISTRIBUTION")
                print("-" * 20)
                for risk, count in report['risk_distribution'].items():
                    print(f"{risk}: {count} bets")
    
    def get_pending_bets(self) -> pd.DataFrame:
        """Get all bets that haven't been settled yet"""
        try:
            df = pd.read_csv(self.csv_path)
            pending = df[df['actual_result'].isna()]
            return pending
        except Exception as e:
            print(f"❌ Error loading pending bets: {e}")
            return pd.DataFrame()
    
    def get_bets_by_event(self, event: str) -> pd.DataFrame:
        """Get all bets for a specific event"""
        try:
            df = pd.read_csv(self.csv_path)
            event_bets = df[df['event'] == event]
            return event_bets
        except Exception as e:
            print(f"❌ Error loading bets for event {event}: {e}")
            return pd.DataFrame()
    
    def export_to_excel(self, filename: str = None) -> str:
        """Export betting data to Excel with multiple sheets"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"betting_analysis_{timestamp}.xlsx"
            
            df = pd.read_csv(self.csv_path)
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # All bets
                df.to_excel(writer, sheet_name='All_Bets', index=False)
                
                # Settled bets only
                settled = df[df['actual_result'].notna()]
                settled.to_excel(writer, sheet_name='Settled_Bets', index=False)
                
                # Pending bets
                pending = df[df['actual_result'].isna()]
                pending.to_excel(writer, sheet_name='Pending_Bets', index=False)
                
                # Performance summary
                report = self.generate_performance_report()
                if report:
                    summary_df = pd.DataFrame([report])
                    summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
            
            print(f"✅ Data exported to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error exporting to Excel: {e}")
            return ""
    
    def _decimal_to_american_odds(self, decimal_odds: float) -> int:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    def validate_data(self) -> Dict:
        """Validate data integrity and return issues found"""
        try:
            df = pd.read_csv(self.csv_path)
            issues = []
            
            # Check for missing required fields
            required_fields = ['bet_id', 'fighter', 'opponent', 'odds_decimal', 'bet_size']
            for field in required_fields:
                missing = df[field].isna().sum()
                if missing > 0:
                    issues.append(f"Missing {field}: {missing} records")
            
            # Check for duplicate bet IDs
            duplicates = df['bet_id'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Duplicate bet IDs: {duplicates}")
            
            # Check for invalid odds
            invalid_odds = df[df['odds_decimal'] <= 1.0]
            if not invalid_odds.empty:
                issues.append(f"Invalid odds (≤1.0): {len(invalid_odds)} records")
            
            # Check for negative bet sizes
            negative_bets = df[df['bet_size'] < 0]
            if not negative_bets.empty:
                issues.append(f"Negative bet sizes: {len(negative_bets)} records")
            
            return {
                'total_records': len(df),
                'issues_found': len(issues),
                'issues': issues,
                'data_quality': 'GOOD' if len(issues) == 0 else 'ISSUES_FOUND'
            }
            
        except Exception as e:
            return {'error': str(e), 'data_quality': 'ERROR'}

# Quick utility functions for notebook integration

def quick_log_notebook_recommendations(recommendations: List[Any], event: str, 
                                     bankroll: float) -> List[str]:
    """Quick function to log recommendations from notebook"""
    tracker = BetTracker()
    return tracker.log_bet_recommendations(recommendations, event, bankroll, "notebook")

def quick_update_results(bet_id: str, result: str, profit_loss: float) -> bool:
    """Quick function to update bet results"""
    tracker = BetTracker()
    return tracker.update_fight_result(bet_id, result, profit_loss)

def quick_performance_report(days: int = 30) -> Dict:
    """Quick function to generate performance report"""
    tracker = BetTracker()
    return tracker.generate_performance_report(days)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Bet Tracking System")
    print("=" * 40)
    
    # Initialize tracker
    tracker = BetTracker()
    
    # Test manual bet logging
    bet_id = tracker.log_bet_manual(
        fighter="Test Fighter",
        opponent="Test Opponent", 
        event="Test Event",
        odds_decimal=2.50,
        bet_size=10.0,
        model_probability=0.65,
        bankroll=100.0,
        notes="Test bet"
    )
    
    # Test result update
    tracker.update_fight_result(bet_id, "WIN", 15.0)
    
    # Test performance report
    tracker.generate_performance_report()
    
    print("\n✅ Bet tracking system test complete")
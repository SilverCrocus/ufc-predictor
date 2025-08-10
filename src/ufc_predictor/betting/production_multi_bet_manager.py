"""
Production Multi-Bet Manager
Complete integration layer for the sophisticated multi-bet UFC betting system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
import warnings
warnings.filterwarnings('ignore')

from .enhanced_multi_bet_system import EnhancedMultiBetSystem, BetLeg, MultiBetRecommendation
from .advanced_correlation_engine import AdvancedCorrelationEngine
from .multi_bet_backtester import MultiBetBacktester, BacktestSummary
from .staking import KellyStaking

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production configuration for multi-bet system."""
    # Bankroll management
    bankroll: float = 1000.0
    max_total_exposure_pct: float = 0.10  # 10% max exposure
    
    # Single bet parameters
    min_single_edge: float = 0.05
    max_single_stake_pct: float = 0.05
    single_bet_threshold: int = 2
    
    # Parlay parameters
    min_parlay_edge: float = 0.10
    max_parlay_stake_pct: float = 0.005
    max_parlay_legs: int = 3
    max_parlays: int = 2
    max_total_parlay_exposure: float = 0.015
    
    # Risk management
    correlation_penalty_alpha: float = 1.2
    confidence_threshold: float = 0.60
    kelly_fraction: float = 0.25
    pessimistic_kelly: bool = True
    
    # Operational parameters
    min_bet_amount: float = 10.0
    max_bet_amount: float = 500.0
    auto_update_correlations: bool = True
    require_feature_data: bool = False


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio monitoring."""
    total_exposure_pct: float
    single_bet_exposure_pct: float
    parlay_exposure_pct: float
    max_correlation: float
    avg_correlation: float
    portfolio_var_95: float
    expected_kelly_growth: float
    risk_score: float


@dataclass
class PerformanceReport:
    """Performance monitoring report."""
    timestamp: datetime
    active_bets: int
    total_exposure: float
    expected_return: float
    risk_metrics: RiskMetrics
    recommendations_summary: Dict[str, Any]
    warnings: List[str]
    status: str


class ProductionMultiBetManager:
    """
    Production-ready multi-bet manager with comprehensive risk management,
    monitoring, and integration capabilities.
    """
    
    def __init__(
        self,
        config: Optional[ProductionConfig] = None,
        historical_data_path: Optional[str] = None,
        auto_initialize: bool = True
    ):
        """
        Initialize production multi-bet manager.
        
        Args:
            config: Production configuration
            historical_data_path: Path to historical data for correlation engine
            auto_initialize: Whether to auto-initialize components
        """
        self.config = config or ProductionConfig()
        self.historical_data_path = historical_data_path
        
        # Core components
        self.multi_bet_system: Optional[EnhancedMultiBetSystem] = None
        self.correlation_engine: Optional[AdvancedCorrelationEngine] = None
        self.backtester: Optional[MultiBetBacktester] = None
        
        # State tracking
        self.current_bankroll = self.config.bankroll
        self.active_recommendations: List[MultiBetRecommendation] = []
        self.performance_history: List[PerformanceReport] = []
        self.correlation_cache: Dict[str, float] = {}
        
        # Risk monitoring
        self.current_exposure = 0.0
        self.risk_alerts: List[str] = []
        
        if auto_initialize:
            self.initialize_components()
    
    def initialize_components(self):
        """Initialize all system components."""
        logger.info("🚀 Initializing production multi-bet manager")
        
        # Initialize correlation engine
        self.correlation_engine = AdvancedCorrelationEngine(
            historical_data_path=self.historical_data_path
        )
        
        # Initialize multi-bet system
        system_config = {
            'single_bet_threshold': self.config.single_bet_threshold,
            'max_parlay_legs': self.config.max_parlay_legs,
            'min_single_edge': self.config.min_single_edge,
            'min_parlay_edge': self.config.min_parlay_edge,
            'max_single_stake_pct': self.config.max_single_stake_pct,
            'max_parlay_stake_pct': self.config.max_parlay_stake_pct,
            'max_total_parlay_exposure': self.config.max_total_parlay_exposure,
            'max_parlays': self.config.max_parlays,
            'correlation_penalty_alpha': self.config.correlation_penalty_alpha,
            'confidence_threshold': self.config.confidence_threshold
        }
        
        self.multi_bet_system = EnhancedMultiBetSystem(
            bankroll=self.current_bankroll,
            **system_config
        )
        
        # Initialize backtester
        self.backtester = MultiBetBacktester(system_config, self.correlation_engine)
        
        logger.info("✅ All components initialized successfully")
    
    def analyze_card(
        self,
        fight_predictions: List[Dict[str, Any]],
        live_odds: List[Dict[str, Any]],
        event_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[MultiBetRecommendation]], PerformanceReport]:
        """
        Analyze a complete UFC card and generate betting recommendations.
        
        Args:
            fight_predictions: List of fight predictions from model
            live_odds: List of current betting odds
            event_metadata: Event information (venue, date, etc.)
            
        Returns:
            (recommendations, performance_report)
        """
        logger.info(f"📊 Analyzing UFC card with {len(fight_predictions)} fights")
        
        try:
            # Validate inputs
            self._validate_inputs(fight_predictions, live_odds)
            
            # Create bet legs
            bet_legs = self._create_bet_legs(fight_predictions, live_odds, event_metadata)
            
            # Update system bankroll
            self.multi_bet_system.bankroll = self.current_bankroll
            
            # Get recommendations
            recommendations = self.multi_bet_system.analyze_betting_opportunities(
                bet_legs, self._prepare_live_features(bet_legs)
            )
            
            # Apply production filters
            filtered_recommendations = self._apply_production_filters(recommendations)
            
            # Update active recommendations
            self.active_recommendations = (
                filtered_recommendations['single_bets'] + 
                filtered_recommendations['parlays']
            )
            
            # Generate performance report
            performance_report = self._generate_performance_report(filtered_recommendations)
            
            # Log analysis summary
            self._log_analysis_summary(filtered_recommendations, performance_report)
            
            return filtered_recommendations, performance_report
            
        except Exception as e:
            logger.error(f"Error analyzing card: {e}")
            raise
    
    def _validate_inputs(
        self,
        fight_predictions: List[Dict[str, Any]],
        live_odds: List[Dict[str, Any]]
    ):
        """Validate input data quality."""
        
        if not fight_predictions:
            raise ValueError("No fight predictions provided")
        
        if not live_odds:
            raise ValueError("No live odds provided")
        
        # Check for required fields in predictions
        required_prediction_fields = ['fighter', 'opponent', 'probability']
        for i, pred in enumerate(fight_predictions):
            missing_fields = [f for f in required_prediction_fields if f not in pred]
            if missing_fields:
                raise ValueError(f"Fight {i} missing fields: {missing_fields}")
        
        # Check for required fields in odds
        required_odds_fields = ['fighter', 'odds']
        for i, odds in enumerate(live_odds):
            missing_fields = [f for f in required_odds_fields if f not in odds]
            if missing_fields:
                raise ValueError(f"Odds {i} missing fields: {missing_fields}")
    
    def _create_bet_legs(
        self,
        fight_predictions: List[Dict[str, Any]],
        live_odds: List[Dict[str, Any]],
        event_metadata: Optional[Dict[str, Any]]
    ) -> List[BetLeg]:
        """Create BetLeg objects from predictions and odds."""
        
        bet_legs = []
        
        # Create odds lookup
        odds_lookup = {}
        for odds_data in live_odds:
            fighter_key = self._normalize_fighter_name(odds_data['fighter'])
            odds_lookup[fighter_key] = odds_data['odds']
        
        # Process each prediction
        for i, prediction in enumerate(fight_predictions):
            fighter = prediction['fighter']
            opponent = prediction['opponent']
            probability = float(prediction['probability'])
            
            # Find matching odds
            fighter_key = self._normalize_fighter_name(fighter)
            if fighter_key not in odds_lookup:
                logger.warning(f"No odds found for {fighter}")
                continue
            
            odds = float(odds_lookup[fighter_key])
            
            # Calculate edge
            edge = (probability * odds) - 1
            
            if edge <= 0:
                continue  # Skip negative edge bets
            
            # Extract features
            features = {}
            for key, value in prediction.items():
                if key not in ['fighter', 'opponent', 'probability']:
                    features[key] = value
            
            # Create confidence interval if available
            confidence_interval = None
            if 'probability_lower' in prediction and 'probability_upper' in prediction:
                confidence_interval = (
                    float(prediction['probability_lower']),
                    float(prediction['probability_upper'])
                )
            
            # Create bet leg
            bet_leg = BetLeg(
                fighter=fighter,
                opponent=opponent,
                probability=probability,
                odds=odds,
                edge=edge,
                event=event_metadata.get('event_name', f'Event_{datetime.now().strftime("%Y%m%d")}') if event_metadata else f'Event_{datetime.now().strftime("%Y%m%d")}',
                division=features.get('division'),
                card_position=i + 1,
                features=features,
                confidence_interval=confidence_interval
            )
            
            bet_legs.append(bet_leg)
        
        logger.info(f"Created {len(bet_legs)} valid bet legs from {len(fight_predictions)} predictions")
        
        return bet_legs
    
    def _normalize_fighter_name(self, name: str) -> str:
        """Normalize fighter name for matching."""
        return name.lower().strip().replace('.', '').replace('-', ' ')
    
    def _prepare_live_features(self, bet_legs: List[BetLeg]) -> pd.DataFrame:
        """Prepare live features DataFrame for correlation analysis."""
        
        if not bet_legs:
            return pd.DataFrame()
        
        # Extract features from bet legs
        feature_data = []
        
        for leg in bet_legs:
            row_data = {
                'fighter': leg.fighter,
                'opponent': leg.opponent,
                'event': leg.event,
                'division': leg.division,
                'card_position': leg.card_position,
                'probability': leg.probability,
                'odds': leg.odds,
                'edge': leg.edge
            }
            
            # Add additional features
            if leg.features:
                row_data.update(leg.features)
            
            feature_data.append(row_data)
        
        return pd.DataFrame(feature_data)
    
    def _apply_production_filters(
        self,
        recommendations: Dict[str, List[MultiBetRecommendation]]
    ) -> Dict[str, List[MultiBetRecommendation]]:
        """Apply production-specific filters and constraints."""
        
        filtered_recommendations = {
            'single_bets': [],
            'parlays': []
        }
        
        # Filter single bets
        for rec in recommendations['single_bets']:
            if self._passes_production_filters(rec):
                # Adjust stake based on constraints
                adjusted_rec = self._adjust_stake_for_production(rec)
                filtered_recommendations['single_bets'].append(adjusted_rec)
        
        # Filter parlays
        for rec in recommendations['parlays']:
            if self._passes_production_filters(rec):
                # Adjust stake based on constraints
                adjusted_rec = self._adjust_stake_for_production(rec)
                filtered_recommendations['parlays'].append(adjusted_rec)
        
        # Apply portfolio-level constraints
        filtered_recommendations = self._apply_portfolio_constraints(filtered_recommendations)
        
        return filtered_recommendations
    
    def _passes_production_filters(self, rec: MultiBetRecommendation) -> bool:
        """Check if recommendation passes production filters."""
        
        # Minimum bet amount
        if rec.recommended_stake < self.config.min_bet_amount:
            return False
        
        # Maximum bet amount
        if rec.recommended_stake > self.config.max_bet_amount:
            rec.recommended_stake = self.config.max_bet_amount
        
        # Confidence threshold
        if rec.confidence_score < self.config.confidence_threshold:
            return False
        
        # Risk score threshold
        if rec.risk_score > 0.8:
            return False
        
        return True
    
    def _adjust_stake_for_production(self, rec: MultiBetRecommendation) -> MultiBetRecommendation:
        """Adjust stake amounts for production constraints."""
        
        # Ensure minimum bet
        rec.recommended_stake = max(rec.recommended_stake, self.config.min_bet_amount)
        
        # Cap at maximum bet
        rec.recommended_stake = min(rec.recommended_stake, self.config.max_bet_amount)
        
        # Round to nearest dollar
        rec.recommended_stake = round(rec.recommended_stake, 0)
        
        return rec
    
    def _apply_portfolio_constraints(
        self,
        recommendations: Dict[str, List[MultiBetRecommendation]]
    ) -> Dict[str, List[MultiBetRecommendation]]:
        """Apply portfolio-level position sizing constraints."""
        
        # Calculate total exposure
        total_stake = sum(
            rec.recommended_stake for rec in 
            recommendations['single_bets'] + recommendations['parlays']
        )
        
        max_total_exposure = self.current_bankroll * self.config.max_total_exposure_pct
        
        # Scale down if over exposure limit
        if total_stake > max_total_exposure:
            scale_factor = max_total_exposure / total_stake
            
            logger.warning(f"Scaling down bets by {scale_factor:.2f} due to exposure limits")
            
            for rec in recommendations['single_bets'] + recommendations['parlays']:
                rec.recommended_stake *= scale_factor
                rec.kelly_fraction *= scale_factor
        
        return recommendations
    
    def _generate_performance_report(
        self,
        recommendations: Dict[str, List[MultiBetRecommendation]]
    ) -> PerformanceReport:
        """Generate comprehensive performance report."""
        
        all_recs = recommendations['single_bets'] + recommendations['parlays']
        
        if not all_recs:
            return PerformanceReport(
                timestamp=datetime.now(),
                active_bets=0,
                total_exposure=0.0,
                expected_return=0.0,
                risk_metrics=RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0),
                recommendations_summary={},
                warnings=[],
                status='no_opportunities'
            )
        
        # Calculate metrics
        total_exposure = sum(rec.recommended_stake for rec in all_recs)
        expected_return = sum(
            rec.expected_value * rec.recommended_stake for rec in all_recs
        )
        
        # Risk metrics
        correlations = [rec.correlation_penalty for rec in recommendations['parlays']]
        max_correlation = max(correlations) if correlations else 0.0
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        single_exposure = sum(rec.recommended_stake for rec in recommendations['single_bets'])
        parlay_exposure = sum(rec.recommended_stake for rec in recommendations['parlays'])
        
        risk_metrics = RiskMetrics(
            total_exposure_pct=(total_exposure / self.current_bankroll) * 100,
            single_bet_exposure_pct=(single_exposure / self.current_bankroll) * 100,
            parlay_exposure_pct=(parlay_exposure / self.current_bankroll) * 100,
            max_correlation=max_correlation,
            avg_correlation=avg_correlation,
            portfolio_var_95=self._calculate_portfolio_var(all_recs),
            expected_kelly_growth=expected_return / self.current_bankroll,
            risk_score=np.mean([rec.risk_score for rec in all_recs])
        )
        
        # Recommendations summary
        recommendations_summary = {
            'single_bets': {
                'count': len(recommendations['single_bets']),
                'total_stake': single_exposure,
                'avg_edge': np.mean([rec.expected_value for rec in recommendations['single_bets']]) if recommendations['single_bets'] else 0,
                'avg_odds': np.mean([rec.combined_odds for rec in recommendations['single_bets']]) if recommendations['single_bets'] else 0
            },
            'parlays': {
                'count': len(recommendations['parlays']),
                'total_stake': parlay_exposure,
                'avg_edge': np.mean([rec.expected_value for rec in recommendations['parlays']]) if recommendations['parlays'] else 0,
                'avg_odds': np.mean([rec.combined_odds for rec in recommendations['parlays']]) if recommendations['parlays'] else 0,
                'avg_legs': np.mean([len(rec.legs) for rec in recommendations['parlays']]) if recommendations['parlays'] else 0
            },
            'total_expected_return': expected_return,
            'expected_roi': (expected_return / total_exposure) * 100 if total_exposure > 0 else 0
        }
        
        # Generate warnings
        warnings = self._generate_warnings(risk_metrics, recommendations)
        
        # Determine status
        status = self._determine_system_status(risk_metrics, recommendations, warnings)
        
        return PerformanceReport(
            timestamp=datetime.now(),
            active_bets=len(all_recs),
            total_exposure=total_exposure,
            expected_return=expected_return,
            risk_metrics=risk_metrics,
            recommendations_summary=recommendations_summary,
            warnings=warnings,
            status=status
        )
    
    def _calculate_portfolio_var(self, recommendations: List[MultiBetRecommendation], confidence: float = 0.05) -> float:
        """Calculate portfolio Value-at-Risk."""
        
        if not recommendations:
            return 0.0
        
        # Simplified VaR calculation
        # In production, would use Monte Carlo simulation
        
        total_stake = sum(rec.recommended_stake for rec in recommendations)
        avg_prob = np.mean([rec.adjusted_probability for rec in recommendations])
        
        # Worst case scenario (all bets lose)
        worst_case_loss = -total_stake
        
        # Probability-weighted loss
        expected_loss = total_stake * (1 - avg_prob)
        
        # Simple VaR estimate
        var_95 = worst_case_loss * confidence + expected_loss * (1 - confidence)
        
        return var_95
    
    def _generate_warnings(
        self,
        risk_metrics: RiskMetrics,
        recommendations: Dict[str, List[MultiBetRecommendation]]
    ) -> List[str]:
        """Generate warning messages based on risk analysis."""
        
        warnings = []
        
        # Exposure warnings
        if risk_metrics.total_exposure_pct > 8.0:
            warnings.append(f"High total exposure: {risk_metrics.total_exposure_pct:.1f}%")
        
        if risk_metrics.parlay_exposure_pct > 2.0:
            warnings.append(f"High parlay exposure: {risk_metrics.parlay_exposure_pct:.1f}%")
        
        # Correlation warnings
        if risk_metrics.max_correlation > 0.25:
            warnings.append(f"High correlation detected: {risk_metrics.max_correlation:.2f}")
        
        # Risk warnings
        if risk_metrics.risk_score > 0.7:
            warnings.append(f"High portfolio risk score: {risk_metrics.risk_score:.2f}")
        
        # Recommendation quality warnings
        single_count = len(recommendations['single_bets'])
        parlay_count = len(recommendations['parlays'])
        
        if single_count == 0 and parlay_count > 0:
            warnings.append("No qualified single bets - parlay-only strategy")
        
        if parlay_count > 3:
            warnings.append(f"High number of parlays recommended: {parlay_count}")
        
        return warnings
    
    def _determine_system_status(
        self,
        risk_metrics: RiskMetrics,
        recommendations: Dict[str, List[MultiBetRecommendation]],
        warnings: List[str]
    ) -> str:
        """Determine overall system status."""
        
        total_bets = len(recommendations['single_bets']) + len(recommendations['parlays'])
        
        if total_bets == 0:
            return 'no_opportunities'
        
        if len(warnings) > 3:
            return 'high_risk'
        
        if risk_metrics.total_exposure_pct > 8.0 or risk_metrics.risk_score > 0.7:
            return 'elevated_risk'
        
        if total_bets >= 3 and risk_metrics.expected_kelly_growth > 0.02:
            return 'optimal'
        
        return 'normal'
    
    def _log_analysis_summary(
        self,
        recommendations: Dict[str, List[MultiBetRecommendation]],
        report: PerformanceReport
    ):
        """Log analysis summary for monitoring."""
        
        logger.info(f"📊 ANALYSIS SUMMARY:")
        logger.info(f"   Status: {report.status}")
        logger.info(f"   Single bets: {len(recommendations['single_bets'])}")
        logger.info(f"   Parlays: {len(recommendations['parlays'])}")
        logger.info(f"   Total exposure: ${report.total_exposure:.2f} ({report.risk_metrics.total_exposure_pct:.1f}%)")
        logger.info(f"   Expected return: ${report.expected_return:.2f}")
        logger.info(f"   Risk score: {report.risk_metrics.risk_score:.2f}")
        
        if report.warnings:
            logger.warning(f"   Warnings: {len(report.warnings)}")
            for warning in report.warnings:
                logger.warning(f"   - {warning}")
    
    def update_bankroll(self, new_bankroll: float):
        """Update current bankroll and system parameters."""
        self.current_bankroll = new_bankroll
        if self.multi_bet_system:
            self.multi_bet_system.bankroll = new_bankroll
        
        logger.info(f"💰 Bankroll updated to ${new_bankroll:,.2f}")
    
    def run_backtest(
        self,
        historical_data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestSummary:
        """Run comprehensive backtest of the multi-bet system."""
        
        if not self.backtester:
            raise ValueError("Backtester not initialized")
        
        logger.info("🔄 Running comprehensive backtest...")
        
        return self.backtester.run_backtest(
            historical_data,
            initial_bankroll=self.config.bankroll,
            start_date=start_date,
            end_date=end_date
        )
    
    def export_recommendations(
        self,
        recommendations: Dict[str, List[MultiBetRecommendation]],
        output_path: str
    ):
        """Export recommendations to structured format."""
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'single_bets': [],
            'parlays': []
        }
        
        # Export single bets
        for rec in recommendations['single_bets']:
            export_data['single_bets'].append({
                'fighter': rec.legs[0].fighter,
                'opponent': rec.legs[0].opponent,
                'probability': rec.adjusted_probability,
                'odds': rec.combined_odds,
                'expected_value': rec.expected_value,
                'recommended_stake': rec.recommended_stake,
                'kelly_fraction': rec.kelly_fraction,
                'confidence_score': rec.confidence_score,
                'risk_score': rec.risk_score
            })
        
        # Export parlays
        for rec in recommendations['parlays']:
            export_data['parlays'].append({
                'legs': [{
                    'fighter': leg.fighter,
                    'opponent': leg.opponent,
                    'probability': leg.probability,
                    'odds': leg.odds
                } for leg in rec.legs],
                'combined_probability': rec.combined_probability,
                'adjusted_probability': rec.adjusted_probability,
                'combined_odds': rec.combined_odds,
                'expected_value': rec.expected_value,
                'correlation_penalty': rec.correlation_penalty,
                'recommended_stake': rec.recommended_stake,
                'kelly_fraction': rec.kelly_fraction,
                'confidence_score': rec.confidence_score,
                'risk_score': rec.risk_score
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📁 Recommendations exported to {output_path}")


if __name__ == "__main__":
    # Demo production system
    print("🏭 Production Multi-Bet Manager Demo")
    print("=" * 50)
    
    # Initialize production manager
    config = ProductionConfig(
        bankroll=5000.0,
        min_single_edge=0.05,
        min_parlay_edge=0.10,
        single_bet_threshold=2,
        max_parlays=2
    )
    
    manager = ProductionMultiBetManager(config=config)
    
    # Sample card data
    fight_predictions = [
        {
            'fighter': 'Jon Jones',
            'opponent': 'Stipe Miocic',
            'probability': 0.75,
            'probability_lower': 0.70,
            'probability_upper': 0.80,
            'division': 'heavyweight'
        },
        {
            'fighter': 'Islam Makhachev',
            'opponent': 'Charles Oliveira',
            'probability': 0.65,
            'probability_lower': 0.60,
            'probability_upper': 0.70,
            'division': 'lightweight'
        },
        {
            'fighter': 'Sean O\'Malley',
            'opponent': 'Marlon Vera',
            'probability': 0.58,
            'probability_lower': 0.54,
            'probability_upper': 0.62,
            'division': 'bantamweight'
        }
    ]
    
    live_odds = [
        {'fighter': 'Jon Jones', 'odds': 1.8},
        {'fighter': 'Islam Makhachev', 'odds': 2.2},
        {'fighter': 'Sean O\'Malley', 'odds': 2.0}
    ]
    
    event_metadata = {
        'event_name': 'UFC 309',
        'date': datetime.now(),
        'venue': 'Madison Square Garden'
    }
    
    # Analyze card
    recommendations, report = manager.analyze_card(
        fight_predictions, live_odds, event_metadata
    )
    
    # Display results
    print(f"\n📊 PRODUCTION ANALYSIS:")
    print(f"Status: {report.status}")
    print(f"Active bets: {report.active_bets}")
    print(f"Total exposure: ${report.total_exposure:.2f}")
    print(f"Expected return: ${report.expected_return:.2f}")
    
    print(f"\n💰 SINGLE BETS ({report.recommendations_summary['single_bets']['count']}):")
    for rec in recommendations['single_bets']:
        print(f"  • {rec.legs[0].fighter}: ${rec.recommended_stake:.0f} @ {rec.combined_odds:.2f} ({rec.expected_value:.1%} EV)")
    
    print(f"\n🎯 PARLAYS ({report.recommendations_summary['parlays']['count']}):")
    for rec in recommendations['parlays']:
        fighters = " + ".join([leg.fighter for leg in rec.legs])
        print(f"  • {fighters}: ${rec.recommended_stake:.0f} @ {rec.combined_odds:.2f} ({rec.expected_value:.1%} EV)")
    
    if report.warnings:
        print(f"\n⚠️  WARNINGS:")
        for warning in report.warnings:
            print(f"  • {warning}")
    
    print(f"\n📈 RISK METRICS:")
    print(f"Total exposure: {report.risk_metrics.total_exposure_pct:.1f}%")
    print(f"Max correlation: {report.risk_metrics.max_correlation:.2f}")
    print(f"Risk score: {report.risk_metrics.risk_score:.2f}")
    print(f"Expected Kelly growth: {report.risk_metrics.expected_kelly_growth:.1%}")
    
    print("\n✅ Production multi-bet manager ready for deployment!")
"""
Risk Agent for UFC Betting System
=================================

Specialized agent for risk management and portfolio optimization:
- Kelly criterion sizing and fractional Kelly implementation
- Portfolio diversification and correlation analysis
- Bankroll management and drawdown protection
- Dynamic position sizing based on confidence and volatility
- Multi-bet portfolio optimization
- Real-time risk monitoring and circuit breakers
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import math

from .base_agent import BaseAgent, AgentPriority, AgentMessage

logger = logging.getLogger(__name__)


@dataclass
class BetRecommendation:
    """Individual bet recommendation with risk metrics"""
    fight_id: str
    fighter: str
    opponent: str
    predicted_probability: float
    market_odds: float
    decimal_odds: float
    expected_value: float
    kelly_fraction: float
    recommended_stake: float
    max_stake: float
    confidence_score: float
    uncertainty: float
    risk_score: float  # 0-1, higher is riskier
    correlation_risk: float  # Correlation with other positions


@dataclass
class PortfolioMetrics:
    """Portfolio-level risk and performance metrics"""
    total_stake: float
    expected_return: float
    expected_variance: float
    sharpe_ratio: float
    max_drawdown_risk: float
    bankroll_utilization: float
    diversification_score: float
    correlation_penalty: float
    risk_adjusted_ev: float
    value_at_risk_95: float  # 95% VaR
    conditional_var_95: float  # Expected shortfall


@dataclass
class RiskLimits:
    """Risk management limits and constraints"""
    max_single_bet_percentage: float = 0.05  # 5%
    max_portfolio_exposure: float = 0.25     # 25%
    max_correlated_exposure: float = 0.15    # 15%
    min_expected_value: float = 0.05         # 5%
    max_kelly_fraction: float = 0.25         # Quarter Kelly
    max_uncertainty_threshold: float = 0.4   # 40%
    max_drawdown_limit: float = 0.20         # 20%
    min_diversification_score: float = 0.6   # 60%


@dataclass
class PortfolioPosition:
    """Current portfolio position tracking"""
    bet_id: str
    fighter: str
    stake: float
    odds: float
    expected_return: float
    entry_time: datetime
    event_date: Optional[datetime] = None
    status: str = "open"  # "open", "won", "lost", "void"
    correlation_group: Optional[str] = None


class RiskAgent(BaseAgent):
    """
    Risk management and portfolio optimization agent
    
    Responsibilities:
    - Calculate optimal bet sizes using Kelly criterion
    - Manage portfolio diversification and correlation
    - Monitor real-time risk exposure and implement circuit breakers
    - Optimize multi-bet portfolios for risk-adjusted returns
    - Track performance and adjust risk parameters dynamically
    - Integrate with existing betting service for enhanced risk management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RiskAgent
        
        Args:
            config: Risk agent configuration
        """
        super().__init__(
            agent_id="risk_agent",
            priority=AgentPriority.CRITICAL
        )
        
        self.config = config
        
        # Risk management configuration
        self.risk_limits = RiskLimits(**config.get('risk_limits', {}))
        self.bankroll = config.get('bankroll', 1000.0)
        self.kelly_multiplier = config.get('kelly_multiplier', 0.25)  # Fractional Kelly
        
        # Portfolio tracking
        self.current_positions: Dict[str, PortfolioPosition] = {}
        self.position_history: List[PortfolioPosition] = []
        self.portfolio_performance: List[Dict[str, float]] = []
        
        # Risk monitoring
        self.risk_monitoring_interval = config.get('risk_monitoring_interval', 300)  # 5 minutes
        self.circuit_breaker_enabled = config.get('circuit_breaker_enabled', True)
        self.daily_loss_limit = config.get('daily_loss_limit', 0.1)  # 10% daily loss limit
        
        # Performance tracking
        self.performance_window = config.get('performance_window', 30)  # 30 days
        self.risk_adjustment_frequency = config.get('risk_adjustment_frequency', 7)  # 7 days
        
        # Correlation analysis
        self.correlation_lookback = config.get('correlation_lookback', 100)  # Historical fights
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_correlation_update = datetime.now()
        
        # Register message handlers
        self.register_message_handler('optimize_portfolio', self._handle_optimize_portfolio)
        self.register_message_handler('calculate_kelly_sizing', self._handle_calculate_kelly_sizing)
        self.register_message_handler('check_risk_limits', self._handle_check_risk_limits)
        self.register_message_handler('update_position', self._handle_update_position)
        self.register_message_handler('get_portfolio_metrics', self._handle_get_portfolio_metrics)
        
        logger.info("RiskAgent initialized")
    
    async def _initialize_agent(self) -> bool:
        """Initialize risk agent components"""
        try:
            # Load historical performance data if available
            await self._load_historical_data()
            
            # Initialize correlation matrix
            await self._initialize_correlation_matrix()
            
            # Perform initial risk assessment
            await self._perform_initial_risk_assessment()
            
            logger.info("RiskAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"RiskAgent initialization failed: {e}")
            return False
    
    async def _start_agent(self) -> bool:
        """Start risk agent operations"""
        try:
            # Start risk monitoring
            self._risk_monitoring_task = asyncio.create_task(
                self._risk_monitoring_loop()
            )
            
            # Start performance tracking
            self._performance_tracking_task = asyncio.create_task(
                self._performance_tracking_loop()
            )
            
            logger.info("RiskAgent started successfully")
            return True
            
        except Exception as e:
            logger.error(f"RiskAgent start failed: {e}")
            return False
    
    async def _stop_agent(self) -> bool:
        """Stop risk agent operations"""
        try:
            # Cancel background tasks
            if hasattr(self, '_risk_monitoring_task'):
                self._risk_monitoring_task.cancel()
            
            if hasattr(self, '_performance_tracking_task'):
                self._performance_tracking_task.cancel()
            
            # Save current state
            await self._save_risk_state()
            
            logger.info("RiskAgent stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"RiskAgent stop failed: {e}")
            return False
    
    async def _process_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Process incoming messages"""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            return await handler(message.payload)
        else:
            logger.warning(f"RiskAgent: No handler for message type '{message.message_type}'")
            return None
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Risk agent health check"""
        current_exposure = sum(pos.stake for pos in self.current_positions.values())
        exposure_ratio = current_exposure / self.bankroll
        
        health_info = {
            'bankroll': self.bankroll,
            'current_positions': len(self.current_positions),
            'total_exposure': current_exposure,
            'exposure_ratio': exposure_ratio,
            'circuit_breaker_enabled': self.circuit_breaker_enabled,
            'within_risk_limits': exposure_ratio <= self.risk_limits.max_portfolio_exposure,
            'correlation_matrix_updated': self.correlation_matrix is not None,
            'last_correlation_update': self.last_correlation_update.isoformat()
        }
        
        return health_info
    
    # === Kelly Criterion and Position Sizing ===
    
    def calculate_kelly_fraction(self, 
                                predicted_probability: float,
                                decimal_odds: float,
                                uncertainty: float = 0.0) -> float:
        """
        Calculate Kelly fraction for optimal bet sizing
        
        Args:
            predicted_probability: Model predicted probability
            decimal_odds: Market decimal odds
            uncertainty: Prediction uncertainty (0-1)
            
        Returns:
            Kelly fraction (0-1)
        """
        try:
            # Basic Kelly formula: f = (bp - q) / b
            # where b = odds - 1, p = probability, q = 1 - p
            
            b = decimal_odds - 1  # Net odds
            p = predicted_probability
            q = 1 - p
            
            if b <= 0 or p <= 0 or p >= 1:
                return 0.0
            
            # Basic Kelly fraction
            kelly_fraction = (b * p - q) / b
            
            # Adjust for uncertainty (reduce position size with higher uncertainty)
            uncertainty_adjustment = 1.0 - (uncertainty * 0.5)  # Max 50% reduction
            kelly_fraction *= uncertainty_adjustment
            
            # Apply fractional Kelly multiplier
            kelly_fraction *= self.kelly_multiplier
            
            # Ensure within bounds
            kelly_fraction = max(0.0, min(kelly_fraction, self.risk_limits.max_kelly_fraction))
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Kelly fraction calculation failed: {e}")
            return 0.0
    
    def calculate_position_size(self, 
                              kelly_fraction: float,
                              confidence_score: float = 1.0,
                              correlation_penalty: float = 0.0) -> float:
        """
        Calculate actual position size considering risk constraints
        
        Args:
            kelly_fraction: Base Kelly fraction
            confidence_score: Model confidence (0-1)
            correlation_penalty: Penalty for correlation with existing positions
            
        Returns:
            Recommended position size as percentage of bankroll
        """
        try:
            # Start with Kelly fraction
            position_size = kelly_fraction
            
            # Adjust for confidence
            position_size *= confidence_score
            
            # Apply correlation penalty
            position_size *= (1.0 - correlation_penalty)
            
            # Apply hard limits
            position_size = min(position_size, self.risk_limits.max_single_bet_percentage)
            
            # Check portfolio constraints
            current_exposure = sum(pos.stake for pos in self.current_positions.values())
            remaining_capacity = (self.risk_limits.max_portfolio_exposure * self.bankroll) - current_exposure
            
            max_affordable = remaining_capacity / self.bankroll
            position_size = min(position_size, max_affordable)
            
            return max(0.0, position_size)
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    # === Portfolio Optimization ===
    
    async def optimize_portfolio(self, 
                               candidate_bets: List[Dict[str, Any]]) -> List[BetRecommendation]:
        """
        Optimize portfolio of candidate bets for risk-adjusted returns
        
        Args:
            candidate_bets: List of candidate bet opportunities
            
        Returns:
            Optimized bet recommendations
        """
        try:
            async with self.track_operation("optimize_portfolio"):
                if not candidate_bets:
                    return []
                
                # Create initial bet recommendations
                recommendations = []
                
                for bet_data in candidate_bets:
                    # Extract bet parameters
                    predicted_prob = bet_data.get('predicted_probability', 0.5)
                    decimal_odds = bet_data.get('decimal_odds', 2.0)
                    confidence = bet_data.get('confidence_score', 0.8)
                    uncertainty = bet_data.get('uncertainty', 0.2)
                    
                    # Calculate Kelly sizing
                    kelly_fraction = self.calculate_kelly_fraction(
                        predicted_prob, decimal_odds, uncertainty
                    )
                    
                    # Calculate correlation risk
                    correlation_risk = await self._calculate_correlation_risk(bet_data)
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        kelly_fraction, confidence, correlation_risk
                    )
                    
                    # Create recommendation
                    if position_size > 0:
                        recommendation = BetRecommendation(
                            fight_id=bet_data.get('fight_id', ''),
                            fighter=bet_data.get('fighter', ''),
                            opponent=bet_data.get('opponent', ''),
                            predicted_probability=predicted_prob,
                            market_odds=decimal_odds,
                            decimal_odds=decimal_odds,
                            expected_value=(predicted_prob * decimal_odds) - 1.0,
                            kelly_fraction=kelly_fraction,
                            recommended_stake=position_size * self.bankroll,
                            max_stake=self.risk_limits.max_single_bet_percentage * self.bankroll,
                            confidence_score=confidence,
                            uncertainty=uncertainty,
                            risk_score=self._calculate_risk_score(bet_data),
                            correlation_risk=correlation_risk
                        )
                        
                        recommendations.append(recommendation)
                
                # Apply portfolio-level optimization
                optimized_recommendations = await self._apply_portfolio_optimization(
                    recommendations
                )
                
                logger.info(f"Portfolio optimization completed: {len(optimized_recommendations)} recommendations")
                
                return optimized_recommendations
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return []
    
    async def _apply_portfolio_optimization(self, 
                                          recommendations: List[BetRecommendation]) -> List[BetRecommendation]:
        """Apply portfolio-level optimization constraints"""
        
        if not recommendations:
            return recommendations
        
        # Sort by risk-adjusted expected value
        recommendations.sort(
            key=lambda x: x.expected_value / (1 + x.risk_score), 
            reverse=True
        )
        
        # Apply portfolio constraints iteratively
        optimized = []
        total_stake = 0.0
        correlation_groups = {}
        
        for rec in recommendations:
            # Check portfolio exposure limit
            if total_stake + rec.recommended_stake > self.risk_limits.max_portfolio_exposure * self.bankroll:
                # Reduce stake to fit within portfolio limit
                remaining_capacity = (self.risk_limits.max_portfolio_exposure * self.bankroll) - total_stake
                if remaining_capacity > 0:
                    rec.recommended_stake = remaining_capacity
                else:
                    continue  # Skip this bet
            
            # Check correlation limits
            correlation_group = self._get_correlation_group(rec)
            if correlation_group:
                group_exposure = correlation_groups.get(correlation_group, 0.0)
                max_group_exposure = self.risk_limits.max_correlated_exposure * self.bankroll
                
                if group_exposure + rec.recommended_stake > max_group_exposure:
                    # Reduce stake to fit within correlation limit
                    remaining_group_capacity = max_group_exposure - group_exposure
                    if remaining_group_capacity > 0:
                        rec.recommended_stake = min(rec.recommended_stake, remaining_group_capacity)
                    else:
                        continue  # Skip this bet
                
                correlation_groups[correlation_group] = group_exposure + rec.recommended_stake
            
            optimized.append(rec)
            total_stake += rec.recommended_stake
        
        return optimized
    
    def _calculate_risk_score(self, bet_data: Dict[str, Any]) -> float:
        """Calculate comprehensive risk score for a bet"""
        
        # Base risk from uncertainty
        uncertainty = bet_data.get('uncertainty', 0.2)
        base_risk = uncertainty
        
        # Risk from odds (very high or very low odds are riskier)
        odds = bet_data.get('decimal_odds', 2.0)
        odds_risk = 0.0
        if odds < 1.5:  # Heavy favorite
            odds_risk = 0.2
        elif odds > 5.0:  # Heavy underdog
            odds_risk = 0.3
        
        # Risk from data quality
        data_quality = bet_data.get('data_quality_score', 0.8)
        data_risk = 1.0 - data_quality
        
        # Combine risks (weighted average)
        total_risk = (
            0.5 * base_risk +
            0.3 * odds_risk +
            0.2 * data_risk
        )
        
        return min(1.0, total_risk)
    
    async def _calculate_correlation_risk(self, bet_data: Dict[str, Any]) -> float:
        """Calculate correlation risk with existing positions"""
        
        if not self.current_positions:
            return 0.0
        
        # Simplified correlation calculation
        # In practice, would use historical correlation matrix
        
        same_event_positions = 0
        for position in self.current_positions.values():
            # Check if same event (simplified check)
            if position.correlation_group and bet_data.get('event_name'):
                if position.correlation_group == bet_data.get('event_name'):
                    same_event_positions += 1
        
        # Correlation penalty increases with same-event positions
        correlation_risk = min(0.5, same_event_positions * 0.1)
        
        return correlation_risk
    
    def _get_correlation_group(self, recommendation: BetRecommendation) -> str:
        """Get correlation group for a bet recommendation"""
        # Simplified implementation - use fight_id prefix as correlation group
        if recommendation.fight_id:
            # Extract event identifier (e.g., "UFC_301" from "UFC_301_Fight_1")
            parts = recommendation.fight_id.split('_')
            if len(parts) >= 2:
                return '_'.join(parts[:2])
        
        return "general"
    
    # === Risk Monitoring ===
    
    async def check_risk_limits(self, 
                              proposed_bets: List[BetRecommendation]) -> Dict[str, Any]:
        """
        Check if proposed bets violate risk limits
        
        Args:
            proposed_bets: List of proposed bet recommendations
            
        Returns:
            Risk assessment with warnings and approvals
        """
        try:
            async with self.track_operation("check_risk_limits"):
                risk_assessment = {
                    'approved_bets': [],
                    'rejected_bets': [],
                    'warnings': [],
                    'risk_metrics': {},
                    'within_limits': True
                }
                
                current_exposure = sum(pos.stake for pos in self.current_positions.values())
                total_proposed_stake = sum(bet.recommended_stake for bet in proposed_bets)
                
                # Check portfolio exposure
                total_exposure = current_exposure + total_proposed_stake
                exposure_ratio = total_exposure / self.bankroll
                
                if exposure_ratio > self.risk_limits.max_portfolio_exposure:
                    risk_assessment['within_limits'] = False
                    risk_assessment['warnings'].append(
                        f"Portfolio exposure {exposure_ratio:.1%} exceeds limit {self.risk_limits.max_portfolio_exposure:.1%}"
                    )
                
                # Check individual bet limits
                for bet in proposed_bets:
                    bet_approved = True
                    bet_warnings = []
                    
                    # Check individual bet size
                    bet_ratio = bet.recommended_stake / self.bankroll
                    if bet_ratio > self.risk_limits.max_single_bet_percentage:
                        bet_approved = False
                        bet_warnings.append(f"Bet size {bet_ratio:.1%} exceeds limit {self.risk_limits.max_single_bet_percentage:.1%}")
                    
                    # Check expected value
                    if bet.expected_value < self.risk_limits.min_expected_value:
                        bet_approved = False
                        bet_warnings.append(f"Expected value {bet.expected_value:.1%} below minimum {self.risk_limits.min_expected_value:.1%}")
                    
                    # Check uncertainty
                    if bet.uncertainty > self.risk_limits.max_uncertainty_threshold:
                        bet_approved = False
                        bet_warnings.append(f"Uncertainty {bet.uncertainty:.1%} exceeds threshold {self.risk_limits.max_uncertainty_threshold:.1%}")
                    
                    if bet_approved:
                        risk_assessment['approved_bets'].append(bet)
                    else:
                        risk_assessment['rejected_bets'].append({
                            'bet': bet,
                            'warnings': bet_warnings
                        })
                
                # Calculate risk metrics
                risk_assessment['risk_metrics'] = {
                    'current_exposure_ratio': current_exposure / self.bankroll,
                    'proposed_exposure_ratio': exposure_ratio,
                    'diversification_score': self._calculate_diversification_score(),
                    'correlation_penalty': self._calculate_portfolio_correlation_penalty(),
                    'estimated_var_95': self._calculate_portfolio_var()
                }
                
                return risk_assessment
                
        except Exception as e:
            logger.error(f"Risk limit check failed: {e}")
            return {'approved_bets': [], 'rejected_bets': [], 'warnings': [str(e)], 'within_limits': False}
    
    def _calculate_diversification_score(self) -> float:
        """Calculate portfolio diversification score"""
        if not self.current_positions:
            return 1.0
        
        # Simple diversification measure based on position concentration
        total_stake = sum(pos.stake for pos in self.current_positions.values())
        if total_stake == 0:
            return 1.0
        
        # Calculate Herfindahl index
        position_weights = [pos.stake / total_stake for pos in self.current_positions.values()]
        herfindahl_index = sum(w ** 2 for w in position_weights)
        
        # Convert to diversification score (1 = perfectly diversified, 0 = concentrated)
        max_herfindahl = 1.0  # All weight in one position
        diversification_score = 1.0 - (herfindahl_index - (1.0 / len(position_weights))) / (max_herfindahl - (1.0 / len(position_weights)))
        
        return max(0.0, min(1.0, diversification_score))
    
    def _calculate_portfolio_correlation_penalty(self) -> float:
        """Calculate portfolio correlation penalty"""
        if not self.current_positions:
            return 0.0
        
        # Simplified correlation penalty based on same-event exposure
        correlation_groups = {}
        total_stake = sum(pos.stake for pos in self.current_positions.values())
        
        for position in self.current_positions.values():
            group = position.correlation_group or "general"
            correlation_groups[group] = correlation_groups.get(group, 0.0) + position.stake
        
        # Calculate penalty based on concentration in correlation groups
        max_group_weight = max(stake / total_stake for stake in correlation_groups.values()) if total_stake > 0 else 0.0
        
        # Penalty increases quadratically with concentration
        correlation_penalty = max_group_weight ** 2
        
        return correlation_penalty
    
    def _calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk"""
        if not self.current_positions:
            return 0.0
        
        # Simplified VaR calculation
        # In practice, would use historical returns and correlation matrix
        
        total_stake = sum(pos.stake for pos in self.current_positions.values())
        
        # Assume normal distribution with estimated portfolio volatility
        portfolio_volatility = 0.3  # 30% volatility assumption
        
        # VaR calculation
        z_score = norm.ppf(1 - confidence_level)  # 95% confidence
        var_95 = total_stake * portfolio_volatility * abs(z_score)
        
        return var_95
    
    # === Position Management ===
    
    async def update_position(self, 
                            bet_id: str,
                            status: str,
                            outcome: Optional[str] = None,
                            payout: Optional[float] = None):
        """
        Update position status and calculate P&L
        
        Args:
            bet_id: Bet identifier
            status: New status ("won", "lost", "void", "open")
            outcome: Bet outcome details
            payout: Payout amount if won
        """
        try:
            async with self.track_operation("update_position"):
                if bet_id not in self.current_positions:
                    logger.warning(f"Position {bet_id} not found")
                    return
                
                position = self.current_positions[bet_id]
                old_status = position.status
                position.status = status
                
                # Calculate P&L if position is closed
                if status in ["won", "lost", "void"]:
                    if status == "won" and payout:
                        profit = payout - position.stake
                        self.bankroll += payout
                    elif status == "lost":
                        profit = -position.stake
                    else:  # void
                        profit = 0.0
                        self.bankroll += position.stake  # Return stake
                    
                    # Record performance
                    self.portfolio_performance.append({
                        'bet_id': bet_id,
                        'profit': profit,
                        'return_percentage': profit / position.stake if position.stake > 0 else 0.0,
                        'settlement_date': datetime.now(),
                        'holding_period': (datetime.now() - position.entry_time).days
                    })
                    
                    # Move to history
                    self.position_history.append(position)
                    del self.current_positions[bet_id]
                    
                    logger.info(f"Position {bet_id} closed: {status}, P&L: {profit:.2f}")
                
                # Update risk assessment
                await self._update_risk_assessment()
                
        except Exception as e:
            logger.error(f"Position update failed: {e}")
    
    async def add_position(self, bet_recommendation: BetRecommendation):
        """Add new position to portfolio"""
        try:
            position = PortfolioPosition(
                bet_id=f"{bet_recommendation.fight_id}_{bet_recommendation.fighter}",
                fighter=bet_recommendation.fighter,
                stake=bet_recommendation.recommended_stake,
                odds=bet_recommendation.decimal_odds,
                expected_return=bet_recommendation.expected_value * bet_recommendation.recommended_stake,
                entry_time=datetime.now(),
                correlation_group=self._get_correlation_group(bet_recommendation)
            )
            
            self.current_positions[position.bet_id] = position
            self.bankroll -= position.stake  # Deduct stake from bankroll
            
            logger.info(f"Added position: {position.bet_id}, stake: {position.stake:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to add position: {e}")
    
    # === Performance Analysis ===
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            if not self.current_positions and not self.portfolio_performance:
                return PortfolioMetrics(
                    total_stake=0.0,
                    expected_return=0.0,
                    expected_variance=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown_risk=0.0,
                    bankroll_utilization=0.0,
                    diversification_score=1.0,
                    correlation_penalty=0.0,
                    risk_adjusted_ev=0.0,
                    value_at_risk_95=0.0,
                    conditional_var_95=0.0
                )
            
            # Current portfolio metrics
            total_stake = sum(pos.stake for pos in self.current_positions.values())
            expected_return = sum(pos.expected_return for pos in self.current_positions.values())
            
            # Portfolio variance (simplified)
            position_variances = []
            for pos in self.current_positions.values():
                # Estimate variance based on odds
                prob_win = 1.0 / pos.odds
                variance = pos.stake ** 2 * prob_win * (1 - prob_win)
                position_variances.append(variance)
            
            expected_variance = sum(position_variances)
            
            # Sharpe ratio (simplified)
            expected_std = math.sqrt(expected_variance) if expected_variance > 0 else 0.0
            sharpe_ratio = expected_return / expected_std if expected_std > 0 else 0.0
            
            # Other metrics
            bankroll_utilization = total_stake / self.bankroll
            diversification_score = self._calculate_diversification_score()
            correlation_penalty = self._calculate_portfolio_correlation_penalty()
            var_95 = self._calculate_portfolio_var()
            
            # Risk-adjusted expected value
            risk_adjusted_ev = expected_return * (1.0 - correlation_penalty) * diversification_score
            
            # CVaR (simplified)
            conditional_var_95 = var_95 * 1.3  # Approximation
            
            return PortfolioMetrics(
                total_stake=total_stake,
                expected_return=expected_return,
                expected_variance=expected_variance,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_risk=self._estimate_max_drawdown_risk(),
                bankroll_utilization=bankroll_utilization,
                diversification_score=diversification_score,
                correlation_penalty=correlation_penalty,
                risk_adjusted_ev=risk_adjusted_ev,
                value_at_risk_95=var_95,
                conditional_var_95=conditional_var_95
            )
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return PortfolioMetrics(
                total_stake=0.0, expected_return=0.0, expected_variance=0.0,
                sharpe_ratio=0.0, max_drawdown_risk=0.0, bankroll_utilization=0.0,
                diversification_score=0.0, correlation_penalty=0.0, risk_adjusted_ev=0.0,
                value_at_risk_95=0.0, conditional_var_95=0.0
            )
    
    def _estimate_max_drawdown_risk(self) -> float:
        """Estimate maximum drawdown risk"""
        if not self.portfolio_performance:
            return 0.0
        
        # Calculate running returns
        returns = [perf['return_percentage'] for perf in self.portfolio_performance[-30:]]  # Last 30 bets
        
        if not returns:
            return 0.0
        
        # Calculate maximum drawdown from recent performance
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        return max_drawdown
    
    # === Message Handlers ===
    
    async def _handle_optimize_portfolio(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle portfolio optimization request"""
        try:
            candidate_bets = payload.get('candidate_bets', [])
            recommendations = await self.optimize_portfolio(candidate_bets)
            
            return {
                'status': 'success',
                'recommendations': [
                    {
                        'fight_id': rec.fight_id,
                        'fighter': rec.fighter,
                        'recommended_stake': rec.recommended_stake,
                        'expected_value': rec.expected_value,
                        'kelly_fraction': rec.kelly_fraction,
                        'risk_score': rec.risk_score
                    }
                    for rec in recommendations
                ],
                'portfolio_metrics': self.calculate_portfolio_metrics().__dict__
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_calculate_kelly_sizing(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Kelly sizing calculation request"""
        try:
            predicted_prob = payload.get('predicted_probability')
            decimal_odds = payload.get('decimal_odds')
            uncertainty = payload.get('uncertainty', 0.0)
            
            if predicted_prob is None or decimal_odds is None:
                return {'status': 'error', 'error': 'Missing required parameters'}
            
            kelly_fraction = self.calculate_kelly_fraction(predicted_prob, decimal_odds, uncertainty)
            
            return {
                'status': 'success',
                'kelly_fraction': kelly_fraction,
                'recommended_stake': kelly_fraction * self.bankroll,
                'max_stake': self.risk_limits.max_single_bet_percentage * self.bankroll
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_check_risk_limits(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk limits check request"""
        try:
            proposed_bets_data = payload.get('proposed_bets', [])
            
            # Convert to BetRecommendation objects
            proposed_bets = []
            for bet_data in proposed_bets_data:
                rec = BetRecommendation(
                    fight_id=bet_data.get('fight_id', ''),
                    fighter=bet_data.get('fighter', ''),
                    opponent=bet_data.get('opponent', ''),
                    predicted_probability=bet_data.get('predicted_probability', 0.5),
                    market_odds=bet_data.get('decimal_odds', 2.0),
                    decimal_odds=bet_data.get('decimal_odds', 2.0),
                    expected_value=bet_data.get('expected_value', 0.0),
                    kelly_fraction=bet_data.get('kelly_fraction', 0.0),
                    recommended_stake=bet_data.get('recommended_stake', 0.0),
                    max_stake=bet_data.get('max_stake', 0.0),
                    confidence_score=bet_data.get('confidence_score', 0.8),
                    uncertainty=bet_data.get('uncertainty', 0.2),
                    risk_score=bet_data.get('risk_score', 0.5),
                    correlation_risk=bet_data.get('correlation_risk', 0.0)
                )
                proposed_bets.append(rec)
            
            risk_assessment = await self.check_risk_limits(proposed_bets)
            
            return {
                'status': 'success',
                'risk_assessment': risk_assessment
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_update_position(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle position update request"""
        try:
            bet_id = payload.get('bet_id')
            status = payload.get('status')
            outcome = payload.get('outcome')
            payout = payload.get('payout')
            
            if not bet_id or not status:
                return {'status': 'error', 'error': 'Missing required parameters'}
            
            await self.update_position(bet_id, status, outcome, payout)
            
            return {
                'status': 'success',
                'updated_position': bet_id,
                'new_status': status,
                'current_bankroll': self.bankroll
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_get_portfolio_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle portfolio metrics request"""
        try:
            metrics = self.calculate_portfolio_metrics()
            
            return {
                'status': 'success',
                'portfolio_metrics': metrics.__dict__,
                'current_positions': len(self.current_positions),
                'bankroll': self.bankroll,
                'performance_history_length': len(self.portfolio_performance)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    # === Background Tasks ===
    
    async def _risk_monitoring_loop(self):
        """Background risk monitoring"""
        logger.info("RiskAgent risk monitoring started")
        
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.risk_monitoring_interval)
                
                if self._stop_event.is_set():
                    break
                
                # Check risk limits
                await self._update_risk_assessment()
                
                # Check circuit breakers
                if self.circuit_breaker_enabled:
                    await self._check_circuit_breakers()
                
            except Exception as e:
                logger.error(f"RiskAgent risk monitoring error: {e}")
        
        logger.info("RiskAgent risk monitoring stopped")
    
    async def _performance_tracking_loop(self):
        """Background performance tracking"""
        logger.info("RiskAgent performance tracking started")
        
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                if self._stop_event.is_set():
                    break
                
                # Calculate performance metrics
                metrics = self.calculate_portfolio_metrics()
                
                # Log performance summary
                logger.info(
                    f"Portfolio update: {metrics.total_stake:.2f} stake, "
                    f"{metrics.expected_return:.2f} expected return, "
                    f"{metrics.sharpe_ratio:.3f} Sharpe ratio"
                )
                
            except Exception as e:
                logger.error(f"RiskAgent performance tracking error: {e}")
        
        logger.info("RiskAgent performance tracking stopped")
    
    async def _update_risk_assessment(self):
        """Update current risk assessment"""
        # This would trigger risk limit checks and alerts
        pass
    
    async def _check_circuit_breakers(self):
        """Check circuit breaker conditions"""
        # Check daily loss limit
        today_performance = [
            perf for perf in self.portfolio_performance
            if perf.get('settlement_date', datetime.now()).date() == datetime.now().date()
        ]
        
        if today_performance:
            daily_pnl = sum(perf['profit'] for perf in today_performance)
            daily_loss_ratio = abs(daily_pnl) / self.bankroll
            
            if daily_pnl < 0 and daily_loss_ratio > self.daily_loss_limit:
                await self.broadcast_message(
                    'circuit_breaker_triggered',
                    {
                        'trigger': 'daily_loss_limit',
                        'daily_loss': daily_pnl,
                        'loss_ratio': daily_loss_ratio,
                        'limit': self.daily_loss_limit
                    }
                )
    
    # === Data Management ===
    
    async def _load_historical_data(self):
        """Load historical performance data"""
        # Implementation would load from persistent storage
        pass
    
    async def _initialize_correlation_matrix(self):
        """Initialize correlation matrix for risk calculations"""
        # Implementation would build correlation matrix from historical data
        pass
    
    async def _perform_initial_risk_assessment(self):
        """Perform initial risk assessment"""
        metrics = self.calculate_portfolio_metrics()
        logger.info(f"Initial portfolio metrics: {metrics}")
    
    async def _save_risk_state(self):
        """Save current risk state for persistence"""
        # Implementation would save to persistent storage
        pass


def create_risk_agent_config(
    bankroll: float,
    risk_limits: Optional[Dict[str, float]] = None,
    kelly_multiplier: float = 0.25,
    circuit_breaker_enabled: bool = True
) -> Dict[str, Any]:
    """
    Factory function for RiskAgent configuration
    
    Args:
        bankroll: Initial bankroll amount
        risk_limits: Risk limit overrides
        kelly_multiplier: Kelly fraction multiplier
        circuit_breaker_enabled: Enable circuit breaker protection
        
    Returns:
        RiskAgent configuration
    """
    default_risk_limits = {
        'max_single_bet_percentage': 0.05,
        'max_portfolio_exposure': 0.25,
        'max_correlated_exposure': 0.15,
        'min_expected_value': 0.05,
        'max_kelly_fraction': 0.25,
        'max_uncertainty_threshold': 0.4,
        'max_drawdown_limit': 0.20,
        'min_diversification_score': 0.6
    }
    
    if risk_limits:
        default_risk_limits.update(risk_limits)
    
    return {
        'bankroll': bankroll,
        'kelly_multiplier': kelly_multiplier,
        'risk_limits': default_risk_limits,
        'circuit_breaker_enabled': circuit_breaker_enabled,
        'daily_loss_limit': 0.1,  # 10%
        'risk_monitoring_interval': 300,  # 5 minutes
        'performance_window': 30,  # 30 days
        'correlation_lookback': 100,  # 100 historical fights
        'risk_adjustment_frequency': 7  # 7 days
    }
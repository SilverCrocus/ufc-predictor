#!/usr/bin/env python3
"""
Main integration script for enhanced UFC predictor pipeline.
Combines all improvement components into a unified workflow.
"""

import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import evaluation components
from src.ufc_predictor.evaluation.temporal_split import TemporalWalkForwardSplitter
from src.ufc_predictor.evaluation.calibration import UFCProbabilityCalibrator
from src.ufc_predictor.evaluation.metrics import UFCMetricsCalculator

# Import feature engineering
from src.ufc_predictor.features.context_features import ContextFeatureGenerator
from src.ufc_predictor.features.matchup_features import MatchupFeatureGenerator
from src.ufc_predictor.features.quality_adjustment import QualityAdjustmentFeatureGenerator
from src.ufc_predictor.features.rolling_profiles import RollingProfileGenerator

# Import models
from src.ufc_predictor.models.stacking import TemporalStackingEnsemble
from src.ufc_predictor.models.bradley_terry import BradleyTerryModel
from src.ufc_predictor.models.survival_moi import CompetingRisksModel

# Import betting components
from src.ufc_predictor.betting.simulator import WalkForwardSimulator
from src.ufc_predictor.betting.staking import KellyStaking
from src.ufc_predictor.betting.correlation import ParlayCorrelationEstimator

# Import monitoring
from src.ufc_predictor.monitoring.dashboards import PerformanceDashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedUFCPipeline:
    """
    Complete enhanced UFC prediction pipeline with all improvements.
    """
    
    def __init__(self, feature_config_path: str, backtest_config_path: str):
        """
        Initialize enhanced pipeline.
        
        Args:
            feature_config_path: Path to features.yaml
            backtest_config_path: Path to backtest.yaml
        """
        # Load configurations
        with open(feature_config_path, 'r') as f:
            self.feature_config = yaml.safe_load(f)
        
        with open(backtest_config_path, 'r') as f:
            self.backtest_config = yaml.safe_load(f)
        
        # Setup output directories BEFORE initializing components
        self.output_dir = Path('artifacts') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Enhanced UFC Pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Feature generators - filter out 'enabled' key
        context_config = {k: v for k, v in self.feature_config.get('context', {}).items() if k != 'enabled'}
        self.context_gen = ContextFeatureGenerator(**context_config)
        
        matchup_config = {k: v for k, v in self.feature_config.get('matchup', {}).items() if k != 'enabled'}
        self.matchup_gen = MatchupFeatureGenerator(**matchup_config)
        
        quality_config = {k: v for k, v in self.feature_config.get('quality_adjustment', {}).items() if k != 'enabled'}
        self.quality_gen = QualityAdjustmentFeatureGenerator(**quality_config)
        
        rolling_config = {k: v for k, v in self.feature_config.get('rolling', {}).items() if k != 'enabled'}
        self.rolling_gen = RollingProfileGenerator(**rolling_config)
        
        # Evaluation components
        self.temporal_splitter = TemporalWalkForwardSplitter(
            **self.backtest_config.get('folding', {})
        )
        self.calibrator = UFCProbabilityCalibrator()
        self.metrics_calc = UFCMetricsCalculator()
        
        # Models
        self.stacking_ensemble = None  # Will be created during training
        self.bradley_terry = BradleyTerryModel()
        self.method_predictor = CompetingRisksModel()
        
        # Betting components
        self.simulator = WalkForwardSimulator()  # Will use default config
        
        # Map config keys to KellyStaking parameters
        staking_params = {}
        staking_cfg = self.backtest_config.get('staking', {})
        if 'kelly_fraction' in staking_cfg:
            staking_params['kelly_fraction'] = staking_cfg['kelly_fraction']
        if 'max_single_pct' in staking_cfg:
            staking_params['max_bet_pct'] = staking_cfg['max_single_pct']
        if 'min_bet_pct' in staking_cfg:
            staking_params['min_bet_pct'] = staking_cfg['min_bet_pct']
        if 'p_lower_quantile' in staking_cfg:
            staking_params['confidence_level'] = staking_cfg['p_lower_quantile']
        
        self.kelly_staking = KellyStaking(**staking_params)
        self.parlay_estimator = ParlayCorrelationEstimator()
        
        # Monitoring
        self.dashboard = PerformanceDashboard(
            output_dir=str(self.output_dir / 'monitoring')
        )
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and prepare UFC fight data.
        
        Args:
            data_path: Path to fight data CSV
            
        Returns:
            Prepared DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Ensure required columns exist
        required_cols = ['date', 'fighter_a', 'fighter_b', 'winner']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} fights from {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def generate_enhanced_features(
        self,
        df: pd.DataFrame,
        historical_fights: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate all enhanced features.
        
        Args:
            df: Fight data
            historical_fights: Historical data for quality adjustment
            
        Returns:
            DataFrame with enhanced features
        """
        logger.info("Generating enhanced features")
        
        # Context features
        if self.feature_config['context']['enabled']:
            df = self.context_gen.generate_features(df)
            logger.info(f"Added {len(self.context_gen.get_feature_names())} context features")
        
        # Matchup features
        if self.feature_config['matchup']['enabled']:
            df = self.matchup_gen.generate_features(df)
            logger.info(f"Added {len(self.matchup_gen.get_feature_names())} matchup features")
        
        # Quality adjustment (requires historical data)
        if self.feature_config['quality_adjustment']['enabled'] and historical_fights is not None:
            # Fit Bradley-Terry ratings
            self.quality_gen.fit_ratings(historical_fights)
            
            # Generate quality-adjusted features
            fighter_stats = df  # In production, would have separate stats DataFrame
            df = self.quality_gen.generate_features(df, fighter_stats)
            logger.info("Added quality-adjusted features")
        
        # Rolling profiles
        if self.feature_config['rolling']['enabled'] and historical_fights is not None:
            df = self.rolling_gen.generate_features(
                df, historical_fights, 'fighter_a', 'date'
            )
            df = self.rolling_gen.generate_features(
                df, historical_fights, 'fighter_b', 'date'
            )
            logger.info("Added rolling profile features")
        
        return df
    
    def train_models(
        self,
        train_data: pd.DataFrame,
        feature_cols: List[str]
    ) -> None:
        """
        Train all models including stacking ensemble.
        
        Args:
            train_data: Training data with features
            feature_cols: List of feature column names
        """
        logger.info("Training models")
        
        # Prepare data
        X = train_data[feature_cols].values
        # Ensure winner is binary integer (handle NaN by dropping)
        winner_mask = train_data['winner'].notna()
        X = X[winner_mask]
        y = train_data.loc[winner_mask, 'winner'].astype(int).values
        
        # Train stacking ensemble
        self.stacking_ensemble = TemporalStackingEnsemble(
            n_splits=5,
            use_probas=True,
            use_features_in_meta=False
        )
        
        if 'date' in train_data.columns:
            self.stacking_ensemble.fit_with_dates(
                train_data.loc[winner_mask, feature_cols],
                y,
                train_data.loc[winner_mask, 'date']
            )
        else:
            self.stacking_ensemble.fit(X, y)
        
        logger.info(f"Trained stacking ensemble with CV scores: {self.stacking_ensemble.cv_scores_}")
        
        # Train method predictor if method data available
        if 'method' in train_data.columns and len(train_data) > 100:  # Need enough samples
            try:
                # Use same filtered data as for main model
                method_data = train_data.loc[winner_mask, 'method']
                finish_round = train_data.loc[winner_mask, 'round'] if 'round' in train_data.columns else None
                
                # Replace NaN in features with 0 for method predictor
                X_method = train_data.loc[winner_mask, feature_cols].fillna(0).values
                
                # Check if we have enough method diversity
                unique_methods = method_data.nunique()
                if unique_methods >= 2 and len(method_data) >= 20:
                    self.method_predictor.fit(
                        X_method,
                        method_data,
                        finish_round
                    )
                    logger.info("Trained method predictor")
                else:
                    logger.info(f"Skipping method predictor - insufficient diversity ({unique_methods} methods, {len(method_data)} samples)")
            except Exception as e:
                logger.warning(f"Could not train method predictor: {e}")
    
    def run_walk_forward_evaluation(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict:
        """
        Run complete walk-forward evaluation.
        
        Args:
            df: Complete dataset
            feature_cols: Feature columns to use
            
        Returns:
            Evaluation results
        """
        logger.info("Running walk-forward evaluation")
        
        # Create temporal folds
        folds = self.temporal_splitter.make_rolling_folds(df)
        
        results = {
            'fold_metrics': [],
            'predictions': [],
            'calibration_results': []
        }
        
        for fold_idx, fold in enumerate(folds):
            logger.info(f"Processing fold {fold_idx + 1}/{len(folds)}")
            
            # Split data
            train_data = df.iloc[fold.train_indices]
            test_data = df.iloc[fold.test_indices]
            
            # Generate features
            train_enhanced = self.generate_enhanced_features(train_data, train_data)
            test_enhanced = self.generate_enhanced_features(test_data, train_data)
            
            # Get feature columns (exclude target and metadata)
            # Only include numeric columns that are either in feature_cols or end with '_feature'
            exclude_cols = {'fighter_a', 'fighter_b', 'winner', 'date', 'event', 
                           'method', 'round', 'time', 'venue', 'division', 'event_id', 
                           'fight_number', 'fighter_url', 'opponent_url'}
            
            # Convert boolean columns to int
            for col in train_enhanced.columns:
                if train_enhanced[col].dtype == 'bool':
                    train_enhanced[col] = train_enhanced[col].astype(int)
                    test_enhanced[col] = test_enhanced[col].astype(int)
                elif train_enhanced[col].dtype == 'object' and col.endswith('_feature'):
                    # Try to convert object features to numeric if possible
                    try:
                        train_enhanced[col] = pd.to_numeric(train_enhanced[col], errors='coerce')
                        test_enhanced[col] = pd.to_numeric(test_enhanced[col], errors='coerce')
                    except:
                        pass
            
            actual_feature_cols = [col for col in train_enhanced.columns 
                                  if col not in exclude_cols
                                  and train_enhanced[col].dtype in ['float64', 'int64', 'float32', 'int32', 'bool']]
            
            # Train models
            self.train_models(train_enhanced, actual_feature_cols)
            
            # Generate predictions
            X_test = test_enhanced[actual_feature_cols]
            # Filter out draws (0.5) and NaN values for binary classification
            valid_mask = test_enhanced['winner'].notna() & (test_enhanced['winner'] != 0.5)
            X_test = X_test[valid_mask]
            y_test = test_enhanced.loc[valid_mask, 'winner'].astype(int).values
            
            predictions = self.stacking_ensemble.predict_proba(X_test)[:, 1]
            
            # Calibrate predictions
            self.calibrator.fit_isotonic_by_segment(
                pd.DataFrame({'prob_pred': predictions, 'winner': y_test})
            )
            calibrated_predictions = self.calibrator.apply_calibration(
                pd.DataFrame({'prob_pred': predictions})
            )
            
            # Calculate metrics
            fold_metrics = self.metrics_calc.calculate_all_metrics(
                y_test,
                calibrated_predictions,
                test_enhanced[['odds_a', 'odds_b']] if 'odds_a' in test_enhanced else None
            )
            
            results['fold_metrics'].append({
                'fold_id': fold_idx,
                **fold_metrics.classification_metrics,
                **fold_metrics.calibration_metrics
            })
            
            results['predictions'].append({
                'fold_id': fold_idx,
                'predictions': predictions,
                'calibrated': calibrated_predictions,
                'actuals': y_test
            })
            
            results['calibration_results'].append(
                self.calibrator.get_calibration_summary()
            )
        
        # Aggregate results
        results['overall_metrics'] = self._aggregate_fold_metrics(results['fold_metrics'])
        
        return results
    
    def run_backtesting_simulation(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict:
        """
        Run complete backtesting simulation.
        
        Args:
            df: Complete dataset with odds
            feature_cols: Feature columns
            
        Returns:
            Backtest results
        """
        logger.info("Running backtesting simulation")
        
        # Create simple model pipeline for simulator
        class ModelPipeline:
            def __init__(self, pipeline_obj):
                self.pipeline = pipeline_obj
            
            def fit(self, data):
                X = data[feature_cols].values if feature_cols else data.values
                y = data['winner'].values
                self.pipeline.train_models(data, feature_cols)
                return self
            
            def predict_proba(self, data):
                X = data[feature_cols].values if feature_cols else data.values
                return self.pipeline.stacking_ensemble.predict_proba(X)
        
        model_pipeline = ModelPipeline(self)
        
        # Run backtest
        backtest_result = self.simulator.run_backtest(
            df,
            model_pipeline,
            feature_pipeline=None  # Features generated inside pipeline
        )
        
        return backtest_result
    
    def generate_reports(self, evaluation_results: Dict, backtest_results: Dict):
        """
        Generate comprehensive reports and visualizations.
        
        Args:
            evaluation_results: Walk-forward evaluation results
            backtest_results: Backtesting simulation results
        """
        logger.info("Generating reports and visualizations")
        
        # Create calibration plots
        if evaluation_results.get('predictions'):
            all_preds = np.concatenate([p['calibrated'] for p in evaluation_results['predictions']])
            all_actuals = np.concatenate([p['actuals'] for p in evaluation_results['predictions']])
            
            calibration_metrics = self.dashboard.create_calibration_plots(
                all_actuals,
                all_preds,
                title="Overall Calibration Analysis"
            )
            logger.info(f"Calibration plots saved. ECE: {calibration_metrics['ece']:.4f}")
        
        # Create betting performance plots
        if backtest_results:
            betting_plots = self.dashboard.create_betting_performance_plots(
                backtest_results,
                title="Betting Performance Analysis"
            )
            logger.info(f"Betting plots saved to {betting_plots['plot_path']}")
        
        # Create CLV analysis if available
        if backtest_results and hasattr(backtest_results, 'bet_history'):
            clv_metrics = self.dashboard.create_clv_analysis_plots(
                backtest_results.bet_history,
                title="Closing Line Value Analysis"
            )
            logger.info(f"CLV win rate: {clv_metrics['clv_win_rate']:.2%}")
        
        # Generate summary report
        summary_metrics = {
            **evaluation_results.get('overall_metrics', {}),
            **(backtest_results.performance_metrics if backtest_results else {})
        }
        
        report_path = self.dashboard.generate_summary_report(
            summary_metrics,
            output_format='json'
        )
        logger.info(f"Summary report saved to {report_path}")
        
        # Save detailed results
        self._save_detailed_results(evaluation_results, backtest_results)
    
    def _aggregate_fold_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across folds."""
        if not fold_metrics:
            return {}
        
        df = pd.DataFrame(fold_metrics)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        aggregated = {}
        for col in numeric_cols:
            aggregated[f'{col}_mean'] = df[col].mean()
            aggregated[f'{col}_std'] = df[col].std()
        
        return aggregated
    
    def _save_detailed_results(self, evaluation_results: Dict, backtest_results: Dict):
        """Save detailed results to files."""
        # Save evaluation results
        eval_path = self.output_dir / 'evaluation_results.json'
        with open(eval_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            eval_save = {}
            for key, value in evaluation_results.items():
                if key == 'predictions':
                    eval_save[key] = [
                        {
                            'fold_id': p['fold_id'],
                            'n_predictions': len(p['predictions'])
                        }
                        for p in value
                    ]
                else:
                    eval_save[key] = value
            
            import json
            json.dump(eval_save, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {eval_path}")
        
        # Save backtest results if available
        if backtest_results:
            backtest_path = self.output_dir / 'backtest_results.json'
            with open(backtest_path, 'w') as f:
                backtest_save = {
                    'performance_metrics': backtest_results.performance_metrics,
                    'n_bets': len(backtest_results.bet_history),
                    'config': backtest_results.config,
                    'metadata': backtest_results.metadata
                }
                json.dump(backtest_save, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {backtest_path}")


def main():
    """Main entry point for enhanced pipeline."""
    parser = argparse.ArgumentParser(description='Run enhanced UFC predictor pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to fight data CSV')
    parser.add_argument('--features-config', type=str, default='configs/features.yaml',
                       help='Path to features configuration')
    parser.add_argument('--backtest-config', type=str, default='configs/backtest.yaml',
                       help='Path to backtest configuration')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['evaluation', 'backtest', 'full'],
                       help='Execution mode')
    parser.add_argument('--feature-cols', type=str, nargs='+',
                       help='Specific feature columns to use')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EnhancedUFCPipeline(args.features_config, args.backtest_config)
    
    # Load data
    df = pipeline.load_data(args.data)
    
    # Default feature columns if not specified
    if not args.feature_cols:
        # Feature columns will be determined after feature generation
        feature_cols = []
    else:
        feature_cols = args.feature_cols
    
    # Run based on mode
    evaluation_results = {}
    backtest_results = None
    
    if args.mode in ['evaluation', 'full']:
        evaluation_results = pipeline.run_walk_forward_evaluation(df, feature_cols)
        logger.info("Walk-forward evaluation complete")
    
    if args.mode in ['backtest', 'full']:
        # Only run backtest if odds data is available
        if 'odds_a' in df.columns or 'odds' in df.columns:
            backtest_results = pipeline.run_backtesting_simulation(df, feature_cols)
            logger.info("Backtesting simulation complete")
        else:
            logger.warning("No odds data found, skipping backtesting")
    
    # Generate reports
    pipeline.generate_reports(evaluation_results, backtest_results)
    
    logger.info(f"Pipeline complete. Results saved to {pipeline.output_dir}")


if __name__ == '__main__':
    main()
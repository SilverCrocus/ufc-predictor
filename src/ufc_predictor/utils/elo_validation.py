"""
ELO System Validation and Testing Framework

This module provides comprehensive validation tools for the UFC ELO rating system,
including backtesting, calibration analysis, and performance benchmarking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import logging

from .ufc_elo_system import UFCELOSystem, ELOConfig
from .multi_dimensional_elo import MultiDimensionalUFCELO
from .elo_historical_processor import UFCHistoricalProcessor
from .elo_integration import ELOIntegration


class ELOValidator:
    """
    Comprehensive validation framework for UFC ELO systems
    """
    
    def __init__(self, elo_system: UFCELOSystem = None, config: ELOConfig = None):
        self.elo_system = elo_system
        self.config = config or ELOConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validation metrics storage
        self.validation_results = {}
        self.backtest_results = []
        
    def split_data_chronologically(self, 
                                 fights_df: pd.DataFrame,
                                 train_ratio: float = 0.7,
                                 validation_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split fight data chronologically for proper time-series validation
        
        Args:
            fights_df: Historical fights DataFrame
            train_ratio: Proportion of data for training
            validation_ratio: Proportion of data for validation
            
        Returns:
            Tuple of (train_df, validation_df, test_df)
        """
        # Ensure fights are sorted by date
        if 'Date' in fights_df.columns:
            fights_df = fights_df.sort_values('Date')
        
        total_fights = len(fights_df)
        train_end = int(total_fights * train_ratio)
        val_end = int(total_fights * (train_ratio + validation_ratio))
        
        train_df = fights_df.iloc[:train_end].copy()
        val_df = fights_df.iloc[train_end:val_end].copy()
        test_df = fights_df.iloc[val_end:].copy()
        
        self.logger.info(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
        
        return train_df, val_df, test_df
    
    def cross_validation_backtest(self,
                                 fights_df: pd.DataFrame,
                                 fighters_df: pd.DataFrame = None,
                                 n_folds: int = 5,
                                 min_train_period_days: int = 365) -> Dict:
        """
        Perform time-series cross-validation backtesting
        
        Args:
            fights_df: Historical fights data
            fighters_df: Fighter information data
            n_folds: Number of validation folds
            min_train_period_days: Minimum training period in days
            
        Returns:
            Cross-validation results
        """
        self.logger.info(f"Starting {n_folds}-fold cross-validation backtest...")
        
        # Ensure chronological order
        if 'Date' in fights_df.columns:
            fights_df['Date'] = pd.to_datetime(fights_df['Date'])
            fights_df = fights_df.sort_values('Date')
            
            start_date = fights_df['Date'].min()
            end_date = fights_df['Date'].max()
            total_period = (end_date - start_date).days
            
            fold_results = []
            
            for fold in range(n_folds):
                # Calculate fold boundaries
                fold_start = start_date + timedelta(days=fold * total_period // n_folds)
                train_end = fold_start + timedelta(days=min_train_period_days)
                test_start = train_end
                test_end = fold_start + timedelta(days=(fold + 1) * total_period // n_folds)
                
                # Skip if insufficient data
                if test_start >= end_date:
                    continue
                
                # Split data
                train_data = fights_df[fights_df['Date'] < train_end]
                test_data = fights_df[
                    (fights_df['Date'] >= test_start) & 
                    (fights_df['Date'] < test_end)
                ]
                
                if len(train_data) < 100 or len(test_data) < 20:
                    continue
                
                self.logger.info(f"Fold {fold + 1}: Training on {len(train_data)} fights, testing on {len(test_data)}")
                
                # Build ELO system for this fold
                fold_elo_system = MultiDimensionalUFCELO(self.config)
                processor = UFCHistoricalProcessor(fold_elo_system)
                
                # Train ELO system
                processor.build_elo_from_history(train_data, fighters_df)
                
                # Validate on test data
                fold_metrics = self.validate_predictions(fold_elo_system, test_data)
                fold_metrics['fold'] = fold + 1
                fold_metrics['train_size'] = len(train_data)
                fold_metrics['test_size'] = len(test_data)
                fold_metrics['train_period'] = (train_end - fold_start).days
                
                fold_results.append(fold_metrics)
            
            # Aggregate results
            cv_results = self._aggregate_cv_results(fold_results)
            self.validation_results['cross_validation'] = cv_results
            
            return cv_results
        
        else:
            self.logger.error("Date column required for cross-validation")
            return {}
    
    def validate_predictions(self, 
                           elo_system: UFCELOSystem,
                           test_fights_df: pd.DataFrame,
                           probability_bins: int = 10) -> Dict:
        """
        Validate ELO predictions against test data
        
        Args:
            elo_system: Trained ELO system
            test_fights_df: Test fights data
            probability_bins: Number of bins for calibration analysis
            
        Returns:
            Comprehensive validation metrics
        """
        predictions = []
        actual_outcomes = []
        prediction_probs = []
        
        processor = UFCHistoricalProcessor(elo_system)
        processed_fights = processor.process_fights_dataframe(test_fights_df)
        
        for fight in processed_fights:
            if fight['winner'] is None:
                continue
            
            try:
                prediction = elo_system.predict_fight_outcome(
                    fight['fighter1'], fight['fighter2'], include_uncertainty=True
                )
                
                if 'error' in prediction:
                    continue
                
                # Get prediction probability
                prob_fighter1_wins = prediction['fighter1_win_prob']
                prediction_probs.append(prob_fighter1_wins)
                
                # Binary prediction (>0.5 threshold)
                predicted_winner = 1 if prob_fighter1_wins > 0.5 else 0
                actual_winner = 1 if fight['winner'] == fight['fighter1'] else 0
                
                predictions.append(predicted_winner)
                actual_outcomes.append(actual_winner)
                
            except Exception as e:
                self.logger.warning(f"Error in prediction validation: {e}")
                continue
        
        if len(predictions) == 0:
            return {'error': 'No valid predictions generated'}
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actual_outcomes = np.array(actual_outcomes)
        prediction_probs = np.array(prediction_probs)
        
        # Calculate metrics
        accuracy = np.mean(predictions == actual_outcomes)
        
        # Calibration metrics
        brier_score = brier_score_loss(actual_outcomes, prediction_probs)
        
        try:
            log_loss_score = log_loss(actual_outcomes, prediction_probs)
            auc_score = roc_auc_score(actual_outcomes, prediction_probs)
        except ValueError:
            log_loss_score = float('inf')
            auc_score = 0.5
        
        # Calibration curve
        fraction_pos, mean_predicted_prob = calibration_curve(
            actual_outcomes, prediction_probs, n_bins=probability_bins, strategy='uniform'
        )
        
        # Reliability diagram data
        calibration_error = np.mean(np.abs(fraction_pos - mean_predicted_prob))
        
        # Confidence analysis
        high_conf_mask = (prediction_probs > 0.7) | (prediction_probs < 0.3)
        high_conf_accuracy = np.mean(predictions[high_conf_mask] == actual_outcomes[high_conf_mask]) if np.sum(high_conf_mask) > 0 else accuracy
        
        medium_conf_mask = (prediction_probs >= 0.4) & (prediction_probs <= 0.6)
        medium_conf_accuracy = np.mean(predictions[medium_conf_mask] == actual_outcomes[medium_conf_mask]) if np.sum(medium_conf_mask) > 0 else accuracy
        
        # Method-specific analysis (if available)
        method_analysis = self._analyze_method_predictions(elo_system, processed_fights)
        
        validation_metrics = {
            'total_predictions': len(predictions),
            'accuracy': accuracy,
            'brier_score': brier_score,
            'log_loss': log_loss_score,
            'auc_score': auc_score,
            'calibration_error': calibration_error,
            'high_confidence_accuracy': high_conf_accuracy,
            'medium_confidence_accuracy': medium_conf_accuracy,
            'high_confidence_predictions': np.sum(high_conf_mask),
            'calibration_curve': {
                'fraction_positive': fraction_pos.tolist(),
                'mean_predicted_prob': mean_predicted_prob.tolist()
            },
            'method_analysis': method_analysis
        }
        
        return validation_metrics
    
    def _analyze_method_predictions(self, 
                                  elo_system: UFCELOSystem,
                                  processed_fights: List[Dict]) -> Dict:
        """Analyze prediction accuracy by fight method"""
        method_analysis = {}
        
        if not hasattr(elo_system, 'predict_fight_outcome'):
            return method_analysis
        
        for method_type in ['KO', 'TKO', 'Submission', 'Decision']:
            method_fights = [f for f in processed_fights if method_type.upper() in f['method'].upper()]
            
            if len(method_fights) < 5:  # Minimum sample size
                continue
            
            correct_predictions = 0
            total_predictions = 0
            
            for fight in method_fights:
                try:
                    prediction = elo_system.predict_fight_outcome(
                        fight['fighter1'], fight['fighter2']
                    )
                    
                    if 'error' in prediction:
                        continue
                    
                    predicted_winner = fight['fighter1'] if prediction['fighter1_win_prob'] > 0.5 else fight['fighter2']
                    
                    if predicted_winner == fight['winner']:
                        correct_predictions += 1
                    total_predictions += 1
                    
                except:
                    continue
            
            if total_predictions > 0:
                method_analysis[method_type] = {
                    'accuracy': correct_predictions / total_predictions,
                    'sample_size': total_predictions
                }
        
        return method_analysis
    
    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate cross-validation results across folds"""
        if not fold_results:
            return {}
        
        # Extract metrics
        metrics = ['accuracy', 'brier_score', 'log_loss', 'auc_score', 'calibration_error']
        
        aggregated = {}
        for metric in metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
        
        aggregated['n_folds'] = len(fold_results)
        aggregated['total_predictions'] = sum(fold['total_predictions'] for fold in fold_results)
        
        return aggregated
    
    def compare_elo_configurations(self,
                                 fights_df: pd.DataFrame,
                                 fighters_df: pd.DataFrame = None,
                                 config_variants: List[ELOConfig] = None) -> Dict:
        """
        Compare different ELO configuration variants
        
        Args:
            fights_df: Historical fights data
            fighters_df: Fighter information
            config_variants: List of ELO configurations to compare
            
        Returns:
            Comparison results
        """
        if config_variants is None:
            # Default configuration variants
            config_variants = [
                ELOConfig(base_k_factor=32, initial_rating=1400),  # Standard
                ELOConfig(base_k_factor=24, initial_rating=1400),  # Conservative
                ELOConfig(base_k_factor=40, initial_rating=1400),  # Aggressive
                ELOConfig(base_k_factor=32, initial_rating=1500),  # Higher initial
                ELOConfig(base_k_factor=32, initial_rating=1300),  # Lower initial
            ]
        
        # Split data
        train_df, val_df, test_df = self.split_data_chronologically(fights_df)
        
        comparison_results = {}
        
        for i, config in enumerate(config_variants):
            self.logger.info(f"Testing configuration {i + 1}/{len(config_variants)}")
            
            # Build ELO system with this configuration
            elo_system = MultiDimensionalUFCELO(config)
            processor = UFCHistoricalProcessor(elo_system)
            processor.build_elo_from_history(train_df, fighters_df)
            
            # Validate on test data
            metrics = self.validate_predictions(elo_system, test_df)
            
            config_name = f"k{config.base_k_factor}_init{config.initial_rating}"
            comparison_results[config_name] = {
                'config': {
                    'base_k_factor': config.base_k_factor,
                    'initial_rating': config.initial_rating,
                    'rookie_k_factor': config.rookie_k_factor,
                    'veteran_k_factor': config.veteran_k_factor
                },
                'metrics': metrics
            }
        
        # Find best configuration
        best_config = max(
            comparison_results.keys(),
            key=lambda x: comparison_results[x]['metrics'].get('accuracy', 0)
        )
        
        comparison_summary = {
            'best_configuration': best_config,
            'best_accuracy': comparison_results[best_config]['metrics'].get('accuracy', 0),
            'all_results': comparison_results
        }
        
        return comparison_summary
    
    def benchmark_against_baseline(self,
                                 fights_df: pd.DataFrame,
                                 fighters_df: pd.DataFrame = None) -> Dict:
        """
        Benchmark ELO system against simple baselines
        
        Args:
            fights_df: Historical fights data
            fighters_df: Fighter information
            
        Returns:
            Benchmarking results
        """
        self.logger.info("Benchmarking ELO system against baselines...")
        
        # Split data
        train_df, val_df, test_df = self.split_data_chronologically(fights_df)
        
        # Build ELO system
        elo_system = MultiDimensionalUFCELO(self.config)
        processor = UFCHistoricalProcessor(elo_system)
        processor.build_elo_from_history(train_df, fighters_df)
        
        # Get ELO predictions
        elo_metrics = self.validate_predictions(elo_system, test_df)
        
        # Baseline 1: Random predictions (50% accuracy expected)
        random_accuracy = 0.5
        
        # Baseline 2: Always predict fighter with more wins
        record_based_accuracy = self._calculate_record_based_accuracy(test_df, fighters_df)
        
        # Baseline 3: Betting odds (if available)
        # This would require odds data, so we'll skip for now
        
        benchmarking_results = {
            'elo_system': {
                'accuracy': elo_metrics.get('accuracy', 0),
                'brier_score': elo_metrics.get('brier_score', 1),
                'auc_score': elo_metrics.get('auc_score', 0.5)
            },
            'baselines': {
                'random': {
                    'accuracy': random_accuracy,
                    'brier_score': 0.25,  # Theoretical maximum
                    'auc_score': 0.5
                },
                'record_based': {
                    'accuracy': record_based_accuracy,
                    'brier_score': None,  # Not applicable
                    'auc_score': None
                }
            },
            'improvements': {
                'vs_random': elo_metrics.get('accuracy', 0) - random_accuracy,
                'vs_record_based': elo_metrics.get('accuracy', 0) - record_based_accuracy
            }
        }
        
        return benchmarking_results
    
    def _calculate_record_based_accuracy(self, 
                                       test_df: pd.DataFrame,
                                       fighters_df: pd.DataFrame) -> float:
        """Calculate accuracy of record-based predictions"""
        if fighters_df is None:
            return 0.5  # Random if no fighter data
        
        # Create win-loss record lookup
        fighter_records = {}
        for _, fighter in fighters_df.iterrows():
            name = fighter.get('Name', '')
            wins = fighter.get('Wins', 0) or 0
            losses = fighter.get('Losses', 0) or 0
            fighter_records[name] = {'wins': wins, 'losses': losses}
        
        correct_predictions = 0
        total_predictions = 0
        
        processor = UFCHistoricalProcessor()
        processed_fights = processor.process_fights_dataframe(test_df)
        
        for fight in processed_fights:
            if fight['winner'] is None:
                continue
            
            fighter1 = fight['fighter1']
            fighter2 = fight['fighter2']
            
            # Get records
            record1 = fighter_records.get(fighter1, {'wins': 0, 'losses': 0})
            record2 = fighter_records.get(fighter2, {'wins': 0, 'losses': 0})
            
            # Predict based on win percentage
            win_pct1 = record1['wins'] / max(1, record1['wins'] + record1['losses'])
            win_pct2 = record2['wins'] / max(1, record2['wins'] + record2['losses'])
            
            predicted_winner = fighter1 if win_pct1 > win_pct2 else fighter2
            
            if predicted_winner == fight['winner']:
                correct_predictions += 1
            total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.5
    
    def generate_validation_report(self, save_path: str = None) -> str:
        """
        Generate comprehensive validation report
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Validation report as string
        """
        report_lines = [
            "UFC ELO SYSTEM VALIDATION REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Cross-validation results
        if 'cross_validation' in self.validation_results:
            cv_results = self.validation_results['cross_validation']
            report_lines.extend([
                "CROSS-VALIDATION RESULTS:",
                f"Number of folds: {cv_results.get('n_folds', 'N/A')}",
                f"Total predictions: {cv_results.get('total_predictions', 'N/A')}",
                f"Mean accuracy: {cv_results.get('accuracy_mean', 0):.3f} ± {cv_results.get('accuracy_std', 0):.3f}",
                f"Mean Brier score: {cv_results.get('brier_score_mean', 0):.3f} ± {cv_results.get('brier_score_std', 0):.3f}",
                f"Mean AUC: {cv_results.get('auc_score_mean', 0):.3f} ± {cv_results.get('auc_score_std', 0):.3f}",
                ""
            ])
        
        # Configuration comparison
        if 'configuration_comparison' in self.validation_results:
            comp_results = self.validation_results['configuration_comparison']
            report_lines.extend([
                "CONFIGURATION COMPARISON:",
                f"Best configuration: {comp_results.get('best_configuration', 'N/A')}",
                f"Best accuracy: {comp_results.get('best_accuracy', 0):.3f}",
                ""
            ])
        
        # Benchmarking results
        if 'benchmarking' in self.validation_results:
            bench_results = self.validation_results['benchmarking']
            report_lines.extend([
                "BENCHMARKING RESULTS:",
                f"ELO accuracy: {bench_results['elo_system'].get('accuracy', 0):.3f}",
                f"Random baseline: {bench_results['baselines']['random'].get('accuracy', 0):.3f}",
                f"Record-based baseline: {bench_results['baselines']['record_based'].get('accuracy', 0):.3f}",
                f"Improvement vs random: {bench_results['improvements'].get('vs_random', 0):.3f}",
                f"Improvement vs record: {bench_results['improvements'].get('vs_record_based', 0):.3f}",
                ""
            ])
        
        report_lines.extend([
            "RECOMMENDATIONS:",
            "- Monitor calibration regularly with new data",
            "- Consider ensemble methods for improved performance", 
            "- Validate method predictions separately",
            "- Update ELO parameters based on validation results"
        ])
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Validation report saved to {save_path}")
        
        return report_text


def main():
    """Example usage of ELO validation framework"""
    
    # Create sample data for testing
    sample_fights = pd.DataFrame({
        'Fighter': ['Jon Jones', 'Daniel Cormier', 'Stipe Miocic', 'Francis Ngannou'] * 10,
        'Opponent': ['Daniel Cormier', 'Stipe Miocic', 'Francis Ngannou', 'Jon Jones'] * 10,
        'Outcome': (['win', 'loss'] * 2) * 10,
        'Method': ['U-DEC', 'KO', 'TKO', 'Submission'] * 10,
        'Event': [f'UFC {200 + i}' for i in range(40)],
        'Date': pd.date_range(start='2010-01-01', periods=40, freq='30D')
    })
    
    sample_fighters = pd.DataFrame({
        'Name': ['Jon Jones', 'Daniel Cormier', 'Stipe Miocic', 'Francis Ngannou'],
        'Weight (lbs)': [205, 205, 240, 250],
        'Wins': [26, 22, 20, 17],
        'Losses': [1, 3, 4, 3]
    })
    
    # Initialize validator
    validator = ELOValidator()
    
    # Run cross-validation
    print("Running cross-validation...")
    cv_results = validator.cross_validation_backtest(
        sample_fights, sample_fighters, n_folds=3
    )
    print("Cross-validation results:", cv_results)
    
    # Compare configurations
    print("\nComparing configurations...")
    config_comparison = validator.compare_elo_configurations(
        sample_fights, sample_fighters
    )
    print("Configuration comparison:", config_comparison['best_configuration'])
    
    # Run benchmarking
    print("\nRunning benchmarking...")
    benchmark_results = validator.benchmark_against_baseline(
        sample_fights, sample_fighters
    )
    print("Benchmark results:", benchmark_results['improvements'])
    
    # Generate report
    report = validator.generate_validation_report()
    print("\nValidation Report:")
    print(report)


if __name__ == "__main__":
    main()
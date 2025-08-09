#!/usr/bin/env python3
"""
Feature Importance Analysis using SHAP values.

Provides comprehensive analysis of feature contributions to model predictions,
identifying the most important features and their interactions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import shap
from typing import Dict, List, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis for UFC prediction models.
    Uses SHAP values for model-agnostic explanations.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize feature importance analyzer.
        
        Args:
            model_path: Path to trained model
        """
        self.model = None
        self.feature_names = None
        self.shap_values = None
        self.shap_explainer = None
        self.X_sample = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model and feature names."""
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        # Load feature names from training metadata
        model_dir = Path(model_path).parent
        column_files = list(model_dir.glob('*_columns.json'))
        
        if column_files:
            with open(column_files[0], 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        else:
            logger.warning("No feature names found, using indices")
    
    def calculate_shap_values(
        self,
        X: pd.DataFrame,
        sample_size: int = 1000,
        use_tree_explainer: bool = True
    ) -> np.ndarray:
        """
        Calculate SHAP values for feature importance.
        
        Args:
            X: Feature data
            sample_size: Number of samples for SHAP calculation
            use_tree_explainer: Use TreeExplainer for tree-based models
            
        Returns:
            SHAP values array
        """
        logger.info("Calculating SHAP values...")
        
        # Sample data if too large
        if len(X) > sample_size:
            logger.info(f"Sampling {sample_size} rows from {len(X)} total")
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        self.X_sample = X_sample
        
        # Create appropriate explainer
        if use_tree_explainer and hasattr(self.model, 'tree_'):
            logger.info("Using TreeExplainer for tree-based model")
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            logger.info("Using KernelExplainer for model-agnostic explanation")
            # Use smaller background sample for KernelExplainer
            background = shap.sample(X_sample, min(100, len(X_sample)))
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
        
        # Calculate SHAP values
        self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        # For binary classification, use positive class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        logger.info(f"SHAP values calculated: shape {self.shap_values.shape}")
        return self.shap_values
    
    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """
        Get feature importance ranking based on mean absolute SHAP values.
        
        Returns:
            DataFrame with feature importance rankings
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create ranking DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else range(len(mean_abs_shap)),
            'importance': mean_abs_shap,
            'importance_pct': mean_abs_shap / mean_abs_shap.sum() * 100
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df['cumulative_importance_pct'] = importance_df['importance_pct'].cumsum()
        
        return importance_df
    
    def identify_redundant_features(
        self,
        correlation_threshold: float = 0.95,
        importance_threshold: float = 0.001
    ) -> Dict[str, List[str]]:
        """
        Identify potentially redundant features based on correlation and importance.
        
        Args:
            correlation_threshold: Correlation threshold for redundancy
            importance_threshold: Minimum importance to keep feature
            
        Returns:
            Dictionary of redundant feature groups
        """
        if self.X_sample is None:
            raise ValueError("Must calculate SHAP values first")
        
        logger.info("Identifying redundant features...")
        
        # Get importance ranking
        importance_df = self.get_feature_importance_ranking()
        
        # Identify low-importance features
        low_importance = importance_df[
            importance_df['importance'] < importance_threshold
        ]['feature'].tolist()
        
        # Calculate correlation matrix
        corr_matrix = self.X_sample.corr().abs()
        
        # Find highly correlated feature pairs
        redundant_groups = {}
        processed = set()
        
        for i, feat1 in enumerate(self.X_sample.columns):
            if feat1 in processed:
                continue
            
            correlated = []
            for j, feat2 in enumerate(self.X_sample.columns):
                if i != j and corr_matrix.iloc[i, j] > correlation_threshold:
                    correlated.append(feat2)
                    processed.add(feat2)
            
            if correlated:
                # Keep the most important feature
                group_importance = importance_df[
                    importance_df['feature'].isin([feat1] + correlated)
                ]
                keep_feature = group_importance.iloc[0]['feature']
                redundant_groups[keep_feature] = [
                    f for f in ([feat1] + correlated) if f != keep_feature
                ]
        
        return {
            'low_importance_features': low_importance,
            'redundant_groups': redundant_groups,
            'total_redundant': len(low_importance) + sum(
                len(v) for v in redundant_groups.values()
            )
        }
    
    def analyze_feature_interactions(
        self,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Analyze feature interactions using SHAP interaction values.
        
        Args:
            top_n: Number of top interactions to return
            
        Returns:
            DataFrame with top feature interactions
        """
        logger.info("Analyzing feature interactions...")
        
        # Calculate interaction values (this can be slow)
        if hasattr(self.shap_explainer, 'shap_interaction_values'):
            interaction_vals = self.shap_explainer.shap_interaction_values(
                self.X_sample.iloc[:100]  # Use subset for speed
            )
            
            if isinstance(interaction_vals, list):
                interaction_vals = interaction_vals[1]
            
            # Get mean absolute interaction strength
            n_features = interaction_vals.shape[1]
            interactions = []
            
            for i in range(n_features):
                for j in range(i+1, n_features):
                    mean_interaction = np.abs(interaction_vals[:, i, j]).mean()
                    feat1 = self.feature_names[i] if self.feature_names else f"feature_{i}"
                    feat2 = self.feature_names[j] if self.feature_names else f"feature_{j}"
                    interactions.append({
                        'feature_1': feat1,
                        'feature_2': feat2,
                        'interaction_strength': mean_interaction
                    })
            
            interactions_df = pd.DataFrame(interactions)
            interactions_df = interactions_df.sort_values(
                'interaction_strength', ascending=False
            ).head(top_n)
            
            return interactions_df
        else:
            logger.warning("Interaction values not available for this explainer type")
            return pd.DataFrame()
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        Categorize features by their semantic meaning.
        
        Returns:
            Dictionary mapping categories to feature lists
        """
        if not self.feature_names:
            return {}
        
        categories = {
            'striking': [],
            'grappling': [],
            'defense': [],
            'cardio': [],
            'reach': [],
            'experience': [],
            'differential': [],
            'other': []
        }
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if 'strike' in feature_lower or 'str' in feature_lower:
                categories['striking'].append(feature)
            elif 'takedown' in feature_lower or 'td' in feature_lower or 'grappl' in feature_lower:
                categories['grappling'].append(feature)
            elif 'def' in feature_lower or 'defense' in feature_lower:
                categories['defense'].append(feature)
            elif 'round' in feature_lower or 'cardio' in feature_lower:
                categories['cardio'].append(feature)
            elif 'reach' in feature_lower or 'height' in feature_lower:
                categories['reach'].append(feature)
            elif 'age' in feature_lower or 'exp' in feature_lower or 'fight' in feature_lower:
                categories['experience'].append(feature)
            elif 'diff' in feature_lower:
                categories['differential'].append(feature)
            else:
                categories['other'].append(feature)
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}
        
        return categories
    
    def analyze_category_importance(self) -> pd.DataFrame:
        """
        Analyze importance by feature category.
        
        Returns:
            DataFrame with category-level importance
        """
        categories = self.get_feature_categories()
        importance_df = self.get_feature_importance_ranking()
        
        category_importance = []
        
        for category, features in categories.items():
            cat_importance = importance_df[
                importance_df['feature'].isin(features)
            ]['importance'].sum()
            
            category_importance.append({
                'category': category,
                'num_features': len(features),
                'total_importance': cat_importance,
                'avg_importance': cat_importance / len(features) if features else 0,
                'top_feature': importance_df[
                    importance_df['feature'].isin(features)
                ].iloc[0]['feature'] if any(importance_df['feature'].isin(features)) else None
            })
        
        cat_importance_df = pd.DataFrame(category_importance)
        cat_importance_df = cat_importance_df.sort_values(
            'total_importance', ascending=False
        )
        
        return cat_importance_df
    
    def create_importance_plots(self, output_dir: str = 'artifacts/feature_importance'):
        """
        Create and save feature importance visualizations.
        
        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating importance plots in {output_dir}")
        
        # 1. Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar plot of top features
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top 20 features detailed
        importance_df = self.get_feature_importance_ranking()
        top_20 = importance_df.head(20)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Importance bar chart
        ax1.barh(range(20), top_20['importance'].values)
        ax1.set_yticks(range(20))
        ax1.set_yticklabels(top_20['feature'].values)
        ax1.set_xlabel('Mean |SHAP value|')
        ax1.set_title('Top 20 Most Important Features')
        ax1.invert_yaxis()
        
        # Cumulative importance
        ax2.plot(range(1, 21), top_20['cumulative_importance_pct'].values, 'b-', linewidth=2)
        ax2.fill_between(range(1, 21), 0, top_20['cumulative_importance_pct'].values, alpha=0.3)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance (%)')
        ax2.set_title('Cumulative Feature Importance')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'top_features_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Category importance
        cat_importance = self.analyze_category_importance()
        if not cat_importance.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Total importance by category
            ax1.bar(cat_importance['category'], cat_importance['total_importance'])
            ax1.set_xlabel('Category')
            ax1.set_ylabel('Total Importance')
            ax1.set_title('Feature Importance by Category')
            ax1.tick_params(axis='x', rotation=45)
            
            # Feature count vs avg importance
            ax2.scatter(cat_importance['num_features'], 
                       cat_importance['avg_importance'],
                       s=cat_importance['total_importance']*100,
                       alpha=0.6)
            for _, row in cat_importance.iterrows():
                ax2.annotate(row['category'], 
                           (row['num_features'], row['avg_importance']),
                           fontsize=8)
            ax2.set_xlabel('Number of Features')
            ax2.set_ylabel('Average Importance')
            ax2.set_title('Category Efficiency')
            
            plt.tight_layout()
            plt.savefig(output_path / 'category_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Plots saved to {output_dir}")
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive feature importance report.
        
        Returns:
            Dictionary containing full analysis results
        """
        logger.info("Generating comprehensive report...")
        
        importance_df = self.get_feature_importance_ranking()
        redundant_analysis = self.identify_redundant_features()
        category_importance = self.analyze_category_importance()
        
        # Find minimum features for 80% importance
        min_features_80 = len(
            importance_df[importance_df['cumulative_importance_pct'] <= 80]
        ) + 1
        
        # Top features analysis
        top_10 = importance_df.head(10)
        
        report = {
            'summary': {
                'total_features': len(importance_df),
                'features_for_80_pct': min_features_80,
                'redundant_features': redundant_analysis['total_redundant'],
                'optimal_feature_count': len(importance_df) - redundant_analysis['total_redundant'],
                'top_feature': importance_df.iloc[0]['feature'],
                'top_feature_importance': float(importance_df.iloc[0]['importance_pct'])
            },
            'top_10_features': top_10[['feature', 'importance_pct', 'cumulative_importance_pct']].to_dict('records'),
            'category_analysis': category_importance.to_dict('records'),
            'redundant_features': redundant_analysis,
            'recommendations': self._generate_recommendations(
                importance_df, redundant_analysis, category_importance
            )
        }
        
        return report
    
    def _generate_recommendations(
        self,
        importance_df: pd.DataFrame,
        redundant_analysis: Dict,
        category_importance: pd.DataFrame
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Feature reduction recommendation
        min_features_80 = len(
            importance_df[importance_df['cumulative_importance_pct'] <= 80]
        ) + 1
        
        if min_features_80 < len(importance_df) * 0.5:
            recommendations.append(
                f"Consider reducing to top {min_features_80} features "
                f"(captures 80% of importance, reduces from {len(importance_df)} features)"
            )
        
        # Redundant features
        if redundant_analysis['total_redundant'] > 5:
            recommendations.append(
                f"Remove {redundant_analysis['total_redundant']} redundant features "
                f"to improve model efficiency"
            )
        
        # Category imbalance
        if not category_importance.empty:
            top_cat = category_importance.iloc[0]
            if top_cat['total_importance'] > 0.5:
                recommendations.append(
                    f"Model heavily relies on {top_cat['category']} features "
                    f"({top_cat['total_importance']:.1%} of importance) - "
                    f"consider diversifying feature set"
                )
        
        # Low importance features
        very_low = importance_df[importance_df['importance_pct'] < 0.1]
        if len(very_low) > 10:
            recommendations.append(
                f"Remove {len(very_low)} features with <0.1% importance "
                f"to simplify model"
            )
        
        # Top feature dominance
        if importance_df.iloc[0]['importance_pct'] > 15:
            recommendations.append(
                f"Top feature '{importance_df.iloc[0]['feature']}' has "
                f"{importance_df.iloc[0]['importance_pct']:.1f}% importance - "
                f"verify this isn't data leakage"
            )
        
        return recommendations
    
    def save_report(self, output_path: str = 'artifacts/feature_importance/report.json'):
        """Save analysis report to file."""
        report = self.generate_report()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_path}")
        
        return report


def main():
    """Run feature importance analysis on trained model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze feature importance')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--data', type=str, help='Path to dataset')
    parser.add_argument('--output', type=str, default='artifacts/feature_importance',
                       help='Output directory')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Sample size for SHAP calculation')
    
    args = parser.parse_args()
    
    # Find model if not specified
    if not args.model:
        from pathlib import Path
        model_dir = Path('model')
        training_dirs = [d for d in model_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('training_')]
        if training_dirs:
            latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
            model_files = list(latest_dir.glob('ufc_winner_model*.joblib'))
            if model_files:
                args.model = str(model_files[0])
        
        if not args.model:
            print("No model found. Please specify --model path")
            return
    
    # Find data if not specified
    if not args.data:
        model_dir = Path(args.model).parent
        data_files = list(model_dir.glob('ufc_fight_dataset_with_diffs*.csv'))
        if data_files:
            args.data = str(data_files[0])
        else:
            print("No data found. Please specify --data path")
            return
    
    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(args.model)
    
    # Load data
    print(f"\nLoading data from {args.data}")
    df = pd.read_csv(args.data)
    
    # Get feature columns
    exclude_cols = ['Date', 'Winner', 'Outcome', 'blue_fighter', 'red_fighter', 
                   'loser_fighter', 'Event', 'Method', 'Time']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    
    print(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    
    # Calculate SHAP values
    analyzer.calculate_shap_values(X, sample_size=args.sample_size)
    
    # Generate report
    report = analyzer.generate_report()
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"\n📊 Feature Statistics:")
    print(f"   Total features: {report['summary']['total_features']}")
    print(f"   Features for 80% importance: {report['summary']['features_for_80_pct']}")
    print(f"   Redundant features: {report['summary']['redundant_features']}")
    print(f"   Optimal feature count: {report['summary']['optimal_feature_count']}")
    
    print(f"\n🏆 Top 10 Most Important Features:")
    for i, feat in enumerate(report['top_10_features'], 1):
        print(f"   {i:2d}. {feat['feature'][:30]:30s} {feat['importance_pct']:5.2f}% (cum: {feat['cumulative_importance_pct']:5.1f}%)")
    
    print(f"\n📂 Category Analysis:")
    for cat in report['category_analysis'][:5]:
        print(f"   {cat['category']:12s}: {cat['num_features']:2d} features, {cat['total_importance']:.3f} total importance")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Create visualizations
    print(f"\n📈 Creating visualizations...")
    analyzer.create_importance_plots(args.output)
    
    # Save report
    analyzer.save_report(f"{args.output}/report.json")
    
    print(f"\n✅ Analysis complete! Results saved to {args.output}/")
    print(f"   • Report: {args.output}/report.json")
    print(f"   • Plots: {args.output}/*.png")
    
    return analyzer, report


if __name__ == "__main__":
    main()
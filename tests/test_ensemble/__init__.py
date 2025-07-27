"""
Ensemble Component Tests

Comprehensive test suite for the XGBoost ensemble system including:
- Bootstrap confidence intervals with thread safety
- Production ensemble manager with memory management  
- Enhanced prediction service with input validation
- Complete integration workflows
- Performance and security testing

Test Coverage:
- Unit tests for individual components
- Integration tests for complete workflows
- Error handling and edge case validation
- Performance benchmarks and memory monitoring
- Security testing for input sanitization
- Real-world UFC event simulation

Usage:
    pytest tests/test_ensemble/ -v                    # Run all ensemble tests
    pytest tests/test_ensemble/test_bootstrap_confidence.py -v  # Bootstrap tests only
    pytest tests/test_ensemble/test_production_ensemble_manager.py -v  # Manager tests only
    pytest tests/test_ensemble/test_enhanced_prediction_service.py -v  # Service tests only
    pytest tests/test_ensemble/test_ensemble_integration.py -v  # Integration tests only
"""
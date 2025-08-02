#!/usr/bin/env python3
"""
UFC Predictor Test Runner
========================

A comprehensive test runner for the UFC predictor system that provides
organized test execution, reporting, and performance metrics.

Features:
- Run specific test categories (unit, integration, performance)
- Generate test coverage reports
- Performance benchmarking
- Continuous integration support
- Detailed test reporting

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --performance      # Run only performance tests
    python run_tests.py --agent            # Run only agent tests
    python run_tests.py --phase2a          # Run only Phase 2A tests
    python run_tests.py --fast             # Skip slow tests
    python run_tests.py --coverage         # Generate coverage report
    python run_tests.py --benchmark        # Run performance benchmarks
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Optional
import json


class UFCTestRunner:
    """Comprehensive test runner for UFC predictor system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.results = {}
        
    def run_pytest(self, args: List[str]) -> tuple:
        """Run pytest with given arguments and return results"""
        cmd = ["python3", "-m", "pytest"] + args
        
        print(f"🚀 Running: {' '.join(cmd)}")
        print("=" * 60)
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True)
        end_time = time.time()
        
        execution_time = end_time - start_time
        return result.returncode, execution_time
    
    def run_unit_tests(self, fast: bool = False) -> tuple:
        """Run unit tests"""
        print("🧪 RUNNING UNIT TESTS")
        print("-" * 30)
        
        args = [
            str(self.tests_dir / "test_feature_engineering.py"),
            "-v",
            "-m", "unit" if not fast else "unit and not slow",
            "--tb=short"
        ]
        
        if fast:
            args.extend(["-x"])  # Stop on first failure
        
        return self.run_pytest(args)
    
    def run_integration_tests(self, fast: bool = False) -> tuple:
        """Run integration tests"""
        print("🔗 RUNNING INTEGRATION TESTS")
        print("-" * 30)
        
        args = [
            str(self.tests_dir / "test_integration.py"),
            "-v",
            "-m", "integration" if not fast else "integration and not slow",
            "--tb=short"
        ]
        
        if fast:
            args.extend(["-x"])
        
        return self.run_pytest(args)
    
    def run_performance_tests(self) -> tuple:
        """Run performance tests"""
        print("⚡ RUNNING PERFORMANCE TESTS")
        print("-" * 30)
        
        args = [
            str(self.tests_dir),
            "-v",
            "-m", "performance",
            "--tb=short",
            "-s"  # Don't capture output for performance logs
        ]
        
        return self.run_pytest(args)
    
    def run_agent_tests(self, fast: bool = False) -> tuple:
        """Run Enhanced ML Pipeline agent tests"""
        print("🤖 RUNNING AGENT TESTS")
        print("-" * 30)
        
        agent_tests_dir = self.tests_dir / "test_agent"
        
        if not agent_tests_dir.exists():
            print("⚠️  Agent tests directory not found, skipping...")
            return 0, 0
        
        args = [
            str(agent_tests_dir),
            "-v", 
            "-m", "agent" if not fast else "agent and not slow",
            "--tb=short"
        ]
        
        if fast:
            args.extend(["-x"])  # Stop on first failure
        
        return self.run_pytest(args)
    
    def run_phase2a_tests(self, fast: bool = False) -> tuple:
        """Run Phase 2A hybrid system tests"""
        print("🔀 RUNNING PHASE 2A TESTS")
        print("-" * 30)
        
        args = [
            str(self.tests_dir),
            "-v",
            "-m", "phase2a" if not fast else "phase2a and not slow",
            "--tb=short"
        ]
        
        if fast:
            args.extend(["-x"])
        
        return self.run_pytest(args)
    
    def run_all_tests(self, fast: bool = False) -> dict:
        """Run all test categories"""
        print("🎯 RUNNING COMPLETE TEST SUITE")
        print("=" * 60)
        
        results = {}
        total_time = 0
        
        # Run unit tests
        returncode, exec_time = self.run_unit_tests(fast)
        results['unit_tests'] = {
            'success': returncode == 0,
            'execution_time': exec_time
        }
        total_time += exec_time
        
        print(f"\n✅ Unit tests completed in {exec_time:.2f}s")
        print("=" * 60)
        
        # Run integration tests
        returncode, exec_time = self.run_integration_tests(fast)
        results['integration_tests'] = {
            'success': returncode == 0,
            'execution_time': exec_time
        }
        total_time += exec_time
        
        print(f"\n✅ Integration tests completed in {exec_time:.2f}s")
        print("=" * 60)
        
        # Run agent tests
        returncode, exec_time = self.run_agent_tests(fast)
        results['agent_tests'] = {
            'success': returncode == 0,
            'execution_time': exec_time
        }
        total_time += exec_time
        
        print(f"\n✅ Agent tests completed in {exec_time:.2f}s")
        print("=" * 60)
        
        # Run Phase 2A tests
        returncode, exec_time = self.run_phase2a_tests(fast)
        results['phase2a_tests'] = {
            'success': returncode == 0,
            'execution_time': exec_time
        }
        total_time += exec_time
        
        print(f"\n✅ Phase 2A tests completed in {exec_time:.2f}s")
        print("=" * 60)
        
        # Only run performance tests if others pass and not in fast mode
        if (not fast and results['unit_tests']['success'] and 
            results['integration_tests']['success'] and results['agent_tests']['success'] and 
            results['phase2a_tests']['success']):
            returncode, exec_time = self.run_performance_tests()
            results['performance_tests'] = {
                'success': returncode == 0,
                'execution_time': exec_time
            }
            total_time += exec_time
            
            print(f"\n✅ Performance tests completed in {exec_time:.2f}s")
        else:
            if fast:
                print("\n⏩ Skipping performance tests in fast mode")
            else:
                print("\n⏭️  Skipping performance tests due to failures")
            results['performance_tests'] = {'success': True, 'execution_time': 0}
        
        results['total_time'] = total_time
        
        return results
    
    def run_coverage_analysis(self) -> tuple:
        """Run tests with coverage analysis"""
        print("📊 RUNNING COVERAGE ANALYSIS")
        print("-" * 30)
        
        # Check if coverage is installed
        try:
            import coverage
        except ImportError:
            print("❌ Coverage.py not installed. Install with: pip install coverage pytest-cov")
            return 1, 0
        
        args = [
            str(self.tests_dir),
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=70",  # Require 70% coverage
            "-v"
        ]
        
        returncode, exec_time = self.run_pytest(args)
        
        if returncode == 0:
            print("\n✅ Coverage analysis completed successfully")
            print("📁 HTML coverage report generated in htmlcov/")
        else:
            print("\n❌ Coverage analysis failed or coverage too low")
        
        return returncode, exec_time
    
    def run_benchmark(self) -> None:
        """Run performance benchmarks"""
        print("🏃 RUNNING PERFORMANCE BENCHMARKS")
        print("-" * 50)
        
        try:
            from conftest import generate_large_fighter_dataset
            from ufc_predictor.features.optimized_feature_engineering import engineer_features_final_optimized, benchmark_feature_engineering
            
            # Generate test data
            print("📊 Benchmarking feature engineering performance...")
            test_data = generate_large_fighter_dataset(500)
            
            # Run benchmark
            benchmark_results = benchmark_feature_engineering(test_data, num_runs=3)
            
            print("\n📈 BENCHMARK RESULTS:")
            print(f"   Dataset size: {benchmark_results['dataset_size']} fighters")
            print(f"   Optimized time: {benchmark_results['optimized_avg_time']:.2f}s")
            print(f"   Estimated original: {benchmark_results['estimated_original_time']:.2f}s")
            print(f"   Performance improvement: {benchmark_results['performance_improvement_pct']:.1f}%")
            print(f"   Speedup factor: {benchmark_results['speedup_factor']:.1f}x")
            
            # Save benchmark results
            benchmark_file = self.project_root / "benchmark_results.json"
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            print(f"\n📁 Benchmark results saved to {benchmark_file}")
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
    
    def print_summary(self, results: dict) -> None:
        """Print test execution summary"""
        print("\n" + "=" * 60)
        print("🎯 TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        total_success = True
        
        for test_type, result in results.items():
            if test_type == 'total_time':
                continue
                
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            time_str = f"{result['execution_time']:.2f}s"
            test_name = test_type.replace('_', ' ').title()
            
            print(f"{test_name:<20} {status:<10} {time_str:>10}")
            
            if not result['success']:
                total_success = False
        
        print("-" * 60)
        total_time = results.get('total_time', 0)
        overall_status = "✅ ALL PASS" if total_success else "❌ SOME FAILED"
        print(f"{'Overall Result':<20} {overall_status:<10} {total_time:.2f}s")
        print("=" * 60)
        
        if total_success:
            print("🎉 All tests passed! The UFC predictor system is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the output above for details.")
    
    def check_dependencies(self) -> bool:
        """Check if required test dependencies are available"""
        print("🔍 CHECKING TEST DEPENDENCIES")
        print("-" * 30)
        
        required_packages = [
            'pytest',
            'pandas', 
            'numpy',
            'unittest.mock'
        ]
        
        optional_packages = [
            'pytest-cov',
            'pytest-benchmark'
        ]
        
        missing_required = []
        missing_optional = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (REQUIRED)")
                missing_required.append(package)
        
        for package in optional_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package} (optional)")
            except ImportError:
                print(f"⚠️  {package} (optional)")
                missing_optional.append(package)
        
        if missing_required:
            print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
            print("Install with: pip install " + " ".join(missing_required))
            return False
        
        if missing_optional:
            print(f"\n💡 Optional packages not installed: {', '.join(missing_optional)}")
            print("Install for enhanced functionality: pip install " + " ".join(missing_optional))
        
        print("\n✅ All required dependencies are available")
        return True


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description="UFC Predictor Test Runner")
    
    # Test selection arguments
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--agent", action="store_true", help="Run only agent tests")
    parser.add_argument("--phase2a", action="store_true", help="Run only Phase 2A tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    
    # Test execution options
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    
    # Output options
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--check-deps", action="store_true", help="Check test dependencies")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = UFCTestRunner()
    
    # Check dependencies if requested
    if args.check_deps:
        if not runner.check_dependencies():
            sys.exit(1)
        return
    
    # Determine which tests to run
    run_specific = args.unit or args.integration or args.agent or args.phase2a or args.performance
    
    try:
        if args.benchmark:
            runner.run_benchmark()
        
        elif args.coverage:
            returncode, _ = runner.run_coverage_analysis()
            sys.exit(returncode)
        
        elif args.unit:
            returncode, _ = runner.run_unit_tests(args.fast)
            sys.exit(returncode)
        
        elif args.integration:
            returncode, _ = runner.run_integration_tests(args.fast)
            sys.exit(returncode)
        
        elif args.agent:
            returncode, _ = runner.run_agent_tests(args.fast)
            sys.exit(returncode)
        
        elif args.phase2a:
            returncode, _ = runner.run_phase2a_tests(args.fast)
            sys.exit(returncode)
        
        elif args.performance:
            returncode, _ = runner.run_performance_tests()
            sys.exit(returncode)
        
        else:
            # Run all tests (default)
            results = runner.run_all_tests(args.fast)
            runner.print_summary(results)
            
            # Exit with error code if any tests failed
            if not all(result['success'] for result in results.values() if isinstance(result, dict)):
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
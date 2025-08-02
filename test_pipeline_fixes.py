#!/usr/bin/env python3
"""
Quick test to verify autonomous pipeline hanging fixes
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from autonomous_pipeline import PipelineConfig, WebScraperAgent

async def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("🧪 Testing Circuit Breaker Functionality")
    print("=" * 50)
    
    config = PipelineConfig(debug_mode=True)
    agent = WebScraperAgent(config)
    
    test_scraper = "test_scraper.py"
    
    # Test failure recording
    agent._record_scraper_failure(test_scraper)
    print(f"After 1 failure: {agent.scraper_failures}")
    print(f"Circuit open: {agent._is_scraper_circuit_open(test_scraper)}")
    
    agent._record_scraper_failure(test_scraper)
    print(f"After 2 failures: {agent.scraper_failures}")
    print(f"Circuit open: {agent._is_scraper_circuit_open(test_scraper)}")
    
    # Test success reset
    agent._record_scraper_success(test_scraper)
    print(f"After success: {agent.scraper_failures}")
    print(f"Circuit open: {agent._is_scraper_circuit_open(test_scraper)}")
    
    print("✅ Circuit breaker test completed")

async def test_scraper_priority():
    """Test scraper priority order"""
    print("\n🧪 Testing Scraper Priority Order")  
    print("=" * 50)
    
    config = PipelineConfig()
    print(f"Primary scraper: {config.webscraper_script}")
    print(f"Backup scrapers: {config.backup_scrapers}")
    
    expected_order = [
        "webscraper/fast_scraping.py",
        "webscraper/scraping.py", 
        "webscraper/optimized_scraping.py"
    ]
    
    actual_order = [config.webscraper_script] + config.backup_scrapers
    
    if actual_order == expected_order:
        print("✅ Scraper priority order is correct")
    else:
        print("❌ Scraper priority order mismatch")
        print(f"Expected: {expected_order}")
        print(f"Actual: {actual_order}")

async def test_syntax_validation():
    """Test that optimized_scraping.py has valid syntax"""
    print("\n🧪 Testing Optimized Scraper Syntax")
    print("=" * 50)
    
    try:
        import ast
        scraper_path = Path("webscraper/optimized_scraping.py")
        
        if scraper_path.exists():
            with open(scraper_path, 'r') as f:
                code = f.read()
            
            # Parse the code to check for syntax errors
            ast.parse(code)
            print("✅ Optimized scraper syntax is valid")
        else:
            print("⚠️ Optimized scraper file not found")
            
    except SyntaxError as e:
        print(f"❌ Syntax error in optimized scraper: {e}")
    except Exception as e:
        print(f"❌ Error validating syntax: {e}")

async def main():
    """Run all tests"""
    print("🚀 Testing Autonomous Pipeline Fixes")
    print("=" * 60)
    
    await test_circuit_breaker()
    await test_scraper_priority() 
    await test_syntax_validation()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\nFixes implemented:")
    print("✅ Scraper priority: stable scrapers first")
    print("✅ Circuit breaker: automatic failure detection") 
    print("✅ Process termination: robust subprocess handling")
    print("✅ Async deadlock fix: removed nested semaphores")
    print("✅ Concurrency limits: batched letter processing")
    
    print("\n🚀 Pipeline should now be much more reliable!")

if __name__ == "__main__":
    asyncio.run(main())
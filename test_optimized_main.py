#!/usr/bin/env python3
"""
Test the main() function of optimized scraper to find hanging point
"""
import asyncio
import sys
from pathlib import Path

print("Testing optimized scraper main function...")

# Add the necessary path
sys.path.append(str(Path(__file__).parent))

# Import the scraper module
try:
    import webscraper.optimized_scraping as opt_scraper
    print("✓ Optimized scraper module imported")
except Exception as e:
    print(f"✗ Module import failed: {e}")
    sys.exit(1)

# Test creating the configuration
try:
    config = opt_scraper.ScrapingConfig()
    print("✓ Configuration created")
except Exception as e:
    print(f"✗ Configuration creation failed: {e}")
    sys.exit(1)

# Test creating the scraper instance
try:
    scraper_class = opt_scraper.OptimizedUFCStatsScraper
    print("✓ Scraper class found")
except Exception as e:
    print(f"✗ Scraper class not found: {e}")
    sys.exit(1)

async def test_scraper_creation():
    """Test creating scraper instance in async context"""
    print("Creating scraper instance...")
    
    try:
        async with opt_scraper.OptimizedUFCStatsScraper(config) as scraper:
            print("✓ Scraper created successfully")
            print("✓ Async context manager works")
            return True
    except Exception as e:
        print(f"✗ Scraper creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_minimal_run():
    """Test a minimal version of the scraper run"""
    print("Testing minimal scraper run...")
    
    try:
        # Create scraper without context manager first
        scraper = opt_scraper.OptimizedUFCStatsScraper(config)
        await scraper.setup_session()
        print("✓ Session setup successful")
        
        # Test just the first step - getting fighter URLs for one letter
        print("Testing single letter URL fetch...")
        
        # This is where it might hang - let's test with timeout
        try:
            urls = await asyncio.wait_for(
                scraper.get_fighter_urls_for_single_letter('a'),
                timeout=10
            )
            print(f"✓ Got {len(urls)} URLs for letter 'a'")
        except asyncio.TimeoutError:
            print("✗ TIMEOUT: Single letter fetch hanging!")
            return False
        except AttributeError:
            print("Method doesn't exist, testing concurrent version...")
            try:
                urls = await asyncio.wait_for(
                    scraper.get_fighter_urls_concurrent(),
                    timeout=10
                )
                print(f"✓ Got {len(urls)} URLs total")
            except asyncio.TimeoutError:
                print("✗ TIMEOUT: Concurrent fetch hanging!")
                return False
        
        await scraper.cleanup()
        print("✓ Cleanup successful")
        return True
        
    except Exception as e:
        print(f"✗ Minimal run failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("=== OPTIMIZED SCRAPER MAIN TEST ===")
    
    print("\n--- Test 1: Scraper creation ---")
    creation_success = await test_scraper_creation()
    
    if creation_success:
        print("\n--- Test 2: Minimal run ---")
        run_success = await test_minimal_run()
    else:
        print("Skipping minimal run test due to creation failure")
    
    print("\n=== MAIN TEST COMPLETED ===")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Test interrupted")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
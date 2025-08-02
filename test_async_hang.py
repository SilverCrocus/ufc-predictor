#!/usr/bin/env python3
"""
Test async operations to isolate hanging issue
"""
import asyncio
import aiohttp
import time
from pathlib import Path

print("Testing async operations...")

async def test_async_session():
    """Test basic async session creation"""
    print("Creating async session...")
    
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=20,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(
        total=30,
        connect=10,
        sock_read=10
    )
    
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    
    print("Session created successfully")
    
    try:
        # Test simple request
        print("Making test request...")
        async with session.get("http://ufcstats.com/statistics/fighters") as response:
            if response.status == 200:
                content = await response.text()
                print(f"Request successful, content length: {len(content)}")
            else:
                print(f"Request failed with status: {response.status}")
    except Exception as e:
        print(f"Request failed: {e}")
    finally:
        await session.close()
        print("Session closed")

async def test_concurrent_requests():
    """Test concurrent requests which might be causing the hang"""
    print("Testing concurrent requests...")
    
    session = aiohttp.ClientSession(
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    
    urls = [
        "http://ufcstats.com/statistics/fighters?char=a&page=all",
        "http://ufcstats.com/statistics/fighters?char=b&page=all",
        "http://ufcstats.com/statistics/fighters?char=c&page=all"
    ]
    
    async def fetch_url(url):
        try:
            print(f"Fetching: {url}")
            async with session.get(url) as response:
                content = await response.text()
                print(f"Completed: {url[-10:]} - Length: {len(content)}")
                return len(content)
        except Exception as e:
            print(f"Failed: {url} - {e}")
            return 0
    
    try:
        # Create tasks
        tasks = [fetch_url(url) for url in urls]
        
        # Run with timeout
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30
        )
        
        print(f"Concurrent requests completed: {results}")
        
    except asyncio.TimeoutError:
        print("TIMEOUT: Concurrent requests hanging!")
    except Exception as e:
        print(f"Concurrent requests failed: {e}")
    finally:
        await session.close()
        print("Session closed")

async def test_semaphores():
    """Test semaphores which might be causing deadlocks"""
    print("Testing semaphores...")
    
    semaphore = asyncio.Semaphore(2)
    
    async def semaphore_task(task_id):
        print(f"Task {task_id} waiting for semaphore...")
        async with semaphore:
            print(f"Task {task_id} acquired semaphore")
            await asyncio.sleep(1)
            print(f"Task {task_id} releasing semaphore")
        return task_id
    
    try:
        tasks = [semaphore_task(i) for i in range(5)]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=20
        )
        print(f"Semaphore test completed: {results}")
    except asyncio.TimeoutError:
        print("TIMEOUT: Semaphore test hanging!")
    except Exception as e:
        print(f"Semaphore test failed: {e}")

async def main():
    print("=== ASYNC OPERATION TESTS ===")
    
    print("\n--- Test 1: Basic async session ---")
    await test_async_session()
    
    print("\n--- Test 2: Concurrent requests ---")
    await test_concurrent_requests()
    
    print("\n--- Test 3: Semaphores ---")
    await test_semaphores()
    
    print("\n=== ASYNC TESTS COMPLETED ===")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Test interrupted")
    except Exception as e:
        print(f"Test failed: {e}")
import json
import requests
import logging
import concurrent.futures
import time

logger = logging.getLogger(__name__)

url = "http://127.0.0.1:9000/run"
payload = {
    "code": "print('Hello, world!')"
}
headers = {
    "Content-Type": "application/json",
}

def make_request():
    """Make a single request to the server"""
    try:
        response = requests.post(
            url,
            json=payload,  # This automatically handles JSON encoding
            headers=headers,
            timeout=10,  # Adding timeout for better error handling
            verify=False  # Disable SSL verification
        )
        
        response.raise_for_status()
        return response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.content}")
        return None

def run_parallel_test(num_requests=100, max_workers=20):
    """Run multiple requests in parallel"""
    logger.info(f"Starting parallel test with {num_requests} requests using {max_workers} workers")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    end_time = time.time()
    successful_requests = sum(1 for r in results if r == 200)
    
    logger.info(f"Parallel test completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Successful requests: {successful_requests}/{num_requests}")

if __name__ == "__main__":
    
    # Then run parallel test
    logger.info("\nRunning parallel test...")
    run_parallel_test()

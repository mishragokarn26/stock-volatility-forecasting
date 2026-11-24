import time
import functools

def retry(max_retries=3, delay=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Retry {attempt+1}/{max_retries} failed: {e}")
                    time.sleep(delay)
            raise Exception("Max retries exceeded")
        return wrapper
    return decorator
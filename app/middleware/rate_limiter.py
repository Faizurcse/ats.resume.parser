"""
Rate Limiting Middleware for high-scale processing.
Limits requests per user/IP to prevent abuse.
"""

import time
import logging
from typing import Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter using Redis for distributed limiting."""
    
    def __init__(self):
        """Initialize rate limiter with Redis."""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=1,  # Use different DB for rate limiting
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Rate limiter Redis connection established")
        except Exception as e:
            logger.error(f"❌ Rate limiter Redis connection failed: {str(e)}")
            self.redis_client = None
    
    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """
        Check if request is allowed based on rate limit.
        
        Args:
            key: Unique identifier (IP, user_id, etc.)
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            bool: True if allowed, False if rate limited
        """
        if not self.redis_client:
            return True  # Allow if Redis is down
        
        try:
            current_time = int(time.time())
            window_start = current_time - window
            
            # Use sliding window counter
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
            
            # Count current requests
            pipe.zcard(f"rate_limit:{key}")
            
            # Add current request
            pipe.zadd(f"rate_limit:{key}", {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(f"rate_limit:{key}", window)
            
            results = pipe.execute()
            current_count = results[1]
            
            return current_count < limit
            
        except Exception as e:
            logger.error(f"❌ Rate limiter error: {str(e)}")
            return True  # Allow if error
    
    def get_remaining_requests(self, key: str, limit: int, window: int) -> int:
        """Get remaining requests for a key."""
        if not self.redis_client:
            return limit
        
        try:
            current_time = int(time.time())
            window_start = current_time - window
            
            # Remove old entries
            self.redis_client.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
            
            # Count current requests
            current_count = self.redis_client.zcard(f"rate_limit:{key}")
            
            return max(0, limit - current_count)
            
        except Exception as e:
            logger.error(f"❌ Rate limiter count error: {str(e)}")
            return limit

# Global rate limiter instance
rate_limiter = RateLimiter()

# Rate limit decorator
def rate_limit(limit: int = 10, window: int = 60):
    """
    Rate limiting decorator.
    
    Args:
        limit: Maximum requests per window
        window: Time window in seconds
    """
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            # Get client IP
            client_ip = request.client.host
            
            # Check rate limit
            if not rate_limiter.is_allowed(client_ip, limit, window):
                remaining = rate_limiter.get_remaining_requests(client_ip, limit, window)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": limit,
                        "window": window,
                        "remaining": remaining,
                        "retry_after": window
                    }
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

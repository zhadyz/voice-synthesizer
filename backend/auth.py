"""
Authentication Module
Provides API key authentication for the backend
"""

from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

# API Key from environment variable
API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")
DEBUG_MODE = os.getenv("DEBUG", "True").lower() == "true"

security = HTTPBearer(auto_error=False)


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Verify API key from header or Bearer token

    Accepts two formats:
    1. X-API-Key: <key>
    2. Authorization: Bearer <key>

    Returns:
        str: User identifier (for logging/tracking)

    Raises:
        HTTPException: 401 if authentication fails
    """
    # Skip auth in debug mode for easier development
    if DEBUG_MODE and API_KEY == "dev-key-change-in-production":
        logger.debug("Authentication skipped (debug mode)")
        return "debug_user"

    # Try X-API-Key header
    if x_api_key and x_api_key == API_KEY:
        return "authenticated"

    # Try Bearer token
    if credentials and credentials.credentials == API_KEY:
        return "authenticated"

    # No valid authentication
    logger.warning("Unauthorized access attempt")
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"}
    )


def get_optional_api_key(
    x_api_key: Optional[str] = Header(None)
) -> Optional[str]:
    """
    Optional authentication for public endpoints

    Returns None if no key provided, validates if key is provided
    """
    if not x_api_key:
        return None

    if x_api_key == API_KEY:
        return "authenticated"

    raise HTTPException(
        status_code=401,
        detail="Invalid API key"
    )

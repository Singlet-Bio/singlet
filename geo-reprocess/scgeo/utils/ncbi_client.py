"""NCBI E-utilities API client with rate limiting and retry logic."""

import time
import logging
from typing import Optional

import requests

from scgeo.config import get_config

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class NCBIClient:
    """Rate-limited, retrying client for NCBI E-utilities.
    
    Automatically handles:
    - Rate limiting (respects API key limits)
    - Exponential backoff on errors
    - Authentication headers
    - JSON/text parsing
    
    Example:
        >>> client = NCBIClient(api_key="YOUR_KEY")
        >>> data = client.get_json("esearch.fcgi", {
        ...     "db": "gds",
        ...     "term": "single cell",
        ...     "retmax": 100
        ... })
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        tool: str = "sc-geo",
        requests_per_second: Optional[int] = None,
    ):
        """Initialize NCBI client.
        
        Args:
            api_key: NCBI API key (falls back to config/env var)
            email: Email for NCBI (falls back to config)
            tool: Tool name for NCBI headers
            requests_per_second: Rate limit (10 with key, 3 without)
        """
        config = get_config()
        
        self.api_key = api_key or config.ncbi.api_key
        self.email = email or config.ncbi.email
        self.tool = tool or config.ncbi.tool
        
        # Set rate limit based on API key
        if requests_per_second is not None:
            self.requests_per_second = requests_per_second
        elif self.api_key:
            self.requests_per_second = 10
        else:
            self.requests_per_second = 3
        
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"{self.tool}/1.0"})
        
        self._min_interval = 1.0 / self.requests_per_second
        self._last_request_time = 0.0
        self._request_count = 0
    
    def _rate_limit(self):
        """Enforce rate limit between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _add_auth(self, params: dict) -> dict:
        """Add authentication params to request."""
        params = dict(params)
        params["tool"] = self.tool
        params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params
    
    def _request_with_retry(
        self,
        url: str,
        max_retries: int = 5,
        params: Optional[dict] = None,
        stream: bool = False,
    ) -> requests.Response:
        """Core retry loop with exponential backoff."""
        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = self.session.get(
                    url, params=params, timeout=60, stream=stream
                )
                self._request_count += 1
                
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 2, 60)
                    logger.warning(
                        f"Rate limited (429). Waiting {wait}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                    continue
                
                if resp.status_code >= 500:
                    wait = min(2 ** attempt * 5, 120)
                    logger.warning(
                        f"Server error ({resp.status_code}). Waiting {wait}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                    continue
                
                resp.raise_for_status()
                return resp
            
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as e:
                wait = min(2 ** attempt * 5, 120)
                logger.warning(
                    f"{type(e).__name__}. Waiting {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
        
        raise RuntimeError(f"Failed after {max_retries} retries: {url}")
    
    def get(
        self,
        endpoint: str,
        params: dict,
        max_retries: int = 5,
    ) -> requests.Response:
        """Rate-limited GET request to E-utility endpoint.
        
        Args:
            endpoint: E-utility endpoint (e.g., "esearch.fcgi")
            params: Query parameters
            max_retries: Maximum retry attempts
            
        Returns:
            requests.Response object
        """
        url = f"{EUTILS_BASE}/{endpoint}"
        params = self._add_auth(params)
        return self._request_with_retry(url, max_retries=max_retries, params=params)
    
    def get_json(
        self,
        endpoint: str,
        params: dict,
        **kwargs
    ) -> dict:
        """GET request returning parsed JSON.
        
        Args:
            endpoint: E-utility endpoint
            params: Query parameters
            **kwargs: Additional arguments for get()
            
        Returns:
            Parsed JSON response
        """
        params["retmode"] = "json"
        resp = self.get(endpoint, params, **kwargs)
        return resp.json()
    
    def get_text(
        self,
        endpoint: str,
        params: dict,
        **kwargs
    ) -> str:
        """GET request returning text.
        
        Args:
            endpoint: E-utility endpoint
            params: Query parameters
            **kwargs: Additional arguments for get()
            
        Returns:
            Response text
        """
        resp = self.get(endpoint, params, **kwargs)
        return resp.text
    
    def get_url(
        self,
        url: str,
        max_retries: int = 5,
        stream: bool = False,
    ) -> requests.Response:
        """Rate-limited GET to an arbitrary URL.
        
        Useful for SRA RunInfo, ENA metadata, etc.
        
        Args:
            url: Full URL to fetch
            max_retries: Maximum retry attempts
            stream: Stream response content
            
        Returns:
            requests.Response object
        """
        return self._request_with_retry(url, max_retries=max_retries, stream=stream)
    
    @property
    def total_requests(self) -> int:
        """Total number of requests made by this client."""
        return self._request_count


# Module-level singleton for convenience
_default_client: Optional[NCBIClient] = None


def get_client() -> NCBIClient:
    """Get default NCBI client (singleton).
    
    Returns:
        Shared NCBIClient instance
    """
    global _default_client
    if _default_client is None:
        _default_client = NCBIClient()
    return _default_client

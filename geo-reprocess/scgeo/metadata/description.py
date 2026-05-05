"""Natural language description fetching from NCBI E-utilities.

Retrieves GSE title, summary, organism, and PubMed IDs via the
NCBI GDS database API.
"""

import json
import logging
import time
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_geo_description(
    gse_id: str,
    email: str = "debruinz@gvsu.edu",
    retries: int = 3,
) -> Dict:
    """Fetch GSE title, summary, and PubMed links from NCBI.

    Uses the GDS ESearch + ESummary pipeline to retrieve the natural
    language description that authors provide when depositing data.

    Args:
        gse_id: GEO series accession (e.g. ``"GSE281311"``).
        email: Contact email for NCBI API (required by their policy).
        retries: Number of retry attempts on transient failures.

    Returns:
        Dictionary with keys ``title``, ``summary``, ``organism``,
        ``n_samples``, ``pubmed_ids``.  Empty dict on failure.
    """
    headers = {"User-Agent": f"cellarium/1.0 ({email})"}

    for attempt in range(retries):
        try:
            # Step 1: ESearch to find GDS UID
            search_url = (
                f"{_EUTILS_BASE}/esearch.fcgi?"
                f"db=gds&term={gse_id}[ACCN]&retmode=json"
                f"&email={email}"
            )
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            uid_list = data.get("esearchresult", {}).get("idlist", [])
            if not uid_list:
                logger.debug("No GDS UID found for %s", gse_id)
                return {}

            # Brief pause between the two API calls
            time.sleep(0.4)

            # Step 2: ESummary to get metadata
            sum_url = (
                f"{_EUTILS_BASE}/esummary.fcgi?"
                f"db=gds&id={uid_list[0]}&retmode=json"
                f"&email={email}"
            )
            req = urllib.request.Request(sum_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                sdata = json.loads(resp.read())

            result = sdata.get("result", {})
            record = result.get(uid_list[0], {})
            return {
                "title": record.get("title", ""),
                "summary": record.get("summary", ""),
                "organism": record.get("taxon", ""),
                "n_samples": record.get("n_samples", 0),
                "pubmed_ids": record.get("pubmedids", []),
            }

        except Exception as e:
            is_rate_limit = "429" in str(e)
            if attempt < retries - 1:
                wait = (10 if is_rate_limit else 2 ** attempt)
                logger.debug(
                    "Retry %d/%d for %s: %s (waiting %ds)",
                    attempt + 1, retries, gse_id, e, wait,
                )
                time.sleep(wait)
            else:
                logger.warning("Failed to fetch GEO description for %s: %s", gse_id, e)
                return {}

    return {}


def fetch_geo_descriptions_batch(
    gse_ids: List[str],
    email: str = "debruinz@gvsu.edu",
    requests_per_second: float = 3.0,
) -> Dict[str, Dict]:
    """Fetch descriptions for multiple GSEs with rate limiting.

    Args:
        gse_ids: List of GEO series accessions.
        email: Contact email for NCBI API.
        requests_per_second: Maximum API request rate.

    Returns:
        ``{gse_id: description_dict}`` for each input GSE.
    """
    results = {}
    interval = 1.0 / requests_per_second

    for i, gse_id in enumerate(gse_ids):
        results[gse_id] = fetch_geo_description(gse_id, email=email)
        if i < len(gse_ids) - 1:
            time.sleep(interval)

    return results

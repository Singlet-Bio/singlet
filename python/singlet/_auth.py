# SPDX-License-Identifier: MIT
"""Retired: authentication for the former token-priced API.

``login()`` configured a key for ``https://api.singlet.bio/v1`` (used by
``query``/``search`` and NMF serving). That API was never brought into
service, so it now raises :class:`NotImplementedError`. The only key the
package uses today is the optional natural-language search key — set it with
:func:`singlet.set_api_key` or ``$SINGLET_API_KEY``.
"""

from __future__ import annotations

from typing import Optional

_RETIRED_MSG = (
    "{name} targeted https://api.singlet.bio/v1, which is not in service. "
    "Downloads need no key: use singlet.load('GSE…') and singlet.find('…'). "
    "To raise the natural-language search limit, create a key at "
    "https://singlet.bio/account and call singlet.set_api_key(key) "
    "(or set $SINGLET_API_KEY)."
)


def retired(name: str) -> str:
    """Error message for a function that needed the retired v1 API."""
    return _RETIRED_MSG.format(name=name)


_LOGIN_MSG = (
    "singlet.login() has been removed: it stored a key for https://api.singlet.bio/v1, "
    "an API that was never in service, so the key was never used. To fix your "
    "script, delete the singlet.login(...) line. singlet.load('GSE…'), "
    "singlet.download() and singlet.open_bundle() need no key. If the key is for "
    "natural-language search, call singlet.set_api_key(key) instead — or just set "
    "$SINGLET_API_KEY, which singlet.find() reads on its own."
)


def login(api_key: Optional[str] = None) -> None:
    """Removed — raises :class:`NotImplementedError` explaining what to do.

    ``login()`` used to store a key for the retired ``api.singlet.bio/v1``
    API without contacting it, so scripts could call it harmlessly. It now
    raises so those calls surface: delete them (downloads need no key), or
    replace them with :func:`singlet.set_api_key` for the natural-language
    search key. The key is never echoed in the message.
    """
    raise NotImplementedError(_LOGIN_MSG)

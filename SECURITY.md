# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 2.x     | ✅ Active          |
| 1.x     | ❌ End of life     |

## Reporting a Vulnerability

If you discover a security vulnerability in singlet, please report it
**privately** via GitHub Security Advisories:

1. Go to <https://github.com/Singlet-Bio/singlet/security/advisories/new>
2. Fill in the details (affected component, severity, reproduction steps)
3. We will respond within 72 hours

**Do NOT open a public issue for security vulnerabilities.**

## Scope

This policy covers:

- The `singlet` Python package (`python/singlet/`)
- The `singlepress` C++ library (`include/`, `src/`)
- The MCP server (`python/singlet/mcp/`)
- File format parsers (.1pz, .spz)

Out of scope:

- The singlet.bio web API (report to security@singlet.bio)
- Third-party dependencies (report to their respective maintainers)

## Security Measures

- Dependencies are audited via `pip-audit` in CI
- File format parsers are fuzz-tested against malformed inputs
- No secrets are stored in the repository
- All network calls use HTTPS with certificate verification

# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | ✅ Active          |
| 0.x and the unreleased 2.x development builds | ❌ Not supported |

## Reporting a Vulnerability

If you discover a security vulnerability in singlet, please report it
**privately** via GitHub Security Advisories:

1. Go to <https://github.com/Singlet-Bio/singlet/security/advisories/new>
2. Fill in the details (affected component, severity, reproduction steps)
3. We will respond within 72 hours

**Do NOT open a public issue for security vulnerabilities.**

## Scope

This policy covers:

- The `singlet-bio` Python package (`python/singlet/`) and the `singlet` R client (`r/`)
- The C++ library (`include/`, `src/`)
- The MCP server (`python/singlet/mcp/`)
- File format parsers (`.1pz`, `.singlet`)

Out of scope:

- The singlet.bio web API (report to security@singlet.bio)
- Third-party dependencies (report to their respective maintainers)

## Security Measures

- Dependencies are audited with `pip-audit` in CI, and a finding fails the build
- The `.1pz` reader treats its input as untrusted: every size, offset and index is
  bounds-checked and allocations are capped (`ReadLimits`). `tests/cpp/test_pz_hardening.cpp`
  feeds it truncated and mutated files, and a libFuzzer target (`tests/fuzz/`) runs under
  AddressSanitizer and UBSan in CI (`.github/workflows/pz-hardening.yml`)
- Downloaded `.singlet` bundles are verified before they are cached: byte count,
  archive structure, schema version and the SHA-256 of every member
- Accessions are validated before they reach a file name or a URL
- No secrets are stored in the repository
- The default data and API hosts are reached over HTTPS with certificate verification
  (`SINGLET_DATA_BASE` and `SINGLET_API_BASE` can point elsewhere, for example at a local test server)

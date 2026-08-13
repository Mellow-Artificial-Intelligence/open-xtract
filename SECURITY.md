# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.8.x   | :white_check_mark: |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing the maintainers directly rather than opening a public issue.

**Please include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

We will acknowledge receipt within 48 hours and aim to provide a fix within 7 days for critical issues.

## URL input security model

When `input_file` is an `http://` or `https://` URL, openextract fetches it over
HTTP. Callers that pass untrusted URLs should treat this as a privileged
operation. The fetcher applies best-effort SSRF defenses; it does **not** make
arbitrary URL fetching safe in every environment.

### Supported schemes

- `http://`
- `https://`

Other schemes are not fetched as URLs.

### Host validation

Unless opted out, openextract refuses hosts that are missing/empty, fail DNS
resolution, or resolve to any non-public address, including:

- Private RFC 1918 ranges
- Loopback
- Link-local (including cloud metadata `169.254.169.254`)
- Multicast and other reserved ranges
- IPv4 and IPv6, including IPv4-mapped IPv6 (for example `::ffff:127.0.0.1`)

Implementation and tests: `src/openextract/_media.py` (`_is_safe_host`,
`_fetch_url`) and the SSRF section of `tests/test_extract.py`.

### Redirect handling

Redirects are followed manually with `follow_redirects=False` at the HTTP
client layer. The host is re-validated on **every** hop so a public URL cannot
redirect to an internal target.

Default maximum hops: `10` (`OPENEXTRACT_MAX_REDIRECTS`).

### Configuration

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `OPENEXTRACT_URL_TIMEOUT` | `30` | HTTP timeout in seconds |
| `OPENEXTRACT_MAX_REDIRECTS` | `10` | Maximum redirect hops |
| `OPENEXTRACT_ALLOW_PRIVATE_URLS` | unset | Set to `1` / `true` / `yes` to disable host validation |
| `OPENEXTRACT_MAX_INPUT_BYTES` | `52428800` | Maximum bytes loaded per input |

Invalid or non-positive timeout/redirect values fall back to the defaults. An
invalid or non-positive input-size limit raises `ValueError` so a bad
configuration cannot silently disable the cap.

`OPENEXTRACT_ALLOW_PRIVATE_URLS` is intended for trusted environments (local
tests, on-prem services). Enabling it removes the private-host guardrail.

### What is protected / not guaranteed

**Protected (best effort):**

- Direct requests to private/loopback/link-local/metadata IPs
- Redirect chains that land on non-public hosts
- Basic timeout and redirect-count limits
- A per-input 50 MiB default cap for paths, URLs, bytes, and binary streams
- URL response streaming that enforces the cap when `Content-Length` is absent or incorrect

**Not guaranteed:**

- DNS rebinding (a hostname resolving to different IPs across lookups)
- Safety of the model provider after bytes are fetched
- Non-HTTP SSRF channels outside this fetcher

If you need a one-off internal fetch without disabling validation globally,
download the bytes with your own HTTP client and pass them to `extract()` as
`bytes` or a file-like object with an explicit `media_type`.

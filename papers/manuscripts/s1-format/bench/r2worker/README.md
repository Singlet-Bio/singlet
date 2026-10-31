# s1bench-r2: the real object store of `netsweep --r2`

A private Cloudflare R2 bucket (`s1bench`) served by byte range through this Worker. Reads go to
`https://s1bench-r2.<subdomain>.workers.dev/<READ_KEY>/<key>` (the key in the path is the read capability); writes also
need the `x-s1-write: <WRITE_KEY>` header. Both are Worker secrets, generated once and never committed.

    npx wrangler r2 bucket create s1bench
    npx wrangler deploy
    printf %s "$READ_KEY"  | npx wrangler secret put READ_KEY
    printf %s "$WRITE_KEY" | npx wrangler secret put WRITE_KEY

On Clipper, `$S1/secrets/r2bench.env` (mode 600) holds `S1_R2_URL=https://.../<READ_KEY>` and `S1_R2_WRITE=<WRITE_KEY>`;
the timed job's `netsweep_r2` stage sources it, uploads the panel datasets' files from the cluster (never from a laptop)
and reads them back. The Worker adds its own invocation to every request; R2's own S3 endpoint would need API keys.

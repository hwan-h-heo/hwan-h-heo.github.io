# Cloudflare Worker Legacy Redirects

This Worker provides true HTTP permanent redirects for legacy blog URLs such as:

```text
/blogs/posts/?id=240823_grt
```

It maps known IDs from `blogs/data/legacy-post-redirects.json` to current slug URLs and returns `308 Permanent Redirect`. `308` is used consistently because these are permanent URL migrations and it preserves the request method. For normal browser `GET` requests it behaves like a `301` for indexing purposes.

## Platform Constraint

GitHub Pages cannot issue query-parameter-specific `301` or `308` redirects. The repository fallback page improves user compatibility, but it is still JavaScript-based.

This Worker only works when traffic is routed through Cloudflare, typically with a custom domain you control. It cannot transparently change redirect behavior for the raw `hwan-h-heo.github.io` hostname because that hostname is controlled by GitHub.

## Configuration

1. Copy `wrangler.toml.example` to `wrangler.toml`.
2. Replace the route pattern with the custom domain that fronts the site.
3. Keep `GITHUB_PAGES_ORIGIN` set to `https://hwan-h-heo.github.io`.
4. Set `CANONICAL_ORIGIN` to the public canonical origin you want redirects to use.
5. Run `wrangler dev` to test locally.
6. Deploy with `wrangler deploy` after Cloudflare DNS/routes are configured.

The Worker proxies unrelated requests to the GitHub Pages origin unchanged.

## Redirect Map Maintenance

The Worker imports:

```text
../../../blogs/data/legacy-post-redirects.json
```

Update that JSON file when a published post ID or slug mapping changes, then run:

```bash
npm run build
npm run validate:seo
```

Do not store credentials or private keys in this directory.

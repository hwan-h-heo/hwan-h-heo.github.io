# Search Indexing Operations

Date: 2026-07-28

These steps happen after deploying the generated `blogs/dist/` output.

## Google Search Console

1. Add and verify the site property for the deployed origin.
2. Submit:

```text
https://hwan-h-heo.github.io/sitemap.xml
```

3. Inspect the blog homepage:

```text
https://hwan-h-heo.github.io/blogs/
```

4. Inspect representative article URLs, especially newly published or migrated URLs.
5. For important migrated URLs, inspect the old URL and confirm whether Google sees a redirect or the repository fallback. True query-specific permanent redirects require the Cloudflare Worker or another edge layer.
6. Request indexing for important new canonical URLs after deployment.

Common statuses:

- `Discovered - currently not indexed`: Google knows the URL but has not crawled it yet. Confirm it is in the sitemap and linked from `/blogs/`.
- `Crawled - currently not indexed`: Google crawled the page but chose not to index it yet. Check content uniqueness, canonical, and internal links.
- `Duplicate`: inspect the selected canonical. Translated pages should self-canonicalize and use reciprocal `hreflang`.
- `Google chose different canonical`: compare the page canonical, sitemap URL, and internal links. They should all point to the same slug URL.
- `Redirect error`: check whether the old URL is being handled by a real HTTP redirect at the edge. GitHub Pages cannot do query-specific `301` or `308` redirects by itself.

## Bing Webmaster Tools

1. Add and verify the site.
2. Submit:

```text
https://hwan-h-heo.github.io/sitemap.xml
```

3. Use URL Inspection for:

```text
https://hwan-h-heo.github.io/blogs/
https://hwan-h-heo.github.io/blogs/posts/optimizing-sparse-3d-generation-inference/
```

4. Verify that representative English and Korean article URLs are crawlable and self-canonical.

## IndexNow

IndexNow can be integrated safely later, but it is intentionally not mandatory for the static build.

Recommended approach if added:

- Generate an IndexNow key outside the repository.
- Store the key as a GitHub Actions secret.
- Publish the key file during deployment without committing the private key.
- Submit only added, updated, or deleted URLs.
- Do not submit the entire site on every build.
- Do not fail the static deployment if the IndexNow API is unavailable.

Current status: not implemented. The core indexing path is sitemap submission plus static internal links from `/blogs/`, tag archives, series archives, and RSS.

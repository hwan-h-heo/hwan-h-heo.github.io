# Search Indexing Operations

Date: 2026-07-28

These steps happen after deploying the generated `blogs/dist/` output.

## Build and CI checks

Portfolio identity lives in `content/portfolio/home.json`: Home/About supply
the name, role, biography and profile links, and `seo` supplies organization,
education, expertise and the sharing-image source. The build generates the
home's title, descriptions, social metadata and linked `WebSite`, `ProfilePage`
and `Person` structured data. Articles, the Blog index, publications and project
pages reference that same Person ID. Chapter navigation preserves the title.

Selected writing is rendered from `featuredPortfolioPosts` in
`blogs/data/site-data.json` into the initial home HTML. Its browser script keeps
those nodes and attaches cover behavior; source previews retain a client-side
fallback. Full and incremental builds regenerate the home.

Run `npm run build`, `npm run validate:seo` and `npm run check:seo:render`.
The static validator covers the portfolio home, projects, published article
routes and archives, including current social-image sources and actual JPEG
dimensions. A fixture regression also replaces an image at the same path and
modification time to verify that thumbnails, deployed search data and social
images receive fresh URLs and pixels. The browser check covers mobile and
desktop, JavaScript disabled,
normal hydration and blocked JSON requests. It also checks title stability
through chapter navigation. GitHub Actions runs both checks, and the deploy
paths retain them as verification gates. These checks validate implementation;
Google's actual indexing state still requires Search Console.

## Google Search Console

1. Add and verify the site property for the deployed origin.
2. Submit:

```text
https://hwan-h-heo.github.io/google-sitemap.xml
```

The legacy `/sitemap.xml` remains available with identical content. Submit the new URL as a separate sitemap entry to bypass any failed-fetch state cached against the old sitemap URL.

If Search Console still reports that it cannot read the new XML URL, submit the plain-text diagnostic sitemap separately:

```text
https://hwan-h-heo.github.io/google-sitemap.txt
```

If TXT succeeds while XML fails, investigate XML-specific parsing or response handling. If both fail, treat the problem as a Google Search Console to GitHub Pages fetch issue rather than a sitemap-content issue.

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

# SEO Refactor Result

Date: 2026-07-28

## Summary

The blog build now produces crawlable static HTML for the blog index, post navigation, archive pages, related links, language alternates, canonical metadata, sitemap entries, robots.txt, and RSS. JavaScript remains as progressive enhancement for interaction, but search-critical links and metadata are present in the initial generated HTML.

The SEO validation script runs against `blogs/dist/` after the static build and fails on missing post links, sitemap/canonical mismatches, invalid hreflang relationships, invalid structured data, unresolved internal links, malformed redirect maps, and Cloudflare Worker redirect regressions.

## Changed Architecture

- `blogs/data/site-data.json` remains the single source of truth for published post metadata.
- `blogs/build-static.js` now generates the blog homepage, post pages, archive pages, sitemap, RSS feed, robots.txt, and legacy redirect fallback from shared route helpers.
- `blogs/lib/site-routes.js` centralizes canonical post routes, archive routes, public route enumeration, and sitemap entries.
- `blogs/lib/seo-utils.js` centralizes URL, metadata, hreflang, post-preview, breadcrumb, tag, series, and related-post rendering helpers.
- `blogs/lib/render-blog-index.js` renders `/blogs/` as static HTML with featured posts, regular posts, notes, and series links.
- `blogs/lib/render-post-page.js` renders static breadcrumbs, language alternates, series navigation, related posts, chronological navigation, article metadata, and JSON-LD.
- `blogs/lib/render-archive-page.js` renders static series and tag archive pages.
- `blogs/data/legacy-post-redirects.json` stores the explicit legacy ID to canonical URL mapping.
- `deploy/cloudflare-worker/src/index.js` provides the true HTTP permanent redirect layer for fronted deployments.
- `scripts/validate-seo-output.js` validates the generated output rather than source templates only.

## Completed P0 Tasks

- Statically rendered `/blogs/` with crawlable `<a href>` links for all 26 published posts.
- Preserved no-JavaScript discoverability for the primary blog index DOM.
- Statically rendered previous/next post links, series navigation, language alternate navigation, breadcrumbs, related-post links, and return-to-blog links.
- Generated a sitemap with absolute HTTPS URLs, canonical post-language URLs, archive URLs, project URLs, and sitemap-level XHTML alternates for translation pairs.
- Generated root `robots.txt` with `Allow: /` and `Sitemap: https://hwan-h-heo.github.io/sitemap.xml`.
- Centralized legacy query redirect mapping in `blogs/data/legacy-post-redirects.json`.
- Replaced the repository-level fallback with deterministic inline mapping, clear unknown-ID handling, canonical declaration where practical, and `location.replace()`.
- Added generated-output SEO validation through `npm run validate:seo`.

## Completed P1 Tasks

- Ensured each article page has exactly one primary `<h1>` matching the article title.
- Changed sidebar site-name headings outside articles to avoid competing `<h1>` elements.
- Generated distinct page titles for blog index, article pages, archive pages, redirect pages, and error/fallback pages.
- Generated unique meta descriptions with explicit metadata first and a prose-derived fallback where needed.
- Ensured each indexable generated page has exactly one absolute HTTPS canonical URL.
- Added reciprocal page-level `hreflang` for English/Korean translation pairs and `x-default`.
- Ensured untranslated pages do not invent language alternates.
- Added absolute Open Graph and Twitter metadata, including a fallback social image.
- Improved article `BlogPosting` JSON-LD with valid dates, language, URL, author, publisher, and image fields.
- Added `Blog`/`WebSite` structured data for the blog homepage and `CollectionPage` structured data for archives.
- Rendered publication dates with semantic `<time datetime="...">`.

## Completed P2 Tasks

- Added deterministic related-post links based on series, tags, category, and language-safe route selection.
- Added static series archive pages for series with at least two posts.
- Added static tag archive pages for tags with at least two posts.
- Added static breadcrumbs to post pages and matching breadcrumb JSON-LD.
- Generated `/blogs/feed.xml` and added feed autodiscovery to the blog homepage.
- Extended metadata support for `seoTitle`, language-specific SEO titles, `socialImage`, and `translationKey`.
- Excluded draft posts from public pages, sitemap, RSS, related-post lists, and validation expectations.

## Intentionally Deferred

- IndexNow was documented but not implemented. It should run only from deployment automation with repository secrets and URL-delta awareness.
- Single-post tag archives were not generated to avoid thin archive pages. Current multi-post tag archives are generated.
- The Cloudflare Worker was not deployed because deployment requires account credentials and a fronted hostname configuration.

## GitHub Pages Limitations

GitHub Pages cannot issue query-parameter-specific HTTP 301 or 308 redirects for URLs such as `/blogs/posts/?id=240823_grt`. The repository fallback improves user compatibility only. True permanent redirects require an edge layer such as Cloudflare Workers in front of the GitHub Pages origin, usually through a custom domain or compatible fronting setup.

## Edge Redirect Requirements

The Worker in `deploy/cloudflare-worker/`:

- inspects `/blogs/posts/` requests;
- reads the `id` query parameter;
- maps known IDs to canonical slug URLs;
- returns HTTP `308 Permanent Redirect`;
- proxies nonlegacy requests to the configured GitHub Pages origin;
- avoids redirect loops.

Deployment requires copying `wrangler.toml.example` to a real Wrangler config, setting `GITHUB_PAGES_ORIGIN` and `CANONICAL_ORIGIN` as needed, and attaching the Worker to the controlled hostname.

## Verification Commands

Commands run from the repository root:

```bash
npm.cmd run check:content
npm.cmd run check:assets
npm.cmd run build
npm.cmd run validate:seo
npm.cmd test --if-present
npm.cmd run check:render
```

Results:

- `check:content`: passed, `26 published posts, 3 portfolio blocks`.
- `check:assets`: passed, `56 content files, 334 local media references, 93 localized assets`; existing large-media warnings only.
- `build`: passed, generated `43 post-language pages`, `6 series archives`, `9 tag archives`, `sitemap.xml`, `robots.txt`, and `blogs/feed.xml`.
- `validate:seo`: passed, `26 published posts, 43 post-language routes`.
- `npm test --if-present`: exited successfully; no test script is currently defined.
- `check:render`: passed after installing Playwright Chromium locally, `75 routes, 680 unique local media resources`.

## Representative Generated Output Inspected

- `blogs/dist/blogs/index.html`
  - title: `3D Generative AI and CUDA Engineering | Hwan Heo`
  - canonical: `https://hwan-h-heo.github.io/blogs/`
  - one `<h1>`: `Research notes for 3D AI systems`
  - static post links present
  - homepage JSON-LD present

- `blogs/dist/blogs/posts/optimizing-sparse-3d-generation-inference/index.html`
  - title: `What the Compiler Missed in Sparse 3D Inference | Hwan Heo`
  - canonical: `https://hwan-h-heo.github.io/blogs/posts/optimizing-sparse-3d-generation-inference/`
  - one `<h1>` matching the article title
  - reciprocal English/Korean/x-default alternates present
  - breadcrumbs, related posts, post navigation, Open Graph, Twitter metadata, and JSON-LD present

- `blogs/dist/blogs/posts/optimizing-sparse-3d-generation-inference-kor/index.html`
  - Korean title, Korean canonical URL, `html lang="ko"`, reciprocal alternates, and article JSON-LD present

- `blogs/dist/blogs/posts/dont-rasterize-triangle/index.html`
  - legacy article now has slug canonical, article metadata, and static internal links

- `blogs/dist/blogs/posts/sdf-and-eikonal-equation/index.html`
  - note page has canonical, single `<h1>`, static navigation, and JSON-LD

- `blogs/dist/blogs/series/3d-generation/index.html`
  - static archive links, canonical, metadata, and `CollectionPage` JSON-LD present

- `blogs/dist/blogs/tags/neural-rendering/index.html`
  - static tag archive links, canonical, metadata, and `CollectionPage` JSON-LD present

- `blogs/dist/blogs/feed.xml`
  - valid RSS 2.0 feed with canonical post links and no legacy query URLs

## Documentation Added

- `docs/seo-refactor-audit.md`
- `docs/search-indexing-operations.md`
- `docs/seo-refactor-result.md`
- `deploy/cloudflare-worker/README.md`

## CI Integration

Added `.github/workflows/validate-site.yml` to run:

- root dependency install with `npm ci`;
- blog dependency install with `npm --prefix blogs ci`;
- content validation;
- asset validation;
- static build;
- SEO validation;
- existing tests with `npm test --if-present`.

The workflow is validation-only and does not deploy.

## Remaining Manual Search Console Actions

- Deploy the generated site normally.
- Register or verify the site in Google Search Console.
- Submit `https://hwan-h-heo.github.io/sitemap.xml`.
- Inspect `/blogs/` and representative article URLs.
- Request indexing for important newly migrated canonical URLs.
- Check for `Discovered - currently not indexed`, `Crawled - currently not indexed`, `Duplicate`, `Google chose different canonical`, and `Redirect error` statuses.
- Add the site to Bing Webmaster Tools and submit the same sitemap.
- Deploy the Cloudflare Worker only after configuring a controlled fronting hostname.

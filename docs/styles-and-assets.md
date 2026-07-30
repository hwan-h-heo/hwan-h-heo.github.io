# Styles and static assets

## Stylesheet ownership

The deployed site has two independent style surfaces.

- `assets/css/portfolio.css` is the shared entry point for the portfolio home and current project pages.
- `blogs/css/blog.css` contains the reduced framework surface and shared blog components.
- `blogs/css/typography.css` owns blog type and article reading styles.
- `blogs/css/sidebar.css` owns the blog sidebar.
- `blogs/css/post.css` owns generated post-only and rich-content components.
- `blogs/css/code-copy.css` and `blogs/css/scroll-progress.css` are loaded only when those post features are enabled.
- `blogs/editor/editor.css` is private to the local blog editor.

Keep selectors in the narrowest stylesheet that owns the markup. Do not add a second
site-wide entry point or restore the retired `used.css`, `main.css`,
`project-legacy.css`, `blog_post_specific.css`, `blog_style.css`, or
`styles.css` paths.

## Vendor files

`assets/vendor/` contains only the browser build that the site loads for each
library. Do not copy whole package distributions into this directory. In
particular, source maps, RTL variants, CommonJS/ESM builds, and unminified
duplicates should stay in package caches rather than the deployment tree.

## Media lifecycle

Post media belongs in `blogs/posts/<post-id>/assets/`. Project media belongs in
`projects/<project>/assets/`. Shared portfolio and runtime media belongs in
`assets/`.

### Sidebar navigation

`css/sidebar-nav.css` and `js/sidebar-controller.js` own the shared responsive
sidebar behavior for portfolio, project, blog index, search, and Labs pages.
Desktop sidebars switch between a 300px panel and a persisted 72px icon rail;
mobile sidebars use an overlay drawer. The portfolio home reserves the saved
sidebar width for sections below the full-width hero, so revealing the sidebar
does not reflow the page. Sidebar content briefly crossfades while the rail
changes width to keep labels from reflowing visibly. Keep page-specific sidebar
colors and Labs layout in the page's existing stylesheet instead of duplicating
the collapse logic.

### Blog cover previews

Keep each post's full-quality `cover` in `blogs/data/site-data.json`. The static
build generates a 960×600 WebP list preview under
`blogs/dist/assets/generated/blog-covers/`; generated previews are deployment
output and must not be edited or committed by hand.

Blog indexes, search results, archive pages, and the portfolio's featured blog
list use these previews with native lazy loading. If the configured cover or
portfolio teaser is a GIF, the static preview remains the default and the
original animation is requested only on pointer hover or keyboard focus.
Reduced-motion and touch-first browsing keep the static preview.
Set `"animatedPreview": true` on a post only when its GIF motion is part of the
preview design. That GIF starts automatically when its preview is within 300px
of the viewport; the generated WebP remains the lightweight placeholder.

Post and project content images preserve explicit alt text. During the build,
missing or placeholder alt text receives a heading-aware fallback and images
receive lazy-loading and asynchronous-decoding attributes. Prefer writing a
specific alt description in Markdown whenever the surrounding context alone is
not sufficient.

Run the following after replacing or removing media:

```bash
npm run audit:assets
npm run check:assets
npm run build
```

The static asset audit is intentionally conservative: it reports a file only
when its filename appears nowhere in current tracked source. Generated output
and ignored editor drafts do not count as live references.

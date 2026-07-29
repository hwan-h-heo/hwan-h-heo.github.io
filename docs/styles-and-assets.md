# Styles and static assets

## Stylesheet ownership

The deployed site has two independent style surfaces.

- `assets/css/portfolio.css` is the shared entry point for the portfolio home and current project pages.
- `blogs/css/blog.css` contains the reduced framework surface and shared blog components.
- `blogs/css/typography.css` owns blog type and article reading styles.
- `blogs/css/sidebar.css` owns the blog sidebar.
- `blogs/css/post.css` owns generated post-only and rich-content components.
- `blogs/css/code-copy.css` and `blogs/css/scroll-progress.css` are loaded only when those post features are enabled.
- `blogs/editor/editor.css` and `blogs/editor/portfolio.css` are private to the local editors.

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

Run the following after replacing or removing media:

```bash
npm run audit:assets
npm run check:assets
npm run build
```

The static asset audit is intentionally conservative: it reports a file only
when its filename appears nowhere in current tracked source. Generated output
and ignored editor drafts do not count as live references.

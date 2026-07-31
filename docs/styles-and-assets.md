# Styles and static assets

## Stylesheet ownership

The deployed site has two independent style surfaces.

- [`portfolio-design-language.md`](portfolio-design-language.md) defines the
  visual grammar and review rules for portfolio-facing UI.
- `assets/css/portfolio.css` is the framework-free shared entry point for the portfolio home and current project pages. It owns the document reset, the responsive `portfolio-shell`, and shared portfolio/sidebar foundations.
- `assets/css/project-detail.css` owns project-page layout and content components, including the project overview columns and responsive media treatments.
- `blogs/css/blog.css` owns the framework-free blog shell, responsive reading layout, and shared blog components.
- `blogs/css/typography.css` owns blog type and article reading styles.
- `blogs/css/sidebar.css` owns the blog sidebar.
- `blogs/css/post.css` owns generated post-only and rich-content components.
- `blogs/css/code-copy.css` and `blogs/css/scroll-progress.css` are loaded only when those post features are enabled.
- `blogs/editor/editor.css` is private to the local blog editor.
- `assets/css/site-icons.css`, `assets/js/site-icons.js`, and
  `assets/icons/site-icons.svg` provide the shared first-party SVG icon
  surface. Use `SiteIcons.render()` for generated markup and `SiteIcons.set()`
  for state changes; do not introduce icon-font classes or a second icon
  runtime.

Keep selectors in the narrowest stylesheet that owns the markup. Do not add a second
site-wide entry point or restore the retired `used.css`, `main.css`,
`project-legacy.css`, `blog_post_specific.css`, `blog_style.css`, or
`styles.css` paths.

Post link-copy controls are generated once by `blogs/build-static.js`. Do not
embed `copyButton` or `myshare_modal` markup in post Markdown; the content check
rejects those legacy copies. Hand-authored table-of-contents markup remains
supported for posts whose curated hierarchy differs from the generated heading
outline.

## Vendor files

`assets/vendor/` has no tracked runtime files. The portfolio, blog shell,
post interactions, and shared icons do not load a third-party UI framework or
icon font. Do not restore a site-wide vendor bundle for layout, components, or
utilities.

The 3D viewer is the separate exception for non-UI libraries. During a build,
the pinned packages under `blogs/node_modules/` are copied selectively into
`blogs/dist/vendor/three/` and `blogs/dist/vendor/tween/`. These generated files
are not authored deployment assets and must not be edited or committed.

The site icon sprite contains only the selected upstream SVG paths needed by
the UI and has no runtime package dependency. Keep the accompanying MIT
license at `assets/icons/LICENSE.site-icons.txt`; when adding an icon,
regenerate the selected sprite with `scripts/build-site-icon-sprite.js` and
verify every `<use>` fragment against a real symbol.

The sprite generator intentionally takes an extracted upstream icon directory
instead of adding a package to the runtime dependency graph. To reproduce the
current 1.11.3 source set, use a temporary npm package extraction:

```bash
icon_tmp_dir="$(mktemp -d)"
npm pack bootstrap-icons@1.11.3 --pack-destination "$icon_tmp_dir"
tar -xzf "$icon_tmp_dir/bootstrap-icons-1.11.3.tgz" -C "$icon_tmp_dir"
node scripts/build-site-icon-sprite.js "$icon_tmp_dir/package/icons" assets/icons/site-icons.svg
npm run build
npm run check:legacy-ui
```

Keep the version in the sprite provenance comment aligned with the extracted
package and retain its matching `LICENSE.site-icons.txt`. The temporary package
is a maintenance input only; do not add it to either `package.json`.

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
Set a post-level `"previewImage"` when a specific still frame should be used
instead of the GIF's first frame; keep the GIF in `"cover"` so hover playback
continues to work. A featured portfolio entry may point `"teaserImage"` to the
same still.
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

Run `npm run check:legacy-ui` after the build whenever shared markup, styles,
scripts, or icons change. The check scans authored runtime files and generated
output, while narrowly allowing the icon provenance/license and the editor's
initial-data API terminology.

The static asset audit is intentionally conservative: it reports a file only
when its filename appears nowhere in current tracked source. Generated output
and ignored editor drafts do not count as live references.

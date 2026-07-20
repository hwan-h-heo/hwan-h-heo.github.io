# Agent Worklog

## 2026-06-26 22:03 KST

### Changed
- Created the Phase 1 maintenance audit in `docs/maintenance-audit.md`.
- Updated `docs/refactor-guide.md` to distinguish first-pass required docs from later-phase docs.
- Documented current build, routing, portfolio/blog data, editor behavior, asset handling, deployment assumptions, risks, and a phase-based migration plan.

### Preserved
- No UI rewrite was performed.
- Existing source content, public route structure, project sources, blog posts, and build scripts were left unchanged.
- `blogs/dist/` was not edited by hand.

### Validation
- `npm run build` passed on 2026-06-26. The build regenerated project pages, blog post pages, sitemap, robots.txt, and `blogs/dist`.

### TODO / Risks
- Add `docs/architecture.md` and `docs/refactor-checklist.md` in Phase 2 before implementation.
- Freeze current generated route inventory before changing blog slugs or post metadata.
- Add asset/content checkers before moving files.

## 2026-06-26 22:17 KST

### Changed
- Added `docs/architecture.md`, `docs/refactor-checklist.md`, `docs/blog-authoring-workflow.md`, `docs/content-maintenance-guide.md`, `docs/asset-handling-audit.md`, and `docs/public-route-compatibility.md`.
- Moved portfolio home/about/resume content into `content/portfolio/home.json`.
- Added `js/portfolio-blocks.js` to render portfolio blocks into the existing root page sections.
- Extended blog metadata with pinned `slug`, `description_*`, `tags`, `cover`, `status`, and `updated` fields.
- Updated blog list/search/article rendering to use descriptions, tags, cover images, and draft filtering.
- Added root scripts: `new:post`, `new:project`, `check:content`, `check:assets`, and `freeze:routes`.

### Preserved
- Existing public blog and project routes are documented in `docs/public-route-compatibility.md`.
- Existing post bodies, project sources, portfolio cards, publications, talks, and deployment model were preserved.
- No backend/database or framework rewrite was introduced.

### Validation
- `npm run freeze:routes` passed and regenerated route compatibility docs.
- `npm run check:content` passed.
- `npm run check:assets` passed after checker path-resolution fixes.
- `npm run build` passed and generated 25 published posts.
- Local HTTP checks against `http://localhost:8080/` passed for root HTML, `/content/portfolio/home.json`, a representative blog post, and a representative project page.
- No lint/typecheck scripts are defined in `package.json` or `blogs/package.json`.

### TODO / Risks
- Add editor UI fields for description/tags/cover/status; the server currently preserves these fields but the form does not expose them.
- Consider build-time rendering for portfolio blocks if no-JavaScript support becomes important.
- Expand asset checking to crawl generated HTML and optionally verify external URLs.

## 2026-06-26 22:36 KST

### Changed
- Replaced all published blog post fallback covers with post-specific images.
- Downloaded rendered remote blog images into each post's local `assets/` directory and rewrote Markdown/HTML references.
- Downloaded rendered remote project images into each project's local `assets/` directory and rewrote project Markdown references.
- Normalized blog `cover` and featured `teaserImage` paths to site-root local paths.
- Tuned blog/search card layout: wider list column, smaller card title/subtitle typography, and steadier thumbnail sizing.
- Strengthened `check:content` and `check:assets` so fallback covers and remote rendered images are caught.

### Preserved
- Existing blog routes, project routes, post bodies, and project page source model were preserved.
- External videos, code examples, and ordinary reference links were left untouched because they are not rendered thumbnail/image assets.

### Validation
- `npm run check:content` passed.
- `npm run check:assets` passed.
- `npm run build` passed and regenerated blog/project output.
- Dist `blogs/data/site-data.json` has 0 fallback covers, 0 remote covers, and 0 remote featured teaser images.

### TODO / Risks
- Some older computer-vision notes use SVG figures as covers because those posts do not have richer raster teaser assets.
- Visual browser QA is still useful for judging exact thumbnail crops across viewport widths.

## 2026-06-26 22:45 KST

### Changed
- Fixed generated blog post image paths for single-quoted `./assets/...` references.
- Fixed generated blog post image paths for legacy cross-post references such as `./210302_cv1/assets/...`.
- Kept post body images rooted at `/blogs/posts/<post_id>/assets/...` in generated HTML so nested slug routes do not resolve them incorrectly.

### Validation
- `npm run build` passed.
- Generated 43 post HTML files now have 0 relative rendered image references and 0 missing local image files.
- Dist post covers all resolve to existing local files.
- `npm run check:content` passed.
- `npm run check:assets` passed.
- Local HTTP checks passed for `http://localhost:8080/blogs/` and a representative post page.

### Follow-up
- Localized two broken Neuralangelo videos in the SDF post because the old NVIDIA `/labs/dir/...` MP4 URLs now return 403 and the replacement files are small.
- Kept large remote videos external by policy; `check:assets` now warns for local GIF/video references over roughly 15 MiB instead of failing.

## 2026-07-21 KST

### Changed
- Localized all rendered remote blog and project media, and converted the five large 2DGS Viewer demonstrations from GIF to local MP4 files.
- Moved portfolio home/about/resume content into structured JSON with build-time rendering and a browser fallback.
- Added complete publishing metadata and matching editor fields for slugs, descriptions, tags, covers, status, and updated dates.
- Centralized public route generation for builds, sitemap output, route documentation, and browser verification.
- Excluded editor drafts, draft assets, and project snapshots from deploy output.
- Replaced the 3D viewer's remote Three.js, Tween.js, texture, and HDR dependencies with pinned local build assets.
- Replaced public social icon CDN dependencies with the repository's local Bootstrap Icons files.
- Added full source-asset checking and Playwright rendering checks for every public route.
- Aligned canonical URLs, sitemap output, social metadata, and production verification with `https://hwan-h-heo.github.io` through a shared site configuration.

### Validation
- `npm run verify` passed for 25 published posts and 3 portfolio blocks.
- Browser verification passed for all 58 public routes and 667 unique local media resources.
- Desktop and 390 px mobile screenshots were reviewed for the portfolio, blog, search, a project page, and the 3D viewer.
- The 3D viewer loaded its default GLB, produced nonblank canvas pixels, and kept its mobile controls inside the viewport.
- Root and blog dependency audits reported 0 vulnerabilities.

### Known Warnings
- Three older local media files remain above the advisory 15 MiB threshold: two NeRF-in-game MP4 files and the 2DGS training GIF.
- External embeds and ordinary outbound links are intentionally outside the asset availability check.

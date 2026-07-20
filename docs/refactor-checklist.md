# Refactor Checklist

## Completed

- [x] Phase 1 audit/spec pass.
- [x] `docs/refactor-guide.md`.
- [x] `docs/maintenance-audit.md`.
- [x] `docs/agent-worklog.md`.
- [x] Phase 2 architecture doc.
- [x] Public route compatibility snapshot.
- [x] Block-driven home/about/resume content in `content/portfolio/home.json`.
- [x] Generic portfolio block renderer in `js/portfolio-blocks.js`.
- [x] Blog metadata supports pinned slug, description, tags, cover, status, and updated date.
- [x] Draft posts are hidden from normalized public/build data.
- [x] Blog list/search cards render cover images, descriptions, and tags.
- [x] Authoring scripts for new posts and projects.
- [x] Content and asset check scripts.
- [x] Editor UI for description, tags, cover, status, updated date, and slug.
- [x] Build-time portfolio block rendering with browser fallback.
- [x] Generated-route browser and media verification.
- [x] Private editor artifacts excluded from deployment.
- [x] Project routes included in sitemap and shared route inventory.
- [x] Local Three.js/Tween runtime and 3D viewer assets.

## Remaining / Future Work

- [ ] Add per-post frontmatter parsing if JSON metadata becomes too indirect.
- [ ] Add visual regression screenshots for portfolio and blog cards.
- [ ] Consider moving publications/talks into dedicated `content/portfolio/*.json` files.
- [ ] Consider automated perceptual screenshot diffs if visual churn increases.

## Manual QA Checklist

- [x] Root route `/` is generated in `blogs/dist/index.html`.
- [x] Portfolio anchors still exist as `#home`, `#about`, `#resume`, `#portfolio`, `#blog`.
- [x] Portfolio cards render and filter correctly in browser QA.
- [x] Blog home renders Posts, Notes, and Series tabs in browser QA.
- [x] Search page returns cards with covers/tags in browser QA.
- [x] Local HTTP checks passed for root HTML, portfolio block JSON, a representative blog post, and a representative project page.
- [x] Representative English and Korean post routes exist in `blogs/dist`.
- [x] Legacy `/blogs/posts/?id=<id>` mappings are documented in `docs/public-route-compatibility.md`.
- [x] Representative project route exists in `blogs/dist`.
- [x] `npm run build` passes.
- [x] `npm run check:content` passes.
- [x] `npm run check:assets` passes.
- [x] `npm run check:render` visits all public routes and validates rendered local media.

## Known Limitations

- The blog metadata model is frontmatter-like JSON, not Markdown frontmatter yet.
- External embeds and ordinary outbound links are not availability-checked.

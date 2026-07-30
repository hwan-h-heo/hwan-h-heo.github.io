# Maintenance Audit

Date: 2026-06-26

This audit documents the current repository before major refactor work. It is evidence for `docs/refactor-guide.md` Phase 1 and should be updated when the implementation changes meaningful build, routing, content, editor, asset, or deployment behavior.

## Scope Checked

Checked current repository structure, root and blog package scripts, portfolio home page, blog static build pipeline, site data loader/validator, blog list scripts, portfolio preview scripts, editor API behavior, project-page generation, and asset reference patterns.

## Framework and Build System

The site is currently plain HTML, CSS, and JavaScript with a Node static generation step. There is no frontend framework, no database, and no production backend.

Root `package.json` delegates all build/dev/deploy commands to `blogs/package.json`:

- `npm run build`: runs `npm --prefix blogs run build`.
- `npm run dev`: runs `npm --prefix blogs run dev`.
- `npm run deploy`: runs `npm --prefix blogs run deploy`.

`blogs/package.json` defines:

- `build`: `node build-static.js`.
- `dev`: build, then serve `blogs/dist` on port `8080`.
- `deploy`: build, then publish `blogs/dist` with `gh-pages -d dist --dotfiles`.
- `edit`: start the local editor API with `node editor-server.js` on port `3030`.

The generator in `blogs/build-static.js`:

- Removes and recreates `blogs/dist`.
- Regenerates project pages from `projects/<slug>/project.json` and `projects/<slug>/content.md`.
- Copies blog static directories and files under `blogs/dist/blogs`.
- Copies `blogs/posts` source assets into `blogs/dist/blogs/posts`.
- Copies root `assets`, `js`, and root `index.html` into the dist root.
- Copies project assets into `blogs/dist/projects`, excluding project source files.
- Validates blog content files against `blogs/data/site-data.json`.
- Generates blog post pages under `blogs/dist/blogs/posts/<slug>/`.
- Emits `sitemap.xml`, `robots.txt`, `.nojekyll`, and an old-site redirect support page.

## Current Routing

Public routes currently implied by source and generated output:

- `/`: root portfolio page copied from `index.html`.
- `/#home`, `/#about`, `/#resume`, `/#portfolio`, `/#blog`: in-page portfolio sections.
- `/blogs/`: blog home copied from `blogs/index.html`.
- `/blogs/search/`: copied search page assets.
- `/blogs/3DViewer/`: copied 3D viewer utility.
- `/blogs/editor/`: copied editor UI, but its API only works when `npm run edit` is running locally.
- `/blogs/posts/<slug>/`: generated English blog post route.
- `/blogs/posts/<slug>-kor/`: generated Korean blog post route when `languages` includes `kor`.
- `/blogs/posts/?id=<old_id>`: legacy style references are rewritten in generated post HTML where `replaceLegacyPostLinks` sees them.
- `/projects/<slug>/`: project routes generated in the source tree and then copied to dist.
- `/hwan-h-heo.io/`: generated redirect support route for the old site path.

Route preservation risk is high for blog posts because the current public slug is derived from `title_eng` unless a `slug` is present in `site-data.json`. Any migration to frontmatter must freeze existing slug output before changing titles or IDs.

## Current Content Organization

Current high-level structure:

- Root portfolio shell: `index.html`.
- Root shared styles and scripts: `assets/css/`, `assets/js/`, and `js/`.
- Portfolio data: partially in `blogs/data/site-data.json`, partially hardcoded in `index.html`.
- Portfolio projects: `projects/<slug>/project.json`, `projects/<slug>/content.md`, generated `index.html`, and local assets.
- Blog metadata: `blogs/data/site-data.json`.
- Blog source: `blogs/posts/<YYMMDD_slug>/content-eng.md`, optional `content-kor.md`, and local assets.
- Blog static shell/assets: `blogs/index.html`, `blogs/css/`, `blogs/js/`, `blogs/search/`, and `blogs/3DViewer/`.
- Local editor: `blogs/editor/`.
- Generated deploy output: `blogs/dist/`, which should not be edited by hand.
- Old public-path compatibility: generated redirects from the current build; no archived generator source is kept in the runtime tree.

Current counts from `blogs/data/site-data.json` and the filesystem:

- 26 blog posts in metadata.
- 26 post source directories under `blogs/posts`.
- 6 portfolio cards in metadata.
- 7 project source directories under `projects`.
- 4 publications.
- 6 talks.
- 3 featured portfolio blog previews.
- 6 blog series entries.

## Current Blog Implementation

`blogs/data/site-data.json` is the central metadata source for posts, series, portfolio cards, publications, talks, and featured blog previews.

Post metadata currently supports:

- `id`
- `slug`
- `title_eng`
- `title_kor`
- `subtitle_eng`
- `subtitle_kor`
- `description_eng`
- `description_kor`
- `date`
- `updated`
- `category`
- `series`
- `tags`
- `cover`
- `status`
- `languages`

Post metadata is now frontmatter-like JSON rather than Markdown frontmatter. It supports preview description, cover image, tags, draft/published status, updated date, and pinned slugs while preserving compatibility with `blogs/data/site-data.json`.

`blogs/lib/site-data.js` validates strict post shapes and rejects unexpected post keys. This is good for safety, but it means adding richer metadata must start with schema changes before data migration.

`blogs/build-static.js` renders every configured post and language. Missing configured `content-*.md` files fail the build. It calculates reading time, injects share UI if missing, generates a TOC for headings, normalizes some legacy asset paths, and renders post pages through `blogs/lib/render-post-page.js`.

`blogs/index.html` is a static shell. `blogs/js/main-list.js` fetches `blogs/data/site-data.json`, sorts posts by date, and renders Posts, Notes, and Series tabs in the browser.

Search is a copied static page. Its client script also depends on `site-data-client.js`.

## Current Portfolio Implementation

The root `index.html` keeps the main shell and stable section anchors. Home/about/resume content is now rendered from `content/portfolio/home.json` by `js/portfolio-blocks.js`; navigation, portfolio grid shell, and blog preview shell remain in `index.html`.

Portfolio cards, selected publications, invited talks, and home-page featured blog previews are now rendered from `blogs/data/site-data.json`:

- `js/portfolio-content.js` renders `portfolioProjects`, `publications`, and `talks`.
- `js/portfolio-blog-preview.js` renders `featuredPortfolioPosts`.
- Both rely on `blogs/js/site-data-client.js` loading `/blogs/data/site-data.json`.

Project detail pages have a better source model:

- `projects/<slug>/project.json` contains project page metadata.
- `projects/<slug>/content.md` contains page body content.
- `blogs/build-static.js` regenerates `projects/<slug>/index.html`.
- `blogs/lib/render-project-page.js` defines the shared page wrapper.

The target block-based portfolio should begin by moving remaining hardcoded `index.html` content into structured data while keeping the current root route and visual shell stable.

## Current Editor and Preview Behavior

The local editor server is `blogs/editor-server.js` on port `3030`. It is not a production backend and should stay local-only.

Observed capabilities:

- Serves the editor UI and static source files.
- Provides `/api/editor-bootstrap` for categories, languages, series, post list, and featured flags.
- Reads and writes post bundles through `/api/post-bundle/...`.
- Saves drafts in `blogs/editor/drafts`.
- Uploads draft images into `blogs/editor/draft-assets`.
- Migrates draft asset references from `./draft-assets/<file>` into a post's `./assets/<file>` on post save.

Maintenance concerns:

- Draft and draft-asset directories contain local artifacts and are ignored by repo guidance, but the build currently copies the whole `blogs/editor` directory into dist.
- The editor follows the current `site-data.json` shape, not a future frontmatter shape.
- Existing post ID renaming and language removal are intentionally unsupported.
- Draft status exists in post metadata. Editor draft files still exist as local work-in-progress files before metadata publication.

## Current Asset Handling

Current asset locations:

- Root shared assets: `assets/`.
- Root/shared CSS and JS: `assets/css/`, `assets/js/`, and `js/`.
- Blog post local assets: usually under `blogs/posts/<id>/assets/`, but older posts also place assets directly in the post folder.
- Project assets: under `projects/<slug>/assets/` or direct files inside a project folder.
- Portfolio card thumbnails: mostly `assets/thumbnails/` and selected root `assets/` files.
- Editor draft assets: `blogs/editor/draft-assets/`.
- Generated output: `blogs/dist/`.

Reference patterns are mixed:

- Absolute site-root paths such as `/blogs/posts/<slug>/` and `/assets/...`.
- Relative HTML paths such as `assets/...`, `../assets/...`, `./assets/...`.
- Legacy post asset paths such as `./<post_id>/assets/...`; the build normalizes some of these for post output.
- External images and embeds from YouTube, GitHub raw URLs, Velog CDN, OpenAI CDN, and project pages.
- Project content may use paths relative to its project directory, such as `assets/file.jpg` or direct files like `vid_poseopt.mp4`.

Current validation covers local references in `site-data.json` for portfolio card URLs/media and featured preview images. It does not fully crawl Markdown/HTML content for broken local references.

Predictability issues:

- Blog assets are not consistently colocated under an `assets/` subfolder.
- Cover images are not first-class metadata.
- Featured post teaser image is a portfolio-specific metadata list, not a post-level field.
- Some project pages depend on external images instead of local copies.
- There is no dedicated `check:assets` command.

## Deployment Assumptions

Deployment is static GitHub Pages output from `blogs/dist`.

Important assumptions to preserve:

- `blogs/dist/.nojekyll` must exist so GitHub Pages serves files as-is.
- Root `index.html` must be copied to dist root.
- `/blogs/data/site-data.json` must exist because both portfolio and blog client scripts fetch it.
- `/blogs/posts/<slug>/index.html` and Korean variants must continue to exist.
- `/projects/<slug>/index.html` must continue to exist.
- No production API, database, server-side rendering, or backend is available.
- The local editor can remain local-only and should not be required for readers.

## Maintenance Pain Points

- Portfolio content is split between `content/portfolio/home.json`, `index.html` shell markup, and `site-data.json`.
- Blog metadata is centralized and now has a richer preview/status model, but it is still separate from Markdown bodies.
- Blog body files have no Markdown frontmatter, so content and metadata are edited in separate places.
- Existing slugs are now pinned in `site-data.json`; new posts should keep using explicit slugs.
- Asset references use several incompatible styles.
- The build has validation but no full content or asset checker.
- `blogs/dist` is generated, but `generateProjectPages()` writes generated `projects/<slug>/index.html` files back into the source tree before dist copy.
- Editor local draft artifacts can accumulate and are separate from publish status.
- Legacy route compatibility is handled partly by rewrite logic, but should be explicitly tested before route changes.

## Refactor Risks

- Breaking public blog URLs by changing slug generation, language suffix rules, or old `?id=` redirects.
- Breaking GitHub Pages deployment by moving files without updating `copyStaticAssets()`.
- Breaking portfolio cards by changing `site-data.json` shape before updating `site-data-client.js` and `portfolio-content.js`.
- Breaking editor save flows by moving metadata to frontmatter before updating `editor-server.js`.
- Losing local assets if post/project folder migrations are not scripted and checked.
- Accidentally editing generated `blogs/dist` or generated project `index.html` instead of source files.
- Introducing a build system that depends on APIs unavailable on GitHub Pages.

## Recommended Migration Plan Summary

1. Freeze route compatibility first: add an inventory of current generated post slugs, Korean slugs, project slugs, and legacy ID mappings before schema migration.
2. Add a second-phase architecture spec for the content model and compatibility rules.
3. Move remaining hardcoded portfolio sections from `index.html` into structured content while rendering the same sections and preserving `/`.
4. Extend post metadata in a backward-compatible way, then migrate toward frontmatter or a generated metadata index without dropping `site-data.json` compatibility until all readers and editor flows are updated.
5. Add asset checking before moving files. Prefer post/project-local assets, but preserve existing external references unless intentionally mirrored.
6. Update the local editor after the data model is stable so authoring creates the same files the build consumes.
7. Add authoring and validation scripts only after schemas are settled.

## Phase 1 Conclusion

No large UI rewrite should happen yet. The current site is buildable static output with partial data centralization. The next phase should create `docs/architecture.md` and `docs/refactor-checklist.md`, then define exact schemas and compatibility tests before implementation.

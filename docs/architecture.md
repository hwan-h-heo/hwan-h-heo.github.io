# Architecture

This repository remains a static GitHub Pages site built with plain HTML, CSS, JavaScript, and a Node build script. The refactor keeps that deployment model and adds content-first layers around the existing files.

## Content Directory Structure

Current source-of-truth content:

```txt
content/
  portfolio/
    home.json              # Ordered home/about/resume blocks

blogs/
  data/
    site-data.json         # Blog metadata, portfolio cards, publications, talks
  posts/<YYMMDD_slug>/
    content-eng.md
    content-kor.md
    assets/

projects/<project_slug>/
  project.json
  content.md
  assets/
```

`blogs/dist/` is generated output. Do not edit it by hand.

## Block Rendering Architecture

The root portfolio page keeps stable section IDs for public anchors:

- `#home`
- `#about`
- `#resume`
- `#portfolio`
- `#blog`

`index.html` now provides render targets for the home/about/resume sections:

```html
<section id="home" data-portfolio-block="home"></section>
```

`js/portfolio-blocks.js` is shared by Node and the browser. The static build pre-renders each block into `blogs/dist/index.html`; the browser loader remains as a development fallback when the source `index.html` is served directly. This keeps JSON as the content source without shipping an empty first render.

Current block types:

- `hero`
- `aboutProfile`
- `resume`

New block types should be added in `content/portfolio/home.json` first, then implemented in `js/portfolio-blocks.js`.

## Portfolio Loading Pipeline

Portfolio data now has two layers:

- `content/portfolio/home.json`: home/about/resume blocks.
- `blogs/data/site-data.json`: portfolio project cards, publications, talks, and featured blog previews.

Runtime scripts:

- `js/portfolio-blocks.js`: renders home/about/resume blocks.
- `js/portfolio-content.js`: renders project cards, publications, and talks.
- `js/portfolio-blog-preview.js`: renders featured blog preview slides.

`blogs/build-static.js` copies `content/` into `blogs/dist/content` so the portfolio block JSON is available after deployment.

## Blog Loading Pipeline

`blogs/data/site-data.json` remains the compatibility metadata index. Posts now support a frontmatter-like metadata shape directly in JSON:

- `slug`
- `description_eng`
- `description_kor`
- `tags`
- `cover`
- `status`
- `updated`

Published posts must define a stable slug, English description, local post-specific cover, at least one tag, status, and updated date. Korean posts also require a Korean description. Drafts may keep incomplete metadata while they are being written.

The build normalizes metadata for readers:

- Missing `description_*` falls back to subtitles.
- Missing `cover` falls back to `/assets/blog_bg.jpeg` only for draft/backward-compatible normalized data; validation rejects that fallback for published posts.
- Missing `status` is treated as `published`.
- `status: "draft"` posts are excluded from generated/public post lists.

Post bodies stay in `blogs/posts/<id>/content-eng.md` and optional `content-kor.md`.

## Project Loading Pipeline

Project pages already use a maintainable source model:

- `projects/<slug>/project.json`
- `projects/<slug>/content.md`
- local assets in the same folder

`blogs/build-static.js` regenerates `projects/<slug>/index.html` from those files and copies project assets into `blogs/dist/projects/<slug>/`.

Use `npm run new:project -- "Project Name"` to scaffold a new project source folder.

## Asset Handling Strategy

Preferred placement:

- Portfolio shared images: `assets/`
- Portfolio block content: `content/portfolio/home.json`
- Blog post images: `blogs/posts/<id>/assets/`
- Project images/files: `projects/<slug>/assets/`
- Draft uploads: `blogs/editor/draft-assets/` until published

Run `npm run check:assets` to scan local image/video references, reject remote rendered images, detect missing and orphaned localized assets, and warn about large animated media.

`npm run check:render` starts an isolated static server and uses Playwright with Chromium to visit every public portfolio, blog, search, post, utility, editor, and project route. It verifies local network responses, image/background decoding, video range responses, dynamic content, and the 3D viewer canvas.

External embeds and ordinary links remain allowed. Rendered first-party images, covers, portfolio media, Three.js runtime files, viewer textures, and viewer environment maps are local.

## Deployment Assumptions

Deployment remains:

```bash
npm run build
npm run deploy
```

`npm run deploy` first runs `npm run verify`, then publishes the already-verified `blogs/dist/` directory. The build copies pinned Three.js/Tween runtime files from `blogs/node_modules` and excludes `blogs/editor/drafts`, `draft-assets`, and `project-snapshots` from public output.

The public origin is `https://hwan-h-heo.github.io`. Node-side canonical URLs, Open Graph images, sitemap entries, robots.txt, and production verification share `SITE_URL` from `blogs/lib/site-config.js`.

Important generated paths:

- `blogs/dist/index.html`
- `blogs/dist/content/portfolio/home.json`
- `blogs/dist/blogs/data/site-data.json`
- `blogs/dist/blogs/posts/<slug>/index.html`
- `blogs/dist/projects/<slug>/index.html`
- `blogs/dist/.nojekyll`

No backend or database is required for production. The editor server remains local-only through `npm run edit` in `blogs/`.

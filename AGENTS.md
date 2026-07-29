# Repository Guidelines

## Project Structure & Module Organization
The repository has two main parts. The portfolio site lives at the root with [`index.html`](/Users/hwanheo/Projects/hwan-h-heo.io/index.html), shared portfolio styles and runtime code in `assets/css/` and `assets/js/`, data-driven scripts in `js/`, media in `assets/`, and project pages in `projects/<project>/index.html`. The portfolio home page’s blog preview is no longer hardcoded; it is rendered from [`blogs/data/site-data.json`](/Users/hwanheo/Projects/hwan-h-heo.io/blogs/data/site-data.json) by [`js/portfolio-blog-preview.js`](/Users/hwanheo/Projects/hwan-h-heo.io/js/portfolio-blog-preview.js). The blog lives under `blogs/` and is built by [`blogs/build-static.js`](/Users/hwanheo/Projects/hwan-h-heo.io/blogs/build-static.js), with shared build helpers in `blogs/lib/`. Blog metadata has a single source of truth in [`blogs/data/site-data.json`](/Users/hwanheo/Projects/hwan-h-heo.io/blogs/data/site-data.json). Blog source content is stored in `blogs/posts/<YYMMDD_slug>/`, typically with `content-eng.md`, optional `content-kor.md`, and local assets. Treat `blogs/dist/` as generated output; do not edit it by hand. Historical blog generator source has been removed; preserve old public URLs through the current redirect generator instead of adding archival runtime code.

## Build, Test, and Development Commands
- `npm install`: install root deployment tooling.
- `npm run build`: rebuild the static site into `blogs/dist/`, validate `blogs/data/site-data.json`, and regenerate blog posts plus the portfolio deployment root.
- `npm run dev`: run the blog build, then serve `blogs/dist/` locally on port `8080`.
- `cd blogs && npm run edit`: start the local editor API from [`blogs/editor-server.js`](/Users/hwanheo/Projects/hwan-h-heo.io/blogs/editor-server.js) on port `3030`.
- `npm run deploy`: publish `blogs/dist/` to GitHub Pages.

## Coding Style & Naming Conventions
Use plain HTML, CSS, and JavaScript with no framework assumptions. Match the surrounding file’s style: Node-side scripts in `blogs/` currently use 4-space indentation. Keep filenames lowercase and descriptive. Blog post directories follow `YYMMDD_topic`, for example `blogs/posts/240602_2dgs/`. Keep Markdown filenames consistent with the current pattern: `content-eng.md` and `content-kor.md`. When adding or editing blog metadata, update [`blogs/data/site-data.json`](/Users/hwanheo/Projects/hwan-h-heo.io/blogs/data/site-data.json) instead of legacy JS files. Keep `languages` entries aligned with the actual `content-*.md` files, because the build now validates them strictly.

## Testing Guidelines
There is no automated test suite yet. Validate changes by running `npm run build` and reviewing the generated site in `blogs/dist/`. For UI changes, run `npm run dev` and manually check portfolio blog preview rendering, blog navigation, TOC behavior, language toggles, image paths, redirect behavior from legacy `?id=` URLs, search results, and project links. For editor changes, also verify draft save/upload flows with `cd blogs && npm run edit`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `add varco3d posts` or `enhance editor`. Keep commits focused on one change. Pull requests should include a brief summary, affected paths, manual test notes, and screenshots for visible site updates. Do not commit ignored local artifacts such as `blogs/node_modules/`, `blogs/dist/`, `blogs/editor/drafts/`, or `blogs/editor/draft-assets/`.

# Repository Guidelines

## Project Structure & Module Organization
The repository has two main parts. The portfolio site lives at the root with [`index.html`](/Users/hwanheo/Projects/hwan-h-heo.io/index.html), shared styles in `css/`, shared scripts in `js/`, media in `assets/`, and project pages in `projects/<project>/index.html`. The blog lives under `blogs/` and is built by [`blogs/build-static.js`](/Users/hwanheo/Projects/hwan-h-heo.io/blogs/build-static.js). Blog source content is stored in `blogs/posts/<YYMMDD_slug>/`, typically with `content-eng.md`, optional `content-kor.md`, and local assets. Treat `blogs/dist/` as generated output; do not edit it by hand.

## Build, Test, and Development Commands
- `npm install`: install root deployment tooling.
- `npm run build`: rebuild the blog static output into `blogs/dist/` and copy shared site assets.
- `npm run dev`: run the blog build, then serve `blogs/dist/` locally on port `8080`.
- `cd blogs && npm run edit`: start the local editor API from [`blogs/editor-server.js`](/Users/hwanheo/Projects/hwan-h-heo.io/blogs/editor-server.js) on port `3030`.
- `npm run deploy`: publish `blogs/dist/` to GitHub Pages.

## Coding Style & Naming Conventions
Use plain HTML, CSS, and JavaScript with no framework assumptions. Match the surrounding file’s style: Node-side scripts in `blogs/` currently use 4-space indentation, while some older browser scripts use existing legacy formatting. Keep filenames lowercase and descriptive. Blog post directories follow `YYMMDD_topic`, for example `blogs/posts/240602_2dgs/`. Keep Markdown filenames consistent with the current pattern: `content-eng.md` and `content-kor.md`.

## Testing Guidelines
There is no automated test suite yet. Validate changes by running `npm run build` and reviewing the generated site in `blogs/dist/`. For UI changes, run `npm run dev` and manually check navigation, TOC behavior, language toggles, image paths, and project links. For editor changes, also verify draft save/upload flows with `cd blogs && npm run edit`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `add varco3d posts` or `enhance editor`. Keep commits focused on one change. Pull requests should include a brief summary, affected paths, manual test notes, and screenshots for visible site updates. Do not commit ignored local artifacts such as `blogs/node_modules/`, `blogs/dist/`, `blogs/editor/drafts/`, or `blogs/editor/draft-assets/`.

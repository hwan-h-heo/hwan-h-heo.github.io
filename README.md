# hwan-h-heo.github.io
Personal Portfolio & Blog of Hwan Heo 

Cite: https://hwan-h-heo.github.io

## Maintenance

- Run `npm run setup` once to install the root verification tools and blog build dependencies.
- Run `npm run verify` before committing. It validates content/assets, builds `blogs/dist/`, and checks every public route in Chromium.
- Run `npm run audit:assets` after replacing media or styles. See [`docs/styles-and-assets.md`](docs/styles-and-assets.md) for stylesheet ownership and asset placement.
- Run `npm run dev` to serve the generated site locally on port `8080`.
- Run `npm run deploy` to run the full verification gate and publish `blogs/dist/` to GitHub Pages.
- Project pages use `projects/<slug>/project.json` and `projects/<slug>/content.md` as source files.
- `projects/<slug>/index.html` files are generated build artifacts and are intentionally ignored by git.
- `blogs/dist/` is generated output. Local editor drafts, uploads, and snapshots are excluded from it.

# hwan-h-heo.github.io
Personal Portfolio & Blog of Hwan Heo 

Cite: https://hwan-h-heo.github.io

## Maintenance

- Run `npm run setup` once to install the root verification tools and blog build dependencies.
- Run `npm run verify` before committing. It validates content/assets, builds `blogs/dist/`, and checks every public route in Chromium.
- Run `npm run check:legacy-ui` after the build to reject retired framework/vendor references, legacy UI class tokens, and broken site-icon sprite hrefs in both source and generated output.
- Run `npm run check:ui` after interface changes. It exercises representative routes at the 390, 767/768, 991/992, 1199/1200, 1440, and 2560 px responsive boundaries.
- Run `npm run capture:ui` to capture reproducible full-page screenshots under the ignored `artifacts/ui-regression/` directory. Pass `-- --screenshots-dir=<path>` to choose another location.
- Run `npm run audit:assets` after replacing media or styles. See [`docs/styles-and-assets.md`](docs/styles-and-assets.md) for stylesheet ownership and asset placement.
- Keep portfolio UI changes aligned with [`docs/portfolio-design-language.md`](docs/portfolio-design-language.md).
- Run `npm run dev` to serve the generated site locally on port `8080`.
- Run `npm run deploy` to run the full verification gate and publish `blogs/dist/` to GitHub Pages.
- Project pages use `projects/<slug>/project.json` and `projects/<slug>/content.md` as source files.
- `projects/<slug>/index.html` files are generated build artifacts and are intentionally ignored by git.
- `blogs/dist/` is generated output. Local editor drafts, uploads, and snapshots are excluded from it.

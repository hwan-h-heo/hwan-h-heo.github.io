# Portfolio sculpture cover

The approved v22 study is the production home hero. Copy is authored in
`content/portfolio/home.json` and rendered by `js/portfolio-blocks.js` at build
time, including the poster and navigation. The three phrase lines are:

```text
Building 3D generative systems,
from geometric representations
to efficient GPU inference.
```

`assets/css/hero-sculpture.css` contains the cover's isolated styles. Name, role
and affiliation retain the study's placement. The paragraph uses Manrope Regular
at 16 px desktop / 15 px mobile; both 12 px chapter links keep 44 px hit targets.
The masthead keeps About, and the bottom chevron links to Projects.

## Rendering and source

`js/hero.js` initializes the Three.js r180 WebGL scene. The small modules in
`js/hero-sculpture/` own the original procedural geometry, packed stone material,
contact shadows and deterministic surface refinement. The scene uses AgX tone
mapping, sRGB output, neutral studio lights, VSM shadows and 24,672 triangles.
Material metalness and clearcoat are zero. There is no ray tracing or model asset
download. The About portrait and article viewers retain their existing r150 runtime.

Refinement starts coarse before its first canvas frame and plays once over four
seconds. Scroll samples that same surface in reverse over 72% of the cover height;
the object stays upright. Scrolling during the entrance retracts only the fraction
already formed. Rendering stops at rest and offscreen, and Motion off freezes the
current state. Reduced motion, data saving, unavailable WebGL and context loss use
the authored responsive posters. JavaScript failure still leaves copy, navigation
and the poster visible, with the inactive motion control hidden.

`npm run build:hero` bundles the pinned root `three` and `esbuild` versions into
`assets/js/hero-sculpture.js` and its linked license notice. These generated assets
are committed with their source so direct static checkout previews work. `npm run
build` regenerates them before the existing blog/static build. The bundled entry is
versioned by the site's existing HTML asset hashing; the entry has no runtime
module imports. Do not deploy the all-in-one experimental preview HTML.

## Fonts and assets

The accepted cover fonts are self-hosted under `assets/hero/sculpture/fonts/`
using cover-only family names. Other sections keep their established fonts.
Font sources are Google Fonts' Manrope, Inter, IBM Plex Mono and Space Grotesk;
the SIL Open Font License notices are retained alongside them:

- `manrope-OFL.txt`
- `inter-OFL.txt`
- `ibmplexmono-OFL.txt`
- `spacegrotesk-OFL.txt`

The original scene posters are `poster-1440.webp`, `poster-768.webp` and
`poster-390.webp`. They depict the final surface and are selected by viewport.
The geometry, mineral texture and lighting are code-authored. The visual reference
was onformative's sculptural spatial composition; no reference mesh or image is used.

## Verification and publishing

Run `npm run build`, `npm run check:hero` and the affected home render check:
`node scripts/check-rendered-site.js --route=/`. The hero check covers the three
phrase lines, touch targets, intro, reverse scroll, idle rendering, pause,
responsive placement and static fallback. Use existing content, asset and SEO
validation before publishing the static output with
`npm --prefix blogs run deploy:dist`.

Build release output from a clean checkout of the committed changes so local
drafts and unrelated metadata edits cannot enter the public artifact. Experiments
under `_workspace/` are not production inputs and should not be committed or sent
to B300 unless requested again.

If `gh-pages` contains already-published posts ahead of `main`, retain that
published tree and overlay only the rebuilt home HTML, home content JSON,
portfolio-block renderer and sculpture assets/source. Restamp deployment versions
and asset hashes across the resulting tree, then validate the actual release.
This preserves public content without committing unrelated local edits.

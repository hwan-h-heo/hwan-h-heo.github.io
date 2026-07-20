# Asset Handling Audit

Date: 2026-06-26

## Current Asset Directories

- `assets/`: shared portfolio/blog images, thumbnails, profile media, CV, favicon, and root vendor assets.
- `assets/thumbnails/`: portfolio card thumbnails.
- `blogs/posts/<id>/assets/`: preferred blog post-local assets.
- `blogs/posts/<id>/`: older posts may still contain loose files directly in the post folder.
- `projects/<slug>/assets/`: preferred project-local assets.
- `projects/<slug>/`: some project files are still stored directly next to `content.md`.
- `blogs/editor/draft-assets/`: local draft uploads before publication.
- `blogs/dist/`: generated output, not source.

## Current Reference Patterns

- Root absolute: `/assets/...`, `/blogs/posts/...`, `/projects/...`.
- Relative to Markdown file: `./assets/...`, `assets/...`.
- Legacy post-local: `./<post_id>/assets/...`.
- External: YouTube embeds, GitHub raw files, Velog CDN, OpenAI CDN, project pages.

The build normalizes local blog image paths during post generation, including `./assets/...` and legacy cross-post `./<post_id>/assets/...` references, so generated slug routes do not break image URLs.

## Improvements Added

- Portfolio block JSON is copied to deployment output via the build.
- Post metadata now has a first-class `cover` field.
- `npm run check:assets` scans local post/project image and video references plus post covers.
- New post scaffolding creates a predictable `assets/` folder.
- New project scaffolding creates a predictable `assets/` folder.
- Published blog posts now must use post-specific cover images instead of `/assets/blog_bg.jpeg`.
- Rendered remote blog/project images have been localized into post/project `assets/` folders.
- `npm run check:assets` fails on remote rendered image references in source content.
- Generated post HTML has been checked for missing local rendered image files after build.
- `npm run check:assets` also scans runtime media references, CSS backgrounds, local videos, and orphaned localized `remote-*` files.
- `npm run check:render` opens every public route and verifies decoded images, videos, dynamic lists, and the 3D viewer canvas.
- The 3D viewer's textures, HDR environments, Three.js modules, and Tween.js runtime are copied locally during the build.
- Large 2DGS Viewer demonstration GIFs were converted to local MP4 files to reduce deployment size and browser decode cost.

## Broken or Fragile Patterns

- Some older blog posts keep images directly in the post directory instead of `assets/`.
- Some older notes use SVG figures as the best available cover assets.
- The asset checker rejects external rendered image URLs but does not fetch or verify ordinary external links and embeds.
- The local editor draft asset folder can accumulate files that are not published.
- Two older NeRF-in-game videos and one 2DGS training GIF remain above the advisory 15 MiB threshold.

## Preferred Rules Going Forward

- Blog post images: `blogs/posts/<id>/assets/<file>`.
- Blog cover image: use a post-specific local image whenever available. Published posts should not fall back to `/assets/blog_bg.jpeg`.
- Large animated media/video: prefer external hosting or compression when a file is around 15 MiB or larger. Keep local copies for small, important assets that are likely to break externally.
- Project assets: `projects/<slug>/assets/<file>`.
- Portfolio card images: `assets/thumbnails/<file>` or project-local generated route if later supported.
- Do not move existing public assets without checking generated routes and references.

## Static Hosting Constraints

GitHub Pages serves static files only. Asset processing must happen before deployment or in client-side code. There is no production upload API, image proxy, database, or server-side optimization.

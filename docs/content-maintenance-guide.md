# Content Maintenance Guide

## Add or Reorder Portfolio Blocks

Edit:

```txt
content/portfolio/home.json
```

Blocks are rendered by ID into matching `index.html` sections. Current block IDs:

- `home`
- `about`
- `resume`

To reorder visible portfolio sections, reorder the corresponding `<section>` elements in `index.html`. To edit content inside a section, edit the JSON block.

When adding a new block type, add both:

- data in `content/portfolio/home.json`
- renderer logic in `js/portfolio-blocks.js`

## Add a Project

Use:

```bash
npm run new:project -- "Project Name"
```

Then edit:

```txt
projects/<project_slug>/project.json
projects/<project_slug>/content.md
```

Put local project assets in:

```txt
projects/<project_slug>/assets/
```

To feature the project on the portfolio grid, add a card entry to `blogs/data/site-data.json` under `portfolioProjects` after a card image is ready.

## Add a Publication or Talk

Edit `blogs/data/site-data.json`:

- `publications`
- `talks`

These render through `js/portfolio-content.js`.

## Add a Blog Post

Use:

```bash
npm run new:post -- "Post Title"
```

Then edit the generated Markdown and metadata in `blogs/data/site-data.json`.

Set `status` from `draft` to `published` when ready. Use `unlisted` for a post
that must keep its direct URL while remaining absent from the blog home, series
and tag archives, search, related links, RSS, and sitemaps. Unlisted pages are
generated with `noindex, nofollow`.

## Post Classification

All articles use the single post category:

```json
"category": "post"
```

Choose the appropriate `series`, `tags`, and `cover`. Series are presented as a
separate browsing view on the blog home rather than as a post category.

## Check Content and Assets

Run:

```bash
npm run check:content
npm run check:assets
npm run build
npm run check:render
```

`check:content` validates metadata, required post language files, portfolio blocks, and project source pairs.

`check:assets` scans rendered media references and localized asset ownership. `check:render` verifies the generated routes in Chromium. `npm run verify` runs the complete sequence and is the preferred pre-commit command.

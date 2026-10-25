# Blog Authoring Workflow

## Create a Post

Use:

```bash
npm run new:post -- "Post Title"
```

This creates:

```txt
blogs/posts/<YYMMDD_post_title>/
  content-eng.md
  assets/
```

It also adds a draft metadata entry to `blogs/data/site-data.json` with:

- pinned `slug`
- `subtitle_eng`
- `tags`
- `cover`
- `status: "draft"`
- `updated`

Draft posts are not generated in public output.

## Add Images

Place post images in:

```txt
blogs/posts/<post_id>/assets/
```

Reference them from Markdown as:

```md
![Figure caption](./assets/figure.webp)
```

The local editor can also upload draft images into `blogs/editor/draft-assets/` and migrate `./draft-assets/<file>` references into the post assets folder when publishing.

## Set Cover Image

Set `cover` in `blogs/data/site-data.json`.

Preferred values:

```json
"cover": "/blogs/posts/<post_id>/assets/cover.webp"
```

Published posts may use a shared local image, but `/assets/blog_bg.jpeg` is reserved as a draft fallback and fails published-content validation.

## Set Subtitle

Use `subtitle_eng` and, when Korean content exists, `subtitle_kor`. This single
editorial deck supplies article headers, home and search previews, RSS summaries,
and search/social metadata. Published posts must define the subtitle for every
available language.

Full and incremental builds also generate a fingerprinted search index under
`blogs/dist/blogs/data/` from published article bodies. Search combines the current
site metadata with this cached plain-text index, retaining the English original
when a Korean body is unavailable. Draft and unlisted bodies are excluded. Do not
edit the generated index; update the post source or metadata and rebuild. Run
`npm test` when changing search extraction, ranking, excerpts, or highlighting.

## Tags

Use the `tags` array in `blogs/data/site-data.json`:

```json
"tags": ["3D Generation", "Neural Rendering"]
```

Tags appear on blog list/search cards and article headers.

## Series

Set the primary `series` in `blogs/data/site-data.json`. To list the same post
in more series, add their IDs to `additionalSeries`:

```json
"series": "graphics",
"additionalSeries": ["3d-generation"]
```

Choose the primary series by the problem the article explains. Use an additional
series when the article substantially serves that readership, and tags for the
techniques and concepts used along the way. For example, sparse 3D inference
optimization belongs primarily to `cuda` and additionally to `3d-generation`.
The PCG-to-Muon article belongs to `graphics` and `3d-generation`; its CUDA tag
does not by itself make it part of the CUDA series.

The primary series supplies preview labels, the article header, and previous/next
navigation. Additional series include the post in their Blog home groups and
series archives. The post keeps one URL per language and counts once in the
overall article total. Each series counts it once.

`additionalSeries` is optional. Every entry must name an existing series, with
no duplicates or repetition of the primary series. Edit these memberships in
the JSON file; the local editor preserves them when saving the post.

Web 3D articles belong to `graphics`; keep `Web 3D` and `Three.js` tags for
more specific discovery. When merging a series, update published and draft
memberships, remove its label from `series`, and add its old ID to
`seriesRedirects` (for example, `"web-3d": "graphics"`). The static redirect
generator preserves the old archive URL without adding it to the sitemap.
Redirect targets must be current series with published posts; chains are not
allowed. Individual post slugs stay unchanged.

## Related Posts

Keep curated recommendations in `blogs/data/post-related.json`, under the post
ID and language (`eng` or `kor`). Use post IDs for internal articles and
`{"title": "...", "url": "https://..."}` for external articles. The shared
renderer supplies the Related Posts cards and resolves internal links to the
appropriate language. Move imported "You may also like" lists into this data
instead of keeping a second recommendation list in the article body. Keep
citations and original-source credits in the body.

## Preview Locally

Build and serve:

```bash
npm run dev
```

The dev server serves `blogs/dist` on port `8080`.

For the local editor:

```bash
cd blogs
npm run edit
```

The editor API runs on port `3030`. Its Publishing panel edits slug, descriptions, tags, cover, status, and updated date together with the Markdown body.

## Publish

Set:

```json
"status": "published"
```

Then run:

```bash
npm run verify
```

Deploy with:

```bash
npm run deploy
```

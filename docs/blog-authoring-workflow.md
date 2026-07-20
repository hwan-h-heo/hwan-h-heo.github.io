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
- `description_eng`
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

## Set Preview Text

Use `description_eng` and, when Korean content exists, `description_kor`.

Older subtitles still render as compatibility fallbacks, but published posts must define `description_eng` and Korean posts must also define `description_kor`.

## Tags

Use the `tags` array in `blogs/data/site-data.json`:

```json
"tags": ["3D Generation", "Neural Rendering"]
```

Tags appear on blog list/search cards and article headers.

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

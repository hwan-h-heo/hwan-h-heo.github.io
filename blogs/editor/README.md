# Blog Editor

A markdown editor for creating blog posts, registering metadata, and publishing both together.

## Features

- ✍️ **Live Preview**: See your content rendered as you type
- 💾 **Draft Management**: Save and load per-language markdown drafts locally
- 📝 **Post Publishing**: Create or update `site-data.json` and `posts/{postId}/content-*.md` in one flow
- 🗂️ **Metadata Editing**: Manage identity, descriptions, slug, tags, cover, status, dates, languages, and portfolio feature settings
- ✅ **Validation**: Duplicate id checks, allowed category/series validation, and language/file alignment
- 🎨 **Blog-Matched Styling**: Preview looks exactly like published posts
- 📐 **Math Support**: Write LaTeX equations with KaTeX
- 🎯 **Code Highlighting**: Automatic syntax highlighting with Prism.js
- ⌨️ **Keyboard Shortcuts**: Ctrl/Cmd + S to save

## Usage

### Edit Mode (Full Features)

```bash
npm run edit
```

Then open http://localhost:3030/editor/

In edit mode, you can:
- Create a new post workspace
- Load an existing post into the editor
- Load/save/delete drafts for the active language tab
- Publish metadata to `data/site-data.json`
- Create or update `posts/{postId}/content-eng.md` and optional `content-kor.md`
- Configure `featuredPortfolioPosts` entry for the current post

### Dev Mode (Preview Only)

```bash
npm run dev
```

Then navigate to the editor - draft management features are hidden.

## Workflow

1. **Start editor**: `npm run edit`
2. **Fill metadata**: Set identity, titles/descriptions, dates, slug, tags, local cover, status, languages, and optional featured teaser info
3. **Write content**: Use the English/Korean tabs to edit markdown
4. **Save draft**: Save the active language tab if you want a local snapshot
5. **Publish**: Click `Publish` to write metadata and markdown files together

## Keyboard Shortcuts

- `Ctrl/Cmd + S`: Save the active language as a draft in edit mode, or download it in preview mode

## File Structure

```
editor/
├── index.html         # Editor UI
├── editor.css         # Editor styles
├── editor.js          # Editor app logic
├── edit.html          # Redirect to index.html
├── drafts/            # Local markdown drafts (gitignored)
├── draft-assets/      # Temporary uploads (gitignored and excluded from deployment)
├── project-snapshots/ # Local project backups (gitignored and excluded from deployment)
└── README.md          # This file

posts/
└── {postId}/
    ├── content-eng.md
    └── content-kor.md
```

## Tips

- Drafts are stored locally in `editor/drafts/`
- Post ID format: `YYMMDD_slug` (e.g., `250101_title`)
- Use standard Markdown syntax
- Math: Inline `$x^2$`, Block `$$x^2$$`
- Code blocks: Use triple backticks with language

## Example Post

```markdown
## Introduction

This is a sample blog post.

### Features

- Bullet points
- **Bold** and *italic*
- Math: $E = mc^2$

\`\`\`javascript
console.log("Hello World!");
\`\`\`

![Image](./assets/image.png)
```

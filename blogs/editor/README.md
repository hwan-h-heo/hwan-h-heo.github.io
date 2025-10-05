# Blog Editor

A simple, elegant Markdown editor for creating and managing blog posts.

## Features

- ✍️ **Live Preview**: See your content rendered as you type
- 💾 **Draft Management**: Save and load drafts locally
- 📝 **Post Publishing**: Directly save to posts directory
- 🎨 **Blog-Matched Styling**: Preview looks exactly like published posts
- 📐 **Math Support**: Write LaTeX equations with KaTeX
- 🎯 **Code Highlighting**: Automatic syntax highlighting with Prism.js
- ⌨️ **Keyboard Shortcuts**: Ctrl/Cmd + S to save

## Usage

### Edit Mode (Full Features)

```bash
npm run edit
```

Then open http://localhost:3030/editor/edit.html

In edit mode, you can:
- Create new drafts
- Load/save/delete drafts
- Save directly to `posts/{postId}/content-{lang}.md`

### Dev Mode (Preview Only)

```bash
npm run dev
```

Then navigate to the editor - draft management features are hidden.

## Workflow

1. **Start editor**: `npm run edit`
2. **Write content**: Use Markdown syntax in the left pane
3. **Preview**: See live preview in the right pane
4. **Save draft**: Click "💾 Save Draft" to save work in progress
5. **Publish**:
   - Enter Post ID (e.g., `250101_my_post`)
   - Select language (eng/kor)
   - Click "✅ Save to Post"

## Keyboard Shortcuts

- `Ctrl/Cmd + S`: Save draft (in edit mode) or download markdown (in dev mode)

## File Structure

```
editor/
├── edit.html          # New improved editor
├── drafts/            # Auto-saved and manual drafts (gitignored)
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

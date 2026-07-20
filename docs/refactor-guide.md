# Portfolio / Blog Refactor Guide

This document is the canonical migration guide for refactoring `hwan-h-heo.github.io`.

The purpose of the refactor is to make the site easier to maintain as a personal portfolio and technical blog. The final site should feel closer to a Notion-like portfolio and Velog/Notion-like blog writing system, while remaining static-hosting friendly and compatible with GitHub Pages.

---

## 1. Main Goals

### Portfolio

The portfolio should become block-based.

Current portfolio/home/about/resume/project elements should be moved out of scattered UI components and into structured content files. Reordering or adding sections should usually require editing one content/config file, not modifying several UI components.

Target behavior:

* Add a section by adding a block entry.
* Reorder sections by reordering block entries.
* Add a project by creating one content entry.
* Keep the UI professional, modern, and easy to scan.
* Preserve existing identity: AI researcher / 3D generation / neural rendering / production-oriented 3D asset generation.

### Blog

The blog should support a modern writing workflow.

Target behavior:

* Each post lives in its own folder.
* Each post has frontmatter metadata.
* Cover image and local assets are stored near the post.
* Blog list preview is generated from frontmatter and excerpt.
* Draft/published status is supported if feasible.
* Local preview/editing workflow is documented.
* Adding a new post should be simple and repeatable.

### Assets

Image and file handling should be predictable.

Target behavior:

* Blog assets are colocated with blog posts.
* Portfolio/project assets are colocated with project entries.
* Broken image references can be checked.
* Existing assets are not deleted without documentation.
* Static-hosting constraints are respected.

### Documentation

All meaningful changes must be documented.

First-pass required docs:

```txt
docs/refactor-guide.md
docs/maintenance-audit.md
docs/agent-worklog.md
```

Additional docs to add in later phases when those phases start:

```txt
docs/architecture.md
docs/blog-authoring-workflow.md
docs/content-maintenance-guide.md
docs/asset-handling-audit.md
docs/refactor-checklist.md
```

---

## 2. Hard Constraints

Do not:

* Break GitHub Pages deployment.
* Delete existing posts, portfolio items, images, or metadata unless clearly obsolete and documented.
* Replace the whole framework unless absolutely necessary.
* Add a backend, database, or heavy CMS.
* Hardcode content into UI components when it can live in content files.
* Break public URLs unless compatibility redirects/routes are provided.
* Claim success without running available checks.

Do:

* Preserve existing content and routes whenever possible.
* Keep the site buildable after each major phase.
* Prefer static-site-friendly architecture.
* Prefer content-first architecture.
* Keep docs concise but useful.
* Append concise progress notes to `docs/agent-worklog.md`.

---

## 3. Recommended Target Structure

Adapt this to the current project stack.

```txt
content/
  portfolio/
    profile.yaml
    experience.yaml
    education.yaml
    publications.yaml
    projects/
      project-slug/
        index.md or index.mdx
        cover.png
        assets/
  blog/
    posts/
      YYYY-MM-DD-post-slug/
        index.md or index.mdx
        cover.png
        assets/
    notes/
      note-slug/
        index.md or index.mdx
        assets/
    series/
      series-slug.yaml
  blocks/
    home.yaml
    about.yaml
    portfolio.yaml

src/
  components/
    blocks/
      BlockRenderer.*
      TextBlock.*
      MarkdownBlock.*
      ImageBlock.*
      GalleryBlock.*
      ProjectBlock.*
      ProjectGridBlock.*
      PublicationListBlock.*
      TimelineBlock.*
      LinkCardBlock.*
      CalloutBlock.*
    blog/
      BlogList.*
      BlogCard.*
      BlogPost.*
      BlogPreview.*
    layout/
      SiteLayout.*
      BlogLayout.*
  lib/
    content/
      loadBlocks.*
      loadPortfolio.*
      loadBlogPosts.*
      loadProjects.*
      assetResolver.*
      frontmatter.*
      slug.*
```

The exact extension should match the existing framework.

---

## 4. Portfolio Block Model

Implement a generic block renderer if the current stack allows it.

Minimum useful block types:

```ts
type PortfolioBlock =
  | { type: "hero"; title: string; subtitle?: string; links?: Link[] }
  | { type: "text"; title?: string; body: string }
  | { type: "markdown"; source: string }
  | { type: "image"; src: string; caption?: string }
  | { type: "gallery"; images: ImageItem[] }
  | { type: "project"; projectId: string }
  | { type: "projectGrid"; projectIds?: string[]; filter?: string }
  | { type: "publicationList"; items?: string[] }
  | { type: "timeline"; items: TimelineItem[] }
  | { type: "experience"; items?: string[] }
  | { type: "education"; items?: string[] }
  | { type: "linkCard"; title: string; url: string; description?: string }
  | { type: "callout"; title?: string; body: string; tone?: string }
  | { type: "divider" }
  | { type: "spacer"; size?: "sm" | "md" | "lg" };
```

Expected result:

* Home/about/portfolio/resume-like pages can be represented as ordered block arrays.
* UI components become reusable renderers.
* Content files determine what appears and in what order.
* Existing content is migrated into the new structure.

---

## 5. Blog Post Format

Recommended post folder:

```txt
content/blog/posts/2026-06-26-example-post/
  index.mdx
  cover.png
  assets/
    fig01.png
    fig02.webp
```

Recommended frontmatter:

```md
---
title: "Post Title"
description: "Short preview text shown in blog list"
date: "2026-06-26"
updated: "2026-06-26"
tags: ["3D Generation", "Neural Rendering"]
series: "optional-series-name"
cover: "./cover.png"
status: "published"
---

Post body here.
```

Requirements:

* Blog list should be generated automatically from content files.
* Blog cards should use title, description, date, tags, cover, and reading time if easy.
* Draft posts should be hidden in production if feasible.
* Existing posts must be preserved and migrated carefully.
* The preview description should be explicit in frontmatter or generated consistently.
* Article rendering should support good typography, code blocks, math, images, captions, and callouts if already supported or easy to add.

---

## 6. Asset Handling

Audit current asset handling first.

Create:

`docs/asset-handling-audit.md`

Include:

* Current asset directories
* Current image reference patterns
* Broken or fragile references
* Whether blog/portfolio assets are mixed
* Whether relative paths work
* Whether image optimization exists
* Whether upload/editor support exists
* What is realistic under GitHub Pages static hosting

Preferred design:

```txt
content/blog/posts/post-slug/
  index.mdx
  cover.png
  assets/

content/portfolio/projects/project-slug/
  index.mdx
  cover.png
  assets/
```

Useful scripts if feasible:

```bash
npm run new:post -- "Post Title"
npm run new:note -- "Note Title"
npm run new:project -- "Project Name"
npm run check:content
npm run check:assets
```

Scripts should avoid overwriting existing files.

---

## 7. Documentation Requirements

### `docs/maintenance-audit.md`

Must include:

* Framework/build system
* Current routing
* Current content organization
* Current blog implementation
* Current portfolio implementation
* Current editor/preview behavior
* Current asset handling
* Deployment assumptions
* Maintenance pain points
* Refactor risks

### `docs/architecture.md`

Must include:

* Content directory structure
* Block rendering architecture
* Blog loading pipeline
* Portfolio loading pipeline
* Asset handling strategy
* Deployment assumptions

### `docs/blog-authoring-workflow.md`

Must include:

* How to create a post
* How to add images
* How to set cover image
* How to set preview text
* How to preview locally
* How to publish
* Draft behavior

### `docs/content-maintenance-guide.md`

Must include:

* How to add/reorder portfolio blocks
* How to add a project
* How to add a publication
* How to add a blog post
* How to add a note
* How to check content/assets

### `docs/agent-worklog.md`

Append after every meaningful phase:

```md
## YYYY-MM-DD HH:MM

### Changed
- ...

### Preserved
- ...

### Validation
- ...

### TODO / Risks
- ...
```

Keep it concise.

### `docs/refactor-checklist.md`

Track:

* Completed phases
* Remaining work
* Manual QA checklist
* Known limitations

---

## 8. Migration Phases

### Phase 1: Audit Only

* Inspect repo structure.
* Identify framework, routing, build, content, assets, deployment.
* Write `docs/maintenance-audit.md`.
* Create/update `docs/agent-worklog.md`.
* Do not rewrite UI yet.
* Keep the current build behavior unchanged.

### Phase 2: Architecture Spec

* Create/update `docs/architecture.md`.
* Decide content schema based on current stack.
* Decide migration path.
* Add `docs/refactor-checklist.md`.
* Define compatibility rules for existing `/`, `/blogs/`, `/blogs/posts/<slug>/`, Korean post slug, legacy `?id=`, and `/projects/<slug>/` routes before moving content.

### Phase 3: Portfolio Content Model

* Move portfolio/home/about/resume data into content files.
* Add block renderer.
* Render existing sections through block architecture.
* Preserve existing visual style unless improving obvious issues.
* Validate build.

### Phase 4: Blog Content Model

* Standardize post metadata.
* Centralize blog loading.
* Improve blog list and article rendering.
* Support draft/published if feasible.
* Preserve existing posts and URLs.
* Validate build.

### Phase 5: Asset Workflow

* Reorganize or document asset placement.
* Add asset resolver/checker if feasible.
* Fix broken image references.
* Document static-hosting limitations.
* Validate build.

### Phase 6: Authoring Scripts

* Add `new:post`, `new:note`, `new:project` if feasible.
* Add `check:content` and `check:assets` if feasible.
* Document commands.

### Phase 7: UI Polish / QA

* Improve navigation clarity.
* Improve blog card UX.
* Improve article typography.
* Check mobile layout.
* Run lint/typecheck/build.
* Update final docs.

---

## 9. Validation Checklist

Run available commands after each major phase.

Check:

* Dependency install works.
* Lint passes if configured.
* Typecheck passes if configured.
* Build passes.
* Existing pages render.
* Blog list renders.
* Individual posts render.
* Images resolve.
* GitHub Pages deployment assumptions are preserved.

If something fails, either fix it or document the failure clearly in `docs/agent-worklog.md`.

---

## 10. Final Report Format

At the end of the refactor, provide:

1. What changed
2. New content structure
3. How to add a blog post
4. How to add a portfolio project
5. How to add images/files
6. Commands run
7. Checks passed/failed
8. Known limitations
9. Recommended next improvements

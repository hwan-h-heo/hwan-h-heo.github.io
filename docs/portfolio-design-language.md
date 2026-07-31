# Portfolio Design Language

## Intent

The portfolio should read as a refined editorial-tech publication rather than a
developer dashboard, research index, or product landing page.

The target order is:

1. Refined
2. Understated
3. Aesthetic
4. Modern
5. Fancy

"Fancy" comes from a small number of signature elements, not from adding more
components. Keep these signatures:

- the immersive hero visual
- strong display typography
- numbered and indexed navigation
- restrained motion
- generous but bounded whitespace

Everything else should stay quiet enough to support the work.

## Source Of Truth

Portfolio tokens and shared portfolio components live in
`assets/css/portfolio.css`. Project-page-only rules live in
`assets/css/project-detail.css`. The shared responsive sidebar lives in
`css/sidebar-nav.css`. Portfolio and project markup is framework-free; the
responsive `portfolio-shell` and the About/Project component layouts are owned
by these stylesheets rather than by generic framework utility classes.
Shared interface icons use the first-party SVG sprite and `.site-icon`
foundation documented in `docs/styles-and-assets.md`; component styles own
only their local size, color, spacing, and motion.
Portfolio and blog UI must remain framework-free at runtime. New components
should use semantic, component-owned class names rather than generic grid,
utility, or icon-font tokens; `npm run check:legacy-ui` enforces that boundary
in authored and generated files.

The blog has its own CSS surface because long-form articles need different
reading, code, math, and dark-theme rules. Do not copy values from
`blogs/css/blog.css` into the portfolio without mapping them to the system in
this document.

## Current Diagnosis

The portfolio home already has a coherent modern and editorial base:

- Manrope, Inter, and IBM Plex Mono have distinct roles.
- Near-black, slate, white, and pale blue surfaces carry most of the design.
- Cards are unframed rows rather than floating panels.
- Hairlines and whitespace create structure.
- Radius and shadow are limited to media and utility controls.
- Project and blog previews share the same hierarchy.
- Motion changes state without changing layout.

The main drift risks are:

- Generic layout helpers can obscure component ownership; keep new portfolio
  layout rules scoped to the shell or the component that uses them.
- Blog CSS retains older framework colors, shadows, radii, and tracking values.
- The shared dark sidebar has a surface-local cyan token.
- Direct hex and rgba values still exist in legacy or media-specific rules.

Treat the current portfolio tokens as canonical. Legacy values are not
precedent for new components.

## Color Grammar

### Neutral hierarchy

Use neutral color for almost all content:

| Token | Role |
| --- | --- |
| `--color-ink` | strongest interactive or editorial emphasis |
| `--color-heading` | titles and primary headings |
| `--color-text-strong` | emphasized body copy |
| `--color-text` | compact UI copy |
| `--color-text-body` | standard descriptions |
| `--color-text-soft` | metadata and tags |
| `--color-text-faint` | tertiary metadata and inactive icons |
| `--color-line` | standard structural hairline |
| `--color-line-soft` | repeated-row and image borders |
| `--action-underline-light` | low-contrast animated underline on light surfaces |
| `--action-underline-dark` | low-contrast animated underline on dark surfaces |

Do not create hierarchy by introducing another hue. Move up or down this
neutral scale first.

### Cyan roles

The three canonical cyan tokens are semantic:

| Token | Meaning | Allowed use |
| --- | --- | --- |
| `--accent-interactive` | active action or control | active navigation, small action icons, contact icons, control hover |
| `--accent-editorial` | quiet editorial emphasis | series/category labels, project type, accolades, card-title hover, inline-link hover underline |
| `--accent-dark-surface` | dark-surface contrast | hero CTA index, hero underline, hero subtitle, dark loading state |

Compatibility aliases remain for existing project and shared styles:

| Existing alias | Canonical role |
| --- | --- |
| `--accent-color` | `--accent-interactive` |
| `--accent-on-light` | `--accent-editorial` |
| `--accent-on-dark` | `--accent-dark-surface` |

Rules:

- Do not use bright cyan for paragraphs, names, institutions, dates, or tags.
- Do not use cyan as a large background or decorative wash.
- A compact component should normally have one persistent cyan signal.
- Inline links rest in `--color-link-muted`; their underline may become
  `--accent-editorial` on hover.
- Focus outlines use `--accent-line-strong` for accessibility.
- The sidebar's `--site-sidebar-accent` is a dark-surface local token. It must
  not leak into light page content.

## Typography

The typography system has three jobs:

| Family | Role |
| --- | --- |
| Manrope / `--heading-font` | names, section titles, project titles, commands |
| Inter / `--default-font` | descriptions, metadata, navigation, contact copy |
| IBM Plex Mono / `--mono-font` | indexes, categories, series names, small structural labels |

Rules:

- Use display weight and scale for hierarchy, not decorative type effects.
- Keep letter spacing at `0` in new portfolio styles.
- Use uppercase mono labels sparingly and keep them short.
- Limit small text to three levels: structural label, metadata, and action.
- Do not create a new font treatment for badges or tags.
- Keep body line height around `1.58` to `1.78`.
- Keep compact metadata around `0.72rem` to `0.84rem`.
- Use `text-wrap: pretty` for display titles where supported.

The editorial quality comes from scale, weight, alignment, and rhythm. It does
not depend on adding a serif family.

## Layout And Spacing

- The hero may use viewport height; content sections should not.
- Standard sections use generous vertical padding but cap their width at large
  viewports.
- Resume is subordinate to About and must align with the About portrait and
  copy grid.
- Project and blog preview rows must share column widths, gaps, and text order.
- Project and Blog remain on one pale output surface. Separate the two chapters
  with balanced whitespace rather than a background-color change or an
  additional divider.
- Fixed-format elements need stable dimensions or aspect ratios.
- Use whitespace to group related content before adding borders or containers.
- Major narrative sections may have larger gaps than reference sections.
- Avoid equal full-page treatment for every section.

At 4K widths, increase content width modestly rather than scaling whitespace
with the viewport indefinitely. Existing 1500px and 1600px caps are the model.

## Surface And Border Grammar

The default component is unframed.

- Use `--color-line-soft` between repeated rows.
- Use `--color-line` for deliberate controls and contact rows.
- Media may use a 1px soft border and a maximum 8px radius.
- Avoid card shadows on portfolio and blog preview rows.
- Avoid nested cards and floating section containers.
- Avoid pill shapes for taxonomy; use inline text separated by a middle dot.
- Circular shapes are reserved for icon-only controls and portraits.
- Remove a border when whitespace already explains the relationship.

Shadows are acceptable only for transient or floating utility UI such as the
mobile sidebar toggle, tooltip, and scroll-to-top control.

## Content Hierarchy

### Project and blog previews

Both preview types use the same order:

1. Editorial label: project type or blog series
2. Title
3. Two-line description
4. Institution and year, or publication date
5. Up to two representative technology tags
6. One meaningful accolade when available

Do not add a badge layer. Do not repeat the same classification in multiple
rows.

### Resume

Resume uses an editorial index rather than cards:

- equal Experience and Education columns
- period aligned separately from entry content
- neutral descriptions and metadata
- muted inline links with a visible underline
- one compact Download CV action

### Papers and talks

Use a shared index/table grammar. Do not introduce a separate card system.

- Papers use an index, copy, and action rail; talks use a date and copy rail.
- Repeated rows share the same hairline, vertical rhythm, title scale, and
  metadata hierarchy.
- Publication actions and Download CV share the light-surface, left-origin
  underline grammar. They are text actions, never bordered buttons.
- Blog preview eyebrows show the series only. `Post` and `Note` remain data
  categories for filtering, not visible metadata.
- Portfolio and Blog home technology tags share the same neutral inline text
  treatment and middle-dot separator. Preview tags do not use pills.
- Blog home preview footers show the publication date only; do not repeat the
  series or language availability already communicated elsewhere.
- Portfolio home preview descriptions reserve a two-line slot before tags and
  metadata so short copy does not disturb row alignment. Dense Blog archive
  rows follow their description's natural height.
- Blog home archive rows mirror the Portfolio preview hierarchy and alignment,
  while retaining a more compact media rail for the denser archive context.
  Keep top-aligned copy, description, tags, and date in one grid.

## Interaction And Motion

Motion should confirm an interaction, not advertise itself.

Canonical timing:

| Token or value | Use |
| --- | --- |
| `--motion-fast` / 180ms | small opacity or color response |
| `--motion-base` / 240ms | text color and compact UI state |
| `--motion-slow` / 360ms | icon movement and image transform |
| `480ms --ease-emphasized` | signature underline and arrow reveal |

Rules:

- Animate `transform`, `opacity`, color, and underline scale.
- Do not animate dimensions, padding, or grid tracks on hover.
- Keep movement within 1 to 4px.
- Respect `prefers-reduced-motion`.
- Keyboard focus must expose the same meaning as hover.

Project and blog card hover has exactly three signals:

1. Media scales to `1.02` with a slight filter adjustment.
2. Title moves from neutral ink to `--accent-editorial`.
3. A small external arrow reveals in place.

Do not add a card lift, background fill, shadow, border-color flash, or summary
animation to this state.

Hero CTA, Selected/All, and Download CV share the left-origin underline
language. Line thickness may differ by surface: 2px on the dark hero, 1px on
light editorial controls. Animated lines use the neutral
`--action-underline-*` tokens rather than a full-strength accent.

Light-surface text actions such as Download CV and publication or talk links
use underline motion only. Do not translate their text or icons on hover.

## Responsive Rules

- Preserve the information hierarchy when columns collapse.
- Keep title, metadata, and tag text large enough to scan on mobile.
- Do not let labels or actions resize their container on hover.
- Use one-column project and blog previews below the existing mobile breakpoint.
- Keep image aspect ratio stable.
- Avoid viewport-scaled font sizes outside established `clamp()` ranges.

## Change Checklist

Before merging a visual change, confirm:

- The change uses an existing token or adds a clearly semantic token.
- Bright cyan indicates action or active state, not general emphasis.
- Project and blog previews still share the same hierarchy.
- No new badge, pill, card surface, shadow, or divider was added unnecessarily.
- Small text still fits one of the three established roles.
- Hover and keyboard focus convey the same action.
- Motion uses the established duration and easing scale.
- The layout is checked at desktop, 390px mobile, and a wide viewport.
- `npm run build` and the rendered-site check pass.

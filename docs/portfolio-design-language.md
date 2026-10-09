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
`assets/css/portfolio.css`. The accepted fixed-chapter home layout lives in `assets/css/portfolio-chapters.css`.
The shared palette, local font roles and reading masthead live in
`assets/css/site-theme.css`, with a pre-paint preference in `js/site-theme.js`.
Project-page rules live in `assets/css/project-detail.css`; `blogs/css/site-reading.css`
shares the sculpture backdrop and local type with the blog while preserving its native
content palette and reading components.
The shared responsive Labs sidebar lives in
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
- Near-black, slate, white, cool pale blue, and soft-neutral surfaces carry
  most of the design.
- Cards are unframed rows rather than floating panels.
- Hairlines and whitespace create structure.
- Radius and shadow are limited to media and utility controls.
- Project and blog previews share the same visual system while preserving
  content-specific hierarchy.
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

### Surface roles

| Token | Role |
| --- | --- |
| `--color-page` | legacy paper alias; fixed chapters use a transparent reading scrim |
| `--color-section` | cool pale-blue Projects surface |
| `--color-writing-section` | near-white soft-neutral Portfolio Writing surface |
| `--color-surface` | neutral component and media surface |

The accepted v32 home uses one dark-default reading scrim across Projects, Blogs
and About. Light paper is an optional persisted theme. The older pale chapter
surfaces remain compatibility tokens for reusable preview rules.

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

Readable metadata such as dates, categories, affiliations, and compact actions
must use at least the surface's `--color-text-soft` or `--blog-color-muted`
contrast. Reserve faint or subtle values for redundant folios, separators, and
inactive icons.

On the Portfolio home, chapter markers, numbered Project/Writing folios, and
row metadata should sit one neutral step above their surrounding separators.
Keep the type compact, but do not rely on low contrast to make it subordinate.
On light Portfolio surfaces, use `0.70rem` for chapter, index, eyebrow, and row
folio labels; use no less than `0.68rem` for compact detail and disclosure
labels. The About point-cloud affordance may sit at `0.66rem` because it is an
ancillary enhancement hint. The dark Hero's out-of-flow margin notation remains
the contained exception and may stay smaller.
Within a featured Project, a meaningful detail label such as `CONTRIBUTION`
uses IBM Plex Mono `600` at `0.70rem` and `--color-text-strong`. Keep it neutral
rather than adding a second cyan signal beside the Project eyebrow, and use
compact tracking so the label reads as structure instead of faint metadata.

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
| Manrope / `--heading-font` | names, project titles, commands |
| Space Grotesk | Portfolio-home chapter `h2` headings and matching Hero section-navigation labels |
| Inter / `--default-font` | descriptions, metadata, navigation, contact copy |
| IBM Plex Mono / `--mono-font` | indexes, categories, series names, small structural labels |

Rules:

- Use display weight and scale for hierarchy, not decorative type effects.
- Optically align Portfolio section headings with the copy beneath them: keep
  description blocks on the structural left edge and apply one shared, subtle
  leftward correction to large Manrope headings instead of adding per-section
  paragraph margins. The compact Projects credibility line is the deliberate
  exception: retain its small positive offset so its smaller glyphs align
  optically with the display heading above.
- Keep Portfolio section headings non-interactive. Put section-level navigation
  in a separate compact text action so headings retain one consistent chapter
  role.
- Let the Portfolio Blogs description use up to 700 px at its existing type size,
  so the full desktop description occupies two lines. Keep “production engineering”
  together; allow the paragraph to reflow naturally on narrower screens.
- Keep letter spacing at `0` in new portfolio styles.
- Use uppercase mono labels sparingly and keep them short.
- Treat the Portfolio About standfirst as descriptive copy, not structure: use
  a high-contrast Cormorant Garamond italic rather than mono, with only a small
  optical size correction for its low x-height. This is a contained accent, not
  a fourth general-purpose text role.
- Limit small text to three levels: structural label, metadata, and action.
- Do not create a new font treatment for badges or tags.
- Keep body line height around `1.58` to `1.78`.
- Keep compact metadata around `0.72rem` to `0.84rem`.
- Use `text-wrap: pretty` for display titles where supported.
- Long-form blog `h4` headings use a short horizontal solid accent rule as an
  index mark, never a gradient or a repeated left-side rail.
- Long-form blog tables use a compact `4px` outer radius, hairline cell borders,
  and no shadow so their hierarchy comes from typography and rules.
- Long-form code blocks use the same compact `4px` editorial radius, a neutral
  hairline and faint wash, and a borderless copy control. Give every block a
  quiet mono `CODE` folio, extended to `CODE / LANGUAGE` when the Markdown fence
  identifies a language. Avoid floating copy buttons, card shadows, and
  utility-panel styling. Collapsible code uses unframed top and bottom
  hairlines, a `CODE /` folio, and a quiet text-only state marker instead of a
  rounded card shell.
- Interactive article embeds, including custom 3D viewers, stay within the
  reading column. Sidebar offsets belong only to standalone Labs surfaces and
  must never shift an embed inside article content.
- Normalize captioned article media into numbered folios such as `FIG. 01 /`.
  Keep the index in neutral mono type, the caption in muted body type, and media
  at a compact `4px` radius; source links retain the article link treatment.
  Fold legacy inline `figcaption` markup, emphasized caption lines, and isolated
  single-item caption lists into this same figure grammar during the static
  build. When one captioned legacy image has an explicit sub-100% width, keep
  the caption centered on a related, readable measure rather than letting it
  span the full article column. Keep ordinary explanatory paragraphs and
  multi-item lists in the body.
- Long-form blockquotes act as editorial annotations because source content uses
  them for questions, theorems, summaries, and quotations. Use a faint neutral
  wash, one neutral hairline, and a short editorial-accent cap; do not add a
  quotation glyph, rounded card, shadow, or semantic label that may misclassify
  the content.
- Optional mathematical derivations in articles use native `details` elements,
  initially closed, with a descriptive `summary`, top and bottom hairlines,
  and a visible keyboard focus. Keep the content in the reading column and
  allow wide equations to scroll within the disclosure. Use article-local
  styles for an article-specific treatment. Short explanatory videos retain
  the numbered figure grammar, use a static poster and native playback controls,
  and load on demand. Articles may opt into muted playback while a video is in
  view; pause it offscreen, in hidden tabs, and inside closed disclosures.
  Honor manual pauses and keep playback manual under reduced-motion preferences.

The core editorial quality comes from scale, weight, alignment, and rhythm. It
does not depend on serif type beyond the contained About standfirst accent.

On the Portfolio Hero, use the approved monochrome sculpture composition. Keep
all identity copy left-aligned, with the small `00 / PORTFOLIO` folio outside the
name's flow. Separate name/role/affiliation from the three-line practice paragraph
with a larger vertical pause. Keep Projects / Blogs as two compact numbered
chapter links below that paragraph. Put a small underlined `About me ↗` link
below the affiliation as part of the identity group. The cover masthead carries
only the signature; reading chapters retain their full navigation. The canonical cover
sizes, material, motion and responsive rules are specified below in Interaction
And Motion and implemented in `assets/css/hero-sculpture.css`.

## Layout And Spacing

- The hero may use viewport height; content sections should not.
- Standard sections use generous vertical padding but cap their width at large
  viewports.
- The career and CV index is subordinate to About. Its rules use the complete
  About shell width, while the expanded Experience and Education content uses
  the same full-width two-column measure before collapsing to one column.
- On wide Portfolio layouts, keep the About copy on a bounded reading rail and
  let the portrait rail absorb the remaining width. Center the portrait within
  that rail so the section balances across the shell instead of leaving unused
  space outside a fixed two-column grid. Treat the person's name as the portrait
  caption headline, clearly above role, affiliation, and point-cloud notation.
  Right-align that caption to the portrait edge on wide layouts so it reads as
  a deliberate margin note, compensating for the source PNG's transparent right
  inset so the text follows the visible portrait rather than the image box;
  return it to left alignment in compact flow.
  Optically lift the wide-layout portrait figure to compensate for transparent
  image headroom; reset that lift when the figure returns to normal single-
  column flow.
- Give the About standfirst and portrait modest visual emphasis without turning
  either into a second hero: the standfirst remains a compact editorial lead,
  and the wide portrait stays near 300px rather than dominating its rail.
- Continue the Portfolio About `--color-page` surface through the home footer
  instead of ending on a contrasting strip. Treat the centered footer note as
  a closing colophon: reuse the About standfirst's Cormorant Garamond italic at
  a smaller but clearly readable scale and neutral contrast, without adding a
  divider. From `1600px` upward, let the About chapter's bottom padding grow
  gently from the standard `80px` to a maximum of `112px`, avoiding both an
  abrupt short ending and viewport-fitted empty space.
- Project and Portfolio-home Blog preview rows share their row spacing, media
  treatment, and interaction grammar; their column proportions and metadata
  order may differ to express artifact versus publication.
- On wide Portfolio layouts, the featured Project row spans the full project
  index measure: keep its media rail bounded and let its copy rail reach the
  shell's right edge so it aligns with the standard project rows below.
- Project and Portfolio-home Blog use complementary pale chapter surfaces:
  Projects stays on cool `--color-section`, while Writing uses the near-white
  soft-neutral `--color-writing-section`. Preserve balanced whitespace at the
  transition and do not add another divider.
- On the Portfolio home, the desktop gutter exposed while the auto-hidden
  sidebar returns at the Hero-to-Projects boundary must use `--color-section`,
  matching the Projects surface without a white transition strip.
- Fixed-format elements need stable dimensions or aspect ratios.
- A case-study overview may place one outcome-focused media figure between its
  overview copy and contributions when the result is the clearest proof of the
  work; do not repeat the same media later in the article. Keep its caption
  muted and italic, and underline only the linked destination text.
- Long-form blog posts use the same heading family, weight, spacing, and scale as
  project-detail titles. Above the title, render a neutral `SERIES /` label and
  an underline-free cyan series link ending in one quiet directional arrow so
  the hierarchy and destination are both explicit without adding a persistent
  rule beneath the text. The subtitle becomes an Inter regular editorial
  standfirst, with a muted neutral `TOPICS /` label and topics directly beneath
  it. Keep clickable topics in editorial cyan, non-clickable topics in a more
  legible neutral, and middle-dot separators in the faintest neutral so link
  state is clear without persistent underlines. Series and topic links reveal a
  restrained underline on hover and keyboard focus only. Render middle-dot
  separators outside the links and never underline them; do not repeat the
  directional arrow on individual topics.
  Opening body paragraphs use the regular body treatment without a lead
  paragraph, drop cap, decorative initial, or separate container. Korean post
  titles use a Manrope-to-Noto Sans KR mixed
  script stack so Latin technical terms retain the heading character; use a
  matching title weight, looser line height, and less aggressive negative
  tracking than English. Protect ASCII hyphenated technical compounds such as
  `IO-Aware` from breaking internally in article titles; allow the surrounding
  title to rebalance naturally. Korean standfirsts keep words intact and use
  additional line height to offset the density of Hangul blocks.
  Treat structural mono labels as a locale-independent publication imprint:
  `SERIES`, `TOPICS`, `AUTHOR`, `PUBLISHED`, `READING`, `FEATURED`, and archive
  chapter labels remain English in every locale. Localize their accessible
  labels and reader-facing actions where useful, but do not mix translated and
  untranslated structural labels in the same composition.
  Place this copy beside
  a narrow margin note with separate author, publication-date, and reading-time
  rows. Keep the standfirst and margin note visually balanced, then let the body
  follow without opening cover media so posts with heterogeneous source imagery
  retain one consistent editorial rhythm. Lead with
  `Author / Hwan Heo` so ownership is visible before the reader reaches the full
  author note at the end of the article.
  Do not add a decorative accent rule or a visible `Article Details` heading;
  do not duplicate the series link in the end matter. On compact screens, turn
  the margin note into three inline metadata columns without a card border. The
  opening may be modestly wider and asymmetric; the article that follows
  returns to the standard reading measure. The
  masthead, utility row, and article share one uninterrupted page surface with
  no divider between title and body. Cover art remains for previews and social
  metadata rather than appearing behind the article title, while the utility
  row stays aligned to the body.
- Source articles should begin with a level-two section. When an article opens
  with summary prose, use `## Abstract`; when it explicitly offers a compact
  takeaway, use `## TL; DR`. Keep that opening as ordinary paragraphs rather
  than a blockquote or summary list so every article enters the reading flow
  with the same hierarchy.
- An implementation article may place one self-contained live figure before the
  opening section when that artifact is the subject of the article. Keep the demo
  on the dark Three.js surface, isolate it from the portfolio hero, and provide
  pause, reduced-motion, offscreen-pause, and static-fallback behavior. Use a
  responsive `16:9` frame that becomes `4:3` on compact screens; do not let its
  scripts or binary assets load on the portfolio home page. When pointer orbit
  helps establish that the artifact is a 3D scene, clamp it to a subtle authored
  envelope (currently ±10 degrees around the submitted view) and generate any
  visibility-culling assets against that complete envelope.
- Public Blog and Post pages are an independent publication without the
  portfolio masthead or sidebar. The home has a quiet `Hwan’s Blog` imprint in
  its existing utility row; posts show `Blog Home` there at every width. Search,
  theme and language remain in that one row. Keep a visible Portfolio link and
  a compact Labs disclosure (3D Viewer / Markdown Editor) in this same row on
  home, post, search and archive pages. Footer links are supplementary. On
  phones, destinations recede while the inline search field is expanded; Labs
  closes on Escape or outside interaction and preserves keyboard navigation.
  Below 1600px, a native Contents disclosure before the body keeps the nested
  heading index reachable, including on mobile. At 1600px and above, retain the
  persistent right-margin TOC after entering the body. Do not add a reading
  progress edge or another navigation bar. Public Labs keep their existing
  72px icon rail; the publication does not load that rail's CSS or controller.
  The post search icon expands an inline field before navigating to results;
  opening search must not discard the reading context, and the Home label may
  recede to preserve input width on phones.
- The public Markdown Editor uses that same dark sidebar as its only persistent
  left rail. Keep browser and Drive draft utilities in a transient right-side
  `Draft tools` drawer, and omit repository publishing, existing-post loading,
  Blog Home curation, and portfolio-feature controls. The local authoring
  console on port `3030` keeps its dedicated left control rail and the
  `Blog Editor` name because it owns those repository operations.
- Blog home, archive, and search utility rows use borderless back links and
  underline-only search fields. Do not use pill containers for back navigation,
  search inputs, or result counts. Search, Tag, and Series pages continue the
  dark-cover and numbered-chapter grammar as `00 / SEARCH·TOPIC·SERIES INDEX`,
  `01 / RESULTS·ARTICLES`, and—when the configured archive boundary exists—
  `02 / FROM THE ARCHIVE`.
  Their cover is the same calm `#101011` ink without the obsolete photographic
  banner. Their preview rows use the Blog home hierarchy: series, title,
  subtitle, localized publication date, then tags. Search only the active locale
  and link directly to that locale's article; do not expose a separate
  `Languages` field. Apply the same current-right / archive-left media signature,
  collapsing back to media-first rows on mobile. Wherever a mobile sidebar
  remains available, its scrim and panel must stack above any fixed utility bar.
- Do not add breadcrumbs before a long-form blog article; the original utility
  row provides the Blog Home return path.
- Close each article with an editorial author note containing a portrait, name,
  professional scope, and restrained `Email` and `LinkedIn` links in English in
  every locale. Treat it as a signature rather than another navigation menu,
  and separate the following related-post section with whitespace instead of a
  hairline. This is the primary ownership signal; the copyright footer remains
  subordinate.
- Author portrait cutouts use a small closed foreground mask to recover isolated
  missing pixels. Match the image underlay to its neutral source matte in both
  themes so transparent edges never read as a clipped second background.
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
- In the desktop rail, Labs and post Contents use the same anchored-flyout
  grammar: align the flyout's top edge with its trigger, open it to the right of
  the rail, and keep only one flyout open. Treat both as transient navigation:
  close on outside click or Escape and restore focus. Labs uses a quiet
  `LABS / COUNT` heading plus short tool descriptions; it is a tool switcher,
  not a card grid or a second sidebar.

Shadows are acceptable only for transient or floating utility UI such as the
mobile sidebar toggle, tooltip, and scroll-to-top control.

## Content Hierarchy

### Blog home cover

The Blog home opens as a compact dark editorial cover, not a photographic
landing-page banner or a light newspaper masthead. Use the Portfolio Hero and
Three.js canvas base ink (`#101011`) so both homes share one branded dark stage;
do not derive the surface from the former source photograph's average color. Its
first role is to create a calm, dark contrast with the pale paper below; avoid a
brown, green-biased, blue-gray,
or saturated accent-colored cover. Keep the title and standfirst as static HTML,
omit the full-bleed background image, and let Featured supply the first content
image. The Hero copy uses the exact same content measure as Featured and Archive
below; do not create a narrower inset for the cover. On wide screens only, make
one visual stage from the Hero's left edge to the shared copy measure's right
edge. Let equal left and right edge regions meet at its midpoint, using the
hierarchical-decoding frame on the left and the generated asset stage on the
right, then bridge their seam with the centered sparse-representation pipeline.
The center visual may overlap the two edge regions; the edge regions themselves
must meet without a gap, and the right artwork must end with the copy measure so
the top-bar side retains dark tension. Scale all three against the Hero height
with dark space around them; they are background atmosphere rather than a
full-bleed collage. Cover them with a strong
deep-ink overlay that is darkest around the central copy and lower edge so the
text retains primary contrast. Reuse the Portfolio Hero name color
(`#efefeb`) for the Blog title, then step the standfirst and
publication folios down through the same neutral family instead of returning to
pure white or blue-gray. Hide all three images on compact layouts before they could
compete with the copy. Connect the cover to the numbered page chapters with one
quiet `00 / TECHNICAL WRITING · YEAR—YEAR` publication imprint so the section's
subject is immediately legible without adding a separate logo treatment. Use a
`WRITTEN BY / HWAN HEO` byline on its own following row, right-aligned to the
copy measure on wide screens and left-aligned on phone layouts. Keep Posts /
Series counts in the `02 ARTICLES` tabs instead of duplicating them in
the Hero. Use a single-line title on wide screens and a single-line standfirst
where the measure allows it. Keep the wide-screen utility search
compact enough that its left edge begins beyond the visible title text instead
of drawing a rule across the title's horizontal field. Below `768px`, reduce
Blog search fields to the search icon; activate the icon to expand the input
toward the left, following the post utility search interaction. Hide the shared
sidebar hamburger throughout compact Blog layouts because the Blog utility row
already supplies the relevant navigation and controls. Let the title wrap
naturally only when the compact viewport requires it. Do not add a vertical
divider or boxed rail between the title and standfirst. On compact layouts,
stack the standfirst and index beneath the title.
Search, theme, and language utilities use the cover's dark-surface contrast. The
treatment must stay compact: never exceed the former `330px` Hero footprint on
wide screens and keep the complete mobile cover within roughly `400px` at the
390px reference viewport. Balance the cover transition around the Hero edge:
the distance from the standfirst's bottom to the Hero edge must equal the
distance from that edge to `01 FEATURED`. Keep the wide reference at `64px` and
the compact reference near `48px`; the right-aligned publication-index row lives
inside the upper interval so the spacing carries hierarchy rather than reading
as empty padding.

### Project and blog previews

Portfolio Project previews use this order:

1. Project folio and editorial type
2. Title
3. Two-line subtitle
4. Up to two representative technology tags
5. Institution and year, with one meaningful accolade in the same neutral
   metadata row when available

When a Portfolio project is explicitly marked `featured`, treat it as the
chapter's opening spread rather than another equal index row. On wide layouts,
use a media rail capped near `480px` beside a copy rail capped at `600px`. Keep
the featured media on the chapter's structural left edge. This restrained
asymmetry gives the copy enough room without making the feature feel like a
full-width hero; when wider canvases leave additional room, let that unused
space remain on the outer right.
Enlarge its title by only one restrained type step, keep its body copy close to
the standard project scale, and vertically center the copy
against the media. Give the copy the rhythm of a publication lead: a concise
service definition followed by one hairline-separated `Contribution` brief.
Do not repeat the service's output classes as a separate Capabilities row when
the definition already states them. The brief uses a small mono label and
compact prose rather than pills, icons, or a feature card.
Add a larger closing pause, then return every remaining Selected project to the
same full-width compact row used by the All index: a media rail capped at
`300px` on the right, copy aligned to the left and to the media's top edge, and
the common thumbnail border and overlay. Keep this orientation consistent in
both Selected and All rather than alternating it by row. The left-to-right
reversal separates the opening feature from the compact index without forcing
a direct size comparison on one media rail. This feature-to-index shift must
remain clear without a card surface, badge, background, two-column card spread,
or additional accent color. Below `768px`, stack projects media-first; signal
the flagship with its larger title, fuller editorial brief, and closing space
rather than a different surface.

The VARCO 3D flagship cover uses a deterministic asymmetric mosaic of public
Explore thumbnails rather than a synthetic hero render. Mix several asset
categories at unequal panel sizes, preserving breathing room around compact
objects while allowing the primary panels to carry a fuller crop. The center
gutter divides material states without cutting an object: the left half begins
textured and the right half begins as untextured geometry. On fine-pointer hover
or keyboard focus, crossfade once to a matching image with those states reversed;
touch layouts retain the initial mixed still. Do not autoplay or loop this
transition, and remove its duration under reduced-motion preferences. Keep both
images free of labels, interface chrome, and decorative effects so the asset
variety and material-state change own the contrast.

The CaPa index cover is a purpose-built `3:1` editorial extraction of the
official pipeline artwork, not the complete paper figure or a second asset
mosaic. Present three large stages—generated geometry, geometry surrounded by
painted multi-view images, and the back-projected 4K textured mesh—as native
transparent artwork. Contain and vertically center that wide extraction inside
the common `16 / 9.4` white thumbnail frame so its top and bottom paper space is
equal. Connect the stages with quiet arrows and retain the source figure's bold
serif stage labels, including the method-defining `w/o Janus`; do not add
Portfolio folio numbers inside the artwork. Omit model names, longer paper
annotations, and the redundant input so the method remains legible at preview
scale and visually distinct from VARCO 3D.

Portfolio-home Blog previews use a publication-first order:

1. Writing folio, series, and publication date in one compact mono line
2. A headline one type step larger than the Project preview title
3. A two-line editorial standfirst
4. A compact `Read post` text action with the boxed external-link icon

Keep both Portfolio Project and Blog preview titles at Manrope `650`. Their
scale, spacing, and content hierarchy distinguish them from body copy; avoid a
heavier display weight that competes with the Space Grotesk chapter headings.

Do not repeat technology tags or the publication date beneath Portfolio-home
Blog standfirsts. Keep the right-side Blog media rail narrower than the Project
media rail so the headline, rather than the thumbnail, carries the row.
Keep the headline free of a trailing destination icon; the explicit action owns
the Portfolio-home Blog preview's persistent external-destination signal. Open
Portfolio-to-Blog navigation in a new tab because the Blog is treated as an
independent publication brand rather than another Portfolio section.

On the Portfolio home, use a complementary desktop orientation to mark the
transition from selected work into writing: Project previews place media on the
left and copy on the right, while Blog previews place copy on the left and media
on the right. Top-align Project copy with one small optical inset for a
consistent artifact-index scan without pinning the eyebrow to the media edge,
while vertically centering the larger Portfolio-home Blog headline stack
against its smaller media rail. On compact single-column layouts, return both
preview types to media-first order and top-align the copy without that inset.
Introduce both Portfolio preview groups with the Blog chapter-head grammar:
a short uppercase mono label, one flexible hairline, and the controls or action
that own the list. Projects keep the `Selected` and `All` filters with counts in
the section intro's right rail, balancing its compact title and description;
`PROJECT INDEX` and its hairline then provide a quiet pause immediately before
the rows. The Technical Blog intro mirrors that composition by keeping
`View all posts` in its right rail, using the boxed external-link icon and a new
tab for the independent Blog brand; `SELECTED WRITING` and its hairline then form
the quiet pause before the writing rows. On compact layouts, move both the
Project filters and Blog action beneath their complete intros. Do not assign
either Portfolio section a competing chapter number.

Give explicit Portfolio controls and collection actions such as `Selected`,
`All`, and `View all posts` the strong small-action weight. Set repeated
per-preview `Read post` actions one weight step lighter so they retain their
editorial cadence without competing with controls that change or leave the
section. The collection-level `View all posts` action may be one modest type
step larger than the compact filter tabs without approaching card-title scale.

Do not add a badge layer. Do not repeat the same classification in multiple
rows.

Preview rows are editorial reading surfaces, not full-row links. Only the media
and title are primary links; the title link may fill the horizontal line box it
occupies. Portfolio-home Blog previews add one compact `Read post` link to make
their publication role explicit; its repetition also supplies a restrained
editorial cadence, while the section-level `View all posts` action owns
collection navigation. Subtitles, tags, dates, organizations, and accolades
remain normal selectable text. Apply the same non-row-link rule to Blog home
featured, archive, and search rows.

On the Blog home, treat Featured and Articles as the two top-level publication
chapters: `01 FEATURED` and `02 ARTICLES`. Set both in the same short uppercase
mono grammar with a quiet trailing hairline. Posts and Series remain
unnumbered tabs within Articles; keep their counts and conventional active-tab
underline so the controls do not compete with the chapter folios. On compact
layouts, let the tab row move beneath the complete `02 ARTICLES` label and rule.
The Posts count represents every published Post, including the separately
presented Featured entry; exclude Featured only from the repeated Archive rows.
Non-featured preview media uses the system's compact `4px` radius, and its copy
begins at the media's top edge rather than being vertically centered. Keep its
tags and publication date as one compact metadata stack beneath the subtitle;
they should not read as separately spaced paragraphs.

Archive preview rows do not use per-item `P–NN` or `N–NN` folios. Start both
Featured and archive-row copy with `SERIES / NAME`, using a muted structural
label and separator before the editorial-accent series value. Follow with title,
subtitle, publication date, and tags in that order. Dates share the compact
mono treatment used by Featured, while archive rows use tighter vertical rhythm.
Only Featured closes with the restrained `Read post` action. Archive rows end
with their tags so copy never grows beyond the adjacent media merely to repeat
an action already available through the cover and title.

Use the Blog home Archive layout as a quiet era signature rather than a repeating
row pattern. Place media on the right for current writing, then return it to the
left from the post configured by `blogHome.archiveStartPostId` onward. The
current boundary begins with `Neural Rendering Beyond Photography`, marking the
shift from broader 3D generation and 3D AI writing into the earlier neural-
rendering body of work. This reverses the opening Archive spread from Featured,
then marks the archive with a single deliberate shift. Keep `02 ARTICLES` as
the tab-owning chapter so Posts and Series remain coherent, then insert one
`03 FROM THE ARCHIVE` folio immediately before the configured boundary Post.
Remove the final current row's bottom hairline at this boundary so it does not
double the new folio rule. Keep all text
left-aligned and reset to media-first stacked rows on the compact single-column
layout.

Featured titles and subtitles are never clipped with a line clamp. Let the
browser fit both through a small, bounded type-size adjustment after fonts load;
keep the copy stack's flex children from shrinking their line boxes. Reserve
16 px between the title and subtitle, reduced to 12 px on compact phones. The
tag row uses its actual tag font size for line-height, not the body font size.
On the side-by-side layout, the complete series-to-action copy stack must not
exceed the cover height. If text still needs more room at the minimum readable
size, release the fixed copy height and preserve the full text instead of hiding
it or compressing its spacing. Give Korean feature titles
the full copy width, retain natural word boundaries with `keep-all`, and balance
both the title and subtitle across their natural line counts so a single
short word never remains as an isolated final line. Keep the Featured technology
tags one type step smaller than archive-row tags, and use flexible space above
`Read post` so its baseline closes exactly at the cover's bottom edge instead of
shrinking the upper hierarchy unnecessarily.
### Career and CV index

Career and CV use the same editorial index grammar as Papers and Talks rather
than introducing a titled Resume subsection or cards:

- Do not render a visible Resume heading. Follow the About profile directly
  with two full-width index rows: Document / Curriculum Vitae first, then
  Career / Experience & Education. Do not split Experience and Education into
  separate disclosures.
- Keep the Document row informational rather than clickable as a whole. Only
  its compact Download text action and icon link to the CV file; do not repeat
  the file format as separate metadata.
- Use one opening hairline above Document and one inter-row hairline above
  Career. Leave the final Career disclosure open without a closing rule.
- Give the expanded index a generous top inset so its category labels do not
  attach visually to the disclosure rule.
- equal Experience and Education columns
- period aligned separately from entry content
- neutral descriptions and metadata
- muted inline links with a visible underline
- one compact Curriculum Vitae download action

### Papers and talks

Use a shared index/table grammar. Do not introduce a separate card system.

- Papers use an index, copy, and action rail; talks use a date and copy rail.
- Repeated rows share the same hairline, vertical rhythm, title scale, and
  metadata hierarchy.
- Papers and Talks keep their count plus chevron as the complete disclosure
  signal; do not add a redundant View details / Close label. The single Career
  disclosure retains that explicit state label because it follows a non-
  collapsible Document row in a different chapter context.
- Publication actions and the compact CV Download link use the light-surface,
  left-origin underline grammar. Neither is rendered as a bordered button.
- Blog-home preview eyebrows show `SERIES / NAME`. The `post` category remains
  an internal data value, not visible metadata or a per-row folio.
- The Blog home featured row retains that series context beneath its indexed
  `Featured` label, but uses tighter type rhythm and a compact
  `1.9:1` media crop on desktop and tablet (`2:1` on mobile).
- Portfolio and Blog home technology tags share the same neutral inline text
  treatment and middle-dot separator. Preview tags do not use pills.
- Blog home preview metadata places publication date before tags; do not repeat
  language availability. Keep `Read post` exclusive to Featured.
- Portfolio home preview subtitles clamp at two lines but follow their
  natural height; they do not reserve an empty second line. Dense Blog archive
  rows follow their subtitle's natural height.
- Blog home archive rows mirror the Portfolio preview hierarchy and alignment,
  while retaining a more compact media rail for the denser archive context.
  Keep top-aligned copy, subtitle, date, tags, and action in one grid.
- Extend the numbered-navigation signature into preview eyebrows with quiet
  folio notation. Use `P–01` for portfolio projects, `W–01` for Portfolio-home
  writing. Keep the folio neutral and the adjacent editorial label accented;
  it is margin notation, never a badge or a separate metadata row. Blog-home
  archive rows are the exception: their shared `02 ARTICLES` chapter label
  provides the index, so individual Posts do not repeat a folio.

## Interaction And Motion

Motion should confirm an interaction, not advertise itself.

### Fixed editorial chapters (v32)

The approved v32 design is integrated in the authored home, project template and
blog surfaces. Its reference sources remain under
`_workspace/2026-10-08-portfolio-editions/`. The rules below are canonical for
`.edition-site`; the earlier v22 continuous-scroll behavior is historical.
Projects, Blogs and About occupy the same viewport and crossfade
over the same fixed cover scene. Dark is the default reading surface; pale paper
is an optional reading theme. Keep the canvas and camera in place while paused,
without a second captured-image layer, crop, zoom or background swap. Fade in a
neutral charcoal reading scrim, hiding most detail behind the reading column and
retaining faint architectural detail at the outer margin. Do not invert lighting.
The portfolio removes the down-scroll cue and sidebar. Keep the cover's name,
role, three-line practice statement and unfilled Projects / Blogs links, with a
secondary About link under the affiliation.

The fixed cover decouples refinement from page scroll. Play the original
four-second coarse-to-fine entrance once on a cover load, then hold the completed
surface permanently. Do not add an idle refinement loop, reverse it on scroll,
or replay it on chapter return or Motion on. Direct reading-chapter links begin
with a finished surface and do not play the intro on return to Cover.

After the entrance, retain the original published homepage's camera parallax:
normalized pointer coordinates translate the camera by at most ±0.22 world units
horizontally and ±0.12 vertically, looking at the same fixed target. Use the
original 0.19 damping per animation frame and 0.002 settling threshold. Do not
rotate the form or alter its support pose in response to the pointer. Pointer
leave eases back to the authored camera. The one-time entrance is capped at
30 fps; pointer easing uses the original display-frame cadence. Cancel all
RAF/timer work once settled. Touch input has no parallax. Motion off pauses 3D
only; re-enabling it never restarts refinement. Suspend the renderer in reading
chapters, hidden tabs and offscreen. Reduced motion, data saving and WebGL failure
retain the static fallback.

The intro uploads ancestor surfaces, endpoint normals and the refinement schedule
once and blends them in the vertex shader. It reuses the stage's cached HDR color
and depth while the moving form uses the lighter mineral material. The completed
form and its pointer response use the authored PBR materials. Keep finished-pose
contact/cast shadows cached until viewport geometry changes. Camera movement does
not invalidate these world-space shadows because geometry and lights stay fixed.
Use a 1024 px shadow map and at most 1.8 million visible pixels (DPR up to 1.25,
1 on phones). A persistently slow entrance may resolve directly to its final
still pose.

Gesture cues use the same left-to-right top rule as a real CTA hover: `scaleX`
from a left origin over 480 ms with `cubic-bezier(.22,1,.36,1)`. The rule leads;
label/index/arrow color follows after 100 ms over 300 ms. Avoid an immediate
full-strength flash. On the cover, downward wheel or an upward finger swipe hints
at Projects; upward intent has no chapter destination. Use quiet cyan for the
rule (`#91c3cd` at 65% opacity), a subdued `#bed2d3` label and `#a8bdbe` index.
Actual hover/focus retains the clearer `#91c3cd` interactive role. Keep arrow
movement within 1 px for hints and 2 px for direct interaction.
Reserve 4 px inside each cover CTA's right edge so the moving arrow stays within
the clipped link, without changing the full-width top rule or overall CTA width.

On an idle cover, reuse that same quiet Projects rule sweep after six seconds
without pointer, touch, keyboard or focus activity. Show it for 1.4 seconds, then
rest seven seconds before repeating. Any interaction immediately clears the idle
cue; actual hover/focus and gesture hints take priority. Pause it outside Cover,
in hidden tabs and for reduced motion. This is a decorative cue, never a focus
change, live-region announcement or 3D animation. Use one quiet-period timeout
and one pulse expiry, with no per-frame polling or pointer-event timer churn.

In Projects, Blogs and About, downward scrolling hints at the right Next link.
Upward scrolling clears any pending gesture hint without emphasizing either link;
reading back through a long page must not imply previous-chapter navigation. Use the reading palette's
muted editorial color for hints and interactive color for direct hover/focus.
Preserve the page's native scrolling: hints never navigate, capture input, move
focus, jump to the footer, or drive the 3D object. Support wheel, touch, keyboard
scroll keys and scrollbar movement. Ignore pinch/zoom and horizontal gestures.
Accumulate 18 px of vertical wheel intent within a 500 ms gap, resetting that
accumulator when direction changes; touch requires 24 px. Keep feedback visible
until 2.4 seconds after the latest gesture. Repeated input extends one timestamp
without restarting the rule animation or writing the DOM on every event.

Direct hover or keyboard selection clears a gesture cue and blocks highlighting
the other action. Keep upward scrolling free of automatic CTA emphasis; Previous
responds only to direct hover or keyboard focus. Never leave both emphasized. When pointer and keyboard
selections differ, the latest direct interaction owns the color and rule while
the keyboard focus outline remains available. Clear hints on chapter changes and
ignore programmatic scroll restoration. Motion off preserves these UI transitions;
only the system's reduced-motion preference removes their animated movement.

Carry the original palette's semantic roles into the reading chapters. Keep the
sculpture and its background monochrome. Titles are off-white; emphasized copy,
body and metadata descend through distinct slate values. Project type and Blog
series labels use a muted cyan; active navigation, filter state, focus and action
feedback use a clearer cyan. Inline links rest in a quiet blue-gray. Folios,
dates, institutions and tags remain neutral. Do not flatten accent tokens to the
heading color or use cyan for entire paragraphs. Preserve the same roles in the
optional light tone with darker values appropriate to paper.

| Preview role | Dark | Light |
| --- | --- | --- |
| Heading / `--ed-ink` | `#e9ecef` | `#1e2b36` |
| Emphasis / `--ed-strong` | `#cfd8e0` | `#344553` |
| Body / `--ed-copy` | `#b2bfcb` | `#505f6b` |
| Metadata / `--ed-soft` | `#929fad` | `#596873` |
| Editorial / `--ed-editorial` | `#83afb9` | `#2f6d7d` |
| Interactive / `--ed-interactive` | `#78c8dc` | `#176b87` |
| Inline link / `--ed-link` | `#bdced4` | `#405f6b` |

Reserve `--ed-faint` for redundant decoration, and keep readable text at metadata
contrast or above. Map the original portfolio tokens to these roles within the
chapter, including separate standard and soft hairlines and tone-aware focus.

Projects and Blogs share a 16 px gap between the chapter title and its description
on desktop and mobile.

Restore the compact editorial hierarchy: 36 px desktop / 30 px mobile chapter
headings, 23 px featured project titles, 19 px project titles, 22 px blog titles
and 14 px reading copy. Keep titles in medium/semibold weights and the About
standfirst near 20 px. Keep desktop blog media at 300 px and featured project
media at 420 px. Paper-backed diagrams remain intact as compact figure plates:
apply a scoped 0.72 brightness multiplier on dark surfaces (0.86 on hover/focus)
to the whole plate, including letterboxing. Mark these assets individually;
do not key out white pixels, invert, blend away or regenerate diagrams. Other
thumbnails keep their original image colors. Public detail views retain originals.
The static About cutout uses a dedicated editorial alpha mask eroded by a 4 px disk
at the original 1049 × 925 resolution, then lightly feathered. Keep the visible
RGB pixels intact and never expand the original alpha. This removes the source
matte's pale contour on charcoal without changing the face or adding a backing
plate. Keep the original portrait and depth assets unchanged; the fixed home does not
load the point-cloud enhancement.

Each chapter scrolls internally, never into the next chapter. Bottom navigation
follows Cover → Projects → Blogs → About → Cover: left arrow is the preceding
chapter, right arrow the following chapter. About closes the sequence with an
explicit `Cover →` action. Keep the arrows at the outer edges of their
labels and align numbers and labels vertically. The persistent chapter index,
brand link, Escape, browser history and direct hashes keep all chapters reachable.
Bound the bottom navigation to a centered 440 px rail with two separate short
top hairlines and a 36 px gap, rather than one full-content-width rule.
Keep at least 44 px touch height and use
the established left-origin rule response on hover and focus. Restore the original
footer note below navigation in each independently scrollable chapter: “A small
collection of work, research, and ideas gathered along the way.” Keep its original
Cormorant Garamond italic at 16 px, neutral metadata color, centered and unframed.
It belongs to the content flow; the tiny fixed folio is not a replacement for it.
Move keyboard focus to the incoming heading, make hidden
chapters inert, and restore cover focus when returning. Fade outgoing text over
220 ms and introduce incoming text after 240 ms with a 440 ms fade, keeping motion
within 3 px. The reading scrim starts immediately and eases over 720 ms; chapter
changes leave it unchanged. Reduced motion removes these transitions. Freeze the
live renderer in place while reading, falling back to an embedded responsive
poster when required. Returning to the cover
resumes the existing scene without replaying its introduction. Preserve the
actual project/blog content, filters and disclosures. All project links use the native project routes generated by the common template.
Portfolio Blog links retain their new-tab behavior and use local article routes
so preview builds and deployed pages share the same navigation. This separation
is intentional: the Blog is an independent, information-dense reading destination
with its own theme preference. Do not merge its theme or tab policy with Portfolio.
Same-tab navigation between the home, project pages and Blog documents uses a
360 ms native root crossfade where cross-document View Transitions are supported.
Keep browser navigation, modified clicks, new tabs, anchors and history native;
never delay the link with an opacity timer or replace documents with fetched HTML.
Retain existing chapter fades for same-document navigation. Reduced motion and
unsupported browsers use ordinary document navigation without a fade.

The common project template follows the approved VARCO3D reference and extends the same dark-default palette,
header, navigation, typography and accent roles to long-form reading. Preserve
the actual project prose, contributions, metadata and original video pixels.
Use native document scrolling, a 720 px reading column and a quiet right-hand
metadata/contents column inside the 1180 px shell. Keep the title at 40 px,
section headings at 24 px and body at 15 px / 1.75; on phones use 28 px, 22 px
and 14 px respectively. Metadata remains 10–12 px. Below 800 px, metadata follows
the overview figure in the single-column flow. The faint cover atmosphere is a
static image under a charcoal veil; do not run another 3D scene behind the text.
Pause the embedded showcase when offscreen, hidden or explicitly paused, and
start paused for reduced motion or data saving. Keep figure colors intact,
provide a quiet play/pause control, and preserve the optional light surface.
Previous/next project links use the compact 440 px rail. This layout is scoped to `.case-study` and applies to every generated project.
Retain authored image sizes within the reading rail, make video/iframe media
responsive, and contain wide equations and tables in native horizontal scroll.

### Cover material and compatibility rules

The home owns chapter navigation in `js/portfolio-chapters.js`; its hidden chapters
are inert. Selected/All, disclosures and each chapter scroll position survive
reloads and detail return through session storage. A native project return restores
the departure link focus. The Blog is an independent publication: keep its original
utility row and editorial reading layout without the portfolio chapter masthead
or sidebar. Shared page backdrops, local fonts and compact type connect it to
the portfolio while the Blog retains its own original color system. Use
`Hwan’s Blog` in the home utility row and the existing Blog Home return on posts.
On phones, shorten `Blog Home` to `Blog` beside the left arrow; keep the destination
visible even for readers entering through a direct article link.
Preserve the post utility row's scroll-away/reveal-on-up behavior. Dark remains
the portfolio default; the independent Blog defaults to light. Portfolio and Labs
must be reachable in this utility row, without
scrolling to the footer or introducing a second bar. Compact Contents is a native
disclosure before the article; wide reading uses the existing right-margin index.
These indexes derive from the same generated heading tree with no duplicate IDs.

The Blog home, search and archives reuse the actual Portfolio sculpture render,
not just its background color. One decorative fixed pseudo-element uses the
existing responsive `assets/hero/sculpture/poster-{1440,768,390}.webp` frames
behind listing content and its footer. Keep those listing surfaces transparent.
Article pages use a plain paper or charcoal background throughout, with no scene
pseudo-element or poster download, to keep long-form reading distraction-free. Preserve
the home's original dark banner above this backdrop, including its hierarchical
decoding, sparse-inference and VARCO3D collage, typography and contrast overlay.
The banner remains dark in both reading themes; its original mobile layout hides
the collage below 768px. Let the same viewport-aligned sculpture frame appear
very faintly beneath the collage: a fixed decorative layer at 18% opacity, clipped
to the banner and beneath its original contrast overlay. Keep this dark banner
layer independent of the light reading veil. The shared sculpture remains visible
below the banner at its established reading strength.
Use a neutral charcoal veil in dark mode
and a warm paper veil in light mode: strongest beneath the reading column,
lighter at the outer margin so the form, architecture and shadows remain visible.
Keep the dark scene subordinate to reading: use a 96% charcoal veil near the
central copy, easing to 84% at the far edge. Phones use the portrait frame with
a 97.8–95% veil. Keep native document scrolling;
do not load the hero runtime, WebGL, animation or pointer tracking on the Blog.
The image is decorative, cannot intercept input and is omitted in print. Sticky
utility backing uses `--ed-header`. Search and archive openings adapt their text
and controls to the reading theme over the shared scene. Home banner controls
keep their light-on-dark treatment until the utility row scrolls onto content.
Blog content ink and syntax colors remain owned by `blogs/css/blog.css`,
`typography.css` and `post.css`, including the established green accent hierarchy.
Do not remap the whole Blog to Portfolio `--ed-*` tokens or replace its syntax hues.
Light article code blocks and tables stay at the page paper's brightness, with
a faint slate tint from the native Blog line color (30% line, 70% paper). Do not
mix white into panels: brighter rectangles interrupt the continuous reading
surface. Inline code, table headers and quotes use a 3% ink wash over this panel;
alternate table rows use just 1%. Keep rules soft with an 8% ink mix into paper.
Define these article-only surfaces in `site-reading.css`. Preserve the native
syntax hue hierarchy while adjusting text lightness for readable comments,
punctuation, language labels, literals and strings against these quiet surfaces.
Preserve dark component colors, caption and feedback ink colors;
copy-code hover uses the same light panel wash. Copy-link and its confirmation
surface use the light panel instead of white; hover and keyboard focus use the
muted panel with the native green accent. Dark keyboard focus matches dark hover.
Apply narrow fixes to actual
theme mismatches, such as unlabelled code blocks missing the dark text color.
Keep tables horizontally scrollable, with
4 px corners, soft rules and subtle alternate rows. Preserve article media source
pixels. In light articles, white-backed figures blend into paper with `multiply`;
the build marks local PNG/WebP/JPEG figures with opaque white corners and substantial
white space as `data-reading-surface="paper"`. Transparent cutouts, ordinary photos,
animated media, dark mode and enlarged dialog images keep their original display.
An authored `data-reading-surface="original"` opts out; `"paper"` can explicitly opt in.
Detection runs only at build time, with no reader-side image analysis or extra downloads.
The scrolled utility row keeps
native Blog text and control colors over the shared backdrop. Copy-link stays outside the left reading margin;
reset legacy sidebar positioning offsets explicitly when the rail is absent.

`js/site-theme.js` applies the theme before CSS paints, with independent preferences:
Portfolio home and project details use `portfolio-theme`, defaulting to dark;
the standalone Blog under `/blogs/` uses `blog-reading-theme`, defaulting to light.
The Blogs preview chapter belongs to the Portfolio and uses its preference.
Migrate the former `site-theme` into the Portfolio only. Ignore the old mirrored
`blog-theme` for the new Blog preference so a prior Portfolio choice cannot
override its light default. Theme toggles, URL overrides, new tabs, storage events
and BFCache `pageshow` synchronize only within the relevant area. The cover itself
remains monochrome ink in either Portfolio reading theme.
User-initiated theme toggles reuse the 360 ms whole-screen crossfade across
Portfolio chapters, project details and Blog pages. Capture complete old/new
palettes by suppressing per-element CSS transitions only during the fade. Keep
the overlay non-interactive; rapid toggles resolve to the latest choice without
moving focus or reading position. Initial paint, stored-preference sync, hidden
tabs, reduced motion and unsupported browsers apply the theme immediately.
Manual 3D Motion is stored separately as `portfolio-motion`. It never suppresses UI
feedback or plays the entrance again.

The lightweight `js/hero-loader.js` avoids the WebGL bundle on reduced-motion,
data-saving and directly opened reading pages. Direct reading-to-Cover and a later
reduced-motion release initialize only the finished geometry. A bundle completing
after chapter departure does not create a renderer until Cover is active again.

The production portfolio hero is the approved v32 sculpture cover, replacing the
former Wave. `js/hero.js` and `js/hero-sculpture/` own its source; the build emits
`assets/js/hero-sculpture.js`. Its material and base cover styles are in
`assets/css/hero-sculpture.css`, with fixed layout in `portfolio-chapters.css`, while copy remains in
`content/portfolio/home.json`. Do not load old Wave or article-demo code on home.

The cover uses a monochrome stone environment informed by onformative's AI Sculpting
spatial studies. A deep flared aperture and an original uneven carved form replace
the earlier ribbon/spiral composition. Build atmosphere through the relative scale
of architecture and sculpture, raking neutral light, cast/contact shadows and
multiscale mineral surface variation. Keep metalness and clearcoat at zero; avoid
colored rim lights, polished floor reflections and screen-space grain. Texture is
part of the material and affects diffuse response, roughness and surface normals.

Place the name, role, practice paragraph and adjacent Projects / Blogs in
the dark left margin as a readable editorial caption. Use real Manrope Regular
outlines for the moderate display name and practice statement: a CSS 400/500 request
must not silently fall back to the older 600/700-only font subset.
Keep role and affiliation in smaller Inter, and the out-of-flow folio in quiet mono.
Keep name, role and affiliation fixed. Place `About me ↗` immediately below the
affiliation, in 11.5 px muted text with a faint underline and a 44 px touch height.
It is a secondary profile link without a number or button frame. On desktop,
accommodate it within the existing pause before the practice statement, keeping
that statement and the primary chapter links in their established positions.
Its hover and keyboard focus share the primary CTA response: a 480 ms underline
sweep from left to right over the faint resting line, 300 ms accent color, a 1 px
lift and a 2 px rightward arrow motion. Keep only the selected Cover link highlighted,
including when focus and pointer target different links. Reduced motion keeps
the color and line feedback immediate, with no lift or arrow translation.
Allow a little more height on phones so the link and practice copy remain separate.
Set the practice paragraph at 16 px on desktop and 15 px on small phones,
with 1.72 line height and a bounded reading width. Distill it to three complete
phrases: building 3D generative systems, geometric representations, and efficient
GPU inference. Give each phrase its own line at standard viewport sizes, retaining
ordinary spaces in the accessible text. Allow phrases to wrap under text zoom;
do not shrink the font or force no-wrap to preserve the line count.
Keep decorative geometry clear of actual text/action bounds during pointer motion.
Projects / Blogs are unfilled chapter links with small mono numbers, Space Grotesk
labels, a single resting top hairline and the established hover line/arrow response.
Keep these labels compact at 12 px, with short rules on a 264 px action rail that
shrinks to the available width. Use a 22 px gap.
Align each small number and label on the same vertical center; do not independently
baseline-align the number against a centered label or add an upward offset.
Do not add a surrounding button frame or a filled primary action. Keep at least
44 px touch height. On compact screens, move the spatial scene below the introduction
and retain both primary actions and the profile link in the first viewport. CTA hover or keyboard focus
may subtly emphasize the sculpture or architecture.
The cover masthead keeps only the signature at left, including on mobile. The
primary actions and the About identity link are already within the introduction;
do not duplicate them in the cover masthead merely to fill the upper-right space. Reading chapters
keep the full masthead navigation.

Place a quiet down-chevron link to Projects at the bottom center of the first
viewport, with a 44 px hit area. Keep it visible even when a compact cover exceeds
the viewport height. Reuse the first-party chevron icon and restrained vertical
pulse; disable the pulse under Motion off or reduced motion. Keep keyboard focus
visible and honor reduced motion for the native section scroll.

The carved form refines once on the first cover load over about four seconds,
then remains finished. The fixed chapter layout never reverses geometry on scroll.
The active rules for scheduling, parallax, pause and fallback are specified above.
Keep source organization and font licenses in `docs/hero-sculpture.md` aligned
with the reproducible build.

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
- Treat the sculpture canvas as the spatial-motion exception: fine-pointer input
  may drive restrained damped camera parallax, while the finished surface stays fixed. Keep the authored object orientation fixed and stop rendering at rest.
- Treat the About portrait point cloud as a progressive enhancement. Keep its
  colored points orthographically aligned to one flat depth plane at rest; on
  fine-pointer hover, restore the authored depth gradually and allow only a
  bounded pointer-relative rotation. Allow one first-view settle when the About
  chapter reaches the same `200px` viewport boundary used by the `#about`
  scrollspy: hold the flat photographic portrait briefly, then ease the point
  layer and depth in with the angle, pass through one shallow left-to-right fan,
  and return to the flat portrait over roughly `4.5s`. Keep the pre-trigger
  portrait flat and never replay the intro. On fine-pointer devices
  only, repeat the same left-to-right-to-front path over `4.5s` after at
  least six seconds at rest. Fade the PNG almost entirely out during that fan so
  it reads as spatial rotation rather than blur, while keeping its depth and
  angle slightly quieter than the intro. Hover interrupts and takes priority.
  Identify the enhancement with one quiet mono `POINT CLOUD` annotation and
  show `/ HOVER FOR DEPTH` from first paint on fine-pointer layouts so
  enhancement readiness never changes the caption layout. Keep the hint hidden
  where hover is unavailable.
  At rest, let the portrait PNG carry `100%` of the image and hide the point
  layer completely, preventing a residual grid on compact displays. As the
  depth response opens, bring the points in on a quicker ease-out curve while
  keeping the PNG fully present through the first fifth of the response; only
  then ease the underlay away. This avoids a thin, blurry midpoint without
  leaving the PNG as a second, misaligned silhouette at full depth.
  Size point sprites from their projected screen-space sampling interval rather
  than DPR alone; overlap neighboring samples enough to prevent grid moire as
  the portrait card changes size.
  Fall back to the portrait PNG for iOS and iPadOS, reduced motion, data-saving
  mode, or an unavailable or lost WebGL context. Detect iPadOS when it presents
  a desktop-style `MacIntel` platform with touch points. In these static modes,
  do not import Three.js, create a WebGL canvas, or expose the point-cloud
  affordance; keep the PNG at full opacity. Desktop Safari and other desktop
  browsers retain the enhancement.
- Respect `prefers-reduced-motion`.
- Keyboard focus must expose the same meaning as hover.
- Keep the blocking portfolio preloader within about one second and exit it on
  `--motion-base`; ambient hero motion may continue independently.

Project and blog preview rows have no row-level hover state. Their media and
title link regions respond independently:

1. Hovering or focusing the media link scales its image to `1.02` with a slight
   filter adjustment.
2. Hovering or focusing the title link moves it to `--accent-editorial` and
   reveals its destination icon where that title owns the destination cue. Use
   the boxed external-link icon for `target="_blank"` and a simple directional
   arrow for same-tab links. Portfolio-home Blog headlines are the exception:
   they use color only because their separate `Read post` action owns a
   persistent boxed external-link icon and the light-surface underline motion.

On the Portfolio About contact index, use one small service icon before the
copy—Envelope for Email and LinkedIn for LinkedIn—and retain one quiet trailing
destination icon to anchor the far edge of each wide contact row. Use the
directional arrow for Email and the boxed external-link icon for LinkedIn. Keep
both neutral at rest and move them to the interactive accent on hover or
keyboard focus.

Do not add a card lift, background fill, shadow, border-color flash, or summary
animation to this state.

Hero CTA and Selected/All share the left-origin action-rule language. The Hero
chapter index places that rule on its top edge; light-surface controls keep it
below the text. Line thickness may differ by surface: 2px on the dark hero, 1px
on light editorial controls. Animated lines use the neutral
`--action-underline-*` tokens rather than a full-strength accent.

Light-surface text actions such as the CV download and publication or talk
links use underline motion only. Do not translate their text or icons on hover.

## Responsive Rules

- Preserve the information hierarchy when columns collapse.
- Keep title, metadata, and tag text large enough to scan on mobile.
- Do not let labels or actions resize their container on hover.
- Use one-column project and blog previews below the existing mobile breakpoint.
- Keep image aspect ratio stable.
- Avoid viewport-scaled font sizes outside established `clamp()` ranges.
- On compact project pages, reduce breadcrumbs to the useful section ancestor
  when the full trail would compete with the sidebar control or title. Blog
  articles omit breadcrumbs entirely.
- On mobile, the floating sidebar toggle hides while scrolling down and returns
  when scrolling up or reaching the top. Keep it visible while navigation is
  open so the control does not sit over long-form headings and preview media.
- Blog-home back-to-top remains desktop-only, while long-form post pages keep
  the compact control available on mobile as an end-to-top reading affordance.
- Persistent long-form TOC and share controls are desktop-only reading rails.
  Anchor both to the shared reading-column tokens so they remain outside the
  article body; compact screens use the native Contents disclosure above it.

## Change Checklist

Before merging a visual change, confirm:

- The change uses an existing token or adds a clearly semantic token.
- Bright cyan indicates action or active state, not general emphasis.
- Project and blog previews still share one visual system while retaining their
  artifact- and publication-specific hierarchy.
- No new badge, pill, card surface, shadow, or divider was added unnecessarily.
- Small text still fits one of the three established roles.
- Hover and keyboard focus convey the same action.
- Motion uses the established duration and easing scale.
- The layout is checked at desktop, 390px mobile, and a wide viewport.
- `npm run build` and the rendered-site check pass.

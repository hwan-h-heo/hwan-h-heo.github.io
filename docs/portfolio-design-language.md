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
- Optically align Portfolio section headings with the copy beneath them: keep
  description blocks on the structural left edge and apply one shared, subtle
  leftward correction to large Manrope headings instead of adding per-section
  paragraph margins.
- Keep letter spacing at `0` in new portfolio styles.
- Use uppercase mono labels sparingly and keep them short.
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

The editorial quality comes from scale, weight, alignment, and rhythm. It does
not depend on adding a serif family.

## Layout And Spacing

- The hero may use viewport height; content sections should not.
- Standard sections use generous vertical padding but cap their width at large
  viewports.
- Resume is subordinate to About and must align with the About portrait and
  copy grid.
- Project and Portfolio-home Blog preview rows share their row spacing, media
  treatment, and interaction grammar; their column proportions and metadata
  order may differ to express artifact versus publication.
- Project and Blog remain on one pale output surface. Separate the two chapters
  with balanced whitespace rather than a background-color change or an
  additional divider.
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
- Public Blog, Post, and Labs surfaces keep the shared dark desktop sidebar as
  a fixed `72px` activity rail; they do not expose its `300px` expanded state.
  The portrait and Blog, Portfolio, and Labs destinations remain the permanent
  rail grammar. Search, theme, and language stay in a slim, right-aligned
  utility row because they are familiar reading-context tools and must not be
  duplicated in the rail. On posts between `1200px` and `1599px`, add one
  contextual Contents trigger that opens the existing TOC as a transient rail
  flyout; at `1600px` and above, retain the persistent right-side TOC and hide
  that trigger. Do not add a separate reading-progress edge: the browser
  scrollbar and, on wide screens, the active TOC already communicate reading
  position. On compact screens, replace the sidebar trigger
  with a borderless `Blog Home` link at the left edge of the utility row. The
  post search icon expands an inline text field before navigating to results;
  opening search must not discard the current reading context, and the Home
  label may recede while the field is open to preserve input width.
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
- Do not render breadcrumbs before a long-form blog article when the persistent
  sidebar already provides the return paths.
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
edge, then divide it into two equal, non-overlapping regions. Center the complete
sparse-representation pipeline in the left region. Right-align the complete
generated asset-stage image in the right region so the artwork ends with the
copy measure and leaves the top-bar side as dark tension. Scale both against the
Hero height with dark space around them; they are background atmosphere rather
than a full-bleed collage. Cover them with a strong
deep-ink overlay that is darkest around the central copy and lower edge so the
text retains primary contrast. Reuse the Portfolio Hero name color
(`rgba(240, 241, 243, 0.88)`) for the Blog title, then step the standfirst and
publication folios down through the same neutral family instead of returning to
pure white or blue-gray. Hide both images on compact layouts before they could
compete with the copy. Connect the cover to the numbered page chapters with one
quiet `00 / TECHNICAL WRITING · YEAR—YEAR` publication imprint so the section's
subject is immediately legible without adding a separate logo treatment. Use a
`WRITTEN BY / HWAN HEO` byline on its own following row, right-aligned to the
copy measure on wide screens and left-aligned on phone layouts. Keep Posts /
Notes / Series counts in the `02 ARTICLES` tabs instead of duplicating them in
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

Portfolio-home Blog previews use a publication-first order:

1. Writing folio, series, and publication date in one compact mono line
2. A headline one type step larger than the Project preview title
3. A two-line editorial standfirst
4. A compact `Read post` text action with the boxed external-link icon

Do not repeat technology tags or the publication date beneath Portfolio-home
Blog standfirsts. Keep the right-side Blog media rail narrower than the Project
media rail so the headline, rather than the thumbnail, carries the row.
Keep the headline free of a trailing destination icon; the explicit action owns
the Portfolio-home Blog preview's persistent external-destination signal.

On the Portfolio home, use a complementary desktop orientation to mark the
transition from selected work into writing: Project previews place media on the
left and copy on the right, while Blog previews place copy on the left and media
on the right. Vertically center both copy stacks against their media so the
shared card grammar remains evident. On compact single-column layouts, return
both preview types to media-first order and top-align the copy.
Introduce the Portfolio Blog previews with one left-aligned `SELECTED NOTES`
mono label and a quiet trailing hairline, echoing the Blog chapter grammar
without assigning the Portfolio section a competing chapter number.

Do not add a badge layer. Do not repeat the same classification in multiple
rows.

Preview rows are editorial reading surfaces, not full-row links. Only the media
and title are primary links; the title link may fill the horizontal line box it
occupies. Portfolio-home Blog previews add one compact `Read post` link to make
their publication role explicit. Subtitles, tags, dates, organizations, and
accolades remain normal selectable text. Apply the same non-row-link rule to
Blog home featured, archive, and search rows.

On the Blog home, treat Featured and Articles as the two top-level publication
chapters: `01 FEATURED` and `02 ARTICLES`. Set both in the same short uppercase
mono grammar with a quiet trailing hairline. Posts, Notes, and Series remain
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
the tab-owning chapter so Posts, Notes, and Series remain coherent, then insert
one `03 FROM THE ARCHIVE` folio immediately before the configured boundary Post
or the first qualifying Note. Remove the final current row's bottom hairline at
this boundary so it does not double the new folio rule. Keep all text
left-aligned and reset to media-first stacked rows on the compact single-column
layout.

Featured titles and subtitles are never clipped with a line clamp. Let the
browser fit both through a small, bounded type-size adjustment after fonts load;
on the side-by-side layout, the complete series-to-action copy stack must not
exceed the cover height. If text still needs more room at the minimum readable
size, preserve the full text instead of hiding it. Give Korean feature titles
the full copy width, retain natural word boundaries with `keep-all`, and balance
both the title and subtitle across their natural line counts so a single
short word never remains as an isolated final line. Keep the Featured technology
tags one type step smaller than archive-row tags, and use flexible space above
`Read post` so its baseline closes exactly at the cover's bottom edge instead of
shrinking the upper hierarchy unnecessarily.
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
- Blog-home preview eyebrows show `SERIES / NAME`. `Post` and `Note` remain data
  categories for filtering, not visible metadata or per-row folios.
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
  provides the index, so individual Posts and Notes do not repeat a folio.

## Interaction And Motion

Motion should confirm an interaction, not advertise itself.

The production portfolio hero remains the established wave visual. The
coarse-to-fine voxel treatment is an experimental one-shot intro available only
through `?hero=voxel`; it must not add hierarchy-asset requests to the default
page load. After the voxel intro fully dissolves, dispose its renderer and fade
the established wave back in at its quiet ambient phase rather than restarting
the voxel sequence.

The experimental voxel visual uses a restrained, right-biased coarse-to-fine
surface refinement in a low-saturation graphite-to-silver family; avoid the
familiar bright blue AI/SaaS illustration palette. Begin with a fully occupied
coarse cube volume. Prune non-surface cells with a slow, legible
corner-to-corner, BFS-like wave. Before that front reaches the far corner, let
already-confirmed coarse surface cells hand directly into the same recursive
refinement so the opening is part of one continuous hierarchy rather than a
separate prelude. At every later refinement, let a similar front travel
recursively through the octree instead of replacing one complete level with the
next. Each reached parent first becomes all eight candidate child
slots; only after those slots separate do rejected children collapse. Retained
children then repeat that local sequence in BFS order within their own octant.
Relieve the otherwise empty black field with one very sparse layer of dim,
screen-wide cool-graphite particles. Keep them nearly static and too restrained
to read as a literal star field, glow effect, or CTA burst. Fade them with the
voxel dissolve so the established Wave remains the only persistent background.
Keep every hierarchy level resident during the cascade so coarse and fine voxels
coexist along the front, and never introduce a whole-level visibility switch.
Advance the complete cascade with a linear clock. Use one consistent local
duration and constant split/collapse velocity for every parent; give the global
cascade enough time that overlapping generations remain individually legible.
Reserve easing
for the initial dense reveal and final dissolve, not for the recursive wave
itself. Let camera-facing children begin their subdivision well before the
parent transition finishes. For the current depth-first treatment, begin the nearest child handoff at roughly 22%
of the parent transition and the farthest around 31%, so about three adjacent
hierarchy levels coexist. Once the upper-left front reaches a coarse sub-volume,
let that region descend toward the finest resolution before the broad front has
passed across the object. Start classification and recursive refinement on the
same linear clock, using the same corner rank and parent-coherent hash, so a
confirmed coarse voxel never pauses before its first split. Delay rear
octants through the same local BFS ordering so visible generations overlap while
the less visible back side catches up without creating a global pause.
Match the first refinement front's spatial duration to its overlap with the
dense prune: the last coarse parent must begin subdivision as the classification
front reaches that same cell, not after a separate trailing gap.
Retain every active index within each displayed resolution rather than thinning
individual levels; lower the complete displayed resolution range when
performance needs to be bounded. Keep the object itself axis-aligned,
visually smaller than the wave field, and near the wave hero's wide-screen
right-center locus. Use the submitted wide-screen reference of 38.5-degree
azimuth, 24-degree elevation, 42.5-degree field of view, and distance 13 for the
three-quarter view. Translate the model
along the camera's view-plane right axis so rightward placement does not alter
its screen-space height or depth.
Do not restart or reverse the hierarchy. Complete the moving refinement within
roughly five seconds of the visitor's initial scan, leave the final voxel silhouette for
only a quarter-second hold, then dissolve it smoothly over roughly 1.15 seconds before the ambient wave
fades in. Do not introduce
a source-mesh reveal after the hierarchy. Keep the final silhouette subordinate
to the left-side typography, retain only minimal
camera-based pointer parallax, and show a static fine surface when reduced motion
is requested.

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

Use the same destination mapping for trailing Contact icons. `mailto:` may keep
the directional arrow; external profiles use the boxed external-link icon.

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
- On compact project pages, reduce breadcrumbs to the useful section ancestor
  when the full trail would compete with the sidebar control or title. Blog
  articles omit breadcrumbs entirely.
- On mobile, the floating sidebar toggle hides while scrolling down and returns
  when scrolling up or reaching the top. Keep it visible while navigation is
  open so the control does not sit over long-form headings and preview media.
- Blog-home back-to-top remains desktop-only, while long-form post pages keep
  the compact control available on mobile as an end-to-top reading affordance.
- Long-form TOC and share controls are desktop-only reading rails. Anchor both
  to the shared reading-column tokens so they remain outside the article body.

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

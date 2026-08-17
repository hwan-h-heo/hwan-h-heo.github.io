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
| `--color-page` | white default page and About surface |
| `--color-section` | cool pale-blue Projects surface |
| `--color-writing-section` | near-white soft-neutral Portfolio Writing surface |
| `--color-surface` | neutral component and media surface |

Use surface tone only at chapter scale. Keep the Projects chapter cool and the
Portfolio Writing chapter softly neutral so the long-form index gains rhythm
without adding more accent colors inside its rows.

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
| Manrope / `--heading-font` | names, project titles, commands |
| Space Grotesk | Portfolio-home chapter `h2` headings and matching Hero section-navigation labels |
| Inter / `--default-font` | descriptions, metadata, navigation, contact copy |
| IBM Plex Mono / `--mono-font` | indexes, categories, series names, small structural labels |

Rules:

- Use display weight and scale for hierarchy, not decorative type effects.
- Optically align Portfolio section headings with the copy beneath them: keep
  description blocks on the structural left edge and apply one shared, subtle
  leftward correction to large Manrope headings instead of adding per-section
  paragraph margins.
- Keep Portfolio section headings non-interactive. Put section-level navigation
  in a separate compact text action so headings retain one consistent chapter
  role.
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

The core editorial quality comes from scale, weight, alignment, and rhythm. It
does not depend on serif type beyond the contained About standfirst accent.

On the Portfolio Hero, keep the display name at its established restrained
scale and place the quiet mono `00 / PORTFOLIO` folio and hairline above it as an
out-of-flow margin notation; it must not push the identity block downward. Then
stack the role, affiliation, and practice statement on a bounded reading rail.
On wide layouts, keep the folio and display name as the composition's
left-aligned anchor, right-align only the role-and-affiliation identity block,
and return the practice statement and CTA contents to the left reading edge.
Reset the identity block to left alignment on compact screens. Use the Portfolio
chapter-heading family for the Project, Blog, and About CTA labels so the cover
leads directly into the publication index; keep their numbers in mono and
keep the three CTAs in compact content-led spans rather than stretching them
across the deck width. On compact screens, let the same links share the available
touch width evenly. Maintain one
consistent, generous vertical rhythm from folio to name and name to deck, then
allow one modestly larger pause before the CTA index so navigation does not read
as another row of identity copy. Give each CTA a top-edge hairline and reveal
its stronger action rule from the left on interaction, making the row read as a
three-column publication index rather than a set of bottom-underlined buttons.
Keep those resting CTA rules optically lighter than the Hero folio hairline so
the navigation remains subordinate. Keep the Wave as the Hero's only
atmospheric element so the typography supplies the identity without another
decorative typeface or louder animation.

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
- Project and Portfolio-home Blog preview rows share their row spacing, media
  treatment, and interaction grammar; their column proportions and metadata
  order may differ to express artifact versus publication.
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
on the side-by-side layout, the complete series-to-action copy stack must not
exceed the cover height. If text still needs more room at the minimum readable
size, preserve the full text instead of hiding it. Give Korean feature titles
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

The production portfolio hero is the established Wave and must remain the only
home-page hero implementation. Its canonical surface is an asymmetric
interference field: domain-warped silk folds cross a restrained radial
diffraction pattern, with sparse cyan-white lustre traveling along the crests.
Keep the parallel base strands subordinate so the moving highlights, rather
than a generic orthogonal grid, describe the surface. Preserve the dark ink
stage, cyan-neutral palette, calm ambient pace, CTA timing, and static fallback.
`js/hero.js` loads the Wave directly; URL parameters, article demos, and
experimental assets must not replace it. Technical visualizations such as the
coarse-to-fine voxel decoder belong to a dedicated article or demo route. Load
their scripts and binary assets only inside that opt-in context, with no
requests or handoff state added to the portfolio home page.

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
- Treat the Wave canvas as the spatial-motion exception: on desktop, pointer
  input may drive a damped camera parallax and surface tilt so the field reads
  as a 3D scene. Keep the combined edge response within roughly 10 degrees,
  preserve the authored ambient drift, and use a smaller envelope for touch.
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

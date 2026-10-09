const { render: renderSiteIcon } = require('../../assets/js/site-icons');
const { renderSiteHeader } = require('./site-header');
const cheerio = require('cheerio');

function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderInlineStrong(value) {
    return escapeHtml(value).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function renderProjectDetailItem(detail) {
    const label = escapeHtml(detail && detail.label);
    const links = Array.isArray(detail && detail.links)
        ? detail.links.filter((link) => link && link.value)
        : [];
    const value = escapeHtml(detail && detail.value);
    const url = detail && detail.url ? String(detail.url) : '';
    const externalAttrs = /^https?:\/\//i.test(url)
        ? ' target="_blank" rel="noopener noreferrer"'
        : '';
    const valueHtml = links.length
        ? `<span class="project-detail-links">${links.map((link) => {
            const linkUrl = link.url ? String(link.url) : '';
            const linkExternalAttrs = /^https?:\/\//i.test(linkUrl)
                ? ' target="_blank" rel="noopener noreferrer"'
                : '';
            return linkUrl
                ? `<a href="${escapeHtml(linkUrl)}"${linkExternalAttrs}>${escapeHtml(link.value)}</a>`
                : `<span>${escapeHtml(link.value)}</span>`;
        }).join('')}</span>`
        : url
            ? `<a href="${escapeHtml(url)}"${externalAttrs}>${value}</a>`
            : value;

    return `              <li><strong>${label}</strong>${valueHtml}</li>`;
}

function renderMathRuntime(contentHtml) {
    if (!/(\$\$|\\\(|\\\[|(?:^|[^\\])\$[^$\n]+\$)/m.test(String(contentHtml || ''))) {
        return '';
    }

    return `  <script src="../../js/mathjax-config.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>`;
}

function renderProjectPage({ project, contentHtml, projectNav = null }) {
    const title = project.title || 'Project';
    const heroTitle = project.heroTitle || title;
    const colon = heroTitle.indexOf(':');
    const heroHtml = colon >= 0 ? `<span>${escapeHtml(heroTitle.slice(0,colon+1))}</span>${escapeHtml(heroTitle.slice(colon+1).trim())}` : escapeHtml(heroTitle);
    const $ = cheerio.load(contentHtml, null, false);
    const entries = [['overview','Overview']];
    const usedIds = new Set(['overview','contributions']);
    if (project.contributions?.length) entries.push(['contributions','Core contributions']);
    $('h2').each((_, node) => {
        const heading = $(node), label = heading.text();
        const base = heading.attr('id') || label.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'') || 'section';
        let id = base, suffix = 2;
        while (usedIds.has(id)) id = `${base}-${suffix++}`;
        usedIds.add(id); heading.attr('id',id); entries.push([id,label]);
    });
    // Keep every authored node, including MathJax source, figures and code.
    const article = $.html();
    const media = project.overviewMedia;
    const mediaHtml = media?.src ? `<figure class="case-visual project-overview-media">
        <video muted loop playsinline preload="none" poster="${escapeHtml(media.poster || '')}" aria-label="${escapeHtml(media.ariaLabel || 'Project showcase')}">
            <source src="${escapeHtml(media.src)}" type="${escapeHtml(media.mimeType || 'video/mp4')}">
        </video>
        <div class="case-caption"><figcaption>${escapeHtml(media.caption || '')} ${media.captionUrl ? `<a href="${escapeHtml(media.captionUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(media.captionLinkLabel || 'Source')}</a>` : ''}</figcaption><button class="case-media-toggle" type="button">Play</button></div>
    </figure>` : '';
    const pager = projectNav?.previous && projectNav?.next ? `<nav class="edition-next project-page-nav" aria-label="Previous and next projects">
        <a class="project-page-nav-prev" href="../${escapeHtml(projectNav.previous.slug)}/" rel="prev"><span>← Previous project</span><strong>${escapeHtml(projectNav.previous.label)}</strong></a>
        <a class="project-page-nav-next" href="../${escapeHtml(projectNav.next.slug)}/" rel="next"><span>Next project →</span><strong>${escapeHtml(projectNav.next.label)}</strong></a>
    </nav>` : '';
    return `<!DOCTYPE html>
<html lang="en"><head>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>${escapeHtml(title)}</title><meta name="description" content="${escapeHtml(project.description || '')}"><meta name="keywords" content="${escapeHtml(project.keywords || '')}">
    <link href="/assets/favicon.ico" rel="icon">
    <script src="/js/site-theme.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="/assets/css/site-theme.css" rel="stylesheet">
    <link href="/assets/css/portfolio-chapters.css" rel="stylesheet">
    <link href="/assets/css/project-detail.css" rel="stylesheet">
    <link href="/assets/css/site-icons.css" rel="stylesheet">
    <script src="/assets/js/site-icons.js"></script>
${renderMathRuntime(contentHtml)}
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-RF7ETSKPK9"></script>
    <script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments)}gtag('js',new Date());gtag('config','G-RF7ETSKPK9');</script>
</head><body class="edition-site case-study portfolio-details-page" data-surface="reading" data-chapter="detail">
    <a class="case-skip" href="#overview">Skip to project</a>
    <div class="case-atmosphere" aria-hidden="true"></div>
${renderSiteHeader()}
    <main id="main" class="portfolio-details"><div class="case-shell">
        <header class="case-opening project-hero-header">
            <a class="case-breadcrumb" href="/#portfolio"><span aria-hidden="true">←</span> Projects</a>
            <p class="case-kicker">01 / PROJECT CASE STUDY</p>
            <h1>${heroHtml}</h1>
            <p class="case-byline">${(project.subtitles || []).map(value => `<span>${escapeHtml(value)}</span>`).join('')}</p>
        </header>
        <div class="case-layout">
            <section class="case-overview project-overview" aria-labelledby="overview"><h2 id="overview">Project Overview</h2>
                ${(project.overview || []).map(value => `<p>${escapeHtml(value)}</p>`).join('')}
            </section>
            <aside class="case-meta" aria-label="Project details">
                <div class="portfolio-info"><h2>Project Details</h2><ul>${(project.details || []).map(renderProjectDetailItem).join('')}</ul></div>
                <nav class="case-contents" aria-label="On this page"><h2>In this case study</h2><ol>${entries.map(([id,label],index) => `<li><a href="#${id}"><span aria-hidden="true">${String(index+1).padStart(2,'0')}</span>${escapeHtml(label)}</a></li>`).join('')}</ol></nav>
                ${projectNav?.items ? `<details class="case-project-switch"><summary>Other projects</summary>${projectNav.items.filter(item => item.slug !== projectNav.currentSlug).map(item => `<a href="../${escapeHtml(item.slug)}/">${escapeHtml(item.label)}</a>`).join('')}</details>` : ''}
            </aside>
            ${mediaHtml}
            <div class="case-body">
                ${project.contributions?.length ? `<section class="case-contributions project-contributions" aria-labelledby="contributions"><h2 id="contributions">Core contributions</h2><ul>${project.contributions.map(value => `<li>${renderInlineStrong(value)}</li>`).join('')}</ul></section>` : ''}
                <article class="case-article project-case-study-article">${article}</article>
            </div>
        </div>
        <div class="case-ending">${pager}</div>
    </div></main>
    <footer class="case-footer"><div class="case-shell"><p>© Hwan Heo. All Rights Reserved.</p><a href="/#portfolio">Back to Projects ↑</a></div></footer>
    <script src="/js/project-detail.js"></script>
</body></html>`;
}

module.exports = { renderProjectPage };

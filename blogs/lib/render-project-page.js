function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderProjectSidebarNav(projectNav) {
    if (!projectNav || !Array.isArray(projectNav.items) || projectNav.items.length < 2) {
        return '';
    }

    const currentItem = projectNav.items.find((item) => item.slug === projectNav.currentSlug);
    if (!currentItem) {
        return '';
    }

    const items = projectNav.items.filter((item) => item.slug !== projectNav.currentSlug).map((item, index) => {
        return `          <a href="../${escapeHtml(item.slug)}/"><span class="project-selector-index">${String(index + 1).padStart(2, '0')}</span><span>${escapeHtml(item.label)}</span></a>`;
    }).join('\n');

    return `        <li class="project-nav-selector">
          <details>
            <summary aria-current="page">
              <i class="bi bi-grid navicon" aria-hidden="true"></i>
              <span class="project-selector-copy"><small>Switch project</small><strong>${escapeHtml(currentItem.label)}</strong></span>
              <i class="bi bi-chevron-down project-selector-toggle" aria-hidden="true"></i>
            </summary>
            <div class="project-selector-options">
${items}
            </div>
          </details>
        </li>`;
}

function renderProjectNavItems(projectNav) {
    const projectSidebarNav = renderProjectSidebarNav(projectNav);
    return `        <li><a href="../../#home"><i class="bi bi-house navicon"></i>Home</a></li>
        <li><a href="../../#portfolio" class="active"><i class="bi bi-images navicon"></i> Project</a></li>
${projectSidebarNav}
        <li><a href="../../blogs/"><i class="bi bi-keyboard navicon"></i> Blog</a></li>
        <li><a href="../../#about"><i class="bi bi-person navicon"></i> About</a></li>
        <li><a href="../../#resume"><i class="bi bi-file-earmark-text navicon"></i> Resume</a></li>`;
}

function renderProjectPager(projectNav) {
    if (!projectNav || !projectNav.previous || !projectNav.next) {
        return '';
    }

    return `<div class="container project-page-nav">
        <a class="project-page-nav-link project-page-nav-prev" href="../${escapeHtml(projectNav.previous.slug)}/">
          <span class="project-page-nav-kicker"><i class="bi bi-arrow-left"></i> Previous Project</span>
          <strong>${escapeHtml(projectNav.previous.label)}</strong>
        </a>
        <a class="project-page-nav-link project-page-nav-next" href="../${escapeHtml(projectNav.next.slug)}/">
          <span class="project-page-nav-kicker">Next Project <i class="bi bi-arrow-right"></i></span>
          <strong>${escapeHtml(projectNav.next.label)}</strong>
        </a>
      </div>`;
}

function normalizeProjectInfoLabels(contentHtml) {
    return String(contentHtml || '').replace(
        /(<div class="portfolio-info">[\s\S]*?<\/div>)/g,
        (block) => block.replace(/(<strong>[^<]+<\/strong>):\s*/g, '$1')
    );
}

function renderProjectHero(project) {
    const title = project.title || 'Project';
    const heroTitle = project.heroTitle || title;
    const subtitles = Array.isArray(project.subtitles) ? project.subtitles : [];

    return `      <div class="row gx-5 justify-content-center">
        <div class="project-hero-header text-center col-11 col-lg-10 col-xl-8 col-xxl-7">
          <span class="project-hero-kicker">Project Case Study</span>
          <h1 class="display-6 fw-bolder mb-0"><span class="text-gradient d-inline">${heroTitle}</span></h1>
          ${subtitles.length ? `<div class="project-hero-meta">${subtitles.map((subtitle) => `<span>${subtitle}</span>`).join('')}</div>` : ''}
        </div>
      </div>`;
}

function renderProjectDetailItem(detail) {
    const label = escapeHtml(detail && detail.label);
    const value = escapeHtml(detail && detail.value);
    const url = detail && detail.url ? String(detail.url) : '';
    const externalAttrs = /^https?:\/\//i.test(url) ? ' target="_blank" rel="noopener noreferrer"' : '';
    const valueHtml = url
        ? `<a href="${escapeHtml(url)}"${externalAttrs}>${value}</a>`
        : value;

    return `              <li><strong>${label}</strong>${valueHtml}</li>`;
}

function renderCaseStudyDetailsInner(project, contentHtml, projectNav = null) {
    const overview = Array.isArray(project.overview) ? project.overview.filter(Boolean) : [];
    const contributions = Array.isArray(project.contributions) ? project.contributions.filter(Boolean) : [];
    const details = Array.isArray(project.details) ? project.details.filter(Boolean) : [];
    const pagerHtml = renderProjectPager(projectNav);

    return `${renderProjectHero(project)}

      <div class="container portfolio-details-container col-11 project-case-study-overview">
        <div class="row gy-4">
          <div class="col-lg-8">
            <div class="portfolio-description project-overview">
              <h2>Project Overview</h2>
              ${overview.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join('\n              ')}
              ${contributions.length ? `<div class="project-contributions">
                <p>Core contributions</p>
                <ul>
${contributions.map((contribution) => `                  <li>${escapeHtml(contribution)}</li>`).join('\n')}
                </ul>
              </div>` : ''}
            </div>
          </div>
          <div class="col-lg-4">
            <aside class="portfolio-info">
              <h3>Project Details</h3>
              <ul>
${details.map(renderProjectDetailItem).join('\n')}
              </ul>
            </aside>
          </div>
        </div>
      </div>

      <div class="container col-11 project-case-study-shell">
        <article class="portfolio-description project-case-study-article">
${contentHtml}
        </article>
      </div>
${pagerHtml}`;
}

function renderProjectDetailsInner(project, contentHtml, projectNav = null) {
    if (project.layout === 'case-study') {
        return renderCaseStudyDetailsInner(project, contentHtml, projectNav);
    }

    const normalizedContentHtml = normalizeProjectInfoLabels(contentHtml);
    const pagerHtml = renderProjectPager(projectNav);

    return `${renderProjectHero(project)}

${normalizedContentHtml}
${pagerHtml}`;
}

function getCommonProjectStyle() {
    return `<style id="project-detail-common-style">
    .portfolio-details-page .header {
      scrollbar-width: none;
    }
    .portfolio-details-page .header::-webkit-scrollbar {
      display: none;
    }
    .navmenu .project-nav-selector {
      margin: 0.2rem 0 0.85rem;
    }
    .navmenu .project-nav-selector details {
      color: var(--nav-color);
      position: relative;
    }
    .navmenu .project-nav-selector summary {
      align-items: center;
      background: rgba(255, 255, 255, 0.055);
      border: 1px solid rgba(255, 255, 255, 0.09);
      border-radius: 12px;
      color: var(--nav-hover-color);
      cursor: pointer;
      display: flex;
      font-family: var(--nav-font);
      gap: 0.6rem;
      list-style: none;
      margin: 0;
      min-height: 58px;
      padding: 0.62rem 0.7rem;
      transition: background-color 0.2s ease, border-color 0.2s ease;
    }
    .navmenu .project-nav-selector summary:hover {
      background: rgba(255, 255, 255, 0.085);
      border-color: rgba(20, 157, 221, 0.36);
    }
    .navmenu .project-nav-selector summary::-webkit-details-marker {
      display: none;
    }
    .navmenu .project-nav-selector summary .navicon {
      color: var(--accent-color);
      flex: 0 0 auto;
      font-size: 1rem;
      margin: 0;
    }
    .navmenu .project-selector-copy {
      display: flex;
      flex: 1 1 auto;
      flex-direction: column;
      gap: 0.13rem;
      min-width: 0;
    }
    .navmenu .project-selector-copy small {
      color: color-mix(in srgb, var(--nav-color), transparent 28%);
      font-family: var(--mono-font);
      font-size: 0.59rem;
      font-weight: 500;
      letter-spacing: 0.07em;
      line-height: 1.2;
      text-transform: uppercase;
    }
    .navmenu .project-selector-copy strong {
      color: var(--nav-hover-color);
      font-size: 0.78rem;
      font-weight: 650;
      line-height: 1.28;
    }
    .navmenu .project-selector-copy strong,
    .navmenu .project-selector-options a span {
      display: -webkit-box;
      overflow: hidden;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
    }
    .navmenu .project-nav-selector .project-selector-toggle {
      color: color-mix(in srgb, var(--nav-color), transparent 20%);
      flex: 0 0 auto;
      font-size: 0.68rem;
      transition: transform 0.2s ease, color 0.2s ease;
    }
    .navmenu .project-nav-selector details[open] .project-selector-toggle {
      color: var(--accent-color);
      transform: rotate(180deg);
    }
    .navmenu .project-selector-options {
      background: rgba(255, 255, 255, 0.035);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 12px;
      display: grid;
      gap: 0.08rem;
      margin-top: 0.38rem;
      overflow-x: hidden;
      padding: 0.35rem;
    }
    .navmenu .project-selector-options a,
    .navmenu .project-selector-options a:focus {
      align-items: center;
      border-radius: 9px;
      color: color-mix(in srgb, var(--nav-color), transparent 8%);
      display: flex;
      font-family: var(--nav-font);
      font-size: 0.75rem;
      gap: 0.55rem;
      line-height: 1.3;
      padding: 0.48rem 0.52rem;
      white-space: normal;
      transition: color 0.2s ease, background-color 0.2s ease;
    }
    .navmenu .project-selector-options a span:not(.project-selector-index) {
      display: block;
      overflow: visible;
      overflow-wrap: anywhere;
      -webkit-box-orient: initial;
      -webkit-line-clamp: initial;
    }
    .navmenu .project-selector-options a:hover {
      background: rgba(255, 255, 255, 0.05);
      color: var(--nav-hover-color);
    }
    .navmenu .project-selector-index {
      color: var(--accent-color);
      display: block !important;
      flex: 0 0 1.5rem;
      font-family: var(--mono-font);
      font-size: 0.61rem;
      font-weight: 500;
      letter-spacing: 0.04em;
      overflow: visible !important;
    }
    .portfolio-details .project-hero-header {
      margin-bottom: 3.4rem;
      max-width: 900px;
    }
    .portfolio-details .project-hero-kicker {
      color: #149ddd;
      display: inline-block;
      font-family: var(--mono-font);
      font-size: 0.67rem;
      font-weight: 500;
      letter-spacing: 0.11em;
      margin-bottom: 0.8rem;
      text-transform: uppercase;
    }
    .portfolio-details .project-hero-header h1 {
      font-size: clamp(2rem, 4.1vw, 3.15rem);
      letter-spacing: -0.045em;
      line-height: 1.08;
      overflow-wrap: anywhere;
      text-wrap: balance;
    }
    .portfolio-details .project-hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      justify-content: center;
      margin-top: 1.15rem;
    }
    .portfolio-details .project-hero-meta span {
      background: #f8fafc;
      border: 1px solid rgba(15, 23, 42, 0.09);
      border-radius: 999px;
      color: #64748b;
      font-size: 0.72rem;
      font-weight: 650;
      line-height: 1.2;
      padding: 0.34rem 0.62rem;
    }
    .portfolio-details .portfolio-description h2 {
      margin-top: 1.25rem;
      margin-bottom: 0.85rem;
      letter-spacing: -0.02em;
    }
    .portfolio-details .portfolio-description h3,
    .portfolio-details .portfolio-description h4,
    .portfolio-details .portfolio-description h5 {
      margin-top: 1.15rem;
      margin-bottom: 0.65rem;
    }
    .portfolio-details .portfolio-description p,
    .portfolio-details .portfolio-description li {
      line-height: 1.72;
    }
    .portfolio-details .portfolio-details-container {
      max-width: 1040px;
    }
    .portfolio-details .container.col-11:not(.portfolio-details-container) {
      max-width: 960px;
    }
    .portfolio-details .project-readable {
      margin-left: auto;
      margin-right: auto;
      max-width: 760px;
      width: min(100%, 760px);
    }
    .portfolio-details .project-overview > p:first-of-type {
      color: #334155;
      font-size: 1.04rem;
      line-height: 1.72;
    }
    .portfolio-details .project-case-study-overview {
      margin-bottom: 3.8rem;
    }
    .portfolio-details .project-case-study-shell {
      max-width: 900px;
    }
    .portfolio-details .project-case-study-article {
      margin: 0 auto;
      max-width: 820px;
    }
    .portfolio-details .project-case-study-article > h2 {
      border-top: 1px solid rgba(15, 23, 42, 0.09);
      font-size: clamp(1.45rem, 2.4vw, 1.8rem);
      margin: 3.8rem 0 1rem;
      padding-top: 2.7rem;
    }
    .portfolio-details .project-case-study-article > h2:first-child {
      border-top: 0;
      margin-top: 0;
      padding-top: 0;
    }
    .portfolio-details .project-case-study-article h3 {
      color: #172033;
      font-size: 1.08rem;
      font-weight: 760;
      margin: 2.15rem 0 0.7rem;
    }
    .portfolio-details .project-case-study-article h4 {
      color: #334155;
      font-size: 0.96rem;
      font-weight: 720;
      margin: 1.7rem 0 0.55rem;
    }
    .portfolio-details .project-case-study-article p,
    .portfolio-details .project-case-study-article li {
      color: #475569;
      font-size: 0.96rem;
      line-height: 1.74;
    }
    .portfolio-details .project-case-study-article blockquote {
      background: linear-gradient(135deg, rgba(20, 157, 221, 0.09), rgba(20, 157, 221, 0.025));
      border: 1px solid rgba(20, 157, 221, 0.2);
      border-radius: 16px;
      margin: 1.35rem 0 2.1rem;
      padding: 1rem 1.1rem;
    }
    .portfolio-details .project-case-study-article blockquote p {
      color: #334155;
      font-size: 0.94rem;
      line-height: 1.66;
      margin: 0;
    }
    .portfolio-details .project-case-study-article blockquote strong {
      color: #0878ad;
    }
    .portfolio-details .project-case-study-article > iframe {
      aspect-ratio: 16 / 9;
      border: 0;
      box-shadow: 0 16px 34px rgba(15, 23, 42, 0.12);
      display: block;
      height: auto;
      margin: 1.35rem 0;
      width: 100%;
    }
    .portfolio-details .project-case-study-article figure {
      margin: 1.4rem 0;
    }
    .portfolio-details .project-case-study-article figcaption {
      color: #7b8491;
      font-size: 0.76rem;
      line-height: 1.5;
      margin-top: 0.55rem;
      text-align: center;
    }
    .portfolio-details .project-case-study-article hr {
      border: 0;
      border-top: 1px solid rgba(15, 23, 42, 0.09);
      margin: 2.6rem 0;
      opacity: 1;
    }
    .portfolio-details .project-case-study-article mjx-container[display="true"] {
      margin: 1.35rem 0 !important;
      max-width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      padding: 0.45rem 0;
    }
    .portfolio-details .project-contributions {
      background: linear-gradient(145deg, #f8fafc, #ffffff);
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 16px;
      margin-top: 1.35rem;
      padding: 1rem 1.1rem 0.85rem;
    }
    .portfolio-details .project-contributions > p:first-child {
      color: #0f172a;
      font-size: 0.78rem;
      font-weight: 760;
      letter-spacing: 0.055em;
      margin-bottom: 0.55rem;
      text-transform: uppercase;
    }
    .portfolio-details .project-contributions ul {
      margin: 0;
      padding-left: 1.1rem;
    }
    .portfolio-details .project-contributions li {
      color: #526070;
      font-size: 0.9rem;
      line-height: 1.58;
      margin: 0.32rem 0;
    }
    .portfolio-details img,
    .portfolio-details video,
    .portfolio-details iframe {
      border-radius: 14px;
      height: auto;
      max-width: 100%;
    }
    .portfolio-details .project-video,
    .portfolio-details video {
      background: #000;
      box-shadow: 0 16px 34px rgba(15, 23, 42, 0.12);
      display: block;
      margin: 1rem auto;
      width: 100%;
    }
    .portfolio-details .text-center .project-video,
    .portfolio-details .text-center video {
      margin-left: auto;
      margin-right: auto;
    }
    .portfolio-details .video-container {
      margin: 1.25rem auto;
      max-width: 100%;
    }
    .portfolio-details .video-container:has(video) {
      background: transparent;
      height: auto;
      overflow: visible;
      padding-bottom: 0;
      position: relative;
    }
    .portfolio-details .video-container:has(video) video {
      height: auto;
      left: auto;
      position: static;
      top: auto;
      width: 100%;
    }
    .portfolio-details .video-container:has(iframe) {
      aspect-ratio: 16 / 9;
      background: #000;
      height: auto;
      overflow: hidden;
      padding-bottom: 0;
      position: relative;
    }
    .portfolio-details .video-container:has(iframe) iframe {
      height: 100%;
      left: 0;
      position: absolute;
      top: 0;
      width: 100%;
    }
    .portfolio-details p > img:only-child {
      display: block;
      margin: 1rem auto;
    }
    .portfolio-details .project-case-study-article .project-compact-result {
      display: block;
      margin: 1rem auto;
      width: 70%;
    }
    .portfolio-details img[width] {
      display: block;
      margin-left: auto;
      margin-right: auto;
    }
    .portfolio-details .text-center img,
    .portfolio-details .gif-container img {
      display: block;
      margin-left: auto;
      margin-right: auto;
    }
    .portfolio-details table img {
      border-radius: 10px;
      width: 100%;
    }
    .portfolio-details .portfolio-info {
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 18px;
      box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
    }
    .portfolio-details .portfolio-info h3 {
      font-size: 1.05rem;
      letter-spacing: 0;
      margin-bottom: 1rem;
    }
    .portfolio-details .portfolio-info ul {
      display: grid;
      gap: 0.78rem;
      list-style: none;
      margin: 0;
      padding: 0;
    }
    .portfolio-details .portfolio-info li {
      color: #475569;
      font-size: 0.84rem;
      line-height: 1.5;
    }
    .portfolio-details .portfolio-info li + li {
      margin-top: 0;
    }
    .portfolio-details .portfolio-info li strong {
      color: #0f172a;
      display: block;
      font-size: 0.84rem;
      font-weight: 760;
      letter-spacing: 0;
      line-height: 1.32;
      margin-bottom: 0.02rem;
      text-transform: none;
    }
    .portfolio-details .portfolio-info a {
      font-weight: 650;
    }
    .portfolio-details table {
      border-radius: 14px;
      overflow: hidden;
    }
    .portfolio-details .project-page-nav {
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 3rem;
      max-width: 960px;
    }
    .portfolio-details .project-page-nav-link {
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      border: 1px solid rgba(15, 23, 42, 0.09);
      border-radius: 18px;
      box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
      color: #1f2937;
      display: flex;
      flex-direction: column;
      min-height: 118px;
      padding: 1.05rem 1.15rem;
      text-decoration: none;
      transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }
    .portfolio-details .project-page-nav-link:hover {
      border-color: rgba(20, 157, 221, 0.45);
      box-shadow: 0 18px 42px rgba(15, 23, 42, 0.12);
      transform: translateY(-2px);
    }
    .portfolio-details .project-page-nav-next {
      align-items: flex-end;
      text-align: right;
    }
    .portfolio-details .project-page-nav-kicker {
      color: #64748b;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      margin-bottom: 0.45rem;
      text-transform: uppercase;
    }
    .portfolio-details .project-page-nav-link strong {
      font-size: 1.02rem;
      line-height: 1.35;
    }
    @media (max-width: 768px) {
      .portfolio-details.section {
        padding-top: 56px;
      }
      .portfolio-details .portfolio-details-container,
      .portfolio-details .container.col-11 {
        width: 100%;
      }
      .portfolio-details .portfolio-description,
      .portfolio-details .portfolio-info {
        margin-bottom: 1.25rem;
      }
      .portfolio-details .project-hero-header {
        margin-bottom: 2.4rem;
      }
      .portfolio-details .project-case-study-overview {
        margin-bottom: 2.6rem;
      }
      .portfolio-details .project-case-study-article > h2 {
        margin-top: 3rem;
        padding-top: 2.2rem;
      }
      .portfolio-details .project-hero-header h1 {
        font-size: clamp(1.8rem, 8.5vw, 2.45rem);
      }
      .portfolio-details .project-case-study-article .project-compact-result {
        width: 100%;
      }
      .portfolio-details .project-page-nav {
        grid-template-columns: 1fr;
      }
      .portfolio-details .project-page-nav-next {
        align-items: flex-start;
        text-align: left;
      }
    }
  </style>`;
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
    const description = project.description || '';
    const keywords = project.keywords || '';
    const detailsInner = renderProjectDetailsInner(project, contentHtml, projectNav);
    const mathRuntime = renderMathRuntime(contentHtml);

    return `<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="utf-8">
  <meta content="width=device-width, initial-scale=1.0" name="viewport">
  <title>${escapeHtml(title)}</title>
  <meta content="${escapeHtml(description)}" name="description">
  <meta content="${escapeHtml(keywords)}" name="keywords">

  <link href="../../assets/favicon.ico" rel="icon">
  <link href="../../assets/favicon.ico" rel="apple-touch-icon">

  <link href="https://fonts.googleapis.com" rel="preconnect">
  <link href="https://fonts.gstatic.com" rel="preconnect" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&family=Manrope:wght@500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700&display=swap" rel="stylesheet">

  <link href="../../assets/vendor/bootstrap/css/bootstrap.min.css" rel="stylesheet">
  <link href="../../assets/vendor/bootstrap-icons/bootstrap-icons.min.css" rel="stylesheet">

  <script src="../../js/sidebar-controller.js"></script>
  <link href="../../assets/css/portfolio.css" rel="stylesheet">
  <link href="/blogs/css/scroll-progress.css" rel="stylesheet">
${mathRuntime}

  <style>
    table { width: 100%; border-collapse: collapse; }
    th, td { width: 50%; border: 1px solid #ddd; padding: 10px; text-align: center; vertical-align: middle; }
    th { width: 15%; background-color: #f2f2f2; }
    .feature-img { max-width: 100%; height: auto; }
    .gif-container { display: flex; justify-content: center; align-items: center; }
    .gif-container img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
  </style>
  ${getCommonProjectStyle()}
  <link href="../../css/sidebar-nav.css" rel="stylesheet">

  <script async src="https://www.googletagmanager.com/gtag/js?id=G-RF7ETSKPK9"></script>
  <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-RF7ETSKPK9');
  </script>
</head>

<body class="portfolio-details-page">
  <header id="header" class="header dark-background d-flex flex-column">
    <div class="profile-img">
      <img src="../../assets/icon.webp" alt="Portrait illustration of Hwan Heo" class="img-fluid rounded-circle">
    </div>

    <a href="../../" class="logo d-flex align-items-center justify-content-center">
      <span class="sitename">Hwan Heo</span>
    </a>

    <div class="social-links text-center">
      <a href="https://github.com/hwanhuh" class="github"><i class="bi bi-github"></i></a>
      <a href="https://www.linkedin.com/in/hwan-heo-0905korea/" class="linkedin"><i class="bi bi-linkedin"></i></a>
      <a href="https://scholar.google.com/citations?user=RulvYTkAAAAJ" class="instagram"><i class="bi bi-mortarboard-fill"></i></a>
      <a href="mailto:hwan.heo.ai@gmail.com" class="google-plus"><i class="bi bi-envelope-fill"></i></a>
    </div>

    <nav id="navmenu" class="navmenu">
      <ul>
${renderProjectNavItems(projectNav)}
      </ul>
    </nav>
  </header>

  <main class="main">
    <div class="page-title dark-background">
      <div class="container d-lg-flex justify-content-between align-items-center">
        <nav class="breadcrumbs">
          <ol>
            <li><a href="../../">Home</a></li>
            <li class="current">${escapeHtml(title)}</li>
          </ol>
        </nav>
      </div>
    </div>

    <section id="portfolio-details" class="portfolio-details section">
${detailsInner}
    </section>
  </main>

  <footer id="footer" class="footer position-relative light-background">
    <div class="container">
      <div class="copyright text-center ">
        <p>© <span>Copyright</span> <strong class="px-1 sitename">Hwan Heo</strong> <span>All Rights Reserved</span></p>
      </div>
    </div>
  </footer>

  <button id="scroll-top" class="scroll-top project-scroll-top d-flex align-items-center justify-content-center" type="button" aria-label="Back to top">
    <i class="bi bi-arrow-up" aria-hidden="true"></i>
  </button>
  <div id="preloader"></div>

  <script src="../../assets/vendor/bootstrap/js/bootstrap.bundle.min.js"></script>
  <script src="../../assets/js/main.js"></script>
  <script src="/blogs/js/scroll-progress.js"></script>
</body>

</html>
`;
}

module.exports = {
    renderProjectPage
};

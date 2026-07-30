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

    const items = projectNav.items
        .filter((item) => item.slug !== projectNav.currentSlug)
        .map((item, index) => {
            return `          <a href="../${escapeHtml(item.slug)}/"><span class="project-selector-index">${String(index + 1).padStart(2, '0')}</span><span>${escapeHtml(item.label)}</span></a>`;
        })
        .join('\n');

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

    return `      <div class="container project-hero-shell">
        <div class="project-hero-header">
          <span class="project-hero-kicker">Project Case Study</span>
          <h1 class="display-6 fw-bolder mb-0"><span class="text-gradient d-inline">${heroTitle}</span></h1>
          ${subtitles.length ? `<div class="project-hero-meta">${subtitles.map((subtitle) => `<span>${escapeHtml(subtitle)}</span>`).join('')}</div>` : ''}
        </div>
      </div>`;
}

function renderProjectDetailItem(detail) {
    const label = escapeHtml(detail && detail.label);
    const value = escapeHtml(detail && detail.value);
    const url = detail && detail.url ? String(detail.url) : '';
    const externalAttrs = /^https?:\/\//i.test(url)
        ? ' target="_blank" rel="noopener noreferrer"'
        : '';
    const valueHtml = url
        ? `<a href="${escapeHtml(url)}"${externalAttrs}>${value}</a>`
        : value;

    return `              <li><strong>${label}</strong>${valueHtml}</li>`;
}

function renderCaseStudyDetailsInner(project, contentHtml, projectNav = null) {
    const overview = Array.isArray(project.overview) ? project.overview.filter(Boolean) : [];
    const contributions = Array.isArray(project.contributions)
        ? project.contributions.filter(Boolean)
        : [];
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
  <link href="../../assets/css/project-detail.css" rel="stylesheet">
  <link href="/blogs/css/scroll-progress.css" rel="stylesheet">
${mathRuntime}
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
      <a href="https://github.com/hwanhuh" class="github" aria-label="GitHub"><i class="bi bi-github" aria-hidden="true"></i></a>
      <a href="https://www.linkedin.com/in/hwan-heo-0905korea/" class="linkedin" aria-label="LinkedIn"><i class="bi bi-linkedin" aria-hidden="true"></i></a>
      <a href="https://scholar.google.com/citations?user=RulvYTkAAAAJ" class="instagram" aria-label="Google Scholar"><i class="bi bi-mortarboard-fill" aria-hidden="true"></i></a>
      <a href="mailto:hwan.heo.ai@gmail.com" class="google-plus" aria-label="Email"><i class="bi bi-envelope-fill" aria-hidden="true"></i></a>
    </div>

    <nav id="navmenu" class="navmenu">
      <ul>
${renderProjectNavItems(projectNav)}
      </ul>
    </nav>
  </header>

  <main class="main">
    <div class="page-title dark-background">
      <div class="container">
        <nav class="breadcrumbs" aria-label="Breadcrumb">
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
      <div class="copyright text-center">
        <p>© <span>Copyright</span> <strong class="px-1 sitename">Hwan Heo</strong> <span>All Rights Reserved</span></p>
      </div>
    </div>
  </footer>

  <button id="scroll-top" class="scroll-top project-scroll-top d-flex align-items-center justify-content-center" type="button" aria-label="Back to top">
    <i class="bi bi-arrow-up" aria-hidden="true"></i>
  </button>

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

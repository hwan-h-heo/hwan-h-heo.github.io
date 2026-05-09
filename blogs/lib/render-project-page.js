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

    const items = projectNav.items.filter((item) => item.slug !== projectNav.currentSlug).map((item) => {
        return `          <a href="../${escapeHtml(item.slug)}/"><i class="bi bi-arrow-return-right navicon"></i><span>${escapeHtml(item.label)}</span></a>`;
    }).join('\n');

    return `        <li class="project-nav-selector">
          <details>
            <summary aria-current="page"><i class="bi bi-dot navicon"></i><span>${escapeHtml(currentItem.label)}</span><i class="bi bi-chevron-down project-selector-toggle"></i></summary>
            <div class="project-selector-options">
${items}
            </div>
          </details>
        </li>`;
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

function renderProjectDetailsInner(project, contentHtml, projectNav = null) {
    const title = project.title || 'Project';
    const heroTitle = project.heroTitle || title;
    const subtitles = Array.isArray(project.subtitles) ? project.subtitles : [];
    const pagerHtml = renderProjectPager(projectNav);

    return `      <div class="row gx-5 justify-content-center">
        <div class="project-hero-header text-center mb-5 col-11 col-lg-10 col-xl-8 col-xxl-7">
          <h1 class="display-6 fw-bolder mb-0"><span class="text-gradient d-inline">${heroTitle}</span></h1>
          ${subtitles.map((subtitle) => `<div class="fs-3 fw-light text-muted">${subtitle}</div>`).join('\n          ')}
        </div>
      </div>

${contentHtml}
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
      margin: -0.35rem 0 0.4rem;
    }
    .navmenu .project-nav-selector details {
      color: var(--nav-color);
    }
    .navmenu .project-nav-selector summary {
      align-items: center;
      background: rgba(255, 255, 255, 0.06);
      border-left: 2px solid var(--accent-color);
      border-radius: 0 10px 10px 0;
      color: var(--nav-hover-color);
      cursor: pointer;
      display: flex;
      font-family: var(--nav-font);
      font-size: 0.86rem;
      gap: 0.35rem;
      line-height: 1.28;
      list-style: none;
      margin: 0 0.35rem 0 1.25rem;
      padding: 0.5rem 0.55rem 0.5rem 0.75rem;
      transition: background-color 0.2s ease, color 0.2s ease;
    }
    .navmenu .project-nav-selector summary::-webkit-details-marker {
      display: none;
    }
    .navmenu .project-nav-selector summary .navicon,
    .navmenu .project-nav-selector .project-selector-options .navicon {
      color: var(--accent-color);
      flex: 0 0 auto;
      font-size: 0.95rem;
      margin-right: 0.05rem;
    }
    .navmenu .project-nav-selector summary span,
    .navmenu .project-selector-options a span {
      display: -webkit-box;
      overflow: hidden;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
    }
    .navmenu .project-nav-selector .project-selector-toggle {
      color: color-mix(in srgb, var(--nav-color), transparent 20%);
      flex: 0 0 auto;
      font-size: 0.75rem;
      margin-left: auto;
      transition: transform 0.2s ease, color 0.2s ease;
    }
    .navmenu .project-nav-selector details[open] .project-selector-toggle {
      color: var(--accent-color);
      transform: rotate(180deg);
    }
    .navmenu .project-selector-options {
      border-left: 1px solid color-mix(in srgb, var(--nav-color), transparent 82%);
      margin: 0.35rem 0.35rem 0.1rem 2.15rem;
      padding: 0.05rem 0 0.1rem 0.35rem;
    }
    .navmenu .project-selector-options a,
    .navmenu .project-selector-options a:focus {
      align-items: flex-start;
      border-radius: 8px;
      color: color-mix(in srgb, var(--nav-color), transparent 8%);
      display: flex;
      font-family: var(--nav-font);
      font-size: 0.8rem;
      gap: 0.25rem;
      line-height: 1.25;
      padding: 0.38rem 0.45rem;
      transition: color 0.2s ease, background-color 0.2s ease;
    }
    .navmenu .project-selector-options a:hover {
      background: rgba(255, 255, 255, 0.05);
      color: var(--nav-hover-color);
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
    .portfolio-details .project-hero-header {
      max-width: 880px;
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
    .portfolio-details .varco-object-embed {
      background: #f8fafc;
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 16px;
      box-shadow: 0 16px 34px rgba(15, 23, 42, 0.1);
      margin: 1.5rem auto 0.65rem;
      overflow: hidden;
      width: 100%;
    }
    .portfolio-details .varco-object-embed iframe {
      border: 0;
      display: block;
      height: min(500px, 70vh);
      width: 100%;
    }
    .portfolio-details p > img:only-child {
      display: block;
      margin: 1rem auto;
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
    .portfolio-details table .varco-lattice-img {
      aspect-ratio: 1 / 1;
      display: block;
      height: clamp(140px, 18vw, 230px);
      margin: 0 auto;
      object-fit: contain;
      padding: 0.35rem;
      width: 100%;
    }
    .portfolio-details table:has(.varco-lattice-img) th,
    .portfolio-details table:has(.varco-lattice-img) td {
      text-align: center;
      vertical-align: middle;
      width: 33.333%;
    }
    .portfolio-details .viewer-feature-group {
      margin: 1.4rem 0 1.8rem;
    }
    .portfolio-details .viewer-feature-group > h4 {
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      margin-bottom: 0.95rem;
    }
    .portfolio-details .viewer-feature-grid {
      display: grid;
      gap: 1rem;
      width: 100%;
    }
    .portfolio-details .viewer-feature-grid-2 {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .portfolio-details .viewer-feature-grid-3 {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .portfolio-details .viewer-feature-card {
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 18px;
      box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
      padding: 1rem;
    }
    .portfolio-details .viewer-feature-card .feature-img {
      aspect-ratio: 16 / 9;
      border-radius: 12px;
      object-fit: cover;
      width: 100%;
    }
    .portfolio-details .viewer-feature-card h5 {
      font-size: 1.02rem;
      font-weight: 700;
      margin: 0.35rem 0 0.45rem;
    }
    .portfolio-details .viewer-feature-card p,
    .portfolio-details .viewer-feature-card li {
      font-size: 0.94rem;
      line-height: 1.62;
    }
    .portfolio-details .portfolio-info {
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 18px;
      box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
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
      .portfolio-details .viewer-feature-grid-2,
      .portfolio-details .viewer-feature-grid-3 {
        grid-template-columns: 1fr;
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

function injectCommonProjectStyle(html) {
    if (html.includes('id="project-detail-common-style"')) {
        return html;
    }

    return html.replace('</head>', `  ${getCommonProjectStyle()}\n</head>`);
}

function replaceOrInsertMeta(html, name, content) {
    const escapedContent = escapeHtml(content || '');
    const pattern = new RegExp(`<meta\\s+content=["'][^"']*["']\\s+name=["']${name}["']>`, 'i');
    if (pattern.test(html)) {
        return html.replace(pattern, `<meta content="${escapedContent}" name="${name}">`);
    }

    return html.replace('</head>', `  <meta content="${escapedContent}" name="${name}">\n</head>`);
}

function patchLegacyLazyLoadingScript(html) {
    return html.replace(
        /lazyImage\.src\s*=\s*lazyImage\.dataset\.src;/g,
        `if (lazyImage.dataset.src) {
            lazyImage.src = lazyImage.dataset.src;
          }`
    );
}

function injectProjectSidebarNav(html, projectNav) {
    const sidebarHtml = renderProjectSidebarNav(projectNav);
    if (!sidebarHtml || html.includes('class="project-nav-selector"')) {
        return html;
    }

    const portfolioItemPattern = /(<li><a\s+href=["']\.\.\/\.\.\/#portfolio["'][^>]*>[\s\S]*?<\/a><\/li>)/i;
    if (portfolioItemPattern.test(html)) {
        return html.replace(portfolioItemPattern, `$1\n${sidebarHtml}`);
    }

    const navPattern = /(<nav\s+id=["']navmenu["'][^>]*>\s*<ul>)([\s\S]*?)(<\/ul>\s*<\/nav>)/i;
    return html.replace(navPattern, (match, openTag, items, closeTag) => `${openTag}${items}\n${sidebarHtml}\n      ${closeTag}`);
}

function renderProjectPageFromLegacyTemplate({ project, contentHtml, legacyHtml, projectNav }) {
    const title = project.title || 'Project';
    const detailsInner = renderProjectDetailsInner(project, contentHtml, projectNav);
    let html = legacyHtml;

    html = html.replace(/<title>[\s\S]*?<\/title>/i, `<title>${escapeHtml(title)}</title>`);
    html = replaceOrInsertMeta(html, 'description', project.description || '');
    html = replaceOrInsertMeta(html, 'keywords', project.keywords || '');
    html = injectCommonProjectStyle(html);
    html = injectProjectSidebarNav(html, projectNav);
    html = patchLegacyLazyLoadingScript(html);
    html = html.replace(/<li class=["']current["']>[\s\S]*?<\/li>/i, `<li class="current">${escapeHtml(title)}</li>`);

    const sectionPattern = /(<section\s+id=["']portfolio-details["'][^>]*>)([\s\S]*?)(<\/section>\s*<!--\s*\/Portfolio Details Section\s*-->)/i;
    if (sectionPattern.test(html)) {
        return html.replace(sectionPattern, (match, openTag, oldContent, closeTag) => `${openTag}\n${detailsInner}\n    ${closeTag}`);
    }

    return html;
}

function renderProjectPage({ project, contentHtml, legacyHtml, projectNav = null }) {
    if (legacyHtml) {
        return renderProjectPageFromLegacyTemplate({ project, contentHtml, legacyHtml, projectNav });
    }

    const title = project.title || 'Project';
    const description = project.description || '';
    const keywords = project.keywords || '';
    const detailsInner = renderProjectDetailsInner(project, contentHtml, projectNav);
    const projectSidebarNav = renderProjectSidebarNav(projectNav);

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
  <link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Raleway:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" rel="stylesheet">

  <link href="../../assets/vendor/bootstrap/css/bootstrap.min.css" rel="stylesheet">
  <link href="../../assets/vendor/bootstrap-icons/bootstrap-icons.css" rel="stylesheet">
  <link href="../../assets/vendor/aos/aos.css" rel="stylesheet">
  <link href="../../assets/vendor/glightbox/css/glightbox.min.css" rel="stylesheet">
  <link href="../../assets/vendor/swiper/swiper-bundle.min.css" rel="stylesheet">

  <link href="../../assets/css/used.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1.9.1/css/academicons.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

  <style>
    table { width: 100%; border-collapse: collapse; }
    th, td { width: 50%; border: 1px solid #ddd; padding: 10px; text-align: center; vertical-align: middle; }
    th { width: 15%; background-color: #f2f2f2; }
    .progress-bar { height: 0.4rem; background: #6EA8FE; width: 0%; z-index: 9999; position: fixed; }
    .feature-img { max-width: 100%; height: auto; }
    .gif-container { display: flex; justify-content: center; align-items: center; }
    .gif-container img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
  </style>
  ${getCommonProjectStyle()}

  <script async src="https://www.googletagmanager.com/gtag/js?id=G-RF7ETSKPK9"></script>
  <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-RF7ETSKPK9');
  </script>
</head>

<body class="portfolio-details-page">
  <div class="progress-container">
    <div class="progress-bar" id="myBar"></div>
  </div>

  <header id="header" class="header dark-background d-flex flex-column">
    <i class="header-toggle d-xl-none bi bi-list"></i>

    <div class="profile-img">
      <img src="../../assets/icon.webp" alt="Hwan Heo icon" class="img-fluid rounded-circle">
    </div>

    <a href="../../" class="logo d-flex align-items-center justify-content-center">
      <h1 class="sitename">Hwan Heo</h1>
    </a>

    <div class="social-links text-center">
      <a href="https://github.com/hwanhuh" class="github"><i class="bi bi-github"></i></a>
      <a href="https://www.linkedin.com/in/hwan-heo-0905korea/" class="linkedin"><i class="bi bi-linkedin"></i></a>
      <a href="https://scholar.google.com/citations?user=RulvYTkAAAAJ" class="instagram"><i class="ai ai-google-scholar"></i></a>
      <a href="mailto:gjghks950@naver.com" class="google-plus"><i class="bi bi-envelope-fill"></i></a>
    </div>

    <nav id="navmenu" class="navmenu">
      <ul>
        <li><a href="../../#home"><i class="bi bi-house navicon"></i>Home</a></li>
        <li><a href="../../#about"><i class="bi bi-person navicon"></i> About</a></li>
        <li><a href="../../#resume"><i class="bi bi-file-earmark-text navicon"></i> Resume</a></li>
        <li><a href="../../#portfolio" class="active"><i class="bi bi-images navicon"></i> Portfolio</a></li>
${projectSidebarNav}
        <li><a href="../../blogs/"><i class="bi bi-keyboard navicon"></i> Blog <i class="bi bi-link-45deg"></i></a></li>
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

  <a href="#" id="scroll-top" class="scroll-top d-flex align-items-center justify-content-center"><i class="bi bi-arrow-up-short"></i></a>
  <div id="preloader"></div>

  <script src="../../assets/vendor/bootstrap/js/bootstrap.bundle.min.js"></script>
  <script src="../../assets/vendor/php-email-form/validate.js"></script>
  <script src="../../assets/vendor/aos/aos.js"></script>
  <script src="../../assets/vendor/typed.js/typed.umd.js"></script>
  <script src="../../assets/vendor/purecounter/purecounter_vanilla.js"></script>
  <script src="../../assets/vendor/waypoints/noframework.waypoints.js"></script>
  <script src="../../assets/vendor/glightbox/js/glightbox.min.js"></script>
  <script src="../../assets/vendor/imagesloaded/imagesloaded.pkgd.min.js"></script>
  <script src="../../assets/vendor/isotope-layout/isotope.pkgd.min.js"></script>
  <script src="../../assets/vendor/swiper/swiper-bundle.min.js"></script>
  <script src="../../assets/js/main.js"></script>
  <script>
    document.addEventListener("DOMContentLoaded", function () {
        window.onscroll = updateScrollIndicator;
        window.onresize = updateScrollIndicator;

        function updateScrollIndicator() {
            const bar = document.getElementById("myBar");
            if (!bar) return;
            var winScroll = document.documentElement.scrollTop;
            var height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            if (height > 0) {
                var scrolled = (winScroll / height) * 100;
                bar.style.width = scrolled + "%";
            }
        }
    });
  </script>
</body>

</html>
`;
}

module.exports = {
    renderProjectPage
};

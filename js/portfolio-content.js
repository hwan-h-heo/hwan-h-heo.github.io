(function() {
    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderMedia(project) {
        if (project.video) {
            return `
                <video poster="${escapeHtml(project.poster || '')}" loop muted playsinline>
                    <source src="${escapeHtml(project.video)}" type="video/mp4">
                </video>
            `;
        }

        const gifAttrs = project.gif
            ? ` data-static="${escapeHtml(project.image)}" data-gif="${escapeHtml(project.gif)}"`
            : '';

        return `
            <img class="img-fluid" src="${escapeHtml(project.image)}"${gifAttrs} alt="${escapeHtml(project.alt || project.title)}">
        `;
    }

    function renderProject(project) {
        const filters = (project.categories || []).map((category) => `filter-${category}`).join(' ');
        const targetAttrs = project.external ? ' target="_blank"' : '';
        const externalIcon = project.external ? ' <i class="bi bi-box-arrow-up-right"></i>' : '';
        const badgeHtml = project.badge
            ? `<div class="top-left"><h6><span class="badge">${escapeHtml(project.badge)}</span></h6></div>`
            : '';
        const spinnerHtml = project.gif || project.video
            ? '<div class="loading-spinner" style="display: none;"></div>'
            : '';

        return `
            <div class="col-lg-4 col-sm-6 portfolio-item isotope-item ${filters}">
                <a class="portfolio-box" href="${escapeHtml(project.url)}"${targetAttrs} rel="noopener noreferrer">
                    <div class="aspect-ratio-box">
                        ${renderMedia(project)}
                    </div>
                    ${spinnerHtml}
                    ${badgeHtml}
                    <div class="polar_content">
                        <h6 data-hover-text="${escapeHtml(project.summary)}">${escapeHtml(project.title)}${externalIcon}</h6>
                    </div>
                    <p class="click-prompt">Click to see details</p>
                </a>
            </div>
        `;
    }

    function renderLink(link) {
        const icon = link.icon || 'bi-link';
        return `
            <a class="btn btn-sm btn-outline-dark" href="${escapeHtml(link.url)}">
                <div class="d-inline-block bi ${escapeHtml(icon)} me-2"></div>${escapeHtml(link.label)}
            </a>
        `;
    }

    function renderPublication(publication) {
        const linksHtml = (publication.links || []).length
            ? `<div>${publication.links.map(renderLink).join('')}</div>`
            : '';

        return `
            <li class="mb-4">
                <p class="mb-1"><i class="bi bi-file-earmark-text"></i><strong> ${escapeHtml(publication.title)}</strong></p>
                <p class="mb-0">
                    <em>
                        ${publication.authorsHtml}<br/>
                        ${publication.venueHtml}
                    </em>
                </p>
                ${linksHtml}
            </li>
        `;
    }

    function renderTalk(talk) {
        const titleHtml = talk.titleHtml || escapeHtml(talk.title);

        return `
            <li class="mb-4">
                <p class="mb-1"><strong>${titleHtml}</strong></p>
                <p class="mb-0"><em>${talk.venueHtml}</em>, <em>${escapeHtml(talk.date)}</em></p>
            </li>
        `;
    }

    function initPortfolioLayout(layoutElement, container) {
        if (!window.Isotope || !window.imagesLoaded || !layoutElement || !container) {
            return;
        }

        const layout = layoutElement.getAttribute('data-layout') || 'masonry';
        const filter = layoutElement.getAttribute('data-default-filter') || '*';
        const sort = layoutElement.getAttribute('data-sort') || 'original-order';

        imagesLoaded(container, function() {
            const isotope = new Isotope(container, {
                itemSelector: '.isotope-item',
                layoutMode: layout,
                filter,
                sortBy: sort,
                percentPosition: true
            });

            layoutElement.querySelectorAll('.isotope-filters li').forEach((filters) => {
                filters.addEventListener('click', function() {
                    const activeFilter = layoutElement.querySelector('.isotope-filters .filter-active');
                    if (activeFilter) {
                        activeFilter.classList.remove('filter-active');
                    }
                    this.classList.add('filter-active');
                    isotope.arrange({ filter: this.getAttribute('data-filter') });
                }, false);
            });
        });
    }

    document.addEventListener('DOMContentLoaded', async function() {
        const projectContainer = document.getElementById('portfolio-projects');
        const publicationsContainer = document.getElementById('portfolio-publications');
        const talksContainer = document.getElementById('portfolio-talks');

        if (!projectContainer || !publicationsContainer || !talksContainer || !window.siteDataClient) {
            return;
        }

        const siteData = await window.siteDataClient.loadSiteData();

        projectContainer.innerHTML = (siteData.portfolioProjects || []).map(renderProject).join('');
        publicationsContainer.innerHTML = (siteData.publications || []).map(renderPublication).join('');
        talksContainer.innerHTML = (siteData.talks || []).map(renderTalk).join('');

        if (window.initPortfolioBoxes) {
            window.initPortfolioBoxes(projectContainer);
        }

        initPortfolioLayout(document.querySelector('.portfolio .isotope-layout'), projectContainer);
    });
})();

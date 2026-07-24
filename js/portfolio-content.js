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
        const targetAttrs = project.external ? ' target="_blank" rel="noopener noreferrer"' : '';
        const externalIcon = project.external ? ' <i class="bi bi-box-arrow-up-right"></i>' : '';
        const badgeHtml = project.badge
            ? `<div class="top-left"><h6><span class="badge">${escapeHtml(project.badge)}</span></h6></div>`
            : '';
        const spinnerHtml = project.gif || project.video
            ? '<div class="loading-spinner" style="display: none;"></div>'
            : '';

        return `
            <article class="col-lg-4 col-sm-6 portfolio-item isotope-item ${filters}">
                <a class="portfolio-box" href="${escapeHtml(project.url)}"${targetAttrs}>
                    <div class="aspect-ratio-box">
                        ${renderMedia(project)}
                    </div>
                    ${spinnerHtml}
                    ${badgeHtml}
                    <div class="polar_content">
                        <h6 data-hover-text="${escapeHtml(project.summary)}">${escapeHtml(project.title)}${externalIcon}</h6>
                        <p class="portfolio-card-summary">${escapeHtml(project.summary)}</p>
                    </div>
                    <p class="click-prompt">Click to see details</p>
                </a>
            </article>
        `;
    }

    function renderLink(link) {
        const icon = link.icon || 'bi-link';
        const external = /^https?:\/\//.test(link.url || '');
        const targetAttrs = external ? ' target="_blank" rel="noopener noreferrer"' : '';
        return `
            <a class="portfolio-paper-link" href="${escapeHtml(link.url)}"${targetAttrs}>
                <i class="bi ${escapeHtml(icon)}" aria-hidden="true"></i>
                ${escapeHtml(link.label)}
            </a>
        `;
    }

    function renderPublication(publication) {
        const linksHtml = (publication.links || []).length
            ? publication.links.map(renderLink).join('')
            : '';

        return `
            <li class="portfolio-publication">
                <div class="portfolio-publication-copy">
                    <h3>${escapeHtml(publication.title)}</h3>
                    <p class="portfolio-publication-authors">${publication.authorsHtml}</p>
                    <p class="portfolio-publication-venue">${publication.venueHtml}</p>
                </div>
                ${linksHtml ? `<div class="portfolio-paper-links">${linksHtml}</div>` : ''}
            </li>
        `;
    }

    function renderTalk(talk) {
        const titleHtml = talk.titleHtml || escapeHtml(talk.title);

        return `
            <li class="portfolio-talk">
                <time>${escapeHtml(talk.date)}</time>
                <div>
                    <h3>${titleHtml}</h3>
                    <p>${talk.venueHtml}</p>
                </div>
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

    const api = {
        renderProject,
        renderPublication,
        renderTalk
    };

    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    }

    if (typeof document === 'undefined') {
        return;
    }

    document.addEventListener('DOMContentLoaded', async function() {
        const section = document.getElementById('portfolio');
        const projectContainer = document.getElementById('portfolio-projects');
        const publicationsContainer = document.getElementById('portfolio-publications');
        const talksContainer = document.getElementById('portfolio-talks');
        const markReady = () => {
            if (!section) return;
            section.dataset.sectionReady = 'true';
            section.dispatchEvent(new CustomEvent('portfolio:section-ready'));
        };

        if (!projectContainer || !publicationsContainer || !talksContainer || !window.siteDataClient) {
            markReady();
            return;
        }

        try {
            const siteData = await window.siteDataClient.loadSiteData();

            if (!projectContainer.querySelector('.portfolio-box')) {
                projectContainer.innerHTML = (siteData.portfolioProjects || []).map(renderProject).join('');
            }
            if (!publicationsContainer.querySelector('.portfolio-publication')) {
                publicationsContainer.innerHTML = (siteData.publications || []).map(renderPublication).join('');
            }
            if (!talksContainer.querySelector('.portfolio-talk')) {
                talksContainer.innerHTML = (siteData.talks || []).map(renderTalk).join('');
            }

            if (window.initPortfolioBoxes) {
                window.initPortfolioBoxes(projectContainer);
            }

            initPortfolioLayout(section, projectContainer);
        } catch (error) {
            console.error(error);
        } finally {
            markReady();
        }
    });
})();

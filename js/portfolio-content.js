(function() {
    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    const CATEGORY_LABELS = {
        app: 'Application',
        research: 'Research',
        per: 'Personal'
    };
    const PROJECT_ACTION_LABELS = {
        varco3d: 'Service Overview',
        capa: 'Research Project',
        deepsfm: 'Project Notes',
        '2dgs-viewer': 'Viewer Project',
        'instant-pose': 'Paper Project',
        'nerf-in-game': 'Engine Project'
    };

    function isSelectedProject(project, index) {
        if (typeof project.selected === 'boolean') {
            return project.selected;
        }
        return index < 3;
    }

    function getProjectCategoryLabels(project) {
        return (project.categories || [])
            .map((category) => CATEGORY_LABELS[category] || category)
            .filter(Boolean);
    }

    function getProjectActionLabel(project) {
        return PROJECT_ACTION_LABELS[project.id] || 'Project Page';
    }

    function renderProjectTags(project) {
        return (project.tags || [])
            .slice(0, 3)
            .map((tag) => `<span class="portfolio-project-tag">${escapeHtml(tag)}</span>`)
            .join('');
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

    function renderProject(project, index = 0) {
        const selected = isSelectedProject(project, index);
        const categoryLabels = getProjectCategoryLabels(project);
        const targetAttrs = project.external ? ' target="_blank" rel="noopener noreferrer"' : '';
        const externalIcon = project.external ? ' <i class="bi bi-box-arrow-up-right"></i>' : '';
        const badgeHtml = project.badge
            ? `<span class="portfolio-project-badge">${escapeHtml(project.badge)}</span>`
            : '';
        const spinnerHtml = project.gif || project.video
            ? '<div class="loading-spinner" style="display: none;"></div>'
            : '';
        const selectedLabel = selected ? '<span>Selected Project</span>' : '<span>Project Archive</span>';
        const actionLabel = getProjectActionLabel(project);
        const tagsHtml = renderProjectTags(project);
        const hidden = selected ? '' : ' hidden';

        return `
            <article class="portfolio-project-item" data-selected="${selected ? 'true' : 'false'}"${hidden}>
                <a class="portfolio-project-link" href="${escapeHtml(project.url)}"${targetAttrs}>
                    <span class="portfolio-project-cover">
                        ${renderMedia(project)}
                        ${spinnerHtml}
                        ${badgeHtml}
                    </span>
                    <span class="portfolio-project-body">
                        <span class="portfolio-project-eyebrow">
                            ${selectedLabel}
                            ${categoryLabels[0] ? `<span>${escapeHtml(categoryLabels[0])}</span>` : ''}
                        </span>
                        <span class="portfolio-project-title">${escapeHtml(project.title)}${externalIcon}</span>
                        <span class="portfolio-project-summary">${escapeHtml(project.summary)}</span>
                        ${tagsHtml ? `<span class="portfolio-project-tags">${tagsHtml}</span>` : ''}
                        <span class="portfolio-project-meta">${escapeHtml(actionLabel)} <i class="bi bi-arrow-up-right" aria-hidden="true"></i></span>
                    </span>
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

    function setPortfolioView(section, view) {
        const resolvedView = view === 'all' ? 'all' : 'selected';
        section.dataset.portfolioView = resolvedView;

        section.querySelectorAll('.portfolio-project-item').forEach((item) => {
            item.hidden = resolvedView === 'selected' && item.dataset.selected !== 'true';
        });

        section.querySelectorAll('[data-portfolio-view]').forEach((control) => {
            const active = control.dataset.portfolioView === resolvedView;
            control.classList.toggle('filter-active', active);
            control.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
    }

    function initPortfolioViewToggle(section) {
        if (!section) return;

        section.querySelectorAll('[data-portfolio-view]').forEach((control) => {
            const activate = () => setPortfolioView(section, control.dataset.portfolioView);
            control.addEventListener('click', activate);
            control.addEventListener('keydown', (event) => {
                if (event.key !== 'Enter' && event.key !== ' ') return;
                event.preventDefault();
                activate();
            });
        });

        setPortfolioView(section, section.dataset.portfolioView || 'selected');
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

            projectContainer.className = 'portfolio-project-list';

            if (!projectContainer.querySelector('.portfolio-project-item')) {
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

            initPortfolioViewToggle(section);
        } catch (error) {
            console.error(error);
        } finally {
            markReady();
        }
    });
})();

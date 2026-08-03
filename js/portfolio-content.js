(function() {
    const icons = typeof window === 'undefined'
        ? require('../assets/js/site-icons')
        : window.SiteIcons;
    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function isSelectedProject(project, index) {
        if (typeof project.selected === 'boolean') {
            return project.selected;
        }
        return index < 3;
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
            <img src="${escapeHtml(project.image)}"${gifAttrs} alt="${escapeHtml(project.alt || project.title)}">
        `;
    }

    function renderProjectTags(project) {
        return (project.tags || [])
            .slice(0, 2)
            .map((tag) => `<span class="portfolio-project-tag">${escapeHtml(tag)}</span>`)
            .join('');
    }

    function renderProject(project, index = 0) {
        const selected = isSelectedProject(project, index);
        const allOrder = Number.isFinite(Number(project.allOrder)) ? Number(project.allOrder) : index;
        const targetAttrs = project.external ? ' target="_blank" rel="noopener noreferrer"' : '';
        const externalIcon = project.external ? ` ${icons.render('box-arrow-up-right')}` : '';
        const eyebrowHtml = project.typeLabel
            ? `<span class="portfolio-project-eyebrow">${escapeHtml(project.typeLabel)}</span>`
            : '';
        const spinnerHtml = project.gif || project.video
            ? '<div class="loading-spinner" style="display: none;"></div>'
            : '';
        const tagsHtml = renderProjectTags(project);
        const accoladeHtml = project.accolade
            ? `<span class="portfolio-project-accolade">${escapeHtml(project.accolade)}</span>`
            : '';
        const hidden = selected ? '' : ' hidden';

        return `
            <article class="portfolio-project-item" data-selected="${selected ? 'true' : 'false'}" data-selected-order="${index}" data-all-order="${allOrder}"${hidden}>
                <div class="portfolio-project-layout">
                    <a class="portfolio-project-cover-link" href="${escapeHtml(project.url)}"${targetAttrs} aria-label="View ${escapeHtml(project.title)}">
                    <span class="portfolio-project-cover">
                        ${renderMedia(project)}
                        ${spinnerHtml}
                    </span>
                    </a>
                    <div class="portfolio-project-body">
                        ${eyebrowHtml}
                        <a class="portfolio-project-title-link" href="${escapeHtml(project.url)}"${targetAttrs}>
                            <span class="portfolio-project-title">${escapeHtml(project.title)}${externalIcon}</span>
                        </a>
                        <span class="portfolio-project-summary">${escapeHtml(project.summary)}</span>
                        <span class="portfolio-project-tags">${tagsHtml}</span>
                        <span class="portfolio-project-meta">
                            <span>${escapeHtml(project.organization)} / ${escapeHtml(project.period)}</span>
                            ${accoladeHtml}
                        </span>
                    </div>
                </div>
            </article>
        `;
    }

    function renderLink(link) {
        const icon = link.icon || 'link';
        const external = /^https?:\/\//.test(link.url || '');
        const targetAttrs = external ? ' target="_blank" rel="noopener noreferrer"' : '';
        return `
            <a class="portfolio-paper-link" href="${escapeHtml(link.url)}"${targetAttrs}>
                ${icons.render(icon)}
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
        const linksHtml = (talk.links || []).length
            ? talk.links.map(renderLink).join('')
            : '';

        return `
            <li class="portfolio-talk">
                <time>${escapeHtml(talk.date)}</time>
                <div>
                    <h3>${titleHtml}</h3>
                    <p>${talk.venueHtml}</p>
                </div>
                ${linksHtml ? `<div class="portfolio-paper-links portfolio-talk-links">${linksHtml}</div>` : ''}
            </li>
        `;
    }

    function setPortfolioView(section, view) {
        const resolvedView = view === 'all' ? 'all' : 'selected';
        section.dataset.portfolioView = resolvedView;

        const projectList = section.querySelector('.portfolio-project-list');
        const orderKey = resolvedView === 'all' ? 'allOrder' : 'selectedOrder';
        const projectItems = Array.from(section.querySelectorAll('.portfolio-project-item'))
            .sort((first, second) => Number(first.dataset[orderKey]) - Number(second.dataset[orderKey]));
        projectItems.forEach((item) => projectList?.appendChild(item));

        const visibleItems = [];
        projectItems.forEach((item) => {
            item.hidden = resolvedView === 'selected' && item.dataset.selected !== 'true';
            item.classList.remove('is-last-visible');
            if (!item.hidden) {
                visibleItems.push(item);
            }
        });
        visibleItems[visibleItems.length - 1]?.classList.add('is-last-visible');

        section.querySelectorAll('[data-portfolio-view]').forEach((control) => {
            const active = control.dataset.portfolioView === resolvedView;
            control.classList.toggle('filter-active', active);
            control.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
    }

    function initPortfolioViewToggle(section) {
        if (!section) return;

        const projectItems = Array.from(section.querySelectorAll('.portfolio-project-item'));
        const counts = {
            selected: projectItems.filter((item) => item.dataset.selected === 'true').length,
            all: projectItems.length
        };

        section.querySelectorAll('[data-portfolio-count]').forEach((count) => {
            count.textContent = counts[count.dataset.portfolioCount] ?? '';
        });

        section.querySelectorAll('[data-portfolio-view]').forEach((control) => {
            const activate = () => setPortfolioView(section, control.dataset.portfolioView);
            control.addEventListener('click', activate);
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

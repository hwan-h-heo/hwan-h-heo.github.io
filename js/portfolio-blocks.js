(function(root, factory) {
    const api = factory(root);
    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    }
    if (root) {
        root.portfolioBlocks = api;
        root.document.addEventListener('DOMContentLoaded', api.mountPortfolioBlocks);
    }
})(typeof window !== 'undefined' ? window : null, function(root) {
    const icons = root?.SiteIcons || require('../assets/js/site-icons');
    const CONTENT_PATH = '/content/portfolio/home.json';
    const HERO_SCROLL_CUE_ENABLED = true;

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderHeroActions(actions) {
        return (actions || []).map((action, index) => {
            const style = ['section', 'download'].includes(action.style)
                ? action.style
                : 'section';
            const isSectionLink = (action.url || '').startsWith('#');
            const download = action.download ? ' download' : '';
            const trailingIcon = action.download
                ? 'download'
                : isSectionLink
                    ? 'arrow-down-right'
                    : 'arrow-up-right';
            return `
                <a class="hero-action hero-action--${style}" href="${escapeHtml(action.url)}"${download}>
                    <small class="hero-action-index" aria-hidden="true">${String(index + 1).padStart(2, '0')}</small>
                    <span class="hero-action-label">${escapeHtml(action.label)}</span>
                    ${icons.render(trailingIcon)}
                </a>
            `;
        }).join('');
    }

    function renderHeroLead(block) {
        const lines = Array.isArray(block.leadLines) && block.leadLines.length
            ? block.leadLines
            : [block.lead];
        return lines
            .filter((line) => line)
            .map((line) => `<span class="hero-lead-line">${escapeHtml(line)}</span>`)
            .join('\n');
    }

    function renderHeroScrollCue() {
        if (!HERO_SCROLL_CUE_ENABLED) {
            return '';
        }

        return `
            <a href="#portfolio" class="scroll-down-arrow" aria-label="Scroll to portfolio">
                ${icons.render('chevron-down')}
            </a>
        `;
    }

    function renderHero(block) {
        const subtitle = block.subtitle || (block.typedItems || []).join(' · ');
        return `
            <div class="hero-content">
                <h1>${escapeHtml(block.title)}</h1>
                <hr/>
                <p class="subtitle">
                    <span class="hero-subtitle-text">${escapeHtml(subtitle)}</span>
                </p>
                <p class="lead">${renderHeroLead(block)}</p>
                <div class="hero-actions">
                    ${renderHeroActions(block.actions)}
                </div>
            </div>
            ${renderHeroScrollCue()}
        `;
    }

    function renderAboutContacts(links) {
        if (!Array.isArray(links) || !links.length) {
            return '';
        }

        const linkHtml = links.map((link) => {
            const externalAttrs = /^https?:\/\//i.test(link.url || '')
                ? ' target="_blank" rel="noopener noreferrer"'
                : '';
            const icon = link.icon || 'arrow-up-right';

            return `
                <a class="about-contact-link" href="${escapeHtml(link.url)}"${externalAttrs}>
                    ${icons.render(icon)}
                    <span>
                        <strong>${escapeHtml(link.label)}</strong>
                        <small>${escapeHtml(link.value)}</small>
                    </span>
                    ${icons.render('arrow-up-right', { className: 'about-contact-arrow' })}
                </a>
            `;
        }).join('');

        return `
            <aside class="about-contact" aria-label="Contact">
                <p class="about-contact-label">Contact</p>
                <div class="about-contact-links">${linkHtml}</div>
            </aside>
        `;
    }

    function renderAbout(block) {
        return `
            <div class="portfolio-shell section-title">
                <h2>${escapeHtml(block.title)}</h2>
            </div>

            <div class="portfolio-shell">
                <div class="about-layout">
                    <div class="about-media">
                        <img src="${escapeHtml(block.image)}" class="about-profile-image" alt="${escapeHtml(block.imageAlt)}">
                    </div>
                    <div class="content about-copy">
                        <div class="about-identity">
                            <h2 class="about-name">${escapeHtml(block.name || block.role)}</h2>
                            <p class="about-meta">
                                <span class="about-role">${escapeHtml(block.role)}</span>
                                <span class="about-affiliation">${escapeHtml(block.affiliation)}</span>
                            </p>
                        </div>
                        <p class="about-intro">${block.introHtml || ''}</p>
                        <p class="about-bio">${block.bodyHtml || ''}</p>
                        ${renderAboutContacts(block.contactLinks)}
                    </div>
                </div>
            </div>
        `;
    }

    function renderResumeItem(item) {
        const bullets = (item.bulletsHtml || []).length
            ? `<ul class="resume-entry-details">${item.bulletsHtml.map((bullet) => `<li>${bullet}</li>`).join('')}</ul>`
            : '';
        const title = item.titleHtml || escapeHtml(item.title);
        const period = item.period
            ? `<time class="resume-period">${escapeHtml(item.period)}</time>`
            : '';
        const meta = item.metaHtml ? `<p class="resume-entry-meta">${item.metaHtml}</p>` : '';
        const body = item.bodyHtml ? `<p class="resume-entry-copy">${item.bodyHtml}</p>` : '';

        return `
            <article class="resume-item${item.period ? '' : ' resume-item-undated'}">
                ${period}
                <div class="resume-entry">
                    <h4>${title}</h4>
                    ${meta}
                    ${body}
                    ${bullets}
                </div>
            </article>
        `;
    }

    function renderResumeSection(section) {
        return `
            <div class="resume-index-section">
                <h3 class="resume-title">${escapeHtml(section.title)}</h3>
                <div class="resume-index-list">
                    ${(section.items || []).map(renderResumeItem).join('')}
                </div>
            </div>
        `;
    }

    function renderResume(block) {
        return `
            <div class="portfolio-shell section-title resume-section-heading">
                <h2>${escapeHtml(block.title)}</h2>
                <a class="resume-cv-link" href="${escapeHtml(block.cvUrl)}" download aria-label="Download curriculum vitae">
                    <span>Download CV</span>
                    ${icons.render('download')}
                </a>
            </div>
            <div class="portfolio-shell resume-content">
                <div class="resume-layout">
                    ${(block.columns || []).map((column) => `
                        <div class="resume-column">
                            ${(column.sections || []).map(renderResumeSection).join('')}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    function renderBlock(block) {
        if (block.type === 'hero') {
            return renderHero(block);
        }
        if (block.type === 'aboutProfile') {
            return renderAbout(block);
        }
        if (block.type === 'resume') {
            return renderResume(block);
        }
        return '';
    }

    function markSectionReady(section) {
        if (!section) return;
        section.dataset.sectionReady = 'true';
        section.dispatchEvent(new root.CustomEvent('portfolio:section-ready'));
    }

    async function loadPortfolioBlocks() {
        const response = await root.fetch(CONTENT_PATH);
        if (!response.ok) {
            throw new Error(`Failed to load ${CONTENT_PATH}`);
        }
        return response.json();
    }

    async function mountPortfolioBlocks() {
        const targets = root.document.querySelectorAll('[data-portfolio-block]');
        if (targets.length === 0) {
            return;
        }

        let coreStatus = 'true';
        try {
            const data = await loadPortfolioBlocks();
            const blocksById = Object.fromEntries((data.blocks || []).map((block) => [block.id, block]));
            let renderedBlock = false;

            targets.forEach((target) => {
                const block = blocksById[target.getAttribute('data-portfolio-block')];
                if (block && !target.innerHTML.trim()) {
                    target.innerHTML = renderBlock(block);
                    renderedBlock = true;
                }
                if (block) markSectionReady(target);
            });

            if (!renderedBlock) {
                return;
            }

        } catch (error) {
            coreStatus = 'error';
            console.error(error);
        } finally {
            root.document.documentElement.dataset.portfolioCoreReady = coreStatus;
            root.document.dispatchEvent(new root.CustomEvent('portfolio:core-ready', {
                detail: { status: coreStatus }
            }));
        }
    }

    return {
        mountPortfolioBlocks,
        renderBlock
    };
});

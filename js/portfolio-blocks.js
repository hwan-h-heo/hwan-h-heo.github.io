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

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderHeroActions(actions) {
        return (actions || []).map((action, index) => `
            <a href="${escapeHtml(action.url)}" data-form="${index}">
                <span class="pbr-action-index" aria-hidden="true">${String(index + 1).padStart(2, '0')}</span>
                <span class="pbr-action-label">${escapeHtml(action.label)}</span>
                <span aria-hidden="true">↘</span>
            </a>
        `).join('');
    }

    function renderHeroLead(block) {
        const lines = Array.isArray(block.leadLines) && block.leadLines.length
            ? block.leadLines
            : [block.lead];
        return lines.filter(Boolean)
            .map(line => `<span class="pbr-lead-line">${escapeHtml(line)}</span>`)
            .join(' ');
    }

    function renderHero(block) {
        const subtitle = block.subtitle || (block.typedItems || []).join(' · ');
        return `
            <picture class="pbr-poster">
                <source media="(max-width:600px)" srcset="/assets/hero/sculpture/poster-390.webp">
                <source media="(max-width:900px)" srcset="/assets/hero/sculpture/poster-768.webp">
                <img src="/assets/hero/sculpture/poster-1440.webp" alt="" aria-hidden="true" fetchpriority="high">
            </picture>
            <div class="pbr-scene" role="img" aria-label="An eroded stone form in a monumental limestone aperture, lit by raking white light"></div>
            <div class="pbr-veil"></div>
            <header class="pbr-masthead">
                <a class="pbr-signature" href="#home">${escapeHtml(block.title)}</a>
                <nav aria-label="Main navigation"><a class="pbr-about-link" href="#about">About</a></nav>
            </header>
            <div class="pbr-intro">
                <div class="pbr-title">
                    <p class="pbr-folio">00 / PORTFOLIO</p>
                    <h1 class="pbr-name">${escapeHtml(block.title)}</h1>
                </div>
                <div class="pbr-copy">
                    <div class="pbr-identity">
                        <p class="pbr-role">${escapeHtml(subtitle)}</p>
                        <p class="pbr-affiliation">${escapeHtml(block.affiliation)}</p>
                    </div>
                    <p class="pbr-lead">${renderHeroLead(block)}</p>
                    <div class="pbr-actions">${renderHeroActions(block.actions)}</div>
                </div>
            </div>
            <div class="pbr-foot"><div class="pbr-foot-left">
                <span>3D generation · Systems · Graphics</span>
                <button class="pbr-motion" type="button" aria-pressed="true" hidden>Motion on</button>
            </div></div>
            <a class="pbr-scroll-cue" href="#portfolio" aria-label="Scroll to Projects">${icons.render('chevron-down')}</a>
        `;
    }

    function renderAboutContacts(links) {
        if (!Array.isArray(links) || !links.length) {
            return '';
        }

        const linkHtml = links.map((link) => {
            const isExternal = /^https?:\/\//i.test(link.url || '');
            const externalAttrs = isExternal
                ? ' target="_blank" rel="noopener noreferrer"'
                : '';
            const destinationIcon = isExternal ? 'box-arrow-up-right' : 'arrow-up-right';

            return `
                <a class="about-contact-link" href="${escapeHtml(link.url)}"${externalAttrs}>
                    ${icons.render(link.icon || 'link', { className: 'about-contact-icon' })}
                    <span class="about-contact-copy">
                        <strong>${escapeHtml(link.label)}</strong>
                        <small>${escapeHtml(link.value)}</small>
                    </span>
                    ${icons.render(destinationIcon, { className: 'about-contact-arrow' })}
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
        const depthAttribute = block.depthImage
            ? ` data-about-depth="${escapeHtml(block.depthImage)}"`
            : '';

        return `
            <div class="portfolio-shell section-title about-section-heading">
                <div class="portfolio-chapter-marker" aria-hidden="true">
                    <span class="portfolio-chapter-label"><span>03 /</span> Profile</span>
                    <span class="portfolio-chapter-rule"></span>
                </div>
                <div class="about-heading-copy">
                    <h2>${escapeHtml(block.title)}</h2>
                </div>
            </div>

            <div class="portfolio-shell">
                <div class="about-editorial-layout">
                    <p class="about-standfirst">${block.introHtml || ''}</p>
                    <figure class="about-profile-note">
                        <div class="about-media" data-about-portrait${depthAttribute}>
                            <img src="${escapeHtml(block.image)}" class="about-profile-image" alt="${escapeHtml(block.imageAlt)}" loading="lazy" decoding="async">
                        </div>
                        <figcaption class="about-identity">
                            <p class="about-portrait-affordance" aria-hidden="true">
                                <span>Point cloud</span><span class="about-portrait-affordance-hint">/ Hover for depth</span>
                            </p>
                            <h3 class="about-name">${escapeHtml(block.name || block.role)}</h3>
                            <p class="about-meta">
                                <span class="about-role">${escapeHtml(block.role)}</span>
                                <span class="about-affiliation">${escapeHtml(block.affiliation)}</span>
                            </p>
                        </figcaption>
                    </figure>
                    <div class="content about-copy">
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
            <div class="portfolio-shell resume-disclosure-shell">
                <div class="resume-document-row">
                    <span>
                        <span class="portfolio-disclosure-kicker">Document</span>
                        <strong>Curriculum Vitae</strong>
                    </span>
                    <span class="portfolio-disclosure-meta resume-document-meta">
                        <a class="resume-document-download" href="${escapeHtml(block.cvUrl)}" download aria-label="Download curriculum vitae PDF">
                            <span>Download</span>
                            ${icons.render('download')}
                        </a>
                    </span>
                </div>
                <details class="portfolio-disclosure resume-disclosure">
                    <summary>
                        <span>
                            <span class="portfolio-disclosure-kicker">Career</span>
                            <strong>Experience &amp; Education</strong>
                        </span>
                        <span class="portfolio-disclosure-meta resume-disclosure-meta">
                            <span class="portfolio-disclosure-state">
                                <span class="portfolio-disclosure-state-closed">View details</span>
                                <span class="portfolio-disclosure-state-open">Close</span>
                            </span>
                            ${icons.render('chevron-down')}
                        </span>
                    </summary>
                    <div class="resume-disclosure-panel">
                        <div class="resume-layout">
                            ${(block.columns || []).map((column) => `
                                <div class="resume-column">
                                    ${(column.sections || []).map(renderResumeSection).join('')}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </details>
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

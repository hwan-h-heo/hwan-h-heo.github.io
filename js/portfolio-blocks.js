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
    const CONTENT_PATH = '/content/portfolio/home.json';

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderIconLinks(links) {
        return (links || []).map((link) => `
            <a aria-label="${escapeHtml(link.label)}" href="${escapeHtml(link.url)}"><i class="${escapeHtml(link.icon)}"></i></a>
        `).join('');
    }

    function renderHero(block) {
        return `
            <div class="hero-content">
                <h1>${escapeHtml(block.title)}</h1>
                <hr/>
                <p class="subtitle">
                    <span class="typed" data-typed-items="${escapeHtml((block.typedItems || []).join(', '))}"></span>
                </p>
                <p class="lead">${escapeHtml(block.lead)}</p>
                <div class="social-links">
                    ${renderIconLinks(block.links)}
                </div>
            </div>
        `;
    }

    function renderFacts(facts) {
        return (facts || []).map((fact) => `
            <li><i class="bi bi-chevron-right"></i> <strong>${escapeHtml(fact.label)}:</strong> <span>${fact.valueHtml || ''}</span></li>
        `).join('');
    }

    function renderAbout(block) {
        return `
            <div class="container section-title" data-aos="fade-up">
                <h2>${escapeHtml(block.title)}</h2>
                <p>${block.introHtml || ''}</p>
            </div>

            <div class="container" data-aos="fade-up" data-aos-delay="100">
                <div class="row gy-4 justify-content-center">
                    <div class="col-lg-4" style="transition: 0.3s ease-in-out;">
                        <img src="${escapeHtml(block.image)}" class="img-fluid" alt="${escapeHtml(block.imageAlt)}">
                    </div>
                    <div class="col-lg-8 content">
                        <h2>${escapeHtml(block.role)}</h2>
                        <p class="fst-italic py-3">${escapeHtml(block.tagline)}</p>
                        <div class="row">
                            <div class="col-10">
                                <ul>${renderFacts(block.facts)}</ul>
                            </div>
                        </div>
                        <p class="py-3">${block.bodyHtml || ''}</p>
                    </div>
                </div>
            </div>
            <a href="#about" class="scroll-down-arrow" aria-label="Scroll down"><i class="bi bi-chevron-double-down"></i></a>
        `;
    }

    function renderResumeItem(item) {
        const bullets = (item.bulletsHtml || []).length
            ? `<ul>${item.bulletsHtml.map((bullet) => `<li>${bullet}</li>`).join('')}</ul>`
            : '';
        const title = item.titleHtml || escapeHtml(item.title);
        const period = item.period ? `<h5>${escapeHtml(item.period)}</h5>` : '';
        const meta = item.metaHtml ? `<p><em>${item.metaHtml}</em></p>` : '';
        const body = item.bodyHtml ? `<p>${item.bodyHtml}</p>` : '';

        return `
            <div class="resume-item${item.period ? '' : ' pb-0'}">
                <h4>${title}</h4>
                ${period}
                ${meta}
                ${body}
                ${bullets}
            </div>
        `;
    }

    function renderResumeSection(section) {
        return `
            <h3 class="resume-title">${escapeHtml(section.title)}</h3>
            ${(section.items || []).map(renderResumeItem).join('')}
        `;
    }

    function renderResume(block) {
        return `
            <div class="container section-title" data-aos="fade-up" style="margin-bottom: -5rem; padding-left: 1rem">
                <div class="d-flex align-items-center justify-content-between mb-4">
                    <h2>${escapeHtml(block.title)}</h2>
                    <a class="btn btn-dark px-3 py-2" href="${escapeHtml(block.cvUrl)}">
                        <div class="d-inline-block bi bi-download me-2"></div>
                        Download CV
                    </a>
                </div>
            </div>
            <div class="container">
                <div class="row">
                    ${(block.columns || []).map((column, index) => `
                        <div class="col-lg-6" data-aos="fade-up" data-aos-delay="${index === 0 ? '100' : '200'}" style="${index === 0 ? 'padding-left: 2rem;' : 'padding-right: 1rem; padding-left: 2rem;'}">
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

    function restartTypedEffect() {
        root.document.querySelectorAll('.typed').forEach((typedElement) => {
            if (root.Typed) {
                const typedStrings = typedElement.getAttribute('data-typed-items');
                if (typedStrings) {
                    new root.Typed(typedElement, {
                        strings: typedStrings.split(',').map((item) => item.trim()),
                        loop: true,
                        typeSpeed: 100,
                        backSpeed: 50,
                        backDelay: 2000
                    });
                }
            }
        });
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
            });

            if (!renderedBlock) {
                return;
            }

            restartTypedEffect();
            if (root.AOS) {
                root.AOS.refresh();
            }
        } catch (error) {
            console.error(error);
        }
    }

    return {
        mountPortfolioBlocks,
        renderBlock
    };
});

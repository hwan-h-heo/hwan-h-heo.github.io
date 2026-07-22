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
        const subtitle = block.subtitle || (block.typedItems || []).join(' · ');
        return `
            <div class="hero-content">
                <h1>${escapeHtml(block.title)}</h1>
                <hr/>
                <p class="subtitle">
                    <span class="hero-subtitle-text">${escapeHtml(subtitle)}</span>
                </p>
                <p class="lead">${escapeHtml(block.lead)}</p>
                <div class="social-links">
                    ${renderIconLinks(block.links)}
                </div>
            </div>
            <a href="#about" class="scroll-down-arrow" aria-label="Scroll down">
                <span class="scroll-down-label">Research · Projects · Notes</span>
                <i class="bi bi-chevron-double-down"></i>
            </a>
        `;
    }

    function renderAbout(block) {
        return `
            <div class="container section-title" data-aos="fade-up">
                <h2>${escapeHtml(block.title)}</h2>
            </div>

            <div class="container" data-aos="fade-up" data-aos-delay="100">
                <div class="row gy-4 align-items-center justify-content-center">
                    <div class="col-lg-4" style="transition: 0.3s ease-in-out;">
                        <img src="${escapeHtml(block.image)}" class="img-fluid about-profile-image" alt="${escapeHtml(block.imageAlt)}">
                    </div>
                    <div class="col-lg-8 content about-copy">
                        <div class="about-identity">
                            <h2 class="about-name">${escapeHtml(block.name || block.role)}</h2>
                            <p class="about-role">${escapeHtml(block.role)}</p>
                            <p class="about-affiliation">${escapeHtml(block.affiliation)}</p>
                        </div>
                        <p class="about-intro">${block.introHtml || ''}</p>
                        <p class="about-bio">${block.bodyHtml || ''}</p>
                    </div>
                </div>
            </div>
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

    function initHeroAboutScrollHandoff() {
        const hero = root.document.getElementById('home');
        const about = root.document.getElementById('about');
        if (!hero || !about || hero.dataset.scrollHandoffBound === 'true') {
            return;
        }

        hero.dataset.scrollHandoffBound = 'true';
        const reducedMotion = root.matchMedia('(prefers-reduced-motion: reduce)').matches;
        let accumulatedWheelDelta = 0;
        let wheelDirection = 0;
        let wheelResetTimer;
        let touchStartY = null;
        let isSnapping = false;

        const heroIsActive = () => root.scrollY < about.offsetTop - 2;
        const aboutStartIsActive = () => {
            const distanceFromAbout = root.scrollY - about.offsetTop;
            return distanceFromAbout >= -2 && distanceFromAbout <= 48;
        };
        const snapToSection = (section, canSnap) => {
            if (isSnapping || !canSnap()) {
                return;
            }

            isSnapping = true;
            section.scrollIntoView({
                behavior: reducedMotion ? 'auto' : 'smooth',
                block: 'start'
            });
            root.setTimeout(() => {
                isSnapping = false;
            }, reducedMotion ? 0 : 900);
        };
        const snapToAbout = () => snapToSection(about, heroIsActive);
        const snapToHero = () => snapToSection(hero, aboutStartIsActive);

        root.addEventListener('wheel', (event) => {
            const direction = Math.sign(event.deltaY);
            const shouldSnapDown = direction > 0 && heroIsActive();
            const shouldSnapUp = direction < 0 && aboutStartIsActive();
            if (!shouldSnapDown && !shouldSnapUp) {
                accumulatedWheelDelta = 0;
                wheelDirection = 0;
                return;
            }

            event.preventDefault();
            if (direction !== wheelDirection) {
                accumulatedWheelDelta = 0;
                wheelDirection = direction;
            }
            accumulatedWheelDelta += Math.abs(event.deltaY);
            root.clearTimeout(wheelResetTimer);
            wheelResetTimer = root.setTimeout(() => {
                accumulatedWheelDelta = 0;
                wheelDirection = 0;
            }, 180);

            if (accumulatedWheelDelta >= 24) {
                accumulatedWheelDelta = 0;
                shouldSnapDown ? snapToAbout() : snapToHero();
            }
        }, { passive: false });

        root.addEventListener('touchstart', (event) => {
            const canSnap = heroIsActive() || aboutStartIsActive();
            touchStartY = canSnap ? event.touches[0]?.clientY ?? null : null;
        }, { passive: true });

        root.addEventListener('touchmove', (event) => {
            if (touchStartY === null) {
                return;
            }

            const currentY = event.touches[0]?.clientY;
            if (typeof currentY !== 'number') {
                return;
            }

            const movement = touchStartY - currentY;
            const shouldSnapDown = movement >= 20 && heroIsActive();
            const shouldSnapUp = movement <= -20 && aboutStartIsActive();
            if (shouldSnapDown || shouldSnapUp) {
                event.preventDefault();
                touchStartY = null;
                shouldSnapDown ? snapToAbout() : snapToHero();
            }
        }, { passive: false });

        root.addEventListener('touchend', () => {
            touchStartY = null;
        }, { passive: true });

        root.addEventListener('keydown', (event) => {
            const target = event.target;
            const isEditable = target instanceof root.HTMLElement
                && (target.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName));
            if (isEditable) {
                return;
            }

            const isSpaceDown = event.key === ' ' && !event.shiftKey;
            const isSpaceUp = event.key === ' ' && event.shiftKey;
            const shouldSnapDown = heroIsActive()
                && (['ArrowDown', 'PageDown'].includes(event.key) || isSpaceDown);
            const shouldSnapUp = aboutStartIsActive()
                && (['ArrowUp', 'PageUp'].includes(event.key) || isSpaceUp);
            if (shouldSnapDown || shouldSnapUp) {
                event.preventDefault();
                shouldSnapDown ? snapToAbout() : snapToHero();
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

            initHeroAboutScrollHandoff();
            if (!renderedBlock) {
                return;
            }

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

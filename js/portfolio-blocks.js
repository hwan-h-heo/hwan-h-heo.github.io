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
        return (links || []).map((link) => {
            const external = /^https?:\/\//.test(link.url || '');
            const target = external ? ' target="_blank" rel="noopener noreferrer"' : '';
            return `
                <a aria-label="${escapeHtml(link.label)}" href="${escapeHtml(link.url)}"${target}>
                    <i class="${escapeHtml(link.icon)}"></i>
                </a>
            `;
        }).join('');
    }

    function renderHeroActions(actions) {
        return (actions || []).map((action) => {
            const style = ['section', 'download'].includes(action.style)
                ? action.style
                : 'section';
            const download = action.download ? ' download' : '';
            const icon = action.download
                ? 'bi-download'
                : (action.url || '').startsWith('#')
                    ? 'bi-arrow-down'
                    : 'bi-arrow-up-right';
            return `
                <a class="hero-action hero-action--${style}" href="${escapeHtml(action.url)}"${download}>
                    ${escapeHtml(action.label)}
                    <i class="bi ${icon}" aria-hidden="true"></i>
                </a>
            `;
        }).join('');
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
                <div class="hero-actions">
                    ${renderHeroActions(block.actions)}
                </div>
                <div class="social-links">
                    ${renderIconLinks(block.links)}
                </div>
            </div>
            <a href="#about" class="scroll-down-arrow" aria-label="Scroll down">
                <span class="scroll-down-label">About · Projects · Notes</span>
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

    function markSectionReady(section) {
        if (!section) return;
        section.dataset.sectionReady = 'true';
        section.dispatchEvent(new root.CustomEvent('portfolio:section-ready'));
    }

    function initSectionScrollHandoff() {
        const sections = Array.from(root.document.querySelectorAll('main > section[id]'));
        const page = root.document.documentElement;
        if (sections.length < 2 || page.dataset.sectionScrollBound === 'true') {
            return;
        }

        page.dataset.sectionScrollBound = 'true';
        const reducedMotion = root.matchMedia('(prefers-reduced-motion: reduce)').matches;
        let accumulatedWheelDelta = 0;
        let wheelDirection = 0;
        let wheelResetTimer;
        let wheelReleaseTimer;
        let touchStartY = null;
        let touchGestureLocked = false;
        let isSnapping = false;
        let wheelGestureLocked = false;
        let activeWheelTransition = '';
        let carriedWheelDelta = 0;
        let carriedWheelEvents = 0;
        let portfolioBoundaryHeld = false;
        const portfolioSection = root.document.getElementById('portfolio');
        const blogSection = root.document.getElementById('blog');

        const delay = (milliseconds) => new Promise(resolve => root.setTimeout(resolve, milliseconds));
        const getPortfolioEndAnchor = () => {
            if (!portfolioSection || !blogSection) return null;
            return Math.max(portfolioSection.offsetTop, blogSection.offsetTop - root.innerHeight);
        };
        const getCurrentSectionIndex = () => {
            const position = root.scrollY + 2;
            let currentIndex = 0;
            sections.forEach((section, index) => {
                if (section.offsetTop <= position) currentIndex = index;
            });
            return currentIndex;
        };
        const getSnapTarget = (direction) => {
            const currentIndex = getCurrentSectionIndex();
            const current = sections[currentIndex];
            const currentTop = current.offsetTop;
            const currentBottom = currentTop + current.offsetHeight;
            const next = sections[currentIndex + 1];
            const snapSized = current.offsetHeight <= root.innerHeight * 1.2;
            const betweenSnapAnchors = next
                && snapSized
                && root.scrollY > currentTop + 48
                && root.scrollY < next.offsetTop - 2;

            if (betweenSnapAnchors) {
                return direction > 0 ? next : current;
            }

            if (direction > 0 && currentIndex < sections.length - 1) {
                const reachesSectionEnd = root.scrollY + root.innerHeight >= currentBottom - 12;
                const fitsViewport = current.offsetHeight <= root.innerHeight + 24;
                const alignedAtStart = Math.abs(root.scrollY - currentTop) <= 48;
                return reachesSectionEnd || (fitsViewport && alignedAtStart)
                    ? sections[currentIndex + 1]
                    : null;
            }

            if (direction < 0 && currentIndex > 0) {
                const previous = sections[currentIndex - 1];
                const atSectionStart = root.scrollY <= currentTop + 48;
                const previousIsSnapSized = previous.offsetHeight <= root.innerHeight * 1.2;
                return atSectionStart && previousIsSnapSized ? previous : null;
            }

            return null;
        };
        const waitForSectionReady = (section) => {
            if (section.dataset.sectionReady !== 'false') {
                return Promise.resolve();
            }

            return new Promise(resolve => {
                const finish = () => {
                    root.clearTimeout(timeoutId);
                    section.removeEventListener('portfolio:section-ready', finish);
                    resolve();
                };
                const timeoutId = root.setTimeout(finish, 560);
                section.addEventListener('portfolio:section-ready', finish, { once: true });
            });
        };
        const prepareSection = async (section) => {
            const wasReady = section.dataset.sectionReady !== 'false';
            await waitForSectionReady(section);

            const imageDecodes = Array.from(section.querySelectorAll('img'))
                .slice(0, 8)
                .map(image => {
                    if (typeof image.decode === 'function') {
                        return image.decode().catch(() => undefined);
                    }
                    return Promise.resolve();
                });
            await Promise.race([Promise.all(imageDecodes), delay(240)]);
            return wasReady;
        };
        const animateScrollToPosition = (getTargetY, duration) => new Promise(resolve => {
            const startY = root.scrollY;
            const distance = getTargetY() - startY;
            if (duration === 0 || Math.abs(distance) < 1) {
                root.scrollTo(0, getTargetY());
                resolve();
                return;
            }

            const startedAt = root.performance.now();
            const step = (timestamp) => {
                const progress = Math.min((timestamp - startedAt) / duration, 1);
                const eased = progress < 0.5
                    ? 4 * progress * progress * progress
                    : 1 - Math.pow(-2 * progress + 2, 3) / 2;
                const liveDistance = getTargetY() - startY;
                root.scrollTo(0, startY + liveDistance * eased);
                if (progress < 1) {
                    root.requestAnimationFrame(step);
                    return;
                }
                root.scrollTo(0, getTargetY());
                resolve();
            };
            root.requestAnimationFrame(step);
        });
        const warmAdjacentSections = (section) => {
            const index = sections.indexOf(section);
            const warm = () => {
                [sections[index - 1], sections[index + 1]]
                    .filter(Boolean)
                    .forEach(adjacent => prepareSection(adjacent));
            };
            if (typeof root.requestIdleCallback === 'function') {
                root.requestIdleCallback(warm, { timeout: 700 });
            } else {
                root.setTimeout(warm, 80);
            }
        };
        const releaseWheelGesture = () => {
            root.clearTimeout(wheelReleaseTimer);
            wheelReleaseTimer = root.setTimeout(() => {
                wheelGestureLocked = false;
            }, 90);
        };
        const snapToPosition = async (section, getTargetY, preferredDuration) => {
            if (!section || isSnapping) return;

            isSnapping = true;
            wheelGestureLocked = true;
            root.clearTimeout(wheelReleaseTimer);
            page.classList.add('section-transitioning');
            try {
                const wasReady = await prepareSection(section);
                const duration = reducedMotion
                    ? 0
                    : (preferredDuration || (wasReady ? 720 : 940));
                await animateScrollToPosition(getTargetY, duration);
                warmAdjacentSections(section);
            } finally {
                page.classList.remove('section-transitioning');
                isSnapping = false;
                releaseWheelGesture();
            }
        };
        const snapToSection = (section, preferredDuration) => {
            return snapToPosition(section, () => section.offsetTop, preferredDuration);
        };
        const snapToPortfolioEnd = () => {
            return snapToPosition(
                portfolioSection,
                () => getPortfolioEndAnchor() ?? root.scrollY,
                360
            );
        };
        const holdAtPortfolioEnd = async () => {
            activeWheelTransition = 'hold-portfolio-end';
            carriedWheelDelta = 0;
            carriedWheelEvents = 0;
            await snapToPortfolioEnd();

            const continueIntoBlog = carriedWheelEvents >= 4 && carriedWheelDelta >= 180;
            activeWheelTransition = '';
            carriedWheelDelta = 0;
            carriedWheelEvents = 0;
            if (!continueIntoBlog) return;

            await delay(70);
            portfolioBoundaryHeld = false;
            activeWheelTransition = 'enter-blog';
            await snapToSection(blogSection, 480);
            activeWheelTransition = '';
        };
        const getPortfolioBoundaryAction = (direction, deltaY) => {
            const endAnchor = getPortfolioEndAnchor();
            if (endAnchor === null || !blogSection || !direction) return null;

            const position = root.scrollY;
            const blogTop = blogSection.offsetTop;
            const projectedPosition = position + deltaY;
            const tolerance = 4;

            if (portfolioBoundaryHeld && position < endAnchor - 48) {
                portfolioBoundaryHeld = false;
            }

            if (direction > 0
                && position < blogTop - tolerance
                && projectedPosition >= endAnchor - tolerance) {
                if (portfolioBoundaryHeld && position >= endAnchor - tolerance) {
                    return { type: 'enter-blog', threshold: 72 };
                }
                return { type: 'hold-portfolio-end', threshold: 18 };
            }

            if (direction < 0
                && position >= blogTop - tolerance
                && projectedPosition <= blogTop + 24) {
                return { type: 'return-portfolio-end', threshold: 18 };
            }

            if (direction < 0 && position <= endAnchor + tolerance) {
                portfolioBoundaryHeld = false;
            }

            return null;
        };

        root.addEventListener('wheel', (event) => {
            if (isSnapping || wheelGestureLocked) {
                event.preventDefault();
                if (isSnapping
                    && activeWheelTransition === 'hold-portfolio-end'
                    && event.deltaY > 0) {
                    carriedWheelDelta += Math.abs(event.deltaY);
                    carriedWheelEvents += 1;
                }
                return;
            }

            const direction = Math.sign(event.deltaY);
            const boundaryAction = getPortfolioBoundaryAction(direction, event.deltaY);
            if (boundaryAction) {
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
                }, 220);

                if (accumulatedWheelDelta >= boundaryAction.threshold) {
                    accumulatedWheelDelta = 0;
                    if (boundaryAction.type === 'hold-portfolio-end') {
                        portfolioBoundaryHeld = true;
                        holdAtPortfolioEnd();
                    } else if (boundaryAction.type === 'enter-blog') {
                        portfolioBoundaryHeld = false;
                        activeWheelTransition = 'enter-blog';
                        snapToSection(blogSection, 480).finally(() => {
                            activeWheelTransition = '';
                        });
                    } else {
                        portfolioBoundaryHeld = false;
                        activeWheelTransition = 'return-portfolio-end';
                        snapToPortfolioEnd().finally(() => {
                            activeWheelTransition = '';
                        });
                    }
                }
                return;
            }

            const target = direction ? getSnapTarget(direction) : null;
            if (!target) {
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
                snapToSection(target);
            }
        }, { passive: false });

        root.addEventListener('touchstart', (event) => {
            touchStartY = event.touches[0]?.clientY ?? null;
            if (isSnapping) touchGestureLocked = true;
        }, { passive: true });

        root.addEventListener('touchmove', (event) => {
            if (isSnapping || touchGestureLocked) {
                event.preventDefault();
                return;
            }
            if (touchStartY === null) return;

            const currentY = event.touches[0]?.clientY;
            if (typeof currentY !== 'number') return;

            const movement = touchStartY - currentY;
            if (Math.abs(movement) < 24) return;

            const target = getSnapTarget(Math.sign(movement));
            if (target) {
                event.preventDefault();
                touchStartY = null;
                touchGestureLocked = true;
                snapToSection(target);
            }
        }, { passive: false });

        root.addEventListener('touchend', () => {
            touchStartY = null;
            touchGestureLocked = false;
        }, { passive: true });

        root.addEventListener('touchcancel', () => {
            touchStartY = null;
            touchGestureLocked = false;
        }, { passive: true });

        root.addEventListener('keydown', (event) => {
            const targetElement = event.target;
            const isEditable = targetElement instanceof root.HTMLElement
                && (targetElement.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(targetElement.tagName));
            if (isEditable || isSnapping || event.repeat) return;

            const isDown = ['ArrowDown', 'PageDown'].includes(event.key)
                || (event.key === ' ' && !event.shiftKey);
            const isUp = ['ArrowUp', 'PageUp'].includes(event.key)
                || (event.key === ' ' && event.shiftKey);
            const direction = isDown ? 1 : (isUp ? -1 : 0);
            const target = direction ? getSnapTarget(direction) : null;
            if (target) {
                event.preventDefault();
                snapToSection(target);
            }
        });

        warmAdjacentSections(sections[0]);
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

            initSectionScrollHandoff();
            if (!renderedBlock) {
                return;
            }

            if (root.AOS) {
                root.AOS.refresh();
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

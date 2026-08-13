const fs = require('fs');
const path = require('path');

const { chromium } = require('playwright');

const { SITE_URL } = require('../blogs/lib/site-config');
const { loadSiteData } = require('../blogs/lib/site-data');
const { getPostRoute } = require('../blogs/lib/site-routes');
const { getLaunchOptions, startStaticServer } = require('./check-rendered-site');

const repoRoot = path.join(__dirname, '..');
const responsiveWidths = {
    portfolio: [390, 767, 768, 1199, 1200, 1440, 2560],
    blog: [390, 767, 768, 991, 992, 1199, 1200, 1440, 2560],
    compact: [390, 768, 1200, 2560],
    utility: [390, 1200, 2560]
};

function parseArguments(argv) {
    const options = {
        baseUrl: '',
        route: '',
        screenshotsDir: '',
        width: 0
    };

    argv.forEach((argument) => {
        if (argument === '--production') {
            options.baseUrl = SITE_URL;
        } else if (argument.startsWith('--base-url=')) {
            options.baseUrl = argument.slice('--base-url='.length).replace(/\/+$/, '');
        } else if (argument.startsWith('--route=')) {
            options.route = argument.slice('--route='.length);
        } else if (argument.startsWith('--screenshots-dir=')) {
            options.screenshotsDir = path.resolve(argument.slice('--screenshots-dir='.length));
        } else if (argument.startsWith('--width=')) {
            options.width = Number.parseInt(argument.slice('--width='.length), 10);
        }
    });

    if (options.width && (!Number.isInteger(options.width) || options.width < 320)) {
        throw new Error('--width must be an integer of at least 320.');
    }

    return options;
}

function requirePost(siteData, id) {
    const post = siteData.posts.find((entry) => entry.id === id);
    if (!post) {
        throw new Error(`Representative post ${id} is missing from site-data.json.`);
    }
    return post;
}

function createCases(siteData) {
    return [
        {
            id: 'portfolio',
            path: '/',
            type: 'portfolio',
            coreSelectors: ['#home', '#about', '#portfolio-projects', '#portfolio-blog-posts'],
            widths: responsiveWidths.portfolio
        },
        {
            id: 'blog-home',
            path: '/blogs/',
            type: 'blog-home',
            coreSelectors: ['.blog-home-topbar', '.blog-home-hero', '.blog-home-archive', '.blog-home-tabs'],
            widths: responsiveWidths.blog
        },
        {
            id: 'blog-search',
            path: '/blogs/search/?q=3d',
            type: 'blog-search',
            coreSelectors: ['.blog-search-page', '#search-results-container'],
            widths: responsiveWidths.blog
        },
        {
            id: 'blog-archive',
            path: '/blogs/series/3d-generation/',
            type: 'blog-archive',
            archiveContext: 'Series Index',
            coreSelectors: ['.blog-archive-page', '.blog-home-archive', '.blog-home-tab-content'],
            widths: responsiveWidths.blog
        },
        {
            id: 'blog-tag-archive',
            path: '/blogs/tags/3d-generation/',
            type: 'blog-archive',
            archiveContext: 'Topic Index',
            coreSelectors: ['.blog-archive-page', '.blog-home-archive', '.blog-home-tab-content'],
            widths: [390, 1440]
        },
        {
            id: 'post',
            path: getPostRoute(requirePost(siteData, '250823_sdf'), 'eng'),
            type: 'post',
            coreSelectors: ['#mainNav', '.masthead', '.main-content'],
            widths: [320, ...responsiveWidths.blog]
        },
        {
            id: 'technical-title-post',
            path: getPostRoute(requirePost(siteData, '250702_building_large_3d_1'), 'eng'),
            type: 'post',
            coreSelectors: ['#mainNav', '.masthead', '.main-content', '.post-title-term'],
            widths: [390, 1440]
        },
        {
            id: 'disclosure-post',
            path: getPostRoute(requirePost(siteData, '240917_3djs'), 'eng'),
            type: 'disclosure-post',
            coreSelectors: ['#mainNav', '.masthead', '.main-content', '.code-disclosure-summary'],
            widths: [390, 991, 992, 1440]
        },
        {
            id: 'embedded-viewer-post',
            path: getPostRoute(requirePost(siteData, '250310_model_viewer'), 'eng'),
            type: 'post',
            coreSelectors: ['#mainNav', '.masthead', '.main-content', '.main-content simple-model-viewer'],
            widths: [390, 1200, 1440]
        },
        {
            id: 'lightbox-post',
            path: getPostRoute(requirePost(siteData, '240823_grt'), 'eng'),
            type: 'lightbox-post',
            coreSelectors: ['#mainNav', '.masthead', '.main-content'],
            widths: [390, 1440]
        },
        {
            id: 'nested-toc-fourier',
            path: getPostRoute(requirePost(siteData, '211128_fourier'), 'eng'),
            type: 'nested-toc-post',
            coreSelectors: ['#mainNav', '.masthead', '.main-content'],
            tocParent: '2. Background',
            tocTarget: '2.1. Kernel Trick',
            widths: [1440]
        },
        {
            id: 'nested-toc-nerf-game',
            path: getPostRoute(requirePost(siteData, '231130_nerf_in_game'), 'eng'),
            type: 'nested-toc-post',
            coreSelectors: ['#mainNav', '.masthead', '.main-content'],
            tocParent: '2. NeRF & Illumination Control',
            tocTarget: "NeRF's Inability to Separate Lighting Effects",
            widths: [1440]
        },
        {
            id: 'project',
            path: '/projects/varco3d/',
            type: 'project',
            coreSelectors: ['main', '.portfolio-details'],
            widths: responsiveWidths.portfolio
        },
        {
            id: 'viewer',
            path: '/blogs/3DViewer/',
            type: 'viewer',
            coreSelectors: ['simple-model-viewer'],
            widths: responsiveWidths.blog
        },
        {
            id: 'editor',
            path: '/blogs/editor/',
            type: 'editor',
            coreSelectors: ['.editor-shell', '#workspace', '.editor-stage'],
            widths: responsiveWidths.utility
        }
    ];
}

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function listFiles(directory) {
    return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const absolutePath = path.join(directory, entry.name);
        return entry.isDirectory() ? listFiles(absolutePath) : [absolutePath];
    });
}

function assertSiteIconSourceConsistency() {
    const spritePath = path.join(repoRoot, 'assets', 'icons', 'site-icons.svg');
    const sprite = fs.readFileSync(spritePath, 'utf8');
    const symbolIds = Array.from(sprite.matchAll(/<symbol\s+id="([^"]+)"/g), (match) => match[1]);
    const uniqueSymbolIds = new Set(symbolIds);
    assert(symbolIds.length > 0, 'The site icon sprite has no symbols.');
    assert(uniqueSymbolIds.size === symbolIds.length, 'The site icon sprite has duplicate symbol ids.');

    const generatedRoot = path.join(repoRoot, 'blogs', 'dist');
    const referencedIds = new Set();
    listFiles(generatedRoot)
        .filter((file) => /\.(?:html|js)$/.test(file))
        .forEach((file) => {
            const source = fs.readFileSync(file, 'utf8');
            Array.from(source.matchAll(/#(icon-[a-z0-9-]+)/g), (match) => match[1])
                .forEach((id) => referencedIds.add(id));
        });
    const missingIds = Array.from(referencedIds).filter((id) => !uniqueSymbolIds.has(id));
    assert(missingIds.length === 0, `Generated pages reference missing site icon symbols: ${missingIds.join(', ')}`);
}

function isScreenshotFontRequest(requestUrl) {
    try {
        const hostname = new URL(requestUrl).hostname;
        return hostname === 'fonts.googleapis.com' || hostname === 'fonts.gstatic.com';
    } catch (error) {
        return false;
    }
}

function isEditorDependency(request) {
    try {
        const url = new URL(request.url());
        return request.frame().url().includes('/blogs/editor/')
            && request.resourceType() === 'script'
            && ['cdn.jsdelivr.net', 'cdnjs.cloudflare.com'].includes(url.hostname);
    } catch (error) {
        return false;
    }
}

async function configureNetwork(context, baseUrl, options) {
    const siteOrigin = new URL(baseUrl).origin;

    await context.route('**/*', async (route) => {
        const request = route.request();
        const requestUrl = request.url();
        const requestOrigin = new URL(requestUrl).origin;

        if (requestOrigin === siteOrigin) {
            await route.continue();
            return;
        }
        if (isEditorDependency(request) || (options.screenshotsDir && isScreenshotFontRequest(requestUrl))) {
            await route.continue();
            return;
        }
        await route.abort();
    });
}

function viewportHeight(width) {
    if (width <= 430) {
        return 844;
    }
    if (width < 1200) {
        return 900;
    }
    return 1000;
}

async function waitForCase(page, testCase) {
    const selector = testCase.type === 'viewer'
        ? 'simple-model-viewer'
        : testCase.coreSelectors[testCase.coreSelectors.length - 1];
    await page.waitForSelector(selector, { state: 'attached', timeout: 45000 });

    if (['portfolio', 'blog-home', 'blog-search'].includes(testCase.type)) {
        const dynamicSelector = {
            portfolio: '#portfolio-projects .portfolio-project-layout',
            'blog-home': '.blog-home-archive .post-preview',
            'blog-search': '#search-results-container .post-preview'
        }[testCase.type];
        await page.waitForSelector(dynamicSelector, { state: 'attached', timeout: 15000 });
    }

    if (testCase.type === 'viewer') {
        await page.waitForFunction(() => {
            const viewer = document.querySelector('simple-model-viewer');
            const canvas = viewer?.shadowRoot?.querySelector('canvas');
            return Boolean(canvas && canvas.width > 0 && canvas.height > 0);
        }, null, { timeout: 45000 });
    }

    await page.waitForTimeout(80);
}

async function inspectLayout(page, testCase, width) {
    return page.evaluate(({ coreSelectors, pageType, viewportWidth }) => {
        const issues = [];
        const visible = (element) => {
            const styles = getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return styles.display !== 'none'
                && styles.visibility !== 'hidden'
                && Number.parseFloat(styles.opacity || '1') > 0
                && rect.width > 0
                && rect.height > 0;
        };
        const describe = (element) => {
            if (!element) {
                return 'unknown';
            }
            if (element.id) {
                return `#${element.id}`;
            }
            const className = String(element.className || '').trim().split(/\s+/).slice(0, 2).join('.');
            return `${element.tagName.toLowerCase()}${className ? `.${className}` : ''}`;
        };

        if (document.documentElement.scrollWidth > document.documentElement.clientWidth + 1) {
            const overflowing = Array.from(document.querySelectorAll('body *'))
                .filter((element) => {
                    const styles = getComputedStyle(element);
                    const rect = element.getBoundingClientRect();
                    return styles.position !== 'fixed'
                        && rect.width > 0
                        && (rect.left < -1 || rect.right > viewportWidth + 1);
                })
                .slice(0, 4)
                .map(describe);
            issues.push(`document has horizontal overflow${overflowing.length ? ` (${overflowing.join(', ')})` : ''}`);
        }

        coreSelectors.forEach((selector) => {
            const element = document.querySelector(selector);
            if (!element) {
                issues.push(`${selector} is missing`);
                return;
            }
            if (!visible(element)) {
                issues.push(`${selector} is not visibly rendered`);
                return;
            }
            const rect = element.getBoundingClientRect();
            if (rect.left < -2 || rect.right > viewportWidth + 2) {
                issues.push(`${selector} exceeds the viewport (${Math.round(rect.left)}..${Math.round(rect.right)})`);
            }
        });

        document.querySelectorAll('.main-content simple-model-viewer').forEach((viewer, index) => {
            const content = viewer.closest('.main-content');
            const contentRect = content?.getBoundingClientRect();
            const viewerRect = viewer.getBoundingClientRect();
            if (contentRect
                && (viewerRect.left < contentRect.left - 1 || viewerRect.right > contentRect.right + 1)) {
                issues.push(`embedded viewer ${index} escapes the article column`);
            }
        });

        document.querySelectorAll('.visually-hidden').forEach((element) => {
            const styles = getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            const clipped = styles.clip !== 'auto' || styles.clipPath !== 'none';
            if (styles.position !== 'absolute'
                || styles.overflow !== 'hidden'
                || rect.width > 2
                || rect.height > 2
                || !clipped) {
                issues.push(`${describe(element)} is not visually hidden`);
            }
        });

        const pairs = [
            ['#portfolio-projects .portfolio-project-layout', '.portfolio-project-cover', '.portfolio-project-body'],
            ['#portfolio-blog-posts .portfolio-blog-preview-item', '.portfolio-blog-preview-cover', '.portfolio-blog-preview-body'],
            ['.post-preview .post-card-link', '.post-card-cover', '.post-card-body']
        ];
        pairs.forEach(([containerSelector, mediaSelector, textSelector]) => {
            document.querySelectorAll(containerSelector).forEach((container, index) => {
                const media = container.querySelector(mediaSelector);
                const text = container.querySelector(textSelector);
                if (!media || !text) {
                    return;
                }
                const first = media.getBoundingClientRect();
                const second = text.getBoundingClientRect();
                const overlapWidth = Math.min(first.right, second.right) - Math.max(first.left, second.left);
                const overlapHeight = Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top);
                if (overlapWidth > 1 && overlapHeight > 1) {
                    issues.push(`${containerSelector}[${index}] media overlaps its text`);
                }
            });
        });

        const header = document.querySelector('#header.header');
        const sharedContent = pageType === 'viewer'
            ? document.querySelector('simple-model-viewer')
            : document.querySelector('.header ~ main');
        if (header && sharedContent) {
            const headerRect = header.getBoundingClientRect();
            const contentRect = sharedContent.getBoundingClientRect();
            const headerVisible = visible(header) && !header.classList.contains('sidebar-auto-hidden');
            if (viewportWidth >= 1200 && headerVisible && contentRect.left < headerRect.right - 2) {
                issues.push(`sidebar overlaps main content (${Math.round(headerRect.right)} > ${Math.round(contentRect.left)})`);
            }
            if (viewportWidth < 1200 && !header.classList.contains('header-show') && headerRect.right > 2) {
                issues.push(`closed mobile sidebar remains in the viewport (${Math.round(headerRect.right)}px)`);
            }
        }

        return issues;
    }, {
        coreSelectors: testCase.coreSelectors,
        pageType: testCase.type,
        viewportWidth: width
    });
}

async function assertSiteIcons(page, testCase) {
    const state = await page.evaluate((pageType) => {
        const iconRoots = [document, ...Array.from(document.querySelectorAll('*'))
            .map((element) => element.shadowRoot)
            .filter(Boolean)];
        const icons = iconRoots.flatMap((root) => Array.from(root.querySelectorAll('svg.site-icon')));
        const visibleIcons = icons.filter((icon) => {
            const rect = icon.getBoundingClientRect();
            const styles = getComputedStyle(icon);
            return styles.display !== 'none'
                && styles.visibility !== 'hidden'
                && rect.width > 0
                && rect.height > 0;
        });
        const invalidHref = icons
            .map((icon) => icon.querySelector('use')?.getAttribute('href') || '')
            .filter((href) => !/^\/assets\/icons\/site-icons\.svg#icon-[a-z0-9-]+$/.test(href));
        const invalidGeometry = visibleIcons.map((icon) => {
            const rect = icon.getBoundingClientRect();
            return {
                className: icon.getAttribute('class'),
                height: Math.round(rect.height * 100) / 100,
                href: icon.querySelector('use')?.getAttribute('href'),
                width: Math.round(rect.width * 100) / 100
            };
        }).filter((icon) => icon.width < 8 || icon.height < 8 || icon.width > 48 || icon.height > 48);
        const emptyGeometry = visibleIcons.filter((icon) => {
            try {
                const box = icon.getBBox();
                return box.width <= 0 || box.height <= 0;
            } catch (error) {
                return true;
            }
        }).length;
        const transparentIcons = visibleIcons.filter((icon) => {
            const fill = getComputedStyle(icon).fill;
            return fill === 'none' || fill === 'rgba(0, 0, 0, 0)';
        }).length;
        const keySelector = {
            portfolio: '#navmenu .navicon',
            project: '#navmenu .navicon',
            'blog-home': '.blog-home-search .site-icon',
            'blog-search': '.blog-home-search .site-icon',
            'blog-archive': '.blog-home-search .site-icon',
            post: '[data-theme-toggle] .site-icon',
            'disclosure-post': '[data-theme-toggle] .site-icon',
            'lightbox-post': '[data-theme-toggle] .site-icon',
            'nested-toc-post': '[data-theme-toggle] .site-icon',
            viewer: '#navmenu .navicon',
            editor: '.toolbar-actions .site-icon'
        }[pageType];

        return {
            emptyGeometry,
            helperAvailable: Boolean(window.SiteIcons?.render && window.SiteIcons?.set),
            iconCount: icons.length,
            invalidGeometry,
            invalidHref,
            keyPresent: Boolean(keySelector && document.querySelector(keySelector)),
            transparentIcons,
            visibleCount: visibleIcons.length
        };
    }, testCase.type);

    assert(state.helperAvailable, `${testCase.id} did not load the site icon helper.`);
    assert(state.iconCount > 0 && state.visibleCount > 0, `${testCase.id} rendered no visible site icons.`);
    assert(state.keyPresent, `${testCase.id} is missing its key site icon.`);
    assert(state.invalidHref.length === 0, `${testCase.id} has invalid site icon hrefs: ${state.invalidHref.join(', ')}`);
    assert(state.invalidGeometry.length === 0, `${testCase.id} has site icons outside the expected square geometry: ${JSON.stringify(state.invalidGeometry.slice(0, 4))}`);
    assert(state.emptyGeometry === 0, `${testCase.id} has unresolved or empty SVG use geometry.`);
    assert(state.transparentIcons === 0, `${testCase.id} has transparent site icons.`);
}

async function prepareScreenshot(page) {
    await page.evaluate(async () => {
        document.querySelectorAll('img').forEach((image) => {
            image.loading = 'eager';
            if (!image.getAttribute('src') && image.dataset.src) {
                image.src = image.dataset.src;
            }
        });
        await document.fonts?.ready;
        const images = Array.from(document.images);
        await Promise.all(images.map((image) => {
            if (image.complete) {
                return image.decode?.().catch(() => null);
            }
            return new Promise((resolve) => {
                image.addEventListener('load', resolve, { once: true });
                image.addEventListener('error', resolve, { once: true });
                window.setTimeout(resolve, 5000);
            });
        }));
        window.scrollTo(0, 0);
    });
}

async function captureScreenshot(page, testCase, width, state, options) {
    if (!options.screenshotsDir) {
        return;
    }
    await prepareScreenshot(page);
    const filename = `${testCase.id}-${width}-${state}.png`;
    await page.screenshot({
        animations: 'disabled',
        caret: 'hide',
        fullPage: true,
        path: path.join(options.screenshotsDir, filename)
    });
}

async function assertBlogTabs(page) {
    const tabs = [
        ['#posts-tab-control', '#posts-tab'],
        ['#series-tab-control', '#series-tab']
    ];

    for (const [controlSelector, paneSelector] of tabs) {
        await page.locator(controlSelector).click();
        await page.waitForFunction(({ control, pane }) => {
            return document.querySelector(control)?.getAttribute('aria-selected') === 'true'
                && document.querySelector(control)?.classList.contains('is-active')
                && document.querySelector(control)?.getAttribute('tabindex') === '0'
                && document.querySelector(pane)?.classList.contains('is-active')
                && !document.querySelector(pane)?.hidden;
        }, { control: controlSelector, pane: paneSelector });

        const inactiveState = await page.evaluate(({ activeControl, activePane }) => {
            return Array.from(document.querySelectorAll('[role="tab"]')).every((tab) => {
                if (tab.matches(activeControl)) {
                    return true;
                }
                const target = tab.getAttribute('data-tab-target');
                const pane = target ? document.querySelector(target) : null;
                return tab.getAttribute('aria-selected') === 'false'
                    && tab.getAttribute('tabindex') === '-1'
                    && !tab.classList.contains('is-active')
                    && (!pane || (!pane.classList.contains('is-active') && pane.hidden));
            }) && Boolean(document.querySelector(activePane));
        }, { activeControl: controlSelector, activePane: paneSelector });
        assert(inactiveState, `${controlSelector} did not deactivate the other tabs and panes.`);
    }

    await page.locator('#series-tab-control').focus();
    await page.keyboard.press('Home');
    assert(await page.locator('#posts-tab-control').getAttribute('aria-selected') === 'true', 'Home did not select the first blog tab.');
    await page.keyboard.press('End');
    assert(await page.locator('#series-tab-control').getAttribute('aria-selected') === 'true', 'End did not select the last blog tab.');
    await page.keyboard.press('ArrowRight');
    assert(await page.locator('#posts-tab-control').getAttribute('aria-selected') === 'true', 'ArrowRight did not wrap blog tab selection.');
    await page.keyboard.press('ArrowLeft');
    assert(await page.locator('#series-tab-control').getAttribute('aria-selected') === 'true', 'ArrowLeft did not wrap blog tab selection.');
}

async function assertBlogHomeInteractions(page, testCase, width, options) {
    if (![390, 1440].includes(width)) {
        if (width <= 767) {
            const scrollTopDisplay = await page.locator('#scroll-top').evaluate((element) => getComputedStyle(element).display);
            assert(scrollTopDisplay === 'none', 'Mobile blog scroll-to-top control should remain hidden below 768px.');
        }
        return;
    }

    await assertBlogTabs(page);
    await captureScreenshot(page, testCase, width, 'series-tab', options);
    await page.locator('#posts-tab-control').click();

    if (width !== 1440) {
        const compactHeroState = await page.evaluate(() => {
            const hero = document.querySelector('.blog-editorial-hero');
            const title = hero?.querySelector('h1');
            const deck = hero?.querySelector('.blog-home-hero-deck');
            const visual = hero?.querySelector('.blog-editorial-visual');
            const leftVisual = hero?.querySelector('.blog-editorial-visual-left');
            const featureLabel = document.querySelector('.blog-feature-label');
            const byline = hero?.querySelector('.blog-publication-byline');
            return {
                backgroundImage: hero ? getComputedStyle(hero).backgroundImage : '',
                heroHeight: hero?.getBoundingClientRect().height || Number.POSITIVE_INFINITY,
                visualDisplay: visual ? getComputedStyle(visual).display : '',
                leftVisualDisplay: leftVisual ? getComputedStyle(leftVisual).display : '',
                featureOpeningGap: featureLabel && hero
                    ? featureLabel.getBoundingClientRect().top - hero.getBoundingClientRect().bottom
                    : Number.NEGATIVE_INFINITY,
                standfirstToHeroEdge: hero && deck
                    ? (() => {
                        const range = document.createRange();
                        range.selectNodeContents(deck);
                        return hero.getBoundingClientRect().bottom - range.getBoundingClientRect().bottom;
                    })()
                    : Number.NEGATIVE_INFINITY,
                deckBelowTitle: title && deck
                    ? deck.getBoundingClientRect().top >= title.getBoundingClientRect().bottom
                    : false,
                bylineBelowDeck: byline && deck
                    ? byline.getBoundingClientRect().top >= deck.getBoundingClientRect().bottom
                    : false,
                bylineLeftAligned: byline && hero
                    ? Math.abs(byline.getBoundingClientRect().left - hero.querySelector('.blog-home-hero-copy').getBoundingClientRect().left) <= 0.5
                    : false
            };
        });
        assert(compactHeroState.backgroundImage === 'none', 'Compact Blog hero restored a photographic background.');
        assert(compactHeroState.heroHeight <= 400, `Compact Blog hero grew beyond 400px: ${compactHeroState.heroHeight}px.`);
        assert(compactHeroState.featureOpeningGap >= 30, `Compact Featured begins too close to the Hero: ${compactHeroState.featureOpeningGap}px.`);
        assert(
            Math.abs(compactHeroState.featureOpeningGap - compactHeroState.standfirstToHeroEdge) <= 1,
            'Compact Blog hero transition spacing is no longer balanced around the Hero edge.'
        );
        assert(
            compactHeroState.visualDisplay === 'none' && compactHeroState.leftVisualDisplay === 'none',
            'Compact Blog hero retained the wide-screen technical edge crops.'
        );
        assert(compactHeroState.deckBelowTitle, 'Compact Blog hero did not stack the standfirst beneath the title.');
        assert(compactHeroState.bylineBelowDeck, 'Compact Blog hero byline no longer follows the standfirst.');
        assert(compactHeroState.bylineLeftAligned, 'Compact Blog hero byline is not left-aligned to the copy measure.');
        const compactUtilityState = await page.evaluate(() => {
            const search = document.querySelector('.blog-home-search');
            const searchButton = search?.querySelector('button[type="submit"]');
            const sidebarToggle = document.querySelector('.sidebar-mobile-toggle');
            const searchRect = search?.getBoundingClientRect();
            return {
                searchExpanded: Boolean(search?.classList.contains('is-expanded')),
                searchButtonExpanded: searchButton?.getAttribute('aria-expanded'),
                searchRight: searchRect?.right || 0,
                searchWidth: searchRect?.width || Number.POSITIVE_INFINITY,
                sidebarToggleVisible: sidebarToggle ? getComputedStyle(sidebarToggle).display !== 'none' : false
            };
        });
        assert(!compactUtilityState.sidebarToggleVisible, 'Compact Blog restored the conflicting sidebar hamburger.');
        assert(
            !compactUtilityState.searchExpanded
                && compactUtilityState.searchButtonExpanded === 'false'
                && compactUtilityState.searchWidth <= 36,
            'Compact Blog search did not begin as an icon-only control.'
        );
        await page.locator('.blog-home-search button[type="submit"]').click();
        await page.waitForFunction(() => {
            const search = document.querySelector('.blog-home-search');
            return search?.classList.contains('is-expanded') && search.getBoundingClientRect().width > 100;
        });
        const expandedUtilityState = await page.evaluate(() => {
            const search = document.querySelector('.blog-home-search');
            const searchRect = search?.getBoundingClientRect();
            return {
                buttonExpanded: search?.querySelector('button[type="submit"]')?.getAttribute('aria-expanded'),
                inputFocused: document.activeElement === search?.querySelector('input'),
                right: searchRect?.right || 0,
                width: searchRect?.width || 0
            };
        });
        assert(
            expandedUtilityState.buttonExpanded === 'true'
                && expandedUtilityState.inputFocused
                && expandedUtilityState.width > 100
                && Math.abs(expandedUtilityState.right - compactUtilityState.searchRight) <= 1,
            'Compact Blog search did not expand toward the left and focus its input.'
        );
        await page.locator('.blog-home-search input').press('Escape');
        await page.waitForFunction(() => !document.querySelector('.blog-home-search')?.classList.contains('is-expanded'));
        const compactArchiveOrder = await page.locator('#posts-tab .post-preview').first().evaluate((card) => {
            const cover = card.querySelector('.post-card-cover')?.getBoundingClientRect();
            const copy = card.querySelector('.post-card-body')?.getBoundingClientRect();
            return cover && copy ? cover.bottom <= copy.top + 1 : false;
        });
        assert(compactArchiveOrder, 'Compact Archive rows did not reset to media-first stacking.');
        return;
    }

    const initialTitleState = await page.evaluate(async () => {
        const heroTitle = document.querySelector('.blog-home-hero-copy h1');
        const heroIntro = document.querySelector('.blog-home-hero-deck');
        const hero = document.querySelector('.blog-editorial-hero');
        const heroCopy = document.querySelector('.blog-home-hero-copy');
        const heroIntroStyle = getComputedStyle(heroIntro);
        const heroIntroRange = document.createRange();
        heroIntroRange.selectNodeContents(heroIntro);
        const heroTitleRange = document.createRange();
        heroTitleRange.selectNodeContents(heroTitle);
        const heroVisual = document.querySelector('.blog-editorial-visual');
        const heroLeftVisual = document.querySelector('.blog-editorial-visual-left');
        const heroByline = document.querySelector('.blog-publication-byline');
        const siteData = await fetch('/blogs/data/site-data.json').then((response) => response.json());
        const featuredPostId = siteData.blogHome?.featuredPostId || '';
        const featuredPost = siteData.posts.find((post) => post.id === featuredPostId);
        const archiveStartPostId = siteData.blogHome?.archiveStartPostId || '';
        const archiveStartPost = siteData.posts.find((post) => post.id === archiveStartPostId);
        const archiveStartDate = archiveStartPost?.date || '';
        const cardIsFromArchive = (card) => {
            const post = siteData.posts.find((entry) => entry.id === card.dataset.postId);
            return Boolean(post && archiveStartDate && post.date <= archiveStartDate);
        };
        const featureCover = document.querySelector('.blog-feature-cover');
        const featureCard = document.querySelector('.blog-feature-card');
        const featureCopy = document.querySelector('.blog-feature-copy');
        const featureTitle = featureCopy?.querySelector('h2');
        const featureSubtitle = featureCopy?.querySelector('.blog-feature-subtitle');
        const featureDate = featureCopy?.querySelector('.blog-feature-date');
        const featureRead = featureCopy?.querySelector('.blog-feature-read');
        const featureLabel = document.querySelector('.blog-feature-label');
        const firstPreview = document.querySelector('.post-preview[data-post-id]');
        const previewSubtitle = firstPreview?.querySelector('.post-subtitle');
        const previewDate = firstPreview?.querySelector('.post-meta');
        const previewTags = firstPreview?.querySelector('.post-tag-row');
        const postArchiveCards = Array.from(document.querySelectorAll('#posts-tab .post-preview'));
        return {
            configuredFeaturedLanguage: featuredPost?.languages?.includes('kor') ? 'ko' : 'en',
            configuredFeaturedPostId: featuredPostId,
            configuredArchiveStartPostId: archiveStartPostId,
            configuredPostCount: siteData.posts.filter((post) => (post.status || 'published') === 'published').length,
            featuredPostId: document.querySelector('.blog-feature-card')?.dataset.postId || '',
            heroIntroFontFamily: heroIntroStyle.fontFamily,
            heroIntroLetterSpacing: heroIntroStyle.letterSpacing,
            heroIntroText: heroIntro.textContent,
            heroIntroWidth: heroIntroRange.getBoundingClientRect().width,
            heroIntroLineCount: heroIntroRange.getClientRects().length,
            heroIsStatic: !heroTitle?.hasAttribute('data-i18n') && !heroIntro?.hasAttribute('data-i18n'),
            heroTitle: heroTitle?.textContent || '',
            heroTitleColor: heroTitle ? getComputedStyle(heroTitle).color : '',
            heroWidth: heroTitle?.getBoundingClientRect().width || 0,
            heroSearchClearsTitle: heroTitle && document.querySelector('.blog-home-search')
                ? document.querySelector('.blog-home-search').getBoundingClientRect().left
                    >= heroTitleRange.getBoundingClientRect().right
                : false,
            heroBackgroundImage: hero ? getComputedStyle(hero).backgroundImage : '',
            heroHeight: hero?.getBoundingClientRect().height || Number.POSITIVE_INFINITY,
            heroFeaturedMeasureMatches: heroCopy && featureCard
                ? Math.abs(heroCopy.getBoundingClientRect().left - featureCard.getBoundingClientRect().left) <= 0.5
                    && Math.abs(heroCopy.getBoundingClientRect().right - featureCard.getBoundingClientRect().right) <= 0.5
                : false,
            heroImprint: document.querySelector('.blog-editorial-imprint')?.textContent.replace(/\s+/g, ' ').trim() || '',
            heroByline: heroByline?.textContent.replace(/\s+/g, ' ').trim() || '',
            articlesChapter: document.querySelector('#blog-home-archive-title')?.textContent.replace(/\s+/g, ' ').trim() || '',
            heroTitleLineCount: (() => {
                if (!heroTitle) return 0;
                const range = document.createRange();
                range.selectNodeContents(heroTitle);
                return range.getClientRects().length;
            })(),
            heroVisualBackgroundImage: heroVisual ? getComputedStyle(heroVisual).backgroundImage : '',
            heroVisualDisplay: heroVisual ? getComputedStyle(heroVisual).display : '',
            heroLeftVisualBackgroundImage: heroLeftVisual ? getComputedStyle(heroLeftVisual).backgroundImage : '',
            heroLeftVisualDisplay: heroLeftVisual ? getComputedStyle(heroLeftVisual).display : '',
            heroOverlayBackground: hero ? getComputedStyle(hero, '::before').backgroundImage : '',
            heroVisualsFillBackground: heroLeftVisual && heroVisual && hero && heroCopy
                ? Math.abs(heroVisual.getBoundingClientRect().left - hero.getBoundingClientRect().left) <= 0.5
                    && Math.abs(heroLeftVisual.getBoundingClientRect().right - heroCopy.getBoundingClientRect().right) <= 0.5
                    && Math.abs(heroVisual.getBoundingClientRect().width - heroLeftVisual.getBoundingClientRect().width) <= 0.5
                    && heroVisual.getBoundingClientRect().right <= heroLeftVisual.getBoundingClientRect().left + 0.5
                : false,
            heroDeckIsBelowTitle: heroTitle && heroIntro
                ? heroIntro.getBoundingClientRect().top >= heroTitle.getBoundingClientRect().bottom
                : false,
            heroBylineFollowsDeck: heroIntro && heroByline
                ? heroByline.getBoundingClientRect().top >= heroIntro.getBoundingClientRect().bottom
                    && Math.abs(heroByline.getBoundingClientRect().right - heroCopy.getBoundingClientRect().right) <= 0.5
                : false,
            tabCounts: [
                document.querySelector('#posts-count')?.textContent.trim(),
                document.querySelector('#series-count')?.textContent.trim()
            ],
            tabControlCount: document.querySelectorAll('.blog-home-tab[role="tab"]').length,
            hasNotesTab: Boolean(document.querySelector('#notes-tab, #notes-tab-control')),
            featureStartsAfterHero: featureCover && hero
                ? featureCover.getBoundingClientRect().top >= hero.getBoundingClientRect().bottom
                : false,
            featureOpeningGap: featureLabel && hero
                ? featureLabel.getBoundingClientRect().top - hero.getBoundingClientRect().bottom
                : Number.NEGATIVE_INFINITY,
            standfirstToHeroEdge: heroIntro && hero
                ? hero.getBoundingClientRect().bottom - heroIntroRange.getBoundingClientRect().bottom
                : Number.NEGATIVE_INFINITY,
            featureCopyOverflow: featureCopy && featureCover
                ? featureCopy.getBoundingClientRect().height - featureCover.getBoundingClientRect().height
                : Number.POSITIVE_INFINITY,
            featureReadBottomDelta: featureRead && featureCover
                ? featureRead.getBoundingClientRect().bottom - featureCover.getBoundingClientRect().bottom
                : Number.POSITIVE_INFINITY,
            featureSubtitleClamp: featureSubtitle ? getComputedStyle(featureSubtitle).webkitLineClamp : '',
            featureTitleClamp: featureTitle ? getComputedStyle(featureTitle).webkitLineClamp : '',
            hasArchiveReadAction: Boolean(document.querySelector('.post-preview .post-card-read')),
            previewDateFontFamily: previewDate ? getComputedStyle(previewDate).fontFamily : '',
            featureDateFontFamily: featureDate ? getComputedStyle(featureDate).fontFamily : '',
            previewMetadataOrder: Boolean(
                previewSubtitle && previewDate && previewTags
                && (previewSubtitle.compareDocumentPosition(previewDate) & Node.DOCUMENT_POSITION_FOLLOWING)
                && (previewDate.compareDocumentPosition(previewTags) & Node.DOCUMENT_POSITION_FOLLOWING)
            ),
            archiveEraSignatureMatches: postArchiveCards.every((card) => {
                return card.classList.contains('post-preview-media-right') === !cardIsFromArchive(card);
            }),
            archiveEraBreaks: [
                document.querySelector('#posts-tab')
            ].map((panel) => {
                const cards = Array.from(panel.querySelectorAll('.post-preview[data-post-id]'));
                const marker = panel.querySelector('.blog-home-era-break');
                const firstFromArchive = cards.find(cardIsFromArchive);
                const lastCurrent = cards.findLast((card) => !cardIsFromArchive(card));
                return {
                    count: panel.querySelectorAll('.blog-home-era-break').length,
                    firstArchivePostId: firstFromArchive?.dataset.postId || '',
                    text: marker?.textContent.replace(/\s+/g, ' ').trim() || '',
                    precedingBorderWidth: marker?.previousElementSibling
                        ? getComputedStyle(marker.previousElementSibling).borderBottomWidth
                        : '',
                    betweenEras: Boolean(
                        marker && firstFromArchive && lastCurrent
                        && (lastCurrent.compareDocumentPosition(marker) & Node.DOCUMENT_POSITION_FOLLOWING)
                        && (marker.compareDocumentPosition(firstFromArchive) & Node.DOCUMENT_POSITION_FOLLOWING)
                    )
                };
            }),
            postArchiveEraGeometry: [
                postArchiveCards.find((card) => !cardIsFromArchive(card)),
                postArchiveCards.find(cardIsFromArchive)
            ].map((card) => {
                const cover = card.querySelector('.post-card-cover')?.getBoundingClientRect();
                const copy = card.querySelector('.post-card-body')?.getBoundingClientRect();
                return cover && copy ? cover.left > copy.left : null;
            }),
            previewSeriesLabel: firstPreview?.querySelector('.post-series-context-label')?.textContent.trim() || '',
            postTitles: Object.fromEntries(Array.from(document.querySelectorAll('.post-preview[data-post-id]')).map((card) => {
                const title = card.querySelector('.post-title');
                const style = getComputedStyle(title);
                return [card.dataset.postId, {
                    fontFamily: style.fontFamily,
                    letterSpacing: style.letterSpacing,
                    text: title.textContent.trim()
                }];
            }))
        };
    });
    assert(initialTitleState.heroIsStatic, 'Blog hero title or intro returned to runtime-managed copy.');
    assert(initialTitleState.heroBackgroundImage === 'none', 'Blog home restored a template-style photographic hero background.');
    assert(initialTitleState.heroHeight <= 330.5, `Wide Blog hero grew beyond its former 330px footprint: ${initialTitleState.heroHeight}px.`);
    assert(initialTitleState.heroFeaturedMeasureMatches, 'Blog Hero copy no longer shares the Featured content measure.');
    assert(initialTitleState.heroImprint === '00 / Technical Writing 2021—2026', 'Blog editorial-cover subject imprint is missing or changed.');
    assert(initialTitleState.heroByline === 'Written by / Hwan Heo', 'Blog editorial-cover author byline is missing or changed.');
    assert(initialTitleState.articlesChapter === '02 Articles', 'Blog Home writing chapter is no longer labeled 02 Articles.');
    assert(initialTitleState.heroTitleLineCount === 1, 'Wide Blog editorial-cover title is no longer a single line.');
    assert(initialTitleState.heroSearchClearsTitle, 'Wide Blog utility search overlaps the Hero title field.');
    assert(
        initialTitleState.heroTitleColor === 'rgba(240, 241, 243, 0.88)',
        'Blog Hero title no longer shares the Portfolio Hero name color.'
    );
    assert(initialTitleState.heroIntroLineCount === 1, 'Wide Blog editorial-cover standfirst is no longer a single line.');
    assert(
        initialTitleState.heroVisualDisplay === 'block'
            && initialTitleState.heroVisualBackgroundImage.includes('sparse-pipeline.png')
            && initialTitleState.heroLeftVisualDisplay === 'block'
            && initialTitleState.heroLeftVisualBackgroundImage.includes('remote-d832175607cd.png'),
        'Wide Blog editorial-cover technical edge crops are missing.'
    );
    assert(initialTitleState.heroVisualsFillBackground, 'Blog editorial-cover images no longer split the left-to-copy-edge background stage.');
    assert(initialTitleState.heroOverlayBackground.includes('linear-gradient'), 'Blog editorial-cover images lost their dark contrast overlay.');
    assert(initialTitleState.heroDeckIsBelowTitle, 'Wide Blog hero standfirst no longer sits beneath its title.');
    assert(initialTitleState.heroBylineFollowsDeck, 'Wide Blog hero byline no longer follows the standfirst on a right-aligned line.');
    assert(
        Number(initialTitleState.tabCounts[0]) === initialTitleState.configuredPostCount,
        'Blog Articles tab count excludes the configured Featured post.'
    );
    assert(
        initialTitleState.tabControlCount === 2 && !initialTitleState.hasNotesTab,
        'Blog home must expose only the Posts and Series tabs.'
    );
    assert(initialTitleState.featureStartsAfterHero, 'Featured media moved into or behind the image-free editorial cover.');
    assert(initialTitleState.featureOpeningGap >= 56, `Featured begins too close to the Hero: ${initialTitleState.featureOpeningGap}px.`);
    assert(
        Math.abs(initialTitleState.featureOpeningGap - initialTitleState.standfirstToHeroEdge) <= 0.5,
        'Blog Hero transition spacing is no longer equal on both sides of the Hero edge.'
    );
    assert(
        initialTitleState.configuredFeaturedPostId
            && initialTitleState.featuredPostId === initialTitleState.configuredFeaturedPostId,
        'Blog home did not render the explicitly configured featured post.'
    );
    assert(initialTitleState.featureCopyOverflow <= 0.5, `Featured copy exceeds its cover by ${initialTitleState.featureCopyOverflow}px.`);
    assert(
        Math.abs(initialTitleState.featureReadBottomDelta) <= 0.5,
        `Featured Read post does not close on the cover edge: ${initialTitleState.featureReadBottomDelta}px.`
    );
    assert(
        ['none', ''].includes(initialTitleState.featureTitleClamp)
            && ['none', ''].includes(initialTitleState.featureSubtitleClamp),
        'Featured title or subtitle restored clipping.'
    );
    assert(!initialTitleState.hasArchiveReadAction, 'Archive rows restored the redundant Read post action.');
    assert(initialTitleState.previewSeriesLabel === 'SERIES', 'Archive rows lost the explicit series hierarchy.');
    assert(initialTitleState.previewMetadataOrder, 'Archive row date and tag hierarchy is out of order.');
    assert(initialTitleState.archiveEraSignatureMatches, 'Archive media sides no longer follow the configured post boundary.');
    assert(
        initialTitleState.archiveEraBreaks.every((entry) => (
            entry.count === 1
                && entry.text === '03 From the Archive'
                && entry.precedingBorderWidth === '0px'
                && entry.betweenEras
        )),
        'Posts lost the single 03 From the Archive boundary at the media-side transition.'
    );
    assert(
        initialTitleState.configuredArchiveStartPostId
            && initialTitleState.archiveEraBreaks[0]?.firstArchivePostId
                === initialTitleState.configuredArchiveStartPostId,
        'The Posts archive does not begin with the configured archive-start post.'
    );
    assert(
        JSON.stringify(initialTitleState.postArchiveEraGeometry) === JSON.stringify([true, false]),
        'Archive era classes do not match the rendered media positions.'
    );
    assert(
        initialTitleState.previewDateFontFamily === initialTitleState.featureDateFontFamily,
        'Featured and archive dates no longer share a font role.'
    );
    await page.evaluate(() => {
        window.__blogStaticPreview = document.querySelector('.post-preview[data-post-id]');
    });
    await page.locator('#lang-toggle-main').click();
    const koreanState = await page.evaluate((initialPostTitles) => {
        const stableTitleIssues = Array.from(document.querySelectorAll('.post-preview[data-post-id]')).flatMap((card) => {
            const title = card.querySelector('.post-title');
            const initial = initialPostTitles[card.dataset.postId];
            if (!initial || initial.text !== title.textContent.trim()) {
                return [];
            }
            const style = getComputedStyle(title);
            return style.fontFamily === initial.fontFamily && style.letterSpacing === initial.letterSpacing
                ? []
                : [card.dataset.postId];
        });
        const heroIntro = document.querySelector('.blog-home-hero-deck');
        const heroIntroStyle = getComputedStyle(heroIntro);
        const heroIntroRange = document.createRange();
        heroIntroRange.selectNodeContents(heroIntro);
        const featureSubtitle = document.querySelector('.blog-feature-subtitle');
        const featureTitle = document.querySelector('.blog-feature-copy h2');
        return {
            button: document.querySelector('#lang-toggle-main')?.textContent.trim(),
            preservedStaticPreview: window.__blogStaticPreview === document.querySelector('.post-preview[data-post-id]'),
            lang: document.documentElement.lang,
            title: document.querySelector('.blog-home-hero-copy h1')?.textContent,
            featureLang: document.querySelector('.blog-feature-copy h2')?.lang,
            featureSubtitleWrap: featureSubtitle ? getComputedStyle(featureSubtitle).textWrap : '',
            featureTitleWrap: featureTitle ? getComputedStyle(featureTitle).textWrap : '',
            heroIntroFontFamily: heroIntroStyle.fontFamily,
            heroIntroLetterSpacing: heroIntroStyle.letterSpacing,
            heroIntroText: heroIntro.textContent,
            heroIntroWidth: heroIntroRange.getBoundingClientRect().width,
            heroWidth: document.querySelector('.blog-home-hero-copy h1')?.getBoundingClientRect().width || 0,
            stableTitleIssues
        };
    }, initialTitleState.postTitles);
    assert(koreanState.lang === 'ko' && koreanState.button === 'A', 'Blog language toggle did not switch to Korean.');
    assert(koreanState.preservedStaticPreview, 'Blog language toggle replaced the server-rendered preview markup.');
    assert(koreanState.title && initialTitleState.heroTitle, 'Blog language toggle left the hero copy empty.');
    assert(
        koreanState.featureLang === initialTitleState.configuredFeaturedLanguage,
        'Blog language toggle used the wrong language for the configured featured title.'
    );
    assert(koreanState.featureSubtitleWrap === 'balance', 'Korean Featured subtitle lost balanced wrapping.');
    assert(koreanState.featureTitleWrap === 'balance', 'Korean Featured title lost balanced wrapping.');
    assert(koreanState.heroIntroText === initialTitleState.heroIntroText, 'Blog language toggle changed the fixed English hero intro.');
    assert(koreanState.heroIntroFontFamily === initialTitleState.heroIntroFontFamily, 'Blog language toggle changed the hero-intro font.');
    assert(koreanState.heroIntroLetterSpacing === initialTitleState.heroIntroLetterSpacing, 'Blog language toggle changed the hero-intro tracking.');
    assert(Math.abs(koreanState.heroIntroWidth - initialTitleState.heroIntroWidth) < 0.5, 'Blog language toggle changed the hero-intro text width.');
    assert(koreanState.stableTitleIssues.length === 0, `Blog language toggle restyled unchanged titles: ${koreanState.stableTitleIssues.join(', ')}`);
    assert(Math.abs(koreanState.heroWidth - initialTitleState.heroWidth) < 0.5, 'Blog language toggle changed the fixed English hero-title width.');
    await page.locator('#lang-toggle-main').click();

    const themeToggle = page.locator('[data-theme-toggle]').first();
    await themeToggle.click();
    const darkState = await page.evaluate(() => ({
        icon: document.querySelector('[data-theme-toggle] .site-icon use')?.getAttribute('href'),
        pressed: document.querySelector('[data-theme-toggle]')?.getAttribute('aria-pressed'),
        theme: document.documentElement.dataset.theme,
        featureTitleColor: getComputedStyle(document.querySelector('.blog-feature-copy h2 a')).color,
        featureTitleParentColor: getComputedStyle(document.querySelector('.blog-feature-copy h2')).color,
        archiveTitleColor: getComputedStyle(document.querySelector('.post-preview .post-title a')).color,
        archiveTitleParentColor: getComputedStyle(document.querySelector('.post-preview .post-title')).color
    }));
    assert(darkState.theme === 'dark' && darkState.pressed === 'true', 'Blog theme toggle did not expose its dark ARIA state.');
    assert(darkState.icon?.endsWith('#icon-sun'), 'Blog theme toggle did not switch to the sun icon.');
    assert(
        darkState.featureTitleColor === darkState.featureTitleParentColor
            && darkState.archiveTitleColor === darkState.archiveTitleParentColor,
        'Dark Blog title links no longer inherit the light editorial title color.'
    );
    await captureScreenshot(page, testCase, width, 'dark-theme', options);
    await themeToggle.click();

    await page.evaluate(() => window.scrollTo(0, Math.min(700, document.documentElement.scrollHeight - innerHeight)));
    await page.waitForFunction(() => document.querySelector('#scroll-top')?.classList.contains('active'));
    await page.locator('#scroll-top').click();
    await page.waitForFunction(() => window.scrollY < 2, null, { timeout: 5000 });
}

async function assertPostNavigation(page, width) {
    const state = await page.evaluate(() => {
        const bodyStyle = getComputedStyle(document.body);
        const masthead = document.querySelector('.post-masthead');
        const mastheadStyle = getComputedStyle(masthead);
        const sidebar = document.querySelector('#header');
        const sidebarToggle = document.querySelector('.sidebar-mobile-toggle');
        const mainContent = document.querySelector('.main-content');
        const introDeck = document.querySelector('.post-intro-deck');
        const articleInfo = document.querySelector('.post-article-info');
        const heroColumn = document.querySelector('.post-intro-column');
        const title = document.querySelector('.post-masthead h1');
        const seriesKicker = document.querySelector('.post-hero-series');
        const seriesLink = document.querySelector('.post-hero-series a');
        const heroTopic = document.querySelector('.post-intro-topics .post-tag');
        const author = document.querySelector('.post-author-note');
        const authorLinks = Array.from(document.querySelectorAll('.post-author-links a'))
            .map((link) => link.textContent.trim());
        const relatedPosts = document.querySelector('.related-posts');
        const searchFormItem = document.querySelector('.post-nav-search-item');
        const searchLinkItem = document.querySelector('.post-nav-search-link-item');
        const searchForm = document.querySelector('.post-nav-search');
        const searchButton = searchForm?.querySelector('button[type="submit"]');
        const postHomeLink = document.querySelector('.post-nav-home');
        const detailKeys = Array.from(document.querySelectorAll('[data-post-detail]'))
            .map((element) => element.dataset.postDetail);
        const detailAuthor = document.querySelector('[data-post-detail="author"] span')?.textContent.trim() || '';
        const introRect = introDeck?.getBoundingClientRect();
        const infoRect = articleInfo?.getBoundingClientRect();
        const titleStyle = getComputedStyle(title);
        const topicAfterDeck = Boolean(
            introDeck && heroTopic
            && (introDeck.compareDocumentPosition(heroTopic) & Node.DOCUMENT_POSITION_FOLLOWING)
        );
        return {
            authorAfterContent: Boolean(mainContent && author && (mainContent.compareDocumentPosition(author) & Node.DOCUMENT_POSITION_FOLLOWING)),
            authorLinks,
            breadcrumbsPresent: Boolean(document.querySelector('.breadcrumbs')),
            detailKeys,
            detailAuthor,
            hasArticleDetailsHeading: Boolean(articleInfo?.querySelector('h2')),
            hasLegacyHeroSummary: Boolean(document.querySelector('.post-hero-summary')),
            figureCaptionsValid: Array.from(document.querySelectorAll('.main-content figure figcaption'))
                .every((caption, index) => {
                    const figure = caption.closest('figure');
                    const figureIndex = caption.querySelector('.post-figure-index');
                    const figureCopy = caption.querySelector('.post-figure-caption');
                    const expectedIndex = `FIG. ${String(index + 1).padStart(2, '0')}`;
                    return figure?.classList.contains('post-media')
                        && figure.classList.contains('post-media--editorial')
                        && figureIndex?.textContent.trim() === expectedIndex
                        && Boolean(figureCopy?.textContent.trim());
                }),
            narrowFigureCaptionsConstrained: Array.from(document.querySelectorAll('.main-content figure.post-media--narrow'))
                .every((figure) => {
                    const caption = figure.querySelector('figcaption');
                    const mediaWidth = getComputedStyle(figure).getPropertyValue('--post-figure-media-width').trim();
                    return Boolean(caption && mediaWidth)
                        && caption.getBoundingClientRect().width < figure.getBoundingClientRect().width - 0.5;
                }),
            codeBlocksValid: Array.from(document.querySelectorAll('.main-content pre'))
                .filter((pre) => pre.querySelector(':scope > code'))
                .every((pre) => {
                    const label = pre.dataset.codeLabel || '';
                    const renderedLabel = getComputedStyle(pre, '::before').content.replace(/^['"]|['"]$/g, '');
                    return pre.classList.contains('post-code-block')
                        && label.startsWith('CODE')
                        && renderedLabel === label;
                }),
            heroWidth: heroColumn?.getBoundingClientRect().width || 0,
            infoBelowDeck: Boolean(introRect && infoRect && infoRect.top >= introRect.bottom),
            infoRightOfDeck: Boolean(introRect && infoRect && infoRect.left >= introRect.right),
            introDeckText: introDeck?.textContent.trim() || '',
            introHasAccentRule: getComputedStyle(introDeck, '::before').content !== 'none',
            languageVisible: getComputedStyle(document.querySelector('[data-language-target]')).display !== 'none',
            mainWidth: mainContent?.getBoundingClientRect().width || 0,
            mastheadBackground: mastheadStyle.backgroundColor,
            mastheadBorderBottom: mastheadStyle.borderBottomWidth,
            pageBackground: bodyStyle.backgroundColor,
            postHomeVisible: Boolean(postHomeLink && getComputedStyle(postHomeLink).display !== 'none'),
            relatedPostsBorderTop: relatedPosts ? getComputedStyle(relatedPosts).borderTopWidth : '0px',
            searchButtonExpanded: searchButton?.getAttribute('aria-expanded'),
            searchFormExpanded: Boolean(searchForm?.classList.contains('is-expanded')),
            searchFormVisible: Boolean(searchFormItem && getComputedStyle(searchFormItem).display !== 'none'),
            searchFormWidth: searchForm?.getBoundingClientRect().width || 0,
            searchLinkVisible: Boolean(searchLinkItem && getComputedStyle(searchLinkItem).display !== 'none'),
            seriesEndPresent: Boolean(document.querySelector('#series-container, .post-end-series')),
            seriesHasArrow: Boolean(seriesLink?.querySelector('.post-hero-series-arrow')),
            seriesKickerText: seriesKicker?.textContent.trim() || '',
            seriesLinkHref: seriesLink?.getAttribute('href') || '',
            seriesLinkDecoration: seriesLink ? getComputedStyle(seriesLink).textDecorationLine : '',
            sidebarRight: sidebar?.getBoundingClientRect().right || 0,
            sidebarToggleVisible: getComputedStyle(sidebarToggle).display !== 'none',
            themeVisible: getComputedStyle(document.querySelector('[data-theme-toggle]')).display !== 'none',
            topicAfterDeck,
            topicColor: heroTopic ? getComputedStyle(heroTopic).color : '',
            topicDecoration: heroTopic ? getComputedStyle(heroTopic).textDecorationLine : '',
            topicHref: heroTopic?.getAttribute('href') || '',
            topicText: heroTopic?.textContent.trim() || '',
            titleColor: titleStyle.color,
            titleFontFamily: titleStyle.fontFamily,
            titleFontWeight: titleStyle.fontWeight,
            titleLetterSpacing: titleStyle.letterSpacing,
            titleTermsUnbroken: Array.from(title.querySelectorAll('.post-title-term'))
                .every((term) => getComputedStyle(term).whiteSpace === 'nowrap' && term.getClientRects().length === 1),
            titleTextWrap: titleStyle.textWrap
        };
    });

    if (width <= 767) {
        assert(
            state.searchFormVisible
                && !state.searchLinkVisible
                && !state.searchFormExpanded
                && state.searchButtonExpanded === 'false'
                && state.searchFormWidth <= 34,
            'Compact post search did not begin as an inline expandable icon.'
        );
    } else {
        assert(
            !state.searchLinkVisible && state.searchFormVisible && state.searchFormWidth > 100,
            'Wide post utilities did not expose the search field.'
        );
    }
    assert(state.themeVisible && state.languageVisible, 'Theme or language controls are missing from the post utility row.');
    assert(state.mastheadBackground === state.pageBackground, 'Post masthead no longer shares the article background.');
    assert(state.mastheadBorderBottom === '0px', 'Post masthead restored an unwanted bottom rule.');
    assert(!state.breadcrumbsPresent, 'Post breadcrumbs should not interrupt the article opening.');
    assert(!state.hasLegacyHeroSummary && state.introDeckText, 'Post subtitle is not in the editorial opening deck.');
    assert(!state.introHasAccentRule, 'Post standfirst restored the decorative accent rule.');
    assert(state.figureCaptionsValid, 'Post figure captions no longer follow the sequential editorial folio grammar.');
    assert(state.narrowFigureCaptionsConstrained, 'Narrow post media no longer keeps its figure caption on a related measure.');
    assert(state.codeBlocksValid, 'Post code blocks no longer expose their editorial code folio.');
    assert(state.titleTermsUnbroken, 'Hyphenated technical terms split across post-title lines.');
    assert(state.seriesKickerText.startsWith('Series') && state.seriesLinkHref.startsWith('/blogs/series/'), 'Post series hierarchy or link is missing above the title.');
    assert(state.seriesHasArrow && state.seriesLinkDecoration === 'none', 'Post series link lost its quiet arrow cue or restored a persistent underline.');
    assert(
        state.topicText
            && state.topicHref.startsWith('/blogs/tags/')
            && state.topicColor !== state.titleColor
            && state.topicDecoration === 'none',
        'Linked post topics no longer use the underline-free semantic accent treatment.'
    );
    assert(state.topicAfterDeck, 'Post topics are not positioned below the subtitle.');
    assert(!state.seriesEndPresent, 'Post page restored the duplicated end-of-article series block.');
    assert(
        !state.hasArticleDetailsHeading
            && ['author', 'published', 'reading'].every((key) => state.detailKeys.includes(key))
            && state.detailAuthor === 'Hwan Heo',
        'Post margin note is missing its concise author, publication, or reading metadata.'
    );
    assert(
        state.titleFontFamily.includes('Manrope')
            && state.titleFontWeight === '700'
            && ['0px', 'normal'].includes(state.titleLetterSpacing)
            && state.titleTextWrap === 'balance',
        `Post title no longer matches project-detail typography: ${JSON.stringify({
            fontFamily: state.titleFontFamily,
            fontWeight: state.titleFontWeight,
            letterSpacing: state.titleLetterSpacing
        })}`
    );
    assert(state.authorAfterContent, 'Post author note is not ordered after the article.');
    assert(
        state.authorLinks.length === 2
            && state.authorLinks.includes('Email')
            && state.authorLinks.includes('LinkedIn'),
        'Post author note no longer closes with the restrained Email and LinkedIn signature.'
    );
    assert(state.relatedPostsBorderTop === '0px', 'Related posts restored the hairline below the author note.');

    if (width <= 767) {
        assert(state.infoBelowDeck, 'Compact Article Details rail did not stack below the subtitle.');
    } else {
        assert(state.infoRightOfDeck, 'Wide Article Details rail did not sit beside the subtitle.');
    }

    if (width >= 992) {
        assert(state.heroWidth > state.mainWidth + 40, 'Post opening no longer widens beyond the article reading measure.');
    }

    if (width < 1200) {
        assert(
            !state.sidebarToggleVisible && state.sidebarRight <= 2 && state.postHomeVisible,
            'Compact post utility row did not replace the sidebar trigger with Blog Home.'
        );
    } else {
        assert(
            !state.sidebarToggleVisible && state.sidebarRight > 60 && !state.postHomeVisible,
            'Desktop post sidebar or compact-only Blog Home state is incorrect.'
        );
    }
}

async function assertBlogArchiveUtilities(page, testCase, width) {
    const state = await page.evaluate(async (pageType) => {
        const back = document.querySelector('.blog-home-back');
        const search = document.querySelector('.blog-home-search');
        const count = document.querySelector('.blog-search-count');
        const hero = document.querySelector('.blog-editorial-utility-hero');
        const sectionHeading = document.querySelector('.blog-editorial-section-head h2');
        const previews = Array.from(document.querySelectorAll('.post-preview[data-post-id]'));
        const backStyle = getComputedStyle(back);
        const searchStyle = getComputedStyle(search);
        const countStyle = count ? getComputedStyle(count) : null;
        const siteData = await fetch('/blogs/data/site-data.json').then((response) => response.json());
        const archiveStartPost = siteData.posts.find((post) => (
            post.id === siteData.blogHome?.archiveStartPostId
        ));
        const archiveStartDate = archiveStartPost?.date || '';
        return {
            backBackground: backStyle.backgroundColor,
            backBorderWidth: backStyle.borderTopWidth,
            backDecoration: backStyle.textDecorationLine,
            backRadius: backStyle.borderRadius,
            countBackground: countStyle?.backgroundColor || '',
            countBorderWidth: countStyle?.borderTopWidth || '',
            countRadius: countStyle?.borderRadius || '',
            editorialHero: Boolean(hero)
                && getComputedStyle(hero).backgroundImage === 'none'
                && getComputedStyle(hero).backgroundColor === 'rgb(16, 16, 17)',
            editorialImprint: document.querySelector('.blog-editorial-imprint')?.textContent.replace(/\s+/g, ' ').trim() || '',
            editorialSectionHeading: sectionHeading?.textContent.replace(/\s+/g, ' ').trim() || '',
            previewHierarchyValid: previews.every((card) => {
                const body = card.querySelector('.post-card-body');
                const series = body?.querySelector('.post-series-context');
                const title = body?.querySelector('.post-title');
                const subtitle = body?.querySelector('.post-subtitle');
                const date = body?.querySelector(':scope > .post-meta');
                const tags = body?.querySelector('.post-tag-row');
                return Boolean(series && title && date)
                    && Boolean(series.compareDocumentPosition(title) & Node.DOCUMENT_POSITION_FOLLOWING)
                    && (!subtitle || Boolean(title.compareDocumentPosition(subtitle) & Node.DOCUMENT_POSITION_FOLLOWING))
                    && Boolean((subtitle || title).compareDocumentPosition(date) & Node.DOCUMENT_POSITION_FOLLOWING)
                    && (!tags || Boolean(date.compareDocumentPosition(tags) & Node.DOCUMENT_POSITION_FOLLOWING));
            }),
            previewEraSignatureValid: previews.every((card) => {
                const postDate = card.querySelector('time[datetime]')?.getAttribute('datetime') || '';
                const fromArchive = Boolean(archiveStartDate && postDate <= archiveStartDate);
                if (window.innerWidth <= 575) {
                    const cover = card.querySelector('.post-card-cover')?.getBoundingClientRect();
                    const body = card.querySelector('.post-card-body')?.getBoundingClientRect();
                    return Boolean(cover && body && cover.bottom <= body.top + 1);
                }
                return card.classList.contains('post-preview-media-right') === !fromArchive;
            }),
            fromArchiveFolio: document.querySelector('.blog-home-era-break')?.textContent.replace(/\s+/g, ' ').trim() || '',
            visibleLanguageField: pageType === 'blog-search'
                ? Array.from(document.querySelectorAll('#search-results-container .post-preview *'))
                    .some((element) => ['Languages:', '언어:'].some((label) => element.textContent.trim().startsWith(label)))
                : false,
            searchBackground: searchStyle.backgroundColor,
            searchBorderBottomWidth: searchStyle.borderBottomWidth,
            searchBorderTopWidth: searchStyle.borderTopWidth,
            searchRadius: searchStyle.borderRadius
        };
    }, testCase.type);

    assert(
        state.backBorderWidth === '0px'
            && state.backRadius === '0px'
            && state.backBackground === 'rgba(0, 0, 0, 0)'
            && state.backDecoration.includes('underline'),
        `${testCase.id} restored the legacy pill back control.`
    );
    assert(
        state.searchBorderTopWidth === '0px'
            && state.searchBorderBottomWidth !== '0px'
            && state.searchRadius === '0px'
            && state.searchBackground === 'rgba(0, 0, 0, 0)',
        `${testCase.id} restored the legacy pill search field.`
    );
    assert(state.editorialHero, `${testCase.id} did not adopt the dark editorial utility cover.`);
    assert(
        state.editorialImprint.startsWith(testCase.type === 'blog-search'
            ? '00 / Search Index'
            : `00 / ${testCase.archiveContext || 'Series Index'}`)
            && state.editorialImprint.endsWith("Hwan's Blog"),
        `${testCase.id} is missing its indexed Blog imprint.`
    );
    assert(
        state.editorialSectionHeading === (testCase.type === 'blog-search' ? '01 Results' : '01 Articles'),
        `${testCase.id} result chapter no longer follows the numbered editorial grammar.`
    );
    assert(state.previewHierarchyValid, `${testCase.id} previews no longer share the Blog home content hierarchy.`);
    assert(state.previewEraSignatureValid, `${testCase.id} previews no longer share the Blog home era alignment.`);
    assert(state.fromArchiveFolio === '02 From the Archive', `${testCase.id} is missing the archive boundary folio.`);
    if (testCase.type === 'blog-search') {
        assert(
            state.countBorderWidth === '0px'
                && state.countRadius === '0px'
                && state.countBackground === 'rgba(0, 0, 0, 0)',
            'Search results restored the legacy pill count.'
        );
        assert(
            !state.visibleLanguageField,
            'Search results restored visible language metadata.'
        );

        if (width === 1440) {
            const languageButton = page.locator('#lang-toggle-main');
            if (await page.locator('html').getAttribute('lang') !== 'en') {
                await languageButton.click();
                await page.waitForFunction(() => document.documentElement.lang === 'en');
            }
            await languageButton.click();
            await page.waitForFunction(() => document.documentElement.lang === 'ko');
            const koreanTargets = await page.locator('#search-results-container .post-card-cover').evaluateAll((links) => (
                links.map((link) => link.getAttribute('href') || '')
            ));
            assert(
                koreanTargets.length > 0 && koreanTargets.every((href) => href.endsWith('-kor/')),
                'Korean search results do not consistently continue into Korean articles.'
            );
            await languageButton.click();
            await page.waitForFunction(() => document.documentElement.lang === 'en');
            const englishTargets = await page.locator('#search-results-container .post-card-cover').evaluateAll((links) => (
                links.map((link) => link.getAttribute('href') || '')
            ));
            assert(
                englishTargets.length > 0 && englishTargets.every((href) => !href.endsWith('-kor/')),
                'English search results do not consistently continue into English articles.'
            );
        }
    }
}

async function assertPostInteractions(page, testCase, width, options) {
    await assertPostNavigation(page, width);

    if (width === 390) {
        const initialUrl = page.url();
        const searchForm = page.locator('#post-nav-search-form');
        const searchInput = page.locator('#post-nav-search-input');
        const searchButton = searchForm.locator('button[type="submit"]');
        await searchButton.click();
        await page.waitForFunction(() => {
            const form = document.querySelector('#post-nav-search-form');
            return form?.classList.contains('is-expanded') && form.getBoundingClientRect().width > 100;
        });
        const expandedState = await searchForm.evaluate((element) => ({
            activeInput: document.activeElement === element.querySelector('input'),
            ariaExpanded: element.querySelector('button')?.getAttribute('aria-expanded'),
            width: element.getBoundingClientRect().width
        }));
        assert(page.url() === initialUrl, 'Opening compact post search navigated away from the article.');
        assert(
            expandedState.activeInput && expandedState.ariaExpanded === 'true' && expandedState.width > 100,
            'Compact post search did not expand and focus its inline field.'
        );
        await searchInput.press('Escape');
        await page.waitForFunction(() => !document.querySelector('#post-nav-search-form')?.classList.contains('is-expanded'));
        assert(await searchButton.getAttribute('aria-expanded') === 'false', 'Escape did not collapse compact post search.');
    }

    if (width === 390 || width === 1440) {
        await page.evaluate(() => {
            const masthead = document.querySelector('.masthead');
            const headerHeight = masthead?.offsetHeight || 260;
            const pinStart = Math.max(96, Math.min(headerHeight * 0.5, headerHeight - 72));
            window.scrollTo(0, pinStart + 12);
        });
        await page.waitForFunction(() => document.querySelector('#mainNav')?.classList.contains('is-fixed'));
        const hiddenNavBottom = await page.locator('#mainNav').evaluate((element) => element.getBoundingClientRect().bottom);
        assert(hiddenNavBottom <= 1, 'Post utility row flashes into view when first entering its fixed state.');
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.waitForFunction(() => !document.querySelector('#mainNav')?.classList.contains('is-fixed'));
    }

    const languageLink = await page.locator('[data-language-target]').first().getAttribute('href');
    assert(languageLink && languageLink.endsWith('-kor/'), 'Post language link no longer points to the Korean counterpart.');

    if (width === 390) {
        const scrollTop = page.locator('#scroll-top');
        assert(await scrollTop.evaluate((element) => getComputedStyle(element).display !== 'none'), 'Post back-to-top control is hidden on mobile.');
        await page.evaluate(() => window.scrollTo(0, Math.min(700, document.documentElement.scrollHeight - innerHeight)));
        await page.waitForFunction(() => document.querySelector('#scroll-top')?.classList.contains('active'));
        await scrollTop.click();
        await page.waitForFunction(() => window.scrollY < 2, null, { timeout: 5000 });
    }

    if (width === 1440) {
        const themeToggle = page.locator('[data-theme-toggle]').first();
        await themeToggle.click();
        const themeState = await page.evaluate(() => ({
            icon: document.querySelector('[data-theme-toggle] .site-icon use')?.getAttribute('href'),
            theme: document.documentElement.dataset.theme
        }));
        assert(themeState.theme === 'dark', 'Post theme toggle did not switch to dark mode.');
        assert(themeState.icon?.endsWith('#icon-sun'), 'Post theme toggle did not switch to the sun icon.');
        await page.waitForFunction(() => {
            const body = getComputedStyle(document.body).backgroundColor;
            return body === getComputedStyle(document.querySelector('.post-masthead')).backgroundColor
                && body === getComputedStyle(document.querySelector('#mainNav')).backgroundColor;
        });
        const darkSurfaceState = await page.evaluate(() => ({
            body: getComputedStyle(document.body).backgroundColor,
            masthead: getComputedStyle(document.querySelector('.post-masthead')).backgroundColor,
            utilities: getComputedStyle(document.querySelector('#mainNav')).backgroundColor
        }));
        assert(
            darkSurfaceState.body === darkSurfaceState.masthead && darkSurfaceState.body === darkSurfaceState.utilities,
            `Dark post opening no longer reads as one continuous surface: ${JSON.stringify(darkSurfaceState)}`
        );
        await themeToggle.click();

        const copyButton = page.locator('.copy-code-button').first();
        if (await copyButton.count()) {
            await copyButton.evaluate((button) => button.click());
            await page.waitForFunction(() => document.querySelector('.copy-code-button .site-icon use')
                ?.getAttribute('href')
                ?.endsWith('#icon-check2'));
        }

        await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight / 2));
        assert(!await page.locator('.scroll-progress-bar').count(), 'Blog posts should not render the legacy scroll-progress line.');
        await page.evaluate(() => window.scrollTo(0, 0));
    }
}

async function assertNestedToc(page, testCase) {
    const initialState = await page.evaluate(({ parentText, targetText }) => {
        const links = Array.from(document.querySelectorAll('.toc a'));
        const childLink = links.find((link) => link.textContent.trim() === targetText);
        const childItem = childLink?.closest('li');
        const parentItem = childItem?.parentElement?.closest('li');
        return {
            hasParent: Boolean(parentItem),
            nestedDirectly: childItem?.parentElement?.parentElement === parentItem,
            parentMatches: parentItem?.querySelector(':scope > a')?.textContent.trim() === parentText,
            targetId: childLink?.getAttribute('href')?.slice(1) || ''
        };
    }, { parentText: testCase.tocParent, targetText: testCase.tocTarget });
    assert(
        initialState.hasParent && initialState.nestedDirectly && initialState.parentMatches,
        `${testCase.id} TOC subitem is not attached to its expected parent.`
    );
    assert(initialState.targetId, `${testCase.id} TOC target is missing.`);

    await page.evaluate((targetId) => {
        const target = document.getElementById(targetId);
        if (target) {
            window.scrollTo(0, target.getBoundingClientRect().top + window.scrollY - 180);
        }
    }, initialState.targetId);
    await page.waitForFunction((targetId) => {
        const childItem = document.querySelector(`.toc a[href="#${CSS.escape(targetId)}"]`)?.closest('li');
        const parentItem = childItem?.parentElement?.closest('li');
        return childItem?.classList.contains('is-current')
            && parentItem?.classList.contains('is-expanded')
            && childItem.parentElement.getBoundingClientRect().height > 0;
    }, initialState.targetId);
}

async function assertDisclosure(page, testCase, width, options) {
    if (![390, 1440].includes(width)) {
        return;
    }
    const disclosure = page.locator('.code-disclosure').first();
    const summary = disclosure.locator('.code-disclosure-summary');
    assert(!await disclosure.evaluate((element) => element.open), 'Code disclosure did not start collapsed.');
    await summary.focus();
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => {
        const details = document.querySelector('.code-disclosure');
        const body = details?.querySelector('.code-disclosure-body');
        return details?.open && body?.getBoundingClientRect().height > 0;
    });
    await captureScreenshot(page, testCase, width, 'disclosure-open', options);
    await summary.click();
    await page.waitForFunction(() => !document.querySelector('.code-disclosure')?.open);
}

async function assertLightbox(page, testCase, width, options) {
    const trigger = page.locator('[data-image-lightbox-target]').first();
    const targetSelector = await trigger.getAttribute('data-image-lightbox-target');
    assert(targetSelector, 'Image lightbox trigger has no target.');
    await trigger.evaluate((element) => {
        element.querySelector('img')?.setAttribute('loading', 'eager');
        element.scrollIntoView({ block: 'center' });
    });
    await page.waitForFunction(() => {
        const image = document.querySelector('[data-image-lightbox-target] img');
        return !(image instanceof HTMLImageElement) || image.naturalWidth > 0;
    }, null, { timeout: 15000 });
    await trigger.click();
    await page.waitForFunction((selector) => {
        const dialog = document.querySelector(selector);
        return dialog?.open && dialog.matches(':modal') && getComputedStyle(dialog).display !== 'none';
    }, targetSelector);

    const openState = await page.evaluate((selector) => {
        const dialog = document.querySelector(selector);
        return {
            closeFocused: document.activeElement === dialog?.querySelector('[data-image-lightbox-close]'),
            hasLabel: Boolean(dialog?.getAttribute('aria-label') || dialog?.getAttribute('aria-labelledby')),
            locked: document.documentElement.classList.contains('post-image-lightbox-open')
                && getComputedStyle(document.documentElement).overflow === 'hidden',
            position: dialog ? getComputedStyle(dialog).position : '',
            viewportSized: dialog
                ? dialog.getBoundingClientRect().width >= window.innerWidth - 1
                    && dialog.getBoundingClientRect().height >= window.innerHeight - 1
                : false
        };
    }, targetSelector);
    assert(openState.position === 'fixed' && openState.viewportSized, 'Image lightbox is not a viewport overlay.');
    assert(openState.hasLabel, 'Image lightbox has no accessible label.');
    assert(openState.closeFocused, 'Image lightbox did not move focus to its close control.');
    assert(openState.locked, 'Image lightbox did not lock background scrolling.');

    await captureScreenshot(page, testCase, width, 'lightbox-open', options);
    await page.keyboard.press('Escape');
    await page.waitForFunction((selector) => !document.querySelector(selector)?.open, targetSelector);
    assert(await trigger.evaluate((element) => document.activeElement === element), 'Escape did not restore focus to the image lightbox trigger.');

    await trigger.click();
    await page.waitForFunction((selector) => document.querySelector(selector)?.open, targetSelector);
    await page.mouse.click(2, 2);
    await page.waitForFunction((selector) => !document.querySelector(selector)?.open, targetSelector);
    assert(await trigger.evaluate((element) => document.activeElement === element), 'Backdrop click did not close the image lightbox and restore focus.');

    await trigger.click();
    await page.waitForFunction((selector) => document.querySelector(selector)?.open, targetSelector);
    await page.locator(`${targetSelector} [data-image-lightbox-close]`).click();
    await page.waitForFunction((selector) => !document.querySelector(selector)?.open, targetSelector);
    assert(await trigger.evaluate((element) => document.activeElement === element), 'Close control did not restore focus to the image lightbox trigger.');
}

async function assertSharedSidebar(page, testCase, width, options) {
    const railOnlyTypes = ['blog-home', 'post', 'viewer', 'editor'];
    const checksCompactSidebar = width === 390 && ['project', 'post'].includes(testCase.type);
    const checksDesktopSidebar = width === 1200 && (testCase.type === 'project' || railOnlyTypes.includes(testCase.type));
    if (!checksCompactSidebar && !checksDesktopSidebar) {
        return;
    }

    if (width < 1200) {
        if (testCase.type === 'post') {
            const compactState = await page.evaluate(() => ({
                homeVisible: getComputedStyle(document.querySelector('.post-nav-home')).display !== 'none',
                toggleVisible: getComputedStyle(document.querySelector('.sidebar-mobile-toggle')).display !== 'none'
            }));
            assert(compactState.homeVisible && !compactState.toggleVisible, 'Compact post restored the conflicting mobile sidebar trigger.');
            return;
        }
        const toggle = page.locator('.sidebar-mobile-toggle');
        await toggle.click();
        const openState = await page.evaluate(() => {
            const header = document.querySelector('#header');
            const nav = document.querySelector('#mainNav');
            const toggleButton = document.querySelector('.sidebar-mobile-toggle');
            return {
                expanded: toggleButton?.getAttribute('aria-expanded'),
                headerOpen: header?.classList.contains('header-show'),
                headerZ: Number.parseInt(getComputedStyle(header).zIndex, 10) || 0,
                icon: toggleButton?.querySelector('.site-icon use')?.getAttribute('href'),
                navFixed: Boolean(nav?.classList.contains('is-fixed')),
                navZ: nav ? Number.parseInt(getComputedStyle(nav).zIndex, 10) || 0 : 0,
                overlayZ: Number.parseInt(getComputedStyle(document.body, '::before').zIndex, 10) || 0,
                rootOpen: document.documentElement.classList.contains('sidebar-mobile-open'),
                toggleZ: Number.parseInt(getComputedStyle(toggleButton).zIndex, 10) || 0
            };
        });
        assert(openState.expanded === 'true' && openState.headerOpen && openState.rootOpen, `${testCase.id} mobile sidebar did not open.`);
        assert(openState.icon?.endsWith('#icon-x'), 'Shared mobile sidebar did not switch to its close icon.');
        await captureScreenshot(page, testCase, width, 'sidebar-open', options);
        await page.keyboard.press('Escape');
        const closedState = await page.evaluate(() => ({
            expanded: document.querySelector('.sidebar-mobile-toggle')?.getAttribute('aria-expanded'),
            focused: document.activeElement === document.querySelector('.sidebar-mobile-toggle'),
            headerOpen: document.querySelector('#header')?.classList.contains('header-show'),
            icon: document.querySelector('.sidebar-mobile-toggle .site-icon use')?.getAttribute('href')
        }));
        assert(closedState.expanded === 'false' && !closedState.headerOpen && closedState.focused, 'Escape did not close the mobile sidebar and restore focus.');
        assert(closedState.icon?.endsWith('#icon-list'), 'Shared mobile sidebar did not restore its menu icon.');
        return;
    }

    if (railOnlyTypes.includes(testCase.type)) {
        const railState = await page.evaluate(() => {
            const header = document.querySelector('#header');
            return {
                collapsed: document.documentElement.classList.contains('sidebar-collapsed'),
                headerWidth: header?.getBoundingClientRect().width || 0,
                collapseToggleCount: document.querySelectorAll('.sidebar-collapse-toggle').length
            };
        });
        assert(
            railState.collapsed && railState.headerWidth <= 80 && railState.collapseToggleCount === 0,
            `${testCase.id} sidebar is no longer a fixed desktop rail: ${JSON.stringify(railState)}`
        );

        const labsSummary = page.locator('.sidebar-labs-menu > summary');
        await labsSummary.click();
        const collapsedLabsState = await page.evaluate(() => {
            const header = document.querySelector('#header');
            const menu = document.querySelector('.sidebar-labs-menu');
            const panel = menu?.querySelector('.sidebar-labs-panel');
            const headerRect = header?.getBoundingClientRect();
            const panelRect = panel?.getBoundingClientRect();
            return {
                collapsed: document.documentElement.classList.contains('sidebar-collapsed'),
                headerWidth: headerRect?.width || 0,
                open: Boolean(menu?.open),
                panelDisplay: panel ? getComputedStyle(panel).display : 'none',
                panelLeft: panelRect?.left || 0,
                sidebarRight: headerRect?.right || 0
            };
        });
        assert(
            collapsedLabsState.collapsed
                && collapsedLabsState.headerWidth <= 80
                && collapsedLabsState.open
                && collapsedLabsState.panelDisplay === 'grid'
                && collapsedLabsState.panelLeft >= collapsedLabsState.sidebarRight - 1,
            `Collapsed Labs did not open as a rail overlay: ${JSON.stringify(collapsedLabsState)}`
        );
        await page.keyboard.press('Escape');
        assert(
            await page.locator('.sidebar-labs-menu').evaluate((element) => !element.open && document.activeElement === element.querySelector('summary')),
            'Escape did not close the collapsed Labs overlay and restore focus.'
        );

        if (testCase.type !== 'post') {
            return;
        }

        const contentsToggle = page.locator('.sidebar-contents-toggle');
        await contentsToggle.click();
        const contentsState = await page.evaluate(() => {
            const header = document.querySelector('#header');
            const toc = document.querySelector('.toc');
            const tocRect = toc?.getBoundingClientRect();
            return {
                expanded: document.querySelector('.sidebar-contents-toggle')?.getAttribute('aria-expanded'),
                visible: toc?.classList.contains('is-visible'),
                ariaHidden: toc?.getAttribute('aria-hidden'),
                tocLeft: tocRect?.left || 0,
                sidebarRight: header?.getBoundingClientRect().right || 0
            };
        });
        assert(
            contentsState.expanded === 'true'
                && contentsState.visible
                && contentsState.ariaHidden === 'false'
                && contentsState.tocLeft >= contentsState.sidebarRight - 1,
            `Post Contents did not open as a rail flyout: ${JSON.stringify(contentsState)}`
        );

        await page.keyboard.press('Escape');
        assert(
            await contentsToggle.evaluate((element) => element.getAttribute('aria-expanded') === 'false' && document.activeElement === element),
            'Escape did not close the Contents flyout and restore focus.'
        );
        await page.evaluate(() => window.scrollTo(0, 0));
        return;
    }

    const collapseToggle = page.locator('.sidebar-collapse-toggle');
    assert(await collapseToggle.getAttribute('aria-expanded') === 'false', 'Desktop sidebar did not honor its default collapsed preference.');
    await collapseToggle.click();
    await page.waitForFunction(() => {
        const header = document.querySelector('#header');
        const main = document.querySelector('.header ~ main');
        return !document.documentElement.classList.contains('sidebar-collapsed')
            && (header?.getBoundingClientRect().right || 0) > 250
            && (main?.getBoundingClientRect().left || 0) >= (header?.getBoundingClientRect().right || 0) - 2;
    });
    const expandedState = await page.evaluate(() => {
        const header = document.querySelector('#header');
        const main = document.querySelector('.header ~ main');
        return {
            collapsed: document.documentElement.classList.contains('sidebar-collapsed'),
            headerRight: header?.getBoundingClientRect().right || 0,
            mainLeft: main?.getBoundingClientRect().left || 0
        };
    });
    assert(!expandedState.collapsed, 'Desktop sidebar did not expand.');
    assert(expandedState.headerRight > 250 && expandedState.mainLeft >= expandedState.headerRight - 2, `Expanded sidebar collides with ${testCase.id} content.`);
    await captureScreenshot(page, testCase, width, 'sidebar-expanded', options);
    await collapseToggle.click();
    assert(await collapseToggle.getAttribute('aria-expanded') === 'false', 'Desktop sidebar did not collapse again.');
}

async function assertEditorIcons(page, width) {
    if (width !== 1200) {
        return;
    }
    const toggle = page.locator('#sidebar-toggle-button');
    const initialIcon = await page.locator('#sidebar-toggle-icon use').getAttribute('href');
    await toggle.click();
    const changedIcon = await page.locator('#sidebar-toggle-icon use').getAttribute('href');
    assert(changedIcon && changedIcon !== initialIcon, 'Editor sidebar icon did not change with its collapsed state.');
    await toggle.click();
    assert(await page.locator('#sidebar-toggle-icon use').getAttribute('href') === initialIcon, 'Editor sidebar icon did not restore its initial state.');

    await page.waitForFunction(() => document.querySelector('#blog-home-featured-post')?.options.length > 0);
    const blogHomeState = await page.evaluate(async () => {
        const siteData = await fetch('/blogs/data/site-data.json').then((response) => response.json());
        const select = document.querySelector('#blog-home-featured-post');
        const saveButton = document.querySelector('#save-blog-home-button');
        return {
            configuredFeaturedPostId: siteData.blogHome?.featuredPostId || '',
            saveDisabled: Boolean(saveButton?.disabled),
            selectedFeaturedPostId: select?.value || ''
        };
    });
    assert(
        blogHomeState.selectedFeaturedPostId === blogHomeState.configuredFeaturedPostId,
        'Blog editor did not load the configured Blog Home Featured post.'
    );
    assert(blogHomeState.saveDisabled, 'Static Blog Editor exposed an active site-data save control.');
}

async function runInteractions(page, testCase, width, options) {
    if (testCase.type === 'blog-home') {
        await assertBlogHomeInteractions(page, testCase, width, options);
    }
    if (['blog-search', 'blog-archive'].includes(testCase.type)) {
        await assertBlogArchiveUtilities(page, testCase, width);
    }
    if (['post', 'disclosure-post', 'lightbox-post'].includes(testCase.type)) {
        await assertPostInteractions(page, testCase, width, options);
    }
    if (testCase.type === 'disclosure-post') {
        await assertDisclosure(page, testCase, width, options);
    }
    if (testCase.type === 'lightbox-post') {
        await assertLightbox(page, testCase, width, options);
    }
    if (testCase.type === 'nested-toc-post') {
        await assertNestedToc(page, testCase);
    }
    if (testCase.type === 'editor') {
        await assertEditorIcons(page, width);
    }
    await assertSharedSidebar(page, testCase, width, options);
}

async function checkViewport(context, baseUrl, testCase, width, options) {
    const page = await context.newPage();
    const iconAssetIssues = [];
    page.on('requestfailed', (request) => {
        if (/site-icons\.(?:css|js|svg)/.test(request.url())) {
            iconAssetIssues.push(`${request.url()} failed: ${request.failure()?.errorText || 'unknown error'}`);
        }
    });
    page.on('response', (response) => {
        if (/site-icons\.(?:css|js|svg)/.test(response.url()) && !response.ok()) {
            iconAssetIssues.push(`${response.url()} returned ${response.status()}`);
        }
    });
    await page.setViewportSize({ width, height: viewportHeight(width) });
    try {
        const response = await page.goto(`${baseUrl}${testCase.path}`, {
            timeout: 45000,
            waitUntil: 'domcontentloaded'
        });
        assert(response?.ok(), `${testCase.path} returned ${response ? response.status() : 'no response'}.`);
        await waitForCase(page, testCase);
        assert(iconAssetIssues.length === 0, iconAssetIssues.join('\n'));

        const layoutIssues = await inspectLayout(page, testCase, width);
        assert(layoutIssues.length === 0, layoutIssues.join('\n'));
        await assertSiteIcons(page, testCase);

        await captureScreenshot(page, testCase, width, 'default', options);
        await runInteractions(page, testCase, width, options);
    } finally {
        await page.close();
    }
}

async function main() {
    const options = parseArguments(process.argv.slice(2));
    assertSiteIconSourceConsistency();
    const localServer = options.baseUrl ? null : await startStaticServer();
    const baseUrl = options.baseUrl || localServer.baseUrl;
    const browser = await chromium.launch(getLaunchOptions());
    const context = await browser.newContext({
        colorScheme: 'light',
        locale: 'en-US',
        reducedMotion: 'reduce',
        serviceWorkers: 'block',
        timezoneId: 'Asia/Seoul',
        viewport: { width: 1440, height: 1000 }
    });
    await context.addInitScript(() => {
        localStorage.setItem('blog-theme', 'light');
        localStorage.setItem('language', 'eng');
        localStorage.setItem('site-sidebar-collapsed', 'true');
    });
    await configureNetwork(context, baseUrl, options);

    if (options.screenshotsDir) {
        fs.mkdirSync(options.screenshotsDir, { recursive: true });
        console.log(`Capturing UI screenshots in ${options.screenshotsDir}`);
    }

    const cases = createCases(loadSiteData()).filter((testCase) => {
        if (!options.route) {
            return true;
        }
        return testCase.id === options.route
            || testCase.path === options.route
            || testCase.path.split('?')[0] === options.route;
    });
    if (cases.length === 0) {
        throw new Error(`No representative UI route matches ${options.route}.`);
    }

    const failures = [];
    let checks = 0;

    try {
        for (const testCase of cases) {
            const widths = options.width ? [options.width] : testCase.widths;
            for (const width of widths) {
                try {
                    await checkViewport(context, baseUrl, testCase, width, options);
                    checks += 1;
                } catch (error) {
                    failures.push({ id: testCase.id, width, message: error.message });
                }
            }
            const testedWidths = (options.width ? [options.width] : testCase.widths).join(', ');
            process.stdout.write(`CHECKED ${testCase.id} at ${testedWidths}px\n`);
        }
    } finally {
        await context.close();
        await browser.close();
        if (localServer) {
            await new Promise((resolve) => localServer.server.close(resolve));
        }
    }

    failures.forEach((failure) => {
        process.stderr.write(`FAIL ${failure.id} at ${failure.width}px\n${failure.message}\n`);
    });
    if (failures.length > 0) {
        throw new Error(`UI regression check failed at ${failures.length} viewport${failures.length === 1 ? '' : 's'}.`);
    }

    console.log(`UI regression check passed: ${checks} representative route/viewports.`);
}

main().catch((error) => {
    console.error(error.message);
    process.exit(1);
});

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
            coreSelectors: ['.blog-archive-page', '.blog-home-archive', '.blog-home-tab-content'],
            widths: responsiveWidths.blog
        },
        {
            id: 'post',
            path: getPostRoute(requirePost(siteData, '250823_sdf'), 'eng'),
            type: 'post',
            coreSelectors: ['#mainNav', '.masthead', '.main-content'],
            widths: responsiveWidths.blog
        },
        {
            id: 'disclosure-post',
            path: getPostRoute(requirePost(siteData, '240917_3djs'), 'eng'),
            type: 'disclosure-post',
            coreSelectors: ['#mainNav', '.masthead', '.main-content', '.code-disclosure-summary'],
            widths: [390, 991, 992, 1440]
        },
        {
            id: 'lightbox-post',
            path: getPostRoute(requirePost(siteData, '240823_grt'), 'eng'),
            type: 'lightbox-post',
            coreSelectors: ['#mainNav', '.masthead', '.main-content'],
            widths: [390, 1440]
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
            portfolio: '#portfolio-projects .portfolio-project-link',
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

        const pairs = [
            ['#portfolio-projects .portfolio-project-link', '.portfolio-project-cover', '.portfolio-project-body'],
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
        ['#notes-tab-control', '#notes-tab'],
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
        return;
    }

    const initialHeroTitle = await page.locator('[data-i18n="heroTitle"]').textContent();
    await page.locator('#lang-toggle-main').click();
    const koreanState = await page.evaluate(() => ({
        button: document.querySelector('#lang-toggle-main')?.textContent.trim(),
        lang: document.documentElement.lang,
        title: document.querySelector('[data-i18n="heroTitle"]')?.textContent
    }));
    assert(koreanState.lang === 'ko' && koreanState.button === 'A', 'Blog language toggle did not switch to Korean.');
    assert(koreanState.title && initialHeroTitle, 'Blog language toggle left the hero copy empty.');
    await page.locator('#lang-toggle-main').click();

    const themeToggle = page.locator('[data-theme-toggle]').first();
    await themeToggle.click();
    const darkState = await page.evaluate(() => ({
        icon: document.querySelector('[data-theme-toggle] .site-icon use')?.getAttribute('href'),
        pressed: document.querySelector('[data-theme-toggle]')?.getAttribute('aria-pressed'),
        theme: document.documentElement.dataset.theme
    }));
    assert(darkState.theme === 'dark' && darkState.pressed === 'true', 'Blog theme toggle did not expose its dark ARIA state.');
    assert(darkState.icon?.endsWith('#icon-sun'), 'Blog theme toggle did not switch to the sun icon.');
    await captureScreenshot(page, testCase, width, 'dark-theme', options);
    await themeToggle.click();

    await page.evaluate(() => window.scrollTo(0, Math.min(700, document.documentElement.scrollHeight - innerHeight)));
    await page.waitForFunction(() => document.querySelector('#scroll-top')?.classList.contains('active'));
    await page.locator('#scroll-top').click();
    await page.waitForFunction(() => window.scrollY < 2, null, { timeout: 5000 });
}

async function assertPostNavigation(page, width) {
    const state = await page.evaluate(() => {
        const toggle = document.querySelector('[data-nav-toggle]');
        const panel = document.querySelector('#postNavPanel');
        return {
            panelVisible: getComputedStyle(panel).display !== 'none',
            toggleVisible: getComputedStyle(toggle).display !== 'none'
        };
    });

    if (width <= 991) {
        assert(state.toggleVisible && !state.panelVisible, 'Mobile post navigation did not start collapsed.');
    } else {
        assert(!state.toggleVisible && state.panelVisible, 'Desktop post navigation did not remain expanded.');
    }
}

async function assertPostInteractions(page, testCase, width, options) {
    await assertPostNavigation(page, width);

    if (width === 390 || width === 991) {
        const toggle = page.locator('[data-nav-toggle]');
        await toggle.click();
        const openState = await page.evaluate(() => ({
            expanded: document.querySelector('[data-nav-toggle]')?.getAttribute('aria-expanded'),
            shown: document.querySelector('#postNavPanel')?.classList.contains('is-open'),
            visible: getComputedStyle(document.querySelector('#postNavPanel')).display !== 'none'
        }));
        assert(openState.expanded === 'true' && openState.shown && openState.visible, 'Mobile post navigation did not open.');
        await captureScreenshot(page, testCase, width, 'nav-open', options);
        await page.keyboard.press('Escape');
        const closedState = await page.evaluate(() => ({
            expanded: document.querySelector('[data-nav-toggle]')?.getAttribute('aria-expanded'),
            focused: document.activeElement === document.querySelector('[data-nav-toggle]'),
            shown: document.querySelector('#postNavPanel')?.classList.contains('is-open')
        }));
        assert(closedState.expanded === 'false' && !closedState.shown && closedState.focused, 'Escape did not close the mobile post navigation and restore focus.');
    }

    const languageLink = await page.locator('[data-language-target]').first().getAttribute('href');
    assert(languageLink && languageLink.endsWith('-kor/'), 'Post language link no longer points to the Korean counterpart.');

    if (width === 1440) {
        const themeToggle = page.locator('[data-theme-toggle]').first();
        await themeToggle.click();
        const themeState = await page.evaluate(() => ({
            icon: document.querySelector('[data-theme-toggle] .site-icon use')?.getAttribute('href'),
            theme: document.documentElement.dataset.theme
        }));
        assert(themeState.theme === 'dark', 'Post theme toggle did not switch to dark mode.');
        assert(themeState.icon?.endsWith('#icon-sun'), 'Post theme toggle did not switch to the sun icon.');
        await themeToggle.click();

        const copyButton = page.locator('.copy-code-button').first();
        if (await copyButton.count()) {
            await copyButton.evaluate((button) => button.click());
            await page.waitForFunction(() => document.querySelector('.copy-code-button .site-icon use')
                ?.getAttribute('href')
                ?.endsWith('#icon-check2'));
        }

        await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight / 2));
        await page.waitForFunction(() => Number.parseFloat(document.querySelector('.scroll-progress-fill')?.style.width || '0') > 0);
        await page.evaluate(() => window.scrollTo(0, 0));
    }
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
    if (testCase.type !== 'project' || ![390, 1200].includes(width)) {
        return;
    }

    if (width < 1200) {
        const toggle = page.locator('.sidebar-mobile-toggle');
        await toggle.click();
        const openState = await page.evaluate(() => ({
            expanded: document.querySelector('.sidebar-mobile-toggle')?.getAttribute('aria-expanded'),
            headerOpen: document.querySelector('#header')?.classList.contains('header-show'),
            icon: document.querySelector('.sidebar-mobile-toggle .site-icon use')?.getAttribute('href'),
            rootOpen: document.documentElement.classList.contains('sidebar-mobile-open')
        }));
        assert(openState.expanded === 'true' && openState.headerOpen && openState.rootOpen, 'Shared mobile sidebar did not open.');
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
    assert(expandedState.headerRight > 250 && expandedState.mainLeft >= expandedState.headerRight - 2, 'Expanded sidebar collides with project content.');
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
}

async function runInteractions(page, testCase, width, options) {
    if (testCase.type === 'blog-home') {
        await assertBlogHomeInteractions(page, testCase, width, options);
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

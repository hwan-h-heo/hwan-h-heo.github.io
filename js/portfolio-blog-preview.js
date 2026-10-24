(function(root, factory) {
    const api = factory(root);
    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    }
    if (root) {
        root.portfolioBlogPreview = api;
        root.document.addEventListener('DOMContentLoaded', api.mount);
    }
})(typeof window !== 'undefined' ? window : null, function(root) {
    const data = root?.siteDataClient || require('../blogs/js/site-data-client');
    const coverMedia = root?.blogCoverMedia || require('../blogs/js/blog-cover-media');
    const icons = root?.SiteIcons || require('../assets/js/site-icons');

    function render(siteData) {
        const { getPostTitle, getPostSubtitle, getPostUrl } = data;
        const readPostIcon = icons.render('box-arrow-up-right', {
            className: 'portfolio-blog-preview-read-icon'
        });

        const escapeHtml = (value) => String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');

        const formatDate = (value) => {
            const date = new Date(`${value}T00:00:00Z`);
            if (Number.isNaN(date.getTime())) {
                return '';
            }
            return date.toLocaleDateString('en-GB', {
                year: 'numeric',
                month: 'short',
                day: '2-digit',
                timeZone: 'UTC'
            });
        };

        const getSeriesTitle = (post) => {
            const series = post.series && siteData.series?.[post.series];
            return series?.eng || post.series || 'Blog';
        };

        return siteData.featuredPortfolioPosts
            .filter((item) => item.post && (item.post.status || 'published') === 'published')
            .map((item) => {
                const title = getPostTitle(item.post, 'eng');
                const subtitle = getPostSubtitle(item.post, 'eng');
                const seriesTitle = getSeriesTitle(item.post);
                const date = formatDate(item.post.date);
                const image = item.teaserImage || item.post.cover || '/assets/blog_bg.jpeg';
                const animatedImage = coverMedia?.isAnimatedCover(image)
                    ? image
                    : item.post.cover;
                const keepAnimated = item.post.animatedPreview === true && coverMedia?.isAnimatedCover(animatedImage);
                const preview = coverMedia?.getBlogCoverPreviewUrl(item.post, 'portfolio') || image;
                const autoplaySource = keepAnimated
                    ? ` data-autoplay-src="${escapeHtml(animatedImage)}"`
                    : '';
                const animatedSource = !keepAnimated && coverMedia?.isAnimatedCover(animatedImage)
                    ? ` data-animated-src="${escapeHtml(animatedImage)}"`
                    : '';
                const imageAlt = item.teaserAlt || `${title} article preview`;

                return `
            <article class="portfolio-blog-preview-item" data-post-id="${escapeHtml(item.id)}">
              <div class="portfolio-blog-preview-layout">
                <a href="${escapeHtml(getPostUrl(item.post, 'eng'))}" target="_blank" rel="noopener noreferrer" class="portfolio-blog-preview-cover-link" aria-label="Read ${escapeHtml(title)}">
                  <span class="portfolio-blog-preview-cover${item.id === '260201_sparse' || getPostUrl(item.post, 'eng').includes('optimizing-sparse-3d-generation-inference') ? ' edition-media-plate' : ''}">
                    <img src="${escapeHtml(preview)}" data-blog-cover data-preview-src="${escapeHtml(preview)}"${autoplaySource}${animatedSource} alt="${escapeHtml(imageAlt)}" loading="lazy" decoding="async">
                  </span>
                </a>
                <div class="portfolio-blog-preview-body">
                  <span class="portfolio-blog-preview-eyebrow">
                    <span class="portfolio-blog-preview-series">${escapeHtml(seriesTitle)}</span>
                    ${date ? `<span class="portfolio-blog-preview-date-separator" aria-hidden="true">·</span><time class="portfolio-blog-preview-date" datetime="${escapeHtml(item.post.date)}">${escapeHtml(date)}</time>` : ''}
                  </span>
                  <a href="${escapeHtml(getPostUrl(item.post, 'eng'))}" target="_blank" rel="noopener noreferrer" class="portfolio-blog-preview-title-link">
                    <span class="portfolio-blog-preview-title">${escapeHtml(title)}</span>
                  </a>
                  <span class="portfolio-blog-preview-summary">${escapeHtml(subtitle)}</span>
                  <a href="${escapeHtml(getPostUrl(item.post, 'eng'))}" target="_blank" rel="noopener noreferrer" class="portfolio-blog-preview-read-link">
                    <span>Read post</span>
                    ${readPostIcon}
                  </a>
                </div>
              </div>
            </article>
        `;
            }).join('');
    }

    async function mount() {
        const section = root.document.getElementById('blog');
        const container = root.document.getElementById('portfolio-blog-posts');
        if (!container) return;

        try {
            // Built pages already contain the complete preview. Only an unbuilt
            // source preview needs the data fetch and client-side renderer.
            if (!container.querySelector('.portfolio-blog-preview-item')) {
                container.innerHTML = render(await data.loadSiteData());
            }
            coverMedia.initializeBlogCoverMedia(container);
        } catch (error) {
            console.error(error);
        } finally {
            if (section) {
                section.dataset.sectionReady = 'true';
                section.dispatchEvent(new root.CustomEvent('portfolio:section-ready'));
            }
        }
    }

    return { render, mount };
});

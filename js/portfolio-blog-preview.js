document.addEventListener('DOMContentLoaded', async function() {
    const section = document.getElementById('blog');
    const container = document.getElementById('portfolio-blog-posts');
    const listShell = document.querySelector('.portfolio-blog-list-shell');
    const markReady = () => {
        if (!section) return;
        section.dataset.sectionReady = 'true';
        section.dispatchEvent(new CustomEvent('portfolio:section-ready'));
    };

    if (!container || !listShell || !window.siteDataClient) {
        markReady();
        return;
    }

    try {
        const { loadSiteData, getPostTitle, getPostSubtitle, getPostUrl } = window.siteDataClient;
        const coverMedia = window.blogCoverMedia;
        const readPostIcon = window.SiteIcons.render('box-arrow-up-right', {
            className: 'portfolio-blog-preview-read-icon'
        });
        const siteData = await loadSiteData();

        const escapeHtml = (value) => String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');

        const formatDate = (value) => {
            const date = new Date(`${value}T00:00:00`);
            if (Number.isNaN(date.getTime())) {
                return '';
            }
            return date.toLocaleDateString('en-GB', {
                year: 'numeric',
                month: 'short',
                day: '2-digit'
            });
        };

        const getSeriesTitle = (post) => {
            const series = post.series && siteData.series?.[post.series];
            return series?.eng || post.series || 'Blog';
        };

        const postsHtml = siteData.featuredPortfolioPosts
            .filter((item) => item.post)
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
                const preview = coverMedia?.getBlogCoverPreviewUrl(item.id, 'portfolio') || image;
                const autoplaySource = keepAnimated
                    ? ` data-autoplay-src="${escapeHtml(animatedImage)}"`
                    : '';
                const animatedSource = !keepAnimated && coverMedia?.isAnimatedCover(animatedImage)
                    ? ` data-animated-src="${escapeHtml(animatedImage)}"`
                    : '';
                const imageAlt = item.teaserAlt || `${title} article preview`;

                return `
            <article class="portfolio-blog-preview-item">
              <div class="portfolio-blog-preview-layout">
                <a href="${escapeHtml(getPostUrl(item.post, 'eng'))}" target="_blank" rel="noopener noreferrer" class="portfolio-blog-preview-cover-link" aria-label="Read ${escapeHtml(title)}">
                  <span class="portfolio-blog-preview-cover">
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

        container.innerHTML = postsHtml;
        coverMedia?.initializeBlogCoverMedia(container);
    } catch (error) {
        console.error(error);
    } finally {
        markReady();
    }
});

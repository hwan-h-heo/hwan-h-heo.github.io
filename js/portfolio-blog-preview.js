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
        const { loadSiteData, getPostTitle, getPostDescription, getPostUrl } = window.siteDataClient;
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
            return date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
            });
        };

        const getSeriesTitle = (post) => {
            const series = post.series && siteData.series?.[post.series];
            return series?.eng || post.series || 'Blog';
        };

        const renderTags = (post) => (post.tags || [])
            .slice(0, 3)
            .map((tag) => `<span class="portfolio-blog-preview-tag">${escapeHtml(tag)}</span>`)
            .join('');

        const postsHtml = siteData.featuredPortfolioPosts
            .filter((item) => item.post)
            .map((item) => {
                const title = getPostTitle(item.post, 'eng');
                const subtitle = item.post.subtitle_eng || getPostDescription(item.post, 'eng');
                const seriesTitle = getSeriesTitle(item.post);
                const category = item.post.category === 'note' ? 'Note' : 'Post';
                const date = formatDate(item.post.date);
                const tagsHtml = renderTags(item.post);
                const image = item.teaserImage || item.post.cover || '/assets/blog_bg.jpeg';

                return `
            <article class="portfolio-blog-preview-item">
              <a href="${escapeHtml(getPostUrl(item.post, 'eng'))}" target="_blank" rel="noopener noreferrer" class="portfolio-blog-preview-link">
                <span class="portfolio-blog-preview-cover" aria-hidden="true">
                  <img src="${escapeHtml(image)}" alt="">
                </span>
                <span class="portfolio-blog-preview-body">
                  <span class="portfolio-blog-preview-eyebrow">
                    <span>${escapeHtml(category)}</span>
                    <span>${escapeHtml(seriesTitle)}</span>
                  </span>
                  <span class="portfolio-blog-preview-title">${escapeHtml(title)}</span>
                  ${subtitle ? `<span class="portfolio-blog-preview-summary">${escapeHtml(subtitle)}</span>` : ''}
                  ${tagsHtml ? `<span class="portfolio-blog-preview-tags">${tagsHtml}</span>` : ''}
                  ${date ? `<span class="portfolio-blog-preview-meta">${escapeHtml(seriesTitle)} / ${escapeHtml(date)}</span>` : ''}
                </span>
              </a>
            </article>
        `;
            }).join('');

        container.innerHTML = postsHtml;
    } catch (error) {
        console.error(error);
    } finally {
        markReady();
    }
});

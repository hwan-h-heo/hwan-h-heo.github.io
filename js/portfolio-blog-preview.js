document.addEventListener('DOMContentLoaded', async function() {
    const section = document.getElementById('blog');
    const container = document.getElementById('portfolio-blog-posts');
    const swiperElement = document.querySelector('.portfolio-blog-swiper');
    const markReady = () => {
        if (!section) return;
        section.dataset.sectionReady = 'true';
        section.dispatchEvent(new CustomEvent('portfolio:section-ready'));
    };

    if (!container || !swiperElement || !window.siteDataClient) {
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

        const slidesHtml = siteData.featuredPortfolioPosts
            .filter((item) => item.post)
            .map((item) => {
                const title = getPostTitle(item.post, 'eng');
                const subtitle = item.post.subtitle_eng || getPostDescription(item.post, 'eng');

                return `
            <div class="swiper-slide">
              <a href="${escapeHtml(getPostUrl(item.post, 'eng'))}" target="_blank" rel="noopener noreferrer" class="portfolio-box blog-preview-card">
                <div class="aspect-ratio-box">
                  <img src="${escapeHtml(item.teaserImage)}" class="img-fluid" alt="${escapeHtml(item.teaserAlt || title)}">
                </div>
                <div class="polar_content">
                  <h6 title="${escapeHtml(title)}">${escapeHtml(title)}</h6>
                  <p class="portfolio-card-summary">${escapeHtml(subtitle)}</p>
                </div>
              </a>
            </div>
        `;
            }).join('');

        container.innerHTML = slidesHtml;

        const configElement = swiperElement.querySelector('.swiper-config');
        if (!configElement || !window.Swiper) {
            return;
        }

        const config = JSON.parse(configElement.textContent.trim());
        new Swiper(swiperElement, config);
    } catch (error) {
        console.error(error);
    } finally {
        markReady();
    }
});

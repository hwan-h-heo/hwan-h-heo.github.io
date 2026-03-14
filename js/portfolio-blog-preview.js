document.addEventListener('DOMContentLoaded', async function() {
    const container = document.getElementById('portfolio-blog-posts');
    const swiperElement = document.querySelector('.portfolio-blog-swiper');

    if (!container || !swiperElement || !window.siteDataClient) {
        return;
    }

    const { loadSiteData, getPostTitle, getPostUrl } = window.siteDataClient;
    const siteData = await loadSiteData();

    const slidesHtml = siteData.featuredPortfolioPosts
        .filter((item) => item.post)
        .map((item) => `
            <div class="swiper-slide">
              <a href="${getPostUrl(item.post, 'eng')}" target="_blank" class="testimonial-link">
                <div class="testimonial-item">
                  <div class="testimonial-content">
                    <h6>${getPostTitle(item.post, 'eng')}</h6>
                  </div>
                  <div class="testimonial-img">
                    <img src="${item.teaserImage}" class="img-fluid" alt="${item.teaserAlt || getPostTitle(item.post, 'eng')}">
                  </div>
                </div>
              </a>
            </div>
        `).join('');

    container.innerHTML = slidesHtml;

    const configElement = swiperElement.querySelector('.swiper-config');
    if (!configElement || !window.Swiper) {
        return;
    }

    const config = JSON.parse(configElement.textContent.trim());
    new Swiper(swiperElement, config);
});

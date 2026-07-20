document.addEventListener('DOMContentLoaded', async function() {
    const resultsContainer = document.getElementById('search-results-container');
    const searchInput = document.getElementById('search-input');
    const urlParams = new URLSearchParams(window.location.search);
    const searchTerm = urlParams.get('q')?.trim().toLowerCase() || '';

    if (searchInput) {
        searchInput.value = urlParams.get('q') || '';
    }

    if (!resultsContainer) {
        return;
    }

    if (!searchTerm) {
        resultsContainer.innerHTML = '<p>Please enter a search term in the box above.</p>';
        return;
    }

    const { loadSiteData, getPostTitle, getPostDescription, getPostUrl } = window.siteDataClient;
    const siteData = await loadSiteData();
    const postContentCache = {};

    async function fetchAllPostContents() {
        await Promise.all(siteData.posts.map(async (post) => {
            try {
                const contents = await Promise.all(post.languages.map(async (language) => {
                    const response = await fetch(`../posts/${post.id}/content-${language}.md`);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status} for ${language}`);
                    }
                    return response.text();
                }));

                postContentCache[post.id] = contents.join('\n').toLowerCase();
            } catch (error) {
                console.warn(`Failed to fetch markdown for ${post.id}:`, error);
                postContentCache[post.id] = '';
            }
        }));
    }

    function renderSearchResults(filteredPosts) {
        if (filteredPosts.length === 0) {
            resultsContainer.innerHTML = `<p>No results found for "${searchInput.value}".</p>`;
            return;
        }

        const currentLang = localStorage.getItem('language') || 'eng';
        const resultsHtml = filteredPosts.map((post) => {
            const title = getPostTitle(post, currentLang);
            const subtitle = getPostDescription(post, currentLang);
            const seriesTitle = siteData.series[post.series]?.eng || 'Series';
            const tagsHtml = (post.tags || []).map((tag) => `<span class="post-tag">${tag}</span>`).join('');

            return `
                <div class="post-preview">
                    <a href="${getPostUrl(post, currentLang)}" class="post-card-link">
                        <div class="post-card-cover" style="background-image: url('${post.cover || '/assets/blog_bg.jpeg'}')"></div>
                        <div class="post-card-body">
                            <h3 class="post-title">${title}</h3>
                            ${subtitle ? `<h5 class="post-subtitle">${subtitle}</h5>` : ''}
                            ${tagsHtml ? `<div class="post-tag-row">${tagsHtml}</div>` : ''}
                        </div>
                    </a>
                    <p class="post-meta">
                        ${seriesTitle} | ${new Date(post.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
                    </p>
                </div>
                <hr class="my-4" />
            `;
        }).join('');

        resultsContainer.innerHTML = resultsHtml;
    }

    await fetchAllPostContents();

    const filteredPosts = [...siteData.posts]
        .sort((a, b) => new Date(b.date) - new Date(a.date))
        .filter((post) => {
            const titleMatch = Object.keys(post).some((key) => key.startsWith('title_') && String(post[key]).toLowerCase().includes(searchTerm));
            const subtitleMatch = Object.keys(post).some((key) => key.startsWith('subtitle_') && String(post[key]).toLowerCase().includes(searchTerm));
            const descriptionMatch = Object.keys(post).some((key) => key.startsWith('description_') && String(post[key]).toLowerCase().includes(searchTerm));
            const tagMatch = (post.tags || []).some((tag) => tag.toLowerCase().includes(searchTerm));
            const contentMatch = postContentCache[post.id]?.includes(searchTerm);
            return titleMatch || subtitleMatch || descriptionMatch || tagMatch || contentMatch;
        });

    renderSearchResults(filteredPosts);
});

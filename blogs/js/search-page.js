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

    const { loadSiteData, getPostTitle, getPostUrl } = window.siteDataClient;
    const siteData = await loadSiteData();
    const postContentCache = {};

    async function fetchAllPostContents() {
        await Promise.all(siteData.posts.map(async (post) => {
            try {
                const [korResponse, engResponse] = await Promise.all([
                    fetch(`../posts/${post.id}/content-kor.md`),
                    fetch(`../posts/${post.id}/content-eng.md`)
                ]);

                const [korText, engText] = await Promise.all([
                    korResponse.ok ? korResponse.text() : '',
                    engResponse.ok ? engResponse.text() : ''
                ]);

                postContentCache[post.id] = `${korText}\n${engText}`.toLowerCase();
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
            const subtitle = post[`subtitle_${currentLang}`] || post.subtitle_eng || '';
            const seriesTitle = siteData.series[post.series]?.eng || 'Series';

            return `
                <div class="post-preview">
                    <a href="${getPostUrl(post, currentLang)}">
                        <h3 class="post-title">${title}</h3>
                        ${subtitle ? `<h5 class="post-subtitle">${subtitle}</h5>` : ''}
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
            const contentMatch = postContentCache[post.id]?.includes(searchTerm);
            return titleMatch || subtitleMatch || contentMatch;
        });

    renderSearchResults(filteredPosts);
});


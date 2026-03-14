document.addEventListener('DOMContentLoaded', async function() {
    const postsContainer = document.querySelector('#posts-tab');
    const seriesContainer = document.querySelector('#series-tab');
    const notesContainer = document.querySelector('#notes-tab');
    const langToggleButton = document.getElementById('lang-toggle-main');

    const { loadSiteData, getPostTitle, getPostUrl } = window.siteDataClient;
    const siteData = await loadSiteData();
    const sortedPosts = [...siteData.posts].sort((a, b) => new Date(b.date) - new Date(a.date));

    function createPostPreviewHTML(post, lang) {
        const title = getPostTitle(post, lang);
        const subtitle = post[`subtitle_${lang}`] || post.subtitle_eng || '';
        const seriesTitle = siteData.series[post.series]?.[lang] || siteData.series[post.series]?.eng || 'Series';

        if (!title) {
            return '';
        }

        return `
        <div class="post-preview">
            <a href="${getPostUrl(post, lang)}">
                <h3 class="post-title">${title}</h3>
                ${subtitle ? `<h5 class="post-subtitle">${subtitle}</h5>` : ''}
            </a>
            <p class="post-meta">
                ${seriesTitle} | ${new Date(post.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
            </p>
        </div>
        <hr class="my-4" />
        `;
    }

    function renderAllPosts(lang) {
        if (!postsContainer) {
            return;
        }

        const allPostsHTML = sortedPosts
            .filter((post) => post.category === 'post')
            .map((post) => createPostPreviewHTML(post, lang))
            .join('');

        postsContainer.innerHTML = allPostsHTML || '<p>No posts yet.</p>';
    }

    function renderNotes(lang) {
        if (!notesContainer) {
            return;
        }

        const notesHTML = sortedPosts
            .filter((post) => post.category === 'note')
            .map((post) => createPostPreviewHTML(post, lang))
            .join('');

        notesContainer.innerHTML = notesHTML || '<p>No notes yet.</p>';
    }

    function renderSeries(lang) {
        if (!seriesContainer) {
            return;
        }

        const postsBySeries = {};
        sortedPosts.forEach((post) => {
            if (!post.series) {
                return;
            }
            if (!postsBySeries[post.series]) {
                postsBySeries[post.series] = [];
            }
            postsBySeries[post.series].push(post);
        });

        const seriesHTML = Object.entries(postsBySeries).map(([seriesId, posts]) => {
            const seriesTitle = siteData.series[seriesId]?.[lang] || siteData.series[seriesId]?.eng || 'Series';
            const itemsHtml = posts.map((post) => `
                <li>
                    <a href="${getPostUrl(post, lang)}">${getPostTitle(post, lang)}</a>
                    <span class="post-meta-sm">${new Date(post.date).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}</span>
                </li>
            `).join('');

            return `
                <div class="series-group mb-5">
                    <h3 class="series-title">${seriesTitle}</h3>
                    <ul class="series-post-list">
                        ${itemsHtml}
                    </ul>
                </div>
            `;
        }).join('');

        seriesContainer.innerHTML = seriesHTML || '<p>No series yet.</p>';
    }

    function renderAllTabs(lang) {
        renderAllPosts(lang);
        renderNotes(lang);
        renderSeries(lang);

        if (langToggleButton) {
            langToggleButton.textContent = lang === 'eng' ? 'KOR' : 'ENG';
        }
    }

    if (langToggleButton) {
        langToggleButton.addEventListener('click', function() {
            const currentLang = localStorage.getItem('language') || 'eng';
            const nextLang = currentLang === 'eng' ? 'kor' : 'eng';
            localStorage.setItem('language', nextLang);
            renderAllTabs(nextLang);
        });
    }

    renderAllTabs(localStorage.getItem('language') || 'eng');
});

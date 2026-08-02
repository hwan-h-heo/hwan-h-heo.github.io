(function() {
    const DATA_PATH = '/blogs/data/site-data.json';
    let siteDataPromise;

    function createSlug(title) {
        return String(title || '')
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-')
            .trim();
    }

    function normalizeSiteData(rawSiteData) {
        const posts = rawSiteData.posts
            .filter((post) => post.status !== 'draft')
            .map((post) => ({
                ...post,
                languages: [...post.languages],
                tags: Array.isArray(post.tags) ? [...post.tags] : [],
                status: post.status || 'published',
                cover: post.cover || '/assets/blog_bg.jpeg',
                updated: post.updated || post.date,
                slug: post.slug || createSlug(post.title_eng || post.id)
            }))
            .sort((a, b) => new Date(b.date) - new Date(a.date));

        const postById = Object.fromEntries(posts.map((post) => [post.id, post]));

        return {
            posts,
            blogHome: { ...(rawSiteData.blogHome || {}) },
            series: rawSiteData.series || {},
            portfolioProjects: rawSiteData.portfolioProjects || [],
            publications: rawSiteData.publications || [],
            talks: rawSiteData.talks || [],
            featuredPortfolioPosts: (rawSiteData.featuredPortfolioPosts || []).map((item) => ({
                ...item,
                post: postById[item.id]
            })),
            postById
        };
    }

    async function loadSiteData() {
        if (!siteDataPromise) {
            siteDataPromise = fetch(DATA_PATH)
                .then((response) => {
                    if (!response.ok) {
                        throw new Error(`Failed to load ${DATA_PATH}`);
                    }
                    return response.json();
                })
                .then(normalizeSiteData);
        }

        return siteDataPromise;
    }

    function getPostTitle(post, lang) {
        return post[`title_${lang}`] || post.title_eng;
    }

    function getPostSubtitle(post, lang) {
        return post[`subtitle_${lang}`] || post.subtitle_eng || '';
    }

    function getPostUrl(post, lang) {
        const resolvedLang = post.languages.includes(lang) ? lang : 'eng';
        return `/blogs/posts/${resolvedLang === 'eng' ? post.slug : `${post.slug}-kor`}/`;
    }

    window.siteDataClient = {
        loadSiteData,
        getPostTitle,
        getPostSubtitle,
        getPostUrl
    };
})();

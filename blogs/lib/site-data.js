const fs = require('fs');
const path = require('path');

const SITE_DATA_PATH = path.join(__dirname, '..', 'data', 'site-data.json');
const POST_RELATED_PATH = path.join(__dirname, '..', 'data', 'post-related.json');
const REPO_ROOT = path.join(__dirname, '..', '..');
const POST_CATEGORIES = ['post', 'note'];
const POST_LANGUAGES = ['eng', 'kor'];
const PORTFOLIO_CATEGORIES = ['research', 'app', 'per'];

const POST_ALLOWED_KEYS = new Set([
    'id',
    'title_eng',
    'title_kor',
    'subtitle_eng',
    'subtitle_kor',
    'date',
    'category',
    'series',
    'languages',
    'slug',
    'description_eng',
    'description_kor',
    'seoTitle',
    'seoTitle_eng',
    'seoTitle_kor',
    'tags',
    'cover',
    'previewImage',
    'animatedPreview',
    'socialImage',
    'translationKey',
    'dependencies',
    'runtimeDependencies',
    'status',
    'updated'
]);

function createSlug(title) {
    return String(title || '')
        .toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-')
        .trim();
}

function validatePostShape(post, seriesMap, errors) {
    Object.keys(post).forEach((key) => {
        if (!POST_ALLOWED_KEYS.has(key)) {
            errors.push(`Unexpected key "${key}" in post "${post.id || 'unknown'}".`);
        }
    });

    ['id', 'title_eng', 'date', 'category', 'series', 'languages'].forEach((key) => {
        if (!post[key]) {
            errors.push(`Missing required "${key}" in post "${post.id || 'unknown'}".`);
        }
    });

    if (post.category && !POST_CATEGORIES.includes(post.category)) {
        errors.push(`Invalid category "${post.category}" in post "${post.id}".`);
    }

    if (post.languages) {
        if (!Array.isArray(post.languages) || post.languages.length === 0) {
            errors.push(`Post "${post.id}" must define a non-empty languages array.`);
        } else {
            const uniqueLanguages = new Set(post.languages);
            if (uniqueLanguages.size !== post.languages.length) {
                errors.push(`Post "${post.id}" contains duplicate languages.`);
            }

            post.languages.forEach((lang) => {
                if (!POST_LANGUAGES.includes(lang)) {
                    errors.push(`Invalid language "${lang}" in post "${post.id}".`);
                }
            });

            if (!post.languages.includes('eng')) {
                errors.push(`Post "${post.id}" must include "eng" in languages.`);
            }

            if (post.languages.includes('kor') && !post.title_kor) {
                errors.push(`Post "${post.id}" must define "title_kor" when Korean content exists.`);
            }
        }
    }

    if (post.series && !seriesMap[post.series]) {
        errors.push(`Post "${post.id}" references unknown series "${post.series}".`);
    }

    ['description_eng', 'description_kor', 'seoTitle', 'seoTitle_eng', 'seoTitle_kor', 'cover', 'previewImage', 'socialImage', 'translationKey', 'status', 'updated', 'slug'].forEach((key) => {
        validateStringField(post, key, errors, `post "${post.id || 'unknown'}"`, false);
    });

    if (post.status && !['published', 'draft'].includes(post.status)) {
        errors.push(`Invalid status "${post.status}" in post "${post.id}".`);
    }

    if (post.updated && !/^\d{4}-\d{2}-\d{2}$/.test(post.updated)) {
        errors.push(`Invalid updated date "${post.updated}" in post "${post.id}".`);
    }

    if (post.updated && post.date && post.updated < post.date) {
        errors.push(`Updated date precedes published date in post "${post.id}".`);
    }

    if (post.slug && !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(post.slug)) {
        errors.push(`Invalid slug "${post.slug}" in post "${post.id}".`);
    }

    if (post.animatedPreview !== undefined && typeof post.animatedPreview !== 'boolean') {
        errors.push(`"animatedPreview" must be a boolean in post "${post.id}".`);
    }

    if (post.animatedPreview && !/\.gif(?:[?#].*)?$/i.test(post.cover || '')) {
        errors.push(`Post "${post.id}" can only enable "animatedPreview" for a GIF cover.`);
    }

    if (post.tags !== undefined) {
        if (!Array.isArray(post.tags)) {
            errors.push(`"tags" must be an array in post "${post.id}".`);
        } else {
            const seenTags = new Set();
            post.tags.forEach((tag) => {
                if (typeof tag !== 'string' || !tag.trim()) {
                    errors.push(`Post "${post.id}" contains an invalid tag.`);
                } else if (seenTags.has(tag)) {
                    errors.push(`Post "${post.id}" contains duplicate tag "${tag}".`);
                }
                seenTags.add(tag);
            });
        }
    }

    ['dependencies', 'runtimeDependencies'].forEach((key) => {
        if (post[key] !== undefined) {
            if (!Array.isArray(post[key])) {
                errors.push(`"${key}" must be an array in post "${post.id}".`);
                return;
            }

            post[key].forEach((dependency) => {
                if (typeof dependency !== 'string' || !dependency.trim()) {
                    errors.push(`Post "${post.id}" contains an invalid ${key} entry.`);
                }
            });
        }
    });

    const status = post.status || 'published';
    if (status === 'published') {
        ['slug', 'description_eng', 'cover', 'updated'].forEach((key) => {
            if (!post[key]) {
                errors.push(`Published post "${post.id}" must define "${key}".`);
            }
        });
        if (post.languages?.includes('kor') && !post.description_kor) {
            errors.push(`Published post "${post.id}" must define "description_kor" for Korean content.`);
        }
        if (!Array.isArray(post.tags) || post.tags.length === 0) {
            errors.push(`Published post "${post.id}" must define at least one tag.`);
        }
        if (post.cover === '/assets/blog_bg.jpeg') {
            errors.push(`Published post "${post.id}" must use a post-specific cover.`);
        }
        if (isExternalUrl(post.cover || '')) {
            errors.push(`Published post "${post.id}" must use a local cover.`);
        }
    }

    validateLocalFileReference(post.cover, errors, `post "${post.id || 'unknown'}"`, 'cover');
    validateLocalFileReference(post.previewImage, errors, `post "${post.id || 'unknown'}"`, 'previewImage');
    validateLocalFileReference(post.socialImage, errors, `post "${post.id || 'unknown'}"`, 'socialImage');
}

function validateStringField(item, key, errors, label, required = true) {
    const value = item[key];
    if (required && !value) {
        errors.push(`Missing required "${key}" in ${label}.`);
        return;
    }

    if (value !== undefined && typeof value !== 'string') {
        errors.push(`"${key}" must be a string in ${label}.`);
    }
}

function isExternalUrl(value) {
    return /^(https?:)?\/\//i.test(value) || /^(mailto|tel):/i.test(value);
}

function normalizeLocalUrlPath(value) {
    const cleaned = String(value || '').split('#')[0].split('?')[0].replace(/^\/+/, '');

    if (!cleaned || cleaned.startsWith('#')) {
        return '';
    }

    return cleaned.endsWith('/') ? `${cleaned}index.html` : cleaned;
}

function projectSourceExistsForRoute(localPath) {
    const match = localPath.match(/^projects\/([^/]+)\/index\.html$/);
    if (!match) {
        return false;
    }

    const projectDir = path.join(REPO_ROOT, 'projects', match[1]);
    return fs.existsSync(path.join(projectDir, 'project.json'))
        && fs.existsSync(path.join(projectDir, 'content.md'));
}

function validateLocalFileReference(value, errors, label, key, { allowGeneratedProjectRoute = false } = {}) {
    if (!value || isExternalUrl(value)) {
        return;
    }

    const localPath = normalizeLocalUrlPath(value);
    if (!localPath) {
        return;
    }

    const absolutePath = path.join(REPO_ROOT, localPath);
    if (fs.existsSync(absolutePath)) {
        return;
    }

    if (allowGeneratedProjectRoute && projectSourceExistsForRoute(localPath)) {
        return;
    }

    errors.push(`Missing local file for "${key}" in ${label}: ${value}`);
}

function validatePortfolioProjectShape(project, index, errors) {
    const label = `portfolio project ${project.id || index}`;

    ['id', 'title', 'summary', 'url', 'typeLabel', 'organization', 'period']
        .forEach((key) => validateStringField(project, key, errors, label));
    ['accolade', 'image', 'gif', 'video', 'poster', 'alt']
        .forEach((key) => validateStringField(project, key, errors, label, false));

    if (!Array.isArray(project.categories) || project.categories.length === 0) {
        errors.push(`"${label}" must define a non-empty categories array.`);
    } else {
        project.categories.forEach((category) => {
            if (!PORTFOLIO_CATEGORIES.includes(category)) {
                errors.push(`Invalid portfolio category "${category}" in ${label}.`);
            }
        });
    }

    if (!Array.isArray(project.tags) || project.tags.length < 2) {
        errors.push(`"${label}" must define at least two tags.`);
    } else {
        project.tags.forEach((tag) => {
            if (typeof tag !== 'string' || !tag.trim()) {
                errors.push(`Invalid empty or non-string tag in ${label}.`);
            }
        });
        if (new Set(project.tags).size !== project.tags.length) {
            errors.push(`"${label}" must not define duplicate tags.`);
        }
    }

    if (!project.image && !project.video) {
        errors.push(`"${label}" must define either image or video.`);
    }

    if (project.external !== undefined && typeof project.external !== 'boolean') {
        errors.push(`"external" must be a boolean in ${label}.`);
    }

    validateLocalFileReference(project.url, errors, label, 'url', { allowGeneratedProjectRoute: true });
    ['image', 'gif', 'video', 'poster'].forEach((key) => {
        if (project[key] && isExternalUrl(project[key])) {
            errors.push(`"${key}" in ${label} must be local.`);
        }
        validateLocalFileReference(project[key], errors, label, key);
    });
}

function validatePublicationShape(publication, index, errors) {
    const label = `publication ${publication.title || index}`;

    ['title', 'authorsHtml', 'venueHtml'].forEach((key) => validateStringField(publication, key, errors, label));

    if (!Array.isArray(publication.links)) {
        errors.push(`"${label}" must define links as an array.`);
        return;
    }

    publication.links.forEach((link, linkIndex) => {
        const linkLabel = `${label} link ${linkIndex}`;
        ['label', 'url'].forEach((key) => validateStringField(link, key, errors, linkLabel));
        validateStringField(link, 'icon', errors, linkLabel, false);
    });
}

function validateTalkShape(talk, index, errors) {
    const label = `talk ${talk.title || index}`;
    ['title', 'venueHtml', 'date'].forEach((key) => validateStringField(talk, key, errors, label));
    validateStringField(talk, 'titleHtml', errors, label, false);
}

function validateBlogHomeShape(blogHome, posts, errors) {
    const allowedKeys = new Set(['featuredPostId']);
    Object.keys(blogHome).forEach((key) => {
        if (!allowedKeys.has(key)) {
            errors.push(`Unexpected key "${key}" in blogHome.`);
        }
    });

    validateStringField(blogHome, 'featuredPostId', errors, 'blogHome');
    if (!blogHome.featuredPostId) {
        return;
    }

    const featuredPost = posts.find((post) => post.id === blogHome.featuredPostId);
    if (!featuredPost) {
        errors.push(`Blog home featured post "${blogHome.featuredPostId}" does not exist.`);
        return;
    }
    if ((featuredPost.status || 'published') !== 'published') {
        errors.push(`Blog home featured post "${blogHome.featuredPostId}" must be published.`);
    }
    if (featuredPost.category !== 'post') {
        errors.push(`Blog home featured post "${blogHome.featuredPostId}" must use the post category.`);
    }
}

function normalizeRelatedItem(item) {
    if (typeof item === 'string') {
        return {
            type: 'post',
            postId: item
        };
    }

    return {
        type: 'external',
        title: item.title,
        url: item.url
    };
}

function normalizeRelatedByLanguage(relatedConfig = {}) {
    return Object.fromEntries(Object.entries(relatedConfig).map(([language, items]) => [
        language,
        items.map(normalizeRelatedItem)
    ]));
}

function normalizeSiteData(rawSiteData, relatedData = {}) {
    const posts = rawSiteData.posts
        .filter((post) => post.status !== 'draft')
        .map((post) => ({
            ...post,
            languages: [...post.languages],
            tags: Array.isArray(post.tags) ? [...post.tags] : [],
            status: post.status || 'published',
            description_eng: post.description_eng || post.subtitle_eng || '',
            description_kor: post.description_kor || post.subtitle_kor || post.description_eng || post.subtitle_eng || '',
            cover: post.cover || '/assets/blog_bg.jpeg',
            previewImage: post.previewImage || '',
            socialImage: post.socialImage || '',
            translationKey: post.translationKey || post.id,
            updated: post.updated || post.date,
            slug: post.slug || createSlug(post.title_eng || post.id),
            relatedByLanguage: normalizeRelatedByLanguage(relatedData[post.id])
        })).sort((a, b) => new Date(b.date) - new Date(a.date));

    const postById = Object.fromEntries(posts.map((post) => [post.id, post]));
    const slugMapping = Object.fromEntries(posts.map((post) => [post.id, post.slug]));
    const slugToId = Object.fromEntries(posts.map((post) => [post.slug, post.id]));

    const featuredPortfolioPosts = (rawSiteData.featuredPortfolioPosts || []).map((item) => ({
        ...item,
        post: postById[item.id]
    }));

    return {
        posts,
        blogHome: { ...rawSiteData.blogHome },
        series: rawSiteData.series,
        portfolioProjects: rawSiteData.portfolioProjects || [],
        publications: rawSiteData.publications || [],
        talks: rawSiteData.talks || [],
        featuredPortfolioPosts,
        slugMapping,
        slugToId,
        postById
    };
}

function validateSiteData(rawSiteData) {
    const errors = [];

    if (!rawSiteData || typeof rawSiteData !== 'object') {
        errors.push('Site data must be a JSON object.');
    }

    if (!rawSiteData.posts || !Array.isArray(rawSiteData.posts)) {
        errors.push('Site data must include a posts array.');
    }

    if (!rawSiteData.series || typeof rawSiteData.series !== 'object') {
        errors.push('Site data must include a series map.');
    }

    if (!rawSiteData.blogHome || typeof rawSiteData.blogHome !== 'object' || Array.isArray(rawSiteData.blogHome)) {
        errors.push('Site data must include a blogHome object.');
    }

    if (errors.length > 0) {
        throw new Error(errors.join('\n'));
    }

    rawSiteData.posts.forEach((post) => validatePostShape(post, rawSiteData.series, errors));
    validateBlogHomeShape(rawSiteData.blogHome, rawSiteData.posts, errors);

    const portfolioProjectIds = new Set();
    (rawSiteData.portfolioProjects || []).forEach((project, index) => {
        validatePortfolioProjectShape(project, index, errors);
        if (project.id) {
            if (portfolioProjectIds.has(project.id)) {
                errors.push(`Duplicate portfolio project id "${project.id}".`);
            }
            portfolioProjectIds.add(project.id);
        }
    });
    (rawSiteData.publications || []).forEach((publication, index) => {
        validatePublicationShape(publication, index, errors);
    });
    (rawSiteData.talks || []).forEach((talk, index) => {
        validateTalkShape(talk, index, errors);
    });

    const ids = new Set();
    const slugs = new Set();
    rawSiteData.posts.forEach((post) => {
        if (ids.has(post.id)) {
            errors.push(`Duplicate post id "${post.id}".`);
        }
        ids.add(post.id);

        const slug = post.slug || createSlug(post.title_eng || post.id);
        if (!slug) {
            errors.push(`Post "${post.id}" produced an empty slug.`);
        } else if (slugs.has(slug)) {
            errors.push(`Duplicate slug "${slug}".`);
        }
        slugs.add(slug);
    });

    const featuredIds = new Set();
    (rawSiteData.featuredPortfolioPosts || []).forEach((item) => {
        if (!ids.has(item.id)) {
            errors.push(`Featured portfolio post "${item.id}" does not exist.`);
        }
        if (!item.teaserImage) {
            errors.push(`Featured portfolio post "${item.id}" is missing teaserImage.`);
        } else if (isExternalUrl(item.teaserImage)) {
            errors.push(`Featured portfolio post "${item.id}" must use a local teaserImage.`);
        } else {
            validateLocalFileReference(item.teaserImage, errors, `featured portfolio post ${item.id}`, 'teaserImage');
        }
        if (featuredIds.has(item.id)) {
            errors.push(`Featured portfolio post "${item.id}" is duplicated.`);
        }
        featuredIds.add(item.id);
    });

    if (errors.length > 0) {
        throw new Error(errors.join('\n'));
    }
}

function validatePostRelatedData(relatedData, rawSiteData) {
    const errors = [];
    const publishedPosts = (rawSiteData.posts || []).filter((post) => (post.status || 'published') === 'published');
    const publishedPostById = Object.fromEntries(publishedPosts.map((post) => [post.id, post]));

    if (!relatedData || typeof relatedData !== 'object' || Array.isArray(relatedData)) {
        throw new Error('Post related data must be a JSON object.');
    }

    Object.entries(relatedData).forEach(([sourceId, languageConfig]) => {
        const sourcePost = publishedPostById[sourceId];
        if (!sourcePost) {
            errors.push(`Related post source "${sourceId}" does not match a published post.`);
            return;
        }

        if (!languageConfig || typeof languageConfig !== 'object' || Array.isArray(languageConfig)) {
            errors.push(`Related post source "${sourceId}" must map to a language object.`);
            return;
        }

        Object.entries(languageConfig).forEach(([language, items]) => {
            if (!POST_LANGUAGES.includes(language)) {
                errors.push(`Related post source "${sourceId}" uses invalid language "${language}".`);
                return;
            }
            if (!sourcePost.languages.includes(language)) {
                errors.push(`Related post source "${sourceId}" defines "${language}" recommendations but has no "${language}" content.`);
            }
            if (!Array.isArray(items)) {
                errors.push(`Related post source "${sourceId}" language "${language}" must be an array.`);
                return;
            }

            const seenItems = new Set();
            items.forEach((item, index) => {
                if (typeof item === 'string') {
                    const itemKey = `post:${item}`;
                    if (seenItems.has(itemKey)) {
                        errors.push(`Related post source "${sourceId}" language "${language}" duplicates target "${item}".`);
                    }
                    seenItems.add(itemKey);

                    if (item === sourceId) {
                        errors.push(`Related post source "${sourceId}" language "${language}" links to itself.`);
                    }
                    if (!publishedPostById[item]) {
                        errors.push(`Related post source "${sourceId}" language "${language}" references unknown post "${item}".`);
                    }
                    return;
                }

                const itemLabel = `related item ${index} for "${sourceId}" language "${language}"`;
                if (!item || typeof item !== 'object' || Array.isArray(item)) {
                    errors.push(`Invalid ${itemLabel}.`);
                    return;
                }

                const allowedKeys = new Set(['title', 'url']);
                Object.keys(item).forEach((key) => {
                    if (!allowedKeys.has(key)) {
                        errors.push(`Unexpected key "${key}" in ${itemLabel}.`);
                    }
                });

                if (typeof item.title !== 'string' || !item.title.trim()) {
                    errors.push(`Missing title in ${itemLabel}.`);
                }
                if (typeof item.url !== 'string' || !/^https?:\/\//i.test(item.url)) {
                    errors.push(`External ${itemLabel} must use an absolute HTTP(S) URL.`);
                }

                const itemKey = `url:${item.url}`;
                if (seenItems.has(itemKey)) {
                    errors.push(`Related post source "${sourceId}" language "${language}" duplicates URL "${item.url}".`);
                }
                seenItems.add(itemKey);
            });
        });
    });

    if (errors.length > 0) {
        throw new Error(errors.join('\n'));
    }
}

function loadRawSiteData(filePath = SITE_DATA_PATH) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function loadRawPostRelatedData(filePath = POST_RELATED_PATH) {
    if (!fs.existsSync(filePath)) {
        return {};
    }
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function loadSiteData(filePath = SITE_DATA_PATH, relatedPath = POST_RELATED_PATH) {
    const rawSiteData = loadRawSiteData(filePath);
    const relatedData = loadRawPostRelatedData(relatedPath);
    validateSiteData(rawSiteData);
    validatePostRelatedData(relatedData, rawSiteData);
    return normalizeSiteData(rawSiteData, relatedData);
}

function writeSiteData(rawSiteData, filePath = SITE_DATA_PATH) {
    validateSiteData(rawSiteData);

    const dirPath = path.dirname(filePath);
    const tempPath = path.join(dirPath, `${path.basename(filePath)}.tmp`);
    const serialized = `${JSON.stringify(rawSiteData, null, 2)}\n`;

    fs.writeFileSync(tempPath, serialized, 'utf8');
    fs.renameSync(tempPath, filePath);
}

module.exports = {
    SITE_DATA_PATH,
    POST_RELATED_PATH,
    POST_CATEGORIES,
    POST_LANGUAGES,
    PORTFOLIO_CATEGORIES,
    createSlug,
    loadRawPostRelatedData,
    loadRawSiteData,
    loadSiteData,
    normalizeSiteData,
    validatePostRelatedData,
    validateSiteData,
    writeSiteData
};

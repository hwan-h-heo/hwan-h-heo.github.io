import legacyRedirects from '../../../blogs/data/legacy-post-redirects.json' with { type: 'json' };

export const REDIRECT_STATUS = 308;
export const DEFAULT_GITHUB_PAGES_ORIGIN = 'https://hwan-h-heo.github.io';

function stripTrailingSlash(value) {
    return String(value || '').replace(/\/+$/, '');
}

function normalizeDirectoryPath(pathname) {
    return pathname.replace(/\/+$/, '/');
}

export function getLegacyRedirectUrl(requestUrl, env = {}) {
    const url = new URL(requestUrl);
    if (normalizeDirectoryPath(url.pathname) !== '/blogs/posts/') {
        return '';
    }

    const legacyId = url.searchParams.get('id') || '';
    const targetPath = legacyRedirects[legacyId];
    if (!targetPath) {
        return '';
    }

    const canonicalOrigin = stripTrailingSlash(env.CANONICAL_ORIGIN || url.origin);
    const targetUrl = new URL(targetPath, `${canonicalOrigin}/`);
    if (targetUrl.pathname === normalizeDirectoryPath(url.pathname) && !targetUrl.search) {
        return '';
    }

    return targetUrl.href;
}

function createOriginRequest(request, env = {}) {
    const requestUrl = new URL(request.url);
    const originUrl = new URL(stripTrailingSlash(env.GITHUB_PAGES_ORIGIN || DEFAULT_GITHUB_PAGES_ORIGIN));
    originUrl.pathname = requestUrl.pathname;
    originUrl.search = requestUrl.search;
    return new Request(originUrl.href, request);
}

export default {
    async fetch(request, env) {
        const redirectUrl = getLegacyRedirectUrl(request.url, env);
        if (redirectUrl) {
            return Response.redirect(redirectUrl, REDIRECT_STATUS);
        }

        return fetch(createOriginRequest(request, env));
    }
};

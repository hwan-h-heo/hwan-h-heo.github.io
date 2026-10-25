const assert = require('node:assert/strict');
const test = require('node:test');
const vm = require('node:vm');
const { loadRawSiteData, normalizeSiteData, validateSiteData } = require('../blogs/lib/site-data');
const { listSeriesRedirectEntries, renderSeriesRedirectPage } = require('../blogs/lib/legacy-redirects');

test('retired series resolve to canonical archives and preserve language queries and fragments', () => {
    const raw = loadRawSiteData();
    validateSiteData(raw);
    const entries = listSeriesRedirectEntries(normalizeSiteData(raw));
    assert.deepEqual(entries, [{ path: '/blogs/series/web-3d/', target: '/blogs/series/graphics/' }]);
    const html = renderSeriesRedirectPage(entries[0].target);
    let destination;
    vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], {
        window: { location: { search: '?lang=kor&from=old', hash: '#articles', replace: (url) => { destination = url; } } }
    });
    assert.equal(destination, '/blogs/series/graphics/?lang=kor&from=old#articles');
    assert(html.includes('content="0;url=/blogs/series/graphics/"'));
    assert(html.includes('href="/blogs/series/graphics/"'));
});

test('series redirects reject collisions, missing archives, chains, and unsafe source paths', () => {
    const invalid = [
        { graphics: 'graphics' },
        { 'web-3d': 'missing-series' },
        { 'web-3d': 'retired', retired: 'graphics' },
        { '../web-3d': 'graphics' },
        { 'web-3d': 'linear-algebra' },
        []
    ];
    invalid.forEach((seriesRedirects) => {
        const data = loadRawSiteData();
        data.seriesRedirects = seriesRedirects;
        assert.throws(() => validateSiteData(data), /series/i);
    });
});

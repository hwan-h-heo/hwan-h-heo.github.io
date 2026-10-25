(function(root, factory) {
    const client = typeof module === 'object' && module.exports ? require('./site-data-client') : root.siteDataClient;
    const api = factory(client);
    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    }
    if (root) {
        root.blogSearch = api;
    }
})(typeof globalThis !== 'undefined' ? globalThis : this, function(client) {
    function normalize(value) {
        return String(value || '').normalize('NFC').toLowerCase().replace(/\s+/g, ' ').trim();
    }

    function getQueryTerms(query) {
        return [...new Set(normalize(query).match(/[\p{L}\p{N}]+(?:[._-][\p{L}\p{N}]+)*[+#]*/gu) || [])];
    }

    function findMatches(text, terms) {
        if (!terms.length) {
            return [];
        }
        const pattern = terms.slice().sort((a, b) => b.length - a.length)
            .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
        return [...String(text).matchAll(new RegExp(pattern, 'giu'))]
            .map((match) => ({ start: match.index, end: match.index + match[0].length, term: normalize(match[0]) }));
    }

    function splitHighlights(text, query) {
        const source = String(text || '');
        const parts = [];
        let cursor = 0;
        findMatches(source, getQueryTerms(query)).forEach((match) => {
            if (match.start > cursor) {
                parts.push({ text: source.slice(cursor, match.start), match: false });
            }
            parts.push({ text: source.slice(match.start, match.end), match: true });
            cursor = match.end;
        });
        if (cursor < source.length) {
            parts.push({ text: source.slice(cursor), match: false });
        }
        return parts;
    }

    function getSearchExcerpt(text, terms, maxLength = 180) {
        const source = String(text || '');
        const matches = findMatches(source, terms);
        let best = null;
        matches.forEach((match, index) => {
            const paragraphStart = source.lastIndexOf('\n', match.start) + 1;
            const nextBreak = source.indexOf('\n', match.end);
            const paragraphEnd = nextBreak < 0 ? source.length : nextBreak;
            const windowEnd = Math.min(paragraphEnd, match.start + maxLength);
            const found = new Set();
            let lastMatchEnd = match.end;
            for (let cursor = index; cursor < matches.length && matches[cursor].end <= windowEnd; cursor += 1) {
                if (!found.has(matches[cursor].term)) {
                    found.add(matches[cursor].term);
                    lastMatchEnd = matches[cursor].end;
                }
            }
            let start = Math.max(paragraphStart, match.start - Math.floor(maxLength / 3), lastMatchEnd - maxLength);
            const end = Math.min(paragraphEnd, start + maxLength);
            start = Math.max(paragraphStart, Math.min(start, end - maxLength));
            if (!best || found.size > best.coverage) {
                best = { start, end, paragraphStart, paragraphEnd, firstMatchStart: match.start, lastMatchEnd, coverage: found.size };
            }
        });
        if (!best) {
            return '';
        }
        let { start, end } = best;
        if (start > best.paragraphStart) {
            const boundary = source.indexOf(' ', start);
            if (boundary >= 0 && boundary < start + 20 && boundary < best.firstMatchStart) {
                start = boundary + 1;
            }
        }
        if (end < best.paragraphEnd) {
            const boundary = source.lastIndexOf(' ', end);
            if (boundary > end - 20 && boundary >= best.lastMatchEnd) {
                end = boundary;
            }
        }
        return `${start > best.paragraphStart ? '…' : ''}${source.slice(start, end).trim()}${end < best.paragraphEnd ? '…' : ''}`;
    }

    function searchPosts(siteData, index, query, language) {
        const terms = getQueryTerms(query);
        if (!terms.length) {
            return [];
        }
        const phrase = normalize(query);
        const results = [];
        siteData.posts.forEach((post) => {
            const contentLanguage = client.getPostLanguage(post, language);
            const titles = [post[`title_${language}`], post[`title_${contentLanguage}`]].map(normalize);
            const subtitles = [post[`subtitle_${language}`], post[`subtitle_${contentLanguage}`]].map(normalize);
            const seriesName = (id) => siteData.series?.[id]?.[language] || siteData.series?.[id]?.eng || id || '';
            const primarySeries = normalize(seriesName(post.series));
            const topics = [...(post.tags || []), ...client.getPostSeriesIds(post).map(seriesName)].map(normalize);
            const body = index.posts[post.id]?.[contentLanguage] || '';
            const normalizedBody = normalize(body);
            const hasTitle = (term) => titles.some((title) => title.includes(term));
            const hasTopic = (term) => topics.some((topic) => topic.includes(term));
            const hasSubtitle = (term) => subtitles.some((subtitle) => subtitle.includes(term));
            const hasMetadata = (term) => hasTitle(term) || hasTopic(term) || hasSubtitle(term);
            if (!terms.every((term) => hasMetadata(term) || normalizedBody.includes(term))) {
                return;
            }
            const tier = terms.every(hasTitle) ? 4
                : terms.every((term) => hasTitle(term) || hasTopic(term)) ? 3
                : terms.every(hasMetadata) ? 2
                : terms.some(hasMetadata) ? 1 : 0;
            const score = [
                tier,
                Number(titles.includes(phrase)),
                Number(titles.some((title) => title.includes(phrase))),
                terms.filter(hasTitle).length,
                Number(topics.includes(phrase)),
                terms.filter((term) => primarySeries.includes(term)).length,
                terms.filter(hasSubtitle).length
            ];
            const excerpt = getSearchExcerpt(body, terms, contentLanguage === 'kor' ? 110 : 180);
            results.push({ post, excerpt, score });
        });
        return results.sort((a, b) => {
            for (let i = 0; i < a.score.length; i += 1) {
                if (a.score[i] !== b.score[i]) {
                    return b.score[i] - a.score[i];
                }
            }
            return b.post.date.localeCompare(a.post.date) || a.post.id.localeCompare(b.post.id);
        });
    }

    return { getQueryTerms, getSearchExcerpt, splitHighlights, searchPosts };
});

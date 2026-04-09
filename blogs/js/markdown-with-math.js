(function(root, factory) {
    const api = factory();

    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    }

    if (root) {
        root.blogMarkdown = api;
    }
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function isEscaped(text, index) {
        let slashCount = 0;
        let cursor = index - 1;

        while (cursor >= 0 && text[cursor] === '\\') {
            slashCount += 1;
            cursor -= 1;
        }

        return slashCount % 2 === 1;
    }

    function isFenceStart(text, index) {
        const marker = text[index];
        if ((marker !== '`' && marker !== '~') || text.slice(index, index + 3) !== marker.repeat(3)) {
            return null;
        }

        const lineStart = text.lastIndexOf('\n', index - 1) + 1;
        const linePrefix = text.slice(lineStart, index);
        if (!/^[ \t]{0,3}$/.test(linePrefix)) {
            return null;
        }

        let markerLength = 0;
        while (text[index + markerLength] === marker) {
            markerLength += 1;
        }

        return { marker, markerLength };
    }

    function findFenceEnd(text, startIndex, marker, markerLength) {
        let cursor = text.indexOf('\n', startIndex);
        if (cursor === -1) {
            return text.length;
        }

        cursor += 1;

        while (cursor < text.length) {
            const lineEnd = text.indexOf('\n', cursor);
            const nextCursor = lineEnd === -1 ? text.length : lineEnd + 1;
            const line = text.slice(cursor, lineEnd === -1 ? text.length : lineEnd);
            const trimmed = line.trimEnd();

            if (/^[ \t]{0,3}/.test(line)) {
                const stripped = trimmed.replace(/^[ \t]{0,3}/, '');
                if (stripped.startsWith(marker.repeat(markerLength)) && /^[`~]+[ \t]*$/.test(stripped)) {
                    return nextCursor;
                }
            }

            cursor = nextCursor;
        }

        return text.length;
    }

    function findInlineCodeEnd(text, startIndex, tickCount) {
        const marker = '`'.repeat(tickCount);
        const endIndex = text.indexOf(marker, startIndex + tickCount);
        return endIndex === -1 ? -1 : endIndex + tickCount;
    }

    function findDelimitedEnd(text, startIndex, delimiter, allowMultiline) {
        let cursor = startIndex + delimiter.length;

        while (cursor < text.length) {
            if (!allowMultiline && text[cursor] === '\n') {
                return -1;
            }

            if (text.startsWith(delimiter, cursor) && !isEscaped(text, cursor)) {
                return cursor + delimiter.length;
            }

            cursor += 1;
        }

        return -1;
    }

    function protectMathSegments(markdown) {
        const segments = [];
        let result = '';
        let index = 0;

        while (index < markdown.length) {
            const fenceInfo = isFenceStart(markdown, index);
            if (fenceInfo) {
                const fenceEnd = findFenceEnd(markdown, index, fenceInfo.marker, fenceInfo.markerLength);
                result += markdown.slice(index, fenceEnd);
                index = fenceEnd;
                continue;
            }

            if (markdown[index] === '`') {
                let tickCount = 0;
                while (markdown[index + tickCount] === '`') {
                    tickCount += 1;
                }

                const codeEnd = findInlineCodeEnd(markdown, index, tickCount);
                if (codeEnd !== -1) {
                    result += markdown.slice(index, codeEnd);
                    index = codeEnd;
                    continue;
                }
            }

            let mathEnd = -1;
            let mathSource = '';

            if (markdown.startsWith('$$', index) && !isEscaped(markdown, index)) {
                mathEnd = findDelimitedEnd(markdown, index, '$$', true);
            } else if (markdown.startsWith('\\[', index)) {
                mathEnd = findDelimitedEnd(markdown, index, '\\]', true);
            } else if (markdown.startsWith('\\(', index)) {
                mathEnd = findDelimitedEnd(markdown, index, '\\)', true);
            } else if (
                markdown[index] === '$'
                && !isEscaped(markdown, index)
                && markdown[index + 1] !== '$'
            ) {
                mathEnd = findDelimitedEnd(markdown, index, '$', false);
            }

            if (mathEnd !== -1) {
                mathSource = markdown.slice(index, mathEnd);
                const placeholder = `MATH_PLACEHOLDER_${segments.length}_TOKEN`;
                segments.push({ placeholder, source: mathSource });
                result += placeholder;
                index = mathEnd;
                continue;
            }

            result += markdown[index];
            index += 1;
        }

        return { text: result, segments };
    }

    function restoreMathSegments(html, segments) {
        return segments.reduce((output, segment) => {
            return output.split(segment.placeholder).join(escapeHtml(segment.source));
        }, html);
    }

    function parseMarkdownWithMath(markdown, parseMarkdown) {
        const parser = typeof parseMarkdown === 'function' ? parseMarkdown : function(source) {
            return source;
        };
        const protectedMarkdown = protectMathSegments(typeof markdown === 'string' ? markdown : '');
        const parsedHtml = parser(protectedMarkdown.text);
        return restoreMathSegments(parsedHtml, protectedMarkdown.segments);
    }

    return {
        parseMarkdownWithMath
    };
});

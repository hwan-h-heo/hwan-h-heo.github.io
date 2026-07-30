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

    function collectMathSegments(markdown) {
        const segments = [];
        let index = 0;

        while (index < markdown.length) {
            const fenceInfo = isFenceStart(markdown, index);
            if (fenceInfo) {
                const fenceEnd = findFenceEnd(markdown, index, fenceInfo.marker, fenceInfo.markerLength);
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
                    index = codeEnd;
                    continue;
                }
            }

            let mathEnd = -1;
            let delimiter = '';
            let display = false;

            if (markdown.startsWith('$$', index) && !isEscaped(markdown, index)) {
                delimiter = '$$';
                display = true;
                mathEnd = findDelimitedEnd(markdown, index, '$$', true);
            } else if (markdown.startsWith('\\[', index)) {
                delimiter = '\\[';
                display = true;
                mathEnd = findDelimitedEnd(markdown, index, '\\]', true);
            } else if (markdown.startsWith('\\(', index)) {
                delimiter = '\\(';
                mathEnd = findDelimitedEnd(markdown, index, '\\)', true);
            } else if (
                markdown[index] === '$'
                && !isEscaped(markdown, index)
                && markdown[index + 1] !== '$'
            ) {
                delimiter = '$';
                mathEnd = findDelimitedEnd(markdown, index, '$', false);
            }

            if (mathEnd !== -1) {
                segments.push({
                    start: index,
                    end: mathEnd,
                    delimiter,
                    display,
                    source: markdown.slice(index, mathEnd)
                });
                index = mathEnd;
                continue;
            }

            index += 1;
        }

        return segments;
    }

    function collectHtmlCommentRanges(markdown) {
        const ranges = [];
        let cursor = 0;

        while (cursor < markdown.length) {
            const start = markdown.indexOf('<!--', cursor);
            if (start === -1) {
                break;
            }
            const commentEnd = markdown.indexOf('-->', start + 4);
            const end = commentEnd === -1 ? markdown.length : commentEnd + 3;
            ranges.push({ start, end });
            cursor = end;
        }

        return ranges;
    }

    function findDisplayMathSegments(markdown) {
        const source = typeof markdown === 'string' ? markdown : '';
        const commentRanges = collectHtmlCommentRanges(source);
        return collectMathSegments(source).filter((segment) => (
            segment.delimiter === '$$'
            && !commentRanges.some((range) => (
                segment.start >= range.start && segment.start < range.end
            ))
        ));
    }

    function normalizeMathMarkdownWithChanges(markdown) {
        const source = typeof markdown === 'string' ? markdown : '';
        const changes = [];
        const correctionPattern = /\\(?:left|right)[ \t]*[{}]|={2,}|-{2,}/g;
        const starredCommandPattern = /\\([A-Za-z]+)$/;
        const validStarredCommands = new Set([
            'DeclareMathOperator',
            'hspace',
            'operatorname',
            'tag',
            'vspace'
        ]);

        findDisplayMathSegments(source).forEach((segment) => {
            const bodyStart = segment.start + 2;
            const bodyEnd = segment.end - 2;
            const body = source.slice(bodyStart, bodyEnd);

            for (const match of body.matchAll(correctionPattern)) {
                const value = match[0];
                const replacement = value.startsWith('=') || value.startsWith('-')
                    ? value[0]
                    : `${value.slice(0, -1)}\\${value.slice(-1)}`;

                changes.push({
                    start: bodyStart + match.index,
                    end: bodyStart + match.index + value.length,
                    replacement
                });
            }

            for (const match of body.matchAll(/\*[ \t]*\{/g)) {
                const prefix = body.slice(0, match.index);
                const commandMatch = prefix.match(starredCommandPattern);
                if (
                    commandMatch
                    && validStarredCommands.has(commandMatch[1])
                ) {
                    continue;
                }

                changes.push({
                    start: bodyStart + match.index,
                    end: bodyStart + match.index + match[0].length,
                    replacement: '_{'
                });
            }
        });

        if (changes.length === 0) {
            return { markdown: source, changes };
        }

        changes.sort((a, b) => a.start - b.start);

        let result = '';
        let cursor = 0;
        changes.forEach((change) => {
            result += source.slice(cursor, change.start);
            result += change.replacement;
            cursor = change.end;
        });
        result += source.slice(cursor);

        return { markdown: result, changes };
    }

    function normalizeMathMarkdown(markdown) {
        return normalizeMathMarkdownWithChanges(markdown).markdown;
    }

    function mergeDisplayMathBlocks(markdown, sourceBlocks) {
        const source = typeof markdown === 'string' ? markdown : '';
        const blocks = Array.isArray(sourceBlocks)
            ? sourceBlocks.map((block) => ({ ...block }))
            : [];

        findDisplayMathSegments(source).forEach((segment) => {
            const firstIndex = blocks.findIndex((block) => (
                block.end > segment.start && block.start < segment.end
            ));
            if (firstIndex < 0) {
                return;
            }

            let lastIndex = firstIndex;
            while (
                lastIndex + 1 < blocks.length
                && blocks[lastIndex + 1].start < segment.end
            ) {
                lastIndex += 1;
            }

            if (lastIndex === firstIndex) {
                return;
            }

            const start = blocks[firstIndex].start;
            const end = blocks[lastIndex].end;
            blocks.splice(firstIndex, lastIndex - firstIndex + 1, {
                type: 'math',
                raw: source.slice(start, end),
                start,
                end
            });
        });

        return blocks;
    }

    function protectMathSegments(markdown) {
        const segments = [];
        let result = '';
        let cursor = 0;

        collectMathSegments(markdown).forEach((segment) => {
            const placeholder = `MATH_PLACEHOLDER_${segments.length}_TOKEN`;
            result += markdown.slice(cursor, segment.start);
            result += placeholder;
            segments.push({ placeholder, source: segment.source });
            cursor = segment.end;
        });
        result += markdown.slice(cursor);

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
        const normalizedMarkdown = normalizeMathMarkdown(markdown);
        const protectedMarkdown = protectMathSegments(normalizedMarkdown);
        const parsedHtml = parser(protectedMarkdown.text);
        return restoreMathSegments(parsedHtml, protectedMarkdown.segments);
    }

    return {
        findDisplayMathSegments,
        mergeDisplayMathBlocks,
        normalizeMathMarkdown,
        normalizeMathMarkdownWithChanges,
        parseMarkdownWithMath
    };
});

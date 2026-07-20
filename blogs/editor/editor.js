(function() {
    const API_BASE = 'http://localhost:3030/api';
    const POST_ID_PATTERN = /^\d{6}_[A-Za-z0-9_]+$/;
    const CONTENT_DELIMITER = '--- 여기부터 실제 콘텐츠 ---';
    const SIDEBAR_COLLAPSED_KEY = 'blog-editor-sidebar-collapsed';
    const PANE_RATIO_KEY = 'blog-editor-pane-ratio';
    const AUTOSAVE_KEY = 'blog-editor-autosave-v2';
    const AUTOSAVE_DELAY_MS = 1500;
    const DIAGNOSTICS_DELAY_MS = 350;
    const AUTOSAVE_VERSION = 2;
    const DEFAULT_CONTENT = {
        eng: '## Introduction\n\nStart writing your post here.\n',
        kor: '## 소개\n\n여기에서 글을 작성하세요.\n'
    };
    const SNIPPETS = [
        {
            id: 'display-math',
            label: 'Display math',
            description: 'Insert a KaTeX block equation.',
            template: '$$\n__CURSOR__\n$$'
        },
        {
            id: 'math-container',
            label: 'Math container',
            description: 'Scrollable display-math wrapper for long equations.',
            template: '<div class="math-container">\n    $$\n    __CURSOR__\n    $$\n</div>'
        },
        {
            id: 'figure',
            label: 'Figure + caption',
            description: 'Image figure with centered caption.',
            template: '<figure>\n    <img class="img-fluid" src="./assets/image.png" alt="Describe the image" width="80%">\n    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 1.</strong> __CURSOR__</figcaption>\n</figure>'
        },
        {
            id: 'video',
            label: 'YouTube embed',
            description: 'Responsive iframe wrapper for videos.',
            template: '<div class="video-container">\n    <iframe src="https://www.youtube.com/embed/VIDEO_ID" title="YouTube video player" frameborder="0" allowfullscreen></iframe>\n</div>\n\n__CURSOR__'
        },
        {
            id: 'table',
            label: 'Markdown table',
            description: 'Simple two-column comparison table.',
            template: '| Column A | Column B |\n| --- | --- |\n| __CURSOR__ | Value |\n| Value | Value |'
        },
        {
            id: 'details',
            label: 'Details block',
            description: 'Collapsible note or appendix.',
            template: '<details>\n    <summary>Summary</summary>\n\n    __CURSOR__\n</details>'
        }
    ];

    const state = {
        editMode: window.location.port === '3030',
        mode: 'create',
        originalId: '',
        lockedLanguages: ['eng'],
        bootstrap: {
            categories: ['post', 'note'],
            languages: ['eng', 'kor'],
            series: {},
            posts: [],
            featuredPortfolioPosts: []
        },
        activeLanguage: 'eng',
        metadata: {},
        contents: {
            eng: DEFAULT_CONTENT.eng,
            kor: DEFAULT_CONTENT.kor
        },
        ui: {
            sidebarCollapsed: false,
            editorPaneRatio: 0.52
        },
        autosave: {
            timerId: null,
            status: 'idle',
            lastSavedAt: null,
            snapshotAvailable: null
        },
        preview: {
            diagnostics: [],
            diagnosticsTimerId: null,
            diagnosticsRevision: 0,
            outline: []
        }
    };

    const el = {};

    document.addEventListener('DOMContentLoaded', init);

    async function init() {
        cacheElements();
        loadUiPreferences();
        bindEvents();
        marked.setOptions({ gfm: true, breaks: false });
        applySidebarState();
        applyPaneRatio();

        resetWorkspace(true);
        renderModeBadge();
        renderLanguageTabs();
        renderFeatureFields();
        renderKoreanFields();
        updateEditor();
        updatePreview();
        updateStats();
        updateControls();

        if (state.editMode) {
            await loadBootstrap();
        } else {
            renderSelectOptions();
            showFeedback('neutral', 'Preview only mode', [
                'Open this page through `npm run edit` to save drafts, register metadata, and publish posts.'
            ]);
        }

        hydrateRecoverySnapshot();
        renderAutosaveState();
    }

    function cacheElements() {
        [
            'mode-badge',
            'workspace-mode',
            'feedback',
            'workspace',
            'control-sidebar',
            'sidebar-toggle-button',
            'sidebar-toggle-icon',
            'sidebar-toggle-label',
            'existing-post-select',
            'post-id',
            'post-date',
            'post-category',
            'post-series',
            'post-slug',
            'post-updated',
            'post-status',
            'post-tags',
            'post-cover',
            'language-eng',
            'language-kor',
            'title-eng',
            'subtitle-eng',
            'title-kor',
            'subtitle-kor',
            'korean-meta-fields',
            'description-eng',
            'description-kor',
            'korean-publishing-fields',
            'featured-enabled',
            'featured-fields',
            'featured-image',
            'featured-alt',
            'featured-order',
            'featured-order-hint',
            'language-tabs',
            'editor-grid',
            'editor-textarea',
            'preview-content',
            'editor-stats',
            'autosave-status',
            'autosave-meta',
            'restore-session-button',
            'discard-session-button',
            'diagnostics-summary',
            'diagnostics-list',
            'outline-count',
            'outline-list',
            'pane-resizer',
            'modal-overlay',
            'modal-title',
            'modal-body',
            'new-post-button',
            'load-post-button',
            'save-draft-button',
            'publish-button',
            'load-selected-post-button',
            'load-draft-button',
            'download-button',
            'upload-button',
            'modal-close-button'
        ].forEach((id) => {
            el[toCamel(id)] = document.getElementById(id);
        });
    }

    function bindEvents() {
        el.sidebarToggleButton.addEventListener('click', toggleSidebar);
        el.newPostButton.addEventListener('click', handleNewPost);
        el.loadPostButton.addEventListener('click', openLoadPostModal);
        el.saveDraftButton.addEventListener('click', saveDraft);
        el.publishButton.addEventListener('click', publishPost);
        el.loadSelectedPostButton.addEventListener('click', () => {
            if (el.existingPostSelect.value) {
                loadExistingPost(el.existingPostSelect.value);
            }
        });
        el.loadDraftButton.addEventListener('click', openDraftModal);
        el.downloadButton.addEventListener('click', downloadActiveMarkdown);
        el.uploadButton.addEventListener('click', uploadMarkdown);
        el.restoreSessionButton.addEventListener('click', restoreRecoverySnapshot);
        el.discardSessionButton.addEventListener('click', () => discardAutosaveSnapshot(true));
        el.languageKor.addEventListener('change', handleLanguageToggle);
        el.featuredEnabled.addEventListener('change', syncFormToState);
        el.modalOverlay.addEventListener('click', (event) => {
            if (event.target === el.modalOverlay) {
                closeModal();
            }
        });
        el.modalCloseButton.addEventListener('click', closeModal);

        [
            el.postId,
            el.postDate,
            el.postCategory,
            el.postSeries,
            el.postSlug,
            el.postUpdated,
            el.postStatus,
            el.postTags,
            el.postCover,
            el.titleEng,
            el.subtitleEng,
            el.titleKor,
            el.subtitleKor,
            el.descriptionEng,
            el.descriptionKor,
            el.featuredImage,
            el.featuredAlt,
            el.featuredOrder
        ].forEach((field) => {
            field.addEventListener('input', syncFormToState);
            field.addEventListener('change', syncFormToState);
        });

        el.editorTextarea.addEventListener('input', () => {
            state.contents[state.activeLanguage] = el.editorTextarea.value;
            updatePreview();
            updateStats();
            queueAutosave();
        });

        el.editorTextarea.addEventListener('paste', handleImagePaste);

        document.querySelectorAll('[data-insert]').forEach((button) => {
            button.addEventListener('click', () => insertMarkdown(button.dataset.insert));
        });

        document.querySelectorAll('[data-snippet]').forEach((button) => {
            button.addEventListener('click', () => insertSnippet(button.dataset.snippet));
        });

        if (el.paneResizer) {
            el.paneResizer.addEventListener('pointerdown', beginPaneResize);
            el.paneResizer.addEventListener('keydown', handlePaneResizerKeydown);
        }

        window.addEventListener('beforeunload', flushAutosave);
        document.addEventListener('visibilitychange', handleVisibilityChange);
        document.addEventListener('keydown', handleKeyboardShortcuts);
    }

    function loadUiPreferences() {
        try {
            state.ui.sidebarCollapsed = window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === '1';

            const storedRatio = Number.parseFloat(window.localStorage.getItem(PANE_RATIO_KEY));
            if (Number.isFinite(storedRatio)) {
                state.ui.editorPaneRatio = clamp(storedRatio, 0.32, 0.68);
            }
        } catch (error) {
            state.ui.sidebarCollapsed = false;
            state.ui.editorPaneRatio = 0.52;
        }
    }

    function persistUiPreference(key, value) {
        try {
            window.localStorage.setItem(key, value);
        } catch (error) {
            // Ignore storage failures and keep the session usable.
        }
    }

    function toCamel(value) {
        return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    }

    function resetWorkspace(initialLoad) {
        const defaultSeries = Object.keys(state.bootstrap.series || {})[0] || '';
        state.mode = 'create';
        state.originalId = '';
        state.lockedLanguages = ['eng'];
        state.activeLanguage = 'eng';
        state.metadata = {
            id: '',
            date: new Date().toISOString().slice(0, 10),
            category: state.bootstrap.categories[0] || 'post',
            series: defaultSeries,
            languages: ['eng'],
            title_eng: '',
            subtitle_eng: '',
            title_kor: '',
            subtitle_kor: '',
            description_eng: '',
            description_kor: '',
            tags: [],
            cover: '',
            status: 'draft',
            updated: new Date().toISOString().slice(0, 10),
            slug: '',
            featured: false,
            teaserImage: '',
            teaserAlt: '',
            featuredOrder: state.bootstrap.featuredPortfolioPosts.length || 0
        };
        state.contents = {
            eng: DEFAULT_CONTENT.eng,
            kor: DEFAULT_CONTENT.kor
        };

        renderFormFromState();
        if (el.existingPostSelect) {
            el.existingPostSelect.value = '';
        }
        renderLanguageTabs();
        updateEditor();
        updatePreview();
        updateStats();
        renderModeBadge();
        renderFeatureFields();
        renderKoreanFields();

        if (!initialLoad) {
            discardAutosaveSnapshot(false);
        }

        if (!initialLoad) {
            showFeedback('neutral', 'New post workspace', [
                'Metadata and markdown were reset. Drafts are not affected.'
            ]);
        }
    }

    function toggleSidebar() {
        state.ui.sidebarCollapsed = !state.ui.sidebarCollapsed;
        applySidebarState();
        persistUiPreference(SIDEBAR_COLLAPSED_KEY, state.ui.sidebarCollapsed ? '1' : '0');
    }

    function applySidebarState() {
        const collapsed = state.ui.sidebarCollapsed;
        el.workspace.classList.toggle('is-sidebar-collapsed', collapsed);
        el.controlSidebar.classList.toggle('is-collapsed', collapsed);
        el.sidebarToggleButton.setAttribute('aria-expanded', String(!collapsed));
        el.sidebarToggleIcon.className = `bi ${collapsed ? 'bi-layout-sidebar-inset' : 'bi-layout-sidebar'}`;
        el.sidebarToggleLabel.textContent = collapsed ? 'Show controls' : 'Hide controls';
        el.sidebarToggleButton.title = collapsed ? 'Show control panel' : 'Hide control panel';
    }

    function applyPaneRatio() {
        if (!el.editorGrid || !el.paneResizer) {
            return;
        }

        const ratioValue = Math.round(state.ui.editorPaneRatio * 100);
        const percentage = `${(state.ui.editorPaneRatio * 100).toFixed(1)}%`;
        el.editorGrid.style.setProperty('--editor-pane-size', percentage);
        el.paneResizer.setAttribute('aria-valuenow', String(ratioValue));
        el.paneResizer.setAttribute('aria-valuetext', `Editor ${ratioValue} percent width`);
        el.paneResizer.title = 'Drag or use arrow keys to resize the editor and preview panels';
    }

    function beginPaneResize(event) {
        if (window.matchMedia('(max-width: 1180px)').matches) {
            return;
        }

        event.preventDefault();
        document.body.classList.add('is-pane-resizing');

        const onPointerMove = (moveEvent) => {
            const rect = el.editorGrid.getBoundingClientRect();
            const nextRatio = clamp((moveEvent.clientX - rect.left) / rect.width, 0.32, 0.68);
            state.ui.editorPaneRatio = nextRatio;
            applyPaneRatio();
        };

        const stopResize = () => {
            document.body.classList.remove('is-pane-resizing');
            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('pointerup', stopResize);
            persistUiPreference(PANE_RATIO_KEY, String(state.ui.editorPaneRatio));
        };

        window.addEventListener('pointermove', onPointerMove);
        window.addEventListener('pointerup', stopResize);
    }

    function handlePaneResizerKeydown(event) {
        if (window.matchMedia('(max-width: 1180px)').matches) {
            return;
        }

        if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') {
            return;
        }

        event.preventDefault();
        const delta = event.key === 'ArrowLeft' ? -0.03 : 0.03;
        state.ui.editorPaneRatio = clamp(state.ui.editorPaneRatio + delta, 0.32, 0.68);
        applyPaneRatio();
        persistUiPreference(PANE_RATIO_KEY, String(state.ui.editorPaneRatio));
    }

    function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }

    function renderModeBadge() {
        el.modeBadge.textContent = state.editMode ? 'Edit mode' : 'Preview only';
        el.workspaceMode.textContent = state.mode === 'update' ? `Editing ${state.originalId}` : 'New post';
        el.postId.disabled = state.mode === 'update';
        el.languageKor.disabled = state.mode === 'update' && state.lockedLanguages.includes('kor');
    }

    function updateControls() {
        const disabled = !state.editMode;
        [
            el.loadPostButton,
            el.saveDraftButton,
            el.publishButton,
            el.loadSelectedPostButton,
            el.loadDraftButton,
            el.existingPostSelect
        ].forEach((button) => {
            button.disabled = disabled;
        });
    }

    function renderSelectOptions() {
        fillSelect(el.postCategory, state.bootstrap.categories, (value) => ({ value, label: value }));
        fillSelect(
            el.postSeries,
            Object.keys(state.bootstrap.series || {}),
            (value) => ({
                value,
                label: `${value} · ${(state.bootstrap.series[value] || {}).eng || value}`
            })
        );
        fillSelect(
            el.existingPostSelect,
            [''].concat(state.bootstrap.posts.map((post) => post.id)),
            (value) => {
                if (!value) {
                    return { value: '', label: 'Select an existing post' };
                }
                const post = state.bootstrap.posts.find((item) => item.id === value);
                return {
                    value,
                    label: `${value} · ${post ? post.title_eng : value}`
                };
            }
        );

        if (!el.postSeries.value && state.metadata.series) {
            el.postSeries.value = state.metadata.series;
        }
        if (!el.postCategory.value && state.metadata.category) {
            el.postCategory.value = state.metadata.category;
        }

        renderFeaturedOrderHint();
    }

    function fillSelect(select, values, mapper) {
        select.innerHTML = values.map((value) => {
            const item = mapper(value);
            return `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`;
        }).join('');
    }

    async function loadBootstrap() {
        try {
            const response = await fetch(`${API_BASE}/editor-bootstrap`);
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.error || 'Failed to load editor bootstrap data');
            }

            state.bootstrap = payload;
            renderSelectOptions();
            if (!state.metadata.series) {
                state.metadata.series = Object.keys(payload.series || {})[0] || '';
            }
            renderFormFromState();
            renderModeBadge();
            showFeedback('neutral', 'Editor connected', [
                'Draft save and publish APIs are available.'
            ]);
        } catch (error) {
            state.editMode = false;
            renderModeBadge();
            updateControls();
            renderSelectOptions();
            showFeedback('error', 'Editor bootstrap failed', [error.message]);
        }
    }

    function renderFormFromState() {
        el.postId.value = state.metadata.id;
        el.postDate.value = state.metadata.date;
        el.postCategory.value = state.metadata.category;
        el.postSeries.value = state.metadata.series;
        el.postSlug.value = state.metadata.slug;
        el.postUpdated.value = state.metadata.updated;
        el.postStatus.value = state.metadata.status;
        el.postTags.value = state.metadata.tags.join(', ');
        el.postCover.value = state.metadata.cover;
        el.languageKor.checked = state.metadata.languages.includes('kor');
        el.titleEng.value = state.metadata.title_eng;
        el.subtitleEng.value = state.metadata.subtitle_eng;
        el.titleKor.value = state.metadata.title_kor;
        el.subtitleKor.value = state.metadata.subtitle_kor;
        el.descriptionEng.value = state.metadata.description_eng;
        el.descriptionKor.value = state.metadata.description_kor;
        el.featuredEnabled.checked = Boolean(state.metadata.featured);
        el.featuredImage.value = state.metadata.teaserImage;
        el.featuredAlt.value = state.metadata.teaserAlt;
        el.featuredOrder.value = state.metadata.featuredOrder;
    }

    function syncFormToState(options = {}) {
        state.metadata.id = el.postId.value.trim();
        state.metadata.date = el.postDate.value;
        state.metadata.category = el.postCategory.value;
        state.metadata.series = el.postSeries.value;
        state.metadata.slug = el.postSlug.value.trim();
        state.metadata.updated = el.postUpdated.value;
        state.metadata.status = el.postStatus.value;
        state.metadata.tags = [...new Set(el.postTags.value.split(',').map((tag) => tag.trim()).filter(Boolean))];
        state.metadata.cover = el.postCover.value.trim();
        state.metadata.languages = el.languageKor.checked ? ['eng', 'kor'] : ['eng'];
        state.metadata.title_eng = el.titleEng.value.trim();
        state.metadata.subtitle_eng = el.subtitleEng.value.trim();
        state.metadata.title_kor = el.titleKor.value.trim();
        state.metadata.subtitle_kor = el.subtitleKor.value.trim();
        state.metadata.description_eng = el.descriptionEng.value.trim();
        state.metadata.description_kor = el.descriptionKor.value.trim();
        state.metadata.featured = el.featuredEnabled.checked;
        state.metadata.teaserImage = el.featuredImage.value.trim();
        state.metadata.teaserAlt = el.featuredAlt.value.trim();
        state.metadata.featuredOrder = el.featuredOrder.value;

        renderLanguageTabs();
        renderFeatureFields();
        renderKoreanFields();
        renderFeaturedOrderHint();

        if (!options.skipAutosave) {
            queueAutosave();
        }
    }

    function handleLanguageToggle() {
        syncFormToState();
        if (!state.metadata.languages.includes(state.activeLanguage)) {
            state.activeLanguage = 'eng';
        }
        if (state.metadata.languages.includes('kor') && !state.contents.kor) {
            state.contents.kor = DEFAULT_CONTENT.kor;
        }
        updateEditor();
        updatePreview();
    }

    function renderLanguageTabs() {
        const tabs = state.metadata.languages.map((language) => {
            const label = language === 'eng' ? 'English' : 'Korean';
            const activeClass = language === state.activeLanguage ? 'is-active' : '';
            return `<button class="language-tab ${activeClass}" data-language="${language}">${label}</button>`;
        }).join('');

        el.languageTabs.innerHTML = tabs;
        el.languageTabs.querySelectorAll('[data-language]').forEach((button) => {
            button.addEventListener('click', () => switchLanguage(button.dataset.language));
        });
    }

    function switchLanguage(language) {
        state.contents[state.activeLanguage] = el.editorTextarea.value;
        state.activeLanguage = language;
        updateEditor();
        updatePreview();
        updateStats();
        renderLanguageTabs();
        queueAutosave();
    }

    function updateEditor() {
        el.editorTextarea.value = state.contents[state.activeLanguage] || '';
    }

    function renderKoreanFields() {
        const hasKorean = state.metadata.languages.includes('kor');
        el.koreanMetaFields.classList.toggle('is-hidden', !hasKorean);
        el.koreanPublishingFields.classList.toggle('is-hidden', !hasKorean);
    }

    function renderFeatureFields() {
        const isFeatured = el.featuredEnabled.checked;
        el.featuredFields.classList.toggle('is-hidden', !isFeatured);
    }

    function renderFeaturedOrderHint() {
        const featuredCount = state.bootstrap.featuredPortfolioPosts.length;
        el.featuredOrderHint.textContent = `Current featured slots: 0 to ${featuredCount}`;
    }

    function stripLegacyContentPreamble(content) {
        const text = typeof content === 'string' ? content : '';
        const parts = text.split(CONTENT_DELIMITER);
        return parts.length > 1 ? parts.slice(1).join(CONTENT_DELIMITER).trim() : text;
    }

    function resolvePreviewAssetBase(postId, targetPostId) {
        const resolvedId = targetPostId || postId;
        if (!resolvedId) {
            return null;
        }

        return state.editMode
            ? `/posts/${resolvedId}/assets/`
            : `/blogs/posts/${resolvedId}/assets/`;
    }

    function updatePreview() {
        const postId = state.metadata.id.trim();
        const content = stripLegacyContentPreamble(el.editorTextarea.value || '');
        const parseMarkdown = window.blogMarkdown && typeof window.blogMarkdown.parseMarkdownWithMath === 'function'
            ? window.blogMarkdown.parseMarkdownWithMath
            : (source, parser) => parser(source);
        let html = parseMarkdown(content, (source) => marked.parse(source));

        html = html.replace(/\.\/draft-assets\//g, '/editor/draft-assets/');
        html = html.replace(/(src|href)=(["'])\.\/assets\//g, (match, attr, quote) => {
            const basePath = resolvePreviewAssetBase(postId, postId);
            return basePath ? `${attr}=${quote}${basePath}` : match;
        });
        html = html.replace(/(src|href)=(["'])\.\/([0-9]{6}_[A-Za-z0-9_]+)\/assets\//g, (match, attr, quote, targetPostId) => {
            const basePath = resolvePreviewAssetBase(postId, targetPostId);
            return basePath ? `${attr}=${quote}${basePath}` : match;
        });

        el.previewContent.innerHTML = html;

        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(el.previewContent, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true }
                ],
                throwOnError: false
            });
        }

        if (window.Prism && Prism.highlightAllUnder) {
            Prism.highlightAllUnder(el.previewContent);
        }

        renderHeadingOutline();
        queuePreviewDiagnostics(postId, content);
    }

    function updateStats() {
        const text = el.editorTextarea.value || '';
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        el.editorStats.textContent = `${words} words · ${text.length} chars`;
    }

    function insertMarkdown(type) {
        const editors = {
            heading2: ['## ', ''],
            heading3: ['### ', ''],
            bold: ['**', '**'],
            italic: ['*', '*'],
            code: ['```\n', '\n```'],
            quote: ['> ', ''],
            list: ['- ', '']
        };

        if (type === 'link') {
            const url = window.prompt('Link URL');
            if (!url) {
                return;
            }
            const text = window.prompt('Link text', 'Link') || 'Link';
            wrapSelection(`[${text}](${url})`, '');
            return;
        }

        if (type === 'image') {
            const url = window.prompt('Image URL or relative asset path', './assets/image.png');
            if (!url) {
                return;
            }
            const alt = window.prompt('Alt text', 'Image') || 'Image';
            wrapSelection(`![${alt}](${url})`, '');
            return;
        }

        const pair = editors[type];
        if (!pair) {
            return;
        }

        wrapSelection(pair[0], pair[1]);
    }

    function insertSnippet(snippetId) {
        const snippet = SNIPPETS.find((item) => item.id === snippetId);
        if (!snippet) {
            return;
        }

        const start = el.editorTextarea.selectionStart;
        const end = el.editorTextarea.selectionEnd;
        const selected = el.editorTextarea.value.slice(start, end);
        const selectionText = selected || '';
        const withSelection = snippet.template.replace(/{{selection}}/g, selectionText);
        const cursorMarker = '__CURSOR__';
        const cursorIndex = withSelection.indexOf(cursorMarker);
        const replacement = withSelection.replace(cursorMarker, '');

        el.editorTextarea.value = `${el.editorTextarea.value.slice(0, start)}${replacement}${el.editorTextarea.value.slice(end)}`;
        el.editorTextarea.focus();

        if (cursorIndex >= 0) {
            const caret = start + cursorIndex;
            el.editorTextarea.selectionStart = caret;
            el.editorTextarea.selectionEnd = caret;
        } else {
            const caret = start + replacement.length;
            el.editorTextarea.selectionStart = caret;
            el.editorTextarea.selectionEnd = caret;
        }

        state.contents[state.activeLanguage] = el.editorTextarea.value;
        updatePreview();
        updateStats();
        queueAutosave();
    }

    function wrapSelection(before, after) {
        const start = el.editorTextarea.selectionStart;
        const end = el.editorTextarea.selectionEnd;
        const selected = el.editorTextarea.value.slice(start, end);
        const replacement = `${before}${selected}${after}`;

        el.editorTextarea.value = `${el.editorTextarea.value.slice(0, start)}${replacement}${el.editorTextarea.value.slice(end)}`;
        el.editorTextarea.focus();
        el.editorTextarea.selectionStart = start + before.length;
        el.editorTextarea.selectionEnd = start + before.length + selected.length;

        state.contents[state.activeLanguage] = el.editorTextarea.value;
        updatePreview();
        updateStats();
        queueAutosave();
    }

    async function handleImagePaste(event) {
        const items = Array.from(event.clipboardData.items || []);
        const imageItem = items.find((item) => item.type.startsWith('image/'));
        if (!imageItem) {
            return;
        }

        if (!state.editMode) {
            showFeedback('error', 'Image upload unavailable', [
                'Paste-to-upload works only when the editor server is running.'
            ]);
            return;
        }

        event.preventDefault();
        const file = imageItem.getAsFile();
        if (!file) {
            return;
        }

        try {
            const reader = new FileReader();
            reader.onload = async (loadEvent) => {
                try {
                    const webpBlob = await convertToWebP(loadEvent.target.result);
                    const uploadResult = await uploadImageBlob(webpBlob, `image-${Date.now()}.webp`);
                    if (!uploadResult.success) {
                        throw new Error(uploadResult.error || 'Image upload failed');
                    }

                    wrapSelection(`![Image](${uploadResult.relativePath})`, '');
                    showFeedback('success', 'Image uploaded', [
                        `Inserted ${uploadResult.filename} into the current markdown.`
                    ]);
                } catch (error) {
                    showFeedback('error', 'Image upload failed', [error.message]);
                }
            };
            reader.readAsDataURL(file);
        } catch (error) {
            showFeedback('error', 'Image upload failed', [error.message]);
        }
    }

    function convertToWebP(dataUrl) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = image.width;
                canvas.height = image.height;
                canvas.getContext('2d').drawImage(image, 0, 0);
                canvas.toBlob((blob) => {
                    if (blob) {
                        resolve(blob);
                        return;
                    }
                    reject(new Error('Failed to convert image to WebP'));
                }, 'image/webp', 0.92);
            };
            image.onerror = () => reject(new Error('Failed to read pasted image'));
            image.src = dataUrl;
        });
    }

    async function uploadImageBlob(blob, filename) {
        const formData = new FormData();
        formData.append('image', blob, filename);

        const response = await fetch(`${API_BASE}/upload-image`, {
            method: 'POST',
            body: formData
        });

        return response.json();
    }

    function handleVisibilityChange() {
        if (document.visibilityState === 'hidden') {
            flushAutosave();
        }
    }

    function queueAutosave() {
        state.autosave.status = 'pending';
        renderAutosaveState();

        if (state.autosave.timerId) {
            window.clearTimeout(state.autosave.timerId);
        }

        state.autosave.timerId = window.setTimeout(() => {
            saveAutosaveSnapshot();
        }, AUTOSAVE_DELAY_MS);
    }

    function flushAutosave() {
        if (!state.autosave.timerId) {
            return;
        }

        window.clearTimeout(state.autosave.timerId);
        state.autosave.timerId = null;
        saveAutosaveSnapshot();
    }

    function readAutosaveSnapshot() {
        try {
            const raw = window.localStorage.getItem(AUTOSAVE_KEY);
            if (!raw) {
                return null;
            }

            const snapshot = JSON.parse(raw);
            if (!snapshot || snapshot.version !== AUTOSAVE_VERSION) {
                return null;
            }

            return snapshot;
        } catch (error) {
            return null;
        }
    }

    function buildAutosaveSnapshot() {
        state.contents[state.activeLanguage] = el.editorTextarea.value;

        return {
            version: AUTOSAVE_VERSION,
            savedAt: new Date().toISOString(),
            mode: state.mode,
            originalId: state.originalId,
            activeLanguage: state.activeLanguage,
            lockedLanguages: [...state.lockedLanguages],
            metadata: { ...state.metadata, languages: [...state.metadata.languages] },
            contents: {
                eng: state.contents.eng || '',
                kor: state.contents.kor || ''
            }
        };
    }

    function snapshotHasRecoverableChanges(snapshot) {
        if (!snapshot) {
            return false;
        }

        const metadata = snapshot.metadata || {};
        const contents = snapshot.contents || {};
        const englishChanged = (contents.eng || '').trim() && (contents.eng || '').trim() !== DEFAULT_CONTENT.eng.trim();
        const koreanChanged = (contents.kor || '').trim() && (contents.kor || '').trim() !== DEFAULT_CONTENT.kor.trim();

        return Boolean(
            metadata.id
            || metadata.title_eng
            || metadata.title_kor
            || metadata.subtitle_eng
            || metadata.subtitle_kor
            || metadata.featured
            || metadata.teaserImage
            || englishChanged
            || koreanChanged
        );
    }

    function saveAutosaveSnapshot() {
        state.autosave.timerId = null;
        const snapshot = buildAutosaveSnapshot();

        if (!snapshotHasRecoverableChanges(snapshot)) {
            discardAutosaveSnapshot(false);
            return;
        }

        try {
            window.localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(snapshot));
            state.autosave.snapshotAvailable = snapshot;
            state.autosave.lastSavedAt = snapshot.savedAt;
            state.autosave.status = 'saved';
            renderAutosaveState();
        } catch (error) {
            state.autosave.status = 'unavailable';
            renderAutosaveState();
        }
    }

    function discardAutosaveSnapshot(notifyUser) {
        if (state.autosave.timerId) {
            window.clearTimeout(state.autosave.timerId);
            state.autosave.timerId = null;
        }

        try {
            window.localStorage.removeItem(AUTOSAVE_KEY);
        } catch (error) {
            // Ignore storage failures and keep the session usable.
        }

        state.autosave.snapshotAvailable = null;
        state.autosave.lastSavedAt = null;
        state.autosave.status = 'idle';
        renderAutosaveState();

        if (notifyUser) {
            showFeedback('neutral', 'Recovery snapshot removed', [
                'The local autosave snapshot was discarded.'
            ]);
        }
    }

    function hydrateRecoverySnapshot() {
        const snapshot = readAutosaveSnapshot();
        state.autosave.snapshotAvailable = snapshot;
        state.autosave.lastSavedAt = snapshot ? snapshot.savedAt : null;
        state.autosave.status = snapshot ? 'recoverable' : 'idle';
        renderAutosaveState();
    }

    function sanitizeRecoveredMetadata(metadata) {
        const categories = state.bootstrap.categories || [];
        const seriesIds = Object.keys(state.bootstrap.series || {});
        const languages = Array.isArray(metadata.languages) && metadata.languages.includes('kor')
            ? ['eng', 'kor']
            : ['eng'];

        return {
            id: metadata.id || '',
            date: metadata.date || new Date().toISOString().slice(0, 10),
            category: categories.includes(metadata.category) ? metadata.category : (categories[0] || 'post'),
            series: seriesIds.includes(metadata.series) ? metadata.series : (seriesIds[0] || ''),
            languages,
            title_eng: metadata.title_eng || '',
            subtitle_eng: metadata.subtitle_eng || '',
            title_kor: metadata.title_kor || '',
            subtitle_kor: metadata.subtitle_kor || '',
            description_eng: metadata.description_eng || '',
            description_kor: metadata.description_kor || '',
            tags: Array.isArray(metadata.tags) ? metadata.tags : [],
            cover: metadata.cover || '',
            status: metadata.status === 'published' ? 'published' : 'draft',
            updated: metadata.updated || metadata.date || new Date().toISOString().slice(0, 10),
            slug: metadata.slug || '',
            featured: Boolean(metadata.featured),
            teaserImage: metadata.teaserImage || '',
            teaserAlt: metadata.teaserAlt || '',
            featuredOrder: metadata.featuredOrder || state.bootstrap.featuredPortfolioPosts.length || 0
        };
    }

    function restoreRecoverySnapshot() {
        const snapshot = state.autosave.snapshotAvailable || readAutosaveSnapshot();
        if (!snapshot) {
            return;
        }

        state.mode = snapshot.mode === 'update' ? 'update' : 'create';
        state.originalId = snapshot.originalId || '';
        state.lockedLanguages = Array.isArray(snapshot.lockedLanguages) && snapshot.lockedLanguages.length > 0
            ? snapshot.lockedLanguages.filter((language) => language === 'eng' || language === 'kor')
            : ['eng'];
        state.metadata = sanitizeRecoveredMetadata(snapshot.metadata || {});
        state.contents = {
            eng: snapshot.contents && typeof snapshot.contents.eng === 'string' ? snapshot.contents.eng : DEFAULT_CONTENT.eng,
            kor: snapshot.contents && typeof snapshot.contents.kor === 'string' ? snapshot.contents.kor : DEFAULT_CONTENT.kor
        };
        state.activeLanguage = state.metadata.languages.includes(snapshot.activeLanguage) ? snapshot.activeLanguage : 'eng';

        renderFormFromState();
        renderLanguageTabs();
        updateEditor();
        updatePreview();
        updateStats();
        renderKoreanFields();
        renderFeatureFields();
        renderModeBadge();
        updateControls();

        if (el.existingPostSelect) {
            el.existingPostSelect.value = state.mode === 'update' ? state.originalId : '';
        }

        state.autosave.status = 'restored';
        renderAutosaveState();
        queueAutosave();
        showFeedback('neutral', 'Recovered local session', [
            'The latest autosave snapshot was restored into the workspace.'
        ]);
    }

    function formatAutosaveTimestamp(timestamp) {
        if (!timestamp) {
            return 'No local recovery snapshot yet.';
        }

        const date = new Date(timestamp);
        if (Number.isNaN(date.getTime())) {
            return 'A local recovery snapshot is available.';
        }

        return `Last local snapshot: ${date.toLocaleString()}`;
    }

    function renderAutosaveState() {
        if (!el.autosaveStatus || !el.autosaveMeta) {
            return;
        }

        const hasSnapshot = Boolean(state.autosave.snapshotAvailable);
        const statusLabels = {
            idle: 'Autosave idle',
            pending: 'Saving…',
            saved: 'Autosaved',
            recoverable: 'Recovery ready',
            restored: 'Recovered',
            unavailable: 'Autosave off'
        };

        el.autosaveStatus.textContent = statusLabels[state.autosave.status] || 'Autosave';
        el.autosaveMeta.textContent = state.autosave.status === 'pending'
            ? 'Saving the current workspace to local storage…'
            : formatAutosaveTimestamp(state.autosave.lastSavedAt);
        el.restoreSessionButton.disabled = !hasSnapshot;
        el.discardSessionButton.disabled = !hasSnapshot;
    }

    function renderHeadingOutline() {
        if (!el.outlineList || !el.outlineCount) {
            return;
        }

        const markdownHeadings = extractMarkdownHeadings(el.editorTextarea.value || '');
        const previewHeadings = Array.from(el.previewContent.querySelectorAll('h2, h3'));
        const headings = markdownHeadings.map((heading, index) => {
            const previewHeading = previewHeadings[index] || null;
            if (previewHeading && !previewHeading.id) {
                previewHeading.id = `preview-heading-${index}`;
            }

            return {
                ...heading,
                outlineIndex: index,
                previewId: previewHeading ? previewHeading.id : '',
                text: previewHeading && previewHeading.textContent.trim()
                    ? previewHeading.textContent.trim()
                    : heading.text || `Heading ${index + 1}`
            };
        });

        state.preview.outline = headings;
        el.outlineCount.textContent = `${headings.length} heading${headings.length === 1 ? '' : 's'}`;

        if (headings.length === 0) {
            el.outlineList.innerHTML = '<p class="outline-empty">Add `##` or `###` headings to build a clickable outline.</p>';
            return;
        }

        el.outlineList.innerHTML = headings.map((heading) => `
            <button class="outline-button outline-button-level-${heading.level}" type="button" data-outline-index="${heading.outlineIndex}">
                <strong>${escapeHtml(heading.text)}</strong>
                <span>${heading.level === 2 ? 'Section' : 'Subsection'}</span>
            </button>
        `).join('');

        el.outlineList.querySelectorAll('[data-outline-index]').forEach((button) => {
            button.addEventListener('click', () => {
                const headingIndex = Number.parseInt(button.dataset.outlineIndex, 10);
                const targetHeading = state.preview.outline[headingIndex];
                if (!targetHeading) {
                    return;
                }

                focusEditorHeading(targetHeading);

                if (targetHeading.previewId) {
                    const previewTarget = document.getElementById(targetHeading.previewId);
                    if (previewTarget) {
                        el.previewContent.scrollTo({
                            top: Math.max(0, previewTarget.offsetTop - 18),
                            behavior: 'smooth'
                        });
                    }
                }
            });
        });
    }

    function extractMarkdownHeadings(content) {
        const headings = [];
        const lines = content.split('\n');
        let lineOffset = 0;
        let fenceMarker = null;

        lines.forEach((line, index) => {
            const fenceMatch = line.match(/^(```+|~~~+)/);
            if (fenceMatch) {
                const nextFenceMarker = fenceMatch[1][0];
                if (!fenceMarker) {
                    fenceMarker = nextFenceMarker;
                } else if (fenceMarker === nextFenceMarker) {
                    fenceMarker = null;
                }
                lineOffset += line.length + 1;
                return;
            }

            if (!fenceMarker) {
                const headingMatch = line.match(/^(#{2,3})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$/);
                if (headingMatch) {
                    headings.push({
                        level: headingMatch[1].length,
                        text: normalizeHeadingText(headingMatch[2]),
                        lineNumber: index + 1,
                        startIndex: lineOffset,
                        endIndex: lineOffset + line.length
                    });
                }
            }

            lineOffset += line.length + 1;
        });

        return headings;
    }

    function normalizeHeadingText(value) {
        return value
            .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
            .replace(/<[^>]+>/g, '')
            .replace(/[*_`~]/g, '')
            .replace(/\s+/g, ' ')
            .trim();
    }

    function focusEditorHeading(heading) {
        if (!el.editorTextarea) {
            return;
        }

        el.editorTextarea.focus();
        el.editorTextarea.setSelectionRange(heading.startIndex, heading.endIndex);

        const textBeforeHeading = el.editorTextarea.value.slice(0, heading.startIndex);
        const lineIndex = textBeforeHeading.split('\n').length - 1;
        const computedStyle = window.getComputedStyle(el.editorTextarea);
        const lineHeight = Number.parseFloat(computedStyle.lineHeight) || 22;
        const paddingTop = Number.parseFloat(computedStyle.paddingTop) || 0;
        const targetScrollTop = Math.max(
            0,
            (lineIndex * lineHeight) - (el.editorTextarea.clientHeight * 0.35) + paddingTop
        );

        if (typeof el.editorTextarea.scrollTo === 'function') {
            el.editorTextarea.scrollTo({
                top: targetScrollTop,
                behavior: 'smooth'
            });
            return;
        }

        el.editorTextarea.scrollTop = targetScrollTop;
    }

    function queuePreviewDiagnostics(postId, content) {
        if (state.preview.diagnosticsTimerId) {
            window.clearTimeout(state.preview.diagnosticsTimerId);
        }

        const revision = ++state.preview.diagnosticsRevision;
        state.preview.diagnosticsTimerId = window.setTimeout(() => {
            runPreviewDiagnostics(revision, postId, content).catch(() => {
                if (revision === state.preview.diagnosticsRevision) {
                    renderDiagnostics([
                        {
                            severity: 'warning',
                            title: 'Diagnostics interrupted',
                            detail: 'Preview checks could not finish for the latest edit.'
                        }
                    ]);
                }
            });
        }, DIAGNOSTICS_DELAY_MS);
    }

    async function runPreviewDiagnostics(revision, postId, content) {
        const diagnostics = [];
        const resourceChecks = [];
        const seenResources = new Set();

        if (/\.\/assets\//.test(content) && !postId) {
            diagnostics.push({
                severity: 'warning',
                title: 'Post ID missing for local assets',
                detail: 'Set the post ID first so `./assets/...` links can be resolved in preview.'
            });
        }

        const mathErrors = Array.from(el.previewContent.querySelectorAll('.katex-error'));
        const seenMath = new Set();
        mathErrors.forEach((node) => {
            const source = (node.textContent || '').trim();
            const reason = (node.getAttribute('title') || '').trim();
            const key = `${source}::${reason}`;
            if (seenMath.has(key)) {
                return;
            }
            seenMath.add(key);
            diagnostics.push({
                severity: 'error',
                title: 'LaTeX render failed',
                detail: reason ? `${source} — ${reason}` : source
            });
        });

        Array.from(el.previewContent.querySelectorAll('img')).forEach((image) => {
            if (!image.getAttribute('alt') || !image.getAttribute('alt').trim()) {
                diagnostics.push({
                    severity: 'warning',
                    title: 'Image alt text missing',
                    detail: image.getAttribute('src') || 'An image in the preview has no alt text.'
                });
            }
        });

        Array.from(el.previewContent.querySelectorAll('img[src], video[src], source[src], a[href]')).forEach((node) => {
            const attribute = node.hasAttribute('href') ? 'href' : 'src';
            const rawValue = node.getAttribute(attribute);
            if (!rawValue || rawValue.startsWith('#') || rawValue.startsWith('mailto:') || rawValue.startsWith('tel:')) {
                return;
            }

            let url;
            try {
                url = new URL(rawValue, window.location.origin);
            } catch (error) {
                return;
            }

            if (url.origin !== window.location.origin) {
                return;
            }

            if (
                !url.pathname.startsWith('/posts/')
                && !url.pathname.startsWith('/blogs/posts/')
                && !url.pathname.startsWith('/editor/draft-assets/')
                && !url.pathname.startsWith('/blogs/editor/draft-assets/')
            ) {
                return;
            }

            const resourceKey = `${attribute}:${url.pathname}`;
            if (seenResources.has(resourceKey)) {
                return;
            }

            seenResources.add(resourceKey);
            resourceChecks.push({
                kind: node.tagName.toLowerCase(),
                url
            });
        });

        const availabilityResults = await Promise.all(resourceChecks.map(async (resource) => {
            const exists = await checkLocalResource(resource.url.href);
            return exists ? null : {
                severity: 'warning',
                title: `${resource.kind} resource not found`,
                detail: resource.url.pathname
            };
        }));

        if (revision !== state.preview.diagnosticsRevision) {
            return;
        }

        renderDiagnostics(diagnostics.concat(availabilityResults.filter(Boolean)));
    }

    async function checkLocalResource(url) {
        try {
            let response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
            if (response.status === 405 || response.status === 501) {
                response = await fetch(url, { method: 'GET', cache: 'no-store' });
            }
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    function renderDiagnostics(diagnostics) {
        if (!el.diagnosticsSummary || !el.diagnosticsList) {
            return;
        }

        state.preview.diagnostics = diagnostics;

        const errorCount = diagnostics.filter((item) => item.severity === 'error').length;
        const warningCount = diagnostics.filter((item) => item.severity === 'warning').length;

        if (diagnostics.length === 0) {
            el.diagnosticsSummary.className = 'feedback feedback-success diagnostics-summary';
            el.diagnosticsSummary.innerHTML = '<strong>Preview diagnostics</strong><p>No math, asset, or accessibility issues detected in the current preview.</p>';
            el.diagnosticsList.innerHTML = '';
            return;
        }

        const summaryType = errorCount > 0 ? 'error' : 'warning';
        el.diagnosticsSummary.className = `feedback feedback-${summaryType} diagnostics-summary`;
        el.diagnosticsSummary.innerHTML = `<strong>Preview diagnostics</strong><p>${errorCount} error(s), ${warningCount} warning(s) found in the current preview.</p>`;
        el.diagnosticsList.innerHTML = diagnostics.map((item) => `
            <div class="diagnostic-item diagnostic-item-${escapeHtml(item.severity)}">
                <strong>${escapeHtml(item.title)}</strong>
                <p>${escapeHtml(item.detail)}</p>
            </div>
        `).join('');
    }

    function buildPayload() {
        syncFormToState({ skipAutosave: true });
        state.contents[state.activeLanguage] = el.editorTextarea.value;

        const payload = {
            mode: state.mode,
            originalId: state.originalId,
            post: {
                id: state.metadata.id,
                title_eng: state.metadata.title_eng,
                subtitle_eng: state.metadata.subtitle_eng,
                title_kor: state.metadata.title_kor,
                subtitle_kor: state.metadata.subtitle_kor,
                description_eng: state.metadata.description_eng,
                description_kor: state.metadata.description_kor,
                tags: [...state.metadata.tags],
                cover: state.metadata.cover,
                status: state.metadata.status,
                updated: state.metadata.updated,
                slug: state.metadata.slug,
                date: state.metadata.date,
                category: state.metadata.category,
                series: state.metadata.series,
                languages: [...state.metadata.languages]
            },
            contents: {},
            featured: {
                enabled: Boolean(state.metadata.featured),
                teaserImage: state.metadata.teaserImage,
                teaserAlt: state.metadata.teaserAlt,
                order: state.metadata.featuredOrder
            }
        };

        payload.post.languages.forEach((language) => {
            payload.contents[language] = state.contents[language] || '';
        });

        return payload;
    }

    function validatePayload(payload) {
        const errors = [];

        if (!payload.post.id) {
            errors.push('Post ID is required.');
        } else if (!POST_ID_PATTERN.test(payload.post.id)) {
            errors.push('Post ID must match YYMMDD_slug and use only letters, numbers, and underscores.');
        }

        if (!payload.post.title_eng) {
            errors.push('English title is required.');
        }

        if (!payload.post.slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(payload.post.slug)) {
            errors.push('URL slug is required and must use lowercase words separated by hyphens.');
        }

        if (!payload.post.date || !/^\d{4}-\d{2}-\d{2}$/.test(payload.post.date)) {
            errors.push('Date must use YYYY-MM-DD.');
        }

        if (!payload.post.updated || !/^\d{4}-\d{2}-\d{2}$/.test(payload.post.updated)) {
            errors.push('Updated date must use YYYY-MM-DD.');
        }

        if (payload.post.status === 'published') {
            if (!payload.post.description_eng) {
                errors.push('Published posts need an English description.');
            }
            if (!payload.post.cover || payload.post.cover === '/assets/blog_bg.jpeg') {
                errors.push('Published posts need a post-specific cover image.');
            }
            if (payload.post.tags.length === 0) {
                errors.push('Published posts need at least one tag.');
            }
        }

        if (!state.bootstrap.categories.includes(payload.post.category)) {
            errors.push('Select a valid category.');
        }

        if (!Object.prototype.hasOwnProperty.call(state.bootstrap.series, payload.post.series)) {
            errors.push('Select a valid series.');
        }

        if (payload.post.languages.includes('kor') && !payload.post.title_kor) {
            errors.push('Korean title is required when Korean content is enabled.');
        }

        if (payload.featured.enabled && !payload.featured.teaserImage) {
            errors.push('Featured posts need a teaser image.');
        }

        const duplicate = state.bootstrap.posts.find((post) => post.id === payload.post.id);
        if (state.mode === 'create' && duplicate) {
            errors.push(`Post ID "${payload.post.id}" already exists.`);
        }

        if (state.mode === 'update' && state.originalId !== payload.post.id) {
            errors.push('Renaming an existing post ID is not supported.');
        }

        return errors;
    }

    async function publishPost() {
        if (!state.editMode) {
            showFeedback('error', 'Publish unavailable', [
                'Start the local editor server with `npm run edit` first.'
            ]);
            return;
        }

        const payload = buildPayload();
        const errors = validatePayload(payload);
        if (errors.length > 0) {
            showFeedback('error', 'Validation failed', errors);
            return;
        }

        const actionLabel = state.mode === 'create' ? 'Create this post and write metadata/files?' : 'Update this post?';
        if (!window.confirm(actionLabel)) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/post-bundle`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                showFeedback('error', 'Publish failed', result.details || [result.error || 'Unknown publish error']);
                return;
            }

            state.mode = 'update';
            state.originalId = payload.post.id;
            state.lockedLanguages = [...payload.post.languages];
            await loadBootstrap();
            discardAutosaveSnapshot(false);
            showFeedback('success', 'Publish complete', [
                `Updated site metadata for ${payload.post.id}.`,
                ...result.savedFiles.map((file) => `${file.language}: ${file.path}`)
            ]);
            renderModeBadge();
        } catch (error) {
            showFeedback('error', 'Publish failed', [error.message]);
        }
    }

    function handleNewPost() {
        if (!window.confirm('Reset the current workspace and start a new post?')) {
            return;
        }
        resetWorkspace(false);
    }

    async function loadExistingPost(postId) {
        if (!state.editMode) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/post-bundle/${encodeURIComponent(postId)}`);
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || 'Failed to load post');
            }

            state.mode = 'update';
            state.originalId = result.post.id;
            state.lockedLanguages = [...result.post.languages];
            state.activeLanguage = 'eng';
            state.metadata = {
                id: result.post.id,
                date: result.post.date,
                category: result.post.category,
                series: result.post.series,
                languages: [...result.post.languages],
                title_eng: result.post.title_eng || '',
                subtitle_eng: result.post.subtitle_eng || '',
                title_kor: result.post.title_kor || '',
                subtitle_kor: result.post.subtitle_kor || '',
                description_eng: result.post.description_eng || '',
                description_kor: result.post.description_kor || '',
                tags: Array.isArray(result.post.tags) ? result.post.tags : [],
                cover: result.post.cover || '',
                status: result.post.status || 'published',
                updated: result.post.updated || result.post.date,
                slug: result.post.slug || '',
                featured: Boolean(result.featured),
                teaserImage: result.featured ? result.featured.teaserImage : '',
                teaserAlt: result.featured ? (result.featured.teaserAlt || '') : '',
                featuredOrder: result.featured ? result.featured.order : state.bootstrap.featuredPortfolioPosts.length
            };
            state.contents = {
                eng: stripLegacyContentPreamble(result.contents.eng || DEFAULT_CONTENT.eng),
                kor: stripLegacyContentPreamble(result.contents.kor || DEFAULT_CONTENT.kor)
            };

            renderFormFromState();
            renderLanguageTabs();
            updateEditor();
            updatePreview();
            updateStats();
            renderKoreanFields();
            renderFeatureFields();
            renderModeBadge();
            el.existingPostSelect.value = result.post.id;
            closeModal();
            queueAutosave();
            showFeedback('success', 'Post loaded', [
                `Editing ${result.post.id}. Post ID is fixed for updates.`
            ]);
        } catch (error) {
            showFeedback('error', 'Load failed', [error.message]);
        }
    }

    async function saveDraft() {
        if (!state.editMode) {
            return;
        }

        const suggestedName = `${state.metadata.id || 'untitled'}-${state.activeLanguage}.md`;
        const filename = window.prompt('Draft filename', suggestedName);
        if (!filename) {
            return;
        }

        state.contents[state.activeLanguage] = el.editorTextarea.value;

        try {
            const response = await fetch(`${API_BASE}/draft`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    content: state.contents[state.activeLanguage]
                })
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Failed to save draft');
            }
            showFeedback('success', 'Draft saved', [
                `${result.filename} was saved for the ${state.activeLanguage} workspace.`
            ]);
        } catch (error) {
            showFeedback('error', 'Draft save failed', [error.message]);
        }
    }

    async function openDraftModal() {
        if (!state.editMode) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/drafts`);
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || 'Failed to load drafts');
            }

            openModal('Load Draft', buildDraftModalHtml(result.drafts || []));
            el.modalBody.querySelectorAll('[data-load-draft]').forEach((button) => {
                button.addEventListener('click', () => loadDraft(button.dataset.loadDraft));
            });
            el.modalBody.querySelectorAll('[data-delete-draft]').forEach((button) => {
                button.addEventListener('click', () => deleteDraft(button.dataset.deleteDraft));
            });
        } catch (error) {
            showFeedback('error', 'Draft list failed', [error.message]);
        }
    }

    function buildDraftModalHtml(drafts) {
        if (drafts.length === 0) {
            return '<p class="panel-copy">No drafts found.</p>';
        }

        return `<div class="draft-list">${drafts.map((draft) => `
            <div class="draft-item">
                <div>
                    <div class="draft-name">${escapeHtml(draft)}</div>
                    <div class="hint">Loads into the active ${state.activeLanguage} tab only.</div>
                </div>
                <div class="draft-actions">
                    <button class="button button-secondary" data-load-draft="${escapeHtml(draft)}">Load</button>
                    <button class="button button-secondary" data-delete-draft="${escapeHtml(draft)}">Delete</button>
                </div>
            </div>
        `).join('')}</div>`;
    }

    async function loadDraft(filename) {
        try {
            const response = await fetch(`${API_BASE}/draft/${encodeURIComponent(filename)}`);
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || 'Failed to load draft');
            }

                state.contents[state.activeLanguage] = result.content;
                state.contents[state.activeLanguage] = stripLegacyContentPreamble(state.contents[state.activeLanguage]);
                updateEditor();
                updatePreview();
                updateStats();
                queueAutosave();
            closeModal();
            showFeedback('success', 'Draft loaded', [
                `${filename} replaced the ${state.activeLanguage} content in this workspace.`
            ]);
        } catch (error) {
            showFeedback('error', 'Draft load failed', [error.message]);
        }
    }

    async function deleteDraft(filename) {
        if (!window.confirm(`Delete draft "${filename}"?`)) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/draft/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Failed to delete draft');
            }
            await openDraftModal();
        } catch (error) {
            showFeedback('error', 'Draft delete failed', [error.message]);
        }
    }

    function downloadActiveMarkdown() {
        const filename = `${state.metadata.id || 'draft'}-${state.activeLanguage}.md`;
        const blob = new Blob([el.editorTextarea.value], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
    }

    function uploadMarkdown() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.md,.markdown,.txt';
        input.onchange = () => {
            const file = input.files && input.files[0];
            if (!file) {
                return;
            }

            const reader = new FileReader();
            reader.onload = (event) => {
                state.contents[state.activeLanguage] = stripLegacyContentPreamble(event.target.result);
                updateEditor();
                updatePreview();
                updateStats();
                queueAutosave();
                showFeedback('success', 'Markdown uploaded', [
                    `${file.name} replaced the ${state.activeLanguage} content in this workspace.`
                ]);
            };
            reader.readAsText(file);
        };
        input.click();
    }

    function handleKeyboardShortcuts(event) {
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
            event.preventDefault();
            if (state.editMode) {
                saveDraft();
            } else {
                downloadActiveMarkdown();
            }
        }

        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'b') {
            event.preventDefault();
            insertMarkdown('bold');
        }

        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'i') {
            event.preventDefault();
            insertMarkdown('italic');
        }
    }

    function openLoadPostModal() {
        const posts = state.bootstrap.posts || [];
        openModal(
            'Load Existing Post',
            posts.length === 0
                ? '<p class="panel-copy">No posts available.</p>'
                : `<div class="draft-list">${posts.map((post) => `
                    <div class="draft-item">
                        <div>
                            <div class="draft-name">${escapeHtml(post.id)}</div>
                            <div class="hint">${escapeHtml(post.title_eng)}</div>
                        </div>
                        <button class="button button-secondary" data-load-post="${escapeHtml(post.id)}">Load</button>
                    </div>
                `).join('')}</div>`
        );

        el.modalBody.querySelectorAll('[data-load-post]').forEach((button) => {
            button.addEventListener('click', () => loadExistingPost(button.dataset.loadPost));
        });
    }

    function openModal(title, html) {
        el.modalTitle.textContent = title;
        el.modalBody.innerHTML = html;
        el.modalOverlay.classList.remove('is-hidden');
    }

    function closeModal() {
        el.modalOverlay.classList.add('is-hidden');
    }

    function showFeedback(type, title, details) {
        el.feedback.className = `feedback feedback-${type}`;
        const detailLines = Array.isArray(details) && details.length > 0
            ? `<ul>${details.map((detail) => `<li>${escapeHtml(detail)}</li>`).join('')}</ul>`
            : '';
        el.feedback.innerHTML = `<strong>${escapeHtml(title)}</strong>${detailLines || '<p></p>'}`;
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }
})();

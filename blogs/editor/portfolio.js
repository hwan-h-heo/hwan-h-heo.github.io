(function() {
    const API_BASE = 'http://localhost:3030/api';
    const SECTION_LABELS = {
        portfolioProjects: 'Projects',
        publications: 'Publications',
        talks: 'Talks',
        projectPages: 'Project Pages'
    };

    const state = {
        section: 'portfolioProjects',
        selectedIndex: 0,
        portfolioCategories: ['research', 'app', 'per'],
        data: {
            portfolioProjects: [],
            publications: [],
            talks: [],
            projectPages: []
        },
        projectPageBundles: {},
        projectPageDirty: false,
        dirty: false
    };

    const el = {};

    document.addEventListener('DOMContentLoaded', init);

    async function init() {
        cacheElements();
        bindEvents();
        await loadBundle();
    }

    function cacheElements() {
        [
            'save-status',
            'add-button',
            'save-button',
            'list-title',
            'item-count',
            'item-list',
            'form-title',
            'feedback',
            'portfolio-form',
            'preview-kind',
            'portfolio-preview',
            'move-up-button',
            'move-down-button',
            'duplicate-button',
            'delete-button',
            'project-create-modal',
            'project-create-form',
            'project-create-close-button',
            'project-create-cancel-button',
            'project-create-slug',
            'project-create-title',
            'project-create-subtitle',
            'project-create-card',
            'project-create-card-image',
            'project-asset-upload'
        ].forEach((id) => {
            el[toCamel(id)] = document.getElementById(id);
        });
    }

    function bindEvents() {
        document.querySelectorAll('.section-tab').forEach((button) => {
            button.addEventListener('click', () => selectSection(button.dataset.section));
        });

        el.addButton.addEventListener('click', addItem);
        el.saveButton.addEventListener('click', saveBundle);
        el.moveUpButton.addEventListener('click', () => moveItem(-1));
        el.moveDownButton.addEventListener('click', () => moveItem(1));
        el.duplicateButton.addEventListener('click', duplicateItem);
        el.deleteButton.addEventListener('click', deleteItem);
        el.projectCreateForm.addEventListener('submit', handleProjectCreateSubmit);
        el.projectCreateCard.addEventListener('change', updateProjectCreateCardImageState);
        el.projectAssetUpload.addEventListener('change', handleProjectAssetUpload);
        el.projectCreateCloseButton.addEventListener('click', closeProjectCreateModal);
        el.projectCreateCancelButton.addEventListener('click', closeProjectCreateModal);
        el.projectCreateModal.addEventListener('click', (event) => {
            if (event.target === el.projectCreateModal) {
                closeProjectCreateModal();
            }
        });
        el.portfolioForm.addEventListener('input', handleFormInput);
        el.portfolioForm.addEventListener('change', handleFormInput);
        el.portfolioForm.addEventListener('click', handleFormClick);

        window.addEventListener('beforeunload', (event) => {
            if (!state.dirty && !state.projectPageDirty) {
                return;
            }
            event.preventDefault();
            event.returnValue = '';
        });
    }

    async function loadBundle() {
        setStatus('Loading', false);
        try {
            const [portfolioResponse, pagesResponse] = await Promise.all([
                fetch(`${API_BASE}/portfolio-bundle`),
                fetch(`${API_BASE}/project-pages`)
            ]);
            if (!portfolioResponse.ok) {
                throw new Error('Failed to load portfolio data.');
            }
            if (!pagesResponse.ok) {
                throw new Error('Failed to load project pages.');
            }
            const bundle = await portfolioResponse.json();
            const pagesPayload = await pagesResponse.json();
            state.portfolioCategories = bundle.portfolioCategories || state.portfolioCategories;
            state.data.portfolioProjects = bundle.portfolioProjects || [];
            state.data.publications = bundle.publications || [];
            state.data.talks = bundle.talks || [];
            state.data.projectPages = pagesPayload.pages || [];
            state.dirty = false;
            state.projectPageDirty = false;
            setStatus('Loaded', false);
            render();
            showFeedback('neutral', 'Loaded portfolio data.', 'Edit an item, then save portfolio.');
        } catch (error) {
            setStatus('Offline', false);
            showFeedback('error', 'Unable to load data.', error.message);
        }
    }

    async function saveBundle() {
        if (state.section === 'projectPages') {
            await saveCurrentProjectPage();
            return;
        }

        syncFormToState();
        setStatus('Saving', true);

        try {
            const response = await fetch(`${API_BASE}/portfolio-bundle`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state.data)
            });
            const payload = await response.json();
            if (!response.ok || !payload.success) {
                throw new Error(payload.error || 'Failed to save portfolio data.');
            }
            state.dirty = false;
            setStatus('Saved', false);
            showFeedback(
                'success',
                'Saved portfolio data.',
                `${payload.projectCount} projects, ${payload.publicationCount} publications, ${payload.talkCount} talks written to site-data.json.`
            );
        } catch (error) {
            setStatus('Save failed', false);
            showFeedback('error', 'Save failed.', error.message);
        }
    }

    async function loadProjectPage(pagePath) {
        if (!pagePath || state.projectPageBundles[pagePath] !== undefined) {
            renderForm();
            renderPreview();
            return;
        }

        setStatus('Loading page', false);
        try {
            const response = await fetch(`${API_BASE}/project-page?path=${encodeURIComponent(pagePath)}`);
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.error || 'Failed to load project page.');
            }
            state.projectPageBundles[pagePath] = {
                metadata: payload.metadata || {},
                content: payload.content || '',
                legacyHtml: payload.legacyHtml || ''
            };
            setStatus(state.projectPageDirty ? 'Unsaved page' : 'Loaded', false);
            renderForm();
            renderPreview();
        } catch (error) {
            setStatus('Page load failed', false);
            showFeedback('error', 'Unable to load project page.', error.message);
        }
    }

    async function saveCurrentProjectPage() {
        const item = currentItem();
        if (!item) {
            return;
        }

        syncFormToState();
        setStatus('Saving page', true);

        try {
            const response = await fetch(`${API_BASE}/project-page`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    path: item.path,
                    metadata: state.projectPageBundles[item.path]?.metadata,
                    content: state.projectPageBundles[item.path]?.content
                })
            });
            const payload = await response.json();
            if (!response.ok || !payload.success) {
                throw new Error(payload.error || 'Failed to save project page.');
            }
            state.projectPageDirty = false;
            setStatus('Saved page', false);
            showFeedback('success', 'Saved project page.', `${payload.path} (${payload.bytes} bytes)`);
        } catch (error) {
            setStatus('Page save failed', false);
            showFeedback('error', 'Project page save failed.', error.message);
        }
    }

    function selectSection(section) {
        syncFormToState();
        state.section = section;
        state.selectedIndex = 0;
        render();
        if (section === 'projectPages') {
            loadProjectPage(currentItem()?.path);
        }
    }

    async function addItem() {
        if (state.section === 'projectPages') {
            openProjectCreateModal();
            return;
        }

        syncFormToState();
        const items = currentItems();
        items.push(createDefaultItem(state.section, items.length + 1));
        state.selectedIndex = items.length - 1;
        markDirty();
        render();
    }

    function openProjectCreateModal() {
        const nextIndex = (state.data.projectPages || []).length + 1;
        el.projectCreateSlug.value = `new_project_${nextIndex}`;
        el.projectCreateTitle.value = `New Project Page ${nextIndex}`;
        el.projectCreateSubtitle.value = 'Project';
        el.projectCreateCard.checked = false;
        el.projectCreateCardImage.value = '';
        updateProjectCreateCardImageState();
        el.projectCreateModal.classList.remove('is-hidden');
        el.projectCreateSlug.focus();
    }

    function updateProjectCreateCardImageState() {
        el.projectCreateCardImage.disabled = !el.projectCreateCard.checked;
        el.projectCreateCardImage.closest('.field')?.classList.toggle('is-disabled', !el.projectCreateCard.checked);
    }

    function closeProjectCreateModal() {
        el.projectCreateModal.classList.add('is-hidden');
    }

    async function handleProjectCreateSubmit(event) {
        event.preventDefault();
        await createProjectPage({
            slug: el.projectCreateSlug.value,
            title: el.projectCreateTitle.value,
            subtitle: el.projectCreateSubtitle.value,
            createPortfolioCard: el.projectCreateCard.checked,
            cardImage: el.projectCreateCardImage.value
        });
    }

    async function createProjectPage(inputPayload) {
        setStatus('Creating page', true);

        try {
            const response = await fetch(`${API_BASE}/project-page-create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputPayload)
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Failed to create project page.');
            }

            state.data.projectPages.push(result.page);
            if (result.portfolioCardCreated) {
                await loadBundle();
                selectSection('projectPages');
                const createdIndex = state.data.projectPages.findIndex((page) => page.path === result.page.path);
                state.selectedIndex = Math.max(0, createdIndex);
            } else {
                state.selectedIndex = state.data.projectPages.length - 1;
            }
            setStatus('Created page', false);
            showFeedback('success', 'Created project page.', `${result.page.path} is ready to edit.`);
            closeProjectCreateModal();
            render();
            await loadProjectPage(result.page.path);
        } catch (error) {
            setStatus('Create failed', false);
            showFeedback('error', 'Project page create failed.', error.message);
        }
    }

    function duplicateItem() {
        syncFormToState();
        const items = currentItems();
        const item = currentItem();
        if (!item) {
            return;
        }
        const duplicate = JSON.parse(JSON.stringify(item));
        if (duplicate.id) {
            duplicate.id = `${duplicate.id}-copy`;
        }
        items.splice(state.selectedIndex + 1, 0, duplicate);
        state.selectedIndex += 1;
        markDirty();
        render();
    }

    function deleteItem() {
        if (state.section === 'projectPages') {
            deleteProjectPage();
            return;
        }

        const items = currentItems();
        if (items.length === 0) {
            return;
        }
        items.splice(state.selectedIndex, 1);
        state.selectedIndex = Math.max(0, Math.min(state.selectedIndex, items.length - 1));
        markDirty();
        render();
    }

    async function deleteProjectPage() {
        const item = currentItem();
        if (!item) {
            return;
        }
        if (item.projectId) {
            showFeedback('warning', 'Cannot delete linked page.', `Remove portfolio card "${item.projectId}" first.`);
            return;
        }
        if (!window.confirm(`Delete ${item.path}? A snapshot backup will be created first.`)) {
            return;
        }

        setStatus('Deleting page', true);
        try {
            const response = await fetch(`${API_BASE}/project-page-delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: item.path })
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Failed to delete project page.');
            }

            delete state.projectPageBundles[item.path];
            state.data.projectPages.splice(state.selectedIndex, 1);
            state.selectedIndex = Math.max(0, Math.min(state.selectedIndex, state.data.projectPages.length - 1));
            setStatus('Deleted page', false);
            showFeedback('success', 'Deleted project page.', `Backup saved to ${result.backup}.`);
            render();
            await loadProjectPage(currentItem()?.path);
        } catch (error) {
            setStatus('Delete failed', false);
            showFeedback('error', 'Project page delete failed.', error.message);
        }
    }

    function moveItem(direction) {
        syncFormToState();
        const items = currentItems();
        const nextIndex = state.selectedIndex + direction;
        if (nextIndex < 0 || nextIndex >= items.length) {
            return;
        }
        const [item] = items.splice(state.selectedIndex, 1);
        items.splice(nextIndex, 0, item);
        state.selectedIndex = nextIndex;
        markDirty();
        render();
    }

    function render() {
        renderTabs();
        renderList();
        renderForm();
        renderPreview();
    }

    function renderTabs() {
        document.querySelectorAll('.section-tab').forEach((button) => {
            button.classList.toggle('is-active', button.dataset.section === state.section);
        });
    }

    function renderList() {
        const items = currentItems();
        el.listTitle.textContent = SECTION_LABELS[state.section];
        el.itemCount.textContent = `${items.length} items`;
        el.itemList.innerHTML = items.map((item, index) => `
            <button class="portfolio-list-item ${index === state.selectedIndex ? 'is-active' : ''}" data-index="${index}">
                <div class="portfolio-list-title-row">
                    <strong>${escapeHtml(getItemTitle(item))}</strong>
                    ${renderListBadge(item)}
                </div>
                <span>${escapeHtml(getItemMeta(item))}</span>
            </button>
        `).join('');

        el.itemList.querySelectorAll('.portfolio-list-item').forEach((button) => {
            button.addEventListener('click', () => {
                syncFormToState();
                state.selectedIndex = Number.parseInt(button.dataset.index, 10);
                render();
                if (state.section === 'projectPages') {
                    loadProjectPage(currentItem()?.path);
                }
            });
        });
    }

    function renderForm() {
        const item = currentItem();
        el.formTitle.textContent = item ? getItemTitle(item) : `No ${SECTION_LABELS[state.section].toLowerCase()} yet`;
        el.portfolioForm.innerHTML = item ? renderFields(item) : '<p class="panel-copy">Add an item to start editing this section.</p>';

        const hasItem = Boolean(item);
        [el.moveUpButton, el.moveDownButton, el.duplicateButton, el.deleteButton].forEach((button) => {
            button.disabled = !hasItem || (state.section === 'projectPages' && button !== el.deleteButton);
        });
        if (state.section === 'projectPages' && item?.projectId) {
            el.deleteButton.disabled = true;
        }
        el.addButton.disabled = false;
        el.saveButton.textContent = state.section === 'projectPages' ? 'Save page' : 'Save portfolio';
    }

    function renderFields(item) {
        if (state.section === 'projectPages') {
            const bundle = state.projectPageBundles[item.path];
            const metadata = bundle?.metadata || {};
            const content = bundle?.content;
            const isLoaded = bundle !== undefined;
            return `
                <label class="field">
                    <span>Path</span>
                    <input name="path" type="text" value="${escapeHtml(item.path)}" disabled />
                </label>
                ${field('projectTitle', 'Title', metadata.title || item.title || '')}
                ${textarea('projectHeroTitle', 'Hero title HTML', metadata.heroTitle || metadata.title || item.title || '', true)}
                ${textarea('projectSubtitles', 'Subtitles (one per line)', (metadata.subtitles || []).join('\n'))}
                ${textarea('projectDescription', 'Meta description', metadata.description || '')}
                ${textarea('projectKeywords', 'Meta keywords', metadata.keywords || '')}
                <div class="project-snippet-bar" aria-label="Project snippets">
                    <div class="project-snippet-group">
                        <button class="button button-secondary" type="button" data-project-snippet="overview">Overview block</button>
                        <button class="button button-secondary" type="button" data-project-snippet="details">Details box</button>
                        <button class="button button-secondary" type="button" data-project-snippet="table">Comparison table</button>
                    </div>
                    <div class="project-snippet-group">
                        <button class="button button-secondary" type="button" data-project-snippet="figure">Figure</button>
                        <button class="button button-secondary" type="button" data-project-snippet="video">Video</button>
                        <button class="button button-secondary" type="button" data-project-upload-asset>Upload asset</button>
                    </div>
                </div>
                <label class="field">
                    <span>Markdown content</span>
                    <textarea name="projectPageContent" class="code-field project-page-editor" spellcheck="false">${escapeHtml(isLoaded ? content : 'Loading project page...')}</textarea>
                </label>
            `;
        }

        if (state.section === 'portfolioProjects') {
            return `
                ${field('id', 'ID', item.id)}
                ${field('title', 'Title', item.title)}
                ${textarea('summary', 'Hover summary', item.summary)}
                ${field('url', 'URL', item.url)}
                <label class="field">
                    <span>Categories</span>
                    <div class="portfolio-category-row">
                        ${state.portfolioCategories.map((category) => `
                            <label class="checkbox-pill">
                                <input name="categories" type="checkbox" value="${escapeHtml(category)}" ${item.categories && item.categories.includes(category) ? 'checked' : ''} />
                                <span>${escapeHtml(category)}</span>
                            </label>
                        `).join('')}
                    </div>
                </label>
                <label class="checkbox-row checkbox-row-stacked">
                    <input name="external" type="checkbox" ${item.external ? 'checked' : ''} />
                    <span>Open as external link</span>
                </label>
                ${field('badge', 'Badge', item.badge || '')}
                ${field('image', 'Image', item.image || '')}
                ${field('gif', 'Hover GIF', item.gif || '')}
                ${field('video', 'Video MP4', item.video || '')}
                ${field('poster', 'Video poster', item.poster || '')}
                ${field('alt', 'Alt text', item.alt || '')}
            `;
        }

        if (state.section === 'publications') {
            return `
                ${field('title', 'Title', item.title)}
                ${textarea('authorsHtml', 'Authors HTML', item.authorsHtml, true)}
                ${textarea('venueHtml', 'Venue HTML', item.venueHtml, true)}
                ${textarea('linksJson', 'Links JSON', JSON.stringify(item.links || [], null, 2), true)}
            `;
        }

        return `
            ${field('title', 'Title', item.title)}
            ${textarea('venueHtml', 'Venue HTML', item.venueHtml, true)}
            ${field('date', 'Date label', item.date)}
        `;
    }

    function handleFormInput() {
        syncFormToState();
        if (state.section === 'projectPages') {
            state.projectPageDirty = true;
            setStatus('Unsaved page', false);
        } else {
            markDirty();
        }
        renderList();
        renderPreview();
    }

    function handleFormClick(event) {
        const uploadButton = event.target.closest('[data-project-upload-asset]');
        if (uploadButton) {
            el.projectAssetUpload.value = '';
            el.projectAssetUpload.click();
            return;
        }

        const snippetButton = event.target.closest('[data-project-snippet]');
        if (!snippetButton) {
            return;
        }
        insertProjectSnippet(snippetButton.dataset.projectSnippet);
    }

    async function handleProjectAssetUpload() {
        const item = currentItem();
        const file = el.projectAssetUpload.files?.[0];
        if (!item || state.section !== 'projectPages' || !file) {
            return;
        }

        const formData = new FormData();
        formData.append('path', item.path);
        formData.append('asset', file);
        setStatus('Uploading asset', true);

        try {
            const response = await fetch(`${API_BASE}/project-asset-upload`, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Failed to upload project asset.');
            }

            insertProjectAssetSnippet(result.relativePath, result.mimeType);
            setStatus('Unsaved page', false);
            showFeedback('success', 'Uploaded project asset.', `${result.relativePath} was inserted into the page content.`);
        } catch (error) {
            setStatus('Upload failed', false);
            showFeedback('error', 'Project asset upload failed.', error.message);
        }
    }

    function insertProjectAssetSnippet(relativePath, mimeType) {
        const isVideo = String(mimeType || '').startsWith('video/');
        const snippet = isVideo
            ? `<div class="text-center">\n  <video controls muted loop class="img-fluid">\n    <source src="${relativePath}" type="${mimeType || 'video/mp4'}">\n  </video>\n</div>`
            : `<figure>\n  <img class="img-fluid" src="${relativePath}" alt="Describe the image">\n  <figcaption class="text-center mt-2"><strong>Figure.</strong> Caption here.</figcaption>\n</figure>`;

        insertTextIntoProjectContent(snippet);
    }

    function insertProjectSnippet(snippetId) {
        const snippets = {
            overview: ':::{.container .portfolio-details-container .col-11}\\n:::{.row .gy-4}\\n:::{.col-lg-8}\\n:::{.portfolio-description}\\n## Project Overview\\n\\nDescribe the project here.\\n:::\\n:::\\n:::\\n:::',
            details: ':::{.portfolio-info}\\n### Project Details\\n\\n- **Category**: Research\\n- **Technology**: Add technologies here\\n- **Project URL**: <a href="#" target="_blank" class="portfolio-link"><i class="bi bi-link-45deg"></i> Project Page</a>\\n:::',
            figure: '<figure>\\n  <img class="img-fluid" src="assets/image.png" alt="Describe the image">\\n  <figcaption class="text-center mt-2"><strong>Figure.</strong> Caption here.</figcaption>\\n</figure>',
            video: '<div class="text-center">\\n  <video controls muted loop class="img-fluid">\\n    <source src="assets/video.mp4" type="video/mp4">\\n  </video>\\n</div>',
            table: '<table>\\n  <tr>\\n    <th>Method A</th>\\n    <th>Method B</th>\\n  </tr>\\n  <tr>\\n    <td>Result A</td>\\n    <td>Result B</td>\\n  </tr>\\n</table>'
        };
        const snippet = snippets[snippetId];
        if (!snippet) {
            return;
        }

        insertTextIntoProjectContent(snippet);
    }

    function insertTextIntoProjectContent(snippet) {
        const textarea = el.portfolioForm.elements.projectPageContent;
        if (!textarea) {
            return;
        }

        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        const before = textarea.value.slice(0, start);
        const after = textarea.value.slice(end);
        const insertion = `${before && !before.endsWith('\\n') ? '\\n\\n' : ''}${snippet}${after && !after.startsWith('\\n') ? '\\n\\n' : ''}`;
        textarea.value = `${before}${insertion}${after}`;
        textarea.focus();
        const cursor = before.length + insertion.length;
        textarea.setSelectionRange(cursor, cursor);
        handleFormInput();
    }

    function syncFormToState() {
        const item = currentItem();
        if (!item || !el.portfolioForm.elements.length) {
            return;
        }

        if (state.section === 'projectPages') {
            const bundle = state.projectPageBundles[item.path];
            const contentField = el.portfolioForm.elements.projectPageContent;
            if (contentField && bundle) {
                bundle.metadata = {
                    title: getFieldValue('projectTitle'),
                    heroTitle: getFieldValue('projectHeroTitle'),
                    subtitles: getFieldValue('projectSubtitles').split('\n').map((line) => line.trim()).filter(Boolean),
                    description: getFieldValue('projectDescription'),
                    keywords: getFieldValue('projectKeywords'),
                    sourceBackup: bundle.metadata.sourceBackup || ''
                };
                bundle.content = contentField.value;
            }
            return;
        }

        if (state.section === 'portfolioProjects') {
            ['id', 'title', 'summary', 'url', 'badge', 'image', 'gif', 'video', 'poster', 'alt'].forEach((key) => {
                item[key] = getFieldValue(key);
            });
            item.categories = [...el.portfolioForm.querySelectorAll('input[name="categories"]:checked')].map((input) => input.value);
            item.external = Boolean(el.portfolioForm.querySelector('input[name="external"]')?.checked);
            removeEmptyFields(item, ['badge', 'image', 'gif', 'video', 'poster', 'alt']);
            if (!item.external) {
                delete item.external;
            }
            return;
        }

        if (state.section === 'publications') {
            item.title = getFieldValue('title');
            item.authorsHtml = getFieldValue('authorsHtml');
            item.venueHtml = getFieldValue('venueHtml');
            try {
                const parsedLinks = JSON.parse(getFieldValue('linksJson') || '[]');
                item.links = Array.isArray(parsedLinks) ? parsedLinks : [];
            } catch (error) {
                item.links = [];
                showFeedback('error', 'Invalid links JSON.', 'Links must be a JSON array.');
            }
            return;
        }

        item.title = getFieldValue('title');
        item.venueHtml = getFieldValue('venueHtml');
        item.date = getFieldValue('date');
    }

    function currentItems() {
        return state.data[state.section] || [];
    }

    function currentItem() {
        return currentItems()[state.selectedIndex] || null;
    }

    function createDefaultItem(section, index) {
        if (section === 'portfolioProjects') {
            return {
                id: `new-project-${index}`,
                title: 'New Portfolio Project',
                summary: 'Short hover summary',
                url: 'projects/new-project/',
                categories: ['app'],
                image: 'assets/thumbnails/example.png',
                alt: 'New project teaser'
            };
        }

        if (section === 'publications') {
            return {
                title: 'New Publication',
                authorsHtml: '<strong>Hwan Heo</strong>',
                venueHtml: 'Venue, Year',
                links: []
            };
        }

        return {
            title: 'New Talk',
            venueHtml: 'Venue',
            date: 'Month Year'
        };
    }

    function getItemTitle(item) {
        return item.title || item.id || 'Untitled';
    }

    function getItemMeta(item) {
        if (state.section === 'projectPages') {
            return [item.path, item.projectId || item.source].filter(Boolean).join(' · ');
        }
        if (state.section === 'portfolioProjects') {
            return [item.id, (item.categories || []).join(', ')].filter(Boolean).join(' · ');
        }
        if (state.section === 'publications') {
            return stripHtml(item.venueHtml || '');
        }
        return [stripHtml(item.venueHtml || ''), item.date].filter(Boolean).join(' · ');
    }

    function renderListBadge(item) {
        if (state.section !== 'projectPages') {
            return '';
        }

        const linked = Boolean(item.projectId);
        return `<span class="list-badge ${linked ? 'list-badge-linked' : 'list-badge-standalone'}">${linked ? 'linked' : 'standalone'}</span>`;
    }

    function renderPreview() {
        const item = currentItem();
        el.previewKind.textContent = state.section === 'portfolioProjects'
            ? 'Project'
            : state.section === 'publications'
                ? 'Publication'
                : state.section === 'talks'
                    ? 'Talk'
                    : 'Page';

        if (!item) {
            el.portfolioPreview.innerHTML = '<div class="preview-empty">No item selected.</div>';
            return;
        }

        if (state.section === 'projectPages') {
            el.portfolioPreview.innerHTML = renderProjectPagePreview(item);
            return;
        }

        if (state.section === 'portfolioProjects') {
            el.portfolioPreview.innerHTML = renderProjectPreview(item);
            return;
        }

        if (state.section === 'publications') {
            el.portfolioPreview.innerHTML = renderPublicationPreview(item);
            return;
        }

        el.portfolioPreview.innerHTML = renderTalkPreview(item);
    }

    function renderProjectPagePreview(item) {
        const bundle = state.projectPageBundles[item.path];
        if (!bundle) {
            return '<div class="preview-empty">Loading project page...</div>';
        }

        const metadata = bundle.metadata || {};
        const sourceContent = bundle.content || '';
        const parsedContentHtml = sourceContent.trimStart().startsWith('<')
            ? sourceContent
            : window.marked
                ? parseProjectMarkdown(bundle.content || '', (source) => marked.parse(source))
                : escapeHtml(bundle.content || '').replace(/\n/g, '<br>');
        const contentHtml = rewriteProjectRelativeUrls(parsedContentHtml, item.path);
        if (bundle.legacyHtml) {
            return `
                <div class="project-page-preview is-frame">
                    <iframe class="project-page-preview-frame" title="${escapeHtml(metadata.title || item.path)} preview" sandbox="allow-scripts allow-same-origin" srcdoc="${escapeHtml(renderLegacyProjectPreview(bundle.legacyHtml, metadata, contentHtml))}"></iframe>
                </div>
            `;
        }

        return `
            <div class="project-page-preview">
                <h1>${metadata.heroTitle || escapeHtml(metadata.title || item.title || item.path)}</h1>
                ${(metadata.subtitles || []).map((subtitle) => `<p class="project-preview-subtitle">${subtitle}</p>`).join('')}
                <div class="project-preview-content">${contentHtml}</div>
            </div>
        `;
    }

    function renderLegacyProjectPreview(legacyHtml, metadata, contentHtml) {
        const title = metadata.title || 'Project';
        const heroTitle = metadata.heroTitle || title;
        const subtitles = Array.isArray(metadata.subtitles) ? metadata.subtitles : [];
        const detailsInner = `
            <div class="row gx-5 justify-content-center">
                <div class="text-center mb-5 col-11 col-lg-10 col-xl-8 col-xxl-7">
                    <h1 class="display-6 fw-bolder mb-0"><span class="text-gradient d-inline">${heroTitle}</span></h1>
                    ${subtitles.map((subtitle) => `<div class="fs-3 fw-light text-muted">${subtitle}</div>`).join('')}
                </div>
            </div>

            ${contentHtml}
        `;

        let html = legacyHtml
            .replace(/<title>[\s\S]*?<\/title>/i, `<title>${escapeHtml(title)}</title>`)
            .replace(/<li class=["']current["']>[\s\S]*?<\/li>/i, `<li class="current">${escapeHtml(title)}</li>`);

        html = html.replace(
            /(<section\s+id=["']portfolio-details["'][^>]*>)([\s\S]*?)(<\/section>\s*<!--\s*\/Portfolio Details Section\s*-->)/i,
            (match, openTag, oldContent, closeTag) => `${openTag}${detailsInner}${closeTag}`
        );

        return html.replace(/(<head[^>]*>)/i, `$1<base href="/">`);
    }

    function renderProjectPreview(item) {
        const media = item.video
            ? `
                <video poster="${escapeHtml(resolvePreviewUrl(item.poster || item.image || ''))}" muted loop playsinline>
                    <source src="${escapeHtml(resolvePreviewUrl(item.video))}" type="video/mp4">
                </video>
            `
            : `<img src="${escapeHtml(resolvePreviewUrl(item.image || ''))}" alt="${escapeHtml(item.alt || item.title || '')}">`;
        const badge = item.badge ? `<div class="preview-project-badge">${escapeHtml(item.badge)}</div>` : '';

        return `
            <div class="preview-project-card">
                <div class="preview-project-media">${media}</div>
                ${badge}
                <div class="preview-project-content">
                    <h3>${escapeHtml(item.title || 'Untitled project')}${item.external ? ' <i class="bi bi-box-arrow-up-right"></i>' : ''}</h3>
                    <p>${escapeHtml(item.summary || 'No summary yet.')}</p>
                </div>
            </div>
        `;
    }

    function renderPublicationPreview(item) {
        return `
            <div class="preview-list-card">
                <h3><i class="bi bi-file-earmark-text"></i> ${escapeHtml(item.title || 'Untitled publication')}</h3>
                <p><em>${item.authorsHtml || ''}<br/>${item.venueHtml || ''}</em></p>
                ${renderPreviewLinks(item.links || [])}
            </div>
        `;
    }

    function renderTalkPreview(item) {
        const titleHtml = item.titleHtml || escapeHtml(item.title || 'Untitled talk');

        return `
            <div class="preview-list-card">
                <h3>${titleHtml}</h3>
                <p><em>${item.venueHtml || ''}</em>, <em>${escapeHtml(item.date || '')}</em></p>
            </div>
        `;
    }

    function renderPreviewLinks(links) {
        if (!links.length) {
            return '';
        }

        return `
            <div class="preview-link-row">
                ${links.map((link) => `<span class="preview-link-pill"><i class="bi ${escapeHtml(link.icon || 'bi-link')}"></i> ${escapeHtml(link.label || 'link')}</span>`).join('')}
            </div>
        `;
    }

    function resolvePreviewUrl(value) {
        const path = String(value || '').trim();
        if (!path || /^(https?:)?\/\//.test(path) || path.startsWith('data:') || path.startsWith('/')) {
            return path;
        }
        return `/${path}`;
    }

    function rewriteProjectRelativeUrls(html, pagePath) {
        const basePath = `/${pagePath.replace(/index\.html$/, '')}`;
        return String(html || '').replace(/\s(src|href)=["']([^"']+)["']/g, (match, attr, value) => {
            if (
                !value ||
                /^(https?:)?\/\//.test(value) ||
                value.startsWith('/') ||
                value.startsWith('#') ||
                value.startsWith('mailto:') ||
                value.startsWith('data:')
            ) {
                return match;
            }

            return ` ${attr}="${basePath}${value}"`;
        });
    }

    function parseProjectMarkdown(markdown, parseMarkdown) {
        const lines = String(markdown || '').split(/\r?\n/);
        const output = [];
        const markdownBuffer = [];
        const stack = [];

        const flushMarkdown = () => {
            const source = markdownBuffer.join('\n').trim();
            markdownBuffer.length = 0;
            if (!source) {
                return;
            }
            output.push(parseMarkdown(`${normalizeMarkdownExtensions(source)}\n`));
        };

        lines.forEach((line) => {
            const openMatch = line.match(/^:::\s*\{([^}]*)\}\s*$/);
            if (openMatch) {
                flushMarkdown();
                output.push(`<div${renderProjectAttributes(openMatch[1])}>`);
                stack.push('div');
                return;
            }

            if (/^:::\s*$/.test(line) && stack.length) {
                flushMarkdown();
                stack.pop();
                output.push('</div>');
                return;
            }

            markdownBuffer.push(line);
        });

        flushMarkdown();
        while (stack.length) {
            stack.pop();
            output.push('</div>');
        }

        return enhanceProjectMedia(output.join('\n'));
    }

    function enhanceProjectMedia(html) {
        return String(html || '')
            .replace(/<img\b([^>]*)>/gi, (match, attrs) => {
                const nextAttrs = ensureLazyImageSource(mergeClassAttribute(attrs, 'img-fluid'));
                return `<img${nextAttrs}>`;
            })
            .replace(/<video\b([^>]*)>/gi, (match, attrs) => {
                let nextAttrs = mergeClassAttribute(attrs, 'img-fluid');
                nextAttrs = mergeClassAttribute(nextAttrs, 'project-video')
                    .replace(/\sstyle=["'][^"']*(?:width|max-width)[^"']*["']/gi, '');
                if (!/\splaysinline\b/i.test(nextAttrs)) {
                    nextAttrs += ' playsinline';
                }
                return `<video${nextAttrs}>`;
            });
    }

    function normalizeMarkdownExtensions(markdown) {
        return String(markdown || '').replace(
            /!\[([^\]]*)\]\(([^)\s]+)\)\{([^}]*)\}/g,
            renderMarkdownImage
        );
    }

    function renderMarkdownImage(match, alt, src, attrs) {
        return `<img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}"${renderProjectAttributes(attrs)}>`;
    }

    function mergeClassAttribute(attrs, className) {
        if (/\sclass\s*=/.test(attrs)) {
            return attrs.replace(/\sclass=(["'])(.*?)\1/i, (match, quote, value) => {
                const classes = new Set(value.split(/\s+/).filter(Boolean));
                classes.add(className);
                return ` class=${quote}${[...classes].join(' ')}${quote}`;
            });
        }

        return `${attrs} class="${className}"`;
    }

    function ensureLazyImageSource(attrs) {
        if (!/\sclass=(["'])(?:(?!\1).)*\blazy-image\b/i.test(attrs) || /\sdata-src=/.test(attrs)) {
            return attrs;
        }

        const src = getAttributeValue(attrs, 'src');
        return src ? `${attrs} data-src="${escapeHtml(src)}"` : attrs;
    }

    function getAttributeValue(attrs, name) {
        const match = String(attrs || '').match(new RegExp(`\\s${name}=(["'])(.*?)\\1`, 'i'));
        return match ? match[2] : '';
    }

    function renderProjectAttributes(source) {
        const tokens = String(source || '').match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [];
        const classes = [];
        const attrs = [];
        let id = '';

        tokens.forEach((token) => {
            if (token.startsWith('.') && token.length > 1) {
                classes.push(token.slice(1));
                return;
            }
            if (token.startsWith('#') && token.length > 1) {
                id = token.slice(1);
                return;
            }
            const separatorIndex = token.indexOf('=');
            if (separatorIndex === -1) {
                attrs.push([token, '']);
                return;
            }
            attrs.push([
                token.slice(0, separatorIndex),
                stripQuotes(token.slice(separatorIndex + 1))
            ]);
        });

        const rendered = [];
        if (id) {
            rendered.push(`id="${escapeHtml(id)}"`);
        }
        if (classes.length) {
            rendered.push(`class="${escapeHtml(classes.join(' '))}"`);
        }
        attrs.forEach(([name, value]) => {
            if (!/^[A-Za-z_:][A-Za-z0-9_:.-]*$/.test(name)) {
                return;
            }
            rendered.push(value ? `${name}="${escapeHtml(value)}"` : name);
        });

        return rendered.length ? ` ${rendered.join(' ')}` : '';
    }

    function stripQuotes(value) {
        const text = String(value || '');
        if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
            return text.slice(1, -1);
        }
        return text;
    }


    function field(name, label, value) {
        return `
            <label class="field">
                <span>${escapeHtml(label)}</span>
                <input name="${escapeHtml(name)}" type="text" value="${escapeHtml(value || '')}" />
            </label>
        `;
    }

    function textarea(name, label, value, code) {
        return `
            <label class="field">
                <span>${escapeHtml(label)}</span>
                <textarea name="${escapeHtml(name)}" class="${code ? 'code-field' : ''}">${escapeHtml(value || '')}</textarea>
            </label>
        `;
    }

    function getFieldValue(name) {
        return String(el.portfolioForm.elements[name]?.value || '').trim();
    }

    function removeEmptyFields(item, keys) {
        keys.forEach((key) => {
            if (!item[key]) {
                delete item[key];
            }
        });
    }

    function markDirty() {
        state.dirty = true;
        setStatus('Unsaved', false);
    }

    function setStatus(text, disabled) {
        el.saveStatus.textContent = text;
        el.saveButton.disabled = Boolean(disabled);
    }

    function showFeedback(type, title, message) {
        el.feedback.className = `feedback feedback-${type}`;
        el.feedback.innerHTML = `<strong>${escapeHtml(title)}</strong><p>${escapeHtml(message)}</p>`;
    }

    function stripHtml(value) {
        const template = document.createElement('template');
        template.innerHTML = value;
        return template.content.textContent || template.content.innerText || '';
    }

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function toCamel(value) {
        return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    }
})();

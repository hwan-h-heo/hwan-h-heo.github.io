document.addEventListener('DOMContentLoaded', function() {
    const postsContainer = document.querySelector('#posts-tab');
    const seriesContainer = document.querySelector('#series-tab');
    const notesContainer = document.querySelector('#notes-tab');
    const langToggleButton = document.getElementById('lang-toggle-main');
    const searchInput = document.getElementById('searchInput');
    const sortedPosts = postsData.sort((a, b) => new Date(b.date) - new Date(a.date));

    /**
     * 단일 포스트 미리보기 HTML을 생성하는 함수 (재사용을 위해 분리)
     * @param {object} post - 포스트 데이터 객체
     * @param {string} lang - 현재 언어 ('eng' 또는 'kor')
     * @returns {string} - 생성된 HTML 문자열
     */
    function createPostPreviewHTML(post, lang) {
        // 현재 언어의 제목/부제가 없으면 영어 버전으로 대체(Fallback)합니다.
        const title = post[`title_${lang}`] || post.title_eng;
        const subtitle = post[`subtitle_${lang}`] || post.subtitle_eng;

        if (!title) {
            return '';
        }

        return `
        <div class="post-preview">
            <a href="./posts/?id=${post.id}">
                <h3 class="post-title">${title}</h3>
                ${subtitle ? `<h5 class="post-subtitle">${subtitle}</h5>` : ''}
            </a>
            <p class="post-meta">
                ${new Date(post.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
            </p>
        </div>
        <hr class="my-4" />
        `;
    }


    /**
     * @param {string} lang - 현재 언어
     */
    function renderAllPosts(lang) {
        if (!postsContainer) return;

        let allPostsHTML = '';
        sortedPosts.forEach(post => {
            allPostsHTML += createPostPreviewHTML(post, lang);
        });

        if (allPostsHTML === '') {
            postsContainer.innerHTML = '<p>작성된 글이 없습니다.</p>';
        } else {
            postsContainer.innerHTML = allPostsHTML;
        }
    }

    /**
     * [Notes 탭] 카테고리가 'note'인 글 목록만 렌더링합니다.
     * @param {string} lang - 현재 언어
     */
    function renderNotes(lang) {
        if (!notesContainer) return;

        const notePosts = sortedPosts.filter(post => post.category === 'note');
        let notesHTML = '';

        if (notePosts.length > 0) {
            notePosts.forEach(post => {
                notesHTML += createPostPreviewHTML(post, lang);
            });
        }
        
        if (notesHTML === '') {
            notesContainer.innerHTML = '<p>작성된 노트가 없습니다.</p>';
        } else {
            notesContainer.innerHTML = notesHTML;
        }
    }

    /**
     * [Series 탭] 시리즈별로 글을 묶어서 렌더링합니다.
     * @param {string} lang - 현재 언어
     */
    function renderSeries(lang) {
        if (!seriesContainer) return;

        const postsBySeries = {};
        sortedPosts.forEach(post => {
            if (post.series) {
                if (!postsBySeries[post.series]) {
                    postsBySeries[post.series] = [];
                }
                postsBySeries[post.series].push(post);
            }
        });

        let seriesHTML = '';
        for (const seriesId in postsBySeries) {
            const seriesTitle = (seriesInfo[seriesId] && seriesInfo[seriesId][lang]) || 
                                (seriesInfo[seriesId] && seriesInfo[seriesId]['eng']) || 
                                'Unnamed Series';
            
            const postsInSeries = postsBySeries[seriesId];

            seriesHTML += `
                <div class="series-group mb-5">
                    <h3 class="series-title">${seriesTitle}</h3>
                    <ol class="series-post-list">
            `;
            
            postsInSeries.forEach(post => {
                const title = post[`title_${lang}`] || post.title_eng;
                seriesHTML += `
                    <li>
                        <a href="./posts/?id=${post.id}">${title}</a>
                        <span class="post-meta-sm">${new Date(post.date).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}</span>
                    </li>
                `;
            });

            seriesHTML += `</ol></div>`;
        }

        if (seriesHTML === '') {
            seriesContainer.innerHTML = '<p>작성된 시리즈가 없습니다.</p>';
        } else {
            seriesContainer.innerHTML = seriesHTML;
        }
    }


    /**
     * 모든 탭의 콘텐츠를 특정 언어로 다시 렌더링하는 마스터 함수
     * @param {string} lang - 렌더링할 언어
     */
    function renderAllTabs(lang) {
        renderAllPosts(lang);
        renderNotes(lang);
        renderSeries(lang);

        if (langToggleButton) {
            langToggleButton.textContent = (lang === 'eng') ? 'kor' : 'eng';
        }
    }

    // 언어 토글 버튼에 클릭 이벤트 리스너 추가
    if (langToggleButton) {
        langToggleButton.addEventListener('click', function() {
            const currentLang = localStorage.getItem('language') || 'eng';
            const newLang = (currentLang === 'eng') ? 'kor' : 'eng';
            
            // 1. localStorage에 새로운 언어 설정 저장
            localStorage.setItem('language', newLang);
            
            // 2. 저장된 새 언어로 화면 전체를 다시 렌더링
            renderAllTabs(newLang);
        });
    }

    const initialLang = localStorage.getItem('language') || 'eng';
    renderAllTabs(initialLang);

});
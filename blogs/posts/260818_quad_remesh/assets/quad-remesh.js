(function () {
    function initializeVideoPlayback() {
        const videos = Array.from(document.querySelectorAll('.quad-remesh-video video'));
        if (!videos.length || !('IntersectionObserver' in window)) {
            return;
        }

        const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        const states = new Map(videos.map((video) => [video, {
            visible: false,
            userPaused: false,
            starting: false
        }]));
        let pageActive = true;

        function canAutoplay(video) {
            const state = states.get(video);
            return state.visible
                && !state.userPaused
                && pageActive
                && !document.hidden
                && !reducedMotion.matches
                && !video.closest('details:not([open])');
        }

        function syncPlayback(video) {
            const state = states.get(video);
            if (!canAutoplay(video)) {
                video.pause();
                return;
            }

            if (!video.paused || state.starting) {
                return;
            }

            state.starting = true;
            video.play().catch(() => {
                // Native controls remain available if the browser blocks playback.
            }).finally(() => {
                state.starting = false;
                if (!canAutoplay(video)) {
                    video.pause();
                }
            });
        }

        function syncAllVideos() {
            videos.forEach(syncPlayback);
        }

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                states.get(entry.target).visible = entry.isIntersecting && entry.intersectionRatio >= 0.35;
                syncPlayback(entry.target);
            });
        }, { threshold: [0, 0.35] });

        videos.forEach((video) => {
            video.muted = true;
            video.addEventListener('pause', () => {
                if (canAutoplay(video) && !video.ended) {
                    states.get(video).userPaused = true;
                }
            });
            video.addEventListener('play', () => {
                states.get(video).userPaused = false;
            });
            observer.observe(video);
        });

        document.querySelectorAll('details').forEach((details) => {
            if (details.querySelector('.quad-remesh-video')) {
                details.addEventListener('toggle', syncAllVideos);
            }
        });
        reducedMotion.addEventListener('change', syncAllVideos);
        document.addEventListener('visibilitychange', syncAllVideos);
        window.addEventListener('pagehide', () => {
            pageActive = false;
            syncAllVideos();
        });
        window.addEventListener('pageshow', () => {
            pageActive = true;
            syncAllVideos();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeVideoPlayback, { once: true });
    } else {
        initializeVideoPlayback();
    }
})();

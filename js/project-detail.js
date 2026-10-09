(() => {
    'use strict';
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const videos = [...document.querySelectorAll('.case-study video')];
    const records = videos.map(video => {
        const toggle = video.closest('figure')?.querySelector('.case-media-toggle');
        const record = {video, toggle, visible:false, manualPaused:reduced.matches || !!navigator.connection?.saveData, internalPause:false};
        video.removeAttribute('autoplay');
        if (!toggle) video.controls = true;
        function syncLabel() {
            if (!toggle) return;
            const playing = !video.paused;
            toggle.textContent = playing ? 'Pause' : 'Play';
            toggle.setAttribute('aria-pressed', String(playing));
            toggle.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} project showcase`);
        }
        record.sync = () => {
            if (record.visible && !document.hidden && !record.manualPaused) {
                video.play().catch(() => {record.manualPaused = true; syncLabel();});
            } else if (!video.paused) {
                record.internalPause = true;
                video.pause();
            }
            syncLabel();
        };
        video.addEventListener('pause', () => {
            if (!record.internalPause) record.manualPaused = true;
            record.internalPause = false; syncLabel();
        });
        video.addEventListener('play', () => {record.manualPaused = false; syncLabel();});
        toggle?.addEventListener('click', () => {record.manualPaused = !video.paused; record.sync();});
        syncLabel();
        return record;
    });
    const observer = new IntersectionObserver(entries => {
        for (const entry of entries) {
            const record = records.find(record => record.video === entry.target);
            record.visible = entry.isIntersecting && entry.intersectionRatio >= .2;
            record.sync();
        }
    }, {threshold:[0,.2]});
    records.forEach(record => observer.observe(record.video));
    document.addEventListener('visibilitychange', () => records.forEach(record => record.sync()));
    addEventListener('pagehide', () => records.forEach(record => {record.visible = false; record.sync();}));
    addEventListener('pageshow', () => records.forEach(record => {
        const rect = record.video.getBoundingClientRect();
        const height = Math.min(rect.bottom,innerHeight) - Math.max(rect.top,0);
        record.visible = rect.width > 0 && height >= rect.height * .2;
        record.sync();
    }));
    reduced.addEventListener('change', () => { if (reduced.matches) records.forEach(record => {record.manualPaused = true; record.sync();}); });
    const headings = [...document.querySelectorAll('#overview,#contributions,.case-article h2')];
    const contents = [...document.querySelectorAll('.case-contents a')];
    let queued = false;
    function markSection() {
        queued = false;
        const active = headings.filter(heading => heading.getBoundingClientRect().top <= 175).at(-1) || headings[0];
        for (const link of contents) {
            if (link.hash === '#' + active.id) link.setAttribute('aria-current', 'location');
            else link.removeAttribute('aria-current');
        }
    }
    addEventListener('scroll', () => {if (!queued) {queued = true; requestAnimationFrame(markSection);}}, {passive:true});
    markSection();
})();

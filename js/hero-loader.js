/* Keep the WebGL bundle out of static and directly opened reading pages. */
(() => {
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let staticPreferred = reduced.matches || !!navigator.connection?.saveData;
    const directReading = ['portfolio','blog','about'].includes(location.hash.slice(1));
    let active = !directReading, loading = false;
    function load() {
        const cover = document.querySelector('.pbr-hero');
        if (!cover || !cover.querySelector('.pbr-scene')) return;
        if (directReading || staticPreferred) cover.dataset.intro = 'finished';
        if (staticPreferred) {
            cover.classList.add('pbr-static');
            cover.querySelector('.pbr-motion').hidden = true;
            return;
        }
        if (!active) return;
        if (loading) { window.initializePortfolioHero?.(); return; }
        loading = true;
        const script = document.createElement('script');
        script.src = document.currentScript?.dataset.bundle || bundle;
        script.onload = () => { if (!active) cover.dataset.intro = 'finished'; };
        script.onerror = () => { cover.classList.add('pbr-static'); cover.querySelector('.pbr-motion').hidden = true; };
        document.head.append(script);
    }
    const bundle = document.currentScript.dataset.bundle;
    window.portfolioHero = {
        setChapterActive(value) { active = !!value; if (active) load(); },
        state: () => ({ready:false, deferred:!loading && !staticPreferred, staticPreferred, suspended:!active, frames:0, framePending:false, settled:true, decoding:{phase:'static', progress:1, plays:0}})
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', load, {once:true});
    else load();
    reduced.addEventListener('change', () => {
        staticPreferred = reduced.matches || !!navigator.connection?.saveData;
        if (!staticPreferred && !loading) {document.querySelector('.pbr-hero')?.classList.remove('pbr-static'); load();}
    });
    document.addEventListener('portfolio:core-ready', load, {once:true});
})();

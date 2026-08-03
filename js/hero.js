async function loadHero() {
    await import('./hero-wave.js');
}

loadHero().catch((error) => {
    // The inline watchdog in index.html reveals the static fallback if both
    // modules fail before either implementation can initialize.
    console.warn('The portfolio hero could not load; using the static fallback.', error);
});

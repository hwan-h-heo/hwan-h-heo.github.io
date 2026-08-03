const requestedHero = new URLSearchParams(window.location.search).get('hero');
const useExperimentalVoxelHero = requestedHero === 'voxel';

async function loadHero() {
    if (!useExperimentalVoxelHero) {
        await import('./hero-wave.js');
        return;
    }

    try {
        await import('./hero-voxel.js');
    } catch (error) {
        console.warn('The experimental voxel hero could not load; using the wave hero.', error);
        await import('./hero-wave.js');
    }
}

loadHero().catch((error) => {
    // The inline watchdog in index.html reveals the static fallback if both
    // modules fail before either implementation can initialize.
    console.warn('The portfolio hero could not load; using the static fallback.', error);
});

import { createMeshRefinement as createSurface } from './gpu-refinement.js';

// One entrance per page load. Chapter return, pointer input and Motion on never
// restart or reverse the completed surface.
export function createMeshRefinement(mesh) {
    const surface = createSurface(mesh);
    let started = false;
    return {
        samplePosition: surface.samplePosition,
        applyMaterial: surface.applyMaterial,
        sampleForCache: surface.seek,
        start() { if (started) return; started = true; surface.start(); },
        resume: surface.resume,
        update(now) { return surface.active ? surface.update(now) : false; },
        finish: surface.finish,
        get active() { return surface.active; },
        state: () => ({ ...surface.state(), surfaceProgress:surface.state().progress,
            phase:surface.active ? 'intro' : 'static', direction:surface.active ? 'forward' : 'none' }),
    };
}

const modelViewerTextureState = {
    exposure: 1,
    textures: []
};

function updateViewerModeButtons(mode) {
    document.querySelectorAll('.viewer-mode-button[data-viewer-mode]').forEach((button) => {
        const isActive = button.dataset.viewerMode === mode;
        button.classList.toggle('is-active', isActive);
        button.setAttribute('aria-pressed', String(isActive));
    });
}

function getToggleableModelViewer() {
    const modelViewer = document.getElementById('model-g');
    const materials = modelViewer?.model?.materials;
    if (!Array.isArray(materials) || materials.length === 0) {
        return null;
    }
    return { materials, modelViewer };
}

window.show_texture = function showTexture() {
    const viewerState = getToggleableModelViewer();
    if (!viewerState) {
        return;
    }

    const { materials, modelViewer } = viewerState;
    const baseColorTexture = materials[0].pbrMetallicRoughness.baseColorTexture;
    if (baseColorTexture.texture !== null) {
        updateViewerModeButtons('texture');
        return;
    }
    if (modelViewerTextureState.textures.length === 0) {
        return;
    }

    modelViewer.environmentImage = 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/white.jpg';
    materials.forEach((material, index) => {
        material.pbrMetallicRoughness.baseColorTexture.setTexture(
            modelViewerTextureState.textures[index] || null
        );
    });
    modelViewer.exposure = modelViewerTextureState.exposure;
    updateViewerModeButtons('texture');
};

window.show_geometry = function showGeometry() {
    const viewerState = getToggleableModelViewer();
    if (!viewerState) {
        return;
    }

    const { materials, modelViewer } = viewerState;
    const baseColorTexture = materials[0].pbrMetallicRoughness.baseColorTexture;
    if (baseColorTexture.texture === null) {
        updateViewerModeButtons('geometry');
        return;
    }

    modelViewerTextureState.textures = materials.map((material) => {
        return material.pbrMetallicRoughness.baseColorTexture.texture;
    });
    modelViewerTextureState.exposure = modelViewer.exposure;
    modelViewer.environmentImage = 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/gradient.jpg';
    materials.forEach((material) => {
        material.pbrMetallicRoughness.baseColorTexture.setTexture(null);
    });
    modelViewer.exposure = 3;
    updateViewerModeButtons('geometry');
};

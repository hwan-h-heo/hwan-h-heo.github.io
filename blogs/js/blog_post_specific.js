var window_state = {};
function show_texture(){
    let modelViewer = document.getElementById('model-g');

    if (modelViewer.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture !== null) return;
    modelViewer.environmentImage = 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/white.jpg';
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(window_state.textures[i]);
    }
    modelViewer.exposure = window_state.exposure;

}
function show_geometry(){
    let modelViewer = document.getElementById('model-g');

    if (modelViewer.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture === null) return;
    window_state.textures = [];
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        window_state.textures.push(modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.texture);
    }
    window_state.exposure = modelViewer.exposure;
    modelViewer.environmentImage = 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/gradient.jpg';
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(null);
    }
    modelViewer.exposure = 3;
}
import { VIEWER_STYLES } from './styles.js';

export function renderViewerTemplate(icons, environmentPresets) {
    return /*html*/`
            <style>${VIEWER_STYLES}</style>

            <div id="viewerRoot">
                <div id="canvasContainer"></div>

                <div id="toolbar" aria-label="Viewer controls">
                    <div class="tool-group">
                        <button id="openPanelBtn" class="icon-button" type="button" title="Open files" aria-label="Open files">${icons.open}</button>
                        <button id="fitBtn" class="icon-button" type="button" title="Fit" aria-label="Fit model">${icons.fit}</button>
                        <button id="resetBtn" class="icon-button" type="button" title="Reset view" aria-label="Reset view">${icons.reset}</button>
                    </div>
                    <div class="mode-segment" aria-label="View mode">
                        <button id="modeToggleBtn" type="button" aria-haspopup="menu" aria-expanded="false">PBR</button>
                        <div id="modeMenu" role="menu" hidden>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="pbr">PBR</button>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="albedo">Albedo</button>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="roughness">Rough</button>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="metalness">Metal</button>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="geometry">Geo</button>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="normal">Normal</button>
                        </div>
                    </div>
                    <div class="tool-group">
                        <button id="wireBtn" class="icon-button" type="button" title="Wireframe" aria-label="Wireframe" aria-pressed="false">${icons.wire}</button>
                        <select id="wireModeSelect" title="Wireframe mode" aria-label="Wireframe mode">
                            <option value="quad">Quad</option>
                            <option value="tri">Tri</option>
                        </select>
                        <button id="sectionBtn" class="icon-button" type="button" title="Section" aria-label="Section" aria-pressed="false">${icons.clip}</button>
                    </div>
                    <div class="toolbar-spacer"></div>
                    <div class="tool-group">
                        <button id="rotateBtn" class="icon-button" type="button" title="Auto rotate" aria-label="Auto rotate" aria-pressed="false">${icons.rotate}</button>
                        <button id="gridBtn" class="icon-button" type="button" title="Grid" aria-label="Grid" aria-pressed="false">${icons.grid}</button>
                        <button id="playBtn" class="icon-button" type="button" title="Play animation" aria-label="Play animation" disabled>${icons.play}</button>
                        <button id="snapshotBtn" class="icon-button" type="button" title="Snapshot" aria-label="Snapshot">${icons.shot}</button>
                        <button id="fullscreenBtn" class="icon-button" type="button" title="Fullscreen" aria-label="Fullscreen">${icons.full}</button>
                    </div>
                </div>

                <div id="environmentControls" aria-label="Environment controls">
                    <div class="pbr-light-control">
                        <div class="pbr-env-picker">
                            <button id="environmentToggle" class="pbr-env-toggle" type="button" aria-expanded="false">
                                <span>Env maps</span>
                                <span id="environmentActiveLabel" class="pbr-env-active-label">Forest</span>
                            </button>
                            <div id="environmentMenu" class="pbr-env-menu" hidden>
                                <button id="environmentBgBtn" class="pbr-env-background-toggle" type="button" aria-pressed="false" title="Show HDRI background">
                                    <span>Background</span>
                                    <span class="pbr-env-background-switch" aria-hidden="true"></span>
                                </button>
                                ${environmentPresets
                                    .map((preset) => `
                                        <button type="button" data-environment-preset="${preset.id}" aria-pressed="false">
                                            <canvas class="pbr-env-preview" width="56" height="56" data-environment-preview="${preset.id}" aria-hidden="true"></canvas>
                                            <span>${preset.label}</span>
                                        </button>
                                    `).join('')}
                            </div>
                        </div>
                        <button id="environmentDial" class="pbr-light-dial" type="button" role="slider" aria-label="Environment rotation" aria-valuemin="0" aria-valuemax="360" aria-valuenow="0">
                            <canvas id="environmentDialPreview" class="pbr-light-env-preview" width="56" height="56" aria-hidden="true"></canvas>
                            <span class="pbr-light-dial-ring"></span>
                            <span id="environmentDialHand" class="pbr-light-dial-hand"></span>
                        </button>
                        <span class="pbr-light-copy"><span id="environmentRotationValue" class="pbr-light-value">0deg</span></span>
                    </div>
                </div>

                <section id="sectionControls" aria-label="Section controls" hidden>
                    <div class="section-grid">
                        <select id="sectionAxis" aria-label="Section axis">
                            <option value="x">X</option>
                            <option value="y">Y</option>
                            <option value="z">Z</option>
                        </select>
                        <input id="sectionSlider" type="range" min="-100" max="100" step="1" value="0" aria-label="Section position">
                        <span id="sectionValue">0%</span>
                    </div>
                </section>

                <section id="fileInputContainer" aria-label="Open 3D model">
                    <div class="panel-head">
                        <div>
                            <p>Local mesh</p>
                            <h2>Open 3D model</h2>
                        </div>
                        <button id="closePanelBtn" class="icon-button" type="button" title="Close" aria-label="Close panel">${icons.close}</button>
                    </div>

                    <div id="dropZone" tabindex="0">
                        ${icons.open}
                        <strong>Drop mesh files</strong>
                        <span>GLB, GLTF, OBJ, FBX, PLY, STL with companion textures or MTL</span>
                    </div>

                    <div class="upload-actions">
                        <button id="chooseFilesBtn" class="text-button primary" type="button">Choose files</button>
                        <button id="chooseFolderBtn" class="text-button secondary" type="button">Choose folder</button>
                    </div>

                    <div class="history-row">
                        <select id="historySelect" aria-label="Recent local meshes">
                            <option value="">No saved meshes</option>
                        </select>
                        <button id="clearHistoryBtn" class="text-button secondary" type="button" disabled>Clear</button>
                    </div>
                    <div id="historyUsage" class="storage-row">Saved 0 B / 100 MB</div>

                    <p id="status">Files stay in this browser.</p>
                    <div id="entries" hidden></div>

                    <section id="preflight" hidden>
                        <dl id="preflightFacts"></dl>
                        <div id="preflightResources" class="preflight-resources"></div>
                        <div class="preflight-actions">
                            <button id="relinkBtn" class="text-button secondary" type="button">Add files</button>
                            <button id="loadBtn" class="text-button primary" type="button">Load mesh</button>
                        </div>
                    </section>
                </section>

                <div id="stats" hidden></div>

                <aside id="textureMaps" aria-label="Texture maps" hidden>
                    <div class="texture-head">
                        <button id="textureToggleBtn" type="button" title="Texture maps" aria-label="Toggle texture map previews" aria-expanded="false">
                            <span>TEX</span>
                            <span id="textureCount"></span>
                        </button>
                    </div>
                    <div id="textureStrip"></div>
                </aside>

                <div id="loading" role="status" aria-live="polite" hidden>
                    <div class="loading-head">
                        <strong id="loadingLabel">Loading mesh</strong>
                        <span id="loadingValue">0%</span>
                    </div>
                    <div class="loading-track" aria-hidden="true"><span id="loadingBar"></span></div>
                </div>

                <dialog id="textureDialog">
                    <button id="textureDialogClose" class="text-button secondary" type="button">Close</button>
                    <canvas id="textureDialogCanvas"></canvas>
                    <p id="textureDialogCaption"></p>
                </dialog>

                <input id="fileInput" class="visually-hidden" type="file" accept=".glb,.gltf,.obj,.fbx,.ply,.stl,.mtl,.bin,.png,.jpg,.jpeg,.webp,.avif,.bmp,.gif,.hdr,.exr,.tga,.ktx2" multiple>
                <input id="folderInput" class="visually-hidden" type="file" webkitdirectory directory multiple>
                <input id="relinkInput" class="visually-hidden" type="file" multiple>
            </div>
        `;
}

// Styles for the model viewer shadow root.
export const VIEWER_STYLES = /*css*/`
                :host {
                    display: block;
                    min-height: 360px;
                    color: #f1ede2;
                    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    --viewer-panel: rgba(8, 8, 7, 0.62);
                    --viewer-panel-strong: rgba(8, 8, 7, 0.84);
                    --viewer-ink: #f1ede2;
                    --viewer-muted: rgba(241, 237, 226, 0.6);
                    --viewer-line: rgba(241, 237, 226, 0.14);
                    --viewer-accent: #5db8a6;
                    --viewer-accent-2: #f3b35b;
                }

                *, *::before, *::after {
                    box-sizing: border-box;
                }

                #viewerRoot {
                    position: relative;
                    width: 100%;
                    height: 100%;
                    min-height: inherit;
                    overflow: hidden;
                    background:
                        radial-gradient(ellipse at 50% 68%, rgba(86, 86, 82, 0.34) 0%, rgba(38, 38, 36, 0.18) 29%, rgba(15, 15, 14, 0) 58%),
                        radial-gradient(ellipse at 50% 52%, rgba(48, 48, 45, 0.2) 0%, rgba(15, 15, 14, 0) 54%),
                        linear-gradient(180deg, #111110 0%, #0d0d0c 48%, var(--viewer-bg-color, #070707) 100%);
                }

                #viewerRoot.upload-open::after {
                    position: absolute;
                    inset: 0;
                    z-index: 8;
                    background: rgba(2, 3, 3, 0.34);
                    content: "";
                    pointer-events: none;
                }

                #canvasContainer,
                #canvasContainer > canvas {
                    position: absolute;
                    inset: 0;
                    width: 100%;
                    height: 100%;
                }

                canvas {
                    display: block;
                }

                #canvasContainer > canvas {
                    outline: none;
                    touch-action: none;
                }

                button,
                select,
                input {
                    font: inherit;
                }

                button {
                    border: 1px solid var(--viewer-line);
                    background: rgba(255, 255, 255, 0.045);
                    color: var(--viewer-ink);
                    cursor: pointer;
                }

                button:disabled {
                    cursor: default;
                    opacity: 0.42;
                }

                button:hover:not(:disabled),
                button:focus-visible,
                select:focus-visible,
                input:focus-visible {
                    border-color: rgba(93, 184, 166, 0.56);
                    outline: 2px solid rgba(93, 184, 166, 0.2);
                    outline-offset: 1px;
                }

                svg {
                    width: 1rem;
                    height: 1rem;
                    fill: none;
                    stroke: currentColor;
                    stroke-linecap: round;
                    stroke-linejoin: round;
                    stroke-width: 1.8;
                }

                .icon-button {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: 2.05rem;
                    height: 2.05rem;
                    border-radius: 7px;
                    padding: 0;
                }

                .text-button {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 2.05rem;
                    border-radius: 7px;
                    padding: 0 0.72rem;
                    font-size: 0.78rem;
                    font-weight: 700;
                    white-space: nowrap;
                }

                .primary {
                    border-color: rgba(93, 184, 166, 0.36);
                    background: rgba(93, 184, 166, 0.22);
                    color: #dffcf7;
                }

                .secondary {
                    background: rgba(255, 255, 255, 0.06);
                }

                .active,
                .icon-button[aria-pressed="true"],
                .text-button[aria-pressed="true"] {
                    border-color: rgba(93, 184, 166, 0.58);
                    background: rgba(93, 184, 166, 0.15);
                    color: #bcfff3;
                }

                #toolbar {
                    position: absolute;
                    z-index: 8;
                    top: 0.75rem;
                    left: 0.75rem;
                    right: 0.75rem;
                    display: flex;
                    align-items: center;
                    gap: 0.45rem;
                    pointer-events: none;
                }

                #toolbar > * {
                    pointer-events: auto;
                }

                .tool-group,
                .mode-segment {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.25rem;
                    min-width: 0;
                    padding: 0.25rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: var(--viewer-panel);
                    box-shadow: 0 12px 34px rgba(17, 24, 39, 0.12);
                    backdrop-filter: blur(14px);
                }

                .mode-segment {
                    position: relative;
                    overflow: visible;
                }

                .toolbar-spacer {
                    flex: 1;
                }

                #modeToggleBtn,
                #modeMenu .mode-button {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 1.78rem;
                    border-radius: 6px;
                    padding: 0 0.48rem;
                    font-size: 0.68rem;
                    font-weight: 800;
                    letter-spacing: 0;
                    text-transform: uppercase;
                }

                #modeToggleBtn {
                    min-width: 4.25rem;
                    gap: 0.42rem;
                }

                #modeToggleBtn::after {
                    width: 0;
                    height: 0;
                    border-left: 0.22rem solid transparent;
                    border-right: 0.22rem solid transparent;
                    border-top: 0.3rem solid currentColor;
                    content: "";
                    opacity: 0.72;
                }

                #modeMenu {
                    position: absolute;
                    z-index: 14;
                    top: calc(100% + 0.35rem);
                    left: 0.25rem;
                    display: grid;
                    gap: 0.25rem;
                    min-width: 7.1rem;
                    padding: 0.35rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: var(--viewer-panel-strong);
                    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.32);
                    backdrop-filter: blur(16px);
                }

                #modeMenu[hidden] {
                    display: none;
                }

                #modeMenu .mode-button {
                    justify-content: flex-start;
                    width: 100%;
                    min-width: 0;
                }

                #modeMenu .mode-button:not(.active) {
                    border-color: transparent;
                    background: transparent;
                    color: rgba(241, 237, 226, 0.62);
                }

                #wireModeSelect,
                #sectionAxis {
                    width: 4.3rem;
                    min-height: 2.05rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 7px;
                    background: rgba(255, 255, 255, 0.06);
                    color: var(--viewer-ink);
                    font-size: 0.74rem;
                    font-weight: 700;
                    padding: 0 0.45rem;
                }

                #stats {
                    position: absolute;
                    z-index: 7;
                    left: 0.75rem;
                    bottom: 0.75rem;
                    max-width: calc(100% - 1.5rem);
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: rgba(8, 8, 7, 0.58);
                    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
                    padding: 0.46rem 0.62rem;
                    color: rgba(241, 237, 226, 0.78);
                    font-size: 0.74rem;
                    font-weight: 700;
                    backdrop-filter: blur(14px);
                }

                #textureMaps {
                    position: absolute;
                    z-index: 7;
                    right: 0.75rem;
                    bottom: 5.25rem;
                    width: min(380px, calc(100% - 1.5rem));
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: rgba(8, 8, 7, 0.72);
                    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
                    padding: 0.48rem;
                    backdrop-filter: blur(14px);
                }

                #textureMaps.collapsed {
                    width: auto;
                    min-width: 0;
                    border-color: transparent;
                    background: transparent;
                    box-shadow: none;
                    padding: 0;
                    backdrop-filter: none;
                }

                #environmentControls {
                    position: absolute;
                    right: 0.75rem;
                    bottom: 0.75rem;
                    z-index: 8;
                    pointer-events: none;
                }

                .pbr-light-control {
                    position: relative;
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    width: 190px;
                    min-height: 4rem;
                    padding: 0.38rem 0.5rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: rgba(15, 15, 14, 0.72);
                    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.34);
                    backdrop-filter: blur(14px);
                    pointer-events: auto;
                }

                .pbr-env-picker {
                    position: relative;
                    flex: 0 0 auto;
                }

                .pbr-env-toggle {
                    display: grid;
                    align-content: center;
                    gap: 0.12rem;
                    min-width: 4.85rem;
                    min-height: 2.75rem;
                    border-radius: 8px;
                    padding: 0.32rem 0.5rem;
                    background: rgba(8, 8, 7, 0.52);
                    color: var(--viewer-ink);
                    font-size: 0.68rem;
                    font-weight: 900;
                    line-height: 1;
                    text-align: left;
                }

                .pbr-env-toggle:hover,
                .pbr-env-toggle:focus-visible,
                .pbr-env-toggle[aria-expanded="true"] {
                    border-color: rgba(243, 179, 91, 0.72);
                    background: rgba(243, 179, 91, 0.14);
                    color: var(--viewer-accent-2);
                }

                .pbr-env-active-label {
                    max-width: 4.25rem;
                    overflow: hidden;
                    color: var(--viewer-muted);
                    font-size: 0.56rem;
                    font-weight: 800;
                    text-overflow: ellipsis;
                    text-transform: uppercase;
                    white-space: nowrap;
                }

                .pbr-env-menu {
                    position: absolute;
                    left: 0;
                    bottom: calc(100% + 0.5rem);
                    z-index: 9;
                    display: grid;
                    gap: 0.32rem;
                    width: 5.9rem;
                    max-height: min(390px, calc(100vh - 160px));
                    overflow-y: auto;
                    padding: 0.44rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: rgba(10, 10, 9, 0.92);
                    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.42);
                    backdrop-filter: blur(12px);
                }

                .pbr-env-menu button {
                    display: grid;
                    justify-items: center;
                    gap: 0.25rem;
                    min-width: 0;
                    min-height: 4.9rem;
                    border: 1px solid rgba(241, 237, 226, 0.16);
                    border-radius: 8px;
                    padding: 0.36rem 0.25rem;
                    background: rgba(8, 8, 7, 0.46);
                    color: var(--viewer-muted);
                    font-size: 0.62rem;
                    font-weight: 900;
                    line-height: 1;
                }

                .pbr-env-menu .pbr-env-background-toggle {
                    grid-template-columns: minmax(0, 1fr) auto;
                    align-items: center;
                    justify-items: stretch;
                    min-height: 1.75rem;
                    padding: 0.32rem 0.38rem;
                    text-align: left;
                }

                .pbr-env-background-toggle span:first-child {
                    overflow: hidden;
                    font-size: 0.5rem;
                    text-overflow: ellipsis;
                    text-transform: uppercase;
                    white-space: nowrap;
                }

                .pbr-env-background-switch {
                    position: relative;
                    width: 1.5rem;
                    height: 0.88rem;
                    border-radius: 999px;
                    background: rgba(241, 237, 226, 0.16);
                    box-shadow: inset 0 0 0 1px rgba(241, 237, 226, 0.18);
                }

                .pbr-env-background-switch::after {
                    position: absolute;
                    top: 0.19rem;
                    left: 0.19rem;
                    width: 0.5rem;
                    height: 0.5rem;
                    border-radius: 50%;
                    background: var(--viewer-muted);
                    content: "";
                    transition: transform 120ms ease, background 120ms ease;
                }

                .pbr-env-background-toggle.active .pbr-env-background-switch {
                    background: rgba(243, 179, 91, 0.32);
                    box-shadow: inset 0 0 0 1px rgba(243, 179, 91, 0.48);
                }

                .pbr-env-background-toggle.active .pbr-env-background-switch::after {
                    transform: translateX(0.62rem);
                    background: var(--viewer-accent-2);
                }

                .pbr-env-menu button:hover,
                .pbr-env-menu button:focus-visible,
                .pbr-env-menu button.active {
                    border-color: rgba(243, 179, 91, 0.72);
                    color: var(--viewer-accent-2);
                    background: rgba(243, 179, 91, 0.14);
                }

                .pbr-env-preview {
                    width: 3rem;
                    height: 3rem;
                    border-radius: 50%;
                    background: rgba(0, 0, 0, 0.28);
                    box-shadow:
                        inset 0 0 0 1px rgba(255, 255, 255, 0.16),
                        0 4px 10px rgba(0, 0, 0, 0.28);
                }

                .pbr-light-dial {
                    position: relative;
                    flex: 0 0 3.5rem;
                    width: 3.5rem;
                    height: 3.5rem;
                    overflow: hidden;
                    border: 1px solid rgba(243, 179, 91, 0.72);
                    border-radius: 50%;
                    padding: 0;
                    background: rgba(243, 179, 91, 0.08);
                    color: var(--viewer-accent-2);
                    cursor: grab;
                    touch-action: none;
                }

                .pbr-light-dial:active {
                    cursor: grabbing;
                }

                .pbr-light-env-preview {
                    position: absolute;
                    inset: 0;
                    width: 100%;
                    height: 100%;
                    border-radius: 50%;
                }

                .pbr-light-dial-ring {
                    position: absolute;
                    inset: 0.25rem;
                    border: 1px solid rgba(250, 249, 245, 0.42);
                    border-radius: 50%;
                    box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.45);
                }

                .pbr-light-dial-hand {
                    position: absolute;
                    left: 50%;
                    top: 0.32rem;
                    width: 2px;
                    height: 1.45rem;
                    margin-left: -1px;
                    border-radius: 999px;
                    background: currentColor;
                    transform-origin: 50% 1.45rem;
                    box-shadow:
                        0 0 0 1px rgba(0, 0, 0, 0.3),
                        0 0 8px rgba(243, 179, 91, 0.48);
                }

                .pbr-light-copy {
                    display: grid;
                    min-width: 1.5rem;
                    justify-items: end;
                }

                .pbr-light-value {
                    color: var(--viewer-accent-2);
                    font-size: 0.68rem;
                    font-weight: 800;
                    line-height: 1.1;
                }

                .texture-head {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 0.65rem;
                    margin-bottom: 0.38rem;
                    color: rgba(241, 237, 226, 0.78);
                    font-size: 0.74rem;
                    font-weight: 800;
                }

                #textureMaps.collapsed .texture-head {
                    margin-bottom: 0;
                }

                #textureToggleBtn {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: auto;
                    min-width: 4rem;
                    min-height: 1.9rem;
                    gap: 0.38rem;
                    border-color: rgba(241, 237, 226, 0.16);
                    border-radius: 999px;
                    background: rgba(8, 8, 7, 0.58);
                    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.055);
                    color: rgba(241, 237, 226, 0.76);
                    font-size: 0.62rem;
                    font-weight: 900;
                    letter-spacing: 0;
                    line-height: 1;
                    padding: 0 0.62rem;
                    text-transform: uppercase;
                }

                #textureToggleBtn:hover,
                #textureToggleBtn:focus-visible,
                #textureToggleBtn[aria-expanded="true"] {
                    border-color: rgba(93, 184, 166, 0.58);
                    background: rgba(93, 184, 166, 0.13);
                    color: #bcfff3;
                }

                #textureCount {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-width: 1.05rem;
                    height: 1.05rem;
                    border-radius: 999px;
                    background: rgba(241, 237, 226, 0.12);
                    color: rgba(241, 237, 226, 0.68);
                    font-size: 0.56rem;
                    font-weight: 900;
                    line-height: 1;
                    padding: 0 0.2rem;
                }

                #textureToggleBtn[aria-expanded="true"] #textureCount {
                    background: rgba(93, 184, 166, 0.22);
                    color: #d9fff8;
                }

                #textureStrip {
                    display: grid;
                    grid-auto-flow: column;
                    grid-auto-columns: 4.85rem;
                    gap: 0.48rem;
                    overflow-x: auto;
                    padding-bottom: 0.05rem;
                }

                #textureMaps.collapsed #textureStrip {
                    display: none;
                }

                .texture-card {
                    display: grid;
                    grid-template-rows: 4.15rem auto;
                    gap: 0.32rem;
                    border: 1px solid rgba(241, 237, 226, 0.12);
                    border-radius: 7px;
                    background: rgba(255, 255, 255, 0.055);
                    padding: 0.3rem;
                    color: rgba(241, 237, 226, 0.8);
                    text-align: left;
                    min-width: 0;
                }

                .texture-card span {
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    font-size: 0.64rem;
                    font-weight: 800;
                }

                .texture-preview {
                    display: grid;
                    place-items: center;
                    min-width: 0;
                    min-height: 0;
                    border-radius: 5px;
                    background:
                        linear-gradient(45deg, rgba(241, 237, 226, 0.16) 25%, transparent 25%),
                        linear-gradient(-45deg, rgba(241, 237, 226, 0.16) 25%, transparent 25%),
                        linear-gradient(45deg, transparent 75%, rgba(241, 237, 226, 0.16) 75%),
                        linear-gradient(-45deg, transparent 75%, rgba(241, 237, 226, 0.16) 75%),
                        rgba(0, 0, 0, 0.34);
                    background-size: 14px 14px;
                    background-position: 0 0, 0 7px, 7px -7px, -7px 0;
                    overflow: hidden;
                }

                .texture-preview img,
                .texture-preview canvas {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }

                #fileInputContainer {
                    position: absolute;
                    z-index: 9;
                    left: 50%;
                    top: 50%;
                    width: min(640px, calc(100% - 2.5rem));
                    max-height: min(78vh, 700px);
                    overflow: auto;
                    transform: translate(-50%, -50%);
                    border: 1px solid var(--viewer-line);
                    border-radius: 14px;
                    background: var(--viewer-panel-strong);
                    box-shadow: 0 24px 72px rgba(0, 0, 0, 0.54);
                    padding: 1.05rem;
                    backdrop-filter: blur(18px);
                }

                .panel-head {
                    display: flex;
                    align-items: start;
                    justify-content: space-between;
                    gap: 0.75rem;
                    margin-bottom: 0.65rem;
                }

                .panel-head p {
                    margin: 0;
                    color: var(--viewer-accent);
                    font-size: 0.7rem;
                    font-weight: 800;
                    text-transform: uppercase;
                }

                .panel-head h2 {
                    margin: 0.12rem 0 0;
                    color: var(--viewer-ink);
                    font-size: 1.05rem;
                    line-height: 1.1;
                    letter-spacing: 0;
                }

                #dropZone {
                    display: grid;
                    gap: 0.3rem;
                    place-items: center;
                    min-height: 11rem;
                    border: 1px dashed rgba(93, 184, 166, 0.46);
                    border-radius: 12px;
                    background:
                        radial-gradient(circle at 50% 12%, rgba(93, 184, 166, 0.12), transparent 52%),
                        rgba(93, 184, 166, 0.035);
                    padding: 0.85rem;
                    text-align: center;
                    color: rgba(241, 237, 226, 0.82);
                }

                #dropZone svg {
                    width: 1.7rem;
                    height: 1.7rem;
                    color: var(--viewer-accent);
                }

                #dropZone strong {
                    font-size: 0.88rem;
                    line-height: 1.2;
                }

                #dropZone span,
                #status,
                .storage-row,
                .preflight-resources {
                    color: var(--viewer-muted);
                    font-size: 0.72rem;
                    line-height: 1.35;
                }

                #viewerRoot.dragging #dropZone {
                    border-color: rgba(93, 184, 166, 0.94);
                    background:
                        radial-gradient(circle at 50% 12%, rgba(93, 184, 166, 0.22), transparent 56%),
                        rgba(93, 184, 166, 0.08);
                }

                .upload-actions,
                .history-row,
                .storage-row,
                .preflight-actions {
                    display: flex;
                    align-items: center;
                    gap: 0.45rem;
                    flex-wrap: wrap;
                    margin-top: 0.62rem;
                }

                .history-row select {
                    flex: 1;
                    min-width: 11rem;
                    min-height: 2.05rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 7px;
                    background: rgba(255, 255, 255, 0.06);
                    color: var(--viewer-ink);
                    padding: 0 0.5rem;
                    font-size: 0.76rem;
                    font-weight: 700;
                }

                #entries {
                    display: grid;
                    gap: 0.35rem;
                    max-height: 8.5rem;
                    overflow: auto;
                    margin-top: 0.6rem;
                }

                .entry-button {
                    display: grid;
                    grid-template-columns: 1fr auto;
                    gap: 0.5rem;
                    width: 100%;
                    min-height: 2.3rem;
                    border-radius: 7px;
                    padding: 0.45rem 0.55rem;
                    text-align: left;
                }

                .entry-button strong,
                .entry-button span {
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }

                .entry-button strong {
                    color: var(--viewer-ink);
                    font-size: 0.76rem;
                }

                .entry-button span {
                    color: rgba(241, 237, 226, 0.54);
                    font-size: 0.68rem;
                }

                #preflight {
                    margin-top: 0.7rem;
                    border-top: 1px solid var(--viewer-line);
                    padding-top: 0.65rem;
                }

                #preflightFacts {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 0.35rem;
                    margin: 0;
                }

                #preflightFacts div {
                    min-width: 0;
                    border: 1px solid rgba(241, 237, 226, 0.1);
                    border-radius: 7px;
                    background: rgba(255, 255, 255, 0.055);
                    padding: 0.43rem 0.48rem;
                }

                #preflightFacts dt,
                #preflightFacts dd {
                    margin: 0;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }

                #preflightFacts dt {
                    color: rgba(241, 237, 226, 0.54);
                    font-size: 0.62rem;
                    font-weight: 800;
                    text-transform: uppercase;
                }

                #preflightFacts dd {
                    color: var(--viewer-ink);
                    font-size: 0.76rem;
                    font-weight: 800;
                }

                #sectionControls {
                    position: absolute;
                    z-index: 8;
                    top: 4.15rem;
                    right: 0.75rem;
                    width: min(320px, calc(100% - 1.5rem));
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: var(--viewer-panel);
                    box-shadow: 0 12px 34px rgba(17, 24, 39, 0.12);
                    padding: 0.55rem;
                    backdrop-filter: blur(14px);
                }

                .section-grid {
                    display: grid;
                    grid-template-columns: auto 1fr auto;
                    gap: 0.45rem;
                    align-items: center;
                }

                #sectionSlider {
                    width: 100%;
                    accent-color: var(--viewer-accent);
                }

                #sectionValue {
                    min-width: 3.4rem;
                    color: rgba(241, 237, 226, 0.8);
                    font-size: 0.72rem;
                    font-weight: 800;
                    text-align: right;
                }

                #loading {
                    position: absolute;
                    z-index: 10;
                    left: 50%;
                    bottom: 1rem;
                    width: min(340px, calc(100% - 2rem));
                    transform: translateX(-50%);
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: rgba(8, 8, 7, 0.86);
                    box-shadow: 0 14px 42px rgba(0, 0, 0, 0.34);
                    padding: 0.65rem 0.75rem;
                    backdrop-filter: blur(14px);
                }

                .loading-head {
                    display: flex;
                    justify-content: space-between;
                    gap: 0.75rem;
                    color: rgba(241, 237, 226, 0.82);
                    font-size: 0.76rem;
                    font-weight: 800;
                    margin-bottom: 0.45rem;
                }

                .loading-track {
                    height: 0.42rem;
                    overflow: hidden;
                    border-radius: 999px;
                    background: rgba(93, 184, 166, 0.16);
                }

                .loading-track span {
                    display: block;
                    width: 0%;
                    height: 100%;
                    border-radius: inherit;
                    background: linear-gradient(90deg, var(--viewer-accent), var(--viewer-accent-2));
                    transition: width 160ms ease;
                }

                #textureDialog {
                    width: min(88vmin, 960px);
                    height: min(88vmin, 960px);
                    max-width: calc(100vw - 2rem);
                    max-height: calc(100vh - 2rem);
                    border: 1px solid rgba(241, 237, 226, 0.16);
                    border-radius: 10px;
                    background: rgba(8, 8, 7, 0.96);
                    padding: 0;
                    color: var(--viewer-ink);
                    overflow: hidden;
                }

                #textureDialog::backdrop {
                    background: rgba(0, 0, 0, 0.68);
                    backdrop-filter: blur(2px);
                }

                #textureDialogCanvas {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                    background:
                        linear-gradient(45deg, rgba(241, 237, 226, 0.12) 25%, transparent 25%),
                        linear-gradient(-45deg, rgba(241, 237, 226, 0.12) 25%, transparent 25%),
                        linear-gradient(45deg, transparent 75%, rgba(241, 237, 226, 0.12) 75%),
                        linear-gradient(-45deg, transparent 75%, rgba(241, 237, 226, 0.12) 75%),
                        rgba(0, 0, 0, 0.42);
                    background-size: 24px 24px;
                    background-position: 0 0, 0 12px, 12px -12px, -12px 0;
                }

                #textureDialogCaption {
                    position: absolute;
                    left: 0.75rem;
                    right: 0.75rem;
                    bottom: 0.75rem;
                    margin: 0;
                    border: 1px solid rgba(241, 237, 226, 0.14);
                    border-radius: 7px;
                    background: rgba(0, 0, 0, 0.46);
                    padding: 0.45rem 0.55rem;
                    color: rgba(241, 237, 226, 0.82);
                    font-size: 0.78rem;
                    font-weight: 700;
                }

                #textureDialogClose {
                    position: absolute;
                    top: 0.75rem;
                    right: 0.75rem;
                    z-index: 1;
                    background: rgba(0, 0, 0, 0.52);
                }

                .visually-hidden {
                    position: absolute;
                    width: 1px;
                    height: 1px;
                    overflow: hidden;
                    clip: rect(0 0 0 0);
                    white-space: nowrap;
                    clip-path: inset(50%);
                }

                [hidden] {
                    display: none !important;
                }

                @media (max-width: 760px) {
                    #toolbar {
                        align-items: start;
                        flex-wrap: wrap;
                    }

                    .toolbar-spacer {
                        display: none;
                    }

                    #modeMenu {
                        left: 0.25rem;
                    }

                    #fileInputContainer {
                        width: calc(100% - 1.5rem);
                        max-height: calc(100% - 7.5rem);
                    }

                    #textureMaps {
                        left: 0.75rem;
                        right: auto;
                        bottom: 5.25rem;
                        width: min(320px, calc(100% - 1.5rem));
                    }

                    #textureMaps.collapsed {
                        width: auto;
                    }

                    #stats {
                        display: none;
                    }

                    #sectionControls {
                        top: auto;
                        right: 0.75rem;
                        bottom: 0.75rem;
                    }
                }
            `;

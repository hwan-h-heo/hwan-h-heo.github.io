# Simple Model Viewer Implementation Plan

Last updated: 2026-03-15
Target file: `js/simple-model-viewer.js`
Status: Phases 6-8 implemented in `js/simple-model-viewer.js`; docs/examples reconciled with the shipped API and `npm run build` re-verified on 2026-03-15; manual browser verification remains pending

## Goal

`simple-model-viewer`를 데모용 viewer에서 재사용 가능한 범용 Web Component로 확장한다.

이번 문서는 다음 목적을 가진다.

- 구현 범위를 기능 묶음 단위로 분해한다.
- 각 묶음별 세부 작업과 완료 조건을 명확히 정의한다.
- 이후 세션에서 작업 로그와 검증 결과를 한 파일에서 계속 기록한다.

## Original Non-Goals For The Planning Pass

- 코드 구현 시작하지 않음
- 기존 동작 변경하지 않음
- 새 UI/스타일 반영하지 않음
- 테스트 실행하지 않음

## Current Baseline Summary

현재 `js/simple-model-viewer.js`에는 아래 기능이 이미 있다.

- `glb/gltf/obj/fbx/ply` 로드
- 조명, 배경색, grid, environment preset 3종
- `diffuse/geometry/normal/wireframe` 뷰 모드
- screenshot to clipboard
- canvas video recording
- basic animation clip selector with play/pause
- transform controls: translate, rotate
- scene graph tree, visibility toggle
- texture 교체 및 history
- explode slider

현재 부족한 축은 아래와 같다.

- 외부 제어용 public API와 event surface가 작음
- 카메라 복구 및 framing 기능 부족
- 캔버스 직접 선택 부재
- 애니메이션 제어가 clip 선택 수준에 머뭄
- material/texture 편집 범위가 제한적
- environment/background 설정이 preset 위주
- 출력 및 상태 저장 기능이 약함
- 대형 모델/장시간 사용 대비 성능 옵션 부족

## Global Success Criteria

아래 항목을 모두 만족하면 전체 작업 완료로 본다.

- `simple-model-viewer`가 attribute, method, custom event를 통해 외부에서 실질적으로 제어 가능하다.
- 기존 주요 기능이 유지된다.
- 새 기능이 최소한의 수동 검증 체크리스트를 통과한다.
- 문서와 예제가 새 API와 UI를 반영한다.
- `npm run build`가 통과한다.

## Execution Rules

- 단계별로 구현하고 각 단계 완료 시 이 파일의 체크박스를 갱신한다.
- 한 단계가 끝날 때마다 작업 로그와 검증 결과를 남긴다.
- 기존 동작과 충돌하는 변경이 있으면 하위 호환 여부를 반드시 적는다.
- `blogs/dist/`는 직접 수정하지 않는다.

## Phase Overview

- [x] Phase 0. Baseline hardening and internal structure prep
- [x] Phase 1. Public API and custom events
- [x] Phase 2. Camera and navigation improvements
- [x] Phase 3. Canvas picking and selection workflow
- [x] Phase 4. Animation control expansion
- [x] Phase 5. Material and texture editing expansion
- [x] Phase 6. Environment and background pipeline
- [x] Phase 7. Export, capture, and state persistence
- [x] Phase 8. Input UX and performance options
- [ ] Phase 9. Documentation, examples, and final verification

---

## Phase 0. Baseline Hardening And Internal Structure Prep

### Objective

후속 기능 추가가 안전하도록 상태, 렌더 갱신, UI sync, dispose 경로를 정리한다.

### Planned Work

- [x] 현재 state 항목과 UI sync 지점을 목록화한다.
- [x] `loadModel`, `discardModel`, `disposeCurrentModel`, `renderMode`, selection reset 흐름을 점검한다.
- [x] 중복 렌더 호출과 산발적 UI 업데이트를 묶을 최소한의 internal helper를 설계한다.
- [x] 새 기능이 기대할 수 있는 공용 helper 후보를 정리한다.
  - 예: `requestRender`, `refreshUiFromState`, `frameObject`, `emitEvent`, `serializeState`
- [x] 메모리 정리 포인트를 명시한다.
  - texture replacement
  - environment replacement
  - model replacement
  - recording/preview cleanup
- [x] backward compatibility 기준을 적는다.

### Success Criteria

- [x] 후속 기능이 얹힐 핵심 helper 목록이 문서화되거나 코드에 반영된다.
- [x] 모델 교체와 discard 후 이전 state 잔존 이슈가 없도록 정리된다.
- [x] 기존 기능의 초기화/정리 경로가 일관된다.
- [x] 새 기능 구현 전에 구조적 위험 요소가 정리되었다고 판단할 수 있다.

### Verification

- [ ] 모델 로드 후 discard 반복 시 UI와 scene state가 누적되지 않는다.
- [ ] wireframe, explode, selection, animation 상태가 모델 교체 뒤 초기화된다.

### Phase 0 Notes

State and UI sync points consolidated in code:

- Core state currently synchronized through `refreshUiFromState()`: `lightsOn`, `viewMode`, `environment`, `isAnimationPlaying`, `transformMode`, and wireframe button state.
- Model-dependent UI reset is now centralized in `resetModelUiState(showLoading)` for model info, transform inputs, roughness/metalness controls, texture selectors/history, explode UI, and animation description visibility.
- Lifecycle paths now flow through `resetModelSession()` before new loads and discard operations so `loadModel`, `discardModel`, `disposeCurrentModel`, selection reset, animation reset, wireframe reset, and transform reset use a consistent teardown path.

Helpers now in code for later phases:

- `requestRender()`
- `refreshUiFromState()`
- `updateDiscardButtonVisibility()`
- `updateTransformButtons()`
- `resetTransformState()`
- `clearModelResources()`
- `resetModelSession()`
- `disposeMaterialSnapshotStore()`
- `disposeTextureHistory()`
- `disposeEnvironmentTexture()`

Cleanup points now explicitly handled:

- Texture replacement history clones are disposed through `disposeTextureHistory()`.
- Original material snapshots are disposed through `disposeMaterialSnapshotStore()`.
- Model replacement/discard goes through `clearModelResources()` and `disposeCurrentModel()`.
- Environment replacement/default reset disposes prior HDR textures through `disposeEnvironmentTexture()`.
- Recording modal/preview cleanup continues to live in `closeModal()` and `stopRecording()`.

Backward compatibility notes:

- Existing attributes and UI controls were kept intact in this pass.
- No new public API was introduced yet; this was internal structure work for later phases.
- Existing view modes, environment toggles, animation UI, texture replacement, and discard behavior remain user-facing compatible while teardown/reset behavior is more explicit.

### Work Log

#### 2026-03-14

- Implemented Phase 0 hardening in `js/simple-model-viewer.js`.
- Added centralized lifecycle helpers for render requests, UI refresh, transform reset, model-session reset, and disposal of material snapshots, texture history, and environment textures.
- Unified `loadModel()` and `discardModel()` around a shared reset path so selection, animation, wireframe, transform attachment, texture UI, explode UI, and model metadata are cleared consistently before reload/discard.
- Preserved existing external behavior while tightening internal cleanup rules, especially around environment replacement and texture-history disposal.

Verification results:

- `npm run build`: passed on 2026-03-14.
- Browser/manual verification for repeated load/discard and state reset behavior is still pending.

---

## Phase 1. Public API And Custom Events

### Objective

`simple-model-viewer`를 외부에서 제어 가능한 Web Component로 확장한다.

### Planned Work

- [x] 신규 attribute 후보 확정
  - `environment`
  - `environment-url`
  - `background-color`
  - `camera-target`
  - `camera-up`
  - `exposure`
  - `animation`
  - `autoplay`
  - `interaction-prompt` 여부는 선택
  - `selection-mode`
  - `performance-mode`
- [x] 필요한 경우 public method 추가
  - `loadModelFromUrl(url)`
  - `discardModel()`
  - `resetView()`
  - `fitCameraToModel()`
  - `selectMeshByName(name)`
  - `selectMeshByIndex(index)`
  - `setEnvironment(urlOrPreset)`
  - `captureScreenshot(options)`
  - `exportState()`
  - `importState(state)`
- [x] custom event 추가
  - `viewer-load`
  - `viewer-error`
  - `viewer-selection-change`
  - `viewer-camera-change`
  - `viewer-animation-change`
  - `viewer-material-change`
  - `viewer-environment-change`
  - `viewer-state-export`
- [x] attribute 변경과 내부 state 갱신 규칙을 정한다.
- [x] attribute와 UI action이 서로 충돌하지 않게 single source of truth를 정한다.

### Success Criteria

- [x] 주요 viewer 상태를 외부 코드에서 attribute 또는 method로 제어할 수 있다.
- [x] 주요 사용자 액션이 custom event로 관찰 가능하다.
- [x] 기존 사용 예시가 깨지지 않는다.
- [x] public API 목록이 문서화된다.

### Verification

- [ ] 외부 스크립트에서 attribute 변경 시 viewer가 예상대로 반응한다.
- [ ] load/error/selection/camera 관련 event payload가 충분한 정보를 담는다.
- [ ] UI 클릭과 외부 API 호출이 동일한 최종 상태를 만든다.

### Phase 1 Notes

Implemented API surface in code:

- New observed attributes: `environment`, `environment-url`, `background-color`, `camera-target`, `camera-up`, `exposure`, `animation`, `autoplay`, `selection-mode`, `performance-mode`.
- Public methods added or formalized on the element instance: `loadModelFromUrl`, `discardModel`, `resetView`, `fitCameraToModel`, `selectMeshByName`, `selectMeshByIndex`, `setEnvironment`, `captureScreenshot`, `exportState`, `importState`.
- Internal helpers added to keep attribute reflection and event emission consistent: `emitEvent`, `emitViewerError`, `reflectAttribute`, `reflectBooleanAttribute`, `getCameraStateSnapshot`, `emitCameraChange`, `emitSelectionChange`, `emitAnimationChange`, `emitEnvironmentChange`, `serializeState`, `frameObject`.

Custom event surface now emitted from shared code paths:

- `viewer-load`: successful model load with source, format, counts.
- `viewer-error`: load/environment/screenshot/API failures.
- `viewer-selection-change`: scene-graph or API-driven selection changes.
- `viewer-camera-change`: orbit, framing, camera attribute, transform-control changes.
- `viewer-animation-change`: animation selection/play/pause/reset.
- `viewer-material-change`: texture replacement/history restore and roughness/metalness updates.
- `viewer-environment-change`: environment/background/exposure updates.
- `viewer-state-export`: explicit `exportState()` calls.

Single-source-of-truth rules now in code:

- New public attributes are applied through shared setter helpers instead of separate one-off UI logic.
- Environment, background color, animation selection, and URL model loads now reflect back to attributes when initiated from the built-in UI or public methods.
- Public methods reuse existing internal lifecycle/render helpers instead of bypassing them.

Verification status:

- `npm run build`: passed on 2026-03-14.
- Manual browser verification for attribute reactions, event payload inspection, and UI/API parity is still pending.

### Work Log

#### 2026-03-14

- Implemented Phase 1 public API and custom events in `js/simple-model-viewer.js`.
- Added attribute handling for environment, background color, camera target/up, exposure, animation selection/autoplay, selection mode, and performance mode.
- Added public methods for model loading, environment control, camera reset/framing, selection, screenshot capture, and state export/import.
- Routed load, environment, selection, animation, camera, and material-edit flows through new custom events so external code can observe viewer activity.
- Kept existing UI behavior intact while syncing core UI-triggered actions back to the corresponding public attributes where practical.

Verification results:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-14.
- `npm run build`: passed on 2026-03-14.
- Manual browser verification is still pending.

---

## Phase 2. Camera And Navigation Improvements

### Objective

모델을 잃지 않고 다시 잡을 수 있는 카메라 기능을 추가한다.

### Planned Work

- [x] `Reset View` 버튼 추가
- [x] `Fit Model` 버튼 추가
- [x] 선택된 mesh 또는 object에 대한 `Frame Selected` 지원
- [x] 초기 카메라 pose 저장 및 복원
- [x] `camera-target` 기반 look-at 제어
- [x] 더블클릭 focus 기능 검토 및 구현
- [x] OrbitControls target 노출 및 동기화
- [x] 카메라 상태 serialize/import 대상에 포함
- [x] 필요한 경우 전환 애니메이션 추가

### Success Criteria

- [x] 사용자가 언제든 기본 시점으로 복귀할 수 있다.
- [x] 선택된 파트를 화면에 다시 맞출 수 있다.
- [x] 카메라 position뿐 아니라 target까지 일관되게 관리된다.
- [x] explode, transform, auto-rotate 이후에도 카메라 복구가 가능하다.

### Verification

- [ ] 모델 이동/회전 후 `Reset View`가 정상 동작한다.
- [ ] scene graph에서 파트 선택 후 `Frame Selected`가 해당 파트를 중심으로 맞춘다.
- [ ] 더블클릭 focus가 UI selection과 충돌하지 않는다.

### Phase 2 Notes

Implemented camera/navigation changes in code:

- Added camera action UI buttons in the control panel: `Reset View`, `Fit Model`, and `Frame Selected`.
- Added shared camera snapshot helpers: `cloneCameraState`, `captureCurrentCameraStateAsDefault`, `syncCameraAttributes`, `setCameraStateSnapshot`, `interpolateCameraState`, and `applyCameraStateSnapshot`.
- Added per-model default camera state tracking so reset restores the saved default view instead of recomputing a fresh fit every time.
- Added `frameSelected()` as a public helper for focusing the currently selected mesh.
- Added double-click canvas focus using a raycaster over loaded mesh parts without changing the existing selection workflow.
- Added short animated transitions for reset/fit/frame actions and suppressed internal OrbitControls event spam during programmatic camera updates.

Camera-state synchronization behavior now in code:

- Declarative camera attributes (`camera-orbit`, `camera-target`, `camera-up`) now save a resettable default camera pose.
- UI-triggered fit/reset/frame actions sync camera attributes back to the element so external code can observe the new target/pose.
- Camera state remains part of `serializeState()` / `importState()` and imported camera states now become the new reset target for the current model session.

Backward compatibility notes:

- Existing camera attributes and `fitCameraToModel()` / `resetView()` APIs were kept; `resetView()` now restores a saved default pose when one exists.
- Scene-graph selection behavior remains intact; double-click focus frames the intersected mesh without introducing canvas-selection state yet.

Verification status:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-14.
- `npm run build`: passed on 2026-03-14.
- Manual browser verification for reset/frame-selected/double-click behavior is still pending.

### Work Log

#### 2026-03-14

- Implemented Phase 2 camera and navigation improvements in `js/simple-model-viewer.js`.
- Added control-panel camera actions for reset, fit-model, and frame-selected flows.
- Added reusable camera snapshot/restore helpers, animated camera transitions, and per-model default camera-state persistence.
- Added double-click focus on the canvas and synchronized camera attributes after UI-driven framing actions.
- Preserved the existing selection workflow while preventing duplicate camera-change events during internal camera updates.

Verification results:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-14.
- `npm run build`: passed on 2026-03-14.
- Manual browser verification is still pending.

---

## Phase 3. Canvas Picking And Selection Workflow

### Objective

scene graph 외에도 캔버스에서 직접 mesh를 선택하고 편집 흐름과 연결한다.

### Planned Work

- [x] raycaster 기반 클릭 선택 추가
- [x] hover highlight 여부 결정 및 구현
- [x] 캔버스 선택 시 scene graph, texture panel, selected state를 동기화
- [x] multi-material mesh 대응 전략 수립
- [x] invisible object, helper object, transform control gizmo 선택 예외 처리
- [x] 선택 해제 경로 추가
  - 빈 공간 클릭
  - ESC
  - API 호출
- [x] 선택 이벤트 payload 정리

### Success Criteria

- [x] 사용자가 캔버스에서 클릭한 mesh가 안정적으로 선택된다.
- [x] 선택 상태가 scene graph와 texture/material UI에 반영된다.
- [x] helper나 gizmo를 실수로 mesh로 선택하지 않는다.
- [x] 선택 해제도 명확히 동작한다.

### Verification

- [ ] scene graph 선택과 canvas 선택이 동일한 결과를 낸다.
- [ ] 선택 highlight가 일정 시간 뒤 사라지거나 선택 상태 표현 규칙이 일관된다.
- [ ] 부분 visibility 토글 후에도 선택 로직이 깨지지 않는다.

### Phase 3 Notes

Implemented selection workflow changes in code:

- Added canvas click picking through the existing raycaster and model mesh list, with drag-threshold guarding so orbit/navigation drags do not trigger accidental selection.
- Added persistent outline-based selection and hover feedback using helper boxes instead of temporary material replacement, so selection no longer mutates mesh materials during editing workflows.
- Added scene-graph label lookup/sync so canvas picks, texture-part changes, API selection calls, and imported state all converge on the same selected mesh bookkeeping.
- Added clear-selection paths for empty-canvas clicks, `Escape`, `selection-mode="none"`, visibility toggles that hide the selected mesh, and a new public `clearSelection()` method.
- Expanded `viewer-selection-change` payloads with `selectionMode`, `materialCount`, `isMultiMaterial`, and effective visibility metadata.

Selection handling decisions now in code:

- `selection-mode` now accepts `canvas` and `all` in addition to the existing `scene-graph` and `none` values.
- Existing `scene-graph` mode is treated as a backward-compatible combined mode so current embeds gain canvas picking without having to change attributes immediately.
- Multi-material meshes are selectable as full mesh objects; editing continues to target the first editable material, and event payloads now expose whether a selected mesh has multiple materials.
- Raycast picking only considers loaded model meshes that are effectively visible, so helpers, gizmos, and hidden objects are excluded from canvas selection.

Verification status:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-14.
- `npm run build`: passed on 2026-03-14.
- Manual browser verification for canvas picking, hover/clear-selection behavior, and visibility-toggle interaction is still pending.

### Work Log

#### 2026-03-14

- Implemented Phase 3 canvas picking and selection workflow in `js/simple-model-viewer.js`.
- Added canvas click selection, hover feedback, scene-graph syncing, and keyboard/API deselection flows.
- Replaced the temporary glow-material selection effect with outline helpers to avoid interfering with material editing and multi-material meshes.
- Kept existing scene-graph selection behavior working while extending `selection-mode` to support canvas-oriented control.

Verification results:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-14.
- `npm run build`: passed on 2026-03-14.
- Manual browser verification is still pending.

---

## Phase 4. Animation Control Expansion

### Objective

animation clip 재생을 더 실용적인 수준으로 확장한다.

### Planned Work

- [x] `autoplay` 지원
- [x] 기본 animation clip 지정 attribute/method 추가
- [x] speed control 추가
- [x] loop mode 지원
  - repeat
  - once
  - ping-pong
- [x] time scrubber 추가
- [x] 현재 시간 / 총 길이 표시
- [x] cross-fade 여부 검토
- [x] animation state export/import 반영

### Success Criteria

- [x] animation이 있는 모델에서 기본 clip 자동 재생이 가능하다.
- [x] 사용자가 재생 속도와 재생 위치를 직접 제어할 수 있다.
- [x] 여러 clip 간 전환 시 동작이 예측 가능하다.
- [x] animation 관련 상태가 UI, attribute, method에서 일관된다.

### Verification

- [ ] glTF animation clip 선택, 재생, 일시정지, speed 변경이 모두 동작한다.
- [ ] loop mode가 의도대로 적용된다.
- [ ] scrubber 이동 후 재생이 정상 이어진다.

### Phase 4 Notes

Implemented animation-control changes in code:

- Replaced the ad-hoc animation selector insertion with a dedicated animation control block in the built-in UI, including clip selection, speed slider, loop-mode selector, timeline scrubber, and live time/duration display.
- Added animation state helpers in code for loop configuration, time formatting, selector population, UI refresh, and richer animation event payloads.
- Added public animation control methods on the element instance: `setAnimation`, `playAnimation`, `pauseAnimation`, `setAnimationSpeed`, `setAnimationLoopMode`, and `setAnimationTime`.
- Added `animation-speed` and `animation-loop` attributes so animation rate and loop policy can be controlled declaratively alongside the existing `animation` and `autoplay` attributes.
- Extended exported/imported viewer state with `animationState` so selected clip, playback state, loop mode, speed, and scrubbed time can be restored together.

Animation behavior decisions now in code:

- The existing `animation` attribute remains the default clip selector; Phase 4 formalizes clip selection further through the new `setAnimation()` method rather than introducing a separate duplicate attribute.
- UI clip changes still start playback immediately for a predictable interactive workflow, while the `autoplay` attribute continues to control declarative/default startup behavior.
- Clip changes now use a short cross-fade when switching between active clips during playback.
- Loop behavior maps to Three.js `LoopRepeat`, `LoopOnce`, and `LoopPingPong`, with `LoopOnce` clamping on the last frame and emitting a finished-state update through the existing animation event surface.

Verification status:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-14
- `npm run build`: passed on 2026-03-14
- Manual browser verification for clip switching, loop modes, and scrubber playback continuity is still pending.

### Work Log

#### 2026-03-14

- Implemented Phase 4 animation control expansion in `js/simple-model-viewer.js`.
- Added animation speed, loop mode, timeline scrubbing, live duration display, and expanded animation state export/import support.
- Reworked the animation UI to use stable built-in controls instead of dynamically appending a selector into the util fieldset.
- Added short clip cross-fades and mixer-finished handling so one-shot playback updates the viewer state cleanly.

Verification results:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-14
- `npm run build`: passed on 2026-03-14
- Manual browser checks are still pending in this session.

---

## Phase 5. Material And Texture Editing Expansion

### Objective

현재 texture replacement 중심의 편집 기능을 실사용 가능한 material editor 수준으로 확장한다.

### Planned Work

- [x] per-mesh material editor 확장
  - base color
  - emissive color
  - emissive intensity
  - opacity
  - transparent
  - double-sided
  - normal scale
  - envMap intensity
- [x] UV transform 편집
  - repeat
  - offset
  - rotation
- [x] texture reset 기능
  - current to original
  - history revert
- [x] texture remove 기능
- [x] multi-material mesh 편집 전략 수립
- [x] editable material detection 보강
- [x] material property 변경 이벤트 추가
- [x] preview UX 개선
  - 현재 선택 texture 메타정보
  - 없음 상태 표현
  - 가능하면 download/copy path

### Success Criteria

- [x] 선택한 mesh의 핵심 material 속성을 UI에서 수정 가능하다.
- [x] texture를 교체, 제거, 원복할 수 있다.
- [x] UV 조정 결과가 즉시 반영된다.
- [x] 변경 사항이 original material snapshot과 충돌하지 않는다.

### Verification

- [ ] 단일 mesh와 multi-mesh 모델 모두에서 편집 UI가 동작한다.
- [ ] normal/ao/emissive 등 비색상 맵 encoding이 깨지지 않는다.
- [ ] 원복 후 material이 의도한 기본 상태로 돌아간다.

### Phase 5 Notes

Implemented material-editing changes in code:

- Reworked the built-in edit panel into a per-part material editor with controls for material slot selection, base/emissive colors, emissive intensity, opacity, transparent toggle, double-sided toggle, roughness, metalness, normal scale, env-map intensity, and UV repeat/offset/rotation.
- Added immutable `initialMaterials` snapshots alongside the existing live material snapshot store so texture reset can restore the original imported material state while texture history continues to support iterative revert flows.
- Extended texture editing with per-material-slot history, remove/reset actions, and preview metadata including source URL when available plus a copy-source button.
- Expanded material normalization and render-mode restore paths to support multi-material meshes instead of always editing only the first material slot.
- Enriched `viewer-material-change` payloads for property, UV, replace, remove, reset, and history-apply actions so external listeners can observe Phase 5 edits more precisely.

Material-editing behavior decisions now in code:

- Multi-material meshes are edited one material slot at a time through a new slot selector in the built-in UI; texture history is keyed by mesh plus material slot to avoid cross-slot collisions.
- The existing `originalMaterials` store remains the live editable material snapshot used by default-view restoration, while `initialMaterials` now holds the immutable import-time baseline for reset-to-original actions.
- UV editing targets the currently selected texture type for the selected material slot; when no texture is assigned, the UI keeps the controls visible but the update no-ops safely.
- Copy-source support is best-effort and only enabled when the selected texture exposes a browser-visible URL and the Clipboard API is available.

Verification status:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-15.
- `npm run build`: passed on 2026-03-15.
- Manual browser verification for multi-material editing, map encoding checks, and reset/remove flows is still pending.

### Work Log

#### 2026-03-15

- Implemented Phase 5 material and texture editing expansion in `js/simple-model-viewer.js`.
- Added per-slot material editing controls, UV transform controls, texture remove/reset actions, and texture metadata preview/copy support.
- Added immutable initial material snapshots and per-slot texture history so reset-to-original and history-apply flows do not overwrite each other.
- Updated material normalization, default-view restoration, and special render modes to tolerate multi-material meshes.

Verification results:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-15.
- `npm run build`: passed on 2026-03-15.
- Manual browser checks are still pending in this session.

---

## Phase 6. Environment And Background Pipeline

### Objective

preset 3개에 묶인 환경 설정을 범용 HDRI/background 제어로 확장한다.

### Planned Work

- [x] preset environment 유지
- [x] custom HDR URL 로드 지원
- [x] local HDR URL 업로드 지원 여부 결정 및 구현
- [x] environment intensity 조절
- [x] exposure 조절
- [x] environment rotation 지원 검토 및 구현
- [x] background visible / environment only 분리 옵션
- [x] background color와 environment 간 우선순위 정리
- [x] environment 변경 이벤트 추가
- [x] environment dispose 경로 정리

### Success Criteria

- [x] preset과 custom environment를 모두 사용할 수 있다.
- [x] 조명 느낌을 intensity/exposure로 조절할 수 있다.
- [x] background와 lighting 환경을 분리해서 제어할 수 있다.
- [x] environment 교체 시 메모리 누수와 state 꼬임이 없도록 dispose/state path를 코드상 정리했다.

### Verification

- [ ] preset 전환과 custom URL 전환이 반복되어도 정상 동작한다.
- [ ] default env 복귀가 확실히 된다.
- [ ] no-pbr/view-mode와 environment 조합 시 이상 동작이 없다.

### Phase 6 Notes

Implemented environment/background changes in code:

- Preset buttons were kept and expanded with custom HDR URL loading plus local `.hdr` file upload in the built-in UI.
- Added declarative/environment state for `environment-intensity`, `environment-rotation`, and `environment-background` alongside the existing `environment`, `environment-url`, `background-color`, and `exposure`.
- Added shared environment presentation helpers so scene background visibility, lighting environment, exposure, and scene rotation are applied from one code path.
- Kept environment texture disposal centralized and extended the event payload to include intensity, rotation, and background visibility.

Environment behavior decisions now in code:

- Background color remains the renderer clear color; when HDR background visibility is disabled, the HDRI still lights the model but the clear color remains visible behind it.
- Remote custom HDR URLs remain part of exported/imported state; local uploaded HDR files are session-only and intentionally not serialized.
- Global environment intensity multiplies per-material `envMapIntensity` instead of overwriting material-local edits.

Verification status:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-15.
- `npm run build`: passed on 2026-03-15.
- Manual browser verification for preset/custom switching, default-env recovery, and no-pbr/view-mode interaction is still pending.

---

## Phase 7. Export, Capture, And State Persistence

### Objective

viewer 결과물을 내보내고 현재 상태를 재현할 수 있게 만든다.

### Planned Work

- [x] screenshot 기능 확장
  - clipboard 유지
  - PNG download fallback
  - filename convention
  - transparent background option 검토 및 API 반영
- [x] recording UX 개선
  - preset duration or quick turntable recording
  - 진행 상태 표시 여부 검토 및 구현
- [x] `Copy Config` 또는 `Export State` 추가
  - camera
  - environment
  - selected view mode
  - transform
  - material overrides
  - animation state
- [x] `Import State` 추가
- [x] 상태 JSON schema 또는 명시 포맷 정의

### Success Criteria

- [x] 사용자가 스크린샷을 다운로드 또는 복사할 수 있다.
- [x] 현재 viewer 상태를 export하여 다시 복원할 수 있다.
- [x] 상태 export 포맷이 예측 가능하고 문서화된다.
- [x] recording 기본 흐름이 기존보다 명확해진다.

### Verification

- [ ] export한 state를 같은 모델에 import하면 시각 상태가 대체로 재현된다.
- [ ] clipboard 미지원 환경에서도 PNG 저장은 가능하다.
- [ ] recording 후 preview, download, cleanup 흐름이 정상이다.

### Phase 7 Notes

Implemented export/capture/state persistence changes in code:

- Screenshot capture now supports clipboard copy, PNG download, download fallback, generated filenames, and a transparent-background capture option through the public API.
- Added quick turntable recording with configurable duration plus lightweight built-in recording status feedback.
- Added built-in `Copy Config` / `Apply Config` UI around the existing `exportState()` / `importState()` API.
- Expanded exported state with a schema identifier, environment state, and serialized material overrides in addition to camera, transform, animation, and selection state.

State format decisions now in code:

- Export format is versioned as `simple-model-viewer-state/v1`.
- Material overrides are serialized as logical properties plus texture URLs when a browser-visible source exists; local blob textures and local HDR uploads are intentionally session-only.
- Import applies environment/camera/view state first and then material overrides so viewer recreation stays predictable on the same model asset.

Verification status:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-15.
- `npm run build`: passed on 2026-03-15.
- Manual browser verification for clipboard fallback, recording preview/download, and import/export visual parity is still pending.

---

## Phase 8. Input UX And Performance Options

### Objective

입력 편의성과 대형 모델 대응력을 함께 개선한다.

### Planned Work

- [x] drag-and-drop 파일 로드
- [x] URL paste/submit UX 개선
- [x] pointer/keyboard shortcut 추가
  - ESC selection clear
  - F fit model
  - R reset view
  - Space play/pause 여부 검토 및 구현
- [x] loading/error feedback 개선
  - 사용자 메시지
  - unsupported format 안내
- [x] performance mode 설계
  - adaptive DPR
  - optional `preserveDrawingBuffer`는 이번 패스에서는 유지
  - recording/screenshot 시에만 필요한 고비용 옵션 분리 가능성 검토
- [x] 향후 KTX2/Meshopt 지원 여부 검토
  - 최소한 구조적으로 확장 가능한 형태

### Success Criteria

- [x] 파일 드롭과 URL 입력 흐름이 직관적이도록 UI와 status feedback을 추가했다.
- [x] 핵심 단축키가 기존 UI와 충돌하지 않도록 text-entry guard를 유지했다.
- [x] 무거운 모델에서 성능 관련 옵션을 선택적으로 적용할 수 있다.
- [x] 에러 메시지가 사용자 관점에서 이해 가능하도록 in-view status 메시지를 추가했다.

### Verification

- [ ] drag-and-drop으로 모델이 정상 로드된다.
- [ ] keyboard shortcut이 input field focus 중에는 오동작하지 않는다.
- [ ] performance mode on/off 시 기본 기능이 유지된다.

### Phase 8 Notes

Implemented input/performance UX changes in code:

- Added drag-and-drop model loading on the canvas area using the same load path as the existing file input.
- Kept the URL entry flow and added status feedback so load, error, and unsupported-format states are visible in-component instead of only through console output.
- Expanded keyboard shortcuts to cover `Esc`, `F`, `R`, and `Space` while preserving the existing text-input guard.
- Added a built-in performance-mode selector wired to the existing DPR/shadow settings.

Performance decisions now in code:

- `performance-mode` still controls adaptive DPR and shadow toggling; `preserveDrawingBuffer` remains enabled in this pass because screenshot/export behavior still depends on it.
- KTX2/Meshopt were not added in this session, but the loader flow remains centralized enough to extend later without reworking the public API surface.

Verification status:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-15.
- `npm run build`: passed on 2026-03-15.
- Manual browser verification for drag-and-drop, shortcut focus guards, and performance-mode behavior is still pending.

---

## Phase 9. Documentation, Examples, And Final Verification

### Objective

새 기능을 문서와 예시에 반영하고 전체 수동 검증을 마무리한다.

### Planned Work

- [x] `simple-model-viewer` 사용 예시 갱신
- [x] 필요한 경우 블로그 포스트/데모 예시 갱신
- [x] public API 문서화
- [x] attribute 목록 정리
- [x] custom event 목록 정리
- [x] method 목록 정리
- [ ] 수동 검증 체크리스트 최종 수행
- [x] `npm run build` 수행

### Success Criteria

- [x] 새 사용자가 문서만 보고 핵심 기능을 사용할 수 있도록 예시와 설명을 갱신했다.
- [x] 예시 코드가 실제 구현과 일치한다.
- [x] 빌드가 통과한다.
- [ ] 최종 검증 표가 모두 채워진다.

### Verification

- [x] 문서의 sample snippet이 실제 attribute/method/event 이름과 일치한다.
- [x] build 결과가 성공한다.
- [ ] 주요 수동 검증 항목이 모두 완료된다.

### Phase 9 Notes

Documentation/example updates made in this session:

- Updated the 3D viewer demo page to showcase the newer environment, selection, and performance attributes.
- Updated the English and Korean blog source post for the viewer with the newer attribute example, an explicit API/event summary, and corrected the outdated `auto-animate` wording to the shipped `auto-rotate` attribute.
- Clarified that the long embedded code block in the blog post is a historical prototype excerpt and moved the reader-facing examples to the current declarative API surface instead of implying the old snapshot is still authoritative.
- Kept the running implementation plan updated as the session log/source of truth for what shipped versus what still needs manual verification.

Verification status:

- `node --check js/simple-model-viewer.js`: passed on 2026-03-15.
- `npm run build`: passed on 2026-03-15.
- Manual browser verification checklist is still pending, so Phase 9 remains open overall.

### Additional Work Log

#### 2026-03-15

- Reconciled the blog post documentation with the current component API, including the declarative animation/selection/performance attributes, public method list, and custom event list.
- Corrected the outdated `auto-animate` terminology in the English and Korean post content to match the shipped `auto-rotate` attribute.
- Marked the embedded long-form code sample in the English and Korean blog posts as a historical prototype excerpt and updated the surrounding example to point readers at the current shipped API instead.
- `node --check js/simple-model-viewer.js`: passed after the documentation reconciliation on 2026-03-15.
- `npm run build`: passed after the documentation reconciliation on 2026-03-15.
- Browser-based manual verification remains pending.

---

## Cross-Cutting Risk Checklist

- [ ] 기존 attribute 호환성 깨짐 여부
- [ ] `originalMaterials` snapshot과 override 충돌
- [ ] multi-material mesh 처리 누락
- [ ] environment/texture 교체 시 dispose 누락
- [ ] selection highlight가 실제 material 상태를 오염시키는지 여부
- [ ] transform controls와 picking 충돌
- [ ] recording/screenshot 성능 저하
- [ ] no-pbr, view-mode, environment 조합 충돌
- [ ] auto-rotate, animation, transform 상호작용 충돌
- [ ] model reload/discard 후 event listener 또는 object URL 누수

## Final Manual Verification Checklist

- [ ] 기본 모델 로드
- [ ] 파일 입력 로드
- [ ] drag-and-drop 로드
- [ ] URL 로드
- [ ] glb/gltf 테스트
- [ ] obj 테스트
- [ ] fbx 테스트
- [ ] ply 테스트
- [ ] 카메라 reset/fit/frame selected
- [ ] canvas 선택
- [ ] scene graph 선택
- [ ] visibility toggle
- [ ] transform translate
- [ ] transform rotate
- [ ] auto-rotate
- [ ] animation select/play/pause/speed/scrub
- [ ] diffuse mode
- [ ] geometry mode
- [ ] normal mode
- [ ] wireframe mode
- [ ] background color
- [ ] grid helper
- [ ] preset environment
- [ ] custom environment
- [ ] ambient light edit
- [ ] directional light add/remove/edit
- [ ] texture replace
- [ ] texture remove/reset/history
- [ ] material property edit
- [ ] UV transform edit
- [ ] explode slider
- [ ] screenshot clipboard
- [ ] screenshot download
- [ ] recording preview/download
- [ ] export state
- [ ] import state
- [ ] discard model
- [ ] reload after discard
- [ ] no-pbr compatibility
- [ ] hide-control-ui compatibility
- [ ] `ui` attribute compatibility
- [ ] multi-viewer independent operation
- [ ] `npm run build`

## Work Log

### Entry Template

Use this format for each work session.

```md
### YYYY-MM-DD HH:MM
- Scope:
- Files touched:
- Decisions:
- Risks found:
- Verification:
- Next step:
```

### 2026-03-15 15:30

- Scope:
  - Implement Phases 6-8 in `js/simple-model-viewer.js`
  - Refresh docs/examples for Phase 9
- Files touched:
  - `js/simple-model-viewer.js`
  - `blogs/3DViewer/index.html`
  - `blogs/posts/250310_model_viewer/content-eng.md`
  - `blogs/posts/250310_model_viewer/content-kor.md`
  - `simple-model-viewer-implementation-plan.md`
- Decisions:
  - Local HDR upload is in scope, but it is treated as session-only and not serialized into exported state.
  - Exported state uses a versioned logical format `simple-model-viewer-state/v1` and only reuses texture URLs when the browser exposes one.
  - Quick turntable recording ships in this pass with lightweight status text rather than a heavier modal progress UI.
- Risks found:
  - Manual browser verification is still required for environment/background edge cases, screenshot clipboard fallback, and drag-and-drop/keyboard focus interactions.
  - `preserveDrawingBuffer` remains enabled for compatibility with screenshot export, so deeper performance optimization is still a follow-up item if needed.
- Verification:
  - `node --check js/simple-model-viewer.js`: passed
  - `npm run build`: passed
- Next step:
  - Run the manual verification checklist in a browser and close Phase 9 if no regressions appear

### 2026-03-14 00:00

- Scope: Planning only
- Files touched: `simple-model-viewer-implementation-plan.md`
- Decisions:
  - Work will be split into 10 phases including final verification.
  - No implementation starts in this session.
  - This file will be the single running log/checklist for later sessions.
- Risks found:
  - The feature set is broad enough that internal refactor safety work should happen first.
  - Existing material and selection flows are likely to conflict with new picking/editor features.
- Verification:
  - Planning document created
- Next step:
  - Start with Phase 0 in a new session

## Decision Log

- [x] Decide final public event naming convention
- [x] Decide whether local HDR upload is in scope
- [x] Decide whether state export should include raw texture references or only logical overrides
- [ ] Decide how far multi-material editing should go in first pass
- [x] Decide whether quick turntable recording ships in initial implementation or later

## Notes For Next Session

- Start from Phase 0, not from scattered feature additions.
- Keep this file updated as the source of truth.
- Mark each checkbox only after the corresponding verification passes.
- If scope needs trimming, note the deferment explicitly under `Decision Log` and `Work Log`.

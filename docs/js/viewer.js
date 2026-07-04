const HF_BASE = 'https://huggingface.co/datasets/willi19/object_processing/resolve/main/';
const DATA_VERSION = '20260704-willi19-9aaa4ce-thumbnail-poses-v26';

// Pipeline stages the viewer can toggle between. `file` is relative to the
// object dir on HuggingFace; `simplified` falls back to the legacy mesh.glb.
// Every stage falls back to mesh.glb so the viewer still shows the object when a
// stage GLB hasn't been uploaded yet (otherwise raw/coacd 404 to a blank screen).
const STAGES = [
  { id: 'raw',        label: 'Raw',        file: 'stages/raw.glb',        fallback: 'mesh.glb' },
  { id: 'manifold',   label: 'Manifold',   file: 'stages/manifold.glb',   fallback: 'mesh.glb' },
  { id: 'coacd',      label: 'CoACD',      file: 'stages/coacd.glb',      fallback: 'mesh.glb' },
  { id: 'simplified', label: 'Simplified', file: 'stages/simplified.glb', fallback: 'mesh.glb' },
];
const DEFAULT_STAGE = STAGES[STAGES.length - 1];  // simplified

// Not user-selectable: the full textured display mesh. Shown automatically while
// a 3D overlay that needs a clean hero shot (symmetry axis / tabletop poses) is
// on, in place of the processed stage mesh.
const MESH_STAGE = { id: 'textured', label: 'Textured', file: 'mesh.glb', fallback: 'mesh.glb' };

const AXIS_COLORS = [
  new BABYLON.Color3(1.0, 0.82, 0.10),  // primary symmetry axis
  new BABYLON.Color3(0.20, 0.85, 0.40),
  new BABYLON.Color3(0.36, 0.62, 0.95),
];

function versionedDataUrl(path) {
  const url = new URL(path, window.location.href);
  if (url.origin === window.location.origin && !url.searchParams.has('v')) {
    url.searchParams.set('v', DATA_VERSION);
  }
  return url.toString();
}

async function fetchJson(path, fallback = null) {
  try {
    const res = await fetch(versionedDataUrl(path), { cache: 'reload' });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.warn(`Could not load ${path}`, err);
    return fallback;
  }
}

function splitUrl(url) {
  const i = url.lastIndexOf('/');
  if (i < 0) return ['', url];
  return [url.slice(0, i + 1), url.slice(i + 1)];
}

function makeContainerMaterialsDoubleSided(container) {
  (container.materials || []).forEach((material) => {
    material.backFaceCulling = false;
    if ('twoSidedLighting' in material) material.twoSidedLighting = true;
  });
}

(async function () {
  const params = new URLSearchParams(window.location.search);
  const objectId = params.get('id');
  if (!objectId) { showError('No object ID specified.'); return; }

  // Stage GLBs normally load from HuggingFace. ?local=1 loads them from
  // objects/{id}/stages/ under this site instead, for testing before upload.
  const useLocal = params.get('local') === '1';

  const catalog = await fetchJson('catalog.json', null);
  if (!catalog) { showError('Failed to load catalog.'); return; }

  const obj = (catalog.objects || []).find(o => o.id === objectId);
  if (!obj) { showError(`Object "${objectId}" not found in catalog.`); return; }
  const meshObjectId = obj.mesh_id || objectId;
  const assetBase = (useLocal ? '' : HF_BASE) + 'objects/' + encodeURIComponent(meshObjectId) + '/';

  const textureManifest = await fetchJson('texture_overrides/manifest.json', { objects: {} });
  const textureOverride = (textureManifest.objects || {})[objectId] || null;
  const catalogMesh = obj.url && !obj.url.startsWith('objects/') ? obj.url : '';
  const replacementMeshUrl = (textureOverride && textureOverride.mesh) || catalogMesh;
  const texturedStage = replacementMeshUrl
    ? { ...MESH_STAGE, label: 'Textured scan', url: replacementMeshUrl }
    : MESH_STAGE;
  const stageOptions = replacementMeshUrl ? [...STAGES, texturedStage] : STAGES;

  document.getElementById('obj-label').textContent = obj.label;
  document.title = `${obj.label} - Object Viewer`;
  if (obj.category) {
    const catEl = document.getElementById('obj-category');
    catEl.textContent = obj.category;
    catEl.style.display = '';
  }

  // Optional IKEA purchase link (separate file, merged client-side).
  try {
    const ikea = (await (await fetch('ikea.json')).json()).objects || {};
    const k = ikea[objectId];
    if (k && k.link) {
      const buy = document.getElementById('buy-link');
      buy.href = k.link;
      buy.title = `${k.name} — ${k.description}`;
      buy.style.display = '';
    }
  } catch (e) { /* no purchase link for this object */ }

  // Overlay metadata (OBB / symmetry / tabletop), served locally. Optional.
  let info = null;
  info = await fetchJson(`objects/${objectId}/info.json`, null);

  // ── Babylon scene ──────────────────────────────────────────────────────────
  const canvas = document.getElementById('renderCanvas');
  const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.051, 0.067, 0.09, 1);
  // glTF and our overlay data are both right-handed (object frame). Keep the
  // scene right-handed so the glTF __root__ is identity and overlays built in
  // object-frame coordinates line up with the mesh without conversion.
  scene.useRightHandedSystem = true;

  const camera = new BABYLON.ArcRotateCamera('cam', Math.PI / 4, Math.PI / 3, 2, BABYLON.Vector3.Zero(), scene);
  camera.attachControl(canvas, true);
  camera.wheelPrecision = 50;
  camera.minZ = 0.001;
  camera.lowerRadiusLimit = 0.05;
  camera.upperRadiusLimit = 50;
  camera.angularSensibilityX = 3000;
  camera.angularSensibilityY = 3000;
  camera.panningSensibility = 3000;
  camera.lowerBetaLimit = null;
  camera.upperBetaLimit = null;

  new BABYLON.HemisphericLight('hemi', new BABYLON.Vector3(0, 1, 0), scene).intensity = 0.85;
  const dir = new BABYLON.DirectionalLight('dir', new BABYLON.Vector3(-1, -2, 1), scene);
  dir.intensity = 0.55;

  // Image-based lighting + ACES tone mapping: glTF materials are PBR, so without
  // an environment to reflect they look flat and matte. Non-fatal if the env
  // asset can't be fetched — the direct lights above still shade the mesh.
  try {
    scene.environmentTexture = BABYLON.CubeTexture.CreateFromPrefilteredData(
      'https://assets.babylonjs.com/environments/environmentSpecular.env', scene);
    scene.environmentIntensity = 0.85;
    const ip = scene.imageProcessingConfiguration;
    ip.toneMappingEnabled = true;
    ip.toneMappingType = BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
    ip.contrast = 1.1;
  } catch (e) { /* direct lights only */ }

  // State shared across stage loads.
  let currentContainer = null;
  let selectedStage = replacementMeshUrl ? texturedStage : DEFAULT_STAGE;  // stage the user picked via the radios
  let loadedStageId = null;           // id of the stage actually in the scene
  let loadedFromReplacement = false;
  let currentRoot = null;          // glTF __root__ node; overlays parent here
  let standWrap = null;            // wraps currentRoot with the standing pose
  let floorMesh = null;            // ground plane under the standing object
  let overlays = { obb: null, symmetry: null, tabletop: null, axes: null };
  const shown = { obb: false, symmetry: false, tabletop: false, axes: false };
  // Compare (raw ↔ simplified wipe) state.
  let cmp = { containers: [], minX: -1, maxX: 1 };
  let cut = 0;
  let compareOn = false;

  function disposeOverlays() {
    for (const k of Object.keys(overlays)) {
      if (overlays[k]) { overlays[k].dispose(); overlays[k] = null; }
    }
  }

  async function loadStageContainer(stage) {
    const candidates = [];
    if (stage.url) candidates.push({ url: stage.url });
    candidates.push({ root: assetBase, file: stage.file });
    if (stage.fallback && stage.fallback !== stage.file) {
      candidates.push({ root: assetBase, file: stage.fallback });
    }

    let lastError = null;
    for (const spec of candidates) {
      try {
        if (spec.url) {
          const [rootUrl, file] = splitUrl(versionedDataUrl(spec.url));
          const container = await BABYLON.SceneLoader.LoadAssetContainerAsync(rootUrl, file, scene);
          return { container, href: spec.url, replacement: true };
        }
        const container = await BABYLON.SceneLoader.LoadAssetContainerAsync(spec.root, spec.file, scene);
        return { container, href: spec.root + spec.file, replacement: false };
      } catch (err) {
        lastError = err;
        console.warn(`mesh load failed for ${objectId}`, spec, err);
      }
    }
    throw lastError || new Error('No mesh URL loaded');
  }

  async function loadStage(stage) {
    document.getElementById('loading').style.display = '';
    document.getElementById('error').style.display = 'none';
    disposeOverlays();
    if (standWrap) { standWrap.dispose(); standWrap = null; }
    if (floorMesh) { floorMesh.dispose(); floorMesh = null; }
    if (currentContainer) { currentContainer.dispose(); currentContainer = null; }

    const loaded = await loadStageContainer(stage);
    currentContainer = loaded.container;
    if (loaded.replacement) makeContainerMaterialsDoubleSided(currentContainer);
    currentContainer.addAllToScene();
    loadedStageId = stage.id;
    loadedFromReplacement = loaded.replacement;
    currentRoot = contentRoot(currentContainer);

    applyReplacementScale();
    applyStandingPose();   // rotate the object upright onto the floor
    rebuildFloor();
    fitCamera(currentContainer);
    rebuildOverlays();
    document.getElementById('download-link').href = loaded.href;
    document.getElementById('download-link').style.display = '';
    document.getElementById('loading').style.display = 'none';
  }

  function fitMeshes(meshes) {
    meshes = meshes.filter(m => m.getTotalVertices && m.getTotalVertices() > 0);
    if (!meshes.length) return;
    let min = new BABYLON.Vector3(Infinity, Infinity, Infinity);
    let max = new BABYLON.Vector3(-Infinity, -Infinity, -Infinity);
    meshes.forEach(m => {
      m.computeWorldMatrix(true);
      const b = m.getBoundingInfo().boundingBox;
      min = BABYLON.Vector3.Minimize(min, b.minimumWorld);
      max = BABYLON.Vector3.Maximize(max, b.maximumWorld);
    });
    camera.target = BABYLON.Vector3.Center(min, max);
    camera.radius = max.subtract(min).length() * 1.4;
  }
  function fitCamera(container) { fitMeshes(container.meshes); }

  function contentRoot(container) {
    const explicit = scene.getTransformNodeByName('__root__')
      || container.transformNodes.find(n => n.name === '__root__')
      || container.meshes.find(m => m.name === '__root__');
    if (explicit) return explicit;

    const root = new BABYLON.TransformNode('asset_root', scene);
    const topNodes = [
      ...container.transformNodes.filter(n => !n.parent && n !== root),
      ...container.meshes.filter(m => !m.parent),
    ];
    topNodes.forEach(n => { n.parent = root; });
    return root;
  }

  function applyReplacementScale() {
    if (!loadedFromReplacement || !currentRoot || !textureOverride || !info || !info.obb) return;
    const sourceExtents = textureOverride.source_extents_mm || [];
    const sourceMax = Math.max(...sourceExtents.map(v => v * 0.001));
    const targetMax = Math.max(...info.obb.extents);
    if (!Number.isFinite(sourceMax) || sourceMax <= 0 || !Number.isFinite(targetMax) || targetMax <= 0) return;
    currentRoot.scaling.scaleInPlace(targetMax / sourceMax);
  }

  // ── Overlays (built in object frame, parented to __root__) ─────────────────
  // Row-major 4x4 (numpy) -> Babylon Matrix (column-major) is a transpose.
  function mat4(rows) {
    return BABYLON.Matrix.FromArray([
      rows[0][0], rows[1][0], rows[2][0], rows[3][0],
      rows[0][1], rows[1][1], rows[2][1], rows[3][1],
      rows[0][2], rows[1][2], rows[2][2], rows[3][2],
      rows[0][3], rows[1][3], rows[2][3], rows[3][3],
    ]);
  }

  // The most upright stable pose = the one giving the tallest footprint
  // (largest z-extent of the OBB after the resting transform). Returns a Babylon
  // Matrix (object -> table) or null.
  function standingPose() {
    // Prefer the curated 'teaser' pose (same one the thumbnails use) so the
    // standing object matches the gallery; fall back to the tallest stable pose.
    if (info && info.display_pose) return mat4(info.display_pose);
    if (!info || !info.tabletop_poses || !info.tabletop_poses.length || !info.obb) return null;
    const e = info.obb.extents, T = mat4(info.obb.transform);
    const hx = e[0] / 2, hy = e[1] / 2, hz = e[2] / 2;
    const local = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ].map(p => new BABYLON.Vector3(p[0], p[1], p[2]));
    let best = null, bestH = -Infinity;
    info.tabletop_poses.forEach(pose => {
      const P = mat4(pose);
      let mn = Infinity, mx = -Infinity;
      local.forEach(c => {
        const w = BABYLON.Vector3.TransformCoordinates(BABYLON.Vector3.TransformCoordinates(c, T), P);
        mn = Math.min(mn, w.z); mx = Math.max(mx, w.z);
      });
      const h = mx - mn;
      if (h > bestH) { bestH = h; best = P; }
    });
    return best;
  }

  // Set a node's local transform from a Babylon Matrix.
  function setNodeMatrix(node, M) {
    const s = new BABYLON.Vector3(), q = new BABYLON.Quaternion(), t = new BABYLON.Vector3();
    M.decompose(s, q, t);
    node.scaling = s; node.rotationQuaternion = q; node.position = t;
  }

  // Rotate the loaded object into its standing pose by wrapping __root__ (so the
  // container's template stays identity — pose instances aren't double-rotated).
  function applyStandingPose() {
    const S = standingPose();
    if (!S || !currentRoot) return;
    standWrap = new BABYLON.TransformNode('stand', scene);
    setNodeMatrix(standWrap, S);
    currentRoot.parent = standWrap;
  }

  function makeFloorSlab(name, size) {
    const m = BABYLON.MeshBuilder.CreateBox(name, { width: size, height: size, depth: 0.002 }, scene);
    m.position.z = -0.001;
    const mat = new BABYLON.StandardMaterial(name + '_mat', scene);
    mat.diffuseColor = new BABYLON.Color3(0.5, 0.5, 0.55);
    mat.alpha = 0.25;
    mat.backFaceCulling = false;
    m.material = mat;
    addFloorGrid(name + '_grid', size).parent = m;
    return m;
  }

  // Faint reference grid on the floor (xy-plane), ~20 divisions, sitting just
  // above the slab's top face so it reads against the dark surface.
  function addFloorGrid(name, size) {
    const half = size / 2, step = size / 20, lines = [];
    for (let i = -half; i <= half + step * 0.01; i += step) {
      lines.push([new BABYLON.Vector3(i, -half, 0), new BABYLON.Vector3(i, half, 0)]);
      lines.push([new BABYLON.Vector3(-half, i, 0), new BABYLON.Vector3(half, i, 0)]);
    }
    const grid = BABYLON.MeshBuilder.CreateLineSystem(name, { lines }, scene);
    grid.color = new BABYLON.Color3(0.42, 0.45, 0.52);
    grid.alpha = 0.55;
    grid.position.z = 0.0015;   // local: slab top sits +0.001 above slab centre
    grid.isPickable = false;
    return grid;
  }

  // Ground plane under the single standing object.
  function rebuildFloor() {
    if (floorMesh) { floorMesh.dispose(); floorMesh = null; }
    const e = info && info.obb ? info.obb.extents : [0.1, 0.1, 0.1];
    floorMesh = makeFloorSlab('floor', Math.max(...e) * 4);
  }

  function buildOBB() {
    if (!info || !info.obb) return null;
    const e = info.obb.extents, T = mat4(info.obb.transform);
    const hx = e[0] / 2, hy = e[1] / 2, hz = e[2] / 2;
    const c = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ].map(p => BABYLON.Vector3.TransformCoordinates(new BABYLON.Vector3(p[0], p[1], p[2]), T));
    const E = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const lines = E.map(([a, b]) => [c[a], c[b]]);
    const m = BABYLON.MeshBuilder.CreateLineSystem('obb', { lines }, scene);
    m.color = new BABYLON.Color3(0.27, 0.51, 1.0);
    return m;
  }

  // Shortest-arc quaternion rotating unit vector `from` onto unit vector `to`.
  function quatFromTo(from, to) {
    const f = from.normalizeToNew(), t = to.normalizeToNew();
    const d = BABYLON.Vector3.Dot(f, t);
    if (d > 0.999999) return BABYLON.Quaternion.Identity();
    if (d < -0.999999) {
      let axis = BABYLON.Vector3.Cross(BABYLON.Axis.X, f);
      if (axis.lengthSquared() < 1e-6) axis = BABYLON.Vector3.Cross(BABYLON.Axis.Y, f);
      return BABYLON.Quaternion.RotationAxis(axis.normalize(), Math.PI);
    }
    const c = BABYLON.Vector3.Cross(f, t);
    return new BABYLON.Quaternion(c.x, c.y, c.z, 1 + d).normalize();
  }

  // A cone (Babylon cylinders point along +Y) placed at `pos`, aimed along `dir`.
  function arrowhead(name, pos, dir, len, color) {
    const cone = BABYLON.MeshBuilder.CreateCylinder(name,
      { height: len, diameterTop: 0, diameterBottom: len * 0.5, tessellation: 14 }, scene);
    const mat = new BABYLON.StandardMaterial(name + '_m', scene);
    mat.emissiveColor = color; mat.disableLighting = true;
    cone.material = mat;
    cone.rotationQuaternion = quatFromTo(BABYLON.Axis.Y, dir);
    cone.position = pos;
    return cone;
  }

  function buildSymmetry() {
    if (!info || !info.symmetry || !info.symmetry.axes.length) return null;
    const ctr = new BABYLON.Vector3(...info.symmetry.center);
    const L = info.symmetry.scale * 0.6;
    const group = new BABYLON.TransformNode('symmetry', scene);
    info.symmetry.axes.forEach((a, i) => {
      const d = new BABYLON.Vector3(...a.axis).normalize();
      const col = AXIS_COLORS[i % AXIS_COLORS.length];
      const p0 = ctr.subtract(d.scale(L)), p1 = ctr.add(d.scale(L));
      const ln = BABYLON.MeshBuilder.CreateLines('sym' + i, { points: [p0, p1] }, scene);
      ln.color = col;
      ln.parent = group;
      // Double-headed arrow so the axis direction reads at a glance.
      arrowhead('symA' + i, p1, d, L * 0.16, col).parent = group;
      arrowhead('symB' + i, p0, d.scale(-1), L * 0.16, col).parent = group;
    });
    return group;
  }

  // Small RGB world-axis gizmo (X red / Y green / Z blue) for orientation + scale
  // reference. Sized to the object so it doubles as a rough ruler.
  function buildAxes() {
    const L = info && info.obb ? Math.max(...info.obb.extents) * 0.6 : 0.1;
    const group = new BABYLON.TransformNode('axes', scene);
    const defs = [
      [new BABYLON.Vector3(L, 0, 0), new BABYLON.Color3(0.92, 0.34, 0.34)],
      [new BABYLON.Vector3(0, L, 0), new BABYLON.Color3(0.30, 0.85, 0.42)],
      [new BABYLON.Vector3(0, 0, L), new BABYLON.Color3(0.36, 0.62, 0.95)],
    ];
    defs.forEach(([v, col], i) => {
      const ln = BABYLON.MeshBuilder.CreateLines('ax' + i, { points: [BABYLON.Vector3.Zero(), v] }, scene);
      ln.color = col; ln.parent = group;
      arrowhead('axh' + i, v, v.normalizeToNew(), L * 0.16, col).parent = group;
    });
    return group;
  }

  // Tabletop poses: instance the object MESH resting in each stable pose, laid
  // out on a grid over a shared floor at table level (z = 0). Each pose matrix
  // (object -> table) plus a grid offset goes straight onto an instance root.
  function buildTabletop() {
    if (!info || !info.tabletop_poses || !info.tabletop_poses.length || !currentContainer || !currentRoot) return null;
    const e = info.obb ? info.obb.extents : [0.1, 0.1, 0.1];
    const n = info.tabletop_poses.length;
    const cols = Math.ceil(Math.sqrt(n));
    const spacing = Math.max(...e) * 1.8;
    const group = new BABYLON.TransformNode('tabletop', scene);

    info.tabletop_poses.forEach((pose, k) => {
      const dx = ((k % cols) - (cols - 1) / 2) * spacing;
      const dy = (Math.floor(k / cols) - (cols - 1) / 2) * spacing;
      const M = mat4(pose).multiply(BABYLON.Matrix.Translation(dx, dy, 0));
      // Deep-clone the loaded object subtree (clone copies the descendant meshes
      // and keeps intermediate node transforms) and drop it at the resting pose.
      // currentRoot's own transform is identity — the standing-pose wrapper lives
      // on its parent and is intentionally not inherited here.
      const clone = currentRoot.clone(`ttpose${k}`, null, false);
      clone.setEnabled(true);
      setNodeMatrix(clone, M);
      clone.parent = group;
    });

    const half = (cols * spacing) / 2 + Math.max(...e);
    makeFloorSlab('tt_floor', half * 2).parent = group;
    return group;
  }

  function rebuildOverlays() {
    disposeOverlays();
    const build = (cond, fn, tag) => {
      try { return cond ? fn() : null; } catch (e) { console.error('overlay ' + tag, e); return null; }
    };
    overlays.obb = build(shown.obb, buildOBB, 'obb');
    overlays.symmetry = build(shown.symmetry, buildSymmetry, 'symmetry');
    overlays.tabletop = build(shown.tabletop, buildTabletop, 'tabletop');
    overlays.axes = build(shown.axes, buildAxes, 'axes');
    // OBB / symmetry / axes live in object frame -> parent to glTF root so they
    // align with the (standing) object. The pose grid is world-frame, at scene root.
    for (const k of ['obb', 'symmetry', 'axes']) {
      if (overlays[k] && currentRoot) overlays[k].parent = currentRoot;
    }
    // Hide the single object only when the pose grid actually built — a failed
    // tabletop build must never leave a blank screen.
    const single = !(shown.tabletop && overlays.tabletop);
    if (standWrap) standWrap.setEnabled(single);
    else if (currentRoot) currentRoot.setEnabled(single);
    if (floorMesh) floorMesh.setEnabled(single);
    // Frame the pose grid (it's bigger and elsewhere than the single object).
    if (!single && overlays.tabletop) fitMeshes(overlays.tabletop.getChildMeshes(false));
  }

  // Show the textured display mesh while symmetry/tabletop overlays are on,
  // otherwise the user-selected pipeline stage. Reload only when the chosen mesh
  // actually changes; loadStage() rebuilds the overlays on the way out.
  async function syncMesh() {
    if (compareOn) { rebuildOverlays(); return; }
    const want = (shown.symmetry || shown.tabletop) ? texturedStage : selectedStage;
    if (loadedStageId === want.id) { rebuildOverlays(); return; }
    await loadStage(want);
  }

  async function selectStage(s) {
    selectedStage = s;
    await syncMesh();
  }

  async function toggleOverlay(kind, on) {
    shown[kind] = on;
    await syncMesh();
  }

  // ── Real-world dimensions (header) ─────────────────────────────────────────
  function showDimensions() {
    if (!info || !info.obb || !info.obb.extents) return;
    const cm = info.obb.extents.map(v => v * 100);
    const el = document.getElementById('obj-dims');
    if (!el) return;
    el.innerHTML = `<b>${cm[0].toFixed(1)} × ${cm[1].toFixed(1)} × ${cm[2].toFixed(1)}</b> cm`;
    el.title = 'Oriented bounding-box size (W × D × H)';
    el.style.display = '';
  }
  showDimensions();

  // ── View presets / turntable / snapshot ────────────────────────────────────
  // The scene is z-up (floor at z = 0, standing pose maximises z), so orbit the
  // camera about +z — keeps the horizon level and makes the presets meaningful.
  camera.upVector = new BABYLON.Vector3(0, 0, 1);
  const VIEWS = {
    iso:   [-Math.PI / 4, Math.PI / 3],
    front: [-Math.PI / 2, Math.PI / 2],
    side:  [0,            Math.PI / 2],
    top:   [-Math.PI / 2, 0.01],
  };
  function setView(which) {
    const v = VIEWS[which]; if (!v) return;
    camera.alpha = v[0]; camera.beta = v[1];
    if (currentContainer) fitCamera(currentContainer);
  }

  let turntable = false;
  function snapshot() {
    BABYLON.Tools.CreateScreenshotUsingRenderTarget(
      engine, camera, { precision: 2 }, undefined, 'image/png', 4, true, `${objectId}.png`);
  }

  // ── Compare: load raw + simplified together, wipe between them with a clip
  // plane in world x driven by the slider. Each mesh sets the clip plane only for
  // its own draw (before/after observers), so the two halves meet at the cut.
  async function loadCmp(stage) {
    try { return await BABYLON.SceneLoader.LoadAssetContainerAsync(assetBase, stage.file, scene); }
    catch (e) {
      if (stage.fallback) return await BABYLON.SceneLoader.LoadAssetContainerAsync(assetBase, stage.fallback, scene);
      throw e;
    }
  }
  async function enterCompare() {
    document.getElementById('loading').style.display = '';
    document.getElementById('error').style.display = 'none';
    const S = standingPose();
    const specs = [{ stage: STAGES[0], plane: 'a' }, { stage: DEFAULT_STAGE, plane: 'b' }];
    // Load BOTH stages first; only hide the live object once they're in hand, so
    // a missing stage can never strand us on a blank screen.
    const loaded = [];
    try {
      for (const sp of specs) loaded.push({ sp, c: await loadCmp(sp.stage) });
    } catch (e) {
      console.error('compare load', e);
      loaded.forEach(l => l.c.dispose());
      document.getElementById('loading').style.display = 'none';
      showError('Compare needs the raw and simplified stage meshes, which are not uploaded for this object yet.');
      return false;
    }

    compareOn = true;
    disposeOverlays();
    document.getElementById('compare-bar').style.display = 'flex';
    if (standWrap) standWrap.setEnabled(false); else if (currentRoot) currentRoot.setEnabled(false);

    cmp.containers = [];
    let mn = Infinity, mx = -Infinity;
    for (const { sp, c } of loaded) {
      c.addAllToScene();
      const root = c.meshes.find(m => m.name === '__root__')
        || c.transformNodes.find(n => n.name === '__root__');
      if (S && root) { const w = new BABYLON.TransformNode('cmpstand', scene); setNodeMatrix(w, S); root.parent = w; }
      c.meshes.filter(m => m.getTotalVertices() > 0).forEach(m => {
        m.computeWorldMatrix(true);
        const b = m.getBoundingInfo().boundingBox;
        mn = Math.min(mn, b.minimumWorld.x); mx = Math.max(mx, b.maximumWorld.x);
        m.onBeforeRenderObservable.add(() => {
          scene.clipPlane = sp.plane === 'a'
            ? new BABYLON.Plane(1, 0, 0, -cut) : new BABYLON.Plane(-1, 0, 0, cut);
        });
        m.onAfterRenderObservable.add(() => { scene.clipPlane = null; });
      });
      cmp.containers.push(c);
    }
    cmp.minX = mn; cmp.maxX = mx; cut = (mn + mx) / 2;
    document.getElementById('compare-range').value = 0.5;
    fitMeshes(cmp.containers[1].meshes);
    document.getElementById('loading').style.display = 'none';
    return true;
  }
  function exitCompare() {
    compareOn = false;
    document.getElementById('compare-bar').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    scene.clipPlane = null;
    cmp.containers.forEach(c => c.dispose());
    cmp.containers = [];
    scene.transformNodes.filter(n => n.name === 'cmpstand').forEach(n => n.dispose());
    if (standWrap) standWrap.setEnabled(true); else if (currentRoot) currentRoot.setEnabled(true);
  }
  document.getElementById('compare-range').addEventListener('input', e => {
    cut = cmp.minX + (cmp.maxX - cmp.minX) * parseFloat(e.target.value);
  });

  const actions = {
    setView, snapshot,
    setTurntable: (v) => { turntable = v; },
    setCompare: async (on) => { if (on) return await enterCompare(); exitCompare(); return true; },
    hasCompare: true,
  };

  // Deep-link state for sharing / testing:
  //   ?overlay=obb,symmetry,tabletop,axes   ?view=front|side|top   ?compare=1
  async function applyUrlState() {
    const ov = (params.get('overlay') || '').split(',').map(s => s.trim()).filter(Boolean);
    ov.forEach(k => { if (k in shown) shown[k] = true; });
    if (ov.length) await syncMesh();
    const v = params.get('view'); if (v) setView(v);
    if (params.get('compare') === '1') await enterCompare();
  }

  // ── Controls ───────────────────────────────────────────────────────────────
  buildControls(stageOptions, info, selectStage, toggleOverlay, actions, selectedStage.id);

  try {
    await loadStage(selectedStage);
    setView('iso');
    await applyUrlState();
  } catch (e) {
    console.error(e);
    showError(`Failed to load 3D model: ${e.message || e}`);
  }

  engine.runRenderLoop(() => {
    if (turntable && !compareOn) camera.alpha += 0.004;
    scene.render();
  });
  window.addEventListener('resize', () => engine.resize());
})();

function buildControls(stages, info, onStage, onOverlay, actions, activeStageId) {
  const panel = document.getElementById('controls');
  if (!panel) return;
  actions = actions || {};

  const stageWrap = document.createElement('div');
  stageWrap.className = 'ctl-group';
  stageWrap.innerHTML = '<div class="ctl-title">Stage</div>';
  const stageInputs = [];
  stages.forEach((s, i) => {
    const id = 'stage-' + s.id;
    const lab = document.createElement('label');
    lab.className = 'ctl-radio';
    const checked = activeStageId ? s.id === activeStageId : i === stages.length - 1;
    lab.innerHTML = `<input type="radio" name="stage" id="${id}" ${checked ? 'checked' : ''}> ${s.label}`;
    const input = lab.querySelector('input');
    input.addEventListener('change', () => onStage(s));
    stageInputs.push(input);
    stageWrap.appendChild(lab);
  });
  panel.appendChild(stageWrap);

  // Compare (raw ↔ simplified). Disables the stage radios while active.
  if (actions.hasCompare) {
    const cmpWrap = document.createElement('div');
    cmpWrap.className = 'ctl-group';
    const lab = document.createElement('label');
    lab.className = 'ctl-check';
    lab.innerHTML = '<input type="checkbox"> Compare raw ↔ simplified';
    lab.querySelector('input').addEventListener('change', async (e) => {
      const on = e.target.checked;
      stageInputs.forEach(inp => { inp.disabled = on; });
      const ok = actions.setCompare ? await actions.setCompare(on) : true;
      if (on && ok === false) {  // stage missing — revert the toggle
        e.target.checked = false;
        stageInputs.forEach(inp => { inp.disabled = false; });
      }
    });
    cmpWrap.appendChild(lab);
    panel.appendChild(cmpWrap);
  }

  // View presets + turntable + snapshot.
  if (actions.setView || actions.snapshot) {
    const vWrap = document.createElement('div');
    vWrap.className = 'ctl-group';
    vWrap.innerHTML = '<div class="ctl-title">View</div>';
    const row = document.createElement('div');
    row.className = 'ctl-btnrow';
    [['Reset', 'iso'], ['Front', 'front'], ['Side', 'side'], ['Top', 'top']].forEach(([label, key]) => {
      const b = document.createElement('button');
      b.className = 'ctl-btn'; b.type = 'button'; b.textContent = label;
      b.addEventListener('click', () => actions.setView && actions.setView(key));
      row.appendChild(b);
    });
    vWrap.appendChild(row);
    const tlab = document.createElement('label');
    tlab.className = 'ctl-check'; tlab.style.marginTop = '4px';
    tlab.innerHTML = '<input type="checkbox"> Turntable';
    tlab.querySelector('input').addEventListener('change', (e) => actions.setTurntable && actions.setTurntable(e.target.checked));
    vWrap.appendChild(tlab);
    const snap = document.createElement('button');
    snap.className = 'ctl-btn'; snap.type = 'button'; snap.textContent = 'Snapshot PNG';
    snap.style.marginTop = '6px';
    snap.addEventListener('click', () => actions.snapshot && actions.snapshot());
    vWrap.appendChild(snap);
    panel.appendChild(vWrap);
  }

  const overlayDefs = [
    { kind: 'obb', label: 'Bounding box', has: info && info.obb },
    { kind: 'symmetry', label: 'Symmetry axis', has: info && info.symmetry && info.symmetry.axes.length },
    { kind: 'tabletop', label: 'Tabletop poses', has: info && info.tabletop_poses && info.tabletop_poses.length },
    { kind: 'axes', label: 'World axes', has: true },
  ];
  const ovWrap = document.createElement('div');
  ovWrap.className = 'ctl-group';
  ovWrap.innerHTML = '<div class="ctl-title">Overlays</div>';
  overlayDefs.forEach(d => {
    const lab = document.createElement('label');
    lab.className = 'ctl-check' + (d.has ? '' : ' disabled');
    const tag = d.kind === 'symmetry' && info && info.symmetry ? ` (${info.symmetry.type})` : '';
    lab.innerHTML = `<input type="checkbox" ${d.has ? '' : 'disabled'}> ${d.label}${tag}`;
    if (d.has) lab.querySelector('input').addEventListener('change', e => onOverlay(d.kind, e.target.checked));
    ovWrap.appendChild(lab);
  });
  panel.appendChild(ovWrap);
}

function showError(msg) {
  const loading = document.getElementById('loading');
  if (loading) loading.style.display = 'none';
  document.getElementById('error-msg').textContent = msg;
  document.getElementById('error').style.display = 'block';
}

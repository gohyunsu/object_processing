const HF_BASE = 'https://huggingface.co/datasets/willi19/object_processing/resolve/main/';
const DATA_VERSION = '20260702-willi19-9aaa4ce';
const AXIS_COLORS = [
  [1.0, 0.82, 0.10],
  [0.20, 0.85, 0.40],
  [0.36, 0.62, 0.95],
];

const state = {
  rows: [],
  textureOverrides: {},
  useTextureOverrides: true,
  active: new Map(),
  observer: null,
};

const els = {
  rows: document.getElementById('rows'),
  empty: document.getElementById('empty'),
  search: document.getElementById('search'),
  symmetryFilter: document.getElementById('symmetry-filter'),
  sort: document.getElementById('sort'),
  textureToggle: document.getElementById('texture-toggle'),
  statVisible: document.getElementById('stat-visible'),
  statObjects: document.getElementById('stat-objects'),
  statTextures: document.getElementById('stat-textures'),
};

function escapeHtml(value) {
  return String(value == null ? '' : value).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
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

function versionedDataUrl(path) {
  const url = new URL(path, window.location.href);
  if (url.origin === window.location.origin && !url.searchParams.has('v')) {
    url.searchParams.set('v', DATA_VERSION);
  }
  return url.toString();
}

async function init() {
  if (!window.BABYLON) {
    els.empty.textContent = 'Babylon.js did not load. Serve this page with network access to use the 3D audit view.';
    els.empty.style.display = 'block';
    return;
  }

  const [catalog, textureManifest] = await Promise.all([
    fetchJson('catalog.json', { objects: [] }),
    fetchJson('texture_overrides/manifest.json', { objects: {} }),
  ]);
  state.textureOverrides = textureManifest.objects || {};

  const rows = await Promise.all((catalog.objects || []).map(async (obj) => {
    const info = await fetchJson(`objects/${obj.id}/info.json`, null);
    return enrichRow(obj, info);
  }));
  state.rows = rows;

  buildSymmetryFilter(rows);
  bindControls();
  applyInitialParams();
  renderRows();
}

function enrichRow(obj, info) {
  const symmetry = info && info.symmetry ? info.symmetry : { type: 'missing', axes: [] };
  const poses = info && Array.isArray(info.tabletop_poses) ? info.tabletop_poses : [];
  const poseCount = Number.isFinite(info && info.n_tabletop_poses)
    ? info.n_tabletop_poses
    : poses.length;
  const axes = Array.isArray(symmetry.axes) ? symmetry.axes : [];
  const axesLabel = axes.length
    ? axes.map((axis) => `${axis.fold}-fold`).join(', ')
    : 'no rotational axis';
  const textureOverride = state.textureOverrides[obj.id] || null;
  return {
    obj,
    info,
    id: obj.id,
    label: obj.label || obj.id,
    symmetry,
    symmetryType: symmetry.type || 'missing',
    poseCount,
    axesLabel,
    textureOverride,
    hasIssue: !info || !poses.length,
  };
}

function buildSymmetryFilter(rows) {
  const types = [...new Set(rows.map((row) => row.symmetryType))].sort((a, b) => {
    if (a === 'none') return -1;
    if (b === 'none') return 1;
    return a.localeCompare(b);
  });
  for (const type of types) {
    const opt = document.createElement('option');
    opt.value = type;
    opt.textContent = type;
    els.symmetryFilter.appendChild(opt);
  }
}

function bindControls() {
  els.search.addEventListener('input', renderRows);
  els.symmetryFilter.addEventListener('change', renderRows);
  els.sort.addEventListener('change', renderRows);
  els.textureToggle.addEventListener('change', () => {
    state.useTextureOverrides = els.textureToggle.checked;
    disposeAll();
    refreshVisibleRows();
  });
}

function applyInitialParams() {
  const params = new URLSearchParams(window.location.search);
  if (params.has('q')) els.search.value = params.get('q') || '';
  if (params.has('sym')) els.symmetryFilter.value = params.get('sym') || '';
  if (params.has('sort')) els.sort.value = params.get('sort') || els.sort.value;
  if (params.get('textures') === '0') {
    els.textureToggle.checked = false;
    state.useTextureOverrides = false;
  }
}

function filteredRows() {
  const q = els.search.value.trim().toLowerCase();
  const sym = els.symmetryFilter.value;
  const rows = state.rows.filter((row) => {
    if (sym && row.symmetryType !== sym) return false;
    if (!q) return true;
    return row.id.toLowerCase().includes(q)
      || row.label.toLowerCase().includes(q)
      || row.symmetryType.toLowerCase().includes(q)
      || row.axesLabel.toLowerCase().includes(q);
  });

  const sort = els.sort.value;
  rows.sort((a, b) => {
    if (sort === 'poses-desc') return b.poseCount - a.poseCount || a.label.localeCompare(b.label);
    if (sort === 'poses-asc') return a.poseCount - b.poseCount || a.label.localeCompare(b.label);
    if (sort === 'symmetry') return a.symmetryType.localeCompare(b.symmetryType) || a.label.localeCompare(b.label);
    return a.label.localeCompare(b.label);
  });
  return rows;
}

function renderRows() {
  disposeAll();
  if (state.observer) state.observer.disconnect();

  const visible = filteredRows();
  els.rows.innerHTML = '';
  els.empty.style.display = visible.length ? 'none' : 'block';
  els.statVisible.textContent = String(visible.length);
  els.statObjects.textContent = String(state.rows.length);
  els.statTextures.textContent = String(Object.keys(state.textureOverrides).length);

  const frag = document.createDocumentFragment();
  visible.forEach((row) => frag.appendChild(renderRow(row)));
  els.rows.appendChild(frag);

  state.observer = new IntersectionObserver(handleIntersect, {
    root: null,
    rootMargin: '560px 0px 560px 0px',
    threshold: 0.01,
  });
  els.rows.querySelectorAll('.object-row').forEach((rowEl) => state.observer.observe(rowEl));
  refreshVisibleRows();
}

function renderRow(row) {
  const article = document.createElement('article');
  article.className = 'object-row';
  article.dataset.id = row.id;

  const symClass = row.symmetryType === 'none' ? 'none' : 'sym';
  const textureChip = row.textureOverride
    ? '<span class="chip texture">texture override</span>'
    : '';
  const issueChip = row.hasIssue
    ? '<span class="chip issue">missing data</span>'
    : '';

  article.innerHTML = `
    <aside class="object-meta">
      <img class="thumb" src="objects/${encodeURIComponent(row.id)}/thumb.png?v=24" alt="${escapeHtml(row.label)}" loading="lazy">
      <div class="obj-title">
        <h2>${escapeHtml(row.label)}</h2>
        <code>${escapeHtml(row.id)}</code>
      </div>
      <div class="chips">
        <span class="chip ${symClass}">${escapeHtml(row.symmetryType)}</span>
        <span class="chip pose">${row.poseCount} poses</span>
        ${textureChip}
        ${issueChip}
      </div>
      <div class="meta-links">
        <a href="viewer.html?id=${encodeURIComponent(row.id)}&overlay=symmetry,tabletop" target="_blank" rel="noopener">viewer</a>
        <a href="objects/${encodeURIComponent(row.id)}/info.json" target="_blank" rel="noopener">info.json</a>
      </div>
    </aside>

    <section class="viewer-cell symmetry" data-kind="symmetry">
      <div class="viewer-title">Symmetry <span>${escapeHtml(row.axesLabel)}</span></div>
      <div class="canvas-wrap">
        <canvas class="audit-canvas"></canvas>
        <div class="scene-state">Waiting for row visibility</div>
      </div>
    </section>

    <section class="viewer-cell tabletop" data-kind="tabletop">
      <div class="viewer-title">Tabletop Poses <span>${row.poseCount} de-duplicated</span></div>
      <div class="canvas-wrap">
        <canvas class="audit-canvas"></canvas>
        <div class="scene-state">Waiting for row visibility</div>
      </div>
    </section>
  `;
  return article;
}

function handleIntersect(entries) {
  for (const entry of entries) {
    const rowEl = entry.target;
    if (entry.isIntersecting) startRow(rowEl);
    else disposeRow(rowEl.dataset.id);
  }
}

function refreshVisibleRows() {
  const viewportH = window.innerHeight || document.documentElement.clientHeight;
  els.rows.querySelectorAll('.object-row').forEach((rowEl) => {
    const box = rowEl.getBoundingClientRect();
    if (box.bottom >= -560 && box.top <= viewportH + 560) startRow(rowEl);
  });
}

function startRow(rowEl) {
  const id = rowEl.dataset.id;
  if (state.active.has(id)) return;
  const row = state.rows.find((r) => r.id === id);
  if (!row) return;

  const symmetryCell = rowEl.querySelector('[data-kind="symmetry"]');
  const tabletopCell = rowEl.querySelector('[data-kind="tabletop"]');
  const scenes = {
    symmetry: new AuditScene(row, 'symmetry', symmetryCell),
    tabletop: new AuditScene(row, 'tabletop', tabletopCell),
  };
  state.active.set(id, scenes);
  scenes.symmetry.load();
  scenes.tabletop.load();
}

function disposeRow(id) {
  const scenes = state.active.get(id);
  if (!scenes) return;
  Object.values(scenes).forEach((scene) => scene.dispose());
  state.active.delete(id);
}

function disposeAll() {
  for (const id of [...state.active.keys()]) disposeRow(id);
}

function splitUrl(url) {
  const i = url.lastIndexOf('/');
  return [url.slice(0, i + 1), url.slice(i + 1)];
}

function mat4(rows) {
  return BABYLON.Matrix.FromArray([
    rows[0][0], rows[1][0], rows[2][0], rows[3][0],
    rows[0][1], rows[1][1], rows[2][1], rows[3][1],
    rows[0][2], rows[1][2], rows[2][2], rows[3][2],
    rows[0][3], rows[1][3], rows[2][3], rows[3][3],
  ]);
}

function setNodeMatrix(node, matrix) {
  const scale = new BABYLON.Vector3();
  const rotation = new BABYLON.Quaternion();
  const position = new BABYLON.Vector3();
  matrix.decompose(scale, rotation, position);
  node.scaling = scale;
  node.rotationQuaternion = rotation;
  node.position = position;
}

function quatFromTo(from, to) {
  const f = from.normalizeToNew();
  const t = to.normalizeToNew();
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

class AuditScene {
  constructor(row, kind, cell) {
    this.row = row;
    this.kind = kind;
    this.cell = cell;
    this.canvas = cell.querySelector('canvas');
    this.status = cell.querySelector('.scene-state');
    this.engine = null;
    this.scene = null;
    this.camera = null;
    this.container = null;
    this.root = null;
    this.resizeObserver = null;
    this.eventCleanups = [];
    this.disposed = false;
    this.loadedFromTextureOverride = false;
  }

  async load() {
    this.setStatus('Loading mesh...');
    try {
      this.engine = new BABYLON.Engine(this.canvas, true, {
        preserveDrawingBuffer: false,
        stencil: true,
        antialias: true,
      });
      this.scene = new BABYLON.Scene(this.engine);
      this.scene.useRightHandedSystem = true;
      this.scene.clearColor = new BABYLON.Color4(0.055, 0.078, 0.106, 1);
      this.bindCanvasInput();
      this.setupCameraAndLights();

      this.container = await this.loadContainer();
      if (this.disposed) return;
      this.container.addAllToScene();
      this.root = this.contentRoot();
      this.applyTextureOverrideScale();

      if (this.kind === 'symmetry') this.buildSymmetryView();
      else this.buildTabletopView();

      this.fitScene();
      this.hideStatus();
      this.engine.runRenderLoop(() => {
        if (!this.disposed && this.scene) this.scene.render();
      });
      this.resizeObserver = new ResizeObserver(() => this.engine && this.engine.resize());
      this.resizeObserver.observe(this.canvas.parentElement);
    } catch (err) {
      console.error(this.row.id, this.kind, err);
      this.setStatus(`Failed to load ${this.kind} view.`, true);
      this.dispose(false);
    }
  }

  setupCameraAndLights() {
    this.camera = new BABYLON.ArcRotateCamera(
      `${this.kind}_camera`,
      -Math.PI / 4,
      Math.PI / 3,
      0.45,
      BABYLON.Vector3.Zero(),
      this.scene,
    );
    this.camera.upVector = new BABYLON.Vector3(0, 0, 1);
    this.camera.attachControl(this.canvas, true);
    this.camera.minZ = 0.001;
    this.camera.lowerRadiusLimit = 0.02;
    this.camera.upperRadiusLimit = 20;
    this.camera.wheelPrecision = 65;
    this.camera.panningSensibility = 2600;
    this.camera.angularSensibilityX = 2600;
    this.camera.angularSensibilityY = 2600;

    const hemi = new BABYLON.HemisphericLight(`${this.kind}_hemi`, new BABYLON.Vector3(0, 0, 1), this.scene);
    hemi.intensity = 0.85;
    const dir = new BABYLON.DirectionalLight(`${this.kind}_dir`, new BABYLON.Vector3(-1, -1.4, -0.9), this.scene);
    dir.intensity = 0.55;
  }

  bindCanvasInput() {
    this.canvas.style.touchAction = 'none';
    this.canvas.style.overscrollBehavior = 'contain';
    this.canvas.style.userSelect = 'none';

    const preventDefault = (event) => event.preventDefault();
    const preventDragScroll = (event) => {
      if (event.pointerType === 'touch' || event.pointerType === 'pen' || event.buttons) {
        event.preventDefault();
      }
    };
    const capturePointer = (event) => {
      preventDragScroll(event);
      if (this.canvas.setPointerCapture && event.pointerId != null) {
        try { this.canvas.setPointerCapture(event.pointerId); } catch (err) { /* pointer already released */ }
      }
    };

    const listeners = [
      ['wheel', preventDefault, { passive: false }],
      ['touchstart', preventDefault, { passive: false }],
      ['touchmove', preventDefault, { passive: false }],
      ['pointerdown', capturePointer, { passive: false }],
      ['pointermove', preventDragScroll, { passive: false }],
      ['contextmenu', preventDefault, false],
    ];

    listeners.forEach(([type, handler, options]) => {
      this.canvas.addEventListener(type, handler, options);
      this.eventCleanups.push(() => this.canvas.removeEventListener(type, handler, options));
    });
  }

  async loadContainer() {
    const urls = [];
    if (state.useTextureOverrides && this.row.textureOverride) {
      urls.push({ url: this.row.textureOverride.mesh, texture: true });
    }
    urls.push({ url: `${HF_BASE}objects/${encodeURIComponent(this.row.id)}/mesh.glb`, texture: false });

    let lastError = null;
    for (const spec of urls) {
      try {
        const [rootUrl, file] = splitUrl(spec.url);
        const container = await BABYLON.SceneLoader.LoadAssetContainerAsync(rootUrl, file, this.scene);
        this.loadedFromTextureOverride = spec.texture;
        const label = spec.texture ? 'texture override' : 'Hugging Face mesh';
        this.cell.querySelector('.viewer-title span').title = label;
        return container;
      } catch (err) {
        lastError = err;
        console.warn(`mesh load failed for ${this.row.id}: ${spec.url}`, err);
      }
    }
    throw lastError || new Error('No mesh URL loaded');
  }

  contentRoot() {
    const explicit = this.container.transformNodes.find((node) => node.name === '__root__')
      || this.scene.getTransformNodeByName('__root__');
    if (explicit) return explicit;

    const root = new BABYLON.TransformNode(`${this.kind}_asset_root`, this.scene);
    const topNodes = [
      ...this.container.transformNodes.filter((node) => !node.parent && node !== root),
      ...this.container.meshes.filter((mesh) => !mesh.parent),
    ];
    topNodes.forEach((node) => { node.parent = root; });
    return root;
  }

  applyTextureOverrideScale() {
    if (!this.loadedFromTextureOverride || !this.root || !this.row.textureOverride || !this.row.info || !this.row.info.obb) return;
    const sourceExtents = this.row.textureOverride.source_extents_mm || [];
    const sourceMax = Math.max(...sourceExtents.map((v) => v * 0.001));
    const targetMax = Math.max(...this.row.info.obb.extents);
    if (!Number.isFinite(sourceMax) || sourceMax <= 0 || !Number.isFinite(targetMax) || targetMax <= 0) return;
    const s = targetMax / sourceMax;
    this.root.scaling.scaleInPlace(s);
  }

  buildSymmetryView() {
    const standing = this.standingPose();
    if (standing && this.root) {
      const stand = new BABYLON.TransformNode(`${this.kind}_stand`, this.scene);
      setNodeMatrix(stand, standing);
      this.root.parent = stand;
    }

    const obb = this.buildOBB();
    if (obb && this.root) obb.parent = this.root;
    const sym = this.buildSymmetryAxes();
    if (sym && this.root) sym.parent = this.root;
    this.makeFloorSlab('sym_floor', this.floorSize());
  }

  buildTabletopView() {
    const poses = this.row.info && Array.isArray(this.row.info.tabletop_poses)
      ? this.row.info.tabletop_poses
      : [];
    if (!poses.length || !this.root) {
      this.setStatus('No tabletop poses found.', true);
      return;
    }

    this.root.setEnabled(false);
    const extents = this.row.info && this.row.info.obb ? this.row.info.obb.extents : [0.1, 0.1, 0.1];
    const cols = Math.ceil(Math.sqrt(poses.length));
    const spacing = Math.max(...extents) * 1.9;
    const group = new BABYLON.TransformNode('tabletop_group', this.scene);

    poses.forEach((pose, index) => {
      const dx = ((index % cols) - (cols - 1) / 2) * spacing;
      const dy = (Math.floor(index / cols) - (cols - 1) / 2) * spacing;
      const clone = this.root.clone(`pose_${index}`, null, false);
      clone.setEnabled(true);
      setNodeMatrix(clone, mat4(pose).multiply(BABYLON.Matrix.Translation(dx, dy, 0)));
      clone.parent = group;
    });

    const half = (cols * spacing) / 2 + Math.max(...extents);
    this.makeFloorSlab('tabletop_floor', Math.max(half * 2, spacing * 2)).parent = group;
  }

  standingPose() {
    const info = this.row.info;
    if (!info) return null;
    if (info.display_pose) return mat4(info.display_pose);
    if (!info.tabletop_poses || !info.tabletop_poses.length || !info.obb) return null;

    const extents = info.obb.extents;
    const transform = mat4(info.obb.transform);
    const hx = extents[0] / 2;
    const hy = extents[1] / 2;
    const hz = extents[2] / 2;
    const corners = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ].map((p) => new BABYLON.Vector3(p[0], p[1], p[2]));

    let best = null;
    let bestHeight = -Infinity;
    info.tabletop_poses.forEach((pose) => {
      const poseM = mat4(pose);
      let minZ = Infinity;
      let maxZ = -Infinity;
      corners.forEach((corner) => {
        const world = BABYLON.Vector3.TransformCoordinates(
          BABYLON.Vector3.TransformCoordinates(corner, transform),
          poseM,
        );
        minZ = Math.min(minZ, world.z);
        maxZ = Math.max(maxZ, world.z);
      });
      const height = maxZ - minZ;
      if (height > bestHeight) {
        bestHeight = height;
        best = poseM;
      }
    });
    return best;
  }

  buildOBB() {
    const info = this.row.info;
    if (!info || !info.obb) return null;
    const extents = info.obb.extents;
    const transform = mat4(info.obb.transform);
    const hx = extents[0] / 2;
    const hy = extents[1] / 2;
    const hz = extents[2] / 2;
    const corners = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ].map((p) => BABYLON.Vector3.TransformCoordinates(new BABYLON.Vector3(p[0], p[1], p[2]), transform));
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const lines = edges.map(([a, b]) => [corners[a], corners[b]]);
    const mesh = BABYLON.MeshBuilder.CreateLineSystem('obb', { lines }, this.scene);
    mesh.color = new BABYLON.Color3(0.27, 0.51, 1.0);
    mesh.isPickable = false;
    return mesh;
  }

  buildSymmetryAxes() {
    const symmetry = this.row.symmetry;
    if (!symmetry || !Array.isArray(symmetry.axes) || !symmetry.axes.length) return null;
    const center = new BABYLON.Vector3(...symmetry.center);
    const length = (symmetry.scale || this.floorSize()) * 0.58;
    const group = new BABYLON.TransformNode('symmetry_axes', this.scene);
    symmetry.axes.forEach((axisInfo, index) => {
      const dir = new BABYLON.Vector3(...axisInfo.axis).normalize();
      const color = new BABYLON.Color3(...AXIS_COLORS[index % AXIS_COLORS.length]);
      const p0 = center.subtract(dir.scale(length));
      const p1 = center.add(dir.scale(length));
      const line = BABYLON.MeshBuilder.CreateLines(`sym_axis_${index}`, { points: [p0, p1] }, this.scene);
      line.color = color;
      line.parent = group;
      this.arrowhead(`sym_head_a_${index}`, p1, dir, length * 0.14, color).parent = group;
      this.arrowhead(`sym_head_b_${index}`, p0, dir.scale(-1), length * 0.14, color).parent = group;
    });
    return group;
  }

  arrowhead(name, position, direction, length, color) {
    const cone = BABYLON.MeshBuilder.CreateCylinder(name, {
      height: length,
      diameterTop: 0,
      diameterBottom: length * 0.52,
      tessellation: 14,
    }, this.scene);
    const material = new BABYLON.StandardMaterial(`${name}_mat`, this.scene);
    material.emissiveColor = color;
    material.diffuseColor = color;
    material.disableLighting = true;
    cone.material = material;
    cone.rotationQuaternion = quatFromTo(BABYLON.Axis.Y, direction);
    cone.position = position;
    cone.isPickable = false;
    return cone;
  }

  floorSize() {
    const extents = this.row.info && this.row.info.obb ? this.row.info.obb.extents : [0.1, 0.1, 0.1];
    return Math.max(...extents) * 3.8;
  }

  makeFloorSlab(name, size) {
    const mesh = BABYLON.MeshBuilder.CreateBox(name, {
      width: size,
      height: size,
      depth: 0.002,
    }, this.scene);
    mesh.position.z = -0.001;
    const material = new BABYLON.StandardMaterial(`${name}_mat`, this.scene);
    material.diffuseColor = new BABYLON.Color3(0.44, 0.48, 0.54);
    material.alpha = 0.28;
    material.backFaceCulling = false;
    mesh.material = material;
    mesh.isPickable = false;

    const half = size / 2;
    const step = size / 12;
    const lines = [];
    for (let v = -half; v <= half + step * 0.01; v += step) {
      lines.push([new BABYLON.Vector3(v, -half, 0.0014), new BABYLON.Vector3(v, half, 0.0014)]);
      lines.push([new BABYLON.Vector3(-half, v, 0.0014), new BABYLON.Vector3(half, v, 0.0014)]);
    }
    const grid = BABYLON.MeshBuilder.CreateLineSystem(`${name}_grid`, { lines }, this.scene);
    grid.color = new BABYLON.Color3(0.54, 0.58, 0.65);
    grid.alpha = 0.5;
    grid.parent = mesh;
    grid.isPickable = false;
    return mesh;
  }

  fitScene() {
    const meshes = this.scene.meshes.filter((mesh) => mesh.getTotalVertices && mesh.getTotalVertices() > 0 && mesh.isEnabled());
    if (!meshes.length) return;
    let min = new BABYLON.Vector3(Infinity, Infinity, Infinity);
    let max = new BABYLON.Vector3(-Infinity, -Infinity, -Infinity);
    meshes.forEach((mesh) => {
      mesh.computeWorldMatrix(true);
      const box = mesh.getBoundingInfo().boundingBox;
      min = BABYLON.Vector3.Minimize(min, box.minimumWorld);
      max = BABYLON.Vector3.Maximize(max, box.maximumWorld);
    });
    const center = BABYLON.Vector3.Center(min, max);
    const radius = Math.max(max.subtract(min).length() * 1.25, 0.08);
    this.camera.target = center;
    this.camera.radius = radius;
    this.camera.alpha = this.kind === 'tabletop' ? -Math.PI / 4 : -Math.PI / 3.5;
    this.camera.beta = Math.PI / 3;
  }

  setStatus(message, isError = false) {
    this.status.textContent = message;
    this.status.classList.toggle('error', isError);
    this.status.style.display = 'flex';
  }

  hideStatus() {
    this.status.style.display = 'none';
  }

  dispose(resetStatus = true) {
    this.disposed = true;
    if (this.resizeObserver) this.resizeObserver.disconnect();
    this.eventCleanups.forEach((cleanup) => cleanup());
    this.eventCleanups = [];
    if (this.engine) this.engine.stopRenderLoop();
    if (this.scene) this.scene.dispose();
    if (this.engine) this.engine.dispose();
    this.resizeObserver = null;
    this.container = null;
    this.root = null;
    this.scene = null;
    this.engine = null;
    if (resetStatus) this.setStatus('Waiting for row visibility');
  }
}

window.addEventListener('beforeunload', disposeAll);
init();

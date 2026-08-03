from pathlib import Path

import numpy as np
from manimlib import *


VOX_IDX_DIR = Path(
    "/data/HHHH/workspace/UltraShape-1.0/outputs/sample_glb_voxel/vox_idx"
)


class FlashVDMVolumeDecodingFlow(ThreeDScene):
    """
    Render:
    manimgl flashvdm/flashvdm_volume_decoding_scene.py FlashVDMVolumeDecodingFlow -w
    """

    default_camera_config = {
        "background_color": "#0D1117",
    }

    RESOLUTIONS = [4, 8, 16, 32]
    VOLUME_SIDE = 3.2
    SPEED_SCALE = 2.0
    PREVIOUS_CELL_OPACITY = 0.015
    GRID_BASE_OPACITY = 0.30
    GRID_FADE_FACTOR = 0.4
    BOUND_BASE_OPACITY = 0.75
    BOUND_FADE_FACTOR = 0.7
    NPZ_AXIS_ORDER = (0, 2, 1)
    POINT_RADII_BY_STROKE_WIDTH = {
        8: 0.055,
        6: 0.038,
        4: 0.024,
        3: 0.016,
    }

    def _dense_cells(self, resolution):
        return np.indices((resolution, resolution, resolution)).reshape(3, -1).T

    def _load_active_cells(self, resolution):
        path = VOX_IDX_DIR / f"sample_r{resolution}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing active-cell index file: {path}")

        with np.load(path) as data:
            if "voxel_idx" not in data:
                raise KeyError(f"`voxel_idx` not found in {path}")
            cells = np.asarray(data["voxel_idx"], dtype=int)

        cells = cells[:, self.NPZ_AXIS_ORDER]

        if cells.ndim != 2 or cells.shape[1] != 3:
            raise ValueError(f"`voxel_idx` in {path} must have shape [N, 3]")

        if len(cells) == 0:
            return np.empty((0, 3), dtype=int)

        cells = np.unique(cells, axis=0)
        if cells.min() < 0 or cells.max() >= resolution:
            raise ValueError(f"Out-of-range voxel index found in {path}")

        order = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))
        return cells[order]

    def _expand_children(self, parent_cells, next_resolution):
        if len(parent_cells) == 0:
            return np.empty((0, 3), dtype=int)

        offsets = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=int,
        )
        children = (parent_cells[:, None, :] * 2 + offsets[None, :, :]).reshape(-1, 3)
        valid = np.all((children >= 0) & (children < next_resolution), axis=1)
        return np.unique(children[valid], axis=0)

    def _cell_tuple_set(self, cells):
        return {(int(x), int(y), int(z)) for x, y, z in cells}

    def _validate_hierarchy(self, active_by_resolution):
        for prev_res, next_res in zip(self.RESOLUTIONS, self.RESOLUTIONS[1:]):
            expanded = self._expand_children(active_by_resolution[prev_res], next_res)
            expanded_set = self._cell_tuple_set(expanded)

            missing = []
            for cell in active_by_resolution[next_res]:
                cell_tuple = (int(cell[0]), int(cell[1]), int(cell[2]))
                if cell_tuple not in expanded_set:
                    missing.append(cell_tuple)
                    if len(missing) >= 5:
                        break

            if missing:
                raise ValueError(
                    f"sample_r{next_res}.npz is not a subset of expanded sample_r{prev_res}.npz. "
                    f"Examples: {missing}"
                )

    def _to_point(self, coord, resolution):
        return ((coord + 0.5) / resolution - 0.5) * self.VOLUME_SIDE

    def _project_to_hud(self, point):
        camera_point = self.frame.to_fixed_frame_point(np.array(point))
        z_scale = self.frame.get_scale() / self.frame.get_focal_distance()
        perspective = 1.0 / max(1e-8, 1.0 - camera_point[2] * z_scale)
        return np.array(
            [camera_point[0] * perspective, camera_point[1] * perspective, 0.0]
        )

    def _grid_wireframe(self, resolution, color=GREY_D, stroke_width=1.0, stride=1):
        lines = Group()
        half_side = self.VOLUME_SIDE / 2
        vals = np.linspace(-half_side, half_side, resolution + 1)
        vals = vals[:: max(1, stride)]
        if not np.isclose(vals[-1], half_side):
            vals = np.append(vals, half_side)

        # ManimCE uses Line3D(thickness=0.004); ManimGL's width is diameter.
        line_width = 0.008

        for y in vals:
            for z in vals:
                line = Line3D(
                    np.array([-half_side, y, z]),
                    np.array([half_side, y, z]),
                    color=color,
                    width=line_width,
                )
                line.set_color(color, opacity=self.GRID_BASE_OPACITY)
                lines.add(line)
        for x in vals:
            for z in vals:
                line = Line3D(
                    np.array([x, -half_side, z]),
                    np.array([x, half_side, z]),
                    color=color,
                    width=line_width,
                )
                line.set_color(color, opacity=self.GRID_BASE_OPACITY)
                lines.add(line)
        for x in vals:
            for y in vals:
                line = Line3D(
                    np.array([x, y, -half_side]),
                    np.array([x, y, half_side]),
                    color=color,
                    width=line_width,
                )
                line.set_color(color, opacity=self.GRID_BASE_OPACITY)
                lines.add(line)

        return lines

    def _make_point_cloud(self, cells, resolution, color, opacity, stroke_width):
        if len(cells) == 0:
            points = np.zeros((0, 3))
        else:
            points = np.array([self._to_point(c, resolution) for c in cells], dtype=float)
        cloud = DotCloud(
            points=points,
            color=color,
            opacity=opacity,
            radius=self.POINT_RADII_BY_STROKE_WIDTH.get(
                stroke_width, 0.006 * max(1, stroke_width)
            ),
        )
        cloud.make_3d(reflectiveness=0.15, gloss=0.05, shadow=0.1)
        return cloud

    def _make_cells(self, cells, resolution, color, opacity):
        voxels = VGroup()
        if len(cells) == 0:
            return voxels

        size = self.VOLUME_SIDE / resolution * 0.90
        stroke_opacity = min(1.0, opacity + 0.14)
        for cell in cells:
            cube = VCube(
                side_length=size,
                fill_color=color,
                fill_opacity=opacity,
                stroke_color=color,
                stroke_width=0.65,
            )
            cube.set_stroke(color, width=0.65, opacity=stroke_opacity)
            cube.move_to(self._to_point(cell, resolution))
            voxels.add(cube)
        return voxels

    def construct(self):
        frame = self.frame
        frame.reorient(-62, 68, 0, ORIGIN, 8 / 1.2)

        query_color = "#7F8B97"
        pruned_color = "#525E6B"
        keep_colors = {
            4: "#B7895A",
            8: "#6E8F77",
            16: "#8C6D76",
            32: "#6E90A8",
        }
        point_sizes = {4: 8, 8: 6, 16: 4, 32: 3}

        active = {res: self._load_active_cells(res) for res in self.RESOLUTIONS}
        self._validate_hierarchy(active)

        expanded = {
            next_res: self._expand_children(active[prev_res], next_res)
            for prev_res, next_res in zip(self.RESOLUTIONS, self.RESOLUTIONS[1:])
        }
        dense4 = self._dense_cells(4)
        active4_set = self._cell_tuple_set(active[4])

        title = Text(
            "Hierarchical Volume Decoding",
            font_size=32,
            color=WHITE,
        )
        title.to_edge(UP, buff=0.18)
        title.fix_in_frame()
        self.play(
            FadeIn(title, shift=UP * 0.1),
            run_time=1.0 * self.SPEED_SCALE,
        )

        bounds = VCube(
            side_length=self.VOLUME_SIDE,
            fill_opacity=0,
            stroke_color=GREY_C,
            stroke_width=1.5,
        )
        bounds.set_stroke(GREY_C, width=1.5, opacity=self.BOUND_BASE_OPACITY)
        self.play(ShowCreation(bounds, lag_ratio=0), run_time=0.9 * self.SPEED_SCALE)

        vae_text = Text("VAE", font_size=20, color=GREY_A)
        vae_rect = RoundedRectangle(
            corner_radius=0.08,
            width=1.4,
            height=0.7,
            stroke_color="#9AA6B2",
            stroke_width=1.6,
            fill_color="#394450",
            fill_opacity=0.32,
        )
        vae_block = VGroup(vae_rect, vae_text.move_to(vae_rect.get_center()))
        vae_block.to_edge(RIGHT, buff=0.6).shift(UP * 0.08)
        vae_block.fix_in_frame()
        self.play(FadeIn(vae_block), run_time=0.6 * self.SPEED_SCALE)

        grid4 = self._grid_wireframe(4, color=GREY_D, stroke_width=0.90, stride=1)
        all4_points = self._make_point_cloud(
            dense4, 4, query_color, opacity=0.22, stroke_width=point_sizes[4]
        )
        active4_cells = self._make_cells(active[4], 4, keep_colors[4], opacity=0.76)

        grid8 = self._grid_wireframe(8, color=GREY_D, stroke_width=0.62, stride=2)
        expand8_points = self._make_point_cloud(
            expanded[8], 8, query_color, opacity=0.18, stroke_width=point_sizes[8]
        )
        active8_cells = self._make_cells(active[8], 8, keep_colors[8], opacity=0.68)

        grid16 = self._grid_wireframe(16, color=GREY_D, stroke_width=0.46, stride=4)
        expand16_points = self._make_point_cloud(
            expanded[16], 16, query_color, opacity=0.14, stroke_width=point_sizes[16]
        )
        active16_cells = self._make_cells(active[16], 16, keep_colors[16], opacity=0.54)

        grid32 = self._grid_wireframe(32, color=GREY_D, stroke_width=0.34, stride=8)
        expand32_points = self._make_point_cloud(
            expanded[32], 32, query_color, opacity=0.11, stroke_width=point_sizes[32]
        )
        active32_cells = self._make_cells(active[32], 32, keep_colors[32], opacity=0.46)

        grid_fade_1 = self.GRID_BASE_OPACITY * self.GRID_FADE_FACTOR
        grid_fade_2 = grid_fade_1 * self.GRID_FADE_FACTOR
        grid_fade_3 = grid_fade_2 * self.GRID_FADE_FACTOR
        bounds_fade_1 = self.BOUND_BASE_OPACITY * self.BOUND_FADE_FACTOR
        bounds_fade_2 = bounds_fade_1 * self.BOUND_FADE_FACTOR
        bounds_fade_3 = bounds_fade_2 * self.BOUND_FADE_FACTOR

        self.play(FadeIn(grid4), FadeIn(all4_points), run_time=0.9 * self.SPEED_SCALE)
        rng = np.random.default_rng(7)
        sample_idx = rng.choice(len(dense4), size=min(16, len(dense4)), replace=False)
        sample_cells = dense4[sample_idx]
        sample_keep = np.array(
            [tuple(int(v) for v in cell) in active4_set for cell in sample_cells],
            dtype=bool,
        )
        sample_dots = VGroup(
            *[
                Dot(
                    point=self._to_point(cell, 4),
                    radius=0.038,
                    fill_color=query_color,
                    stroke_width=0,
                )
                for cell in sample_cells
            ]
        )
        self.play(FadeIn(sample_dots), run_time=0.35 * self.SPEED_SCALE)

        sample_hud_dots = VGroup(
            *[
                Dot(
                    point=self._project_to_hud(dot.get_center()),
                    radius=0.032,
                    fill_color=query_color,
                    stroke_width=0,
                )
                for dot in sample_dots
            ]
        )
        sample_hud_dots.fix_in_frame()
        self.play(
            sample_dots.animate.set_opacity(0.05),
            FadeIn(sample_hud_dots),
            run_time=0.45 * self.SPEED_SCALE,
        )
        self.play(
            LaggedStart(
                *[
                    dot.animate.move_to(vae_block.get_center()).set_opacity(0.95)
                    for dot in sample_hud_dots
                ],
                lag_ratio=0.08,
            ),
            Indicate(vae_block, color="#A4B0BC"),
            run_time=1.1 * self.SPEED_SCALE,
        )
        self.play(
            LaggedStart(
                *[
                    dot.animate.move_to(self._project_to_hud(self._to_point(cell, 4)))
                    .set_color(keep_colors[4] if keep else pruned_color)
                    .set_opacity(1.0 if keep else 0.20)
                    for dot, cell, keep in zip(sample_hud_dots, sample_cells, sample_keep)
                ],
                lag_ratio=0.08,
            ),
            run_time=1.5 * self.SPEED_SCALE,
        )
        self.play(
            FadeOut(sample_hud_dots),
            FadeOut(sample_dots),
            run_time=0.3 * self.SPEED_SCALE,
        )
        self.play(FadeIn(active4_cells), run_time=0.8 * self.SPEED_SCALE)
        self.play(
            all4_points.animate.set_opacity(0.015),
            run_time=0.8 * self.SPEED_SCALE,
        )
        self.wait(0.25 * self.SPEED_SCALE)

        self.play(
            active4_cells.animate.set_opacity(self.PREVIOUS_CELL_OPACITY),
            bounds.animate.set_stroke(opacity=bounds_fade_1),
            FadeIn(grid8),
            FadeIn(expand8_points),
            run_time=1.0 * self.SPEED_SCALE,
        )
        self.play(FadeIn(active8_cells), run_time=0.8 * self.SPEED_SCALE)
        self.play(
            expand8_points.animate.set_opacity(0.018),
            grid4.animate.set_opacity(grid_fade_1),
            run_time=0.7 * self.SPEED_SCALE,
        )
        self.wait(0.2 * self.SPEED_SCALE)

        self.play(
            active8_cells.animate.set_opacity(self.PREVIOUS_CELL_OPACITY),
            expand8_points.animate.set_opacity(0.006),
            bounds.animate.set_stroke(opacity=bounds_fade_2),
            FadeIn(grid16),
            FadeIn(expand16_points),
            run_time=1.0 * self.SPEED_SCALE,
        )
        self.play(FadeIn(active16_cells), run_time=0.8 * self.SPEED_SCALE)
        self.play(
            expand16_points.animate.set_opacity(0.015),
            grid8.animate.set_opacity(grid_fade_1),
            grid4.animate.set_opacity(grid_fade_2),
            run_time=0.7 * self.SPEED_SCALE,
        )
        self.wait(0.2 * self.SPEED_SCALE)

        self.play(
            active16_cells.animate.set_opacity(self.PREVIOUS_CELL_OPACITY),
            expand16_points.animate.set_opacity(0.005),
            bounds.animate.set_stroke(opacity=bounds_fade_3),
            FadeIn(grid32),
            FadeIn(expand32_points),
            run_time=1.0 * self.SPEED_SCALE,
        )
        ambient_rotation = lambda m, dt: m.increment_theta(0.1 * dt)
        frame.add_updater(ambient_rotation)
        self.play(FadeIn(active32_cells), run_time=0.9 * self.SPEED_SCALE)
        self.play(
            expand32_points.animate.set_opacity(0.005),
            grid32.animate.set_opacity(0.11),
            grid16.animate.set_opacity(grid_fade_1),
            grid8.animate.set_opacity(grid_fade_2),
            grid4.animate.set_opacity(grid_fade_3),
            run_time=0.8 * self.SPEED_SCALE,
        )
        self.wait((0.8 * self.SPEED_SCALE) + 2.0)
        frame.remove_updater(ambient_rotation)

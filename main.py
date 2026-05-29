# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")


# Basic geometry and physics constants. Distances use km, rates use MB/s.
R_EARTH = 6371.0  # Mean Earth radius used for ECEF geometry and eclipse checks.
ATM_HEIGHT = 100.0  # Extra atmospheric margin used when testing Earth blockage.
C_LIGHT = 299792.458  # Speed of light in km/s, kept for propagation-related extensions.
OMEGA_E = 7.2921150e-5  # Earth rotation angular velocity, rad/s.

# Single ground station location: Beijing, in geodetic degrees.
GS_NAME = "Beijing"
GS_LAT = 39.90
GS_LON = 116.40

# Remote sensing task regions. The total task volume/rate is unchanged and
# split by these weights.
TASK_REGIONS = (
    {"name": "South China Sea", "lat": 15.0, "lon": 115.0, "weight": 0.70},
    {"name": "Chengdu", "lat": 30.57, "lon": 104.07, "weight": 0.30},
)

# Satellite-count samples used by the scalability figures.
SCALE_NODE_TARGETS = (
    100,
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
    5500,
    6000,
    6500,
    7000,
    7500,
    8000,
    8500,
    9000,
    9500,
    10000,
    11000,
    12000,
    13000,
    14000,
    15000,
    16000,
    17000,
    18000,
    19000,
    20000,
)


def ground_station_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.array(
        [
            R_EARTH * np.cos(lat) * np.cos(lon),
            R_EARTH * np.cos(lat) * np.sin(lon),
            R_EARTH * np.sin(lat),
        ],
        dtype=np.float64,
    )


GS_ECEF = ground_station_ecef(GS_LAT, GS_LON)
GS_UNIT = GS_ECEF / np.linalg.norm(GS_ECEF)
TASK_REGION_NAMES = tuple(str(region["name"]) for region in TASK_REGIONS)
TASK_REGION_WEIGHTS = np.asarray([float(region["weight"]) for region in TASK_REGIONS], dtype=np.float64)
TASK_REGION_WEIGHTS = TASK_REGION_WEIGHTS / max(float(TASK_REGION_WEIGHTS.sum()), 1e-9)
TASK_REGION_ECEF = np.stack(
    [ground_station_ecef(float(region["lat"]), float(region["lon"])) for region in TASK_REGIONS],
    axis=0,
)
TASK_REGION_UNIT = TASK_REGION_ECEF / np.linalg.norm(TASK_REGION_ECEF, axis=1, keepdims=True)


def select_best_objective_index(F: np.ndarray, utility_tol: float = 1e-6) -> int:
    """Pick max delivery utility, then lower active/energy/buffer/loss."""
    obj = np.asarray(F, dtype=float)
    if obj.ndim != 2 or obj.shape[0] == 0 or obj.shape[1] < 2:
        raise ValueError("objective matrix must be non-empty with at least 2 columns")
    util = -obj[:, 0]
    max_util = float(np.max(util))
    tied = np.where(util >= max_util - max(float(utility_tol), 0.0))[0]
    if obj.shape[1] >= 5:
        order = np.lexsort((obj[tied, 4], obj[tied, 3], obj[tied, 1], obj[tied, 2]))
    elif obj.shape[1] >= 3:
        total_active = obj[tied, 1] + obj[tied, 2]
        order = np.lexsort((obj[tied, 1], obj[tied, 2], total_active))
    else:
        order = np.argsort(obj[tied, 1])
    return int(tied[order[0]])


def scale_node_targets(n_total: int) -> np.ndarray:
    targets = [n for n in SCALE_NODE_TARGETS if 0 <= n <= n_total]
    if n_total > 0 and n_total not in targets:
        targets.append(n_total)
    return np.asarray(sorted(set(targets)), dtype=int)


@dataclass
class Config:
    # Input/output paths. The default data pipeline uses one satellite state CSV.
    SAT_STATE_CSV: str = "satellite_state_all.csv"
    OUTPUT_DIR: str = "Simulation_Experiment_Results"
    CACHE_DIR: str = ".cache_experiment"

    # Run control. Use SAMPLE_EVERY/MAX_STEPS/SAT_LIMIT to downsample for tests.
    RNG_SEED: int = 42
    SAMPLE_EVERY: int = 1
    MAX_STEPS: Optional[int] = None
    SAT_LIMIT: Optional[int] = None

    # Queue and energy model. Energy is normalized to [0, E_MAX].
    Q_MAX_COMM: float = 5000.0
    E_MAX: float = 100.0
    E_MELT: float = 20.0
    E_WARN_RATIO: float = 1.35
    E_BASE_DRAIN: float = 0.08
    E_SOLAR_CHG: float = 0.12
    E_TX_DRAIN: float = 0.03
    E_SENSE_DRAIN: float = 0.02
    E_RX_RATIO: float = 0.5
    E_MELT_RECOVER_RATIO: float = 1.5
    E_GUARD_RECOVER_RATIO: float = 1.6

    # Communication model and sparse ISL graph controls.
    ELEVATION_MIN: float = 30.0
    TX_RATE_ISL: float = 150.0
    TX_RATE_SGL: float = 300.0
    ISL_MAX_DIST: float = 5000.0
    ISL_NEIGHBOR_K: int = 24
    TOPK_CANDIDATES: int = 24
    ADAPTIVE_K_BASE: int = 8
    ADAPTIVE_K_MIN: int = 4
    ADAPTIVE_K_MAX: int = 24
    ADAPTIVE_K_ALPHA: float = 8.0
    ADAPTIVE_K_BETA: float = 6.0
    ADAPTIVE_K_GAMMA: float = 6.0
    HUNGARIAN_MAX_CELLS: int = 250_000

    # Task generation and utility model.
    # TASK_GEN_RATE is the total task data released per simulated second.
    # R_SENSOR is the per-satellite sensing ingestion cap.
    R_SENSOR: float = 200.0
    TASK_GEN_RATE: float = 100.0
    TASK_RADIUS: float = 1000.0
    TASK_DATA_MB: float = 5000.0
    URGENCY: float = 8.5
    PENALTY: float = 0.1
    GS_ANTENNAS: int = 4
    WINDOW_LOOKAHEAD: int = 10

    # NSGA-III search parameters.
    GA_POP: int = 120
    GA_GEN: int = 60
    GA_CR: float = 0.9
    GA_PARTITIONS: int = 4
    GA_CONV_WINDOW: int = 10
    GA_CONV_EPS: float = 0.5

    # Extra experiments used by Fig. 5 and Fig. 6.
    COMPLEXITY_REPEATS: int = 3
    COMPLEXITY_MAX_POINTS: int = 8
    COMPLEXITY_MAX_STEPS: int = 96
    BASELINE_REPEATS: int = 1

    # Simplified astronomical epoch used for Sun direction and ICRF->ECEF rotation.
    T0_DOY: float = 76.0
    T0_GMST: float = 0.5

    # Feature switches.
    PLOT: bool = True
    USE_ISL_CACHE: bool = True


def apply_profile(cfg: Config, profile: str) -> Config:
    if profile == "smoke":
        # Minimal profile for syntax/flow checks. It should finish quickly.
        cfg.MAX_STEPS = 24
        cfg.SAT_LIMIT = 120
        cfg.GA_POP = 8
        cfg.GA_GEN = 2
        cfg.GA_PARTITIONS = 2
        cfg.HUNGARIAN_MAX_CELLS = 60_000
        cfg.COMPLEXITY_REPEATS = 1
        cfg.COMPLEXITY_MAX_POINTS = 3
        cfg.COMPLEXITY_MAX_STEPS = 12
    elif profile == "fast":
        # Medium profile for quick experiments while preserving the main workflow.
        cfg.MAX_STEPS = 240
        cfg.SAT_LIMIT = 400
        cfg.GA_POP = 24
        cfg.GA_GEN = 12
        cfg.GA_PARTITIONS = 3
        cfg.COMPLEXITY_REPEATS = 1
        cfg.COMPLEXITY_MAX_POINTS = 5
        cfg.COMPLEXITY_MAX_STEPS = 48
    elif profile != "full":
        raise ValueError(f"unknown profile: {profile}")
    cfg.ISL_NEIGHBOR_K = int(cfg.ADAPTIVE_K_MAX)
    cfg.TOPK_CANDIDATES = int(cfg.ADAPTIVE_K_MAX)
    return cfg


class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        print("=" * 72)
        print("S1 Data loading and geometry preprocessing")
        print("=" * 72)
        self._load_state_csv()
        self._icrf_to_ecef()
        self._compute_ground_geometry()
        self._compute_eclipse()
        self._precompute_isl_neighbors()
        print(
            f"Data ready: N={self.N}, T={self.T}, dt={self.dt_eff:.1f}s, "
            f"span={self.times_s[0]:.1f}..{self.times_s[-1]:.1f}s"
        )

    def _resolve_csv_path(self) -> Path:
        path = Path(self.cfg.SAT_STATE_CSV)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"satellite state CSV not found: {path}")
        return path

    def _load_state_csv(self) -> None:
        csv_path = self._resolve_csv_path()
        print(f"  loading CSV: {csv_path}")
        usecols = [
            "sat_id",
            "time_s",
            "x_km",
            "y_km",
            "z_km",
            "vx_km_s",
            "vy_km_s",
            "vz_km_s",
        ]
        dtype = {
            "sat_id": "string",
            "time_s": "float64",
            "x_km": "float32",
            "y_km": "float32",
            "z_km": "float32",
            "vx_km_s": "float32",
            "vy_km_s": "float32",
            "vz_km_s": "float32",
        }
        df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=usecols, dtype=dtype)
        if df.empty:
            raise ValueError("satellite state CSV is empty")

        sat_names = np.array(sorted(df["sat_id"].dropna().unique().tolist()), dtype=object)
        times = np.sort(df["time_s"].dropna().unique().astype(np.float64))
        if self.cfg.SAT_LIMIT is not None:
            sat_names = sat_names[: self.cfg.SAT_LIMIT]
        if self.cfg.SAMPLE_EVERY > 1:
            times = times[:: self.cfg.SAMPLE_EVERY]
        if self.cfg.MAX_STEPS is not None:
            times = times[: self.cfg.MAX_STEPS]

        df = df[df["sat_id"].isin(sat_names) & df["time_s"].isin(times)].copy()
        sat_index = pd.Index(sat_names)
        time_index = pd.Index(times)
        ti = time_index.get_indexer(df["time_s"].to_numpy(dtype=np.float64))
        si = sat_index.get_indexer(df["sat_id"].to_numpy())
        valid = (ti >= 0) & (si >= 0)
        if not np.all(valid):
            df = df.iloc[np.where(valid)[0]]
            ti = ti[valid]
            si = si[valid]

        self.sat_names = sat_names
        self.times_s = times.astype(np.float64)
        self.T = len(self.times_s)
        self.N = len(self.sat_names)
        if self.T == 0 or self.N == 0:
            raise ValueError("no satellite samples remain after filtering")

        self.pos_icrf = np.full((self.T, self.N, 3), np.nan, dtype=np.float32)
        self.vel_icrf = np.full((self.T, self.N, 3), np.nan, dtype=np.float32)
        self.pos_icrf[ti, si, 0] = df["x_km"].to_numpy(dtype=np.float32)
        self.pos_icrf[ti, si, 1] = df["y_km"].to_numpy(dtype=np.float32)
        self.pos_icrf[ti, si, 2] = df["z_km"].to_numpy(dtype=np.float32)
        self.vel_icrf[ti, si, 0] = df["vx_km_s"].to_numpy(dtype=np.float32)
        self.vel_icrf[ti, si, 1] = df["vy_km_s"].to_numpy(dtype=np.float32)
        self.vel_icrf[ti, si, 2] = df["vz_km_s"].to_numpy(dtype=np.float32)

        missing = int(np.isnan(self.pos_icrf[..., 0]).sum())
        if missing:
            raise ValueError(
                f"CSV is not rectangular after filtering: missing {missing} position samples"
            )

        if self.T > 1:
            dts = np.diff(self.times_s)
            if not np.allclose(dts, np.median(dts), rtol=0.0, atol=1e-6):
                raise ValueError("time_s values are not uniformly spaced")
            self.dt_eff = float(np.median(dts))
        else:
            self.dt_eff = 60.0

        rows_expected = self.T * self.N
        if len(df) != rows_expected:
            raise ValueError(
                f"expected {rows_expected} rows after filtering, got {len(df)}"
            )
        print(f"  loaded rows={len(df)}, satellites={self.N}, time steps={self.T}")

    def _icrf_to_ecef(self) -> None:
        print("  converting ICRF positions to ECEF by Earth rotation")
        theta = self.cfg.T0_GMST + OMEGA_E * self.times_s
        ct = np.cos(theta).astype(np.float32)
        st = np.sin(theta).astype(np.float32)
        self.pos_ecef = np.empty_like(self.pos_icrf)
        self.pos_ecef[:, :, 0] = (
            ct[:, None] * self.pos_icrf[:, :, 0]
            + st[:, None] * self.pos_icrf[:, :, 1]
        )
        self.pos_ecef[:, :, 1] = (
            -st[:, None] * self.pos_icrf[:, :, 0]
            + ct[:, None] * self.pos_icrf[:, :, 1]
        )
        self.pos_ecef[:, :, 2] = self.pos_icrf[:, :, 2]

    def _compute_ground_geometry(self) -> None:
        print("  computing GS visibility and task coverage")
        self.vis_gs = np.zeros((self.T, self.N), dtype=bool)
        self.task_coverage_by_region = np.zeros((len(TASK_REGIONS), self.T, self.N), dtype=bool)
        self.task_coverage = np.zeros((self.T, self.N), dtype=bool)
        sin_el_min = np.sin(np.radians(self.cfg.ELEVATION_MIN))
        cos_task = np.cos(self.cfg.TASK_RADIUS / R_EARTH)
        chunk = 128
        for start in range(0, self.T, chunk):
            end = min(start + chunk, self.T)
            pos = self.pos_ecef[start:end].astype(np.float64, copy=False)
            vec = pos - GS_ECEF[None, None, :]
            dist = np.linalg.norm(vec, axis=2) + 1e-9
            sin_el = np.sum(vec * GS_UNIT[None, None, :], axis=2) / dist
            self.vis_gs[start:end] = sin_el >= sin_el_min

            r_norm = np.linalg.norm(pos, axis=2) + 1e-9
            for region_i, region_unit in enumerate(TASK_REGION_UNIT):
                cos_ground = np.sum(pos * region_unit[None, None, :], axis=2) / r_norm
                self.task_coverage_by_region[region_i, start:end] = cos_ground >= cos_task
        self.task_region_weights = TASK_REGION_WEIGHTS.copy()
        self.task_coverage = np.any(self.task_coverage_by_region, axis=0)
        region_means = self.task_coverage_by_region.mean(axis=(1, 2))
        region_msg = ", ".join(
            f"{name}={mean:.4f}" for name, mean in zip(TASK_REGION_NAMES, region_means)
        )
        print(
            f"  GS visibility mean={self.vis_gs.mean():.4f}, "
            f"task coverage mean={self.task_coverage.mean():.4f} ({region_msg})"
        )

    def _compute_eclipse(self) -> None:
        print("  computing eclipse state")
        t_days = self.cfg.T0_DOY + self.times_s / 86400.0
        lam = np.radians(280.46 + 0.9856474 * t_days)
        eps = np.radians(23.439)
        sx = np.cos(lam)
        sy = np.cos(eps) * np.sin(lam)
        sz = np.sin(eps) * np.sin(lam)
        theta = self.cfg.T0_GMST + OMEGA_E * self.times_s
        ct = np.cos(theta)
        st = np.sin(theta)
        sun_ecef = np.stack(
            [ct * sx + st * sy, -st * sx + ct * sy, sz],
            axis=1,
        )
        sun_ecef = sun_ecef / (np.linalg.norm(sun_ecef, axis=1, keepdims=True) + 1e-9)
        self.eclipse = np.zeros((self.T, self.N), dtype=bool)
        chunk = 128
        for start in range(0, self.T, chunk):
            end = min(start + chunk, self.T)
            pos = self.pos_ecef[start:end].astype(np.float64, copy=False)
            sun = sun_ecef[start:end]
            dot = np.sum(pos * sun[:, None, :], axis=2)
            perp2 = np.sum(pos * pos, axis=2) - dot * dot
            self.eclipse[start:end] = (dot < 0.0) & (perp2 < R_EARTH * R_EARTH)
        print(f"  eclipse ratio={self.eclipse.mean():.4f}")

    def _cache_path(self) -> Path:
        cache_dir = Path(self.cfg.CACHE_DIR)
        if not cache_dir.is_absolute():
            cache_dir = Path.cwd() / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"isl_topk_N{self.N}_T{self.T}_K{self.cfg.ISL_NEIGHBOR_K}_"
            f"D{int(self.cfg.ISL_MAX_DIST)}_"
            f"t{int(self.times_s[0])}-{int(self.times_s[-1])}.npz"
        )
        return cache_dir / stem

    def _precompute_isl_neighbors(self) -> None:
        K = int(max(1, self.cfg.ISL_NEIGHBOR_K))
        cache_path = self._cache_path()
        if self.cfg.USE_ISL_CACHE and cache_path.exists():
            print(f"  loading ISL Top-K cache: {cache_path}")
            data = np.load(cache_path)
            self.isl_neighbor_idx = data["idx"]
            self.isl_neighbor_dist = data["dist"]
            if self.isl_neighbor_idx.shape == (self.T, self.N, K):
                return
            print("  cache shape mismatch; rebuilding")

        print(f"  precomputing ISL nearest neighbors: K={K}")
        idx_all = np.full((self.T, self.N, K), -1, dtype=np.int32)
        dist_all = np.full((self.T, self.N, K), np.inf, dtype=np.float32)
        t0 = time.perf_counter()
        for t in range(self.T):
            if t % 100 == 0 or t == self.T - 1:
                print(f"    ISL cache step {t + 1}/{self.T}")
            pos = self.pos_icrf[t].astype(np.float64, copy=False)
            tree = cKDTree(pos)
            try:
                dist, idx = tree.query(
                    pos,
                    k=K + 1,
                    distance_upper_bound=self.cfg.ISL_MAX_DIST,
                    workers=-1,
                )
            except TypeError:
                dist, idx = tree.query(
                    pos,
                    k=K + 1,
                    distance_upper_bound=self.cfg.ISL_MAX_DIST,
                )
            if K + 1 == 1:
                dist = dist[:, None]
                idx = idx[:, None]
            dist = dist[:, 1 : K + 1]
            idx = idx[:, 1 : K + 1]
            invalid = (idx >= self.N) | ~np.isfinite(dist)
            idx = idx.astype(np.int32, copy=False)
            idx[invalid] = -1
            dist = dist.astype(np.float32, copy=False)
            dist[invalid] = np.inf
            idx_all[t] = idx
            dist_all[t] = dist
        self.isl_neighbor_idx = idx_all
        self.isl_neighbor_dist = dist_all
        print(f"  ISL Top-K cache built in {time.perf_counter() - t0:.1f}s")
        if self.cfg.USE_ISL_CACHE:
            np.savez(cache_path, idx=idx_all, dist=dist_all)
            print(f"  saved ISL Top-K cache: {cache_path}")


class SimulationEngine:
    def __init__(self, cfg: Config, dl: DataLoader):
        self.cfg = cfg
        self.dl = dl
        scale = dl.dt_eff
        self.e_base = cfg.E_BASE_DRAIN * scale
        self.e_solar = cfg.E_SOLAR_CHG * scale
        self.e_tx = cfg.E_TX_DRAIN * scale
        self.e_rx = cfg.E_TX_DRAIN * cfg.E_RX_RATIO * scale
        self.e_sense = cfg.E_SENSE_DRAIN * scale
        self.e_melt_recover = cfg.E_MELT * cfg.E_MELT_RECOVER_RATIO
        self.e_warn = cfg.E_MELT * cfg.E_WARN_RATIO
        self.e_guard_recover = cfg.E_MELT * cfg.E_GUARD_RECOVER_RATIO
        self.blockage_radius = R_EARTH + ATM_HEIGHT

    def split_genes(self, genes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = self.dl.N
        g = np.asarray(genes, dtype=bool)
        if g.size != 2 * N:
            raise ValueError(f"gene length must be {2 * N}, got {g.size}")
        sense_on = g[:N].copy()
        comm_on = g[N:].copy()
        comm_on |= sense_on
        return sense_on, comm_on

    def _los_mask(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        vec = p1 - p0
        dist2 = np.sum(vec * vec, axis=1)
        dist = np.sqrt(np.maximum(dist2, 1e-12))
        s = -np.sum(p0 * vec, axis=1) / np.maximum(dist2, 1e-12)
        cross = np.cross(p0, p1)
        d_min = np.linalg.norm(cross, axis=1) / np.maximum(dist, 1e-9)
        blocked = (s > 0.0) & (s < 1.0) & (d_min < self.blockage_radius)
        return ~blocked

    @staticmethod
    def _fair_allocate(budget: float, capacities: np.ndarray) -> np.ndarray:
        cap = np.maximum(np.asarray(capacities, dtype=np.float64), 0.0)
        alloc = np.zeros_like(cap)
        remaining = float(min(max(float(budget), 0.0), float(cap.sum())))
        active = cap > 1e-9
        while remaining > 1e-9 and np.any(active):
            active_idx = np.where(active)[0]
            before = remaining
            share = remaining / max(len(active_idx), 1)
            take = np.minimum(share, cap[active_idx])
            alloc[active_idx] += take
            cap[active_idx] -= take
            remaining -= float(take.sum())
            active = cap > 1e-9
            if before - remaining < 1e-9:
                break
        return alloc

    @staticmethod
    def _task_priority(q_region: np.ndarray, idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
        idx_arr = np.asarray(idx, dtype=np.int64)
        if idx_arr.size == 0:
            return np.zeros(0, dtype=np.float64)
        q_sub = q_region[:, idx_arr]
        q_total = q_sub.sum(axis=0)
        weighted = (q_sub * weights[:, None]).sum(axis=0)
        max_weight = max(float(np.max(weights)), 1e-9)
        return np.clip(weighted / np.maximum(q_total, 1e-9) / max_weight, 0.0, 1.0)

    def _future_gs_score(self, t: int, sats: np.ndarray) -> np.ndarray:
        sats_arr = np.asarray(sats, dtype=np.int64)
        if sats_arr.size == 0:
            return np.zeros(0, dtype=np.float64)
        t_end = min(int(t) + max(1, int(self.cfg.WINDOW_LOOKAHEAD)), self.dl.T)
        future = self.dl.vis_gs[int(t):t_end, sats_arr]
        denom = max(float(t_end - int(t)), 1.0)
        return np.clip(future.sum(axis=0).astype(np.float64) / denom, 0.0, 1.0)

    def _adaptive_k_value(
        self,
        buffer_pressure: float,
        gs_direction_score: float,
        task_priority: float,
    ) -> int:
        cfg = self.cfg
        raw = (
            float(cfg.ADAPTIVE_K_BASE)
            + float(cfg.ADAPTIVE_K_ALPHA) * float(np.clip(buffer_pressure, 0.0, 1.0))
            + float(cfg.ADAPTIVE_K_BETA) * float(np.clip(gs_direction_score, 0.0, 1.0))
            + float(cfg.ADAPTIVE_K_GAMMA) * float(np.clip(task_priority, 0.0, 1.0))
        )
        return int(np.clip(round(raw), int(cfg.ADAPTIVE_K_MIN), int(cfg.ADAPTIVE_K_MAX)))

    @staticmethod
    def _deplete_region_queue(
        q_region: np.ndarray,
        sat: int,
        amount: float,
        weights: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        remaining = max(float(amount), 0.0)
        moved = np.zeros(q_region.shape[0], dtype=np.float64)
        for region_i in np.argsort(-weights):
            if remaining <= 1e-9:
                break
            take = min(float(q_region[region_i, sat]), remaining)
            if take <= 0.0:
                continue
            q_region[region_i, sat] -= take
            moved[region_i] = take
            remaining -= take
        return float(moved.sum()), moved

    def _move_region_queue(
        self,
        q_region: np.ndarray,
        sender: int,
        receiver: int,
        amount: float,
    ) -> float:
        moved_total, moved_by_region = self._deplete_region_queue(
            q_region,
            sender,
            amount,
            self.dl.task_region_weights,
        )
        if moved_total > 0.0:
            q_region[:, receiver] += moved_by_region
        return moved_total

    def _edge_scores(
        self,
        t: int,
        sender: int,
        cand: np.ndarray,
        dist: np.ndarray,
        q_region: np.ndarray,
        q_total: np.ndarray,
        e: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        distance_score = 1.0 - np.clip(dist / max(cfg.ISL_MAX_DIST, 1e-9), 0.0, 1.0)
        source_buffer_pressure = np.full(
            len(cand),
            np.clip(q_total[sender] / max(cfg.Q_MAX_COMM, 1e-9), 0.0, 1.0),
            dtype=np.float64,
        )
        receiver_free_capacity = np.clip(
            (cfg.Q_MAX_COMM - q_total[cand]) / max(cfg.Q_MAX_COMM, 1e-9),
            0.0,
            1.0,
        )
        gs_direction_score = self._future_gs_score(t, cand)
        energy_score = np.clip(
            np.minimum(e[sender], e[cand]) / max(cfg.E_MAX, 1e-9),
            0.0,
            1.0,
        )
        task_priority = np.full(
            len(cand),
            float(self._task_priority(q_region, np.asarray([sender]), self.dl.task_region_weights)[0]),
            dtype=np.float64,
        )
        return (
            0.15 * distance_score
            + 0.20 * source_buffer_pressure
            + 0.15 * receiver_free_capacity
            + 0.20 * gs_direction_score
            + 0.15 * energy_score
            + 0.15 * task_priority
        )

    def _downlink_priority(self, t: int, cand: np.ndarray, q_region: np.ndarray, e: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        q_total = q_region.sum(axis=0)
        task_priority = self._task_priority(q_region, cand, self.dl.task_region_weights)
        buffer_pressure = np.clip(q_total[cand] / max(cfg.Q_MAX_COMM, 1e-9), 0.0, 1.0)
        t_end = min(int(t) + max(1, int(cfg.WINDOW_LOOKAHEAD)), self.dl.T)
        future_vis = self.dl.vis_gs[int(t):t_end, cand]
        remaining_window = future_vis.sum(axis=0).astype(np.float64) + 1.0
        urgency_raw = q_total[cand] / np.maximum(remaining_window, 1e-9)
        window_urgency = urgency_raw / max(float(np.max(urgency_raw)), 1e-9)
        energy_score = np.clip(e[cand] / max(cfg.E_MAX, 1e-9), 0.0, 1.0)
        return (
            0.35 * task_priority
            + 0.30 * buffer_pressure
            + 0.20 * window_urgency
            + 0.15 * energy_score
        )

    def _isl_transfer(
        self,
        t: int,
        q_region: np.ndarray,
        e: np.ndarray,
        avail: np.ndarray,
        total_energy_demand: float,
        matcher_mode: str = "hungarian",
        use_topk: bool = True,
        use_window_urgency: bool = True,
        use_energy_score: bool = True,
    ) -> float:
        cfg = self.cfg
        dl = self.dl
        if matcher_mode not in {"hungarian", "greedy"}:
            raise ValueError(f"unknown matcher_mode: {matcher_mode}")

        q_total = q_region.sum(axis=0)
        senders = np.where(avail & (q_total > 1e-9))[0]
        receivers_allowed = avail & (q_total < cfg.Q_MAX_COMM - 1e-9)
        if len(senders) == 0 or int(receivers_allowed.sum()) <= 1:
            return total_energy_demand

        pos_t = dl.pos_icrf[t].astype(np.float64, copy=False)
        if use_topk:
            idx_k = dl.isl_neighbor_idx[t, senders]
            dist_k = dl.isl_neighbor_dist[t, senders]
            full_neighbor_lists = None
            full_receiver_idx = None
        else:
            full_receiver_idx = np.where(receivers_allowed)[0]
            if len(full_receiver_idx) == 0:
                return total_energy_demand
            full_tree = cKDTree(pos_t[full_receiver_idx])
            try:
                full_neighbor_lists = full_tree.query_ball_point(
                    pos_t[senders],
                    r=cfg.ISL_MAX_DIST,
                    workers=-1,
                )
            except TypeError:
                full_neighbor_lists = full_tree.query_ball_point(
                    pos_t[senders],
                    r=cfg.ISL_MAX_DIST,
                )
            idx_k = None
            dist_k = None
        rows: List[int] = []
        cols: List[int] = []
        scores: List[float] = []
        dvals: List[float] = []

        for row, s in enumerate(senders):
            if use_topk:
                cand = idx_k[row]
                dist = dist_k[row]
            else:
                local = full_neighbor_lists[row]
                if len(local) == 0:
                    continue
                cand = full_receiver_idx[np.asarray(local, dtype=np.int32)]
                dist = np.linalg.norm(pos_t[cand] - pos_t[s], axis=1)
            valid = (
                (cand >= 0)
                & (cand != s)
                & np.isfinite(dist)
                & (dist > 0.0)
                & (dist < cfg.ISL_MAX_DIST)
            )
            if not np.any(valid):
                continue
            cand = cand[valid]
            dist = dist[valid]
            active = receivers_allowed[cand]
            if not np.any(active):
                continue
            cand = cand[active]
            dist = dist[active]
            if use_topk:
                sender_pressure = q_total[s] / max(cfg.Q_MAX_COMM, 1e-9)
                sender_gs = float(self._future_gs_score(t, np.asarray([s]))[0])
                sender_priority = float(
                    self._task_priority(q_region, np.asarray([s]), dl.task_region_weights)[0]
                )
                k_s = self._adaptive_k_value(sender_pressure, sender_gs, sender_priority)
                if len(cand) > k_s:
                    cand = cand[:k_s]
                    dist = dist[:k_s]
            los = self._los_mask(
                np.repeat(pos_t[s][None, :], len(cand), axis=0),
                pos_t[cand],
            )
            if not np.any(los):
                continue
            cand = cand[los]
            dist = dist[los]
            if use_energy_score:
                score = self._edge_scores(t, int(s), cand, dist, q_region, q_total, e)
            else:
                d_norm = dist / max(cfg.ISL_MAX_DIST, 1e-9)
                q_norm = (cfg.Q_MAX_COMM - q_total[cand]) / max(cfg.Q_MAX_COMM, 1e-9)
                downlink_bonus = dl.vis_gs[t, cand].astype(float)
                score = (
                    0.50 * (1.0 - d_norm)
                    + 0.40 * q_norm
                    + 0.10 * downlink_bonus
                )
            for c, sc, dd in zip(cand, score, dist):
                rows.append(row)
                cols.append(int(c))
                scores.append(float(sc))
                dvals.append(float(dd))

        if not rows:
            return total_energy_demand

        rows_arr = np.asarray(rows, dtype=np.int32)
        cols_arr = np.asarray(cols, dtype=np.int32)
        scores_arr = np.asarray(scores, dtype=np.float64)
        receiver_unique = np.unique(cols_arr)
        assignments: List[Tuple[int, int]] = []

        n_cells = len(senders) * len(receiver_unique)
        if matcher_mode == "hungarian" and n_cells <= cfg.HUNGARIAN_MAX_CELLS:
            col_pos = {int(c): i for i, c in enumerate(receiver_unique)}
            score_mat = np.full((len(senders), len(receiver_unique)), -1e9)
            for r, c, sc in zip(rows_arr, cols_arr, scores_arr):
                j = col_pos[int(c)]
                if sc > score_mat[r, j]:
                    score_mat[r, j] = sc
            rr, cc = linear_sum_assignment(-score_mat)
            for r, c in zip(rr, cc):
                if score_mat[r, c] > -1e8:
                    assignments.append((int(senders[r]), int(receiver_unique[c])))
        else:
            used_s = set()
            used_r = set()
            order = np.argsort(-scores_arr)
            for edge_i in order:
                s = int(senders[rows_arr[edge_i]])
                r = int(cols_arr[edge_i])
                if s in used_s or r in used_r:
                    continue
                used_s.add(s)
                used_r.add(r)
                assignments.append((s, r))

        tx_cap = cfg.TX_RATE_ISL * self.dl.dt_eff
        for s, r in assignments:
            if q_total[s] <= 0.0 or q_total[r] >= cfg.Q_MAX_COMM:
                continue
            tx = min(q_total[s], tx_cap, cfg.Q_MAX_COMM - q_total[r])
            if tx <= 0.0:
                continue
            moved = self._move_region_queue(q_region, s, r, tx)
            if moved <= 0.0:
                continue
            q_total[s] -= moved
            q_total[r] += moved
            e[s] = max(0.0, e[s] - self.e_tx)
            e[r] = max(0.0, e[r] - self.e_rx)
            total_energy_demand += self.e_tx + self.e_rx
        return total_energy_demand

    def evaluate(
        self,
        genes: np.ndarray,
        track_history: bool = False,
        return_summary: bool = False,
        matcher_mode: str = "hungarian",
        use_topk: bool = True,
        use_window_urgency: bool = True,
        use_energy_score: bool = True,
    ) -> Tuple[float, Tuple[int, int], Optional[Dict]]:
        cfg = self.cfg
        dl = self.dl
        eval_t0 = time.perf_counter()
        sense_on, comm_on = self.split_genes(genes)
        n_sense = int(sense_on.sum())
        n_comm = int(comm_on.sum())
        if n_comm == 0:
            empty_summary = {
                "delivered_mb": 0.0,
                "delivery_ratio": 0.0,
                "sensing_progress": 0.0,
                "unsensed_mb": cfg.TASK_DATA_MB,
                "generated_mb": 0.0,
                "dropped_mb": 0.0,
                "packet_loss_rate": 1.0,
                "avg_queue_backlog_mb": 0.0,
                "sense_active_count": n_sense,
                "comm_active_count": n_comm,
                "active_satellite_count": n_comm,
                "total_energy_demand": 0.0,
                "unit_delivered_energy": None,
                "completion_time_s": None,
                "online_scheduling_time_s": 0.0,
                "simulation_runtime_s": float(time.perf_counter() - eval_t0),
                "avg_remaining_energy": 0.0,
                "min_remaining_energy": 0.0,
                "low_power_protection_count": 0,
                "melt_count": 0,
                "final_buffer_mb": 0.0,
                "utility": 0.0,
                "delivery_utility": 0.0,
            }
            return 0.0, (n_sense, n_comm), {"summary": empty_summary} if return_summary else None

        active_comm = np.where(comm_on)[0]
        e = np.full(dl.N, cfg.E_MAX, dtype=np.float64)
        q_region = np.zeros((len(TASK_REGIONS), dl.N), dtype=np.float64)
        melt = np.zeros(dl.N, dtype=bool)
        guard = np.zeros(dl.N, dtype=bool)

        delivered = 0.0
        delivered_by_region = np.zeros(len(TASK_REGIONS), dtype=np.float64)
        remaining = cfg.TASK_DATA_MB
        remaining_by_region = cfg.TASK_DATA_MB * dl.task_region_weights.astype(np.float64, copy=True)
        generated_total = 0.0
        generated_by_region = np.zeros(len(TASK_REGIONS), dtype=np.float64)
        dropped_sensing = 0.0
        dropped_melt = 0.0
        dropped_by_region = np.zeros(len(TASK_REGIONS), dtype=np.float64)
        queue_integral = 0.0
        total_energy_demand = 0.0
        completion_time_s = None
        online_scheduling_time_s = 0.0
        low_power_events = 0
        melt_events = 0

        hist = None
        if track_history:
            hist = {
                "time_s": [],
                "delivered_cum": [],
                "queue_total": [],
                "energy_mean": [],
                "energy_min": [],
                "melt_count": [],
                "guard_count": [],
                "drop_cum": [],
            }

        for t in range(dl.T):
            ecl = dl.eclipse[t, active_comm]
            solar = np.where(ecl, 0.0, self.e_solar)
            e[active_comm] = np.clip(
                e[active_comm] + solar - self.e_base,
                0.0,
                cfg.E_MAX,
            )
            total_energy_demand += self.e_base * n_comm

            new_melt = melt.copy()
            new_melt[comm_on & (e < cfg.E_MELT)] = True
            new_melt[(e > self.e_melt_recover) & comm_on] = False
            just_melt = new_melt & ~melt
            if np.any(just_melt):
                melt_drop = q_region[:, just_melt].sum(axis=1)
                dropped_by_region += melt_drop
                dropped_melt += float(melt_drop.sum())
                q_region[:, just_melt] = 0.0
                melt_events += int(just_melt.sum())
            melt = new_melt

            prev_guard = guard.copy()
            low_energy = comm_on & ~melt & (e <= self.e_warn)
            guard[low_energy] = True
            guard[(e >= self.e_guard_recover) & comm_on] = False
            guard[melt | ~comm_on] = False
            low_power_events += int((guard & ~prev_guard).sum())

            avail = comm_on & ~melt & ~guard

            if remaining > 0.0:
                sensor_step_cap = np.full(dl.N, cfg.R_SENSOR * dl.dt_eff, dtype=np.float64)
                q_total = q_region.sum(axis=0)
                for region_i, region_weight in enumerate(dl.task_region_weights):
                    if remaining_by_region[region_i] <= 0.0:
                        continue
                    can_sense = (
                        sense_on
                        & avail
                        & dl.task_coverage_by_region[region_i, t]
                        & (sensor_step_cap > 1e-9)
                    )
                    sense_idx = np.where(can_sense)[0]
                    if len(sense_idx) == 0:
                        continue
                    step_budget = min(
                        cfg.TASK_GEN_RATE * dl.dt_eff * float(region_weight),
                        remaining_by_region[region_i],
                    )
                    per_sat_cap = np.minimum(
                        np.maximum(cfg.Q_MAX_COMM - q_total[sense_idx], 0.0),
                        sensor_step_cap[sense_idx],
                    )
                    alloc = self._fair_allocate(step_budget, per_sat_cap)
                    used = alloc > 1e-9
                    if np.any(used):
                        target = sense_idx[used]
                        stored = alloc[used]
                        q_region[region_i, target] += stored
                        q_total[target] += stored
                        sensor_step_cap[target] = np.maximum(0.0, sensor_step_cap[target] - stored)
                        actual_generated = float(stored.sum())
                        generated_total += actual_generated
                        generated_by_region[region_i] += actual_generated
                        remaining_by_region[region_i] = max(
                            0.0,
                            float(remaining_by_region[region_i]) - actual_generated,
                        )
                        remaining -= actual_generated
                        e[target] = np.maximum(0.0, e[target] - self.e_sense)
                        total_energy_demand += self.e_sense * int(len(target))

            sched_t0 = time.perf_counter()
            total_energy_demand = self._isl_transfer(
                t,
                q_region,
                e,
                avail,
                total_energy_demand,
                matcher_mode=matcher_mode,
                use_topk=use_topk,
                use_window_urgency=use_window_urgency,
                use_energy_score=use_energy_score,
            )
            online_scheduling_time_s += time.perf_counter() - sched_t0

            tx_sgl = cfg.TX_RATE_SGL * dl.dt_eff
            sched_t0 = time.perf_counter()
            q_total = q_region.sum(axis=0)
            cand = np.where(avail & dl.vis_gs[t] & (q_total > 1e-9))[0]
            if len(cand) > 0:
                if use_energy_score:
                    priority = self._downlink_priority(t, cand, q_region, e)
                else:
                    priority = q_total[cand]
                order = np.argsort(-priority)
                ant_used = 0
                for sat in cand[order]:
                    if ant_used >= cfg.GS_ANTENNAS:
                        break
                    tx = min(q_total[sat], tx_sgl)
                    if tx <= 0.0:
                        continue
                    actual_tx, delivered_regions = self._deplete_region_queue(
                        q_region,
                        int(sat),
                        tx,
                        dl.task_region_weights,
                    )
                    if actual_tx <= 0.0:
                        continue
                    q_total[sat] -= actual_tx
                    delivered += actual_tx
                    delivered_by_region += delivered_regions
                    e[sat] = max(0.0, e[sat] - self.e_tx)
                    total_energy_demand += self.e_tx
                    ant_used += 1
                if completion_time_s is None and delivered >= cfg.TASK_DATA_MB - 1e-9:
                    completion_time_s = float(dl.times_s[t] - dl.times_s[0])
            online_scheduling_time_s += time.perf_counter() - sched_t0

            q_region = np.clip(q_region, 0.0, cfg.Q_MAX_COMM)
            q_total = q_region.sum(axis=0)
            queue_integral += float(q_total[active_comm].sum())

            if track_history and hist is not None:
                hist["time_s"].append(float(dl.times_s[t]))
                hist["delivered_cum"].append(float(delivered))
                hist["queue_total"].append(float(q_total[active_comm].sum()))
                hist["energy_mean"].append(float(e[active_comm].mean()))
                hist["energy_min"].append(float(e[active_comm].min()))
                hist["melt_count"].append(int(melt[active_comm].sum()))
                hist["guard_count"].append(int(guard[active_comm].sum()))
                hist["drop_cum"].append(float(dropped_sensing + dropped_melt))

        q_total = q_region.sum(axis=0)
        total_buffer = float(q_total[active_comm].sum())
        delivery_utility = float(np.sum(delivered_by_region * dl.task_region_weights) * cfg.URGENCY)
        utility = delivered * cfg.URGENCY - cfg.PENALTY * total_buffer
        simulation_runtime_s = float(time.perf_counter() - eval_t0)
        remaining = float(max(float(remaining_by_region.sum()), 0.0))
        dropped_total = float(dropped_by_region.sum())
        packet_loss_rate = 1.0 if generated_total <= 1e-9 else float(dropped_total / max(generated_total, 1e-9))
        summary = {
            "delivered_mb": float(delivered),
            "delivery_ratio": float(min(delivered / max(cfg.TASK_DATA_MB, 1e-9), 1.0)),
            "sensing_progress": float(min(generated_total / max(cfg.TASK_DATA_MB, 1e-9), 1.0)),
            "unsensed_mb": float(max(remaining, 0.0)),
            "generated_mb": float(generated_total),
            "generated_by_region_mb": generated_by_region.tolist(),
            "delivered_by_region_mb": delivered_by_region.tolist(),
            "dropped_by_region_mb": dropped_by_region.tolist(),
            "dropped_mb": dropped_total,
            "packet_loss_rate": packet_loss_rate,
            "avg_queue_backlog_mb": float(queue_integral / max(dl.T, 1)),
            "sense_active_count": int(n_sense),
            "comm_active_count": int(n_comm),
            "active_satellite_count": int(n_comm),
            "total_energy_demand": float(total_energy_demand),
            "unit_delivered_energy": (
                float(total_energy_demand / delivered) if delivered > 1e-9 else None
            ),
            "completion_time_s": completion_time_s,
            "online_scheduling_time_s": float(online_scheduling_time_s),
            "simulation_runtime_s": simulation_runtime_s,
            "avg_remaining_energy": float(e[active_comm].mean()),
            "min_remaining_energy": float(e[active_comm].min()),
            "low_power_protection_count": int(low_power_events),
            "melt_count": int(melt_events),
            "final_buffer_mb": float(total_buffer),
            "utility": float(utility),
            "delivery_utility": float(delivery_utility),
            "constraint_valid": bool(np.all(~sense_on | comm_on)),
        }
        if hist is not None:
            hist["summary"] = summary
        elif return_summary:
            hist = {"summary": summary}
        return utility, (n_sense, n_comm), hist


class NSGA3Solver:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _ref_dirs(n_obj: int, H: int) -> np.ndarray:
        out: List[List[float]] = []

        def rec(d: int, left: int, total: int, cur: List[float]) -> None:
            if d == 1:
                out.append(cur + [left / total])
                return
            for i in range(left + 1):
                rec(d - 1, left - i, total, cur + [i / total])

        rec(n_obj, H, H, [])
        return np.asarray(out, dtype=float)

    @staticmethod
    def _nds(F: np.ndarray) -> List[List[int]]:
        n = len(F)
        dom_count = np.zeros(n, dtype=int)
        dom_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = F[i] - F[j]
                if np.all(d <= 0.0) and np.any(d < 0.0):
                    dom_list[i].append(j)
                    dom_count[j] += 1
                elif np.all(d >= 0.0) and np.any(d > 0.0):
                    dom_list[j].append(i)
                    dom_count[i] += 1
        fronts: List[List[int]] = []
        cur = np.where(dom_count == 0)[0].tolist()
        while cur:
            fronts.append(cur)
            nxt = []
            for i in cur:
                for j in dom_list[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        nxt.append(j)
            cur = nxt
        return fronts

    @staticmethod
    def _assoc(F_n: np.ndarray, refs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        refs_n = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-9)
        proj = F_n @ refs_n.T
        d2 = np.maximum(np.sum(F_n * F_n, axis=1, keepdims=True) - proj * proj, 0.0)
        ri = np.argmin(d2, axis=1)
        return ri, np.sqrt(d2[np.arange(len(F_n)), ri])

    def _survive(
        self,
        X: np.ndarray,
        F: np.ndarray,
        pop: int,
        refs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        fronts = self._nds(F)
        selected: List[int] = []
        for fr in fronts:
            if len(selected) + len(fr) <= pop:
                selected.extend(fr)
                continue
            need = pop - len(selected)
            fmin = F.min(axis=0)
            fmax = F.max(axis=0)
            span = np.maximum(fmax - fmin, 1e-9)
            F_last_n = (F[fr] - fmin) / span
            if selected:
                F_sel_n = (F[selected] - fmin) / span
                F_all_n = np.vstack([F_sel_n, F_last_n])
            else:
                F_all_n = F_last_n
            ri_all, di_all = self._assoc(F_all_n, refs)
            ri_last = ri_all[len(selected) :]
            di_last = di_all[len(selected) :]
            niche = np.zeros(len(refs), dtype=int)
            for r in ri_all[: len(selected)]:
                niche[r] += 1
            pool = list(range(len(fr)))
            picked: List[int] = []
            for _ in range(need):
                if not pool:
                    break
                active_refs = np.unique(ri_last[pool])
                min_niche = niche[active_refs].min()
                cand_refs = active_refs[niche[active_refs] == min_niche]
                ref = np.random.choice(cand_refs)
                cand_pool = [p for p in pool if ri_last[p] == ref]
                best = min(cand_pool, key=lambda p: di_last[p])
                picked.append(fr[best])
                niche[ref] += 1
                pool.remove(best)
            selected.extend(picked)
            break
        return X[selected], F[selected]

    @staticmethod
    def _repair(x: np.ndarray, n_sat: int) -> np.ndarray:
        y = x.astype(bool).copy()
        y[n_sat:] |= y[:n_sat]
        return y.astype(float)

    @staticmethod
    def _cx(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.random.rand(len(p1)) < 0.5
        return np.where(mask, p1, p2), np.where(mask, p2, p1)

    def _mut(self, x: np.ndarray, pm: float, n_sat: int) -> np.ndarray:
        y = np.logical_xor(x.astype(bool), np.random.rand(len(x)) < pm)
        return self._repair(y.astype(float), n_sat)

    def _init_population(self, pop: int, n_sat: int) -> np.ndarray:
        n_var = 2 * n_sat
        X = np.zeros((pop, n_var), dtype=float)
        n_biased = pop // 2
        n_random = pop // 4
        n_greedy = pop - n_biased - n_random

        sense = np.random.rand(n_biased, n_sat) < 0.12
        comm = np.random.rand(n_biased, n_sat) < 0.30
        comm |= sense
        X[:n_biased, :n_sat] = sense
        X[:n_biased, n_sat:] = comm

        start = n_biased
        end = start + n_random
        sense = np.random.rand(n_random, n_sat) < 0.25
        comm = np.random.rand(n_random, n_sat) < 0.50
        comm |= sense
        X[start:end, :n_sat] = sense
        X[start:end, n_sat:] = comm

        target_sense = max(2, n_sat // 12)
        target_comm = max(target_sense, n_sat // 4)
        for k in range(end, end + n_greedy):
            x = np.zeros(n_var, dtype=float)
            sense_sel = np.random.choice(n_sat, min(target_sense, n_sat), replace=False)
            comm_sel = np.random.choice(n_sat, min(target_comm, n_sat), replace=False)
            x[sense_sel] = 1.0
            x[n_sat + comm_sel] = 1.0
            x[n_sat + sense_sel] = 1.0
            X[k] = x
        return X

    def optimize(self, eval_fn, n_sat: int) -> Dict:
        c = self.cfg
        pop = c.GA_POP
        gen = c.GA_GEN
        n_var = 2 * n_sat
        refs = self._ref_dirs(5, c.GA_PARTITIONS)
        pm = 1.0 / max(n_var, 1)
        X = self._init_population(pop, n_sat)
        print("=" * 72)
        print(f"S2 NSGA-III optimization: pop={pop}, gen={gen}, dim={n_var}")
        print("=" * 72)
        t0 = time.perf_counter()
        F = np.asarray([eval_fn(X[i].astype(bool)) for i in range(pop)], dtype=float)
        history: List[Dict] = []
        best_util_hist: List[float] = []
        best_total_hist: List[int] = []
        utility_ceiling = float(c.TASK_DATA_MB * c.URGENCY)
        tie_tol = max(float(c.GA_CONV_EPS), 1e-6)

        for g in range(gen):
            off_X = []
            idx = np.random.permutation(pop)
            for k in range(0, pop, 2):
                p1 = X[idx[k]]
                p2 = X[idx[min(k + 1, pop - 1)]]
                if np.random.rand() < c.GA_CR:
                    c1, c2 = self._cx(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                off_X.extend([self._mut(c1, pm, n_sat), self._mut(c2, pm, n_sat)])
            off_X = np.asarray(off_X[:pop], dtype=float)
            off_F = np.asarray(
                [eval_fn(off_X[i].astype(bool)) for i in range(pop)],
                dtype=float,
            )
            X, F = self._survive(
                np.vstack([X, off_X]),
                np.vstack([F, off_F]),
                pop,
                refs,
            )
            best_i = select_best_objective_index(F, utility_tol=tie_tol)
            best_util = float(-F[best_i, 0])
            best_gene = X[best_i].astype(bool)
            best_sense = int(best_gene[:n_sat].sum())
            best_comm = int(best_gene[n_sat:].sum())
            best_total = int(F[best_i, 2])
            util = -F[:, 0]
            near_best = np.where(util >= float(np.max(util)) - tie_tol)[0]
            near_best_total = F[near_best, 2]
            near_best_min_active = int(np.min(near_best_total)) if len(near_best_total) else best_total
            best_util_hist.append(best_util)
            best_total_hist.append(best_total)
            history.append(
                {
                    "gen": g + 1,
                    "best_util": best_util,
                    "sense_count": best_sense,
                    "comm_count": best_comm,
                    "total_active": best_total,
                    "near_best_min_active": near_best_min_active,
                    "near_best_count": int(len(near_best)),
                    "utility_ceiling": utility_ceiling,
                    "utility_gap_to_ceiling": float(max(0.0, utility_ceiling - best_util)),
                    "ceiling_reached": bool(best_util >= utility_ceiling - tie_tol),
                }
            )
            if g == 0 or (g + 1) % 5 == 0 or g == gen - 1:
                print(
                    f"  Gen {g + 1:3d}/{gen} "
                    f"best_U={best_util:.1f} "
                    f"sense={best_sense} comm={best_comm}"
                )
            w = c.GA_CONV_WINDOW
            util_flat = max(best_util_hist[-w:]) - min(best_util_hist[-w:]) < c.GA_CONV_EPS
            total_flat = max(best_total_hist[-w:]) == min(best_total_hist[-w:])
            if g >= w and util_flat and total_flat:
                print(f"  convergence stop at generation {g + 1}")
                break
        runtime_s = float(time.perf_counter() - t0)
        print(f"  optimization time: {runtime_s:.1f}s")
        return {
            "X": X,
            "F": F,
            "history": history,
            "runtime_s": runtime_s,
            "generations_completed": len(history),
            "evaluations": int(pop * (1 + len(history))),
            "utility_ceiling": utility_ceiling,
        }


class ConstellationProblem:
    def __init__(self, cfg: Config, dl: DataLoader):
        self.cfg = cfg
        self.dl = dl
        self.engine = SimulationEngine(cfg, dl)

    def evaluate(self, genes: np.ndarray) -> List[float]:
        _, counts, hist = self.engine.evaluate(genes, return_summary=True)
        n_sense, n_comm = counts
        summary = dict(hist.get("summary", {})) if hist else {}
        generated = float(summary.get("generated_mb", 0.0))
        packet_loss = 1.0 if generated <= 1e-9 else float(summary.get("packet_loss_rate", 1.0))
        return [
            -float(summary.get("delivery_utility", 0.0)),
            float(summary.get("total_energy_demand", 0.0)),
            float(n_comm),
            float(summary.get("final_buffer_mb", 0.0)),
            packet_loss,
        ]


def build_method_configs(dl: DataLoader, best_gene: np.ndarray) -> List[Dict]:
    full_gene = np.ones(2 * dl.N, dtype=bool)
    nsga_gene = best_gene.astype(bool).copy()
    return [
        {
            "method_id": "M1",
            "method_name": "Full + Greedy",
            "label": "M1 Full + Greedy",
            "activation": "Full",
            "candidate_graph": "Full",
            "matcher": "Greedy",
            "gene": full_gene,
            "matcher_mode": "greedy",
            "use_topk": False,
        },
        {
            "method_id": "M2",
            "method_name": "Full + Adaptive-K + Greedy",
            "label": "M2 Full + Adaptive-K + Greedy",
            "activation": "Full",
            "candidate_graph": "Adaptive-K",
            "matcher": "Greedy",
            "gene": full_gene,
            "matcher_mode": "greedy",
            "use_topk": True,
        },
        {
            "method_id": "M3",
            "method_name": "Full + Hungarian",
            "label": "M3 Full + Hungarian",
            "activation": "Full",
            "candidate_graph": "Full",
            "matcher": "Hungarian",
            "gene": full_gene,
            "matcher_mode": "hungarian",
            "use_topk": False,
        },
        {
            "method_id": "M4",
            "method_name": "Full + Adaptive-K + Hungarian",
            "label": "M4 Full + Adaptive-K + Hungarian",
            "activation": "Full",
            "candidate_graph": "Adaptive-K",
            "matcher": "Hungarian",
            "gene": full_gene,
            "matcher_mode": "hungarian",
            "use_topk": True,
        },
        {
            "method_id": "M5",
            "method_name": "NSGA-III + Greedy",
            "label": "M5 NSGA-III + Greedy",
            "activation": "NSGA-III",
            "candidate_graph": "Full",
            "matcher": "Greedy",
            "gene": nsga_gene,
            "matcher_mode": "greedy",
            "use_topk": False,
        },
        {
            "method_id": "M6",
            "method_name": "NSGA-III + Adaptive-K + Greedy",
            "label": "M6 NSGA-III + Adaptive-K + Greedy",
            "activation": "NSGA-III",
            "candidate_graph": "Adaptive-K",
            "matcher": "Greedy",
            "gene": nsga_gene,
            "matcher_mode": "greedy",
            "use_topk": True,
        },
        {
            "method_id": "M7",
            "method_name": "NSGA-III + Hungarian",
            "label": "M7 NSGA-III + Hungarian",
            "activation": "NSGA-III",
            "candidate_graph": "Full",
            "matcher": "Hungarian",
            "gene": nsga_gene,
            "matcher_mode": "hungarian",
            "use_topk": False,
        },
        {
            "method_id": "M8",
            "method_name": "Proposed",
            "label": "M8 Proposed",
            "activation": "NSGA-III",
            "candidate_graph": "Adaptive-K",
            "matcher": "Hungarian",
            "gene": nsga_gene,
            "matcher_mode": "hungarian",
            "use_topk": True,
        },
    ]


def _method_color(method_id: str) -> str:
    colors = {
        "M1": "#4D4D4D",
        "M2": "#1B9E77",
        "M3": "#D95F02",
        "M4": "#7570B3",
        "M5": "#E7298A",
        "M6": "#66A61E",
        "M7": "#E6AB02",
        "M8": "#1F78B4",
    }
    return colors.get(method_id, "#777777")


def run_method_comparison_experiment(
    cfg: Config,
    dl: DataLoader,
    best_gene: np.ndarray,
    nsga_runtime_s: float = 0.0,
) -> Dict:
    print("=" * 72)
    print("S3 M1-M8 comparison experiment")
    print("=" * 72)
    engine = SimulationEngine(cfg, dl)
    methods = build_method_configs(dl, best_gene)
    repeats = max(1, int(cfg.BASELINE_REPEATS))
    rows: List[Dict] = []
    for item in methods:
        util, counts, hist = engine.evaluate(
            item["gene"],
            track_history=False,
            return_summary=True,
            matcher_mode=item["matcher_mode"],
            use_topk=item["use_topk"],
            use_window_urgency=True,
            use_energy_score=True,
        )
        summary = dict(hist.get("summary", {})) if hist else {}
        sim_samples = [float(summary.get("simulation_runtime_s", 0.0))]
        online_samples = [float(summary.get("online_scheduling_time_s", 0.0))]
        for _ in range(max(0, repeats - 1)):
            _, _, sample_hist = engine.evaluate(
                item["gene"],
                track_history=False,
                return_summary=True,
                matcher_mode=item["matcher_mode"],
                use_topk=item["use_topk"],
                use_window_urgency=True,
                use_energy_score=True,
            )
            sample_summary = dict(sample_hist.get("summary", {})) if sample_hist else {}
            sim_samples.append(float(sample_summary.get("simulation_runtime_s", 0.0)))
            online_samples.append(float(sample_summary.get("online_scheduling_time_s", 0.0)))

        activation_runtime_s = float(nsga_runtime_s) if item["activation"] == "NSGA-III" else 0.0
        summary["method_id"] = item["method_id"]
        summary["method_name"] = item["method_name"]
        summary["label"] = item["label"]
        summary["activation_strategy"] = item["activation"]
        summary["candidate_graph"] = item["candidate_graph"]
        summary["matcher"] = item["matcher"]
        summary["scoring"] = "Unified multi-factor score + adaptive K"
        summary["utility"] = float(util)
        summary["sense_active_count"] = int(counts[0])
        summary["comm_active_count"] = int(counts[1])
        summary["active_satellite_count"] = int(counts[1])
        summary["simulation_runtime_s"] = float(np.median(sim_samples))
        summary["online_scheduling_time_s"] = float(np.median(online_samples))
        summary["activation_runtime_s"] = activation_runtime_s
        summary["total_runtime_s"] = float(activation_runtime_s + summary["simulation_runtime_s"])
        summary["color"] = _method_color(item["method_id"])
        rows.append(summary)
        completion = summary.get("completion_time_s")
        completion_txt = "not finished" if completion is None else f"{float(completion):.1f}s"
        print(
            f"  {item['label']:<32s} active={counts[1]:5d} "
            f"delivery={summary.get('delivery_ratio', 0.0) * 100:6.2f}% "
            f"loss={summary.get('packet_loss_rate', 0.0) * 100:6.2f}% "
            f"online={summary['online_scheduling_time_s']:.3f}s "
            f"total={summary['total_runtime_s']:.3f}s "
            f"finish={completion_txt}"
        )
    return {"methods": rows, "nsga_runtime_s": float(nsga_runtime_s)}


def save_method_comparison_outputs(out_dir: Path, comparison: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = comparison.get("methods", []) if comparison else []
    table1 = []
    table2 = []
    for m in methods:
        table1.append(
            {
                "方法": m.get("label", ""),
                "激活卫星数": m.get("active_satellite_count", 0),
                "交付率": m.get("delivery_ratio", 0.0),
                "丢包率": m.get("packet_loss_rate", 0.0),
                "平均队列": m.get("avg_queue_backlog_mb", 0.0),
                "总能耗": m.get("total_energy_demand", 0.0),
                "单位交付能耗": m.get("unit_delivered_energy"),
                "完成时间": m.get("completion_time_s"),
                "在线调度时间": m.get("online_scheduling_time_s", 0.0),
                "总运行时间": m.get("total_runtime_s", 0.0),
            }
        )
        table2.append(
            {
                "方法": m.get("label", ""),
                "平均剩余能量": m.get("avg_remaining_energy", 0.0),
                "最低剩余能量": m.get("min_remaining_energy", 0.0),
                "低电量保护次数": m.get("low_power_protection_count", 0),
                "熔断次数": m.get("melt_count", 0),
            }
        )
    pd.DataFrame(table1).to_csv(out_dir / "table1_method_performance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(table2).to_csv(out_dir / "table2_energy_protection.csv", index=False, encoding="utf-8-sig")
    print(f"  saved M1-M8 comparison tables in {out_dir}")


def run_scalability_experiment(
    cfg: Config,
    dl: DataLoader,
    best_gene: np.ndarray,
    nsga_runtime_s: float = 0.0,
) -> Dict:
    print("=" * 72)
    print("S4 Satellite-scale experiment for Fig1-Fig4")
    print("=" * 72)
    n_total = int(dl.N)
    if n_total <= 1:
        return {"nodes": [], "methods": []}

    rng = np.random.default_rng(int(cfg.RNG_SEED) + 20260515)
    methods = build_method_configs(dl, best_gene)
    node_targets = scale_node_targets(n_total)
    if len(node_targets) == 0:
        node_targets = np.asarray([n_total], dtype=int)

    steps = np.arange(min(dl.T, max(1, int(cfg.COMPLEXITY_MAX_STEPS))), dtype=int)
    repeats = max(1, int(cfg.COMPLEXITY_REPEATS))
    all_idx = np.arange(dl.N, dtype=np.int32)
    best_sense_mask = best_gene[: dl.N].astype(bool)
    best_comm_mask = best_gene[dl.N :].astype(bool) | best_sense_mask
    sense_ratio = float(np.mean(best_sense_mask)) if dl.N else 0.0
    comm_ratio = float(np.mean(best_comm_mask)) if dl.N else 0.0
    sense_ratio = float(np.clip(sense_ratio, 1.0 / max(n_total, 1), 1.0))
    comm_ratio = float(np.clip(comm_ratio, sense_ratio, 1.0))

    e_base = cfg.E_BASE_DRAIN * dl.dt_eff
    e_solar = cfg.E_SOLAR_CHG * dl.dt_eff
    e_tx = cfg.E_TX_DRAIN * dl.dt_eff
    e_rx = cfg.E_TX_DRAIN * cfg.E_RX_RATIO * dl.dt_eff
    e_sense = cfg.E_SENSE_DRAIN * dl.dt_eff
    tx_isl = cfg.TX_RATE_ISL * dl.dt_eff
    tx_sgl = cfg.TX_RATE_SGL * dl.dt_eff
    task_release = cfg.TASK_GEN_RATE * dl.dt_eff
    per_sat_sensor_cap = cfg.R_SENSOR * dl.dt_eff
    los_engine = SimulationEngine(cfg, dl)

    def choose(pool: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(pool) == 0:
            return np.asarray([], dtype=np.int32)
        count = int(np.clip(count, 1, len(pool)))
        if count >= len(pool):
            return np.sort(pool.astype(np.int32))
        return np.sort(rng.choice(pool, count, replace=False).astype(np.int32))

    def choose_preferred(pool: np.ndarray, preferred_mask: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(pool) == 0:
            return np.asarray([], dtype=np.int32)
        count = int(np.clip(count, 1, len(pool)))
        preferred = pool[preferred_mask[pool]]
        picked: List[int] = []
        if len(preferred) > 0:
            take = min(count, len(preferred))
            picked.extend(choose(preferred, take).tolist())
        if len(picked) < count:
            remaining = np.asarray([p for p in pool if int(p) not in set(picked)], dtype=np.int32)
            if len(remaining) > 0:
                picked.extend(choose(remaining, count - len(picked)).tolist())
        return np.sort(np.asarray(picked, dtype=np.int32))

    def activation_for_method(pool: np.ndarray, activation: str) -> Tuple[np.ndarray, np.ndarray]:
        if len(pool) == 0:
            empty = np.asarray([], dtype=np.int32)
            return empty, empty
        if activation == "Full":
            return pool.copy(), pool.copy()
        n = len(pool)
        n_sense = max(1, int(np.ceil(n * sense_ratio)))
        n_comm = max(n_sense, int(np.ceil(n * comm_ratio)))
        sense_idx = choose_preferred(pool, best_sense_mask, n_sense)
        comm_idx = choose_preferred(pool, best_comm_mask, n_comm)
        comm_idx = np.unique(np.concatenate([comm_idx, sense_idx])).astype(np.int32)
        return sense_idx, comm_idx

    def benchmark(
        sense_idx: np.ndarray,
        comm_idx: np.ndarray,
        matcher_mode: str,
        use_topk: bool,
    ) -> Dict:
        comm_idx = np.sort(np.asarray(comm_idx, dtype=np.int32))
        sense_idx = np.sort(np.asarray(sense_idx, dtype=np.int32))
        if len(comm_idx) == 0:
            return {"online_scheduling_time_s": 0.0, "delivery_ratio": 0.0}
        comm_map = {int(sat): local for local, sat in enumerate(comm_idx)}
        sense_local = np.asarray([comm_map[int(sat)] for sat in sense_idx if int(sat) in comm_map], dtype=np.int32)
        q_region = np.zeros((len(TASK_REGIONS), len(comm_idx)), dtype=np.float64)
        e = np.full(len(comm_idx), cfg.E_MAX, dtype=np.float64)
        delivered = 0.0
        delivered_by_region = np.zeros(len(TASK_REGIONS), dtype=np.float64)
        generated = 0.0
        generated_by_region = np.zeros(len(TASK_REGIONS), dtype=np.float64)
        dropped = 0.0
        remaining_task = cfg.TASK_DATA_MB
        remaining_by_region = cfg.TASK_DATA_MB * dl.task_region_weights.astype(np.float64, copy=True)
        online_s = 0.0

        def task_priority_local(idx: np.ndarray) -> np.ndarray:
            idx_arr = np.asarray(idx, dtype=np.int64)
            if idx_arr.size == 0:
                return np.zeros(0, dtype=np.float64)
            q_sub = q_region[:, idx_arr]
            q_total_sub = q_sub.sum(axis=0)
            weighted = (q_sub * dl.task_region_weights[:, None]).sum(axis=0)
            return np.clip(
                weighted / np.maximum(q_total_sub, 1e-9) / max(float(dl.task_region_weights.max()), 1e-9),
                0.0,
                1.0,
            )

        def future_gs_local(ti: int, idx: np.ndarray) -> np.ndarray:
            idx_arr = np.asarray(idx, dtype=np.int64)
            if idx_arr.size == 0:
                return np.zeros(0, dtype=np.float64)
            t_end = min(int(ti) + max(1, int(cfg.WINDOW_LOOKAHEAD)), dl.T)
            future = dl.vis_gs[int(ti):t_end, comm_idx[idx_arr]]
            return np.clip(future.sum(axis=0).astype(np.float64) / max(float(t_end - int(ti)), 1.0), 0.0, 1.0)

        def adaptive_k_local(ti: int, s_local: int, q_total_local: np.ndarray) -> int:
            raw = (
                float(cfg.ADAPTIVE_K_BASE)
                + float(cfg.ADAPTIVE_K_ALPHA) * float(np.clip(q_total_local[s_local] / max(cfg.Q_MAX_COMM, 1e-9), 0.0, 1.0))
                + float(cfg.ADAPTIVE_K_BETA) * float(future_gs_local(ti, np.asarray([s_local]))[0])
                + float(cfg.ADAPTIVE_K_GAMMA) * float(task_priority_local(np.asarray([s_local]))[0])
            )
            return int(np.clip(round(raw), int(cfg.ADAPTIVE_K_MIN), int(cfg.ADAPTIVE_K_MAX)))

        def deplete_local(local: int, amount: float) -> Tuple[float, np.ndarray]:
            remaining_amount = max(float(amount), 0.0)
            moved = np.zeros(len(TASK_REGIONS), dtype=np.float64)
            for region_i in np.argsort(-dl.task_region_weights):
                if remaining_amount <= 1e-9:
                    break
                take = min(float(q_region[region_i, local]), remaining_amount)
                if take <= 0.0:
                    continue
                q_region[region_i, local] -= take
                moved[region_i] = take
                remaining_amount -= take
            return float(moved.sum()), moved

        def move_local(sender: int, receiver: int, amount: float) -> float:
            moved_total, moved_by_region = deplete_local(sender, amount)
            if moved_total > 0.0:
                q_region[:, receiver] += moved_by_region
            return moved_total

        def edge_score_local(ti: int, sender: int, cand: np.ndarray, dist: np.ndarray, q_total_local: np.ndarray) -> np.ndarray:
            distance_score = 1.0 - np.clip(dist / max(cfg.ISL_MAX_DIST, 1e-9), 0.0, 1.0)
            source_buffer_pressure = np.full(
                len(cand),
                np.clip(q_total_local[sender] / max(cfg.Q_MAX_COMM, 1e-9), 0.0, 1.0),
                dtype=np.float64,
            )
            receiver_free_capacity = np.clip(
                (cfg.Q_MAX_COMM - q_total_local[cand]) / max(cfg.Q_MAX_COMM, 1e-9),
                0.0,
                1.0,
            )
            gs_direction_score = future_gs_local(ti, cand)
            energy_score = np.clip(np.minimum(e[sender], e[cand]) / max(cfg.E_MAX, 1e-9), 0.0, 1.0)
            priority = np.full(len(cand), float(task_priority_local(np.asarray([sender]))[0]), dtype=np.float64)
            return (
                0.15 * distance_score
                + 0.20 * source_buffer_pressure
                + 0.15 * receiver_free_capacity
                + 0.20 * gs_direction_score
                + 0.15 * energy_score
                + 0.15 * priority
            )

        def downlink_priority_local(ti: int, cand: np.ndarray, q_total_local: np.ndarray) -> np.ndarray:
            task_priority = task_priority_local(cand)
            buffer_pressure = np.clip(q_total_local[cand] / max(cfg.Q_MAX_COMM, 1e-9), 0.0, 1.0)
            t_end = min(int(ti) + max(1, int(cfg.WINDOW_LOOKAHEAD)), dl.T)
            future = dl.vis_gs[int(ti):t_end, comm_idx[cand]]
            remaining_window = future.sum(axis=0).astype(np.float64) + 1.0
            urgency_raw = q_total_local[cand] / np.maximum(remaining_window, 1e-9)
            window_urgency = urgency_raw / max(float(np.max(urgency_raw)), 1e-9)
            energy_score = np.clip(e[cand] / max(cfg.E_MAX, 1e-9), 0.0, 1.0)
            return 0.35 * task_priority + 0.30 * buffer_pressure + 0.20 * window_urgency + 0.15 * energy_score

        for t in steps:
            ecl = dl.eclipse[t, comm_idx]
            e = np.clip(e + np.where(ecl, 0.0, e_solar) - e_base, 0.0, cfg.E_MAX)
            avail = e > cfg.E_MELT * cfg.E_WARN_RATIO
            q_total = q_region.sum(axis=0)

            if remaining_task > 0.0 and len(sense_local) > 0:
                sensor_step_cap = np.full(len(comm_idx), per_sat_sensor_cap, dtype=np.float64)
                for region_i, region_weight in enumerate(dl.task_region_weights):
                    if remaining_by_region[region_i] <= 0.0:
                        continue
                    can_sense = sense_local[
                        avail[sense_local]
                        & dl.task_coverage_by_region[region_i, t, comm_idx[sense_local]]
                        & (sensor_step_cap[sense_local] > 1e-9)
                    ]
                    if len(can_sense) == 0:
                        continue
                    step_budget = min(
                        task_release * float(region_weight),
                        remaining_by_region[region_i],
                    )
                    per_sat_cap = np.minimum(
                        np.maximum(cfg.Q_MAX_COMM - q_total[can_sense], 0.0),
                        sensor_step_cap[can_sense],
                    )
                    alloc = SimulationEngine._fair_allocate(step_budget, per_sat_cap)
                    used = alloc > 1e-9
                    if np.any(used):
                        target = can_sense[used]
                        stored = alloc[used]
                        q_region[region_i, target] += stored
                        q_total[target] += stored
                        sensor_step_cap[target] = np.maximum(0.0, sensor_step_cap[target] - stored)
                        actual_generated = float(stored.sum())
                        generated += actual_generated
                        generated_by_region[region_i] += actual_generated
                        remaining_by_region[region_i] = max(
                            0.0,
                            float(remaining_by_region[region_i]) - actual_generated,
                        )
                        remaining_task -= actual_generated
                        e[target] = np.maximum(0.0, e[target] - e_sense)

            sched_t0 = time.perf_counter()
            q_total = q_region.sum(axis=0)
            senders = np.where(avail & (q_total > 1e-9))[0]
            receivers_allowed = avail & (q_total < cfg.Q_MAX_COMM - 1e-9)
            rows: List[int] = []
            cols: List[int] = []
            scores: List[float] = []
            pos_t = dl.pos_icrf[t].astype(np.float64, copy=False)

            if len(senders) > 0 and int(receivers_allowed.sum()) > 1:
                if use_topk:
                    for s in senders:
                        sat = int(comm_idx[s])
                        cand_global = dl.isl_neighbor_idx[t, sat]
                        cand_dist = dl.isl_neighbor_dist[t, sat]
                        cand_local = np.asarray([comm_map.get(int(c), -1) for c in cand_global], dtype=np.int32)
                        valid_local = cand_local >= 0
                        cand_local = cand_local[valid_local]
                        cand_dist = cand_dist[valid_local]
                        if len(cand_local) == 0:
                            continue
                        valid = (
                            (cand_local != s)
                            & receivers_allowed[cand_local]
                            & np.isfinite(cand_dist)
                            & (cand_dist > 0.0)
                            & (cand_dist < cfg.ISL_MAX_DIST)
                        )
                        cand_local = cand_local[valid]
                        cand_dist = cand_dist[valid]
                        if len(cand_local) == 0:
                            continue
                        k_s = adaptive_k_local(int(t), int(s), q_total)
                        if len(cand_local) > k_s:
                            cand_local = cand_local[:k_s]
                            cand_dist = cand_dist[:k_s]
                        p0 = np.repeat(pos_t[sat][None, :], len(cand_local), axis=0)
                        los = los_engine._los_mask(p0, pos_t[comm_idx[cand_local]])
                        cand_local = cand_local[los]
                        cand_dist = cand_dist[los]
                        if len(cand_local) == 0:
                            continue
                        score = edge_score_local(int(t), int(s), cand_local, cand_dist, q_total)
                        for c, sc in zip(cand_local, score):
                            rows.append(int(s))
                            cols.append(int(c))
                            scores.append(float(sc))
                else:
                    recv_local = np.where(receivers_allowed)[0]
                    if len(recv_local) > 0:
                        tree = cKDTree(pos_t[comm_idx[recv_local]])
                        try:
                            neigh_lists = tree.query_ball_point(
                                pos_t[comm_idx[senders]],
                                r=cfg.ISL_MAX_DIST,
                                workers=-1,
                            )
                        except TypeError:
                            neigh_lists = tree.query_ball_point(
                                pos_t[comm_idx[senders]],
                                r=cfg.ISL_MAX_DIST,
                            )
                        for row_i, s in enumerate(senders):
                            local = neigh_lists[row_i]
                            if len(local) == 0:
                                continue
                            cand_local = recv_local[np.asarray(local, dtype=np.int32)]
                            valid = cand_local != s
                            cand_local = cand_local[valid]
                            if len(cand_local) == 0:
                                continue
                            cand_dist = np.linalg.norm(pos_t[comm_idx[cand_local]] - pos_t[comm_idx[s]], axis=1)
                            valid = (cand_dist > 0.0) & (cand_dist < cfg.ISL_MAX_DIST)
                            cand_local = cand_local[valid]
                            cand_dist = cand_dist[valid]
                            if len(cand_local) == 0:
                                continue
                            p0 = np.repeat(pos_t[comm_idx[s]][None, :], len(cand_local), axis=0)
                            los = los_engine._los_mask(p0, pos_t[comm_idx[cand_local]])
                            cand_local = cand_local[los]
                            cand_dist = cand_dist[los]
                            if len(cand_local) == 0:
                                continue
                            score = edge_score_local(int(t), int(s), cand_local, cand_dist, q_total)
                            for c, sc in zip(cand_local, score):
                                rows.append(int(s))
                                cols.append(int(c))
                                scores.append(float(sc))

            if rows:
                rows_arr = np.asarray(rows, dtype=np.int32)
                cols_arr = np.asarray(cols, dtype=np.int32)
                scores_arr = np.asarray(scores, dtype=np.float64)
                pairs: List[Tuple[int, int]] = []
                receiver_unique = np.unique(cols_arr)
                sender_unique = np.unique(rows_arr)
                n_cells = len(sender_unique) * len(receiver_unique)
                if matcher_mode == "hungarian" and n_cells <= cfg.HUNGARIAN_MAX_CELLS:
                    row_pos = {int(s): i for i, s in enumerate(sender_unique)}
                    col_pos = {int(c): i for i, c in enumerate(receiver_unique)}
                    score_mat = np.full((len(sender_unique), len(receiver_unique)), -1e9, dtype=np.float64)
                    for s, c, sc in zip(rows_arr, cols_arr, scores_arr):
                        i = row_pos[int(s)]
                        j = col_pos[int(c)]
                        if sc > score_mat[i, j]:
                            score_mat[i, j] = sc
                    rr, cc = linear_sum_assignment(-score_mat)
                    for r, c in zip(rr, cc):
                        if score_mat[r, c] > -1e8:
                            pairs.append((int(sender_unique[r]), int(receiver_unique[c])))
                else:
                    used_s = set()
                    used_r = set()
                    for edge_i in np.argsort(-scores_arr):
                        s = int(rows_arr[edge_i])
                        r = int(cols_arr[edge_i])
                        if s in used_s or r in used_r:
                            continue
                        used_s.add(s)
                        used_r.add(r)
                        pairs.append((s, r))
                q_total = q_region.sum(axis=0)
                for s, r in pairs:
                    tx = min(q_total[s], tx_isl, cfg.Q_MAX_COMM - q_total[r])
                    if tx > 0.0:
                        moved = move_local(s, r, tx)
                        if moved <= 0.0:
                            continue
                        q_total[s] -= moved
                        q_total[r] += moved
                        e[s] = max(0.0, e[s] - e_tx)
                        e[r] = max(0.0, e[r] - e_rx)

            q_total = q_region.sum(axis=0)
            cand = np.where(avail & dl.vis_gs[t, comm_idx] & (q_total > 1e-9))[0]
            if len(cand) > 0:
                priority = downlink_priority_local(int(t), cand, q_total)
                ant_used = 0
                for local in cand[np.argsort(-priority)]:
                    if ant_used >= cfg.GS_ANTENNAS:
                        break
                    tx = min(q_total[local], tx_sgl)
                    if tx > 0.0:
                        actual_tx, moved_by_region = deplete_local(int(local), tx)
                        if actual_tx <= 0.0:
                            continue
                        q_total[local] -= actual_tx
                        delivered += actual_tx
                        delivered_by_region += moved_by_region
                        e[local] = max(0.0, e[local] - e_tx)
                        ant_used += 1
            online_s += time.perf_counter() - sched_t0

        return {
            "online_scheduling_time_s": float(online_s),
            "delivery_ratio": float(min(delivered / max(cfg.TASK_DATA_MB, 1e-9), 1.0)),
            "sensing_progress": float(min(generated / max(cfg.TASK_DATA_MB, 1e-9), 1.0)),
            "generated_mb": float(generated),
            "delivered_mb": float(delivered),
            "dropped_mb": float(dropped),
        }

    series = {
        item["method_id"]: {
            "method_id": item["method_id"],
            "label": item["label"],
            "activation": item["activation"],
            "candidate_graph": item["candidate_graph"],
            "matcher": item["matcher"],
            "color": _method_color(item["method_id"]),
            "online_scheduling_time_s": [],
            "total_runtime_s": [],
            "activation_ratio": [],
            "delivery_ratio": [],
        }
        for item in methods
    }

    for n in node_targets:
        pools = [choose(all_idx, int(n)) for _ in range(repeats)]
        for item in methods:
            online_samples: List[float] = []
            delivery_samples: List[float] = []
            active_ratio_samples: List[float] = []
            for pool in pools:
                sense_idx, comm_idx = activation_for_method(pool, item["activation"])
                stats = benchmark(sense_idx, comm_idx, item["matcher_mode"], item["use_topk"])
                online_samples.append(float(stats["online_scheduling_time_s"]))
                delivery_samples.append(float(stats["delivery_ratio"]))
                active_ratio_samples.append(float(len(comm_idx) / max(len(pool), 1)))
            online = float(np.median(online_samples))
            activation_scale_s = (
                float(nsga_runtime_s) * (float(n) / max(float(n_total), 1.0))
                if item["activation"] == "NSGA-III"
                else 0.0
            )
            series[item["method_id"]]["online_scheduling_time_s"].append(online)
            series[item["method_id"]]["total_runtime_s"].append(float(online + activation_scale_s))
            series[item["method_id"]]["activation_ratio"].append(float(np.median(active_ratio_samples)))
            series[item["method_id"]]["delivery_ratio"].append(float(np.median(delivery_samples)))
        print(f"  scale point nodes={int(n)} completed")

    return {
        "nodes": [int(x) for x in node_targets],
        "methods": list(series.values()),
        "steps_used": int(len(steps)),
        "repeats": int(repeats),
        "delivery_ratio_definition": "delivered/TASK_DATA_MB within sampled scale benchmark",
        "nsga_runtime_scale_model": "linear_scale_from_measured_full_run",
        "measured_full_nsga_s": float(nsga_runtime_s),
    }


def save_scalability_outputs(out_dir: Path, scale: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = scale
    print("  skipped scalability result tables")


class Visualizer:
    def __init__(self, out_dir: Path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        self.plt = plt
        self.sns = sns
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", rc={"axes.unicode_minus": False})
        plt.rcParams["mathtext.fontset"] = "stix"

    def _save(self, name: str) -> None:
        path = self.out_dir / name
        try:
            self.plt.tight_layout()
        except Exception:
            pass
        self.plt.savefig(path, dpi=240, bbox_inches="tight")
        self.plt.close()
        print(f"  saved {path}")

    def _project2d(self, pos_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        az, el = np.radians(30.0), np.radians(35.0)
        x, y, z = pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2]
        px = -x * np.sin(az) + y * np.cos(az)
        py = (-x * np.cos(az) - y * np.sin(az)) * np.sin(el) + z * np.cos(el)
        return px, py

    def _draw_globe(self, ax) -> None:
        plt = self.plt
        theta = np.linspace(0.0, 2.0 * np.pi, 240)
        gx = R_EARTH * np.cos(theta)
        gy = R_EARTH * np.sin(theta)
        ax.plot(gx, gy, color="#88BBDD", lw=1.4, zorder=1)
        ax.fill(gx, gy, color="#E8F4FD", alpha=0.32, zorder=0)
        for lat_d in range(-60, 90, 30):
            lat = np.radians(lat_d)
            lon = np.linspace(0.0, 2.0 * np.pi, 120)
            pts = np.column_stack(
                [
                    R_EARTH * np.cos(lat) * np.cos(lon),
                    R_EARTH * np.cos(lat) * np.sin(lon),
                    R_EARTH * np.sin(lat) * np.ones_like(lon),
                ]
            )
            px, py = self._project2d(pts)
            ax.plot(px, py, color="#CCDDEE", lw=0.4, alpha=0.65, zorder=0)
        ax.set_aspect("equal")
        ax.axis("off")

    def _los_pair_mask(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        vec = p1 - p0
        dist2 = np.sum(vec * vec, axis=1)
        dist = np.sqrt(np.maximum(dist2, 1e-12))
        s = -np.sum(p0 * vec, axis=1) / np.maximum(dist2, 1e-12)
        cross = np.cross(p0, p1)
        d_min = np.linalg.norm(cross, axis=1) / np.maximum(dist, 1e-9)
        blocked = (s > 0.0) & (s < 1.0) & (d_min < R_EARTH + ATM_HEIGHT)
        return ~blocked

    def _empty_figure(self, name: str, title: str, message: str) -> None:
        plt = self.plt
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        self._save(name)

    def fig1_topology(self, dl: DataLoader, best_genes: np.ndarray) -> None:
        plt = self.plt
        mid = dl.T // 2
        pos = dl.pos_ecef[mid].astype(np.float64, copy=False)
        sense_on = best_genes[: dl.N].astype(bool)
        comm_on = best_genes[dl.N :].astype(bool)
        inactive = ~comm_on

        px, py = self._project2d(pos)
        gs_px, gs_py = self._project2d(GS_ECEF.reshape(1, 3))
        fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.5))

        ax = axes[0]
        self._draw_globe(ax)
        ax.scatter(gs_px, gs_py, color="red", s=245, marker="*", zorder=15, label="Ground station")
        step = max(1, dl.N // 650)
        cover = dl.task_coverage[mid]
        ax.scatter(px[::step], py[::step], c="#6BAED6", s=10, alpha=0.52, zorder=3, label=f"Candidate relay ({dl.N})")
        if np.any(cover):
            cover_idx = np.where(cover)[0]
            step_cover = max(1, len(cover_idx) // 220)
            ax.scatter(
                px[cover_idx][::step_cover],
                py[cover_idx][::step_cover],
                c="darkorange",
                s=28,
                alpha=0.82,
                edgecolors="saddlebrown",
                linewidths=0.3,
                zorder=5,
                label=f"Task-covering ({len(cover_idx)})",
            )
        for i in np.where(cover)[0][:30]:
            ax.plot([gs_px[0], px[i]], [gs_py[0], py[i]], color="lightcoral", alpha=0.12, lw=0.5, zorder=2)
        ax.set_title("(a) Full candidate constellation", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.set_xlim(-10500, 10500)
        ax.set_ylim(-9500, 9500)

        ax = axes[1]
        self._draw_globe(ax)
        ax.scatter(gs_px, gs_py, color="red", s=245, marker="*", zorder=15, label="Ground station")
        if np.any(inactive):
            inactive_idx = np.where(inactive)[0]
            step_inactive = max(1, len(inactive_idx) // 400)
            ax.scatter(px[inactive_idx][::step_inactive], py[inactive_idx][::step_inactive], c="lightgray", s=5, alpha=0.18, zorder=2)
        comm_only = comm_on & ~sense_on
        if np.any(comm_only):
            comm_idx = np.where(comm_only)[0]
            step_comm = max(1, len(comm_idx) // 180)
            ax.scatter(
                px[comm_idx][::step_comm],
                py[comm_idx][::step_comm],
                c="royalblue",
                s=18,
                alpha=0.72,
                edgecolors="navy",
                linewidths=0.3,
                zorder=6,
                label=f"Active relay ({len(comm_idx)})",
            )
        if np.any(sense_on):
            sense_idx = np.where(sense_on)[0]
            ax.scatter(
                px[sense_idx],
                py[sense_idx],
                c="forestgreen",
                s=72,
                marker="^",
                edgecolors="darkgreen",
                linewidths=0.8,
                zorder=8,
                label=f"Active sensing ({len(sense_idx)})",
            )

        active_sense = np.where(sense_on)[0]
        drawn = 0
        for sat in active_sense[:30]:
            neigh = dl.isl_neighbor_idx[mid, sat]
            for target in neigh:
                if drawn >= 24:
                    break
                if target >= 0 and comm_on[target]:
                    ax.plot([px[sat], px[target]], [py[sat], py[target]], color="limegreen", alpha=0.35, lw=0.75, zorder=3)
                    drawn += 1
            if drawn >= 24:
                break
        ax.set_title("(b) Sparse dual-layer backbone", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.set_xlim(-10500, 10500)
        ax.set_ylim(-9500, 9500)

        fig.suptitle("Fig 1. Sparse heterogeneous constellation topology snapshot", fontsize=14, fontweight="bold", y=1.01)
        self._save("Fig1_Topology.png")

    def fig2_matching(self, dl: DataLoader, cfg: Config, best_genes: np.ndarray) -> None:
        plt = self.plt
        sense_on = best_genes[: dl.N].astype(bool)
        comm_on = best_genes[dl.N :].astype(bool)
        sense_all = np.where(sense_on)[0]
        comm_all = np.where(comm_on)[0]
        if len(sense_all) == 0 or len(comm_all) == 0:
            self._empty_figure("Fig2_Matching.png", "Fig 2. Normalized-score bipartite matching", "No active matching candidates.")
            return

        show_sense = sense_all[: min(6, len(sense_all))]
        show_comm = comm_all[: min(6, len(comm_all))]
        ns, nc = len(show_sense), len(show_comm)
        best_t = 0
        best_matches: List[Tuple[int, int, float]] = []
        best_count = -1
        for t in np.linspace(0, dl.T - 1, min(50, dl.T), dtype=int):
            ps = dl.pos_icrf[t, show_sense].astype(np.float64, copy=False)
            pc = dl.pos_icrf[t, show_comm].astype(np.float64, copy=False)
            dist = np.linalg.norm(ps[:, None, :] - pc[None, :, :], axis=2)
            score = np.full((ns, nc), -1e9, dtype=np.float64)
            for i in range(ns):
                p0 = np.repeat(ps[i][None, :], nc, axis=0)
                los = self._los_pair_mask(p0, pc)
                valid = (show_comm != show_sense[i]) & los & (dist[i] > 0.0) & (dist[i] < cfg.ISL_MAX_DIST)
                d_norm = dist[i] / max(cfg.ISL_MAX_DIST, 1e-9)
                score[i, valid] = (1.0 - d_norm[valid]) * 0.4 + 0.4 + 0.2
            rr, cc = linear_sum_assignment(-score)
            matches = [(int(r), int(c), float(score[r, c])) for r, c in zip(rr, cc) if score[r, c] > -1e8]
            if len(matches) > best_count:
                best_count = len(matches)
                best_t = int(t)
                best_matches = matches

        fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
        y_max = max(ns, nc) * 1.2 + 0.5
        rp = [(0.3, i * 1.2 + 0.3) for i in range(ns)]
        cp = [(2.7, i * 1.2 + 0.3 + (ns - nc) * 0.6) for i in range(nc)]
        titles = ["(a) Original complete bipartite graph", "(b) Normalized-score optimal matching"]

        for ax, title in zip(axes, titles):
            for i, (x, y) in enumerate(rp):
                circle = plt.Circle((x, y), 0.18, color="#E74C3C", ec="darkred", lw=1.2, zorder=5)
                ax.add_patch(circle)
                ax.text(x, y, f"$S_{{{i + 1}}}$", va="center", ha="center", fontsize=10, fontweight="bold", color="white", zorder=6)
            for j, (x, y) in enumerate(cp):
                rect = plt.Rectangle((x - 0.18, y - 0.18), 0.36, 0.36, color="#3498DB", ec="darkblue", lw=1.2, zorder=5)
                ax.add_patch(rect)
                ax.text(x, y, f"$C_{{{j + 1}}}$", va="center", ha="center", fontsize=10, fontweight="bold", color="white", zorder=6)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, y_max)
            ax.set_aspect("equal")

        for rx, ry in rp:
            for cx, cy in cp:
                axes[0].plot([rx, cx], [ry, cy], color="gray", alpha=0.35, ls="--", lw=0.8)
        for r, c, sc in best_matches:
            axes[1].plot([rp[r][0], cp[c][0]], [rp[r][1], cp[c][1]], color="#E74C3C", linewidth=2.8, zorder=3)
            mx = (rp[r][0] + cp[c][0]) / 2.0
            my = (rp[r][1] + cp[c][1]) / 2.0
            axes[1].text(
                mx,
                my + 0.12,
                f"{sc:.3f}",
                fontsize=7.5,
                color="#8B0000",
                fontweight="bold",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="lightyellow", alpha=0.8, lw=0.5),
                zorder=7,
            )
        fig.suptitle(f"Fig 2. Normalized-score Hungarian matching graph\n(t={best_t}, matched pairs={max(best_count, 0)})", fontsize=13, fontweight="bold")
        self._save("Fig2_Matching.png")

    def fig4_pareto_convergence(self, F: np.ndarray, history: List[Dict]) -> None:
        plt = self.plt
        fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.2))
        gens = np.asarray([h.get("gen", i + 1) for i, h in enumerate(history)], dtype=float)
        sense = np.asarray([h.get("sense_count", 0) for h in history], dtype=float)
        relay = np.asarray([h.get("comm_count", 0) for h in history], dtype=float)
        total = np.asarray(
            [h.get("total_active", h.get("sense_count", 0) + h.get("comm_count", 0)) for h in history],
            dtype=float,
        )
        near_best_min = np.asarray(
            [
                h.get(
                    "near_best_min_active",
                    h.get("total_active", h.get("sense_count", 0) + h.get("comm_count", 0)),
                )
                for h in history
            ],
            dtype=float,
        )
        if len(gens) == 0:
            gens = np.asarray([0.0])
            sense = np.asarray([0.0])
            relay = np.asarray([0.0])
            total = np.asarray([0.0])
            near_best_min = np.asarray([0.0])

        ax.plot(
            gens,
            sense,
            color="#238B45",
            marker="^",
            ms=5.0,
            lw=2.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Active sensing",
        )
        ax.plot(
            gens,
            relay,
            color="#2171B5",
            marker="o",
            ms=5.0,
            lw=2.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Active relay",
        )
        ax.plot(
            gens,
            total,
            color="#D7301F",
            marker="s",
            ms=4.8,
            lw=2.0,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label="Total active",
        )
        if len(history) > 0 and np.any(near_best_min > 0):
            ax.plot(
                gens,
                near_best_min,
                color="#756BB1",
                linestyle=(0, (2, 2)),
                lw=1.4,
                alpha=0.85,
                label="Min active near best",
            )
        ax.set_xlabel("NSGA-III generation", fontsize=11)
        ax.set_ylabel("Active satellites", fontsize=11)
        ax.set_title("Activation-scale convergence", fontsize=11.5, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.24)
        if len(gens) > 0:
            ax.axvline(gens[-1], color="#7A7A7A", lw=1.0, ls=(0, (3, 3)), alpha=0.9)
            ax.annotate(
                "stop",
                xy=(gens[-1], total[-1]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8.8,
                color="#4D4D4D",
            )
            ax.text(
                0.98,
                0.94,
                f"Final: S={int(sense[-1])}, R={int(relay[-1])}, Total={int(total[-1])}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9.0,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BDBDBD", alpha=0.92),
            )
            ceiling_hits = [
                h for h in history
                if bool(h.get("ceiling_reached", False))
            ]
            if ceiling_hits:
                hit_gen = int(ceiling_hits[0].get("gen", 0))
                note = f"Utility ceiling reached at generation {hit_gen}"
                if hit_gen == 1:
                    note += "; tie-break tracks sparse activation"
                ax.text(
                    0.02,
                    0.96,
                    note,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.7,
                    color="#4D4D4D",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BDBDBD", alpha=0.92),
                )
            elif len(total) > 1 and float(np.ptp(total)) < 1e-9:
                ax.text(
                    0.02,
                    0.96,
                    "Activation unchanged; finite-task/queue bottleneck caps utility",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.7,
                    color="#4D4D4D",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BDBDBD", alpha=0.92),
                )
        y_min = min(float(np.min(sense)), float(np.min(relay)), float(np.min(total)))
        y_max = max(float(np.max(sense)), float(np.max(relay)), float(np.max(total)), float(np.max(near_best_min)))
        pad = max(2.0, (y_max - y_min) * 0.12)
        ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
        ax.legend(loc="best", fontsize=8.8, framealpha=0.96, facecolor="white", edgecolor="#BDBDBD")
        fig.suptitle("Fig. 4  Activation-scale convergence of NSGA-III", fontsize=13.5, fontweight="bold", y=0.98)
        self._save("Fig4_Pareto_Convergence.png")

    def fig4_activation_utility_3d(self, X: np.ndarray, F: np.ndarray, dl: DataLoader, best_gene: np.ndarray) -> None:
        plt = self.plt
        sense_count = np.sum(X[:, : dl.N] > 0.5, axis=1).astype(float)
        relay_count = np.sum(X[:, dl.N :] > 0.5, axis=1).astype(float)
        total_active = sense_count + relay_count
        z_util = np.asarray(-F[:, 0], dtype=float)
        z_pct = z_util / max(float(np.max(z_util)), 1e-9) * 100.0
        best_sense = float(np.sum(best_gene[: dl.N]))
        best_comm = float(np.sum(best_gene[dl.N :]))
        best_i = select_best_objective_index(F)
        best_util = float(z_pct[best_i])
        sizes = 35.0 + 95.0 * (total_active - np.min(total_active)) / max(float(np.ptp(total_active)), 1e-9)

        fig, ax = plt.subplots(figsize=(8.8, 6.4))
        sc = ax.scatter(
            relay_count,
            sense_count,
            c=z_pct,
            s=sizes,
            cmap="viridis",
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        ax.scatter(
            best_comm,
            best_sense,
            c="#FFD02A",
            marker="*",
            s=310,
            edgecolors="black",
            linewidths=1.0,
            zorder=8,
            label="Best trade-off",
        )
        ax.annotate(
            f"Best: S={int(best_sense)}, R={int(best_comm)}, U={best_util:.1f}%",
            xy=(best_comm, best_sense),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=9.2,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BDBDBD", alpha=0.94),
            arrowprops=dict(arrowstyle="->", color="#4D4D4D", lw=0.9),
            zorder=9,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.78)
        cbar.set_label("Delivery utility U (%)", fontsize=10)
        ax.set_xlabel("Active relay satellites", fontsize=11)
        ax.set_ylabel("Active sensing satellites", fontsize=11)
        ax.set_title("Fig. 4-S  Activation trade-off distribution", fontsize=12.5, fontweight="bold", pad=12)
        ax.grid(True, ls="--", alpha=0.22)
        ax.legend(loc="best", fontsize=8.8, framealpha=0.96, facecolor="white", edgecolor="#BDBDBD")

        utility_spread = float(np.ptp(z_pct)) if len(z_pct) else 0.0
        if utility_spread < 1.0 and len(z_pct) > 0:
            ax.text(
                0.02,
                0.96,
                "Utility saturated near 100%; compare activation scale",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.0,
                color="#4D4D4D",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BDBDBD", alpha=0.92),
            )
        ax.text(
            0.02,
            0.04,
            "Marker size: total active satellites",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#4D4D4D",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#BDBDBD", alpha=0.93, lw=0.7),
        )
        self._save("Fig4_Activation_Utility_3D.png")

    def fig5_complexity_overall(self, timing: Dict) -> None:
        plt = self.plt
        nodes = np.asarray(timing.get("nodes", []), dtype=float)
        if len(nodes) == 0:
            self._empty_figure("Fig5_Complexity.png", "Fig. 5 Runtime scalability comparison", "No timing data.")
            return
        t_full = np.maximum(np.asarray(timing.get("t_full_greedy", []), dtype=float), 1e-6)
        t_graph = np.maximum(np.asarray(timing.get("t_graph_only", []), dtype=float), 1e-6)
        dual_key = "dual_total_s" if "dual_total_s" in timing else "t_dual"
        t_dual = np.maximum(np.asarray(timing.get(dual_key, []), dtype=float), 1e-6)

        fig, ax = plt.subplots(figsize=(11.4, 6.2))
        ax.axhspan(1e-3, 1.0, color="#EDF8E9", alpha=0.28, zorder=0)
        line1, = ax.plot(
            nodes,
            t_full,
            color="#2B2B2B",
            marker="s",
            linestyle=(0, (4, 2)),
            lw=1.9,
            ms=7.2,
            mfc="white",
            mew=1.8,
            label="Full activation + greedy",
            zorder=6,
        )
        line2, = ax.plot(
            nodes,
            t_graph,
            color="#E34A33",
            marker="d",
            linestyle="-",
            lw=1.9,
            ms=7.2,
            mfc="none",
            mew=1.8,
            label="Graph matching only",
            zorder=5,
        )
        line3, = ax.plot(
            nodes,
            t_dual,
            color="#2171B5",
            marker="o",
            linestyle="-",
            lw=2.1,
            ms=7.8,
            mfc="none",
            mew=1.9,
            label="Dual-layer end-to-end",
            zorder=7,
        )
        ax.set_xlim(max(0.0, nodes.min() * 0.85), nodes.max() * 1.03)
        ax.set_yscale("log")
        y_min = min(float(np.min(t_full)), float(np.min(t_graph)), float(np.min(t_dual)))
        y_max = max(float(np.max(t_full)), float(np.max(t_graph)), float(np.max(t_dual)))
        ax.set_ylim(max(1e-6, y_min * 0.55), y_max * 1.35)
        ax.set_xlabel("Constellation node count", fontsize=11)
        ax.set_ylabel("Algorithm runtime (s, log scale)", fontsize=11)
        ax.set_title("Fig. 5  Runtime Scalability Comparison", fontsize=13, fontweight="bold")
        ax.grid(True, which="major", ls="--", alpha=0.24)
        ax.grid(True, which="minor", ls=":", alpha=0.12)
        ax.legend(handles=[line1, line2, line3], fontsize=9.5, loc="upper left", framealpha=1.0, facecolor="white", edgecolor="#444444")
        ax.text(
            0.985,
            0.035,
            "Dual-layer includes NSGA-III optimization + online scheduling",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            color="#4D4D4D",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BDBDBD", alpha=0.90),
        )

        if len(nodes) >= 4:
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

                inset_len = min(4, len(nodes))
                axins = inset_axes(ax, width="34%", height="34%", loc="center right", bbox_to_anchor=(0.02, 0.05, 0.82, 0.88), bbox_transform=ax.transAxes, borderpad=0.8)
                xs = nodes[:inset_len]
                axins.plot(
                    xs,
                    t_full[:inset_len],
                    color="#2B2B2B",
                    marker="s",
                    linestyle=(0, (4, 2)),
                    lw=1.5,
                    ms=5.5,
                    mfc="white",
                    mew=1.4,
                )
                axins.plot(
                    xs,
                    t_graph[:inset_len],
                    color="#E34A33",
                    marker="d",
                    linestyle="-",
                    lw=1.5,
                    ms=5.5,
                    mfc="none",
                    mew=1.4,
                )
                axins.plot(
                    xs,
                    t_dual[:inset_len],
                    color="#2171B5",
                    marker="o",
                    linestyle="-",
                    lw=1.6,
                    ms=5.8,
                    mfc="none",
                    mew=1.5,
                )
                small_min = min(float(np.min(t_full[:inset_len])), float(np.min(t_graph[:inset_len])), float(np.min(t_dual[:inset_len])))
                small_max = max(float(np.max(t_full[:inset_len])), float(np.max(t_graph[:inset_len])), float(np.max(t_dual[:inset_len])))
                axins.set_xlim(xs.min(), xs.max())
                axins.set_yscale("log")
                axins.set_ylim(max(1e-6, small_min * 0.8), small_max * 1.12)
                axins.grid(True, which="major", ls=":", alpha=0.16)
                axins.grid(True, which="minor", ls=":", alpha=0.08)
                axins.tick_params(axis="both", labelsize=8.5, direction="in", top=True, right=True, length=3)
                axins.set_title("Small-scale zoom (log y)", fontsize=9.5, pad=2)
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#333333", lw=1.0, ls=(0, (4, 4)))
            except Exception:
                pass
        self._save("Fig5_Complexity.png")

    def fig6_baseline_comparison(self, compare: Dict) -> None:
        plt = self.plt
        methods = compare.get("methods", []) if compare else []
        if not methods:
            self._empty_figure("Fig6_Baseline_Comparison.png", "Fig. 6 Comparative experimental results", "No baseline data.")
            return
        labels = [m["label"] for m in methods]
        colors = [m.get("color", "#777777") for m in methods]
        delivered = np.asarray([m.get("delivery_ratio", 0.0) for m in methods], dtype=float)
        queue = np.asarray([m.get("avg_queue_backlog_mb", 0.0) for m in methods], dtype=float)
        energy = np.asarray([m.get("total_energy_demand", 0.0) for m in methods], dtype=float)
        runtime = np.asarray([m.get("runtime_s", 0.0) for m in methods], dtype=float)
        metrics = [
            ("Delivered data", delivered / max(float(delivered[0]), 1e-9), "Higher is better"),
            ("Average queue backlog", queue / max(float(queue[0]), 1e-9), "Lower is better"),
            ("Total energy demand", energy / max(float(energy[0]), 1e-9), "Lower is better"),
            ("Runtime", runtime / max(float(runtime[0]), 1e-9), "Lower is better"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))
        axes = axes.ravel()
        x = np.arange(len(labels), dtype=float)
        for ax, (title, vals, note) in zip(axes, metrics):
            bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.55, alpha=0.92)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9.5, rotation=8, ha="right")
            ax.set_title(title, fontsize=11.5, fontweight="bold")
            ax.grid(True, axis="y", ls="--", alpha=0.22)
            y_top = max(1.25, float(np.max(vals)) * 1.18)
            ax.set_ylim(0.0, y_top)
            ax.axhline(1.0, color="#C51B29", lw=1.1, ls=(0, (3.5, 3.5)), alpha=0.8)
            for bar, val, color in zip(bars, vals, colors):
                ax.text(bar.get_x() + bar.get_width() / 2.0, min(bar.get_height() + y_top * 0.035, y_top * 0.97), f"{val:.2f}", ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
            ax.text(0.03, 0.93, f"{note}; full+greedy = 1", transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color="#7A1F1F")
        fig.suptitle("Fig. 6  Comparative Experimental Results", fontsize=14, fontweight="bold", y=0.995)
        self._save("Fig6_Baseline_Comparison.png")

    def _plot_scale_metric(
        self,
        scale: Dict,
        metric_key: str,
        ylabel: str,
        title: str,
        filename: str,
        log_y: bool = False,
    ) -> None:
        plt = self.plt
        nodes = np.asarray(scale.get("nodes", []), dtype=float)
        methods = scale.get("methods", []) if scale else []
        if len(nodes) == 0 or not methods:
            self._empty_figure(filename, title, "No scale experiment data.")
            return
        fig, ax = plt.subplots(figsize=(11.0, 6.2))
        markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
        for i, method in enumerate(methods):
            vals = np.asarray(method.get(metric_key, []), dtype=float)
            if len(vals) != len(nodes):
                continue
            linestyle = "-" if method.get("candidate_graph") == "Adaptive-K" else (0, (4, 2))
            ax.plot(
                nodes,
                np.maximum(vals, 1e-9) if log_y else vals,
                marker=markers[i % len(markers)],
                linestyle=linestyle,
                lw=1.8,
                ms=6.4,
                color=method.get("color", "#777777"),
                label=method.get("label", f"M{i + 1}"),
                alpha=0.95,
            )
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Satellite count", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, which="major", ls="--", alpha=0.24)
        ax.grid(True, which="minor", ls=":", alpha=0.12)
        ax.legend(ncol=2, fontsize=8.4, framealpha=0.96, facecolor="white", edgecolor="#BDBDBD")
        self._save(filename)

    def fig1_scale_online_time(self, scale: Dict) -> None:
        self._plot_scale_metric(
            scale,
            "online_scheduling_time_s",
            "Online scheduling time (s)",
            "Fig. 1  Satellite Scale vs Online Scheduling Time",
            "Fig1_Scale_Online_Scheduling_Time.png",
            log_y=True,
        )

    def fig2_scale_total_runtime(self, scale: Dict) -> None:
        self._plot_scale_metric(
            scale,
            "total_runtime_s",
            "Total runtime (s)",
            "Fig. 2  Satellite Scale vs Total Runtime",
            "Fig2_Scale_Total_Runtime.png",
            log_y=True,
        )

    def fig3_scale_activation_ratio(self, scale: Dict) -> None:
        self._plot_scale_metric(
            scale,
            "activation_ratio",
            "Active satellite ratio",
            "Fig. 3  Satellite Scale vs Activation Ratio",
            "Fig3_Scale_Activation_Ratio.png",
            log_y=False,
        )

    def fig4_scale_delivery_ratio(self, scale: Dict) -> None:
        self._plot_scale_metric(
            scale,
            "delivery_ratio",
            "Delivery ratio",
            "Fig. 4  Satellite Scale vs Delivery Ratio",
            "Fig4_Scale_Delivery_Ratio.png",
            log_y=False,
        )

    def fig5_nsga_pareto_front(self, X: np.ndarray, F: np.ndarray, dl: DataLoader) -> None:
        plt = self.plt
        if X.size == 0 or F.size == 0:
            self._empty_figure("Fig5_NSGA_Pareto_Front.png", "Fig. 5 NSGA-III Pareto Front", "No Pareto data.")
            return
        sense = np.sum(X[:, : dl.N] > 0.5, axis=1).astype(float)
        comm = np.sum(X[:, dl.N :] > 0.5, axis=1).astype(float)
        util = np.asarray(-F[:, 0], dtype=float)
        fig, ax = plt.subplots(figsize=(8.4, 6.2))
        sc = ax.scatter(
            sense,
            comm,
            c=util,
            cmap="viridis",
            s=58,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.45,
        )
        best_i = select_best_objective_index(F)
        ax.scatter(
            sense[best_i],
            comm[best_i],
            marker="*",
            s=280,
            c="#FFD02A",
            edgecolors="black",
            linewidths=0.9,
            label="Selected trade-off",
            zorder=8,
        )
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Utility", fontsize=10)
        ax.set_xlabel("Sensing active count", fontsize=11)
        ax.set_ylabel("Communication active count", fontsize=11)
        ax.set_title("Fig. 5  NSGA-III Pareto Front", fontsize=13, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.24)
        ax.legend(loc="best", fontsize=9.0, framealpha=0.96, facecolor="white", edgecolor="#BDBDBD")
        self._save("Fig5_NSGA_Pareto_Front.png")

    def fig6_nsga_convergence(self, history: List[Dict]) -> None:
        plt = self.plt
        if not history:
            self._empty_figure("Fig6_NSGA_Convergence.png", "Fig. 6 NSGA-III Convergence", "No convergence history.")
            return
        gens = np.asarray([h.get("gen", i + 1) for i, h in enumerate(history)], dtype=float)
        best_util = np.asarray([h.get("best_util", 0.0) for h in history], dtype=float)
        active = np.asarray([h.get("total_active", 0.0) for h in history], dtype=float)
        fig, ax1 = plt.subplots(figsize=(9.2, 5.8))
        line1, = ax1.plot(gens, best_util, color="#1F78B4", marker="o", lw=2.0, ms=5.5, label="Best utility")
        ax1.set_xlabel("Generation", fontsize=11)
        ax1.set_ylabel("Best utility", fontsize=11, color="#1F78B4")
        ax1.tick_params(axis="y", labelcolor="#1F78B4")
        ax1.grid(True, ls="--", alpha=0.24)
        ax2 = ax1.twinx()
        line2, = ax2.plot(gens, active, color="#D95F02", marker="s", lw=1.9, ms=5.2, label="Active satellite count")
        ax2.set_ylabel("Active satellite count", fontsize=11, color="#D95F02")
        ax2.tick_params(axis="y", labelcolor="#D95F02")
        lines = [line1, line2]
        ax1.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=9.2, framealpha=0.96, facecolor="white", edgecolor="#BDBDBD")
        fig.suptitle("Fig. 6  NSGA-III Convergence Curve", fontsize=13, fontweight="bold")
        self._save("Fig6_NSGA_Convergence.png")

    def fig7_activation_utility_3d(self, X: np.ndarray, F: np.ndarray, dl: DataLoader, best_gene: np.ndarray) -> None:
        plt = self.plt
        if X.size == 0 or F.size == 0:
            self._empty_figure("Fig7_Activation_Utility_3D.png", "Fig. 7 3D Activation-Utility Distribution", "No activation data.")
            return
        sense = np.sum(X[:, : dl.N] > 0.5, axis=1).astype(float)
        comm = np.sum(X[:, dl.N :] > 0.5, axis=1).astype(float)
        util = np.asarray(-F[:, 0], dtype=float)
        fig = plt.figure(figsize=(9.2, 7.2))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            sense,
            comm,
            util,
            c=util,
            cmap="viridis",
            s=42,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )
        best_i = select_best_objective_index(F)
        ax.scatter(
            [sense[best_i]],
            [comm[best_i]],
            [util[best_i]],
            marker="*",
            s=260,
            c="#FFD02A",
            edgecolors="black",
            linewidths=0.9,
            label="Selected trade-off",
        )
        ax.set_xlabel("Sensing active count", labelpad=10)
        ax.set_ylabel("Communication active count", labelpad=10)
        ax.set_zlabel("Utility", labelpad=8)
        ax.set_title("Fig. 7  3D Activation-Utility Distribution", fontsize=13, fontweight="bold", pad=16)
        ax.view_init(elev=25, azim=-52)
        fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.08, label="Utility")
        ax.legend(loc="upper left", fontsize=9.0)
        self._save("Fig7_Activation_Utility_3D.png")

    def fig_topology(self, dl: DataLoader, sense_on: np.ndarray, comm_on: np.ndarray) -> None:
        plt = self.plt
        mid = dl.T // 2
        pos = dl.pos_icrf[mid]
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        inactive = ~comm_on
        comm_only = comm_on & ~sense_on
        sense = sense_on
        ax.scatter(
            pos[inactive, 0],
            pos[inactive, 1],
            pos[inactive, 2],
            s=4,
            c="#bdbdbd",
            alpha=0.25,
            label="inactive",
        )
        ax.scatter(
            pos[comm_only, 0],
            pos[comm_only, 1],
            pos[comm_only, 2],
            s=8,
            c="#3182bd",
            alpha=0.65,
            label="comm",
        )
        ax.scatter(
            pos[sense, 0],
            pos[sense, 1],
            pos[sense, 2],
            s=16,
            c="#de2d26",
            alpha=0.9,
            label="sense+comm",
        )
        drawn = 0
        comm_idx = np.where(comm_on)[0]
        for i in comm_idx[: min(120, len(comm_idx))]:
            neigh = dl.isl_neighbor_idx[mid, i]
            for j in neigh:
                if j >= 0 and comm_on[j]:
                    ax.plot(
                        [pos[i, 0], pos[j, 0]],
                        [pos[i, 1], pos[j, 1]],
                        [pos[i, 2], pos[j, 2]],
                        c="#636363",
                        alpha=0.18,
                        linewidth=0.5,
                    )
                    drawn += 1
                    break
            if drawn >= 80:
                break
        ax.set_title("Dual-mode constellation topology snapshot")
        ax.set_xlabel("ICRF X (km)")
        ax.set_ylabel("ICRF Y (km)")
        ax.set_zlabel("ICRF Z (km)")
        ax.legend(loc="upper right")
        self._save("Fig1_Topology.png")

    def fig_pareto(self, result: Dict) -> None:
        plt = self.plt
        F = result["F"]
        hist = result["history"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        util = -F[:, 0]
        sc = axes[0].scatter(F[:, 1], util, c=F[:, 2], cmap="viridis", s=42)
        axes[0].set_xlabel("Active sensing satellites")
        axes[0].set_ylabel("Utility")
        axes[0].set_title("Pareto population")
        cb = fig.colorbar(sc, ax=axes[0])
        cb.set_label("Active communication satellites")
        if hist:
            axes[1].plot([h["gen"] for h in hist], [h["best_util"] for h in hist], lw=2)
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Best utility")
        axes[1].set_title("NSGA-III convergence")
        self._save("Fig2_Pareto_Convergence.png")

    def fig_activation(self, sat_names: np.ndarray, sense_on: np.ndarray, comm_on: np.ndarray) -> None:
        plt = self.plt
        x = np.arange(len(sat_names))
        fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True)
        axes[0].scatter(x[comm_on], np.ones(int(comm_on.sum())), s=8, c="#3182bd")
        axes[0].set_ylim(0.5, 1.5)
        axes[0].set_yticks([1])
        axes[0].set_yticklabels(["comm"])
        axes[0].set_title("Communication activation")
        axes[1].scatter(x[sense_on], np.ones(int(sense_on.sum())), s=10, c="#de2d26")
        axes[1].set_ylim(0.5, 1.5)
        axes[1].set_yticks([1])
        axes[1].set_yticklabels(["sense"])
        axes[1].set_xlabel("Satellite index")
        axes[1].set_title("Sensing activation")
        self._save("Fig3_Activation_Distribution.png")

    def fig_replay(self, hist: Dict) -> None:
        plt = self.plt
        time_h = np.asarray(hist["time_s"], dtype=float) / 3600.0
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        axes[0, 0].plot(time_h, hist["delivered_cum"], lw=2, c="#238b45")
        axes[0, 0].set_title("Delivered data")
        axes[0, 0].set_ylabel("MB")
        axes[0, 1].plot(time_h, hist["queue_total"], lw=1.8, c="#756bb1")
        axes[0, 1].set_title("Queue backlog")
        axes[0, 1].set_ylabel("MB")
        axes[1, 0].plot(time_h, hist["energy_mean"], label="mean", c="#3182bd")
        axes[1, 0].plot(time_h, hist["energy_min"], label="min", c="#de2d26")
        axes[1, 0].set_title("Energy state")
        axes[1, 0].set_xlabel("Time (h)")
        axes[1, 0].legend()
        axes[1, 1].plot(time_h, hist["melt_count"], label="melt", c="#e6550d")
        axes[1, 1].plot(time_h, hist["guard_count"], label="guard", c="#636363")
        axes[1, 1].set_title("Protection states")
        axes[1, 1].set_xlabel("Time (h)")
        axes[1, 1].legend()
        self._save("Fig4_Replay.png")


def run_complexity_experiment(
    cfg: Config,
    dl: DataLoader,
    best_gene: np.ndarray,
    nsga_runtime_s: float = 0.0,
) -> Dict:
    print("=" * 72)
    print("S3a Runtime scalability experiment for Fig5")
    print("=" * 72)
    rng = np.random.default_rng(int(cfg.RNG_SEED) + 20260407)
    n_total = int(dl.N)
    if n_total <= 1:
        return {
            "nodes": [],
            "t_full_greedy": [],
            "t_graph_only": [],
            "t_dual": [],
            "dual_nsga_s": [],
            "dual_online_s": [],
            "dual_total_s": [],
        }

    max_points = max(1, int(cfg.COMPLEXITY_MAX_POINTS))
    min_nodes = min(n_total, max(24, min(60, n_total)))
    if max_points == 1 or min_nodes >= n_total:
        node_targets = np.asarray([n_total], dtype=int)
    else:
        node_targets = np.unique(np.rint(np.geomspace(min_nodes, n_total, max_points)).astype(int))
    node_targets = node_targets[(node_targets >= 2) & (node_targets <= n_total)]
    if len(node_targets) == 0:
        node_targets = np.asarray([n_total], dtype=int)

    sense_ratio = float(np.mean(best_gene[: dl.N])) if dl.N else 0.1
    comm_ratio = float(np.mean(best_gene[dl.N :])) if dl.N else 0.3
    sense_ratio = float(np.clip(sense_ratio, 0.05, 1.0))
    comm_ratio = float(np.clip(comm_ratio, max(sense_ratio, 0.10), 1.0))
    steps = np.arange(min(dl.T, max(1, int(cfg.COMPLEXITY_MAX_STEPS))), dtype=int)

    def choose(pool: np.ndarray, count: int) -> np.ndarray:
        count = int(np.clip(count, 1, len(pool)))
        if count >= len(pool):
            return np.sort(pool)
        return np.sort(rng.choice(pool, count, replace=False))

    def benchmark(comm_idx: np.ndarray, sense_idx: np.ndarray, matcher: str, use_topk: bool, use_window: bool) -> float:
        comm_idx = np.asarray(comm_idx, dtype=np.int32)
        sense_idx = np.asarray(sense_idx, dtype=np.int32)
        if len(comm_idx) == 0:
            return 0.0
        comm_map = {int(sat): i for i, sat in enumerate(comm_idx)}
        sense_local = np.asarray([comm_map[int(sat)] for sat in sense_idx if int(sat) in comm_map], dtype=np.int32)
        q = np.full(len(comm_idx), cfg.Q_MAX_COMM * 0.28, dtype=np.float64)
        e = np.full(len(comm_idx), cfg.E_MAX * 0.90, dtype=np.float64)
        tx_isl = cfg.TX_RATE_ISL * dl.dt_eff
        tx_sgl = cfg.TX_RATE_SGL * dl.dt_eff
        production = cfg.R_SENSOR * dl.dt_eff * 0.15
        lookahead = max(1, int(cfg.WINDOW_LOOKAHEAD))

        t0 = time.perf_counter()
        for t in steps:
            senders = np.where(q > 1e-9)[0]
            receivers_allowed = q < cfg.Q_MAX_COMM - 1e-9
            rows: List[int] = []
            cols: List[int] = []
            scores: List[float] = []
            if len(senders) > 0 and np.any(receivers_allowed):
                for row in senders:
                    sat = int(comm_idx[row])
                    cand_global = dl.isl_neighbor_idx[t, sat]
                    cand_dist = dl.isl_neighbor_dist[t, sat]
                    cand_local = np.asarray([comm_map.get(int(c), -1) for c in cand_global], dtype=np.int32)
                    valid = (
                        (cand_local >= 0)
                        & (cand_local != row)
                        & receivers_allowed[np.maximum(cand_local, 0)]
                        & np.isfinite(cand_dist)
                        & (cand_dist > 0.0)
                        & (cand_dist < cfg.ISL_MAX_DIST)
                    )
                    if not np.any(valid):
                        continue
                    cand_local = cand_local[valid]
                    cand_dist = cand_dist[valid]
                    if use_topk and len(cand_local) > cfg.TOPK_CANDIDATES:
                        cand_local = cand_local[: cfg.TOPK_CANDIDATES]
                        cand_dist = cand_dist[: cfg.TOPK_CANDIDATES]
                    q_norm = (cfg.Q_MAX_COMM - q[cand_local]) / max(cfg.Q_MAX_COMM, 1e-9)
                    e_norm = e[cand_local] / max(cfg.E_MAX, 1e-9)
                    d_norm = cand_dist / max(cfg.ISL_MAX_DIST, 1e-9)
                    score = (1.0 - d_norm) * 0.4 + q_norm * 0.4 + e_norm * 0.2
                    for col, sc in zip(cand_local, score):
                        rows.append(int(row))
                        cols.append(int(col))
                        scores.append(float(sc))

            if rows:
                rows_arr = np.asarray(rows, dtype=np.int32)
                cols_arr = np.asarray(cols, dtype=np.int32)
                scores_arr = np.asarray(scores, dtype=np.float64)
                pairs: List[Tuple[int, int]] = []
                receiver_unique = np.unique(cols_arr)
                n_cells = len(senders) * len(receiver_unique)
                if matcher == "hungarian" and n_cells <= cfg.HUNGARIAN_MAX_CELLS:
                    col_pos = {int(c): i for i, c in enumerate(receiver_unique)}
                    score_mat = np.full((len(comm_idx), len(receiver_unique)), -1e9, dtype=np.float64)
                    for row, col, sc in zip(rows_arr, cols_arr, scores_arr):
                        j = col_pos[int(col)]
                        if sc > score_mat[row, j]:
                            score_mat[row, j] = sc
                    rr, cc = linear_sum_assignment(-score_mat)
                    for row, colp in zip(rr, cc):
                        if score_mat[row, colp] > -1e8:
                            pairs.append((int(row), int(receiver_unique[colp])))
                else:
                    used_s = set()
                    used_r = set()
                    for edge_i in np.argsort(-scores_arr):
                        s = int(rows_arr[edge_i])
                        r = int(cols_arr[edge_i])
                        if s in used_s or r in used_r:
                            continue
                        used_s.add(s)
                        used_r.add(r)
                        pairs.append((s, r))
                for s, r in pairs:
                    tx = min(q[s], tx_isl, cfg.Q_MAX_COMM - q[r])
                    if tx > 0.0:
                        q[s] -= tx
                        q[r] += tx

            cand = np.where(dl.vis_gs[t, comm_idx] & (q > 1e-9))[0]
            if len(cand) > 0:
                if use_window:
                    t_end = min(int(t) + lookahead, dl.T - 1)
                    if t_end > t:
                        future_vis = dl.vis_gs[int(t) : t_end, comm_idx[cand]]
                        remaining_window = future_vis.sum(axis=0).astype(float) + 1.0
                    else:
                        remaining_window = np.ones(len(cand), dtype=float)
                    priority = q[cand] / remaining_window
                else:
                    priority = q[cand]
                used = 0
                for local in cand[np.argsort(-priority)]:
                    if used >= cfg.GS_ANTENNAS:
                        break
                    tx = min(q[local], tx_sgl)
                    if tx > 0.0:
                        q[local] -= tx
                        used += 1

            if len(sense_local) > 0:
                q[sense_local] = np.minimum(q[sense_local] + production, cfg.Q_MAX_COMM)
        return time.perf_counter() - t0

    nodes: List[int] = []
    t_full_greedy: List[float] = []
    t_graph_only: List[float] = []
    t_dual: List[float] = []
    dual_nsga_s: List[float] = []
    dual_online_s: List[float] = []
    dual_total_s: List[float] = []
    repeats = max(1, int(cfg.COMPLEXITY_REPEATS))
    all_idx = np.arange(dl.N, dtype=np.int32)
    nsga_runtime_s = max(float(nsga_runtime_s), 0.0)
    for n in node_targets:
        full_times: List[float] = []
        graph_times: List[float] = []
        dual_times: List[float] = []
        for _ in range(repeats):
            pool = choose(all_idx, int(n))
            sparse_sense_n = max(1, int(np.ceil(len(pool) * sense_ratio)))
            sparse_comm_n = max(sparse_sense_n, int(np.ceil(len(pool) * comm_ratio)))
            sparse_sense = choose(pool, sparse_sense_n)
            sparse_comm = np.unique(np.concatenate([choose(pool, sparse_comm_n), sparse_sense])).astype(np.int32)
            full_times.append(benchmark(pool, pool, matcher="greedy", use_topk=False, use_window=False))
            graph_times.append(benchmark(pool, pool, matcher="hungarian", use_topk=False, use_window=False))
            dual_times.append(benchmark(sparse_comm, sparse_sense, matcher="hungarian", use_topk=True, use_window=True))
        nodes.append(int(n))
        t_full_greedy.append(float(np.median(full_times)))
        t_graph_only.append(float(np.median(graph_times)))
        dual_online = float(np.median(dual_times))
        dual_nsga = nsga_runtime_s * (float(n) / max(float(n_total), 1.0))
        dual_total = dual_nsga + dual_online
        dual_online_s.append(dual_online)
        dual_nsga_s.append(dual_nsga)
        dual_total_s.append(dual_total)
        t_dual.append(dual_total)
        print(
            f"  nodes={int(n):5d} full+greedy={t_full_greedy[-1]:.4f}s "
            f"graph-only={t_graph_only[-1]:.4f}s "
            f"dual-online={dual_online:.4f}s dual-total={dual_total:.4f}s"
        )

    return {
        "nodes": nodes,
        "t_full_greedy": t_full_greedy,
        "t_graph_only": t_graph_only,
        "t_dual": t_dual,
        "dual_nsga_s": dual_nsga_s,
        "dual_online_s": dual_online_s,
        "dual_total_s": dual_total_s,
        "dual_nsga_runtime_model": "linear_scale_from_measured_full_run",
        "measured_full_nsga_s": nsga_runtime_s,
    }


def run_baseline_comparison_experiment(cfg: Config, dl: DataLoader, best_gene: np.ndarray) -> Dict:
    print("=" * 72)
    print("S3b Baseline comparison experiment for Fig6")
    print("=" * 72)
    engine = SimulationEngine(cfg, dl)
    full_gene = np.ones(2 * dl.N, dtype=bool)
    methods_cfg = [
        {
            "label": "Full+greedy",
            "color": "#2B2B2B",
            "gene": full_gene,
            "matcher_mode": "greedy",
            "use_topk": False,
            "use_window_urgency": False,
            "use_energy_score": False,
        },
        {
            "label": "Graph-only",
            "color": "#E34A33",
            "gene": full_gene,
            "matcher_mode": "hungarian",
            "use_topk": False,
            "use_window_urgency": False,
            "use_energy_score": True,
        },
        {
            "label": "Dual-layer",
            "color": "#2171B5",
            "gene": best_gene.astype(bool),
            "matcher_mode": "hungarian",
            "use_topk": True,
            "use_window_urgency": True,
            "use_energy_score": True,
        },
    ]
    repeats = max(1, int(cfg.BASELINE_REPEATS))
    methods: List[Dict] = []
    for item in methods_cfg:
        _, counts, hist = engine.evaluate(
            item["gene"],
            track_history=False,
            return_summary=True,
            matcher_mode=item["matcher_mode"],
            use_topk=item["use_topk"],
            use_window_urgency=item["use_window_urgency"],
            use_energy_score=item["use_energy_score"],
        )
        summary = dict(hist.get("summary", {})) if hist else {}
        runtime_samples: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            engine.evaluate(
                item["gene"],
                track_history=False,
                return_summary=False,
                matcher_mode=item["matcher_mode"],
                use_topk=item["use_topk"],
                use_window_urgency=item["use_window_urgency"],
                use_energy_score=item["use_energy_score"],
            )
            runtime_samples.append(time.perf_counter() - t0)
        summary["runtime_s"] = float(np.median(runtime_samples))
        summary["label"] = item["label"]
        summary["color"] = item["color"]
        summary["active_satellites"] = int(counts[1])
        methods.append(summary)
        print(
            f"  {item['label']:<12s} delivered={summary.get('delivered_mb', 0.0):.1f} MB "
            f"queue={summary.get('avg_queue_backlog_mb', 0.0):.1f} MB "
            f"energy={summary.get('total_energy_demand', 0.0):.1f} "
            f"runtime={summary['runtime_s']:.3f}s"
        )
    return {"methods": methods}


def save_outputs(
    out_dir: Path,
    cfg: Config,
    dl: DataLoader,
    sense_on: np.ndarray,
    comm_on: np.ndarray,
    summary: Dict,
    result: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = cfg, dl, sense_on, comm_on, summary, result
    print("  skipped best activation, summary, config, and NSGA history files")


def save_complexity_outputs(out_dir: Path, timing: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "complexity_timing.json").open("w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    nodes = timing.get("nodes", [])
    rows: List[Dict] = []
    for i in range(len(nodes)):
        row = {}
        for key, value in timing.items():
            if isinstance(value, list) and len(value) == len(nodes):
                row[key] = value[i]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "complexity_timing.csv", index=False)
    print(f"  saved complexity timing tables in {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-mode constellation sensing/communication activation simulation"
    )
    parser.add_argument("--profile", choices=["full", "fast", "smoke"], default="full")
    parser.add_argument("--csv", dest="csv_path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ga-pop", type=int, default=None)
    parser.add_argument("--ga-gen", type=int, default=None)
    parser.add_argument("--sat-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_profile(Config(), args.profile)
    if args.csv_path:
        cfg.SAT_STATE_CSV = args.csv_path
    if args.output:
        cfg.OUTPUT_DIR = args.output
    if args.seed is not None:
        cfg.RNG_SEED = args.seed
    if args.ga_pop is not None:
        cfg.GA_POP = args.ga_pop
    if args.ga_gen is not None:
        cfg.GA_GEN = args.ga_gen
    if args.sat_limit is not None:
        cfg.SAT_LIMIT = args.sat_limit
    if args.max_steps is not None:
        cfg.MAX_STEPS = args.max_steps
    if args.no_plots:
        cfg.PLOT = False
    if args.no_cache:
        cfg.USE_ISL_CACHE = False

    np.random.seed(cfg.RNG_SEED)
    out_dir = Path(cfg.OUTPUT_DIR)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nDual-mode constellation joint sensing/communication activation")
    print(f"profile={args.profile}, output={out_dir}")
    print(f"ground station: {GS_NAME} ({GS_LAT}N, {GS_LON}E)")
    task_desc = ", ".join(
        f"{name} {weight * 100:.0f}%"
        for name, weight in zip(TASK_REGION_NAMES, TASK_REGION_WEIGHTS)
    )
    print(f"task regions: {task_desc}")

    t0 = time.perf_counter()
    dl = DataLoader(cfg)
    problem = ConstellationProblem(cfg, dl)
    solver = NSGA3Solver(cfg)
    result = solver.optimize(problem.evaluate, dl.N)

    best_i = select_best_objective_index(result["F"], utility_tol=max(float(cfg.GA_CONV_EPS), 1e-6))
    best_gene = result["X"][best_i].astype(bool)
    engine = SimulationEngine(cfg, dl)
    sense_on, comm_on = engine.split_genes(best_gene)
    _, counts, replay = engine.evaluate(best_gene, track_history=True, return_summary=True)
    summary = dict(replay["summary"])
    summary["runtime_total_s"] = float(time.perf_counter() - t0)
    summary["profile"] = args.profile
    summary["satellites_total"] = int(dl.N)
    summary["time_steps"] = int(dl.T)
    summary["dt_s"] = float(dl.dt_eff)
    summary["runtime_nsga_s"] = float(result.get("runtime_s", 0.0))
    summary["nsga_generations_completed"] = int(result.get("generations_completed", 0))
    summary["nsga_evaluations"] = int(result.get("evaluations", 0))

    print("=" * 72)
    print("Best solution")
    print("=" * 72)
    print(f"delivery utility     : {summary['delivery_utility']:.2f}")
    print(f"raw utility          : {summary['utility']:.2f}")
    print(f"delivered            : {summary['delivered_mb']:.1f} MB")
    print(f"delivery ratio       : {summary['delivery_ratio'] * 100:.2f}%")
    print(f"sensing active       : {counts[0]} / {dl.N}")
    print(f"communication active : {counts[1]} / {dl.N}")
    print(f"constraint valid     : {summary['constraint_valid']}")
    print(f"total runtime        : {summary['runtime_total_s']:.1f}s")

    save_outputs(out_dir, cfg, dl, sense_on, comm_on, summary, result)
    comparison = run_method_comparison_experiment(
        cfg,
        dl,
        best_gene,
        nsga_runtime_s=float(result.get("runtime_s", 0.0)),
    )
    save_method_comparison_outputs(out_dir, comparison)

    if cfg.PLOT:
        print("=" * 72)
        print("S5 Generating requested experiment figures")
        print("=" * 72)
        scale = run_scalability_experiment(
            cfg,
            dl,
            best_gene,
            nsga_runtime_s=float(result.get("runtime_s", 0.0)),
        )
        save_scalability_outputs(out_dir, scale)
        vis = Visualizer(out_dir)
        vis.fig1_scale_online_time(scale)
        vis.fig2_scale_total_runtime(scale)
        vis.fig3_scale_activation_ratio(scale)
        vis.fig4_scale_delivery_ratio(scale)
        vis.fig5_nsga_pareto_front(result["X"], result["F"], dl)
        vis.fig6_nsga_convergence(result["history"])
        vis.fig7_activation_utility_3d(result["X"], result["F"], dl, best_gene)

    print("=" * 72)
    print("Done")
    print("=" * 72)


if __name__ == "__main__":
    main()

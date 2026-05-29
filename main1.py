# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import time
import warnings
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")


# Basic geometry and physics constants. Distances use km, rates use MB/s.
R_EARTH = 6371.0
ATM_HEIGHT = 100.0
C_LIGHT = 300000.0
OMEGA_E = 7.2921150e-5

# Single ground station: Beijing.
GS_NAME = "Beijing"
GS_LAT = 39.90
GS_LON = 116.40

# Multi-task remote-sensing regions. The simulation uses the normalized
# scheduling weights requested for the virtual observation node workload.
TASK_REGIONS = (
    {"name": "South China Sea", "lat": 15.0, "lon": 115.0, "weight": 0.40},
    {"name": "Chengdu", "lat": 30.57, "lon": 104.07, "weight": 0.20},
    {"name": "Harbin", "lat": 45.80, "lon": 126.53, "weight": 0.20},
    {"name": "Urumqi", "lat": 43.82, "lon": 87.62, "weight": 0.20},
)

SCALE_NODE_TARGETS = tuple([100, 500] + list(range(1000, 20001, 1000)))
TOPK_SENSITIVITY_VALUES = (1, 2, 3, 4, 6, 8, 10, 12)
SPARSE_ACTIVATION_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
SPARSE_RANDOM_REPEATS = 5
STRESS_TASK_DATA_FACTOR = 1.8
STRESS_TASK_GEN_FACTOR = 1.4
STRESS_TX_RATE_FACTOR = 0.55
STRESS_QUEUE_FACTOR = 0.65
ABLATION_CONFIGS = (
    ("Proposed", {"future_gs": True, "energy": True, "task_priority": True, "buffer_pressure": True}),
    ("No Future GS", {"future_gs": False, "energy": True, "task_priority": True, "buffer_pressure": True}),
    ("No Energy", {"future_gs": True, "energy": False, "task_priority": True, "buffer_pressure": True}),
    ("No Task Priority", {"future_gs": True, "energy": True, "task_priority": False, "buffer_pressure": True}),
    ("No Buffer Pressure", {"future_gs": True, "energy": True, "task_priority": True, "buffer_pressure": False}),
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
    """Select max utility first, then fewer activated roles."""
    obj = np.asarray(F, dtype=float)
    if obj.ndim != 2 or obj.shape[0] == 0 or obj.shape[1] < 2:
        raise ValueError("objective matrix must be non-empty with at least 2 columns")
    util = -obj[:, 0]
    max_util = float(np.max(util))
    tied = np.where(util >= max_util - max(float(utility_tol), 0.0))[0]
    if obj.shape[1] >= 3:
        total_roles = obj[tied, 1] + obj[tied, 2]
        order = np.lexsort((obj[tied, 1], obj[tied, 2], total_roles))
    else:
        order = np.argsort(obj[tied, 1])
    return int(tied[order[0]])


def scale_node_targets(n_total: int) -> np.ndarray:
    targets = [n for n in SCALE_NODE_TARGETS if 0 < n <= n_total]
    if n_total > 0 and n_total not in targets:
        targets.append(int(n_total))
    return np.asarray(sorted(set(targets)), dtype=int)


def _compute_isl_topk(pos_icrf: np.ndarray, k_neighbors: int, max_dist: float, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    T, N, _ = pos_icrf.shape
    K = int(max(1, k_neighbors))
    idx_all = np.full((T, N, K), -1, dtype=np.int32)
    dist_all = np.full((T, N, K), np.inf, dtype=np.float32)
    if N <= 1:
        return idx_all, dist_all

    query_k = min(K + 1, N)
    for t in range(T):
        if verbose and (t % 100 == 0 or t == T - 1):
            print(f"    ISL cache step {t + 1}/{T}")
        pos = pos_icrf[t].astype(np.float64, copy=False)
        tree = cKDTree(pos)
        try:
            dist, idx = tree.query(pos, k=query_k, distance_upper_bound=max_dist, workers=-1)
        except TypeError:
            dist, idx = tree.query(pos, k=query_k, distance_upper_bound=max_dist)
        if query_k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        dist = dist[:, 1:]
        idx = idx[:, 1:]
        width = dist.shape[1]
        if width > 0:
            invalid = (idx >= N) | ~np.isfinite(dist)
            idx = idx.astype(np.int32, copy=False)
            idx[invalid] = -1
            dist = dist.astype(np.float32, copy=False)
            dist[invalid] = np.inf
            take = min(K, width)
            idx_all[t, :, :take] = idx[:, :take]
            dist_all[t, :, :take] = dist[:, :take]
    return idx_all, dist_all


@dataclass
class Config:
    SAT_STATE_CSV: str = "satellite_state_all.csv"
    OUTPUT_DIR: str = "Simulation_Experiment_Results"
    CACHE_DIR: str = ".cache_experiment"

    RNG_SEED: int = 42
    SAMPLE_EVERY: int = 1
    MAX_STEPS: Optional[int] = None
    SAT_LIMIT: Optional[int] = None

    Q_MAX_COMM: float = 5000.0
    E_MAX: float = 100.0
    E_MELT: float = 10.0
    E_WARN_RATIO: float = 1.35
    E_BASE_DRAIN: float = 0.07
    E_SOLAR_CHG: float = 0.13
    E_TX_DRAIN: float = 0.02
    E_SENSE_DRAIN: float = 0.02
    E_RX_RATIO: float = 0.5
    E_MELT_RECOVER_RATIO: float = 2.0
    E_GUARD_RECOVER_RATIO: float = 2.4

    ELEVATION_MIN: float = 15.0
    TX_RATE_ISL: float = 150.0
    TX_RATE_SGL: float = 250.0
    ISL_MAX_DIST: float = 4000.0
    ISL_NEIGHBOR_K: int = 12
    TOPK_CANDIDATES: int = 12
    ADAPTIVE_K_BASE: int = 8
    ADAPTIVE_K_MIN: int = 4
    ADAPTIVE_K_MAX: int = 12
    ADAPTIVE_K_ALPHA: float = 8.0
    ADAPTIVE_K_BETA: float = 6.0
    ADAPTIVE_K_GAMMA: float = 6.0
    HUNGARIAN_MAX_CELLS: int = 250_000

    R_SENSOR: float = 100.0
    TASK_GEN_RATE: float = 100.0
    TASK_ARRIVAL_MODE: str = "poisson"
    TASK_PACKET_MB: float = 500.0
    TASK_POISSON_SCALE: float = 1.0
    TASK_RADIUS: float = 300.0
    TASK_DATA_MB: float = 20000.0
    URGENCY: float = 8.5
    PENALTY: float = 0.15
    GS_ANTENNAS: int = 4
    WINDOW_LOOKAHEAD: int = 10
    MAX_RELAY_HOPS: int = 4
    CACHE_TTL_S: float = 1800.0

    GA_POP: int = 60
    GA_GEN: int = 10
    GA_CR: float = 0.8
    GA_PARTITIONS: int = 4
    GA_CONV_WINDOW: int = 10
    GA_CONV_EPS: float = 0.5

    COMPLEXITY_REPEATS: int = 3
    COMPLEXITY_MAX_STEPS: int = 96
    BASELINE_REPEATS: int = 1

    T0_DOY: float = 76.0
    T0_GMST: float = 0.5

    PLOT: bool = True
    USE_ISL_CACHE: bool = True


def apply_profile(cfg: Config, profile: str) -> Config:
    if profile == "smoke":
        cfg.MAX_STEPS = 24
        cfg.SAT_LIMIT = 120
        cfg.GA_POP = 8
        cfg.GA_GEN = 2
    elif profile == "fast":
        cfg.MAX_STEPS = 240
        cfg.SAT_LIMIT = 400
        cfg.GA_POP = 24
        cfg.GA_GEN = 12
    elif profile != "full":
        raise ValueError(f"unknown profile: {profile}")
    cfg.ISL_NEIGHBOR_K = int(cfg.ADAPTIVE_K_MAX)
    cfg.TOPK_CANDIDATES = int(cfg.ADAPTIVE_K_MAX)
    return cfg


def make_stress_config(cfg: Config) -> Config:
    """Create a resource-constrained scenario without changing nominal defaults."""
    stress = replace(cfg)
    stress.TASK_DATA_MB = float(cfg.TASK_DATA_MB) * STRESS_TASK_DATA_FACTOR
    stress.TASK_GEN_RATE = float(cfg.TASK_GEN_RATE) * STRESS_TASK_GEN_FACTOR
    stress.TX_RATE_ISL = float(cfg.TX_RATE_ISL) * STRESS_TX_RATE_FACTOR
    stress.TX_RATE_SGL = float(cfg.TX_RATE_SGL) * STRESS_TX_RATE_FACTOR
    stress.Q_MAX_COMM = float(cfg.Q_MAX_COMM) * STRESS_QUEUE_FACTOR
    stress.GS_ANTENNAS = max(1, int(cfg.GS_ANTENNAS) // 2)
    return stress


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

        sat_names_all = np.array(sorted(df["sat_id"].dropna().unique().tolist()), dtype=object)
        times = np.sort(df["time_s"].dropna().unique().astype(np.float64))
        if self.cfg.SAMPLE_EVERY > 1:
            times = times[:: int(self.cfg.SAMPLE_EVERY)]
        if self.cfg.MAX_STEPS is not None:
            times = self._select_representative_time_window(df, times, int(self.cfg.MAX_STEPS))
        sat_names = sat_names_all
        if self.cfg.SAT_LIMIT is not None:
            sat_names = self._select_representative_satellites(df, sat_names_all, times, int(self.cfg.SAT_LIMIT))

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
            raise ValueError(f"CSV is not rectangular after filtering: missing {missing} position samples")

        if self.T > 1:
            dts = np.diff(self.times_s)
            if not np.allclose(dts, np.median(dts), rtol=0.0, atol=1e-6):
                raise ValueError("time_s values are not uniformly spaced")
            self.dt_eff = float(np.median(dts))
        else:
            self.dt_eff = 60.0

        rows_expected = self.T * self.N
        if len(df) != rows_expected:
            raise ValueError(f"expected {rows_expected} rows after filtering, got {len(df)}")
        print(f"  loaded rows={len(df)}, satellites={self.N}, time steps={self.T}")

    def _mission_region_scores(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if frame.empty:
            return np.zeros((0, len(TASK_REGION_NAMES)), dtype=np.float64), np.zeros(0, dtype=np.float64)
        time_s = frame["time_s"].to_numpy(dtype=np.float64)
        x = frame["x_km"].to_numpy(dtype=np.float64)
        y = frame["y_km"].to_numpy(dtype=np.float64)
        z = frame["z_km"].to_numpy(dtype=np.float64)
        theta = self.cfg.T0_GMST + OMEGA_E * time_s
        ct = np.cos(theta)
        st = np.sin(theta)
        pos_ecef = np.column_stack((ct * x + st * y, -st * x + ct * y, z))
        r_norm = np.linalg.norm(pos_ecef, axis=1) + 1e-9

        cos_task = np.cos(float(self.cfg.TASK_RADIUS) / R_EARTH)
        region_scores = np.zeros((len(frame), len(TASK_REGION_NAMES)), dtype=np.float64)
        for region_i, region_unit in enumerate(TASK_REGION_UNIT):
            covered = (pos_ecef @ region_unit) / r_norm >= cos_task
            region_scores[:, region_i] = covered.astype(np.float64)

        vec = pos_ecef - GS_ECEF[None, :]
        dist = np.linalg.norm(vec, axis=1) + 1e-9
        sin_el_min = np.sin(np.radians(self.cfg.ELEVATION_MIN))
        gs_visible = (vec @ GS_UNIT) / dist >= sin_el_min
        gs_score = gs_visible.astype(np.float64)
        return region_scores, gs_score

    def _mission_relevance_scores(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        region_scores, gs_score = self._mission_region_scores(frame)
        if len(region_scores) == 0:
            return np.zeros(0, dtype=np.float64), gs_score
        task_score = region_scores @ TASK_REGION_WEIGHTS
        return task_score, gs_score

    def _select_representative_time_window(self, df: pd.DataFrame, times: np.ndarray, max_steps: int) -> np.ndarray:
        max_steps = int(max(1, max_steps))
        if len(times) <= max_steps:
            return times
        frame = df[df["time_s"].isin(times)]
        region_scores, gs_score = self._mission_region_scores(frame)
        region_by_time = []
        for region_i in range(len(TASK_REGION_NAMES)):
            by_time = (
                pd.Series(region_scores[:, region_i], index=frame.index)
                .groupby(frame["time_s"])
                .sum()
                .reindex(times, fill_value=0.0)
                .to_numpy(dtype=np.float64)
            )
            region_by_time.append(by_time)
        region_by_time = np.vstack(region_by_time)
        gs_by_time = (
            pd.Series(gs_score, index=frame.index)
            .groupby(frame["time_s"])
            .sum()
            .reindex(times, fill_value=0.0)
            .to_numpy(dtype=np.float64)
        )
        kernel = np.ones(max_steps, dtype=np.float64)
        window_region = np.vstack([np.convolve(region_by_time[i], kernel, mode="valid") for i in range(len(TASK_REGION_NAMES))])
        window_gs = np.convolve(gs_by_time, kernel, mode="valid")
        active_regions = (window_region > 0.0).sum(axis=0).astype(np.float64)
        min_region = window_region.min(axis=0)
        weighted_region = TASK_REGION_WEIGHTS @ window_region
        denom = len(TASK_REGION_NAMES) * np.sum(window_region * window_region, axis=0)
        fairness = np.divide(
            np.sum(window_region, axis=0) ** 2,
            np.maximum(denom, 1e-9),
            out=np.zeros(window_region.shape[1], dtype=np.float64),
            where=denom > 0.0,
        )
        window_score = active_regions * 1e6 + min_region * 1e3 + fairness * 1e2 + weighted_region + 0.05 * window_gs
        start = int(np.argmax(window_score)) if len(window_score) else 0
        selected = times[start : start + max_steps]
        if start != 0:
            print(
                "  representative time window selected: "
                f"{selected[0]:.1f}..{selected[-1]:.1f}s "
                f"(mission score={float(window_score[start]):.1f}, "
                f"covered regions={int(active_regions[start])}/{len(TASK_REGION_NAMES)})"
            )
        return selected

    def _select_representative_satellites(
        self,
        df: pd.DataFrame,
        sat_names: np.ndarray,
        times: np.ndarray,
        sat_limit: int,
    ) -> np.ndarray:
        sat_limit = int(max(1, sat_limit))
        if len(sat_names) <= sat_limit:
            return sat_names
        frame = df[df["time_s"].isin(times)]
        region_scores, gs_score = self._mission_region_scores(frame)
        region_by_sat = []
        for region_i in range(len(TASK_REGION_NAMES)):
            by_sat = (
                pd.Series(region_scores[:, region_i], index=frame.index)
                .groupby(frame["sat_id"])
                .sum()
                .reindex(sat_names, fill_value=0.0)
                .to_numpy(dtype=np.float64)
            )
            region_by_sat.append(by_sat)
        region_by_sat = np.vstack(region_by_sat).T
        gs_by_sat = (
            pd.Series(gs_score, index=frame.index)
            .groupby(frame["sat_id"])
            .sum()
            .reindex(sat_names, fill_value=0.0)
            .to_numpy(dtype=np.float64)
        )
        aggregate_score = region_by_sat @ TASK_REGION_WEIGHTS + 0.20 * gs_by_sat
        selected: List[object] = []
        selected_set = set()
        if sat_limit >= len(TASK_REGION_NAMES):
            quotas = np.maximum(1, np.floor(TASK_REGION_WEIGHTS * sat_limit).astype(int))
            while int(quotas.sum()) > sat_limit:
                candidates = np.where(quotas > 1)[0]
                if len(candidates) == 0:
                    break
                quotas[candidates[np.argmax(quotas[candidates])]] -= 1
            while int(quotas.sum()) < sat_limit:
                quotas[int(np.argmax(TASK_REGION_WEIGHTS))] += 1
            for region_i in np.argsort(-TASK_REGION_WEIGHTS):
                order = np.lexsort((sat_names.astype(str), -region_by_sat[:, region_i]))
                picked = 0
                for idx in order:
                    if region_by_sat[idx, region_i] <= 0.0:
                        break
                    sat = sat_names[idx]
                    if sat in selected_set:
                        continue
                    selected.append(sat)
                    selected_set.add(sat)
                    picked += 1
                    if picked >= int(quotas[region_i]) or len(selected) >= sat_limit:
                        break
                if len(selected) >= sat_limit:
                    break
        fill_order = np.lexsort((sat_names.astype(str), -aggregate_score))
        for idx in fill_order:
            if len(selected) >= sat_limit:
                break
            sat = sat_names[idx]
            if sat in selected_set:
                continue
            selected.append(sat)
            selected_set.add(sat)
        selected = np.array(sorted(selected[:sat_limit]), dtype=object)
        selected_mask = np.isin(sat_names, selected)
        selected_scores = aggregate_score[selected_mask]
        if float(np.max(selected_scores, initial=0.0)) > 0.0:
            region_counts = (region_by_sat[selected_mask] > 0.0).sum(axis=0)
            region_msg = ", ".join(f"{name}={int(count)}" for name, count in zip(TASK_REGION_NAMES, region_counts))
            print(
                "  representative satellite subset selected: "
                f"{sat_limit}/{len(sat_names)} satellites "
                f"(mean mission score={float(np.mean(selected_scores)):.3f}; {region_msg})"
            )
        else:
            selected = sat_names[:sat_limit]
        return selected

    def _icrf_to_ecef(self) -> None:
        print("  converting ICRF positions to ECEF by Earth rotation")
        theta = self.cfg.T0_GMST + OMEGA_E * self.times_s
        ct = np.cos(theta).astype(np.float32)
        st = np.sin(theta).astype(np.float32)
        self.pos_ecef = np.empty_like(self.pos_icrf)
        self.pos_ecef[:, :, 0] = ct[:, None] * self.pos_icrf[:, :, 0] + st[:, None] * self.pos_icrf[:, :, 1]
        self.pos_ecef[:, :, 1] = -st[:, None] * self.pos_icrf[:, :, 0] + ct[:, None] * self.pos_icrf[:, :, 1]
        self.pos_ecef[:, :, 2] = self.pos_icrf[:, :, 2]

    def _compute_ground_geometry(self) -> None:
        print("  computing Beijing GS visibility and task-region coverage")
        self.vis_gs = np.zeros((self.T, self.N), dtype=bool)
        # task_coverage_by_region[r, t, i] is the coverage predicate for the
        # implicit virtual observation node O(t, i, r). No explicit O class is
        # created; data generation and buffering are represented by
        # remaining_by_region[r], task_region_weights[r], and q_region[r, i].
        self.task_coverage_by_region = np.zeros((len(TASK_REGIONS), self.T, self.N), dtype=bool)
        sin_el_min = np.sin(np.radians(self.cfg.ELEVATION_MIN))
        cos_task = np.cos(float(self.cfg.TASK_RADIUS) / R_EARTH)
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
        region_msg = ", ".join(f"{name}={mean:.4f}" for name, mean in zip(TASK_REGION_NAMES, region_means))
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
        sun_ecef = np.stack([ct * sx + st * sy, -st * sx + ct * sy, sz], axis=1)
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
        sat_hash = hashlib.sha1("\n".join(map(str, self.sat_names)).encode("utf-8")).hexdigest()[:10]
        stem = (
            f"isl_topk_N{self.N}_T{self.T}_K{self.cfg.ISL_NEIGHBOR_K}_"
            f"D{int(self.cfg.ISL_MAX_DIST)}_t{int(self.times_s[0])}-{int(self.times_s[-1])}_"
            f"s{sat_hash}.npz"
        )
        return cache_dir / stem

    def _precompute_isl_neighbors(self) -> None:
        K = int(max(1, self.cfg.ISL_NEIGHBOR_K))
        cache_path = self._cache_path() if self.cfg.USE_ISL_CACHE else None
        if cache_path is not None and cache_path.exists():
            print(f"  loading ISL Top-K cache: {cache_path}")
            data = np.load(cache_path)
            idx = data["idx"]
            dist = data["dist"]
            if idx.shape == (self.T, self.N, K) and dist.shape == (self.T, self.N, K):
                self.isl_neighbor_idx = idx
                self.isl_neighbor_dist = dist
                return
            print("  cache shape mismatch; rebuilding")

        print(f"  precomputing ISL nearest neighbors with cKDTree: K={K}")
        t0 = time.perf_counter()
        self.isl_neighbor_idx, self.isl_neighbor_dist = _compute_isl_topk(
            self.pos_icrf,
            K,
            self.cfg.ISL_MAX_DIST,
            verbose=True,
        )
        print(f"  ISL Top-K cache built in {time.perf_counter() - t0:.1f}s")
        if cache_path is not None:
            np.savez(cache_path, idx=self.isl_neighbor_idx, dist=self.isl_neighbor_dist)
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
        self.task_arrival_by_region = self._build_task_arrivals()

    def _build_task_arrivals(self) -> np.ndarray:
        """Build one reproducible task-arrival stream shared by all evaluations.

        fixed mode reproduces the old deterministic per-step workload. poisson
        mode keeps the same expected MB per region and time step, while turning
        the number of remote-sensing data batches into a Poisson random variable.
        """
        cfg = self.cfg
        dl = self.dl
        mode = str(getattr(cfg, "TASK_ARRIVAL_MODE", "fixed")).strip().lower()
        weights = dl.task_region_weights.astype(np.float64, copy=True)
        expected = float(cfg.TASK_GEN_RATE) * float(dl.dt_eff) * weights[:, None]
        expected = np.repeat(expected, int(dl.T), axis=1)
        if mode == "fixed":
            return expected.astype(np.float64, copy=False)
        if mode != "poisson":
            raise ValueError(f"unknown TASK_ARRIVAL_MODE={cfg.TASK_ARRIVAL_MODE!r}; use fixed or poisson")
        packet_mb = max(float(getattr(cfg, "TASK_PACKET_MB", 500.0)), 1e-9)
        scale = max(float(getattr(cfg, "TASK_POISSON_SCALE", 1.0)), 1e-9)
        seed_payload = (
            f"{int(cfg.RNG_SEED)}|{int(dl.T)}|{float(dl.dt_eff):.9f}|"
            f"{float(cfg.TASK_GEN_RATE):.9f}|{packet_mb:.9f}|{scale:.9f}|"
            f"{','.join(f'{w:.9f}' for w in weights)}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little") % (2**32 - 1)
        rng = np.random.default_rng(seed)
        lam = np.maximum(expected / packet_mb * scale, 0.0)
        return rng.poisson(lam).astype(np.float64) * packet_mb / scale

    @staticmethod
    def _score_flag(score_flags: Optional[Dict[str, bool]], name: str) -> bool:
        return True if score_flags is None else bool(score_flags.get(name, True))

    @staticmethod
    def _weighted_sum(components: List[Tuple[bool, float, np.ndarray]], fallback: np.ndarray) -> np.ndarray:
        out = np.zeros_like(fallback, dtype=np.float64)
        weight_sum = 0.0
        for enabled, weight, values in components:
            if not enabled:
                continue
            out += float(weight) * np.asarray(values, dtype=np.float64)
            weight_sum += float(weight)
        if weight_sum <= 1e-12:
            return fallback.astype(np.float64, copy=False)
        return out / weight_sum

    def split_genes(self, genes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split a 2N gene into sensing and communication roles.

        The physical coupling comm_on |= sense_on models a dual-mode satellite:
        a sensing role depends on its communication role to cache, forward, and
        downlink data through the cross-layer access topology.
        """
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
        return np.clip(weighted / np.maximum(q_total, 1e-9) / max(float(np.max(weights)), 1e-9), 0.0, 1.0)

    def _future_gs_score(self, t: int, sats: np.ndarray) -> np.ndarray:
        sats_arr = np.asarray(sats, dtype=np.int64)
        if sats_arr.size == 0:
            return np.zeros(0, dtype=np.float64)
        t_end = min(int(t) + max(1, int(self.cfg.WINDOW_LOOKAHEAD)), self.dl.T)
        future = self.dl.vis_gs[int(t):t_end, sats_arr]
        return np.clip(future.sum(axis=0).astype(np.float64) / max(float(t_end - int(t)), 1.0), 0.0, 1.0)

    def _adaptive_k_value(self, buffer_pressure: float, gs_direction_score: float, task_priority: float) -> int:
        cfg = self.cfg
        raw = (
            float(cfg.ADAPTIVE_K_BASE)
            + float(cfg.ADAPTIVE_K_ALPHA) * float(np.clip(buffer_pressure, 0.0, 1.0))
            + float(cfg.ADAPTIVE_K_BETA) * float(np.clip(gs_direction_score, 0.0, 1.0))
            + float(cfg.ADAPTIVE_K_GAMMA) * float(np.clip(task_priority, 0.0, 1.0))
        )
        return int(np.clip(round(raw), int(cfg.ADAPTIVE_K_MIN), int(cfg.ADAPTIVE_K_MAX)))

    @staticmethod
    def _deplete_region_queue(q_region: np.ndarray, sat: int, amount: float, weights: np.ndarray) -> Tuple[float, np.ndarray]:
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

    @staticmethod
    def _empty_transfer_stats() -> Dict:
        return {"senders": set(), "receivers": set(), "moved_mb": 0.0, "max_hop_after_transfer": 0}

    @staticmethod
    def _append_batch(batch_queues: Dict[Tuple[int, int], deque], region_i: int, sat: int, amount: float, born_time_s: float, hop: int) -> None:
        if amount <= 1e-9:
            return
        key = (int(region_i), int(sat))
        batch_queues.setdefault(key, deque()).append(
            {"amount": float(amount), "born_time_s": float(born_time_s), "hop": int(hop)}
        )

    @staticmethod
    def _remove_sat_batches(batch_queues: Dict[Tuple[int, int], deque], sats: np.ndarray) -> None:
        sat_set = {int(s) for s in np.asarray(sats, dtype=np.int64).tolist()}
        if not sat_set:
            return
        for key in list(batch_queues.keys()):
            if int(key[1]) in sat_set:
                del batch_queues[key]

    def _expire_ttl_batches(
        self,
        batch_queues: Dict[Tuple[int, int], deque],
        q_region: np.ndarray,
        current_time_s: float,
    ) -> Tuple[float, np.ndarray]:
        ttl_s = float(self.cfg.CACHE_TTL_S)
        dropped_by_region = np.zeros(q_region.shape[0], dtype=np.float64)
        if ttl_s <= 0.0:
            return 0.0, dropped_by_region
        for key in list(batch_queues.keys()):
            region_i, sat = int(key[0]), int(key[1])
            kept = deque()
            dropped = 0.0
            for batch in batch_queues[key]:
                amount = float(batch.get("amount", 0.0))
                if amount <= 1e-9:
                    continue
                age_s = float(current_time_s) - float(batch.get("born_time_s", current_time_s))
                if age_s > ttl_s:
                    dropped += amount
                else:
                    kept.append(batch)
            if dropped > 0.0:
                dropped_by_region[region_i] += dropped
                q_region[region_i, sat] = max(0.0, float(q_region[region_i, sat]) - dropped)
            if kept:
                batch_queues[key] = kept
            else:
                del batch_queues[key]
        return float(dropped_by_region.sum()), dropped_by_region

    def _movable_amount_by_sat(self, batch_queues: Dict[Tuple[int, int], deque], n_sat: int) -> np.ndarray:
        movable = np.zeros(int(n_sat), dtype=np.float64)
        max_hops = int(self.cfg.MAX_RELAY_HOPS)
        for (_region_i, sat), batches in batch_queues.items():
            total = 0.0
            for batch in batches:
                if int(batch.get("hop", 0)) < max_hops:
                    total += float(batch.get("amount", 0.0))
            if total > 1e-9:
                movable[int(sat)] += total
        return movable

    def _deplete_batch_queue(
        self,
        batch_queues: Dict[Tuple[int, int], deque],
        q_region: np.ndarray,
        sat: int,
        amount: float,
        forward: bool,
    ) -> Tuple[float, np.ndarray, Dict[int, List[Dict]], float, int]:
        remaining = max(float(amount), 0.0)
        moved_by_region = np.zeros(q_region.shape[0], dtype=np.float64)
        moved_batches_by_region: Dict[int, List[Dict]] = {}
        hop_weighted_sum = 0.0
        max_hop = 0
        max_hops = int(self.cfg.MAX_RELAY_HOPS)

        for region_i in np.argsort(-self.dl.task_region_weights):
            if remaining <= 1e-9:
                break
            key = (int(region_i), int(sat))
            queue = batch_queues.get(key)
            if not queue:
                continue

            kept = deque()
            moved_batches: List[Dict] = []
            while queue:
                batch = queue.popleft()
                batch_amount = float(batch.get("amount", 0.0))
                batch_hop = int(batch.get("hop", 0))
                if batch_amount <= 1e-9:
                    continue
                can_take = remaining > 1e-9 and ((not forward) or batch_hop < max_hops)
                if not can_take:
                    kept.append(batch)
                    continue

                take = min(batch_amount, remaining)
                out_hop = batch_hop + 1 if forward else batch_hop
                moved_batch = {
                    "amount": float(take),
                    "born_time_s": float(batch.get("born_time_s", 0.0)),
                    "hop": int(out_hop),
                }
                moved_batches.append(moved_batch)
                moved_by_region[int(region_i)] += take
                hop_weighted_sum += take * float(out_hop)
                max_hop = max(max_hop, int(out_hop))
                remaining -= take

                leftover = batch_amount - take
                if leftover > 1e-9:
                    batch["amount"] = float(leftover)
                    kept.append(batch)
                if remaining <= 1e-9:
                    kept.extend(queue)
                    break

            if kept:
                batch_queues[key] = kept
            else:
                batch_queues.pop(key, None)
            moved_total_region = float(moved_by_region[int(region_i)])
            if moved_total_region > 0.0:
                q_region[int(region_i), int(sat)] = max(0.0, float(q_region[int(region_i), int(sat)]) - moved_total_region)
                moved_batches_by_region[int(region_i)] = moved_batches

        return float(moved_by_region.sum()), moved_by_region, moved_batches_by_region, float(hop_weighted_sum), int(max_hop)

    def _move_region_queue(
        self,
        q_region: np.ndarray,
        batch_queues: Dict[Tuple[int, int], deque],
        sender: int,
        receiver: int,
        amount: float,
    ) -> Tuple[float, int]:
        moved_total, moved_by_region, moved_batches, _hop_sum, max_hop = self._deplete_batch_queue(
            batch_queues, q_region, sender, amount, forward=True
        )
        if moved_total > 0.0:
            q_region[:, int(receiver)] += moved_by_region
            for region_i, batches in moved_batches.items():
                for batch in batches:
                    self._append_batch(
                        batch_queues,
                        int(region_i),
                        int(receiver),
                        float(batch["amount"]),
                        float(batch["born_time_s"]),
                        int(batch["hop"]),
                    )
        return moved_total, int(max_hop)

    def _deplete_downlink_queue(
        self,
        q_region: np.ndarray,
        batch_queues: Dict[Tuple[int, int], deque],
        sat: int,
        amount: float,
    ) -> Tuple[float, np.ndarray, float, int]:
        moved_total, moved_by_region, _moved_batches, hop_sum, max_hop = self._deplete_batch_queue(
            batch_queues, q_region, sat, amount, forward=False
        )
        return moved_total, moved_by_region, float(hop_sum), int(max_hop)

    def _edge_scores(
        self,
        t: int,
        sender: int,
        cand: np.ndarray,
        dist: np.ndarray,
        q_region: np.ndarray,
        q_total: np.ndarray,
        e: np.ndarray,
        score_flags: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
        cfg = self.cfg
        distance_score = 1.0 - np.clip(dist / max(cfg.ISL_MAX_DIST, 1e-9), 0.0, 1.0)
        source_buffer_pressure = np.full(
            len(cand),
            np.clip(q_total[sender] / max(cfg.Q_MAX_COMM, 1e-9), 0.0, 1.0),
            dtype=np.float64,
        )
        receiver_free_capacity = np.clip((cfg.Q_MAX_COMM - q_total[cand]) / max(cfg.Q_MAX_COMM, 1e-9), 0.0, 1.0)
        gs_direction_score = self._future_gs_score(t, cand)
        energy_score = np.clip(np.minimum(e[sender], e[cand]) / max(cfg.E_MAX, 1e-9), 0.0, 1.0)
        priority_value = float(self._task_priority(q_region, np.asarray([sender]), self.dl.task_region_weights)[0])
        task_priority = np.full(len(cand), priority_value, dtype=np.float64)
        return self._weighted_sum(
            [
                (True, 0.15, distance_score),
                (self._score_flag(score_flags, "buffer_pressure"), 0.20, source_buffer_pressure),
                (True, 0.15, receiver_free_capacity),
                (self._score_flag(score_flags, "future_gs"), 0.20, gs_direction_score),
                (self._score_flag(score_flags, "energy"), 0.15, energy_score),
                (self._score_flag(score_flags, "task_priority"), 0.15, task_priority),
            ],
            distance_score,
        )

    def _downlink_priority(
        self,
        t: int,
        cand: np.ndarray,
        q_region: np.ndarray,
        e: np.ndarray,
        score_flags: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
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
        return self._weighted_sum(
            [
                (self._score_flag(score_flags, "task_priority"), 0.35, task_priority),
                (self._score_flag(score_flags, "buffer_pressure"), 0.30, buffer_pressure),
                (self._score_flag(score_flags, "future_gs"), 0.20, window_urgency),
                (self._score_flag(score_flags, "energy"), 0.15, energy_score),
            ],
            q_total[cand],
        )

    def _isl_transfer(
        self,
        t: int,
        q_region: np.ndarray,
        batch_queues: Dict[Tuple[int, int], deque],
        e: np.ndarray,
        avail: np.ndarray,
        total_energy_demand: float,
        matcher_mode: str = "hungarian",
        use_topk: bool = True,
        use_window_urgency: bool = True,
        use_energy_score: bool = True,
        fixed_topk_k: Optional[int] = None,
        score_flags: Optional[Dict[str, bool]] = None,
    ) -> Tuple[float, Dict]:
        """Run Greedy/Hungarian online scheduling over Full or Top-K graph."""
        cfg = self.cfg
        dl = self.dl
        stats = self._empty_transfer_stats()
        if matcher_mode not in {"hungarian", "greedy"}:
            raise ValueError(f"unknown matcher_mode: {matcher_mode}")

        q_total = q_region.sum(axis=0)
        movable_total = self._movable_amount_by_sat(batch_queues, dl.N)
        senders = np.where(avail & (movable_total > 1e-9))[0]
        receivers_allowed = avail & (q_total < cfg.Q_MAX_COMM - 1e-9)
        if len(senders) == 0 or int(receivers_allowed.sum()) <= 1:
            return total_energy_demand, stats

        pos_t = dl.pos_icrf[t].astype(np.float64, copy=False)
        if use_topk:
            idx_k = dl.isl_neighbor_idx[t, senders]
            dist_k = dl.isl_neighbor_dist[t, senders]
            full_receiver_idx = None
            full_neighbor_lists = None
        else:
            full_receiver_idx = np.where(receivers_allowed)[0]
            if len(full_receiver_idx) == 0:
                return total_energy_demand, stats
            full_tree = cKDTree(pos_t[full_receiver_idx])
            try:
                full_neighbor_lists = full_tree.query_ball_point(pos_t[senders], r=cfg.ISL_MAX_DIST, workers=-1)
            except TypeError:
                full_neighbor_lists = full_tree.query_ball_point(pos_t[senders], r=cfg.ISL_MAX_DIST)
            idx_k = None
            dist_k = None

        rows: List[int] = []
        cols: List[int] = []
        scores: List[float] = []

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
            valid = (cand >= 0) & (cand != s) & np.isfinite(dist) & (dist > 0.0) & (dist < cfg.ISL_MAX_DIST)
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
                if fixed_topk_k is None:
                    sender_pressure = q_total[s] / max(cfg.Q_MAX_COMM, 1e-9)
                    sender_gs = float(self._future_gs_score(t, np.asarray([s]))[0]) if use_window_urgency else float(dl.vis_gs[t, s])
                    sender_priority = float(self._task_priority(q_region, np.asarray([s]), dl.task_region_weights)[0])
                    k_s = self._adaptive_k_value(sender_pressure, sender_gs, sender_priority)
                else:
                    k_s = int(np.clip(int(fixed_topk_k), 1, dl.isl_neighbor_idx.shape[2]))
                if len(cand) > k_s:
                    cand = cand[:k_s]
                    dist = dist[:k_s]

            los = self._los_mask(np.repeat(pos_t[s][None, :], len(cand), axis=0), pos_t[cand])
            if not np.any(los):
                continue
            cand = cand[los]
            dist = dist[los]
            if use_energy_score:
                score = self._edge_scores(t, int(s), cand, dist, q_region, q_total, e, score_flags=score_flags)
            else:
                d_norm = dist / max(cfg.ISL_MAX_DIST, 1e-9)
                free = (cfg.Q_MAX_COMM - q_total[cand]) / max(cfg.Q_MAX_COMM, 1e-9)
                score = 0.50 * (1.0 - d_norm) + 0.40 * free + 0.10 * dl.vis_gs[t, cand].astype(float)
            for c, sc in zip(cand, score):
                rows.append(row)
                cols.append(int(c))
                scores.append(float(sc))

        if not rows:
            return total_energy_demand, stats

        rows_arr = np.asarray(rows, dtype=np.int32)
        cols_arr = np.asarray(cols, dtype=np.int32)
        scores_arr = np.asarray(scores, dtype=np.float64)
        receiver_unique = np.unique(cols_arr)
        assignments: List[Tuple[int, int]] = []

        n_cells = len(senders) * len(receiver_unique)
        if matcher_mode == "hungarian" and n_cells <= cfg.HUNGARIAN_MAX_CELLS:
            col_pos = {int(c): i for i, c in enumerate(receiver_unique)}
            score_mat = np.full((len(senders), len(receiver_unique)), -1e9, dtype=np.float64)
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
            for edge_i in np.argsort(-scores_arr):
                s = int(senders[rows_arr[edge_i]])
                r = int(cols_arr[edge_i])
                if s in used_s or r in used_r:
                    continue
                used_s.add(s)
                used_r.add(r)
                assignments.append((s, r))

        tx_cap = cfg.TX_RATE_ISL * dl.dt_eff
        for s, r in assignments:
            if movable_total[s] <= 0.0 or q_total[r] >= cfg.Q_MAX_COMM:
                continue
            tx = min(movable_total[s], tx_cap, cfg.Q_MAX_COMM - q_total[r])
            if tx <= 0.0:
                continue
            moved, max_hop = self._move_region_queue(q_region, batch_queues, s, r, tx)
            if moved <= 0.0:
                continue
            q_total[s] -= moved
            q_total[r] += moved
            movable_total[s] = max(0.0, movable_total[s] - moved)
            e[s] = max(0.0, e[s] - self.e_tx)
            e[r] = max(0.0, e[r] - self.e_rx)
            total_energy_demand += self.e_tx + self.e_rx
            stats["senders"].add(int(s))
            stats["receivers"].add(int(r))
            stats["moved_mb"] = float(stats["moved_mb"]) + float(moved)
            stats["max_hop_after_transfer"] = max(int(stats["max_hop_after_transfer"]), int(max_hop))
        return total_energy_demand, stats

    def evaluate(
        self,
        genes: np.ndarray,
        track_history: bool = False,
        return_summary: bool = False,
        matcher_mode: str = "hungarian",
        use_topk: bool = True,
        use_window_urgency: bool = True,
        use_energy_score: bool = True,
        fixed_topk_k: Optional[int] = None,
        score_flags: Optional[Dict[str, bool]] = None,
    ) -> Tuple[float, Tuple[int, int], Optional[Dict]]:
        """Evaluate one sparse activation scheme.

        Virtual observation node O(t, i, r) is implicit: when satellite i covers
        task region r at step t and sense_on, comm_on, energy, melt, and guard
        states are feasible, the generated remote-sensing batch is recorded in
        q_region[r, i]. The same q_region buffers drive communication forwarding
        nodes, downlink priority, regional weights, and final accounting.
        """
        cfg = self.cfg
        dl = self.dl
        eval_t0 = time.perf_counter()
        sense_on, comm_on = self.split_genes(genes)
        n_sense = int(sense_on.sum())
        n_comm = int(comm_on.sum())
        n_regions = len(TASK_REGIONS)
        if n_comm == 0:
            empty_summary = {
                "delivered_mb": 0.0,
                "delivery_ratio": 0.0,
                "sensing_progress": 0.0,
                "unsensed_mb": float(cfg.TASK_DATA_MB),
                "generated_mb": 0.0,
                "task_arrival_mode": str(cfg.TASK_ARRIVAL_MODE).lower(),
                "task_packet_mb": float(cfg.TASK_PACKET_MB),
                "task_arrival_offered_mb": float(self.task_arrival_by_region.sum()),
                "task_arrival_by_region_mb": self.task_arrival_by_region.sum(axis=1).tolist(),
                "generated_by_region_mb": [0.0] * n_regions,
                "delivered_by_region_mb": [0.0] * n_regions,
                "dropped_by_region_mb": [0.0] * n_regions,
                "dropped_mb": 0.0,
                "dropped_melt_mb": 0.0,
                "dropped_ttl_mb": 0.0,
                "packet_loss_rate": 0.0,
                "avg_queue_backlog_mb": 0.0,
                "sense_active_count": n_sense,
                "comm_active_count": n_comm,
                "active_satellite_count": n_comm,
                "candidate_comm_relay_ratio": 0.0,
                "actual_forward_unique_satellite_count": 0,
                "actual_forward_participation_ratio": 0.0,
                "actual_downlink_unique_satellite_count": 0,
                "avg_actual_isl_send_count": 0.0,
                "max_actual_isl_send_count": 0,
                "avg_actual_isl_receive_count": 0.0,
                "max_actual_isl_receive_count": 0,
                "avg_actual_downlink_count": 0.0,
                "max_actual_downlink_count": 0,
                "avg_delivered_relay_hops": 0.0,
                "max_observed_relay_hops": 0,
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
                "constraint_valid": bool(np.all(~sense_on | comm_on)),
            }
            return 0.0, (n_sense, n_comm), {"summary": empty_summary} if return_summary else None

        active_comm = np.where(comm_on)[0]
        e = np.full(dl.N, cfg.E_MAX, dtype=np.float64)
        q_region = np.zeros((n_regions, dl.N), dtype=np.float64)
        batch_queues: Dict[Tuple[int, int], deque] = {}
        melt = np.zeros(dl.N, dtype=bool)
        guard = np.zeros(dl.N, dtype=bool)

        delivered = 0.0
        delivered_by_region = np.zeros(n_regions, dtype=np.float64)
        remaining_by_region = cfg.TASK_DATA_MB * dl.task_region_weights.astype(np.float64, copy=True)
        generated_total = 0.0
        generated_by_region = np.zeros(n_regions, dtype=np.float64)
        dropped_melt = 0.0
        dropped_by_region = np.zeros(n_regions, dtype=np.float64)
        queue_integral = 0.0
        total_energy_demand = 0.0
        completion_time_s = None
        online_scheduling_time_s = 0.0
        low_power_events = 0
        melt_events = 0
        dropped_ttl = 0.0
        actual_forward_unique: set = set()
        actual_downlink_unique: set = set()
        actual_isl_send_counts: List[int] = []
        actual_isl_receive_counts: List[int] = []
        actual_downlink_counts: List[int] = []
        delivered_hop_sum = 0.0
        delivered_hop_amount = 0.0
        max_observed_relay_hops = 0
        candidate_comm_relay_ratio = float(n_comm / max(dl.N, 1))

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
                "actual_isl_send_count": [],
                "actual_isl_receive_count": [],
                "actual_downlink_count": [],
                "actual_forward_unique_count": [],
                "candidate_comm_relay_ratio": [],
                "actual_forward_participation_ratio": [],
                "dropped_ttl_cum": [],
                "avg_relay_hop_cum": [],
            }

        for t in range(dl.T):
            ecl = dl.eclipse[t, active_comm]
            solar = np.where(ecl, 0.0, self.e_solar)
            e[active_comm] = np.clip(e[active_comm] + solar - self.e_base, 0.0, cfg.E_MAX)
            total_energy_demand += self.e_base * n_comm

            new_melt = melt.copy()
            new_melt[comm_on & (e < cfg.E_MELT)] = True
            new_melt[(e >= self.e_melt_recover) & comm_on] = False
            just_melt = new_melt & ~melt
            if np.any(just_melt):
                melt_drop = q_region[:, just_melt].sum(axis=1)
                dropped_by_region += melt_drop
                dropped_melt += float(melt_drop.sum())
                q_region[:, just_melt] = 0.0
                self._remove_sat_batches(batch_queues, np.where(just_melt)[0])
                melt_events += int(just_melt.sum())
            melt = new_melt

            prev_guard = guard.copy()
            guard[comm_on & ~melt & (e <= self.e_warn)] = True
            guard[(e >= self.e_guard_recover) & comm_on] = False
            guard[melt | ~comm_on] = False
            low_power_events += int((guard & ~prev_guard).sum())

            avail = comm_on & ~melt & ~guard

            ttl_drop, ttl_drop_by_region = self._expire_ttl_batches(batch_queues, q_region, float(dl.times_s[t]))
            if ttl_drop > 0.0:
                dropped_ttl += float(ttl_drop)
                dropped_by_region += ttl_drop_by_region

            if float(remaining_by_region.sum()) > 1e-9:
                sensor_step_cap = np.full(dl.N, cfg.R_SENSOR * dl.dt_eff, dtype=np.float64)
                q_total = q_region.sum(axis=0)
                for region_i, region_weight in enumerate(dl.task_region_weights):
                    if remaining_by_region[region_i] <= 1e-9:
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
                        float(self.task_arrival_by_region[region_i, t]),
                        float(remaining_by_region[region_i]),
                    )
                    if step_budget <= 1e-9:
                        continue
                    per_sat_cap = np.minimum(
                        np.maximum(cfg.Q_MAX_COMM - q_total[sense_idx], 0.0),
                        sensor_step_cap[sense_idx],
                    )
                    alloc = self._fair_allocate(step_budget, per_sat_cap)
                    used = alloc > 1e-9
                    if not np.any(used):
                        continue
                    target = sense_idx[used]
                    stored = alloc[used]
                    q_region[region_i, target] += stored
                    for sat_i, amount_i in zip(target, stored):
                        self._append_batch(batch_queues, int(region_i), int(sat_i), float(amount_i), float(dl.times_s[t]), hop=0)
                    q_total[target] += stored
                    sensor_step_cap[target] = np.maximum(0.0, sensor_step_cap[target] - stored)
                    actual_generated = float(stored.sum())
                    generated_total += actual_generated
                    generated_by_region[region_i] += actual_generated
                    remaining_by_region[region_i] = max(0.0, float(remaining_by_region[region_i]) - actual_generated)
                    e[target] = np.maximum(0.0, e[target] - self.e_sense)
                    total_energy_demand += self.e_sense * int(len(target))

            sched_t0 = time.perf_counter()
            total_energy_demand, transfer_stats = self._isl_transfer(
                t,
                q_region,
                batch_queues,
                e,
                avail,
                total_energy_demand,
                matcher_mode=matcher_mode,
                use_topk=use_topk,
                use_window_urgency=use_window_urgency,
                use_energy_score=use_energy_score,
                fixed_topk_k=fixed_topk_k,
                score_flags=score_flags,
            )
            online_scheduling_time_s += time.perf_counter() - sched_t0
            step_isl_senders = set(transfer_stats.get("senders", set()))
            step_isl_receivers = set(transfer_stats.get("receivers", set()))
            actual_forward_unique.update(step_isl_senders)
            actual_forward_unique.update(step_isl_receivers)
            actual_isl_send_counts.append(int(len(step_isl_senders)))
            actual_isl_receive_counts.append(int(len(step_isl_receivers)))
            max_observed_relay_hops = max(
                int(max_observed_relay_hops),
                int(transfer_stats.get("max_hop_after_transfer", 0)),
            )

            sched_t0 = time.perf_counter()
            q_total = q_region.sum(axis=0)
            cand = np.where(avail & dl.vis_gs[t] & (q_total > 1e-9))[0]
            step_downlink_sats = set()
            if len(cand) > 0:
                priority = self._downlink_priority(t, cand, q_region, e, score_flags=score_flags) if use_energy_score else q_total[cand]
                ant_used = 0
                tx_sgl = cfg.TX_RATE_SGL * dl.dt_eff
                for sat in cand[np.argsort(-priority)]:
                    if ant_used >= cfg.GS_ANTENNAS:
                        break
                    tx = min(q_total[sat], tx_sgl)
                    if tx <= 0.0:
                        continue
                    actual_tx, delivered_regions, hop_sum, max_hop = self._deplete_downlink_queue(
                        q_region, batch_queues, int(sat), tx
                    )
                    if actual_tx <= 0.0:
                        continue
                    q_total[sat] -= actual_tx
                    delivered += actual_tx
                    delivered_by_region += delivered_regions
                    delivered_hop_sum += float(hop_sum)
                    delivered_hop_amount += float(actual_tx)
                    max_observed_relay_hops = max(int(max_observed_relay_hops), int(max_hop))
                    e[sat] = max(0.0, e[sat] - self.e_tx)
                    total_energy_demand += self.e_tx
                    step_downlink_sats.add(int(sat))
                    actual_downlink_unique.add(int(sat))
                    ant_used += 1
                if completion_time_s is None and delivered >= cfg.TASK_DATA_MB - 1e-9:
                    completion_time_s = float(dl.times_s[t] - dl.times_s[0])
            online_scheduling_time_s += time.perf_counter() - sched_t0
            actual_downlink_counts.append(int(len(step_downlink_sats)))

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
                hist["drop_cum"].append(float(dropped_melt + dropped_ttl))
                hist["actual_isl_send_count"].append(int(actual_isl_send_counts[-1] if actual_isl_send_counts else 0))
                hist["actual_isl_receive_count"].append(int(actual_isl_receive_counts[-1] if actual_isl_receive_counts else 0))
                hist["actual_downlink_count"].append(int(actual_downlink_counts[-1] if actual_downlink_counts else 0))
                hist["actual_forward_unique_count"].append(int(len(actual_forward_unique)))
                hist["candidate_comm_relay_ratio"].append(float(candidate_comm_relay_ratio))
                hist["actual_forward_participation_ratio"].append(float(len(actual_forward_unique) / max(dl.N, 1)))
                hist["dropped_ttl_cum"].append(float(dropped_ttl))
                hist["avg_relay_hop_cum"].append(
                    float(delivered_hop_sum / delivered_hop_amount) if delivered_hop_amount > 1e-9 else 0.0
                )

        q_total = q_region.sum(axis=0)
        total_buffer = float(q_total[active_comm].sum())
        dropped_total = float(dropped_by_region.sum())
        utility = float(delivered * cfg.URGENCY - cfg.PENALTY * total_buffer)
        delivery_utility = float(np.sum(delivered_by_region * dl.task_region_weights) * cfg.URGENCY)
        packet_loss_rate = 0.0 if generated_total <= 1e-9 else float(dropped_total / max(generated_total, 1e-9))
        simulation_runtime_s = float(time.perf_counter() - eval_t0)
        send_counts = np.asarray(actual_isl_send_counts, dtype=np.float64)
        recv_counts = np.asarray(actual_isl_receive_counts, dtype=np.float64)
        down_counts = np.asarray(actual_downlink_counts, dtype=np.float64)
        actual_forward_unique_count = int(len(actual_forward_unique))
        actual_forward_ratio = float(actual_forward_unique_count / max(dl.N, 1))
        actual_downlink_unique_count = int(len(actual_downlink_unique))
        avg_delivered_hops = float(delivered_hop_sum / delivered_hop_amount) if delivered_hop_amount > 1e-9 else 0.0
        summary = {
            "delivered_mb": float(delivered),
            "delivery_ratio": float(min(delivered / max(cfg.TASK_DATA_MB, 1e-9), 1.0)),
            "sensing_progress": float(min(generated_total / max(cfg.TASK_DATA_MB, 1e-9), 1.0)),
            "unsensed_mb": float(max(remaining_by_region.sum(), 0.0)),
            "generated_mb": float(generated_total),
            "task_arrival_mode": str(cfg.TASK_ARRIVAL_MODE).lower(),
            "task_packet_mb": float(cfg.TASK_PACKET_MB),
            "task_arrival_offered_mb": float(self.task_arrival_by_region.sum()),
            "task_arrival_by_region_mb": self.task_arrival_by_region.sum(axis=1).tolist(),
            "generated_by_region_mb": generated_by_region.tolist(),
            "delivered_by_region_mb": delivered_by_region.tolist(),
            "dropped_by_region_mb": dropped_by_region.tolist(),
            "dropped_mb": dropped_total,
            "dropped_melt_mb": float(dropped_melt),
            "dropped_ttl_mb": float(dropped_ttl),
            "packet_loss_rate": packet_loss_rate,
            "avg_queue_backlog_mb": float(queue_integral / max(dl.T, 1)),
            "sense_active_count": int(n_sense),
            "comm_active_count": int(n_comm),
            "active_satellite_count": int(n_comm),
            "candidate_comm_relay_ratio": candidate_comm_relay_ratio,
            "actual_forward_unique_satellite_count": actual_forward_unique_count,
            "actual_forward_participation_ratio": actual_forward_ratio,
            "actual_downlink_unique_satellite_count": actual_downlink_unique_count,
            "avg_actual_isl_send_count": float(send_counts.mean()) if send_counts.size else 0.0,
            "max_actual_isl_send_count": int(send_counts.max()) if send_counts.size else 0,
            "avg_actual_isl_receive_count": float(recv_counts.mean()) if recv_counts.size else 0.0,
            "max_actual_isl_receive_count": int(recv_counts.max()) if recv_counts.size else 0,
            "avg_actual_downlink_count": float(down_counts.mean()) if down_counts.size else 0.0,
            "max_actual_downlink_count": int(down_counts.max()) if down_counts.size else 0,
            "avg_delivered_relay_hops": avg_delivered_hops,
            "max_observed_relay_hops": int(max_observed_relay_hops),
            "total_energy_demand": float(total_energy_demand),
            "unit_delivered_energy": float(total_energy_demand / delivered) if delivered > 1e-9 else None,
            "completion_time_s": completion_time_s,
            "online_scheduling_time_s": float(online_scheduling_time_s),
            "simulation_runtime_s": simulation_runtime_s,
            "avg_remaining_energy": float(e[active_comm].mean()),
            "min_remaining_energy": float(e[active_comm].min()),
            "low_power_protection_count": int(low_power_events),
            "melt_count": int(melt_events),
            "final_buffer_mb": float(total_buffer),
            "utility": utility,
            "delivery_utility": delivery_utility,
            "constraint_valid": bool(np.all(~sense_on | comm_on) and int(max_observed_relay_hops) <= int(cfg.MAX_RELAY_HOPS)),
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

        rec(int(n_obj), int(max(1, H)), int(max(1, H)), [])
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

    def _survive(self, X: np.ndarray, F: np.ndarray, pop: int, refs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            ri_last = ri_all[len(selected):]
            di_last = di_all[len(selected):]
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
        y = np.asarray(x, dtype=bool).copy()
        y[n_sat:] |= y[:n_sat]
        return y.astype(float)

    @staticmethod
    def _cx(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.random.rand(len(p1)) < 0.5
        return np.where(mask, p1, p2), np.where(mask, p2, p1)

    def _mut(self, x: np.ndarray, pm: float, n_sat: int) -> np.ndarray:
        y = np.logical_xor(np.asarray(x, dtype=bool), np.random.rand(len(x)) < pm)
        return self._repair(y, n_sat)

    def _init_population(self, pop: int, n_sat: int) -> np.ndarray:
        n_var = 2 * n_sat
        X = np.zeros((pop, n_var), dtype=float)
        n_sparse = pop // 2
        n_random = pop // 4
        n_heur = pop - n_sparse - n_random

        if n_sparse > 0:
            sense = np.random.rand(n_sparse, n_sat) < 0.12
            comm = np.random.rand(n_sparse, n_sat) < 0.30
            comm |= sense
            X[:n_sparse, :n_sat] = sense
            X[:n_sparse, n_sat:] = comm

        start = n_sparse
        end = start + n_random
        if n_random > 0:
            sense = np.random.rand(n_random, n_sat) < 0.25
            comm = np.random.rand(n_random, n_sat) < 0.50
            comm |= sense
            X[start:end, :n_sat] = sense
            X[start:end, n_sat:] = comm

        target_sense = max(1, n_sat // 12)
        target_comm = max(target_sense, n_sat // 4)
        for k in range(end, end + n_heur):
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
        pop = int(max(1, c.GA_POP))
        gen = int(max(0, c.GA_GEN))
        n_var = 2 * int(n_sat)
        pm = 1.0 / max(n_var, 1)
        X = self._init_population(pop, int(n_sat))
        print("=" * 72)
        print(f"S2 NSGA-III sparse activation optimization: pop={pop}, gen={gen}, dim={n_var}")
        print("=" * 72)
        t0 = time.perf_counter()
        F = np.asarray([eval_fn(X[i].astype(bool)) for i in range(pop)], dtype=float)
        refs = self._ref_dirs(int(F.shape[1]), int(c.GA_PARTITIONS))
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
                off_X.extend([self._mut(c1, pm, int(n_sat)), self._mut(c2, pm, int(n_sat))])
            off_X = np.asarray(off_X[:pop], dtype=float)
            off_F = np.asarray([eval_fn(off_X[i].astype(bool)) for i in range(pop)], dtype=float)
            X, F = self._survive(np.vstack([X, off_X]), np.vstack([F, off_F]), pop, refs)

            best_i = select_best_objective_index(F, utility_tol=tie_tol)
            best_util = float(-F[best_i, 0])
            best_gene = X[best_i].astype(bool)
            best_sense = int(best_gene[:n_sat].sum())
            best_comm = int((best_gene[n_sat:] | best_gene[:n_sat]).sum())
            best_total = best_comm
            util = -F[:, 0]
            near_best = np.where(util >= float(np.max(util)) - tie_tol)[0]
            near_best_min_active = int(np.min(F[near_best, 2])) if len(near_best) else best_total
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
                    "elapsed_s": float(time.perf_counter() - t0),
                    "utility_ceiling": utility_ceiling,
                    "utility_gap_to_ceiling": float(max(0.0, utility_ceiling - best_util)),
                    "ceiling_reached": bool(best_util >= utility_ceiling - tie_tol),
                }
            )
            if g == 0 or (g + 1) % 5 == 0 or g == gen - 1:
                print(f"  Gen {g + 1:3d}/{gen} best_U={best_util:.1f} sense={best_sense} comm={best_comm}")
            w = int(max(1, c.GA_CONV_WINDOW))
            if len(best_util_hist) >= w:
                util_flat = max(best_util_hist[-w:]) - min(best_util_hist[-w:]) < c.GA_CONV_EPS
                total_flat = max(best_total_hist[-w:]) == min(best_total_hist[-w:])
                if util_flat and total_flat:
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


class TBLSOptimizer:
    """Time-Budgeted Local Search baseline for sparse activation.

    TBLS searches only the sensing/communication activation vector. It is a
    traditional heuristic optimizer used for equal wall-clock budget comparison
    with NSGA-III; Greedy/Hungarian online scheduling is still performed inside
    SimulationEngine.evaluate().
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _repair(gene: np.ndarray, n_sat: int) -> np.ndarray:
        out = np.asarray(gene, dtype=bool).copy()
        sense = out[:n_sat]
        comm = out[n_sat:]
        comm |= sense
        out[:n_sat] = sense
        out[n_sat:] = comm
        return out

    @staticmethod
    def _better(
        utility_a: float,
        counts_a: Tuple[int, int],
        utility_b: float,
        counts_b: Tuple[int, int],
        tol: float = 1e-9,
    ) -> bool:
        if float(utility_a) > float(utility_b) + tol:
            return True
        if abs(float(utility_a) - float(utility_b)) <= tol:
            active_a = int(counts_a[0]) + int(counts_a[1])
            active_b = int(counts_b[0]) + int(counts_b[1])
            if active_a < active_b:
                return True
            if active_a == active_b and int(counts_a[1]) < int(counts_b[1]):
                return True
        return False

    def _initial_pool(self, n_sat: int, dl: DataLoader, scores: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
        order = np.argsort(-scores)
        genes: List[np.ndarray] = []
        for comm_ratio, sense_ratio in ((0.18, 0.35), (0.30, 0.35), (0.45, 0.30), (0.60, 0.25)):
            comm_count = int(np.clip(round(n_sat * comm_ratio), 1, n_sat))
            sense_count = int(np.clip(round(comm_count * sense_ratio), 1, comm_count))
            genes.append(_make_gene(n_sat, order[:sense_count], order[:comm_count]))
        for comm_ratio, sense_ratio in ((0.24, 0.30), (0.35, 0.25), (0.50, 0.20)):
            comm_count = int(np.clip(round(n_sat * comm_ratio), 1, n_sat))
            sense_count = int(np.clip(round(comm_count * sense_ratio), 1, comm_count))
            pool = order[: max(comm_count * 3, comm_count)]
            comm_idx = rng.choice(pool, size=comm_count, replace=False)
            sense_idx = rng.choice(comm_idx, size=sense_count, replace=False)
            genes.append(_make_gene(n_sat, sense_idx, comm_idx))
        for _ in range(4):
            p_comm = float(rng.uniform(0.15, 0.55))
            p_sense = float(rng.uniform(0.08, 0.28))
            comm_idx = np.where(rng.random(n_sat) < p_comm)[0]
            if len(comm_idx) == 0:
                comm_idx = order[: max(1, int(round(0.2 * n_sat)))]
            sense_pool = comm_idx
            sense_idx = sense_pool[rng.random(len(sense_pool)) < p_sense]
            if len(sense_idx) == 0:
                sense_idx = sense_pool[:1]
            genes.append(_make_gene(n_sat, sense_idx, comm_idx))
        return [self._repair(g, n_sat) for g in genes]

    @staticmethod
    def _weighted_choice(candidates: np.ndarray, scores: np.ndarray, rng: np.random.Generator, prefer_high: bool) -> Optional[int]:
        if len(candidates) == 0:
            return None
        vals = np.asarray(scores[candidates], dtype=np.float64)
        if prefer_high:
            weights = vals - float(np.min(vals)) + 1e-6
        else:
            weights = float(np.max(vals)) - vals + 1e-6
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return int(rng.choice(candidates))
        return int(rng.choice(candidates, p=weights / total))

    def _mutate_local(
        self,
        gene: np.ndarray,
        n_sat: int,
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = self._repair(gene, n_sat)
        sense = out[:n_sat].copy()
        comm = out[n_sat:].copy() | sense
        move = str(
            rng.choice(
                [
                    "add_sense",
                    "drop_sense",
                    "swap_sense",
                    "add_comm",
                    "drop_comm",
                    "swap_comm",
                    "shrink",
                ],
                p=[0.16, 0.12, 0.18, 0.14, 0.12, 0.20, 0.08],
            )
        )
        all_idx = np.arange(n_sat, dtype=np.int32)
        if move == "add_sense":
            pick = self._weighted_choice(all_idx[~sense], scores, rng, prefer_high=True)
            if pick is not None:
                sense[pick] = True
                comm[pick] = True
        elif move == "drop_sense":
            pick = self._weighted_choice(all_idx[sense], scores, rng, prefer_high=False)
            if pick is not None and int(sense.sum()) > 1:
                sense[pick] = False
        elif move == "swap_sense":
            drop = self._weighted_choice(all_idx[sense], scores, rng, prefer_high=False)
            add = self._weighted_choice(all_idx[~sense], scores, rng, prefer_high=True)
            if drop is not None and add is not None and int(sense.sum()) > 1:
                sense[drop] = False
                sense[add] = True
                comm[add] = True
        elif move == "add_comm":
            pick = self._weighted_choice(all_idx[~comm], scores, rng, prefer_high=True)
            if pick is not None:
                comm[pick] = True
        elif move == "drop_comm":
            optional = all_idx[comm & ~sense]
            pick = self._weighted_choice(optional, scores, rng, prefer_high=False)
            if pick is not None and int(comm.sum()) > int(sense.sum()):
                comm[pick] = False
        elif move == "swap_comm":
            optional = all_idx[comm & ~sense]
            inactive = all_idx[~comm]
            drop = self._weighted_choice(optional, scores, rng, prefer_high=False)
            add = self._weighted_choice(inactive, scores, rng, prefer_high=True)
            if drop is not None and add is not None:
                comm[drop] = False
                comm[add] = True
        else:
            optional = all_idx[comm & ~sense]
            if len(optional) > 0:
                order = optional[np.argsort(scores[optional])]
                n_drop = int(np.clip(rng.integers(1, 4), 1, len(order)))
                comm[order[:n_drop]] = False
        out = np.zeros(2 * n_sat, dtype=bool)
        out[:n_sat] = sense
        out[n_sat:] = comm | sense
        return out

    def optimize(
        self,
        eval_fn: Callable[[np.ndarray], Tuple[float, Tuple[int, int], Dict]],
        n_sat: int,
        dl: DataLoader,
        max_budget_s: float,
        checkpoint_s: List[float],
    ) -> Dict:
        rng = np.random.default_rng(int(self.cfg.RNG_SEED) + 23017)
        n_sat = int(n_sat)
        max_budget_s = float(max(max_budget_s, 0.01))
        checkpoints = sorted(float(x) for x in checkpoint_s if float(x) > 0.0)
        if not checkpoints:
            checkpoints = [max_budget_s]
        scores = _satellite_selection_scores(dl)
        cache: Dict[bytes, Tuple[float, Tuple[int, int], Dict]] = {}
        eval_count = 0
        t0 = time.perf_counter()

        best_gene: Optional[np.ndarray] = None
        best_summary: Dict = {}
        best_counts = (0, 0)
        best_utility = -np.inf
        current_gene: Optional[np.ndarray] = None
        current_summary: Dict = {}
        current_counts = (0, 0)
        current_utility = -np.inf
        recorded: List[Dict] = []
        next_checkpoint_idx = 0

        def evaluate_gene(gene: np.ndarray) -> Tuple[float, Tuple[int, int], Dict, bool]:
            nonlocal eval_count
            fixed = self._repair(gene, n_sat)
            key = fixed.tobytes()
            if key not in cache:
                util, counts, summary = eval_fn(fixed)
                cache[key] = (float(util), (int(counts[0]), int(counts[1])), dict(summary))
                eval_count += 1
                return cache[key][0], cache[key][1], cache[key][2], True
            return cache[key][0], cache[key][1], cache[key][2], False

        def record_due(force: bool = False) -> None:
            nonlocal next_checkpoint_idx
            elapsed = float(time.perf_counter() - t0)
            while next_checkpoint_idx < len(checkpoints) and (force or elapsed >= checkpoints[next_checkpoint_idx]):
                cp = checkpoints[next_checkpoint_idx]
                recorded.append(
                    {
                        "time_s": cp,
                        "elapsed_s": elapsed,
                        "best_utility": float(best_utility if np.isfinite(best_utility) else 0.0),
                        "best_gene": None if best_gene is None else best_gene.copy(),
                        "best_summary": dict(best_summary),
                        "best_counts": tuple(best_counts),
                        "evaluations": int(eval_count),
                    }
                )
                next_checkpoint_idx += 1
                if not force:
                    continue

        for gene in self._initial_pool(n_sat, dl, scores, rng):
            util, counts, summary, _ = evaluate_gene(gene)
            if current_gene is None or self._better(util, counts, current_utility, current_counts):
                current_gene = gene.copy()
                current_utility = float(util)
                current_counts = counts
                current_summary = dict(summary)
            if best_gene is None or self._better(util, counts, best_utility, best_counts):
                best_gene = gene.copy()
                best_utility = float(util)
                best_counts = counts
                best_summary = dict(summary)
            record_due()
            if float(time.perf_counter() - t0) >= max_budget_s:
                break

        if current_gene is None:
            current_gene = self._repair(np.zeros(2 * n_sat, dtype=bool), n_sat)
            current_utility, current_counts, current_summary, _ = evaluate_gene(current_gene)
            best_gene = current_gene.copy()
            best_utility = float(current_utility)
            best_counts = current_counts
            best_summary = dict(current_summary)

        temp0 = max(abs(float(best_utility)) * 0.03, 1000.0)
        while float(time.perf_counter() - t0) < max_budget_s:
            cand = self._mutate_local(current_gene, n_sat, scores, rng)
            util, counts, summary, _ = evaluate_gene(cand)
            elapsed = float(time.perf_counter() - t0)
            if self._better(util, counts, best_utility, best_counts):
                best_gene = cand.copy()
                best_utility = float(util)
                best_counts = counts
                best_summary = dict(summary)
            delta = float(util) - float(current_utility)
            remaining = max(0.0, 1.0 - elapsed / max_budget_s)
            temp = max(temp0 * remaining, 1e-6)
            accept = delta >= 0.0 or float(rng.random()) < float(np.exp(np.clip(delta / temp, -80.0, 20.0)))
            if accept:
                current_gene = cand.copy()
                current_utility = float(util)
                current_counts = counts
                current_summary = dict(summary)
            record_due()

        record_due(force=True)
        runtime_s = float(time.perf_counter() - t0)
        return {
            "best_gene": self._repair(best_gene if best_gene is not None else np.zeros(2 * n_sat, dtype=bool), n_sat),
            "best_summary": dict(best_summary),
            "best_utility": float(best_utility if np.isfinite(best_utility) else 0.0),
            "runtime_s": runtime_s,
            "evaluations": int(eval_count),
            "checkpoints": recorded,
        }


class ConstellationProblem:
    """Outer sparse activation problem solved by NSGA-III.

    NSGA-III returns sparse activation schemes for sensing and communication
    roles. It does not replace Greedy/Hungarian online scheduling, which remains
    inside SimulationEngine.evaluate() over the selected cross-layer topology.
    """

    def __init__(self, cfg: Config, dl: DataLoader):
        self.cfg = cfg
        self.dl = dl
        self.engine = SimulationEngine(cfg, dl)

    def evaluate(self, genes: np.ndarray) -> List[float]:
        _, counts, hist = self.engine.evaluate(genes, return_summary=True)
        n_sense, n_comm = counts
        summary = dict(hist.get("summary", {})) if hist else {}
        return [-float(summary.get("utility", 0.0)), float(n_sense), float(n_comm)]


def build_method_configs(dl: DataLoader, best_gene: np.ndarray) -> List[Dict]:
    full_gene = np.ones(2 * dl.N, dtype=bool)
    nsga_gene = np.asarray(best_gene, dtype=bool).copy()
    return [
        {
            "method_id": "M1",
            "method_name": "Full Graph + Greedy",
            "label": "M1 Full Graph + Greedy",
            "activation": "Full",
            "candidate_graph": "Full Graph",
            "matcher": "Greedy",
            "gene": full_gene,
            "matcher_mode": "greedy",
            "use_topk": False,
        },
        {
            "method_id": "M2",
            "method_name": "Top-K Graph + Greedy",
            "label": "M2 Top-K Graph + Greedy",
            "activation": "Full",
            "candidate_graph": "Top-K Graph",
            "matcher": "Greedy",
            "gene": full_gene,
            "matcher_mode": "greedy",
            "use_topk": True,
        },
        {
            "method_id": "M3",
            "method_name": "Full Graph + Hungarian",
            "label": "M3 Full Graph + Hungarian",
            "activation": "Full",
            "candidate_graph": "Full Graph",
            "matcher": "Hungarian",
            "gene": full_gene,
            "matcher_mode": "hungarian",
            "use_topk": False,
        },
        {
            "method_id": "M4",
            "method_name": "Top-K Graph + Hungarian",
            "label": "M4 Top-K Graph + Hungarian",
            "activation": "Full",
            "candidate_graph": "Top-K Graph",
            "matcher": "Hungarian",
            "gene": full_gene,
            "matcher_mode": "hungarian",
            "use_topk": True,
        },
        {
            "method_id": "M5",
            "method_name": "NSGA-III + Full Graph + Greedy",
            "label": "M5 NSGA-III + Full Graph + Greedy",
            "activation": "NSGA-III",
            "candidate_graph": "Full Graph",
            "matcher": "Greedy",
            "gene": nsga_gene,
            "matcher_mode": "greedy",
            "use_topk": False,
        },
        {
            "method_id": "M6",
            "method_name": "NSGA-III + Top-K Graph + Greedy",
            "label": "M6 NSGA-III + Top-K Graph + Greedy",
            "activation": "NSGA-III",
            "candidate_graph": "Top-K Graph",
            "matcher": "Greedy",
            "gene": nsga_gene,
            "matcher_mode": "greedy",
            "use_topk": True,
        },
        {
            "method_id": "M7",
            "method_name": "NSGA-III + Full Graph + Hungarian",
            "label": "M7 NSGA-III + Full Graph + Hungarian",
            "activation": "NSGA-III",
            "candidate_graph": "Full Graph",
            "matcher": "Hungarian",
            "gene": nsga_gene,
            "matcher_mode": "hungarian",
            "use_topk": False,
        },
        {
            "method_id": "M8",
            "method_name": "Proposed: NSGA-III + Top-K Graph + Hungarian",
            "label": "M8 Proposed: NSGA-III + Top-K Graph + Hungarian",
            "activation": "NSGA-III",
            "candidate_graph": "Top-K Graph",
            "matcher": "Hungarian",
            "gene": nsga_gene,
            "matcher_mode": "hungarian",
            "use_topk": True,
        },
    ]


def build_tbls_method_configs(dl: DataLoader, tbls_best_gene: np.ndarray) -> List[Dict]:
    tbls_gene = np.asarray(tbls_best_gene, dtype=bool).copy()
    return [
        {
            "method_id": "M9",
            "method_name": "TBLS + Full Graph + Greedy",
            "label": "M9 TBLS + Full Graph + Greedy",
            "activation": "TBLS",
            "candidate_graph": "Full Graph",
            "matcher": "Greedy",
            "gene": tbls_gene,
            "matcher_mode": "greedy",
            "use_topk": False,
        },
        {
            "method_id": "M10",
            "method_name": "TBLS + Top-K Graph + Greedy",
            "label": "M10 TBLS + Top-K Graph + Greedy",
            "activation": "TBLS",
            "candidate_graph": "Top-K Graph",
            "matcher": "Greedy",
            "gene": tbls_gene,
            "matcher_mode": "greedy",
            "use_topk": True,
        },
        {
            "method_id": "M11",
            "method_name": "TBLS + Full Graph + Hungarian",
            "label": "M11 TBLS + Full Graph + Hungarian",
            "activation": "TBLS",
            "candidate_graph": "Full Graph",
            "matcher": "Hungarian",
            "gene": tbls_gene,
            "matcher_mode": "hungarian",
            "use_topk": False,
        },
        {
            "method_id": "M12",
            "method_name": "TBLS + Top-K Graph + Hungarian",
            "label": "M12 TBLS + Top-K Graph + Hungarian",
            "activation": "TBLS",
            "candidate_graph": "Top-K Graph",
            "matcher": "Hungarian",
            "gene": tbls_gene,
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
        "M9": "#8C564B",
        "M10": "#17BECF",
        "M11": "#9467BD",
        "M12": "#FF7F0E",
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
        for _ in range(repeats - 1):
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
        summary["scoring"] = "Cross-layer score with Full/Top-K sparse candidate graph and Greedy/Hungarian online scheduling"
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
            f"  {item['label']:<45s} active={counts[1]:5d} "
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
                "方法编号": m.get("method_id", ""),
                "方法": m.get("label", ""),
                "感知角色激活数": m.get("sense_active_count", 0),
                "通信转发角色激活数": m.get("comm_active_count", 0),
                "活跃卫星数": m.get("active_satellite_count", 0),
                "候选通信转发节点数": m.get("comm_active_count", 0),
                "候选通信转发节点比例": m.get("candidate_comm_relay_ratio", 0.0),
                "实际参与转发唯一卫星数": m.get("actual_forward_unique_satellite_count", 0),
                "实际参与转发节点比例": m.get("actual_forward_participation_ratio", 0.0),
                "实际下传唯一卫星数": m.get("actual_downlink_unique_satellite_count", 0),
                "平均每时刻ISL发送卫星数": m.get("avg_actual_isl_send_count", 0.0),
                "最大每时刻ISL发送卫星数": m.get("max_actual_isl_send_count", 0),
                "平均每时刻ISL接收卫星数": m.get("avg_actual_isl_receive_count", 0.0),
                "最大每时刻ISL接收卫星数": m.get("max_actual_isl_receive_count", 0),
                "平均每时刻下传卫星数": m.get("avg_actual_downlink_count", 0.0),
                "最大每时刻下传卫星数": m.get("max_actual_downlink_count", 0),
                "平均交付中继跳数": m.get("avg_delivered_relay_hops", 0.0),
                "最大观测中继跳数": m.get("max_observed_relay_hops", 0),
                "交付数据量(MB)": m.get("delivered_mb", 0.0),
                "生成数据量(MB)": m.get("generated_mb", 0.0),
                "丢弃数据量(MB)": m.get("dropped_mb", 0.0),
                "TTL丢弃数据量(MB)": m.get("dropped_ttl_mb", 0.0),
                "熔断丢弃数据量(MB)": m.get("dropped_melt_mb", 0.0),
                "交付率(delivered/TASK_DATA_MB)": m.get("delivery_ratio", 0.0),
                "丢包率(dropped/generated)": m.get("packet_loss_rate", 0.0),
                "平均缓存积压(MB)": m.get("avg_queue_backlog_mb", 0.0),
                "完成时间(s)": m.get("completion_time_s"),
                "在线调度时间(s)": m.get("online_scheduling_time_s", 0.0),
                "单次仿真评估时间(s)": m.get("simulation_runtime_s", 0.0),
                "NSGA-III优化时间(s)": m.get("activation_runtime_s", 0.0),
                "方法总运行时间(s)": m.get("total_runtime_s", 0.0),
            }
        )
        table2.append(
            {
                "方法编号": m.get("method_id", ""),
                "方法": m.get("label", ""),
                "总能耗需求": m.get("total_energy_demand", 0.0),
                "单位交付能耗": m.get("unit_delivered_energy"),
                "平均剩余能量": m.get("avg_remaining_energy", 0.0),
                "最低剩余能量": m.get("min_remaining_energy", 0.0),
                "低电量保护次数": m.get("low_power_protection_count", 0),
                "熔断次数": m.get("melt_count", 0),
                "TTL丢弃数据量(MB)": m.get("dropped_ttl_mb", 0.0),
                "熔断丢弃数据量(MB)": m.get("dropped_melt_mb", 0.0),
                "最终缓存(MB)": m.get("final_buffer_mb", 0.0),
            }
        )
    pd.DataFrame(table1).to_csv(out_dir / "table1_method_performance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(table2).to_csv(out_dir / "table2_energy_protection.csv", index=False, encoding="utf-8-sig")
    print(f"  saved M1-M8 comparison tables in {out_dir}")


def _subset_dataloader(dl: DataLoader, cfg: Config, sat_idx: np.ndarray) -> DataLoader:
    sat_idx = np.asarray(sat_idx, dtype=np.int32)
    if len(sat_idx) == dl.N and np.array_equal(sat_idx, np.arange(dl.N, dtype=np.int32)):
        return dl
    sub = DataLoader.__new__(DataLoader)
    sub.cfg = cfg
    sub.sat_names = dl.sat_names[sat_idx].copy()
    sub.times_s = dl.times_s.copy()
    sub.T = int(dl.T)
    sub.N = int(len(sat_idx))
    sub.dt_eff = float(dl.dt_eff)
    sub.pos_icrf = dl.pos_icrf[:, sat_idx, :].copy()
    sub.vel_icrf = dl.vel_icrf[:, sat_idx, :].copy()
    sub.pos_ecef = dl.pos_ecef[:, sat_idx, :].copy()
    sub.vis_gs = dl.vis_gs[:, sat_idx].copy()
    sub.task_coverage_by_region = dl.task_coverage_by_region[:, :, sat_idx].copy()
    sub.task_coverage = np.any(sub.task_coverage_by_region, axis=0)
    sub.eclipse = dl.eclipse[:, sat_idx].copy()
    sub.task_region_weights = dl.task_region_weights.copy()
    sub.isl_neighbor_idx, sub.isl_neighbor_dist = _compute_isl_topk(
        sub.pos_icrf,
        int(cfg.ISL_NEIGHBOR_K),
        float(cfg.ISL_MAX_DIST),
        verbose=False,
    )
    return sub


def run_scalability_experiment(
    cfg: Config,
    dl: DataLoader,
    best_gene: np.ndarray,
    nsga_runtime_s: float = 0.0,
) -> Dict:
    _ = best_gene, nsga_runtime_s
    print("=" * 72)
    print("S4 Satellite-scale experiment for Fig1-Fig4")
    print("=" * 72)
    n_total = int(dl.N)
    if n_total <= 0:
        return {"nodes": [], "methods": [], "definitions": {}}

    rng = np.random.default_rng(int(cfg.RNG_SEED) + 20260519)
    node_targets = scale_node_targets(n_total)
    nested_order = rng.permutation(n_total).astype(np.int32)
    series: Dict[str, Dict] = {}

    for n in node_targets:
        if int(n) == n_total:
            sat_idx = np.arange(n_total, dtype=np.int32)
        else:
            sat_idx = np.sort(nested_order[: int(n)].astype(np.int32))
        sub_dl = _subset_dataloader(dl, cfg, sat_idx)
        print(f"  scale point nodes={sub_dl.N}: running NSGA-III activation optimization")
        scale_problem = ConstellationProblem(cfg, sub_dl)
        scale_solver = NSGA3Solver(cfg)
        scale_result = scale_solver.optimize(scale_problem.evaluate, sub_dl.N)
        best_i = select_best_objective_index(scale_result["F"], utility_tol=max(float(cfg.GA_CONV_EPS), 1e-6))
        scale_best_gene = scale_result["X"][best_i].astype(bool)
        scale_nsga_runtime = float(scale_result.get("runtime_s", 0.0))

        engine = SimulationEngine(cfg, sub_dl)
        for item in build_method_configs(sub_dl, scale_best_gene):
            if item["method_id"] not in series:
                series[item["method_id"]] = {
                    "method_id": item["method_id"],
                    "label": item["label"],
                    "activation": item["activation"],
                    "candidate_graph": item["candidate_graph"],
                    "matcher": item["matcher"],
                    "color": _method_color(item["method_id"]),
                    "online_scheduling_time_s": [],
                    "total_runtime_s": [],
                    "activation_ratio": [],
                    "candidate_comm_relay_ratio": [],
                    "actual_forward_participation_ratio": [],
                    "actual_forward_unique_satellite_count": [],
                    "actual_downlink_unique_satellite_count": [],
                    "avg_actual_isl_send_count": [],
                    "max_actual_isl_send_count": [],
                    "avg_actual_isl_receive_count": [],
                    "max_actual_isl_receive_count": [],
                    "avg_actual_downlink_count": [],
                    "max_actual_downlink_count": [],
                    "avg_delivered_relay_hops": [],
                    "max_observed_relay_hops": [],
                    "dropped_ttl_mb": [],
                    "dropped_melt_mb": [],
                    "delivery_ratio": [],
                    "packet_loss_rate": [],
                    "utility": [],
                }
            _, counts, hist = engine.evaluate(
                item["gene"],
                track_history=False,
                return_summary=True,
                matcher_mode=item["matcher_mode"],
                use_topk=item["use_topk"],
                use_window_urgency=True,
                use_energy_score=True,
            )
            summary = dict(hist.get("summary", {})) if hist else {}
            activation_runtime_s = scale_nsga_runtime if item["activation"] == "NSGA-III" else 0.0
            total_runtime_s = activation_runtime_s + float(summary.get("simulation_runtime_s", 0.0))
            series[item["method_id"]]["online_scheduling_time_s"].append(float(summary.get("online_scheduling_time_s", 0.0)))
            series[item["method_id"]]["total_runtime_s"].append(float(total_runtime_s))
            series[item["method_id"]]["activation_ratio"].append(float(counts[1] / max(sub_dl.N, 1)))
            series[item["method_id"]]["candidate_comm_relay_ratio"].append(float(summary.get("candidate_comm_relay_ratio", counts[1] / max(sub_dl.N, 1))))
            series[item["method_id"]]["actual_forward_participation_ratio"].append(float(summary.get("actual_forward_participation_ratio", 0.0)))
            series[item["method_id"]]["actual_forward_unique_satellite_count"].append(int(summary.get("actual_forward_unique_satellite_count", 0)))
            series[item["method_id"]]["actual_downlink_unique_satellite_count"].append(int(summary.get("actual_downlink_unique_satellite_count", 0)))
            series[item["method_id"]]["avg_actual_isl_send_count"].append(float(summary.get("avg_actual_isl_send_count", 0.0)))
            series[item["method_id"]]["max_actual_isl_send_count"].append(int(summary.get("max_actual_isl_send_count", 0)))
            series[item["method_id"]]["avg_actual_isl_receive_count"].append(float(summary.get("avg_actual_isl_receive_count", 0.0)))
            series[item["method_id"]]["max_actual_isl_receive_count"].append(int(summary.get("max_actual_isl_receive_count", 0)))
            series[item["method_id"]]["avg_actual_downlink_count"].append(float(summary.get("avg_actual_downlink_count", 0.0)))
            series[item["method_id"]]["max_actual_downlink_count"].append(int(summary.get("max_actual_downlink_count", 0)))
            series[item["method_id"]]["avg_delivered_relay_hops"].append(float(summary.get("avg_delivered_relay_hops", 0.0)))
            series[item["method_id"]]["max_observed_relay_hops"].append(int(summary.get("max_observed_relay_hops", 0)))
            series[item["method_id"]]["dropped_ttl_mb"].append(float(summary.get("dropped_ttl_mb", 0.0)))
            series[item["method_id"]]["dropped_melt_mb"].append(float(summary.get("dropped_melt_mb", 0.0)))
            series[item["method_id"]]["delivery_ratio"].append(float(summary.get("delivery_ratio", 0.0)))
            series[item["method_id"]]["packet_loss_rate"].append(float(summary.get("packet_loss_rate", 0.0)))
            series[item["method_id"]]["utility"].append(float(summary.get("utility", 0.0)))
        print(f"  scale point nodes={sub_dl.N} completed")

    return {
        "nodes": [int(x) for x in node_targets],
        "methods": [series[k] for k in sorted(series.keys())],
        "definitions": {
            "delivery_ratio": "delivered_mb/TASK_DATA_MB",
            "packet_loss_rate": "dropped_mb/generated_mb",
            "online_time": "online ISL matching plus Beijing ground-station downlink selection time",
            "total_runtime": "NSGA-III sparse activation runtime plus simulation evaluation runtime for NSGA-III methods; simulation runtime only for Full activation methods",
        },
    }


def save_scalability_outputs(out_dir: Path, scale: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = [int(x) for x in scale.get("nodes", [])] if scale else []
    rows = []
    for method in scale.get("methods", []) if scale else []:
        for i, n in enumerate(nodes):
            rows.append(
                {
                    "卫星数量": n,
                    "方法编号": method.get("method_id", ""),
                    "方法": method.get("label", ""),
                    "候选通信转发节点比例": method.get("candidate_comm_relay_ratio", [None] * len(nodes))[i],
                    "实际参与转发节点比例": method.get("actual_forward_participation_ratio", [None] * len(nodes))[i],
                    "实际参与转发唯一卫星数": method.get("actual_forward_unique_satellite_count", [None] * len(nodes))[i],
                    "实际下传唯一卫星数": method.get("actual_downlink_unique_satellite_count", [None] * len(nodes))[i],
                    "平均每时刻ISL发送卫星数": method.get("avg_actual_isl_send_count", [None] * len(nodes))[i],
                    "最大每时刻ISL发送卫星数": method.get("max_actual_isl_send_count", [None] * len(nodes))[i],
                    "平均每时刻ISL接收卫星数": method.get("avg_actual_isl_receive_count", [None] * len(nodes))[i],
                    "最大每时刻ISL接收卫星数": method.get("max_actual_isl_receive_count", [None] * len(nodes))[i],
                    "平均每时刻下传卫星数": method.get("avg_actual_downlink_count", [None] * len(nodes))[i],
                    "最大每时刻下传卫星数": method.get("max_actual_downlink_count", [None] * len(nodes))[i],
                    "平均交付中继跳数": method.get("avg_delivered_relay_hops", [None] * len(nodes))[i],
                    "最大观测中继跳数": method.get("max_observed_relay_hops", [None] * len(nodes))[i],
                    "TTL丢弃数据量(MB)": method.get("dropped_ttl_mb", [None] * len(nodes))[i],
                    "熔断丢弃数据量(MB)": method.get("dropped_melt_mb", [None] * len(nodes))[i],
                    "在线调度时间(s)": method.get("online_scheduling_time_s", [None] * len(nodes))[i],
                    "方法总运行时间(s)": method.get("total_runtime_s", [None] * len(nodes))[i],
                    "交付率": method.get("delivery_ratio", [None] * len(nodes))[i],
                    "丢包率": method.get("packet_loss_rate", [None] * len(nodes))[i],
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "table6_scalability_actual_participation.csv", index=False, encoding="utf-8-sig")
    print(f"  saved scalability actual participation table in {out_dir}")


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    span = float(np.ptp(arr))
    if span <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - float(np.min(arr))) / span


def _make_gene(n_sat: int, sense_idx: np.ndarray, comm_idx: np.ndarray) -> np.ndarray:
    gene = np.zeros(2 * int(n_sat), dtype=bool)
    sense_idx = np.asarray(sense_idx, dtype=np.int32)
    comm_idx = np.asarray(comm_idx, dtype=np.int32)
    comm_idx = np.unique(np.concatenate([comm_idx, sense_idx])).astype(np.int32)
    gene[sense_idx] = True
    gene[int(n_sat) + comm_idx] = True
    return gene


def _satellite_selection_scores(dl: DataLoader) -> np.ndarray:
    region_cov = dl.task_coverage_by_region.mean(axis=1)
    weighted_coverage = dl.task_region_weights @ region_cov
    gs_vis = dl.vis_gs.mean(axis=0).astype(np.float64)
    degree = np.isfinite(dl.isl_neighbor_dist).sum(axis=2).mean(axis=0).astype(np.float64)
    return (
        0.50 * _normalize_vector(weighted_coverage)
        + 0.30 * _normalize_vector(gs_vis)
        + 0.20 * _normalize_vector(degree)
    )


def _summary_for_experiment(method: str, ratio: Optional[float], summary: Dict, counts: Tuple[int, int], extra: Optional[Dict] = None) -> Dict:
    row = {
        "method": method,
        "activation_ratio": None if ratio is None else float(ratio),
        "sense_active_count": int(counts[0]),
        "comm_active_count": int(counts[1]),
        "utility": float(summary.get("utility", 0.0)),
        "delivery_ratio": float(summary.get("delivery_ratio", 0.0)),
        "packet_loss_rate": float(summary.get("packet_loss_rate", 0.0)),
        "online_scheduling_time_s": float(summary.get("online_scheduling_time_s", 0.0)),
        "simulation_runtime_s": float(summary.get("simulation_runtime_s", 0.0)),
        "unit_delivered_energy": summary.get("unit_delivered_energy"),
        "low_power_protection_count": int(summary.get("low_power_protection_count", 0)),
        "candidate_comm_relay_ratio": float(summary.get("candidate_comm_relay_ratio", 0.0)),
        "actual_forward_unique_satellite_count": int(summary.get("actual_forward_unique_satellite_count", 0)),
        "actual_forward_participation_ratio": float(summary.get("actual_forward_participation_ratio", 0.0)),
        "actual_downlink_unique_satellite_count": int(summary.get("actual_downlink_unique_satellite_count", 0)),
        "avg_actual_isl_send_count": float(summary.get("avg_actual_isl_send_count", 0.0)),
        "max_actual_isl_send_count": int(summary.get("max_actual_isl_send_count", 0)),
        "avg_actual_isl_receive_count": float(summary.get("avg_actual_isl_receive_count", 0.0)),
        "max_actual_isl_receive_count": int(summary.get("max_actual_isl_receive_count", 0)),
        "avg_actual_downlink_count": float(summary.get("avg_actual_downlink_count", 0.0)),
        "max_actual_downlink_count": int(summary.get("max_actual_downlink_count", 0)),
        "avg_delivered_relay_hops": float(summary.get("avg_delivered_relay_hops", 0.0)),
        "max_observed_relay_hops": int(summary.get("max_observed_relay_hops", 0)),
        "dropped_ttl_mb": float(summary.get("dropped_ttl_mb", 0.0)),
        "dropped_melt_mb": float(summary.get("dropped_melt_mb", 0.0)),
    }
    if extra:
        row.update(extra)
    return row


def _budget_constrained_nsga_gene(
    cfg: Config,
    dl: DataLoader,
    comm_budget: int,
    sense_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(seed)
    n_sat = int(dl.N)
    comm_budget = int(np.clip(comm_budget, 1, n_sat))
    sense_target = int(np.clip(round(comm_budget * float(sense_fraction)), 1, comm_budget))
    pop = int(max(1, cfg.GA_POP))
    gen = int(max(0, cfg.GA_GEN))
    n_var = 2 * n_sat
    pm = 1.0 / max(n_var, 1)
    engine = SimulationEngine(cfg, dl)
    solver = NSGA3Solver(cfg)
    refs = solver._ref_dirs(3, int(cfg.GA_PARTITIONS))

    def repair_budget(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=bool).copy()
        sense = y[:n_sat].copy()
        comm = y[n_sat:].copy() | sense
        if int(sense.sum()) > comm_budget:
            keep = rng.choice(np.where(sense)[0], comm_budget, replace=False)
            new_sense = np.zeros(n_sat, dtype=bool)
            new_sense[keep] = True
            sense = new_sense
            comm &= sense | (rng.random(n_sat) < 0.0)
            comm[keep] = True
        if int(comm.sum()) > comm_budget:
            comm_idx = np.where(comm)[0]
            sense_idx = np.where(sense)[0]
            optional = np.setdiff1d(comm_idx, sense_idx, assume_unique=False)
            drop_count = int(comm.sum()) - comm_budget
            if drop_count > 0 and len(optional) > 0:
                drop = rng.choice(optional, min(drop_count, len(optional)), replace=False)
                comm[drop] = False
            if int(comm.sum()) > comm_budget:
                keep = rng.choice(np.where(comm)[0], comm_budget, replace=False)
                new_comm = np.zeros(n_sat, dtype=bool)
                new_comm[keep] = True
                comm = new_comm
                sense &= comm
        comm |= sense
        out = np.zeros(n_var, dtype=bool)
        out[:n_sat] = sense
        out[n_sat:] = comm
        return out

    def random_budget_gene() -> np.ndarray:
        comm_idx = rng.choice(n_sat, comm_budget, replace=False)
        sense_idx = rng.choice(comm_idx, min(sense_target, len(comm_idx)), replace=False)
        return _make_gene(n_sat, sense_idx, comm_idx)

    X = np.vstack([random_budget_gene() for _ in range(pop)]).astype(float)

    def eval_budget(x: np.ndarray) -> List[float]:
        gene = repair_budget(x)
        _, counts, hist = engine.evaluate(
            gene,
            return_summary=True,
            matcher_mode="hungarian",
            use_topk=True,
            use_window_urgency=True,
            use_energy_score=True,
        )
        summary = dict(hist.get("summary", {})) if hist else {}
        return [-float(summary.get("utility", 0.0)), float(counts[0]), float(counts[1])]

    t0 = time.perf_counter()
    F = np.asarray([eval_budget(X[i]) for i in range(pop)], dtype=float)
    for _ in range(gen):
        off_X = []
        idx = rng.permutation(pop)
        for k in range(0, pop, 2):
            p1 = X[idx[k]]
            p2 = X[idx[min(k + 1, pop - 1)]]
            if rng.random() < cfg.GA_CR:
                mask = rng.random(n_var) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for child in (c1, c2):
                child_bool = np.logical_xor(child.astype(bool), rng.random(n_var) < pm)
                off_X.append(repair_budget(child_bool).astype(float))
        off_X = np.asarray(off_X[:pop], dtype=float)
        off_F = np.asarray([eval_budget(off_X[i]) for i in range(pop)], dtype=float)
        X, F = solver._survive(np.vstack([X, off_X]), np.vstack([F, off_F]), pop, refs)
        X = np.asarray([repair_budget(x).astype(float) for x in X], dtype=float)
    best_i = select_best_objective_index(F, utility_tol=max(float(cfg.GA_CONV_EPS), 1e-6))
    gene = repair_budget(X[best_i]).astype(bool)
    return gene, {"runtime_s": float(time.perf_counter() - t0), "evaluations": int(pop * (1 + gen))}


def run_topk_sensitivity_experiment(cfg: Config, dl: DataLoader, best_gene: np.ndarray) -> Dict:
    print("=" * 72)
    print("S6 Top-K efficiency-performance experiment")
    print("=" * 72)
    engine = SimulationEngine(cfg, dl)
    rows: List[Dict] = []
    for k in TOPK_SENSITIVITY_VALUES:
        _, counts, hist = engine.evaluate(
            best_gene,
            return_summary=True,
            matcher_mode="hungarian",
            use_topk=True,
            fixed_topk_k=int(k),
        )
        summary = dict(hist.get("summary", {})) if hist else {}
        rows.append(_summary_for_experiment(f"Top-K K={k}", None, summary, counts, {"k": int(k), "graph": "Top-K", "scenario": "Stress"}))
    _, counts, hist = engine.evaluate(best_gene, return_summary=True, matcher_mode="hungarian", use_topk=False)
    summary = dict(hist.get("summary", {})) if hist else {}
    rows.append(_summary_for_experiment("Full Graph", None, summary, counts, {"k": "Full Graph", "graph": "Full Graph", "scenario": "Stress"}))
    full_utility = max(float(rows[-1].get("utility", 0.0)), 1e-9)
    full_online = max(float(rows[-1].get("online_scheduling_time_s", 0.0)), 1e-9)
    for row in rows:
        row["normalized_utility_vs_full"] = float(row.get("utility", 0.0)) / full_utility
        row["speedup_vs_full"] = full_online / max(float(row.get("online_scheduling_time_s", 0.0)), 1e-9)
    return {"rows": rows, "scenario": "Stress"}


def run_sparse_activation_curve_experiment(cfg: Config, dl: DataLoader, best_gene: np.ndarray) -> Dict:
    print("=" * 72)
    print("S7 Sparse activation utility curve experiment")
    print("=" * 72)
    rng = np.random.default_rng(int(cfg.RNG_SEED) + 20260701)
    engine = SimulationEngine(cfg, dl)
    n_sat = int(dl.N)
    scores = _satellite_selection_scores(dl)
    order = np.argsort(-scores)
    base_sense = int(best_gene[:n_sat].sum())
    base_comm = int((best_gene[n_sat:] | best_gene[:n_sat]).sum())
    sense_fraction = float(np.clip(base_sense / max(base_comm, 1), 0.10, 0.75))
    rows: List[Dict] = []

    full_gene = np.ones(2 * n_sat, dtype=bool)
    _, full_counts, full_hist = engine.evaluate(full_gene, return_summary=True, matcher_mode="hungarian", use_topk=True)
    full_summary = dict(full_hist.get("summary", {})) if full_hist else {}

    for ratio in SPARSE_ACTIVATION_RATIOS:
        comm_count = int(np.clip(round(float(ratio) * n_sat), 1, n_sat))
        sense_count = int(np.clip(round(comm_count * sense_fraction), 1, comm_count))

        random_rows = []
        for rep in range(SPARSE_RANDOM_REPEATS):
            comm_idx = rng.choice(n_sat, comm_count, replace=False)
            sense_idx = rng.choice(comm_idx, sense_count, replace=False)
            gene = _make_gene(n_sat, sense_idx, comm_idx)
            _, counts, hist = engine.evaluate(gene, return_summary=True, matcher_mode="hungarian", use_topk=True)
            summary = dict(hist.get("summary", {})) if hist else {}
            random_rows.append(_summary_for_experiment("Random Sparse", ratio, summary, counts, {"repeat": rep + 1}))
        random_df = pd.DataFrame(random_rows)
        median_random = random_df.median(numeric_only=True).to_dict()
        rows.append(
            {
                "method": "Random Sparse",
                "activation_ratio": float(ratio),
                "sense_active_count": int(median_random.get("sense_active_count", sense_count)),
                "comm_active_count": int(median_random.get("comm_active_count", comm_count)),
                "utility": float(median_random.get("utility", 0.0)),
                "utility_q25": float(random_df["utility"].quantile(0.25)),
                "utility_q75": float(random_df["utility"].quantile(0.75)),
                "delivery_ratio": float(median_random.get("delivery_ratio", 0.0)),
                "delivery_ratio_q25": float(random_df["delivery_ratio"].quantile(0.25)),
                "delivery_ratio_q75": float(random_df["delivery_ratio"].quantile(0.75)),
                "packet_loss_rate": float(median_random.get("packet_loss_rate", 0.0)),
                "online_scheduling_time_s": float(median_random.get("online_scheduling_time_s", 0.0)),
                "simulation_runtime_s": float(median_random.get("simulation_runtime_s", 0.0)),
                "scenario": "Stress",
            }
        )

        comm_idx = order[:comm_count]
        sense_idx = comm_idx[np.argsort(-scores[comm_idx])[:sense_count]]
        heuristic_gene = _make_gene(n_sat, sense_idx, comm_idx)
        _, counts, hist = engine.evaluate(heuristic_gene, return_summary=True, matcher_mode="hungarian", use_topk=True)
        summary = dict(hist.get("summary", {})) if hist else {}
        row = _summary_for_experiment("Coverage Heuristic", ratio, summary, counts, {"scenario": "Stress"})
        row["utility_q25"] = row["utility_q75"] = row["utility"]
        row["delivery_ratio_q25"] = row["delivery_ratio_q75"] = row["delivery_ratio"]
        rows.append(row)

        nsga_gene, nsga_meta = _budget_constrained_nsga_gene(
            cfg,
            dl,
            comm_count,
            sense_fraction,
            seed=int(cfg.RNG_SEED) + int(round(ratio * 1000)),
        )
        _, counts, hist = engine.evaluate(nsga_gene, return_summary=True, matcher_mode="hungarian", use_topk=True)
        summary = dict(hist.get("summary", {})) if hist else {}
        row = _summary_for_experiment("Budget-constrained NSGA-III", ratio, summary, counts, {**nsga_meta, "scenario": "Stress"})
        row["utility_q25"] = row["utility_q75"] = row["utility"]
        row["delivery_ratio_q25"] = row["delivery_ratio_q75"] = row["delivery_ratio"]
        rows.append(row)

        row = _summary_for_experiment("Full Activation Reference", ratio, full_summary, full_counts, {"scenario": "Stress"})
        row["utility_q25"] = row["utility_q75"] = row["utility"]
        row["delivery_ratio_q25"] = row["delivery_ratio_q75"] = row["delivery_ratio"]
        rows.append(row)
        print(f"  activation ratio={ratio:.1f} completed")
    return {"rows": rows}


def run_ablation_experiment(cfg: Config, dl: DataLoader, best_gene: np.ndarray) -> Dict:
    print("=" * 72)
    print("S8 Cross-layer scoring ablation experiment")
    print("=" * 72)
    engine = SimulationEngine(cfg, dl)
    rows: List[Dict] = []
    for name, flags in ABLATION_CONFIGS:
        _, counts, hist = engine.evaluate(
            best_gene,
            return_summary=True,
            matcher_mode="hungarian",
            use_topk=True,
            use_energy_score=True,
            score_flags=flags,
        )
        summary = dict(hist.get("summary", {})) if hist else {}
        rows.append(_summary_for_experiment(name, None, summary, counts, {"score_flags": str(flags), "scenario": "Stress"}))
        print(f"  ablation {name} completed")
    _, counts, hist = engine.evaluate(
        best_gene,
        return_summary=True,
        matcher_mode="hungarian",
        use_topk=True,
        use_energy_score=True,
        score_flags={"future_gs": False, "energy": False, "task_priority": False, "buffer_pressure": False},
    )
    summary = dict(hist.get("summary", {})) if hist else {}
    rows.append(_summary_for_experiment("Distance Only", None, summary, counts, {"score_flags": "all cross-layer flags disabled", "scenario": "Stress"}))
    _, counts, hist = engine.evaluate(
        best_gene,
        return_summary=True,
        matcher_mode="hungarian",
        use_topk=True,
        use_energy_score=False,
    )
    summary = dict(hist.get("summary", {})) if hist else {}
    rows.append(_summary_for_experiment("No Cross-layer Score", None, summary, counts, {"score_flags": "legacy distance/free/downlink score", "scenario": "Stress"}))
    prop = rows[0]
    prop_u = max(abs(float(prop.get("utility", 0.0))), 1e-9)
    prop_d = float(prop.get("delivery_ratio", 0.0))
    prop_loss = float(prop.get("packet_loss_rate", 0.0))
    prop_energy = float(prop.get("unit_delivered_energy") or 0.0)
    prop_online = float(prop.get("online_scheduling_time_s", 0.0))
    for row in rows:
        row["utility_change_pct_vs_proposed"] = (float(row.get("utility", 0.0)) - float(prop.get("utility", 0.0))) / prop_u * 100.0
        row["delivery_ratio_change_vs_proposed"] = float(row.get("delivery_ratio", 0.0)) - prop_d
        row["packet_loss_increase_vs_proposed"] = float(row.get("packet_loss_rate", 0.0)) - prop_loss
        row["unit_energy_change_pct_vs_proposed"] = (
            (float(row.get("unit_delivered_energy") or 0.0) - prop_energy) / max(abs(prop_energy), 1e-9) * 100.0
        )
        row["online_time_change_pct_vs_proposed"] = (
            (float(row.get("online_scheduling_time_s", 0.0)) - prop_online) / max(abs(prop_online), 1e-9) * 100.0
        )
    return {"rows": rows}


def run_matcher_quality_experiment(scale: Dict) -> Dict:
    rows: List[Dict] = []
    nodes = scale.get("nodes", []) if scale else []
    wanted = {"M1", "M2", "M3", "M4"}
    for method in scale.get("methods", []) if scale else []:
        if method.get("method_id") not in wanted:
            continue
        for i, node in enumerate(nodes):
            rows.append(
                {
                    "satellite_count": int(node),
                    "method_id": method.get("method_id", ""),
                    "label": method.get("label", ""),
                    "utility": float(method.get("utility", [0.0] * len(nodes))[i]),
                    "delivery_ratio": float(method.get("delivery_ratio", [0.0] * len(nodes))[i]),
                    "online_scheduling_time_s": float(method.get("online_scheduling_time_s", [0.0] * len(nodes))[i]),
                }
            )
    return {"rows": rows}


def run_energy_safety_timeseries_experiment(cfg: Config, dl: DataLoader, best_gene: np.ndarray) -> Dict:
    engine = SimulationEngine(cfg, dl)
    full_gene = np.ones(2 * dl.N, dtype=bool)
    cases = [
        ("Proposed", best_gene, True, None),
        ("Full Activation", full_gene, True, None),
        ("No Energy Score", best_gene, True, {"future_gs": True, "energy": False, "task_priority": True, "buffer_pressure": True}),
    ]
    series = []
    for label, gene, use_energy_score, score_flags in cases:
        _, _, hist = engine.evaluate(
            gene,
            track_history=True,
            return_summary=True,
            matcher_mode="hungarian",
            use_topk=True,
            use_energy_score=use_energy_score,
            score_flags=score_flags,
        )
        series.append({"label": label, "history": hist, "summary": dict(hist.get("summary", {})) if hist else {}})
    return {
        "series": series,
        "e_melt": float(cfg.E_MELT),
        "e_warn": float(cfg.E_MELT * cfg.E_WARN_RATIO),
        "scenario": "Stress",
    }


def _time_budget_random_search(
    cfg: Config,
    dl: DataLoader,
    eval_fn: Callable[[np.ndarray], Tuple[float, Tuple[int, int], Dict]],
    max_budget_s: float,
    checkpoint_s: List[float],
) -> Dict:
    rng = np.random.default_rng(int(cfg.RNG_SEED) + 41077)
    n_sat = int(dl.N)
    max_budget_s = float(max(max_budget_s, 0.01))
    checkpoints = sorted(float(x) for x in checkpoint_s if float(x) > 0.0)
    if not checkpoints:
        checkpoints = [max_budget_s]
    t0 = time.perf_counter()
    cache: Dict[bytes, Tuple[float, Tuple[int, int], Dict]] = {}
    eval_count = 0
    best_gene: Optional[np.ndarray] = None
    best_summary: Dict = {}
    best_counts = (0, 0)
    best_utility = -np.inf
    recorded: List[Dict] = []
    next_checkpoint_idx = 0

    def repair(gene: np.ndarray) -> np.ndarray:
        out = np.asarray(gene, dtype=bool).copy()
        out[n_sat:] |= out[:n_sat]
        return out

    def evaluate(gene: np.ndarray) -> Tuple[float, Tuple[int, int], Dict]:
        nonlocal eval_count
        fixed = repair(gene)
        key = fixed.tobytes()
        if key not in cache:
            util, counts, summary = eval_fn(fixed)
            cache[key] = (float(util), (int(counts[0]), int(counts[1])), dict(summary))
            eval_count += 1
        return cache[key]

    def better(u: float, c: Tuple[int, int], bu: float, bc: Tuple[int, int]) -> bool:
        return TBLSOptimizer._better(u, c, bu, bc)

    def random_gene() -> np.ndarray:
        ratio = float(rng.choice([0.15, 0.25, 0.35, 0.50, 0.65]))
        comm_count = int(np.clip(round(n_sat * ratio), 1, n_sat))
        sense_fraction = float(rng.uniform(0.15, 0.45))
        sense_count = int(np.clip(round(comm_count * sense_fraction), 1, comm_count))
        comm_idx = rng.choice(n_sat, size=comm_count, replace=False)
        sense_idx = rng.choice(comm_idx, size=sense_count, replace=False)
        return _make_gene(n_sat, sense_idx, comm_idx)

    def record_due(force: bool = False) -> None:
        nonlocal next_checkpoint_idx
        elapsed = float(time.perf_counter() - t0)
        while next_checkpoint_idx < len(checkpoints) and (force or elapsed >= checkpoints[next_checkpoint_idx]):
            cp = checkpoints[next_checkpoint_idx]
            recorded.append(
                {
                    "time_s": cp,
                    "elapsed_s": elapsed,
                    "best_utility": float(best_utility if np.isfinite(best_utility) else 0.0),
                    "best_gene": None if best_gene is None else best_gene.copy(),
                    "best_summary": dict(best_summary),
                    "best_counts": tuple(best_counts),
                    "evaluations": int(eval_count),
                }
            )
            next_checkpoint_idx += 1

    while float(time.perf_counter() - t0) < max_budget_s or best_gene is None:
        gene = random_gene()
        util, counts, summary = evaluate(gene)
        if best_gene is None or better(util, counts, best_utility, best_counts):
            best_gene = gene.copy()
            best_utility = float(util)
            best_counts = counts
            best_summary = dict(summary)
        record_due()
        if float(time.perf_counter() - t0) >= max_budget_s:
            break
    record_due(force=True)
    runtime_s = float(time.perf_counter() - t0)
    return {
        "best_gene": repair(best_gene if best_gene is not None else np.zeros(2 * n_sat, dtype=bool)),
        "best_summary": dict(best_summary),
        "best_utility": float(best_utility if np.isfinite(best_utility) else 0.0),
        "runtime_s": runtime_s,
        "evaluations": int(eval_count),
        "checkpoints": recorded,
    }


def _coverage_heuristic_reference_gene(dl: DataLoader, reference_gene: np.ndarray) -> np.ndarray:
    n_sat = int(dl.N)
    scores = _satellite_selection_scores(dl)
    order = np.argsort(-scores)
    ref = np.asarray(reference_gene, dtype=bool)
    if len(ref) >= 2 * n_sat:
        ref_sense = int(ref[:n_sat].sum())
        ref_comm = int((ref[n_sat:] | ref[:n_sat]).sum())
    else:
        ref_sense = int(round(0.12 * n_sat))
        ref_comm = int(round(0.35 * n_sat))
    comm_count = int(np.clip(ref_comm if ref_comm > 0 else round(0.35 * n_sat), 1, n_sat))
    sense_count = int(np.clip(ref_sense if ref_sense > 0 else round(0.30 * comm_count), 1, comm_count))
    return _make_gene(n_sat, order[:sense_count], order[:comm_count])


def _checkpoint_at_ratio(result: Dict, ratio: float, base_budget_s: float) -> Dict:
    target = float(base_budget_s) * float(ratio)
    checkpoints = list(result.get("checkpoints", []))
    if not checkpoints:
        return {}
    eligible = [cp for cp in checkpoints if float(cp.get("time_s", 0.0)) <= target + 1e-9]
    return eligible[-1] if eligible else checkpoints[0]


def _time_budget_method_row(
    method_id: str,
    method_name: str,
    optimizer: str,
    ratio: float,
    budget_s: float,
    opt_runtime_s: float,
    evaluations: int,
    summary: Dict,
    counts: Tuple[int, int],
) -> Dict:
    row = _summary_for_experiment(
        method_name,
        None,
        summary,
        counts,
        extra={
            "method_id": method_id,
            "optimizer": optimizer,
            "time_budget_ratio": float(ratio),
            "time_budget_s": float(budget_s),
            "actual_optimization_time_s": float(opt_runtime_s),
            "optimizer_evaluations": int(evaluations),
        },
    )
    row["total_runtime_s"] = float(opt_runtime_s) + float(summary.get("simulation_runtime_s", 0.0))
    return row


def run_time_budget_optimizer_experiment(
    cfg: Config,
    dl: DataLoader,
    nsga_result: Dict,
    nsga_best_gene: np.ndarray,
    nsga_runtime_s: float,
) -> Dict:
    print("=" * 72)
    print("S7 Time-budget optimizer comparison: NSGA-III vs TBLS")
    print("=" * 72)
    base_budget_s = float(max(float(nsga_runtime_s), 0.05))
    budget_ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
    checkpoint_s = [base_budget_s * ratio for ratio in budget_ratios]
    max_budget_s = float(max(checkpoint_s))
    engine = SimulationEngine(cfg, dl)

    def eval_activation(gene: np.ndarray) -> Tuple[float, Tuple[int, int], Dict]:
        utility, counts, hist = engine.evaluate(
            gene,
            track_history=False,
            return_summary=True,
            matcher_mode="hungarian",
            use_topk=True,
            use_window_urgency=True,
            use_energy_score=True,
        )
        summary = dict(hist.get("summary", {})) if hist else {}
        return float(summary.get("utility", utility)), counts, summary

    nsga_util, nsga_counts, nsga_summary = eval_activation(np.asarray(nsga_best_gene, dtype=bool))
    print(f"  baseline NSGA-III optimizer time: {base_budget_s:.3f}s")
    tbls_result = TBLSOptimizer(cfg).optimize(eval_activation, dl.N, dl, max_budget_s, checkpoint_s)
    random_result = _time_budget_random_search(cfg, dl, eval_activation, max_budget_s, checkpoint_s)
    coverage_gene = _coverage_heuristic_reference_gene(dl, nsga_best_gene)
    coverage_util, coverage_counts, coverage_summary = eval_activation(coverage_gene)
    coverage_eval_runtime = float(coverage_summary.get("simulation_runtime_s", 0.0))

    curve_rows: List[Dict] = []
    hist = list(nsga_result.get("history", []))
    hist_points = []
    best_so_far = -np.inf
    for item in sorted(hist, key=lambda h: float(h.get("elapsed_s", 0.0))):
        best_so_far = max(best_so_far, float(item.get("best_util", 0.0)))
        hist_points.append((float(item.get("elapsed_s", 0.0)), float(best_so_far)))
    for ratio, budget in zip(budget_ratios, checkpoint_s):
        nsga_curve_util = float(nsga_util)
        for elapsed, util in hist_points:
            if elapsed <= budget + 1e-9:
                nsga_curve_util = max(nsga_curve_util if budget >= base_budget_s else -np.inf, util)
        if budget >= base_budget_s:
            nsga_curve_util = max(nsga_curve_util, float(nsga_util))
        elif hist_points:
            eligible = [util for elapsed, util in hist_points if elapsed <= budget + 1e-9]
            nsga_curve_util = float(max(eligible)) if eligible else float(hist_points[0][1])
        curve_rows.append(
            {
                "optimizer": "NSGA-III",
                "time_budget_ratio": float(ratio),
                "time_budget_s": float(budget),
                "best_utility": float(nsga_curve_util),
            }
        )
        tbls_cp = _checkpoint_at_ratio(tbls_result, ratio, base_budget_s)
        random_cp = _checkpoint_at_ratio(random_result, ratio, base_budget_s)
        curve_rows.append(
            {
                "optimizer": "TBLS",
                "time_budget_ratio": float(ratio),
                "time_budget_s": float(budget),
                "best_utility": float(tbls_cp.get("best_utility", tbls_result.get("best_utility", 0.0))),
            }
        )
        curve_rows.append(
            {
                "optimizer": "Random Search",
                "time_budget_ratio": float(ratio),
                "time_budget_s": float(budget),
                "best_utility": float(random_cp.get("best_utility", random_result.get("best_utility", 0.0))),
            }
        )
        curve_rows.append(
            {
                "optimizer": "Coverage Heuristic",
                "time_budget_ratio": float(ratio),
                "time_budget_s": float(budget),
                "best_utility": float(coverage_util),
            }
        )

    method_rows: List[Dict] = [
        _time_budget_method_row(
            "M8",
            "NSGA-III + Top-K Graph + Hungarian",
            "NSGA-III",
            1.0,
            base_budget_s,
            float(nsga_runtime_s),
            int(nsga_result.get("evaluations", 0)),
            nsga_summary,
            nsga_counts,
        )
    ]
    for ratio, budget in zip(budget_ratios, checkpoint_s):
        tbls_cp = _checkpoint_at_ratio(tbls_result, ratio, base_budget_s)
        if tbls_cp:
            method_rows.append(
                _time_budget_method_row(
                    "M12",
                    "TBLS + Top-K Graph + Hungarian",
                    "TBLS",
                    ratio,
                    budget,
                    float(tbls_cp.get("elapsed_s", budget)),
                    int(tbls_cp.get("evaluations", tbls_result.get("evaluations", 0))),
                    dict(tbls_cp.get("best_summary", {})),
                    tuple(tbls_cp.get("best_counts", (0, 0))),
                )
            )
        random_cp = _checkpoint_at_ratio(random_result, ratio, base_budget_s)
        if random_cp:
            method_rows.append(
                _time_budget_method_row(
                    "R1",
                    "Random Search + Top-K Graph + Hungarian",
                    "Random Search",
                    ratio,
                    budget,
                    float(random_cp.get("elapsed_s", budget)),
                    int(random_cp.get("evaluations", random_result.get("evaluations", 0))),
                    dict(random_cp.get("best_summary", {})),
                    tuple(random_cp.get("best_counts", (0, 0))),
                )
            )
        method_rows.append(
            _time_budget_method_row(
                "H1",
                "Coverage Heuristic + Top-K Graph + Hungarian",
                "Coverage Heuristic",
                ratio,
                budget,
                coverage_eval_runtime,
                1,
                coverage_summary,
                coverage_counts,
            )
        )

    tbls_1x_cp = _checkpoint_at_ratio(tbls_result, 1.0, base_budget_s)
    tbls_1x_gene = tbls_1x_cp.get("best_gene") if tbls_1x_cp else None
    if tbls_1x_gene is None:
        tbls_1x_gene = tbls_result.get("best_gene")
    for method in build_tbls_method_configs(dl, np.asarray(tbls_1x_gene, dtype=bool)):
        utility, counts, hist_eval = engine.evaluate(
            method["gene"],
            track_history=False,
            return_summary=True,
            matcher_mode=method["matcher_mode"],
            use_topk=bool(method["use_topk"]),
        )
        summary = dict(hist_eval.get("summary", {})) if hist_eval else {}
        summary["utility"] = float(summary.get("utility", utility))
        method_rows.append(
            _time_budget_method_row(
                str(method["method_id"]),
                str(method["method_name"]),
                "TBLS",
                1.0,
                base_budget_s,
                float(tbls_1x_cp.get("elapsed_s", base_budget_s)) if tbls_1x_cp else float(tbls_result.get("runtime_s", 0.0)),
                int(tbls_1x_cp.get("evaluations", tbls_result.get("evaluations", 0))) if tbls_1x_cp else int(tbls_result.get("evaluations", 0)),
                summary,
                counts,
            )
        )

    print(
        "  fair 1.0x utility: "
        f"NSGA={nsga_summary.get('utility', nsga_util):.1f}, "
        f"TBLS={tbls_1x_cp.get('best_utility', 0.0) if tbls_1x_cp else tbls_result.get('best_utility', 0.0):.1f}, "
        f"Random={_checkpoint_at_ratio(random_result, 1.0, base_budget_s).get('best_utility', 0.0):.1f}, "
        f"Coverage={coverage_util:.1f}"
    )
    return {
        "base_budget_s": base_budget_s,
        "budget_ratios": budget_ratios,
        "checkpoint_s": checkpoint_s,
        "curve_rows": curve_rows,
        "method_rows": method_rows,
        "tbls_result": tbls_result,
        "random_result": random_result,
        "coverage_result": {
            "best_gene": coverage_gene,
            "best_utility": float(coverage_util),
            "best_summary": coverage_summary,
            "best_counts": coverage_counts,
            "runtime_s": coverage_eval_runtime,
            "evaluations": 1,
        },
    }


def save_academic_experiment_outputs(
    out_dir: Path,
    topk: Dict,
    sparse: Dict,
    ablation: Dict,
    matcher: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(topk.get("rows", [])).to_csv(out_dir / "table3_topk_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sparse.get("rows", [])).to_csv(out_dir / "table4_sparse_activation_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ablation.get("rows", [])).to_csv(out_dir / "table5_ablation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matcher.get("rows", [])).to_csv(out_dir / "table7_matcher_quality.csv", index=False, encoding="utf-8-sig")
    print(f"  saved academic comparison tables in {out_dir}")


def save_time_budget_optimizer_outputs(out_dir: Path, time_budget_exp: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = time_budget_exp.get("method_rows", []) if time_budget_exp else []
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "方法编号": row.get("method_id", ""),
                "方法": row.get("method", ""),
                "优化器": row.get("optimizer", ""),
                "时间预算倍率": row.get("time_budget_ratio", np.nan),
                "优化时间预算(s)": row.get("time_budget_s", np.nan),
                "实际优化时间(s)": row.get("actual_optimization_time_s", np.nan),
                "评估次数": row.get("optimizer_evaluations", np.nan),
                "最佳utility": row.get("utility", np.nan),
                "交付率": row.get("delivery_ratio", np.nan),
                "丢包率": row.get("packet_loss_rate", np.nan),
                "候选通信转发节点数": row.get("comm_active_count", np.nan),
                "候选通信转发节点比例": row.get("candidate_comm_relay_ratio", np.nan),
                "实际参与转发唯一卫星数": row.get("actual_forward_unique_satellite_count", np.nan),
                "实际参与转发节点比例": row.get("actual_forward_participation_ratio", np.nan),
                "平均交付中继跳数": row.get("avg_delivered_relay_hops", np.nan),
                "最大观测中继跳数": row.get("max_observed_relay_hops", np.nan),
                "在线调度时间(s)": row.get("online_scheduling_time_s", np.nan),
                "单次仿真评估时间(s)": row.get("simulation_runtime_s", np.nan),
                "方法总运行时间(s)": row.get("total_runtime_s", np.nan),
            }
        )
    pd.DataFrame(table_rows).to_csv(
        out_dir / "table8_time_budget_optimizer_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  saved time-budget optimizer comparison table in {out_dir}")


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
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        sns.set_theme(
            style="whitegrid",
            rc={
                "font.sans-serif": plt.rcParams["font.sans-serif"],
                "axes.unicode_minus": False,
            },
        )

    @staticmethod
    def _zh_label(label: str) -> str:
        mapping = {
            "M1 Full Graph + Greedy": "M1 全图 + 贪婪",
            "M2 Top-K Graph + Greedy": "M2 Top-K图 + 贪婪",
            "M3 Full Graph + Hungarian": "M3 全图 + 匈牙利",
            "M4 Top-K Graph + Hungarian": "M4 Top-K图 + 匈牙利",
            "M5 NSGA-III + Full Graph + Greedy": "M5 NSGA-III + 全图 + 贪婪",
            "M6 NSGA-III + Top-K Graph + Greedy": "M6 NSGA-III + Top-K图 + 贪婪",
            "M7 NSGA-III + Full Graph + Hungarian": "M7 NSGA-III + 全图 + 匈牙利",
            "M8 Proposed: NSGA-III + Top-K Graph + Hungarian": "M8 本文方法：NSGA-III + Top-K图 + 匈牙利",
            "Random Sparse": "随机稀疏激活",
            "Coverage Heuristic": "覆盖启发式",
            "Budget-constrained NSGA-III": "预算约束NSGA-III",
            "Full Activation Reference": "全激活参考",
            "Proposed": "本文方法",
            "No Future GS": "去除未来地面站可见性",
            "No Energy": "去除能量评分",
            "No Task Priority": "去除任务优先级",
            "No Buffer Pressure": "去除缓存压力",
            "Distance Only": "仅距离评分",
            "No Cross-layer Score": "无跨层评分",
            "Full Activation": "全激活",
            "No Energy Score": "无能量评分",
        }
        mapping.update(
            {
                "M9 TBLS + Full Graph + Greedy": "M9 TBLS + 全图 + 贪婪",
                "M10 TBLS + Top-K Graph + Greedy": "M10 TBLS + Top-K图 + 贪婪",
                "M11 TBLS + Full Graph + Hungarian": "M11 TBLS + 全图 + 匈牙利",
                "M12 TBLS + Top-K Graph + Hungarian": "M12 TBLS + Top-K图 + 匈牙利",
                "TBLS": "TBLS时间预算局部搜索",
                "Random Search": "随机搜索",
                "Coverage Heuristic": "覆盖启发式",
                "NSGA-III": "NSGA-III多目标优化",
            }
        )
        return mapping.get(str(label), str(label))

    def _save(self, name: str) -> None:
        path = self.out_dir / name
        try:
            self.plt.tight_layout()
        except Exception:
            pass
        self.plt.savefig(path, dpi=240, bbox_inches="tight")
        self.plt.close()
        print(f"  saved {path}")

    def _empty_figure(self, name: str, title: str, message: str) -> None:
        fig, ax = self.plt.subplots(figsize=(8.0, 4.5))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        self._save(name)

    def _plot_scale_metric(
        self,
        scale: Dict,
        metric_key: str,
        ylabel: str,
        title: str,
        filename: str,
        log_y: bool = False,
        note: Optional[str] = None,
    ) -> None:
        nodes = np.asarray(scale.get("nodes", []), dtype=float)
        methods = scale.get("methods", []) if scale else []
        if len(nodes) == 0 or not methods:
            self._empty_figure(filename, title, "没有规模实验数据。")
            return
        fig, ax = self.plt.subplots(figsize=(11.0, 6.2))
        markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
        for i, method in enumerate(methods):
            vals = np.asarray(method.get(metric_key, []), dtype=float)
            if len(vals) != len(nodes):
                continue
            linestyle = "-" if method.get("candidate_graph") == "Top-K Graph" else (0, (4, 2))
            plot_vals = np.maximum(vals, 1e-9) if log_y else vals
            ax.plot(
                nodes,
                plot_vals,
                marker=markers[i % len(markers)],
                linestyle=linestyle,
                lw=1.8,
                ms=6.4,
                color=method.get("color", "#777777"),
                label=self._zh_label(method.get("label", f"M{i + 1}")),
                alpha=0.95,
            )
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("卫星数量", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, which="major", ls="--", alpha=0.24)
        ax.grid(True, which="minor", ls=":", alpha=0.12)
        ax.legend(ncol=2, fontsize=8.4, framealpha=0.96, facecolor="white", edgecolor="#BDBDBD")
        if note:
            ax.text(
                0.02,
                0.03,
                note,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.8,
                color="#4D4D4D",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BDBDBD", alpha=0.92),
            )
        self._save(filename)

    def fig1_scale_online_time(self, scale: Dict) -> None:
        self._plot_scale_metric(
            scale,
            "online_scheduling_time_s",
            "在线调度时间(s)",
            "图1  卫星数量与在线调度时间",
            "Fig1_Scale_Online_Scheduling_Time.png",
            log_y=True,
        )

    def fig2_scale_total_runtime(self, scale: Dict) -> None:
        self._plot_scale_metric(
            scale,
            "total_runtime_s",
            "总运行时间(s)",
            "图2  卫星数量与总运行时间",
            "Fig2_Scale_Total_Runtime.png",
            log_y=True,
        )

    def fig3_scale_activation_ratio(self, scale: Dict) -> None:
        nodes = np.asarray(scale.get("nodes", []), dtype=float)
        methods = scale.get("methods", []) if scale else []
        if len(nodes) == 0 or not methods:
            self._empty_figure("Fig3_Scale_Activation_Ratio.png", "图3  卫星数量与候选/实际转发参与比例", "没有规模实验数据。")
            return
        fig, axes = self.plt.subplots(2, 1, figsize=(11.0, 7.4), sharex=True)
        markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
        for i, method in enumerate(methods):
            candidate = np.asarray(method.get("candidate_comm_relay_ratio", method.get("activation_ratio", [])), dtype=float)
            actual = np.asarray(method.get("actual_forward_participation_ratio", []), dtype=float)
            if len(candidate) != len(nodes) or len(actual) != len(nodes):
                continue
            linestyle = "-" if method.get("candidate_graph") == "Top-K Graph" else (0, (4, 2))
            label = self._zh_label(method.get("label", f"M{i + 1}"))
            color = method.get("color", "#777777")
            axes[0].plot(nodes, candidate, marker=markers[i % len(markers)], linestyle=linestyle, lw=1.8, ms=6.0, color=color, label=label)
            axes[1].plot(nodes, actual, marker=markers[i % len(markers)], linestyle=linestyle, lw=1.8, ms=6.0, color=color, label=label)
        axes[0].set_ylabel("候选通信转发节点比例", fontsize=11)
        axes[1].set_ylabel("实际参与转发节点比例", fontsize=11)
        axes[1].set_xlabel("卫星数量", fontsize=11)
        axes[0].set_title("图3  卫星数量与候选/实际转发参与比例", fontsize=13, fontweight="bold")
        for ax in axes:
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, ls="--", alpha=0.24)
        axes[0].legend(ncol=2, fontsize=8.3, framealpha=0.96, facecolor="white", edgecolor="#BDBDBD")
        axes[1].text(
            0.02,
            0.05,
            "实际参与比例只统计发生过ISL发送或接收的唯一卫星，不包含仅被候选激活但未转发数据的节点。",
            transform=axes[1].transAxes,
            ha="left",
            va="bottom",
            fontsize=8.6,
            color="#4D4D4D",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BDBDBD", alpha=0.92),
        )
        self._save("Fig3_Scale_Activation_Ratio.png")

    def fig4_scale_delivery_ratio(self, scale: Dict) -> None:
        self._plot_scale_metric(
            scale,
            "delivery_ratio",
            "交付率",
            "图4  Top-K稀疏候选图的交付性能权衡",
            "Fig4_Scale_Delivery_Ratio.png",
            note="交付率用于量化Top-K稀疏候选图在在线计算效率与传输性能之间的权衡。",
        )

    def fig5_nsga_pareto_front(self, X: np.ndarray, F: np.ndarray, dl: DataLoader) -> None:
        if X.size == 0 or F.size == 0:
            self._empty_figure("Fig5_NSGA_Pareto_Front.png", "图5  NSGA-III Pareto前沿", "没有Pareto数据。")
            return
        sense = np.sum(X[:, : dl.N] > 0.5, axis=1).astype(float)
        comm = np.sum(X[:, dl.N:] > 0.5, axis=1).astype(float)
        util = np.asarray(-F[:, 0], dtype=float)
        fig, ax = self.plt.subplots(figsize=(8.4, 6.2))
        sc = ax.scatter(sense, comm, c=util, cmap="viridis", s=58, alpha=0.82, edgecolors="white", linewidths=0.45)
        best_i = select_best_objective_index(F)
        ax.scatter(
            sense[best_i],
            comm[best_i],
            marker="*",
            s=280,
            c="#FFD02A",
            edgecolors="black",
            linewidths=0.9,
            label="选定折中解",
            zorder=8,
        )
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("任务效用", fontsize=10)
        ax.set_xlabel("感知角色激活数", fontsize=11)
        ax.set_ylabel("通信角色激活数", fontsize=11)
        ax.set_title("图5  NSGA-III Pareto前沿", fontsize=13, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.24)
        ax.legend(loc="best", fontsize=9.0, framealpha=0.96, facecolor="white", edgecolor="#BDBDBD")
        self._save("Fig5_NSGA_Pareto_Front.png")

    def fig6_nsga_convergence(self, history: List[Dict]) -> None:
        if not history:
            self._empty_figure("Fig6_NSGA_Convergence.png", "图6  NSGA-III收敛过程", "没有收敛历史。")
            return
        gens = np.asarray([h.get("gen", i + 1) for i, h in enumerate(history)], dtype=float)
        best_util = np.asarray([h.get("best_util", 0.0) for h in history], dtype=float)
        active = np.asarray([h.get("total_active", 0.0) for h in history], dtype=float)
        fig, ax1 = self.plt.subplots(figsize=(9.2, 5.8))
        line1, = ax1.plot(gens, best_util, color="#1F78B4", marker="o", lw=2.0, ms=5.5, label="最优效用")
        ax1.set_xlabel("迭代代数", fontsize=11)
        ax1.set_ylabel("最优效用", fontsize=11, color="#1F78B4")
        ax1.tick_params(axis="y", labelcolor="#1F78B4")
        ax1.grid(True, ls="--", alpha=0.24)
        ax2 = ax1.twinx()
        line2, = ax2.plot(gens, active, color="#D95F02", marker="s", lw=1.9, ms=5.2, label="活跃卫星数")
        ax2.set_ylabel("活跃卫星数", fontsize=11, color="#D95F02")
        ax2.tick_params(axis="y", labelcolor="#D95F02")
        lines = [line1, line2]
        ax1.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=9.2, framealpha=0.96)
        fig.suptitle("图6  NSGA-III收敛过程", fontsize=13, fontweight="bold")
        self._save("Fig6_NSGA_Convergence.png")

    def fig7_activation_utility_3d(self, X: np.ndarray, F: np.ndarray, dl: DataLoader, best_gene: np.ndarray) -> None:
        _ = best_gene
        if X.size == 0 or F.size == 0:
            self._empty_figure("Fig7_Activation_Utility_3D.png", "图7  激活-效用三维分布", "没有激活数据。")
            return
        sense = np.sum(X[:, : dl.N] > 0.5, axis=1).astype(float)
        comm = np.sum(X[:, dl.N:] > 0.5, axis=1).astype(float)
        util = np.asarray(-F[:, 0], dtype=float)
        fig = self.plt.figure(figsize=(9.2, 7.2))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(sense, comm, util, c=util, cmap="viridis", s=42, alpha=0.82, edgecolors="white", linewidths=0.35)
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
            label="选定折中解",
        )
        ax.set_xlabel("感知角色激活数", labelpad=10)
        ax.set_ylabel("通信角色激活数", labelpad=10)
        ax.set_zlabel("任务效用", labelpad=8)
        ax.set_title("图7  激活-效用三维视图", fontsize=13, fontweight="bold", pad=16)
        ax.view_init(elev=25, azim=-52)
        fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.08, label="任务效用")
        ax.legend(loc="upper left", fontsize=9.0)
        self._save("Fig7_Activation_Utility_3D.png")

    def fig8_topk_efficiency_performance(self, topk: Dict) -> None:
        rows = topk.get("rows", []) if topk else []
        if not rows:
            self._empty_figure("Fig8_TopK_Efficiency_Performance.png", "图8  Top-K效率-性能权衡", "没有Top-K数据。")
            return
        df = pd.DataFrame(rows)
        labels = ["全图" if str(x) == "Full Graph" else str(x) for x in df["k"].tolist()]
        x = np.arange(len(labels))
        fig, ax1 = self.plt.subplots(figsize=(10.2, 5.8))
        ax2 = ax1.twinx()
        y_util = df.get("normalized_utility_vs_full", df["utility"]).to_numpy(dtype=float)
        line1, = ax1.plot(x, y_util, color="#1F78B4", marker="o", lw=2.2, label="相对全图效用")
        line3, = ax1.plot(x, df["delivery_ratio"], color="#2CA25F", marker="^", lw=1.8, ls="--", label="交付率")
        line2, = ax2.plot(x, df["online_scheduling_time_s"], color="#D95F02", marker="s", lw=2.0, label="在线调度时间")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_xlabel("候选图K值")
        ax1.set_ylabel("归一化效用 / 交付率", color="#1F78B4")
        ax2.set_ylabel("在线调度时间(s)", color="#D95F02")
        ax2.set_yscale("log")
        ax1.tick_params(axis="y", labelcolor="#1F78B4")
        ax2.tick_params(axis="y", labelcolor="#D95F02")
        ax1.set_ylim(0.0, min(1.08, max(1.02, float(np.max(y_util)) * 1.06)))
        for xi, speedup in zip(x, df.get("speedup_vs_full", np.ones(len(df)))):
            ax2.annotate(f"{speedup:.1f}倍", (xi, df["online_scheduling_time_s"].iloc[int(xi)]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.0, color="#A34A00")
        ax1.grid(True, ls="--", alpha=0.25)
        ax1.set_title("图8  压力场景下Top-K效率-性能权衡", fontsize=13, fontweight="bold")
        ax1.legend([line1, line3, line2], [line1.get_label(), line3.get_label(), line2.get_label()], loc="best", fontsize=9.0)
        self._save("Fig8_TopK_Efficiency_Performance.png")

    def fig9_sparse_activation_utility_curve(self, sparse: Dict) -> None:
        rows = sparse.get("rows", []) if sparse else []
        if not rows:
            self._empty_figure("Fig9_Sparse_Activation_Utility_Curve.png", "图9  稀疏激活率与任务效用", "没有稀疏激活数据。")
            return
        df = pd.DataFrame(rows)
        methods = list(df["method"].drop_duplicates())
        colors = {
            "Random Sparse": "#7A7A7A",
            "Coverage Heuristic": "#1B9E77",
            "Budget-constrained NSGA-III": "#1F78B4",
            "Full Activation Reference": "#D95F02",
        }
        fig, axes = self.plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
        for method in methods:
            sub = df[df["method"] == method].sort_values("activation_ratio")
            color = colors.get(method, None)
            zh_method = self._zh_label(method)
            if method == "Full Activation Reference":
                axes[0].axhline(float(sub["utility"].iloc[0]), color=color, lw=2.0, ls="--", label=zh_method)
                axes[1].axhline(float(sub["delivery_ratio"].iloc[0]), color=color, lw=2.0, ls="--", label=zh_method)
                continue
            axes[0].plot(sub["activation_ratio"], sub["utility"], marker="o", lw=2.0, color=color, label=zh_method)
            axes[1].plot(sub["activation_ratio"], sub["delivery_ratio"], marker="s", lw=1.9, color=color, label=zh_method)
            if method == "Random Sparse" and {"utility_q25", "utility_q75"}.issubset(set(sub.columns)):
                axes[0].fill_between(sub["activation_ratio"], sub["utility_q25"], sub["utility_q75"], color=color, alpha=0.16, linewidth=0)
                axes[1].fill_between(sub["activation_ratio"], sub["delivery_ratio_q25"], sub["delivery_ratio_q75"], color=color, alpha=0.16, linewidth=0)
        axes[0].set_ylabel("任务效用")
        axes[1].set_ylabel("交付率")
        axes[1].set_xlabel("通信角色激活比例")
        axes[0].set_title("图9  压力场景下稀疏激活率与任务效用", fontsize=13, fontweight="bold")
        for ax in axes:
            ax.grid(True, ls="--", alpha=0.25)
            ax.legend(loc="best", fontsize=8.8, framealpha=0.96)
        self._save("Fig9_Sparse_Activation_Utility_Curve.png")

    def fig10_cross_layer_ablation(self, ablation: Dict) -> None:
        rows = ablation.get("rows", []) if ablation else []
        if not rows:
            self._empty_figure("Fig10_Cross_Layer_Ablation.png", "图10  跨层评分消融实验", "没有消融实验数据。")
            return
        df = pd.DataFrame(rows)
        metrics = [
            ("utility", "任务效用(x10^3)", 1.0 / 1000.0, lambda v: f"{v / 1000.0:.1f}k"),
            ("delivery_ratio", "交付率", 1.0, lambda v: f"{v:.2f}"),
            ("packet_loss_rate", "丢包率", 1.0, lambda v: f"{v:.2f}"),
            ("online_scheduling_time_s", "在线调度时间(s)", 1.0, lambda v: f"{v:.3f}"),
            ("unit_delivered_energy", "单位交付能耗", 1.0, lambda v: f"{v:.1f}"),
            ("low_power_protection_count", "低电量保护次数", 1.0, lambda v: f"{int(round(v))}"),
        ]
        fig, axes = self.plt.subplots(2, 3, figsize=(14.0, 8.2))
        axes = axes.ravel()
        x = np.arange(len(df))
        labels = [self._zh_label(label) for label in df["method"].tolist()]
        colors = ["#D95F02" if str(label) == "本文方法" else "#5DA5DA" for label in labels]
        for ax, (key, title, scale, fmt) in zip(axes, metrics):
            raw_vals = pd.to_numeric(df[key], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            vals = raw_vals * float(scale)
            bars = ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.6)
            ax.axhline(0.0, color="#555555", lw=0.8, ls="--")
            ax.set_title(title, fontsize=11.0, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8.2)
            ax.grid(True, axis="y", ls="--", alpha=0.22)
            finite_vals = vals[np.isfinite(vals)]
            if len(finite_vals) == 0:
                finite_vals = np.array([0.0])
            y_min = min(0.0, float(np.min(finite_vals)))
            y_max = max(0.0, float(np.max(finite_vals)))
            span = max(y_max - y_min, 1.0)
            ax.set_ylim(y_min - 0.06 * span if y_min < 0.0 else 0.0, y_max + 0.18 * span)
            for bar, raw, plotted in zip(bars, raw_vals, vals):
                if plotted >= 0.0:
                    y = plotted + 0.025 * span
                    va = "bottom"
                else:
                    y = plotted - 0.025 * span
                    va = "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y,
                    fmt(float(raw)),
                    ha="center",
                    va=va,
                    fontsize=7.2,
                    rotation=90,
                    color="#222222",
                )
        fig.suptitle(
            "图10  压力场景下跨层评分消融实验\n"
            "柱状图表示各指标绝对值；零高度柱表示有效的零值，并非数据缺失。",
            fontsize=13.0,
            fontweight="bold",
        )
        self._save("Fig10_Cross_Layer_Ablation.png")

    def fig11_greedy_hungarian_quality_gap(self, matcher: Dict) -> None:
        rows = matcher.get("rows", []) if matcher else []
        if not rows:
            self._empty_figure("Fig11_Greedy_Hungarian_Quality_Gap.png", "图11  贪婪与匈牙利在线调度质量差距", "没有匹配器对比数据。")
            return
        df = pd.DataFrame(rows)
        metrics = [
            ("utility", "任务效用"),
            ("delivery_ratio", "交付率"),
            ("online_scheduling_time_s", "在线调度时间(s，对数)"),
        ]
        fig, axes = self.plt.subplots(2, 2, figsize=(14.0, 8.0))
        axes_flat = axes.ravel()
        for ax, (key, ylabel) in zip(axes_flat[:3], metrics):
            for label, sub in df.groupby("label"):
                sub = sub.sort_values("satellite_count")
                ax.plot(sub["satellite_count"], sub[key], marker="o", lw=1.8, label=self._zh_label(label))
            ax.set_xlabel("卫星数量")
            ax.set_ylabel(ylabel)
            if key == "online_scheduling_time_s":
                ax.set_yscale("log")
            ax.grid(True, ls="--", alpha=0.24)
        gain_ax = axes_flat[3]
        pivot = df.pivot_table(index="satellite_count", columns="method_id", values="utility", aggfunc="median").sort_index()
        if {"M1", "M3"}.issubset(set(pivot.columns)):
            gain_ax.plot(pivot.index, pivot["M3"] - pivot["M1"], marker="o", lw=1.9, label="全图：匈牙利 - 贪婪")
        if {"M2", "M4"}.issubset(set(pivot.columns)):
            gain_ax.plot(pivot.index, pivot["M4"] - pivot["M2"], marker="s", lw=1.9, label="Top-K图：匈牙利 - 贪婪")
        gain_ax.axhline(0.0, color="#555555", lw=0.9, ls="--")
        gain_ax.set_xlabel("卫星数量")
        gain_ax.set_ylabel("效用增益")
        gain_ax.grid(True, ls="--", alpha=0.24)
        gain_ax.legend(fontsize=8.0, framealpha=0.95)
        axes_flat[0].legend(fontsize=7.8, framealpha=0.95)
        fig.suptitle("图11  贪婪与匈牙利在线调度质量差距", fontsize=13.5, fontweight="bold")
        self._save("Fig11_Greedy_Hungarian_Quality_Gap.png")

    def fig12_energy_safety_timeseries(self, energy: Dict) -> None:
        if not energy:
            self._empty_figure("Fig12_Energy_Safety_Timeseries.png", "图12  能量安全与队列积压时序", "没有回放数据。")
            return
        series = energy.get("series") if isinstance(energy, dict) else None
        if not series and "time_s" in energy:
            series = [{"label": "Proposed", "history": energy, "summary": energy.get("summary", {})}]
        if not series:
            self._empty_figure("Fig12_Energy_Safety_Timeseries.png", "图12  能量安全与队列积压时序", "没有回放数据。")
            return
        fig, axes = self.plt.subplots(3, 1, figsize=(11.0, 8.4), sharex=True)
        for item in series:
            hist = item.get("history", {})
            label = self._zh_label(str(item.get("label", "case")))
            if "time_s" not in hist:
                continue
            time_h = np.asarray(hist["time_s"], dtype=float) / 3600.0
            active = max(float(item.get("summary", {}).get("active_satellite_count", 1.0)), 1.0)
            axes[0].plot(time_h, hist["energy_mean"], lw=2.0, label=f"{label} 平均能量")
            axes[1].plot(time_h, np.asarray(hist["melt_count"], dtype=float) / active, lw=1.8, label=f"{label} 熔断比例")
            axes[1].plot(time_h, np.asarray(hist["guard_count"], dtype=float) / active, lw=1.3, ls="--", label=f"{label} 保护比例")
            axes[2].plot(time_h, hist["queue_total"], lw=2.0, label=label)
        if "e_melt" in energy:
            axes[0].axhline(float(energy["e_melt"]), color="#C51B29", lw=1.0, ls="--", label="熔断阈值 E_MELT")
        if "e_warn" in energy:
            axes[0].axhline(float(energy["e_warn"]), color="#D95F02", lw=1.0, ls=":", label="预警阈值 E_WARN")
        axes[0].set_ylabel("能量")
        axes[1].set_ylabel("保护比例")
        axes[2].set_ylabel("队列积压(MB)")
        axes[2].set_xlabel("时间(h)")
        for ax in axes:
            ax.grid(True, ls="--", alpha=0.24)
            ax.legend(loc="best", fontsize=8.8)
        fig.suptitle("图12  压力场景下能量安全与队列积压时序", fontsize=13.5, fontweight="bold")
        self._save("Fig12_Energy_Safety_Timeseries.png")


    def fig13_time_budget_search_comparison(self, time_budget_exp: Dict) -> None:
        rows = time_budget_exp.get("curve_rows", []) if time_budget_exp else []
        title = "图13  相同时间预算下的激活优化搜索效率对比"
        if not rows:
            self._empty_figure("Fig13_Time_Budget_Search_Comparison.png", title, "没有时间预算优化对比数据。")
            return
        df = pd.DataFrame(rows)
        if df.empty:
            self._empty_figure("Fig13_Time_Budget_Search_Comparison.png", title, "没有时间预算优化对比数据。")
            return
        fig, ax = self.plt.subplots(figsize=(10.5, 5.8))
        order = ["NSGA-III", "TBLS", "Random Search", "Coverage Heuristic"]
        colors = {
            "NSGA-III": "#1F78B4",
            "TBLS": "#D95F02",
            "Random Search": "#777777",
            "Coverage Heuristic": "#1B9E77",
        }
        markers = {
            "NSGA-III": "o",
            "TBLS": "s",
            "Random Search": "^",
            "Coverage Heuristic": "D",
        }
        for optimizer in order:
            sub = df[df["optimizer"] == optimizer].sort_values("time_budget_ratio")
            if sub.empty:
                continue
            ax.plot(
                sub["time_budget_ratio"],
                sub["best_utility"],
                marker=markers.get(optimizer, "o"),
                lw=2.2,
                ms=6.0,
                color=colors.get(optimizer),
                label=self._zh_label(optimizer),
            )
        ax.axvline(1.0, color="#C51B29", ls="--", lw=1.2)
        finite_util = pd.to_numeric(df["best_utility"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        ymax = float(finite_util.max()) if len(finite_util) else 1.0
        ymin = float(finite_util.min()) if len(finite_util) else 0.0
        ax.text(1.03, ymax - 0.04 * max(ymax - ymin, 1.0), "1.0x 同等时间预算", color="#C51B29", fontsize=9.2)
        ax.set_xscale("log", base=2)
        ticks = [0.25, 0.5, 1.0, 2.0, 4.0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"])
        ax.set_xlabel("优化时间 / NSGA-III优化时间")
        ax.set_ylabel("最佳任务效用")
        ax.set_title(title, fontsize=13.5, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.25)
        ax.legend(loc="best", fontsize=9.0, framealpha=0.95)
        self._save("Fig13_Time_Budget_Search_Comparison.png")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于虚拟观测节点与通信转发节点的联合感知/通信跨层拓扑优化方法"
    )
    parser.add_argument("--csv", dest="csv_path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--profile", choices=["full", "fast", "smoke"], default="full")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ga-pop", type=int, default=None)
    parser.add_argument("--ga-gen", type=int, default=None)
    parser.add_argument("--sat-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--arrival", choices=["fixed", "poisson"], default=None)
    parser.add_argument("--task-packet-mb", type=float, default=None)
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
        cfg.RNG_SEED = int(args.seed)
    if args.ga_pop is not None:
        cfg.GA_POP = int(args.ga_pop)
    if args.ga_gen is not None:
        cfg.GA_GEN = int(args.ga_gen)
    if args.sat_limit is not None:
        cfg.SAT_LIMIT = int(args.sat_limit)
    if args.max_steps is not None:
        cfg.MAX_STEPS = int(args.max_steps)
    if args.arrival is not None:
        cfg.TASK_ARRIVAL_MODE = str(args.arrival)
    if args.task_packet_mb is not None:
        cfg.TASK_PACKET_MB = float(args.task_packet_mb)
    if args.no_plots:
        cfg.PLOT = False
    if args.no_cache:
        cfg.USE_ISL_CACHE = False

    np.random.seed(int(cfg.RNG_SEED))
    out_dir = Path(cfg.OUTPUT_DIR)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n基于虚拟观测节点与通信转发节点的联合感知/通信跨层拓扑优化方法")
    print(f"profile={args.profile}, output={out_dir}")
    print(f"ground station: {GS_NAME} lat={GS_LAT:.2f}, lon={GS_LON:.2f}")
    task_desc = ", ".join(f"{name}={weight:.1f}" for name, weight in zip(TASK_REGION_NAMES, TASK_REGION_WEIGHTS))
    print(f"task region weights: {task_desc}")
    print(
        "task arrival model: "
        f"{cfg.TASK_ARRIVAL_MODE}, packet={cfg.TASK_PACKET_MB:.1f} MB, "
        f"expected_rate={cfg.TASK_GEN_RATE:.1f} MB/s"
    )

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
    print(f"delivery ratio       : {summary['delivery_ratio'] * 100:.2f}% (delivered/TASK_DATA_MB)")
    print(f"packet loss rate     : {summary['packet_loss_rate'] * 100:.2f}% (dropped/generated)")
    print(f"sensing role active  : {counts[0]} / {dl.N}")
    print(f"comm role active     : {counts[1]} / {dl.N}")
    print(f"candidate relay ratio: {summary.get('candidate_comm_relay_ratio', 0.0) * 100:.2f}%")
    print(
        "actual forward nodes : "
        f"{summary.get('actual_forward_unique_satellite_count', 0)} / {dl.N} "
        f"({summary.get('actual_forward_participation_ratio', 0.0) * 100:.2f}%)"
    )
    print(
        "actual ISL per step  : "
        f"send avg/max={summary.get('avg_actual_isl_send_count', 0.0):.2f}/{summary.get('max_actual_isl_send_count', 0)}, "
        f"recv avg/max={summary.get('avg_actual_isl_receive_count', 0.0):.2f}/{summary.get('max_actual_isl_receive_count', 0)}"
    )
    print(
        "downlink per step    : "
        f"avg/max={summary.get('avg_actual_downlink_count', 0.0):.2f}/{summary.get('max_actual_downlink_count', 0)}, "
        f"unique={summary.get('actual_downlink_unique_satellite_count', 0)}"
    )
    print(
        "relay hop / TTL drop : "
        f"avg_hop={summary.get('avg_delivered_relay_hops', 0.0):.2f}, "
        f"max_hop={summary.get('max_observed_relay_hops', 0)}, "
        f"ttl_drop={summary.get('dropped_ttl_mb', 0.0):.1f} MB"
    )
    print(f"online scheduling    : {summary['online_scheduling_time_s']:.3f}s")
    print(f"simulation runtime   : {summary['simulation_runtime_s']:.3f}s")
    print(f"NSGA-III runtime     : {summary['runtime_nsga_s']:.3f}s")
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
        stress_cfg = make_stress_config(cfg)
        print("=" * 72)
        print(
            "S6 Stress scenario for academic comparison figures: "
            f"TASK_DATA_MB={stress_cfg.TASK_DATA_MB:.1f}, "
            f"TX_ISL={stress_cfg.TX_RATE_ISL:.1f}, TX_SGL={stress_cfg.TX_RATE_SGL:.1f}, "
            f"GS_ANTENNAS={stress_cfg.GS_ANTENNAS}, Q_MAX={stress_cfg.Q_MAX_COMM:.1f}"
        )
        print("=" * 72)
        stress_problem = ConstellationProblem(stress_cfg, dl)
        stress_result = NSGA3Solver(stress_cfg).optimize(stress_problem.evaluate, dl.N)
        stress_best_i = select_best_objective_index(stress_result["F"], utility_tol=max(float(stress_cfg.GA_CONV_EPS), 1e-6))
        stress_best_gene = stress_result["X"][stress_best_i].astype(bool)
        topk_exp = run_topk_sensitivity_experiment(stress_cfg, dl, stress_best_gene)
        sparse_exp = run_sparse_activation_curve_experiment(stress_cfg, dl, stress_best_gene)
        ablation_exp = run_ablation_experiment(stress_cfg, dl, stress_best_gene)
        matcher_exp = run_matcher_quality_experiment(scale)
        energy_exp = run_energy_safety_timeseries_experiment(stress_cfg, dl, stress_best_gene)
        save_academic_experiment_outputs(out_dir, topk_exp, sparse_exp, ablation_exp, matcher_exp)
        vis.fig8_topk_efficiency_performance(topk_exp)
        vis.fig9_sparse_activation_utility_curve(sparse_exp)
        vis.fig10_cross_layer_ablation(ablation_exp)
        vis.fig11_greedy_hungarian_quality_gap(matcher_exp)
        vis.fig12_energy_safety_timeseries(energy_exp)
        time_budget_exp = run_time_budget_optimizer_experiment(
            cfg,
            dl,
            result,
            best_gene,
            nsga_runtime_s=float(result.get("runtime_s", 0.0)),
        )
        save_time_budget_optimizer_outputs(out_dir, time_budget_exp)
        vis.fig13_time_budget_search_comparison(time_budget_exp)

    print("=" * 72)
    print("Done")
    print("=" * 72)


if __name__ == "__main__":
    main()

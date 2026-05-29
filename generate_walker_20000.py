import csv
import math
import numpy as np

# =========================
# 1. Walker 星座参数
# =========================
T = 20000    # 卫星总数
P = 100            # 轨道面数
S = T // P         # 每个轨道面卫星数
F = 20             # 相位因子

h_km = 550         # 轨道高度 km
Re_km = 6378.137   # 地球半径 km
mu = 398600.4418   # 地球引力常数 km^3/s^2

inclination_deg = 86
inclination = math.radians(inclination_deg)

a = Re_km + h_km   # 圆轨道半长轴
n = math.sqrt(mu / a**3)   # 平均角速度 rad/s
v_mag = math.sqrt(mu / a)  # 圆轨道速度 km/s

# =========================
# 2. 仿真时间设置
# =========================
duration_s = 3 * 3600     # simulation duration: 3 hours
step_s = 150               # 每 300 秒采样一次，数据量适中

times = np.arange(0, duration_s + step_s, step_s)

# =========================
# 3. 坐标旋转函数
# =========================
def rotation_matrix(raan, inc):
    cosO = math.cos(raan)
    sinO = math.sin(raan)
    cosi = math.cos(inc)
    sini = math.sin(inc)

    return np.array([
        [cosO, -sinO * cosi,  sinO * sini],
        [sinO,  cosO * cosi, -cosO * sini],
        [0,     sini,         cosi]
    ])

# =========================
# 4. 生成 satellite_state_all.csv
# =========================
output_file = "satellite_state_all.csv"

with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow([
        "sat_id",
        "time_s",
        "x_km",
        "y_km",
        "z_km",
        "vx_km_s",
        "vy_km_s",
        "vz_km_s"
    ])

    for p in range(P):
        raan = 2 * math.pi * p / P
        R = rotation_matrix(raan, inclination)

        for s in range(S):
            sat_index = p * S + s + 1
            sat_id = f"Sat_{sat_index:05d}"

            # Walker Delta 相位设置
            M0 = 2 * math.pi * s / S + 2 * math.pi * F * p / T

            for t in times:
                M = M0 + n * t

                # 圆轨道下，轨道平面坐标
                r_orb = np.array([
                    a * math.cos(M),
                    a * math.sin(M),
                    0
                ])

                v_orb = np.array([
                    -v_mag * math.sin(M),
                    v_mag * math.cos(M),
                    0
                ])

                # 转换到惯性坐标系
                r_eci = R @ r_orb
                v_eci = R @ v_orb

                writer.writerow([
                    sat_id,
                    float(t),
                    r_eci[0],
                    r_eci[1],
                    r_eci[2],
                    v_eci[0],
                    v_eci[1],
                    v_eci[2]
                ])

print(f"生成完成：{output_file}")
print(f"卫星数量：{T}")
print(f"轨道面数量：{P}")
print(f"每面卫星数：{S}")
print(f"仿真步长：{step_s} s")
print(f"时间点数量：{len(times)}")
print(f"总数据行数约：{T * len(times)}")

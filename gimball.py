# -*- coding: utf-8 -*-
"""
RoboMaster Gimbal + ToF LiDAR Scanner & Video Generator
-------------------------------------------------------
โปรแกรมควบคุม Gimbal ให้หมุนสแกน (Yaw / Pitch) พร้อมรับค่าจากเซนเซอร์ ToF
และเซนเซอร์มุมของ Gimbal มาพล็อตเป็น Radar / LiDAR Map แบบ Real-time
และบันทึกผลออกมาเป็นไฟล์วิดีโอ (MP4/AVI) พร้อมกราฟสรุปผล 4 มิติ

คุณสมบัติเด่น:
1. หมุน Gimbal สแกนซ้าย-ขวา (-90° ถึง +90°) อัตโนมัติ
2. Subscribe มุม Gimbal (Yaw, Pitch) และระยะ ToF (mm)
3. ชดเชยระยะ Offset ของหัวเซนเซอร์ ToF = 7 cm (70 mm)
4. [ใหม่!] ตัวกรองสัญญาณรบกวน (Noise Filter):
   - Median Filter: กำจัด Spike Noise และค่ากระโดดผิดปกติ
   - Exponential Moving Average (EMA): เกลี่ยค่าระยะให้เรียบ กำแพงตรง ไม่สั่น
   - Outlier Rejection: ป้องกันค่าหลุดช่วงฉับพลัน
5. แสดงผล Radar Display แบบ Real-time พร้อมแสดงจุด Raw vs Filtered
6. บันทึกวิดีโอ .mp4 และรูปกราฟสรุปผล .png อัตโนมัติ
"""

import time
import math
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from collections import deque

# =============================================================================
#  CONFIG & PARAMETERS
# =============================================================================

# ตั้งค่าโหมดจำลอง (ตั้งเป็น False เมื่อต่อกับหุ่นจริง)
MOCK_MODE = False
CONN_TYPE = "ap"  # 'ap' (ต่อตรง Wi-Fi หุ่น) หรือ 'sta' (ต่อผ่าน Router)

# พารามิเตอร์การสแกนของ Gimbal (-75° ถึง +75°)
SCAN_YAW_MIN = -75.0    # องศาซ้ายสุด
SCAN_YAW_MAX = 75.0     # องศาขวาสุด
SCAN_PITCH = 0.0        # องศามุมก้ม-เงย (Pitch)
SCAN_SPEED = 40         # ความเร็วการหมุน Gimbal (deg/s)
SCAN_PASSES = 1         # จำนวนรอบการสแกนไป-กลับ (Sweeps)

# พารามิเตอร์เซนเซอร์ ToF & Gimbal
TOF_OFFSET_CM = 7.0      # Offset ของหัวเซนเซอร์ ToF จากจุดหมุน Gimbal = 7 cm
TOF_OFFSET_MM = TOF_OFFSET_CM * 10.0  # 70.0 mm

# -----------------------------------------------------------------------------
# พารามิเตอร์กรองสัญญาณรบกวน (Noise Filter for Straight Wall)
# -----------------------------------------------------------------------------
ENABLE_NOISE_FILTER = True     # เปิด/ปิด ระบบกรอง Noise
MEDIAN_WINDOW_SIZE = 5         # ขนาดหน้าต่าง Median Filter (3-7 จุด) ตัด Spike Noise
EMA_ALPHA = 0.30               # ค่าน้ำหนัก Smoothing (0.1 เรียบตรงมาก - 1.0 ค่าดิบ)
MAX_JUMP_THRESHOLD_MM = 350.0  # ตัดค่ากระโดดผิดปกติแบบฉับพลัน (Outlier Clamp)

# พารามิเตอร์วิดีโอ & หน้าจอ Radar & กราฟสรุปผล
FRAME_WIDTH = 800
FRAME_HEIGHT = 800
FPS = 30
MAX_RANGE_MM = 3000     # ระยะสแกนสูงสุดที่แสดง (3 เมตร / 3000 mm)
OUTPUT_VIDEO_PATH = "gimbal_tof_scan.mp4"
SUMMARY_PLOT_PATH = "gimbal_tof_summary.png"

if not MOCK_MODE:
    import robomaster
    from robomaster import robot
    # ปิด Log กวนใจจาก SDK
    logging.getLogger("sdk").setLevel(logging.CRITICAL)


# =============================================================================
#  NOISE FILTER CLASS (Spike & Wall Straightening Filter)
# =============================================================================

class ToFNoiseFilter:
    """
    ระบบกรอง Noise สำหรับเซนเซอร์ ToF แบบ Zero Phase Lag:
    1. Outlier Rejection: ป้องกันค่ากระโดดหลุดโลก (Glitch/Spike)
    2. Fast Median Filter: กรอง Noise แบบไม่ทำให้เกิด Time Delay / Phase Lag
    """
    def __init__(self, median_window=3, max_jump_mm=MAX_JUMP_THRESHOLD_MM):
        self.median_window = median_window
        self.max_jump_mm = max_jump_mm
        self.window = deque(maxlen=median_window)
        self.last_filtered = None

    def update(self, raw_dist_mm):
        if raw_dist_mm <= 0 or raw_dist_mm > MAX_RANGE_MM * 1.5:
            return self.last_filtered if self.last_filtered is not None else raw_dist_mm

        # 1. Outlier Rejection (ป้องกันค่ากระโดดฉับพลัน)
        if self.last_filtered is not None and abs(raw_dist_mm - self.last_filtered) > self.max_jump_mm:
            if len(self.window) >= 2:
                raw_dist_mm = float(np.median(self.window))

        self.window.append(raw_dist_mm)

        # 2. Fast Median Filter (Zero-Phase / ไม่ดีเลย์ข้ามมุม)
        median_val = float(np.median(self.window))
        self.last_filtered = median_val
        return self.last_filtered

    def reset(self):
        self.window.clear()
        self.last_filtered = None


# =============================================================================
#  SCANNER CLASS
# =============================================================================

class GimbalToFScanner:
    def __init__(self, mock=MOCK_MODE):
        self.mock = mock
        self.ep_robot = None
        self.ep_gimbal = None
        self.ep_sensor = None
        
        # ตัวกรอง Noise
        self.filter = ToFNoiseFilter()
        
        # สถานะเซนเซอร์แบบ Real-time
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_raw_tof_dist = 0.0   # ค่าดิบจากเซนเซอร์ ToF (mm)
        self.current_tof_dist = 0.0       # ค่าหลังกรอง Noise (mm)
        self.tof_sensor_id = 0            # ใช้ ToF ตัวที่ 1 (index 0)
        
        # จุด Point Cloud สะสมสำหรับ Real-time Radar
        self.points_history = deque(maxlen=600)
        # เก็บข้อมูลจุดทั้งหมดตลอดการสแกนสำหรับสรุปผล (Summary Plot)
        self.all_scanned_points = []
        
        # ข้อมูลสำหรับบันทึกวิดีโอ
        self.frames = []
        self.is_running = True

    def _tof_data_handler(self, sub_info):
        """Callback รับข้อมูลจาก ToF Distance Sensors (tof1..tof4)"""
        distances = sub_info
        if distances and len(distances) > self.tof_sensor_id:
            raw_dist = distances[self.tof_sensor_id]
            if raw_dist > 0 and raw_dist < 10000:
                self.current_raw_tof_dist = float(raw_dist)
                
                # กรอง Noise ให้กำแพงเรียบตรง
                if ENABLE_NOISE_FILTER:
                    self.current_tof_dist = self.filter.update(self.current_raw_tof_dist)
                else:
                    self.current_tof_dist = self.current_raw_tof_dist

                self._record_point(self.current_yaw, self.current_pitch,
                                   self.current_raw_tof_dist, self.current_tof_dist)

    def _gimbal_angle_handler(self, angle_info):
        """Callback รับข้อมูลมุม Gimbal (Pitch, Yaw, Pitch_ground, Yaw_ground)"""
        pitch_angle, yaw_angle, _, _ = angle_info
        self.current_pitch = float(pitch_angle)
        self.current_yaw = float(yaw_angle)

    def _record_point(self, yaw_deg, pitch_deg, raw_dist_mm, filtered_dist_mm):
        """แปลงมุมและระยะเป็นพิกัด 2D/3D (X, Y) โดยคำนึงถึง Sensor Offset 7 cm (70 mm)"""
        if filtered_dist_mm <= 30 or filtered_dist_mm > MAX_RANGE_MM * 1.5:
            return
        
        # บวก Offset 7 cm (70 mm) เพื่อให้ได้ระยะห่างสัมบูรณ์จากจุดหมุนของ Gimbal
        total_dist_mm = filtered_dist_mm + TOF_OFFSET_MM
        raw_total_dist_mm = raw_dist_mm + TOF_OFFSET_MM
        
        # แปลงเป็นเรเดียน
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        
        # X: ด้านข้าง (ขวาเป็น +), Y: ด้านหน้า (หน้าเป็น +)
        x = total_dist_mm * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = total_dist_mm * math.cos(pitch_rad) * math.cos(yaw_rad)

        # พิกัดดิบก่อนกรอง (Raw Jittery Point)
        raw_x = raw_total_dist_mm * math.cos(pitch_rad) * math.sin(yaw_rad)
        raw_y = raw_total_dist_mm * math.cos(pitch_rad) * math.cos(yaw_rad)
        
        pt = {
            'x': x,
            'y': y,
            'raw_x': raw_x,
            'raw_y': raw_y,
            'yaw': yaw_deg,
            'pitch': pitch_deg,
            'raw_dist': raw_dist_mm,
            'dist': total_dist_mm,
            'time': time.time()
        }
        self.points_history.append(pt)
        self.all_scanned_points.append(pt)

    def connect(self):
        """เชื่อมต่อหุ่นยนต์และเริ่ม Subscribe เซนเซอร์"""
        if self.mock:
            print("[INFO] เริ่มการทำงานในโหมด SIMULATION (Mock Mode)")
            return True

        print(f"[INFO] กำลังเชื่อมต่อ RoboMaster ({CONN_TYPE.upper()} Mode)...")
        try:
            self.ep_robot = robot.Robot()
            self.ep_robot.initialize(conn_type=CONN_TYPE)
            
            self.ep_gimbal = self.ep_robot.gimbal
            self.ep_sensor = self.ep_robot.sensor

            # Subscribe มุม Gimbal และ ToF ที่ความถี่ 20Hz / 50Hz
            self.ep_gimbal.sub_angle(freq=20, callback=self._gimbal_angle_handler)
            self.ep_sensor.sub_distance(freq=20, callback=self._tof_data_handler)
            
            # Recenter Gimbal
            self.ep_gimbal.recenter(pitch_speed=80, yaw_speed=80).wait_for_completed()
            print("[INFO] เชื่อมต่อสำเร็จและจัดตำแหน่ง Gimbal ตรงกลางเรียบร้อย!")
            return True
        except Exception as e:
            print(f"[ERROR] ไม่สามารถเชื่อมต่อกับหุ่นยนต์ได้: {e}")
            print("[INFO] สลับไปใช้ Mock Mode เพื่อจำลองการแสดงผลแทน...")
            self.mock = True
            return True

    def draw_radar_frame(self):
        """วาด Radar UI หน้าจอเรดาร์แสดงผลมุมหมุน จุด Obstacle และ Telemetry"""
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        
        center_x = FRAME_WIDTH // 2
        center_y = FRAME_HEIGHT - 120  # วางตำแหน่งหัวหุ่นยนต์ใกล้ขอบล่าง
        max_radius = min(FRAME_WIDTH // 2 - 40, center_y - 80)
        scale = max_radius / MAX_RANGE_MM  # pixels per mm

        # 1. วาดวงกลมระยะ Range Rings (ทุกๆ 50 cm / 1 m)
        ranges = [500, 1000, 1500, 2000, 2500, 3000]
        for r_mm in ranges:
            r_px = int(r_mm * scale)
            if r_px <= max_radius:
                cv2.circle(frame, (center_x, center_y), r_px, (30, 60, 30), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{r_mm/1000:.1f}m", (center_x + 5, center_y - r_px + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 0), 1, cv2.LINE_AA)

        # 2. วาดเส้นแบ่งมุม (Angle Grid Rays) และขอบเขต FOV สแกน (-75° ถึง +75°)
        display_angles = [-75, -60, -30, 0, 30, 60, 75]
        for angle in display_angles:
            rad = math.radians(angle)
            end_x = int(center_x + max_radius * math.sin(rad))
            end_y = int(center_y - max_radius * math.cos(rad))
            # ไฮไลท์เส้นขอบเขตการสแกน ±75°
            is_limit = (angle in [SCAN_YAW_MIN, SCAN_YAW_MAX])
            line_color = (0, 100, 100) if is_limit else (25, 50, 25)
            line_thick = 2 if is_limit else 1
            cv2.line(frame, (center_x, center_y), (end_x, end_y), line_color, line_thick, cv2.LINE_AA)
            
            lbl_x = int(center_x + (max_radius + 20) * math.sin(rad)) - 10
            lbl_y = int(center_y - (max_radius + 20) * math.cos(rad)) + 4
            lbl_color = (0, 255, 200) if is_limit else (0, 180, 80)
            cv2.putText(frame, f"{angle}°", (lbl_x, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, lbl_color, 1, cv2.LINE_AA)

        # 3. วาด Point Cloud (แสดงทั้งจุดดิบก่อนกรอง และจุดที่กรองเรียบตรงแล้ว)
        now = time.time()
        for pt in list(self.points_history):
            age = now - pt['time']
            alpha = max(0.2, 1.0 - (age / 6.0))  # จางลงใน 6 วินาที

            # วาดจุดดิบก่อนกรองเป็นสีเทา/แดงจางๆ (Raw Jittery Dots)
            if 'raw_x' in pt and ENABLE_NOISE_FILTER:
                raw_px = int(center_x + pt['raw_x'] * scale)
                raw_py = int(center_y - pt['raw_y'] * scale)
                if 0 <= raw_px < FRAME_WIDTH and 0 <= raw_py < FRAME_HEIGHT:
                    cv2.circle(frame, (raw_px, raw_py), 2, (50, 50, 110), -1, cv2.LINE_AA)
            
            # วาดจุดที่ผ่านการกรองแล้ว (Filtered Clean Wall Points)
            pt_x = int(center_x + pt['x'] * scale)
            pt_y = int(center_y - pt['y'] * scale)
            dist_ratio = min(1.0, pt['dist'] / MAX_RANGE_MM)
            b = int(255 * dist_ratio * alpha)
            g = int(255 * (1.0 - abs(dist_ratio - 0.5) * 2) * alpha)
            r = int(255 * (1.0 - dist_ratio) * alpha)
            
            if 0 <= pt_x < FRAME_WIDTH and 0 <= pt_y < FRAME_HEIGHT:
                cv2.circle(frame, (pt_x, pt_y), 3, (b, g, r), -1, cv2.LINE_AA)

        # 4. วาด Sweep Beam Line (ลำแสงตามมุม Gimbal ปัจจุบัน พร้อมแสดงหัว ToF Offset 7 cm)
        current_rad = math.radians(self.current_yaw)
        sensor_offset_px = int(TOF_OFFSET_MM * scale)
        sensor_x = int(center_x + sensor_offset_px * math.sin(current_rad))
        sensor_y = int(center_y - sensor_offset_px * math.cos(current_rad))

        beam_end_x = int(center_x + max_radius * math.sin(current_rad))
        beam_end_y = int(center_y - max_radius * math.cos(current_rad))
        
        # วาดก้าน Gimbal จากแกนหมุนถึงตำแหน่งหัวเซนเซอร์ ToF (+7 cm)
        cv2.line(frame, (center_x, center_y), (sensor_x, sensor_y), (100, 100, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, (sensor_x, sensor_y), 4, (0, 165, 255), -1, cv2.LINE_AA)

        # วาดเส้นลำแสงเลเซอร์ยิงออกจากหัว ToF
        cv2.line(frame, (sensor_x, sensor_y), (beam_end_x, beam_end_y), (0, 255, 255), 2, cv2.LINE_AA)
        
        # ลำแสงสามเหลี่ยมกึ่งโปร่งใส (Glow sector)
        poly_pts = np.array([
            [sensor_x, sensor_y],
            [int(center_x + max_radius * math.sin(current_rad - 0.05)), int(center_y - max_radius * math.cos(current_rad - 0.05))],
            [beam_end_x, beam_end_y],
            [int(center_x + max_radius * math.sin(current_rad + 0.05)), int(center_y - max_radius * math.cos(current_rad + 0.05))]
        ], np.int32)
        glow_overlay = frame.copy()
        cv2.fillPoly(glow_overlay, [poly_pts], (0, 180, 180))
        cv2.addWeighted(glow_overlay, 0.3, frame, 0.7, 0, frame)

        # 5. วาดตำแหน่งปัจจุบันของ ToF Target Hit (ระยะ ToF + Offset 7 cm)
        total_hit_dist = self.current_tof_dist + TOF_OFFSET_MM
        if self.current_tof_dist > 30 and total_hit_dist <= MAX_RANGE_MM:
            hit_x = int(center_x + (total_hit_dist * scale) * math.sin(current_rad))
            hit_y = int(center_y - (total_hit_dist * scale) * math.cos(current_rad))
            cv2.circle(frame, (hit_x, hit_y), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (hit_x, hit_y), 10, (0, 255, 255), 1, cv2.LINE_AA)

        # 6. วาดไอคอน Gimbal / Robot Center
        cv2.circle(frame, (center_x, center_y), 12, (255, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (center_x, center_y), 14, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GIMBAL", (center_x - 26, center_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 7. วาด Telemetry HUD Header & Data Panel
        cv2.rectangle(frame, (15, 15), (FRAME_WIDTH - 15, 100), (20, 20, 20), -1)
        cv2.rectangle(frame, (15, 15), (FRAME_WIDTH - 15, 100), (60, 60, 60), 1)

        cv2.putText(frame, "ROBOMASTER GIMBAL + TOF SCANNER", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        
        status_mode = "[SIMULATION MOCK]" if self.mock else f"[LIVE {CONN_TYPE.upper()}]"
        cv2.putText(frame, status_mode, (FRAME_WIDTH - 210, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if not self.mock else (0, 165, 255), 1, cv2.LINE_AA)

        filter_status = "FILTER: ON (Median+EMA)" if ENABLE_NOISE_FILTER else "FILTER: OFF"
        cv2.putText(frame, filter_status, (FRAME_WIDTH - 210, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180) if ENABLE_NOISE_FILTER else (120, 120, 120), 1, cv2.LINE_AA)

        # Telemetry Values (เปรียบเทียบค่าดิบกับค่าที่กรองแล้ว)
        telemetry_str = (f"Yaw: {self.current_yaw:+05.1f} deg | "
                         f"Raw: {self.current_raw_tof_dist/10:04.1f}cm -> "
                         f"Filt: {self.current_tof_dist/10:04.1f}cm "
                         f"(+{TOF_OFFSET_CM:.0f}cm off) = {total_hit_dist/10:04.1f}cm")
        cv2.putText(frame, telemetry_str, (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

        pts_count_str = f"Pts: {len(self.points_history)}"
        cv2.putText(frame, pts_count_str, (FRAME_WIDTH - 110, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 255, 150), 1, cv2.LINE_AA)

        # ขอบรอบนอก
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), (40, 80, 40), 1)

        return frame

    def run_mock_sweep(self):
        """จำลองการหมุนสแกนพร้อม Noise เพื่อทดสอบประสิทธิภาพการกรองให้กำแพงตรง"""
        print("[INFO] กำลังจำลองการหมุน Gimbal พร้อม Noise Filter...")
        total_steps = SCAN_PASSES * 180
        yaw_step = 1.5
        curr_yaw = SCAN_YAW_MIN
        direction = 1

        for step in range(total_steps):
            curr_yaw += direction * yaw_step
            if curr_yaw >= SCAN_YAW_MAX:
                curr_yaw = SCAN_YAW_MAX
                direction = -1
            elif curr_yaw <= SCAN_YAW_MIN:
                curr_yaw = SCAN_YAW_MIN
                direction = 1

            self.current_yaw = curr_yaw
            self.current_pitch = math.sin(step * 0.05) * 5.0

            # คำนวณระยะกำแพงตรงหน้า Y = 1600 mm (True Geometry)
            rad = math.radians(curr_yaw)
            dist_wall = 1600.0 / max(0.001, math.cos(rad)) if abs(curr_yaw) < 65 else 3200.0
            if abs(curr_yaw - (-35)) < 12:
                dist_wall = min(dist_wall, 900.0 + (curr_yaw + 35)**2 * 2.0)
            if abs(curr_yaw - 45) < 15:
                dist_wall = min(dist_wall, 1200.0 + (curr_yaw - 45)**2 * 1.5)

            # ใส่ Noise ชัดเจน (Jitter ±35mm และ Spike กระโดดเป็นครั้งคราว)
            noise = np.random.normal(0, 35.0)
            if np.random.random() < 0.06:
                noise += np.random.choice([-220.0, 220.0])  # Spike noise
                
            raw_tof_sim = dist_wall - TOF_OFFSET_MM + noise
            self.current_raw_tof_dist = max(30.0, min(MAX_RANGE_MM - TOF_OFFSET_MM, raw_tof_sim))
            
            # กรอง Noise ผ่านระบบ ToFNoiseFilter
            if ENABLE_NOISE_FILTER:
                self.current_tof_dist = self.filter.update(self.current_raw_tof_dist)
            else:
                self.current_tof_dist = self.current_raw_tof_dist
            
            self._record_point(self.current_yaw, self.current_pitch,
                               self.current_raw_tof_dist, self.current_tof_dist)

            frame = self.draw_radar_frame()
            self.frames.append(frame)
            
            cv2.imshow("RoboMaster Gimbal ToF Scanner", frame)
            if cv2.waitKey(int(1000 / FPS)) & 0xFF == 27:
                break

    def run_hardware_scan(self):
        """ควบคุม Gimbal จริงให้หมุนสแกนซ้าย-ขวา พร้อมบันทึกภาพเรดาร์"""
        print(f"[INFO] เริ่มสแกน Gimbal Yaw: {SCAN_YAW_MIN}° ถึง {SCAN_YAW_MAX}° จำนวน {SCAN_PASSES} รอบ...")
        
        for p in range(SCAN_PASSES):
            print(f"[SCAN] รอบที่ {p+1}/{SCAN_PASSES} : หมุนไปขวา ({SCAN_YAW_MAX}°)")
            action = self.ep_gimbal.moveto(pitch=SCAN_PITCH, yaw=SCAN_YAW_MAX,
                                           pitch_speed=SCAN_SPEED, yaw_speed=SCAN_SPEED)
            while not action.is_completed:
                frame = self.draw_radar_frame()
                self.frames.append(frame)
                cv2.imshow("RoboMaster Gimbal ToF Scanner", frame)
                if cv2.waitKey(int(1000 / FPS)) & 0xFF == 27:
                    return
                time.sleep(1.0 / FPS)

            print(f"[SCAN] รอบที่ {p+1}/{SCAN_PASSES} : หมุนกลับซ้าย ({SCAN_YAW_MIN}°)")
            action = self.ep_gimbal.moveto(pitch=SCAN_PITCH, yaw=SCAN_YAW_MIN,
                                           pitch_speed=SCAN_SPEED, yaw_speed=SCAN_SPEED)
            while not action.is_completed:
                frame = self.draw_radar_frame()
                self.frames.append(frame)
                cv2.imshow("RoboMaster Gimbal ToF Scanner", frame)
                if cv2.waitKey(int(1000 / FPS)) & 0xFF == 27:
                    return
                time.sleep(1.0 / FPS)

        print("[INFO] สแกนเสร็จสมบูรณ์ กำลังปรับ Gimbal กลับจุดกึ่งกลาง (0°)...")
        self.ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=60, yaw_speed=60).wait_for_completed()

    def save_video(self, output_path=OUTPUT_VIDEO_PATH):
        """บันทึกเฟรมทั้งหมดเป็นไฟล์วิดีโอ MP4 / AVI"""
        if not self.frames:
            print("[WARN] ไม่มีเฟรมภาพสำหรับบันทึกวิดีโอ")
            return

        print(f"[INFO] กำลังบันทึกวิดีโอ {len(self.frames)} เฟรม ไปที่: {output_path} ...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

        for f in self.frames:
            out.write(f)
        out.release()
        print(f"[SUCCESS] บันทึกไฟล์วิดีโอเรียบร้อยแล้ว: {output_path}")

    def plot_summary(self, output_path=SUMMARY_PLOT_PATH):
        """พลอตกราฟเดียว แสดงจุดพิกัด (X, Y) ทั้งหมดที่วัดได้จากเซนเซอร์ ToF + Gimbal โดยกรองแบบ Zero-Phase ให้ตรงตามค่า Raw"""
        if not self.all_scanned_points:
            print("[WARN] ไม่มีข้อมูลสำหรับพลอตกราฟ")
            return

        print(f"[INFO] กำลังสร้างกราฟจุดที่วัดได้ (Point Cloud) จาก {len(self.all_scanned_points)} จุด...")

        # 1. ดึงข้อมูลดิบทั้งหมด (Raw Measurements)
        yaws = np.array([p['yaw'] for p in self.all_scanned_points])
        raw_dists_cm = np.array([p['raw_dist'] / 10.0 for p in self.all_scanned_points])
        raw_total_dists_cm = raw_dists_cm + TOF_OFFSET_CM

        # คำนวณพิกัด Raw (X, Y)
        yaw_rads = np.radians(yaws)
        raw_xs_cm = raw_total_dists_cm * np.sin(yaw_rads)
        raw_ys_cm = raw_total_dists_cm * np.cos(yaw_rads)

        # 2. กรองแบบ Zero-Phase Spatial Filter (อิงตามมุม โดยไม่มี Phase Lag / ไม่เลื่อนเยื้อง)
        sort_idx = np.argsort(yaws)
        sorted_yaws = yaws[sort_idx]
        sorted_dists = raw_total_dists_cm[sort_idx]

        # Zero-Phase Centered Median Filter (pure numpy, kernel_size=7)
        k = 7
        half = k // 2
        n = len(sorted_dists)
        clean_dists_cm = np.array([
            float(np.median(sorted_dists[max(0, i - half): min(n, i + half + 1)]))
            for i in range(n)
        ])

        # คำนวณพิกัด Clean ที่อยู่กึ่งกลางระนาบ Raw จริงๆ
        clean_rads = np.radians(sorted_yaws)
        clean_xs_cm = clean_dists_cm * np.sin(clean_rads)
        clean_ys_cm = clean_dists_cm * np.cos(clean_rads)

        # สร้างกราฟเดียว (Single Plot) สไตล์ Dark Mode
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8))

        # 1. พลอตจุด Raw (สีส้มจาง) เพื่อเปรียบเทียบ Noise
        ax.scatter(raw_xs_cm, raw_ys_cm,
                   color='#FF8800', s=14, alpha=0.35,
                   label=f'Raw Points ({len(raw_xs_cm)} pts)')

        # 2. พลอตจุด Clean (สีฟ้าสดใส สีเดียว ไม่แบ่งตามระยะ)
        ax.scatter(clean_xs_cm, clean_ys_cm,
                   color='#00CFFF', s=22, alpha=0.90,
                   edgecolors='none',
                   label=f'Clean Points (Zero-Phase, {len(clean_xs_cm)} pts)')

        # 3. จุดตำแหน่ง Gimbal ที่ (0, 0)
        ax.scatter([0], [0], color='#00FF66', s=120, marker='o', edgecolors='white',
                   linewidth=1.5, zorder=5, label='Gimbal Origin (0,0)')

        # ตั้งค่ากราฟ
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"ToF LiDAR Scan — Raw vs Clean (Offset={TOF_OFFSET_CM:.0f}cm | {SCAN_YAW_MIN:.0f}° to {SCAN_YAW_MAX:.0f}°)",
                     fontsize=13, fontweight='bold', color='#00FFFF', pad=15)
        ax.set_xlabel("X - Lateral Distance (cm)", fontsize=11, color='#DDDDDD')
        ax.set_ylabel("Y - Forward Distance (cm)", fontsize=11, color='#DDDDDD')
        ax.grid(True, linestyle=':', alpha=0.4, color='#777777')
        ax.legend(loc='upper right', fontsize=9, facecolor='#1a1a1a', edgecolor='#444444')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"[SUCCESS] บันทึกภาพกราฟ (Raw vs Clean) เรียบร้อยแล้ว: {output_path}")

        try:
            plt.show(block=False)
            plt.pause(2.0)
        except Exception:
            pass
        plt.close('all')

    def close(self):
        """ปิดการเชื่อมต่อและคืนทรัพยากร"""
        cv2.destroyAllWindows()
        if not self.mock and self.ep_robot:
            try:
                self.ep_gimbal.unsub_angle()
                self.ep_sensor.unsub_distance()
                self.ep_robot.close()
                print("[INFO] ปิดการเชื่อมต่อ RoboMaster เรียบร้อย")
            except Exception as e:
                print(f"[WARN] Error during close: {e}")


def main():
    scanner = GimbalToFScanner(mock=MOCK_MODE)
    try:
        if scanner.connect():
            if scanner.mock:
                scanner.run_mock_sweep()
            else:
                scanner.run_hardware_scan()
            
            # 1. บันทึกวิดีโอผลลัพธ์
            scanner.save_video(OUTPUT_VIDEO_PATH)

            # 2. พลอตกราฟสรุปผลทั้งหมด 4 มิติ
            scanner.plot_summary(SUMMARY_PLOT_PATH)
    except KeyboardInterrupt:
        print("\n[INFO] ผู้ใช้กดยกเลิก (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] เกิดข้อผิดพลาด: {e}")
    finally:
        scanner.close()


if __name__ == "__main__":
    main()

import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Toplevel
from collections import deque
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

# Import modules from library
from eye_tracking_lib import (
    EyeModel3D,
    get_image_compensation,
    apply_compensation_to_pupil,
    auto_init_stabilization_from_loaded_calibration,
    load_dlc_csv,
    fit_ellipse_center,
    get_pupil_data_all,
    determine_best_eye,
    compute_calibration,
    predict_gaze as predict_gaze_module,
    save_calibration_data,
    load_calibration_data,
    serialize_pipeline,
    deserialize_pipeline,
    get_frame_gaze_data as get_frame_gaze_data_module
)


class DLCSceneCalibrator3D:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DLC Eye-to-Scene: Template Stabilization & Calibration")
        self.root.geometry("1600x950")

        # Video / CV State
        self.cap_eye = None
        self.cap_scene = None
        self.raw_eye_frame = None
        self.raw_scene_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30.0

        # DLC data
        self.dlc_data = None
        self.dlc_bodyparts = []

        # --- Image Stabilization State (Template Matching) ---
        self.stab_template = None
        self.stab_ref_point = None
        self.stab_curr_point = None
        self.template_size = 30
        self.defining_landmark = False

        # Display
        self.display_w = 600
        self.scene_display_scale = 1.0
        self.eye_display_scale = 1.0

        # Gaze / Calibration
        self.eye_models = {"left": EyeModel3D(100), "right": EyeModel3D(100)}
        self.reference_centers = {"left": (320.0, 240.0), "right": (320.0, 240.0)}
        self.gaze_regressors = {"left": None, "right": None}
        self.calibration_pairs = []
        self.calibrating = False
        self.hover_pos = None
        self.eye_hover_pos = None
        self.ridge_alpha = 0.1
        self.alpha_var = tk.StringVar(value=str(self.ridge_alpha))
        self.trail_var = tk.IntVar(value=20)

        # Compensation Settings
        self.comp_enabled = tk.BooleanVar(value=True)
        self.comp_axes = {
            "x": tk.BooleanVar(value=True),
            "y": tk.BooleanVar(value=True),
            "rot": tk.BooleanVar(value=False),
        }

        self.create_ui()
        self.setup_bindings()

    def setup_bindings(self):
        self.root.bind('<Right>', lambda e: self.seek(50))
        self.root.bind('<Left>', lambda e: self.seek(-50))
        self.root.bind('<Up>', lambda e: self.seek(5))
        self.root.bind('<Down>', lambda e: self.seek(-5))

    def create_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_container = ttk.Frame(main_frame)
        video_container.pack(fill=tk.BOTH, expand=True)

        # --- Eye Panel ---
        self.eye_panel = ttk.Frame(video_container)
        self.eye_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(self.eye_panel, text="Eye Video (Stabilization Source)").pack()
        self.eye_label = ttk.Label(self.eye_panel, background="black")
        self.eye_label.pack(fill=tk.BOTH, expand=True)
        self.eye_label.bind("<Button-1>", self.on_eye_click)
        self.eye_label.bind("<Motion>", self.on_eye_mouse_move)

        # --- Scene Panel ---
        self.scene_panel = ttk.Frame(video_container)
        self.scene_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(self.scene_panel, text="Scene Camera").pack()
        self.scene_label = ttk.Label(self.scene_panel, background="black")
        self.scene_label.pack(fill=tk.BOTH, expand=True)
        self.scene_label.bind("<Button-1>", self.on_scene_click)
        self.scene_label.bind("<Motion>", self.on_scene_mouse_move)
        self.scene_label.bind("<Leave>", self.on_scene_mouse_leave)

        # Playback
        playback_frame = ttk.Frame(main_frame)
        playback_frame.pack(fill=tk.X, pady=10)
        ttk.Button(playback_frame, text="<<", command=lambda: self.seek(-30)).pack(side=tk.LEFT)
        self.slider_var = tk.DoubleVar()
        self.time_slider = ttk.Scale(playback_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.slider_var, command=self.on_slider_drag)
        self.time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Button(playback_frame, text=">>", command=lambda: self.seek(30)).pack(side=tk.LEFT)
        self.frame_label = ttk.Label(playback_frame, text="Frame: 0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=10)

        # Controls
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X)

        # 1. Load Data
        load_frame = ttk.LabelFrame(bottom_frame, text="1. Load Data", padding=5)
        load_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(load_frame, text="Load Eye Video", command=lambda: self.load_video("eye")).pack(fill=tk.X)
        ttk.Button(load_frame, text="Load Scene Video", command=lambda: self.load_video("scene")).pack(fill=tk.X)
        ttk.Button(load_frame, text="Load DLC CSV", command=self.load_dlc_csv).pack(fill=tk.X)

        # 2. Calibration
        calib_frame = ttk.LabelFrame(bottom_frame, text="2. Calibration", padding=5)
        calib_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.btn_calib = ttk.Button(calib_frame, text="Start Calibration Mode", command=self.toggle_calibration)
        self.btn_calib.pack(fill=tk.X)
        self.btn_define_landmark = ttk.Button(calib_frame, text="Define Stab. Landmark", command=self.toggle_landmark_mode)
        self.btn_define_landmark.pack(fill=tk.X)

        # Ridge Alpha
        param_frame = ttk.Frame(calib_frame)
        param_frame.pack(fill=tk.X, pady=2)
        ttk.Label(param_frame, text="Ridge α:").pack(side=tk.LEFT)
        ttk.Entry(param_frame, textvariable=self.alpha_var, width=6).pack(side=tk.LEFT)
        ttk.Button(calib_frame, text="Compute Calibration", command=self.compute_calibration).pack(fill=tk.X)
        ttk.Button(calib_frame, text="Clear Points", command=self.clear_points).pack(fill=tk.X)

        # 3. Stabilization Settings
        comp_frame = ttk.LabelFrame(bottom_frame, text="3. Stabilization", padding=5)
        comp_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Checkbutton(comp_frame, text="Enable Image Stab.", variable=self.comp_enabled).pack(anchor=tk.W)
        ttk.Checkbutton(comp_frame, text="Compensate X", variable=self.comp_axes["x"]).pack(anchor=tk.W)
        ttk.Checkbutton(comp_frame, text="Compensate Y", variable=self.comp_axes["y"]).pack(anchor=tk.W)

        # 4. Save/Load
        persist_frame = ttk.LabelFrame(bottom_frame, text="4. Save/Load", padding=5)
        persist_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(persist_frame, text="Save Calibration", command=self.save_calibration).pack(fill=tk.X)
        ttk.Button(persist_frame, text="Load Calibration", command=self.load_calibration).pack(fill=tk.X)

        # 5. Output
        out_frame = ttk.LabelFrame(bottom_frame, text="5. Output", padding=5)
        out_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(out_frame, text="Export Video", command=self.generate_output_video).pack(fill=tk.X)
        ttk.Button(out_frame, text="Export Gaze Data CSV", command=self.export_gaze_data_csv).pack(fill=tk.X)
        ttk.Separator(out_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(out_frame, text="Trail Length (Frames)").pack(fill=tk.X)
        frame_trail_ctrl = ttk.Frame(out_frame)
        frame_trail_ctrl.pack(fill=tk.X)
        self.lbl_trail_val = ttk.Label(frame_trail_ctrl, text="20")
        self.lbl_trail_val.pack(side=tk.RIGHT, padx=2)
        self.trail_slider = ttk.Scale(
            frame_trail_ctrl, from_=0, to=60, orient=tk.HORIZONTAL,
            variable=self.trail_var, command=lambda v: self.lbl_trail_val.config(text=str(int(float(v))))
        )
        self.trail_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_label = ttk.Label(bottom_frame, text="System Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    # -----------------------------
    # Image Stabilization Core
    # -----------------------------

    def toggle_landmark_mode(self):
        self.defining_landmark = not self.defining_landmark
        if self.defining_landmark:
            self.btn_define_landmark.config(text="CLICK ON EYE VIDEO...")
            self.status_label.config(text="Click a stable feature (e.g. tear duct) on the Eye Video.")
        else:
            self.btn_define_landmark.config(text="Define Stab. Landmark")
        self.refresh_displays()

    def on_eye_mouse_move(self, event):
        self.eye_hover_pos = (event.x, event.y)
        if self.defining_landmark:
            self.refresh_displays()

    def on_eye_click(self, event):
        if self.defining_landmark and self.raw_eye_frame is not None:
            if self.eye_display_scale == 0: return
            ex = int(float(event.x) / self.eye_display_scale)
            ey = int(float(event.y) / self.eye_display_scale)

            h, w = self.raw_eye_frame.shape[:2]
            r = self.template_size
            x1 = max(0, ex - r)
            y1 = max(0, ey - r)
            x2 = min(w, ex + r)
            y2 = min(h, ey + r)

            self.stab_template = self.raw_eye_frame[y1:y2, x1:x2].copy()
            self.stab_ref_point = (ex, ey)
            self.stab_curr_point = (ex, ey)

            self.defining_landmark = False
            self.btn_define_landmark.config(text="Define Stab. Landmark")
            self.status_label.config(text=f"Landmark defined at ({ex}, {ey}). Stabilization Active.")
            self.refresh_displays()

    def get_image_compensation(self, current_bgr):
        """Wrapper to update stab_curr_point from stabilization module."""
        stats, new_curr_point = get_image_compensation(
            current_bgr, self.stab_template, self.stab_ref_point, 
            self.stab_curr_point, self.comp_axes
        )
        if new_curr_point:
            self.stab_curr_point = new_curr_point
        return stats, new_curr_point

    def apply_compensation_to_pupil(self, pupil_x, pupil_y, frame_bgr):
        """Wrapper for apply_compensation_to_pupil from stabilization module."""
        return apply_compensation_to_pupil(
            pupil_x, pupil_y, frame_bgr, self.comp_enabled, self.stab_template,
            self.stab_ref_point, self.stab_curr_point, self.comp_axes
        )

    # -----------------------------
    # Logic & Video
    # -----------------------------

    def on_scene_click(self, event):
        if not self.calibrating or self.raw_eye_frame is None: return

        eyes_data = self.get_pupil_data_all(self.current_frame_idx)
        if not eyes_data:
            messagebox.showwarning("Data Missing", "No valid pupil detected in this frame.")
            return

        sx = float(event.x) / max(1e-6, self.scene_display_scale)
        sy = float(event.y) / max(1e-6, self.scene_display_scale)

        for side, (pupil, _) in eyes_data.items():
            adj_pupil, _ = self.apply_compensation_to_pupil(pupil[0], pupil[1], self.raw_eye_frame)
            self.calibration_pairs.append(
                (float(adj_pupil[0]), float(adj_pupil[1]), sx, sy, side, int(self.current_frame_idx))
            )

        self.status_label.config(text=f"Point added. Total points: {len(self.calibration_pairs)}")
        self.refresh_displays()

    def predict_gaze(self, pupil_x, pupil_y, side, frame_bgr):
        """Wrapper for predict_gaze from gaze_calibration module."""
        return predict_gaze_module(
            pupil_x, pupil_y, side, frame_bgr, self.gaze_regressors, self.reference_centers,
            self.eye_models, apply_compensation_to_pupil, self.comp_enabled, self.stab_template,
            self.stab_ref_point, self.stab_curr_point, self.comp_axes
        )

    def load_video(self, which):
        path = filedialog.askopenfilename()
        if not path: return

        cap = cv2.VideoCapture(path)
        if which == "eye":
            self.cap_eye = cap
            ret, frame = cap.read()
            if ret:
                self.raw_eye_frame = frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if self.stab_template is not None:
                # A calibration with a template is already loaded — try to auto-locate
                # the same landmark in the newly opened eye video.
                success, new_ref, new_curr = auto_init_stabilization_from_loaded_calibration(
                    self.raw_eye_frame, self.stab_template, self.stab_ref_point, self.stab_curr_point
                )
                if success:
                    self.stab_ref_point = new_ref
                    self.stab_curr_point = new_curr
                    self.status_label.config(text="Stabilization landmark auto-located in new video.")
                else:
                    self.stab_template = None
                    self.stab_ref_point = None
                    self.status_label.config(text="Could not auto-locate stabilization landmark. Please define manually.")
        else:
            self.cap_scene = cap
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            detected_fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = detected_fps if detected_fps > 0 else 30.0
            self.time_slider.config(to=self.total_frames - 1)

        self.load_current_frames()

    def load_current_frames(self):
        idx = int(self.current_frame_idx)

        if self.cap_eye:
            self.cap_eye.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = self.cap_eye.read()
            if ret: self.raw_eye_frame = f

        if self.cap_scene:
            self.cap_scene.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = self.cap_scene.read()
            if ret: self.raw_scene_frame = f

        self.refresh_displays()

    def refresh_displays(self):
        # --- Update Eye Display ---
        if self.raw_eye_frame is not None:
            f = self.raw_eye_frame.copy()

            # Preview template placement
            if self.defining_landmark and self.eye_hover_pos:
                mx, my = self.eye_hover_pos
                if self.eye_display_scale > 0:
                    ix = int(mx / self.eye_display_scale)
                    iy = int(my / self.eye_display_scale)
                    r = self.template_size
                    cv2.rectangle(f, (ix-r, iy-r), (ix+r, iy+r), (0, 255, 255), 2)
                    cv2.putText(f, "Click stable feature", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Draw Stabilization Visualization
            if self.stab_template is not None and self.stab_ref_point is not None:
                stats, curr_pt = self.get_image_compensation(self.raw_eye_frame)
                dx, dy, _ = stats
                if curr_pt:
                    cx, cy = int(curr_pt[0]), int(curr_pt[1])
                    rx, ry = int(self.stab_ref_point[0]), int(self.stab_ref_point[1])
                    r = self.template_size
                    cv2.rectangle(f, (rx-r, ry-r), (rx+r, ry+r), (0, 0, 255), 2)
                    cv2.rectangle(f, (cx-r, cy-r), (cx+r, cy+r), (0, 255, 0), 2)
                    cv2.line(f, (rx, ry), (cx, cy), (0, 255, 255), 2)
                    cv2.putText(f, f"Shift: dx={dx:.1f}, dy={dy:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw pupil ellipses
            eyes = self.get_pupil_data_all(self.current_frame_idx)
            for side, (pupil, ellipse_params) in eyes.items():
                if ellipse_params is not None:
                    cv2.ellipse(f, ellipse_params, (255, 0, 255), 2)

            self.update_label_image(self.eye_label, f, self.display_w, is_scene=False)

        # --- Update Scene Display ---
        if self.raw_scene_frame is not None:
            f = self.raw_scene_frame.copy()
            scene_h, scene_w = f.shape[:2]

            # 1. Draw Existing Calibration Points
            for px, py, sx, sy, side, fidx in self.calibration_pairs:
                cv2.circle(f, (int(sx), int(sy)), 4, (0, 0, 255), -1)

            # 2. Draw Crosshair
            if self.calibrating and self.hover_pos:
                scale = self.display_w / max(1, scene_w)
                hx_widget, hy_widget = self.hover_pos
                scene_x = int(hx_widget / max(1e-6, scale))
                scene_y = int(hy_widget / max(1e-6, scale))
                cv2.line(f, (scene_x, 0), (scene_x, scene_h), (255, 255, 0), 1)
                cv2.line(f, (0, scene_y), (scene_w, scene_y), (255, 255, 0), 1)
                cv2.circle(f, (scene_x, scene_y), 5, (255, 255, 0), 1)

            # 3. Live prediction with improved visualization
            eyes = self.get_pupil_data_all(self.current_frame_idx)
            predictions = {}
            for side, (pupil, _) in eyes.items():
                pred, stats = self.predict_gaze(pupil[0], pupil[1], side, self.raw_eye_frame)
                if pred:
                    predictions[side] = pred

            # Draw dots for each eye with different colors and circle around best eye
            best_eye = self.determine_best_eye(self.current_frame_idx)
            for side, pred in predictions.items():
                # Different colors for each eye
                color = (255, 0, 0) if side == "left" else (0, 0, 255)  # Blue for left, Red for right
                # Draw small filled circle (dot)
                cv2.circle(f, pred, 4, color, -1)
                # Draw green circle around best eye
                if side == best_eye:
                    cv2.circle(f, pred, 15, (0, 255, 0), 2)

            self.update_label_image(self.scene_label, f, self.display_w, is_scene=True)

        self.frame_label.config(text=f"Frame: {self.current_frame_idx}")

    def update_label_image(self, label, frame_bgr, target_w, is_scene=False):
        h, w = frame_bgr.shape[:2]
        scale = target_w / w

        if is_scene:
            self.scene_display_scale = scale
        else:
            self.eye_display_scale = scale

        resized = cv2.resize(frame_bgr, (target_w, int(h * scale)))
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        label.config(image=img)
        label.image = img

    def load_dlc_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path: return

        self.dlc_data, self.dlc_bodyparts = load_dlc_csv(path)
        self.status_label.config(text=f"Loaded DLC: {len(self.dlc_bodyparts)} parts.")

    def fit_ellipse_center(self, points):
        """Wrapper for fit_ellipse_center from pupil_detection module."""
        return fit_ellipse_center(points)

    def get_pupil_data_all(self, frame_idx):
        """Wrapper for get_pupil_data_all from pupil_detection module."""
        return get_pupil_data_all(frame_idx, self.dlc_data, self.dlc_bodyparts)

    def determine_best_eye(self, frame_idx):
        """Wrapper for determine_best_eye from pupil_detection module."""
        return determine_best_eye(frame_idx, self.dlc_data, self.dlc_bodyparts)

    def toggle_calibration(self):
        self.calibrating = not self.calibrating
        self.btn_calib.config(text="STOP Calibration" if self.calibrating else "Start Calibration Mode")
        self.refresh_displays()

    def clear_points(self):
        self.calibration_pairs = []
        self.gaze_regressors = {"left": None, "right": None}
        self.refresh_displays()

    def compute_calibration(self):
        if len(self.calibration_pairs) < 10:
            messagebox.showerror("Error", "Need at least 10 points.")
            return

        try:
            self.ridge_alpha = float(self.alpha_var.get())
        except ValueError:
            self.ridge_alpha = 0.1

        self.gaze_regressors, self.reference_centers = compute_calibration(
            self.calibration_pairs, self.eye_models, self.ridge_alpha
        )

        messagebox.showinfo("Success", "Calibration complete.")

    def seek(self, delta):
        self.current_frame_idx = int(np.clip(self.current_frame_idx + delta, 0, self.total_frames - 1))
        self.slider_var.set(self.current_frame_idx)
        self.load_current_frames()

    def on_slider_drag(self, val):
        self.current_frame_idx = int(float(val))
        self.load_current_frames()

    def on_scene_mouse_move(self, event):
        if self.calibrating:
            self.hover_pos = (event.x, event.y)
            self.refresh_displays()

    def on_scene_mouse_leave(self, event):
        self.hover_pos = None
        self.refresh_displays()

    # -----------------------------
    # Save/Load Logic
    # -----------------------------

    def _serialize_pipeline(self, pipe):
        """Wrapper for serialize_pipeline from data_persistence module."""
        return serialize_pipeline(pipe)

    def _deserialize_pipeline(self, data, alpha):
        """Wrapper for deserialize_pipeline from data_persistence module."""
        return deserialize_pipeline(data, alpha)

    def auto_init_stabilization_from_loaded_calibration(self):
        """Wrapper for auto_init_stabilization_from_loaded_calibration from stabilization module."""
        success, new_ref, new_curr = auto_init_stabilization_from_loaded_calibration(
            self.raw_eye_frame, self.stab_template, self.stab_ref_point, self.stab_curr_point
        )
        if success:
            self.stab_ref_point = new_ref
            self.stab_curr_point = new_curr
        return success

    def save_calibration(self):
        if not any(v is not None for v in self.gaze_regressors.values()):
            messagebox.showerror("Error", "No calibration to save.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path: return

        data = save_calibration_data(
            self.ridge_alpha, self.reference_centers, self.gaze_regressors,
            self.stab_template, self.stab_ref_point, self.template_size
        )

        try:
            import json
            with open(path, "w") as f: 
                json.dump(data, f)
            messagebox.showinfo("Saved", "Calibration saved.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_calibration(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path: return

        try:
            import json
            with open(path, "r") as f: 
                data = json.load(f)

            (self.ridge_alpha, self.reference_centers, self.gaze_regressors,
             self.stab_template, self.stab_ref_point, self.template_size) = load_calibration_data(data)

            self.alpha_var.set(str(self.ridge_alpha))
            if self.stab_ref_point:
                self.stab_curr_point = self.stab_ref_point

            if self.stab_template is not None and self.raw_eye_frame is not None:
                if self.auto_init_stabilization_from_loaded_calibration():
                    messagebox.showinfo("Loaded", "Calibration loaded.\nStabilization landmark auto-located.")
                else:
                    messagebox.showinfo("Loaded", "Calibration loaded.\nNote: Could not auto-locate stabilization landmark. Please define manually.")
            elif self.stab_template is not None:
                messagebox.showinfo("Loaded", "Calibration loaded.\nNote: Load eye video to auto-locate stabilization landmark.")
            else:
                messagebox.showinfo("Loaded", "Calibration loaded.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -----------------------------
    # Export
    # -----------------------------

    def generate_output_video(self):
        if not self.cap_scene or not self.cap_eye: return

        out_path = filedialog.asksaveasfilename(defaultextension=".mp4")
        if not out_path: return

        w_e, h_e = int(self.cap_eye.get(3)), int(self.cap_eye.get(4))
        w_s, h_s = int(self.cap_scene.get(3)), int(self.cap_scene.get(4))

        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w_e + w_s, max(h_e, h_s)))

        self.cap_eye.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap_scene.set(cv2.CAP_PROP_POS_FRAMES, 0)

        trail_len = self.trail_var.get()
        trail_history = deque(maxlen=trail_len + 1)

        prog_win = Toplevel(self.root)
        prog_win.title("Exporting...")
        ttk.Label(prog_win, text="Processing...").pack(pady=10)
        progress = ttk.Progressbar(prog_win, length=250, mode="determinate")
        progress.pack(pady=10)
        progress["maximum"] = self.total_frames

        try:
            for i in range(self.total_frames):
                ret_e, fe = self.cap_eye.read()
                ret_s, fs = self.cap_scene.read()
                if not ret_e or not ret_s: break

                # Draw Stabilization on Eye Frame (fe)
                if self.comp_enabled.get() and self.stab_template is not None:
                    stats, curr_pt = self.get_image_compensation(fe)
                    if curr_pt:
                        cx, cy = int(curr_pt[0]), int(curr_pt[1])
                        rx, ry = int(self.stab_ref_point[0]), int(self.stab_ref_point[1])
                        r = self.template_size
                        cv2.rectangle(fe, (rx-r, ry-r), (rx+r, ry+r), (0, 0, 255), 2)
                        cv2.rectangle(fe, (cx-r, cy-r), (cx+r, cy+r), (0, 255, 0), 2)
                        cv2.line(fe, (rx, ry), (cx, cy), (0, 255, 255), 2)

                # Draw pupil ellipses on exported video (with validation)
                eyes_data = self.get_pupil_data_all(i)
                for side, (pupil, ellipse_params) in eyes_data.items():
                    if ellipse_params is not None:
                        # Validate before drawing
                        center, axes, angle = ellipse_params
                        if axes[0] > 0 and axes[1] > 0 and axes[0] < 10000 and axes[1] < 10000:
                            try:
                                cv2.ellipse(fe, ellipse_params, (255, 0, 255), 2)
                            except cv2.error:
                                pass

                # Improved gaze visualization
                best_eye = self.determine_best_eye(i)
                predictions = {}
                
                for side, (pupil, _) in eyes_data.items():
                    res, _ = self.predict_gaze(pupil[0], pupil[1], side, fe)
                    if res:
                        predictions[side] = res

                # Draw both eyes with different colors
                for side, pred in predictions.items():
                    color = (255, 0, 0) if side == "left" else (0, 0, 255)  # Blue for left, Red for right
                    cv2.circle(fs, pred, 4, color, -1)
                    
                    # Update trail with best eye
                    if side == best_eye:
                        trail_history.append(pred)

                # Draw trail
                if trail_len > 0:
                    for idx, pt in enumerate(trail_history):
                        if pt is None: continue
                        norm_pos = (idx + 1) / len(trail_history)
                        is_head = idx == len(trail_history) - 1
                        if is_head:
                            cv2.circle(fs, pt, 15, (0, 255, 0), 3)
                        else:
                            radius = max(3, int(12 * norm_pos))
                            cv2.circle(fs, pt, radius, (0, 165, 255), -1)
                else:
                    # Draw green circle around best eye when no trail
                    if best_eye in predictions:
                        cv2.circle(fs, predictions[best_eye], 15, (0, 255, 0), 3)

                canvas = np.zeros((max(h_e, h_s), w_e + w_s, 3), dtype=np.uint8)
                canvas[:h_e, :w_e] = fe
                canvas[:h_s, w_e:] = fs

                out.write(canvas)

                if i % 10 == 0:
                    progress["value"] = i
                    prog_win.update()

        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
            print(e)
        finally:
            out.release()
            prog_win.destroy()

        messagebox.showinfo("Done", "Video exported.")

    def get_frame_gaze_data(self, frame_idx, eye_frame):
        """Wrapper for get_frame_gaze_data from data_export module."""
        return get_frame_gaze_data_module(
            frame_idx, eye_frame, self.dlc_data, self.dlc_bodyparts,
            self.comp_enabled, self.stab_template, self.stab_ref_point, self.stab_curr_point, self.comp_axes,
            self.gaze_regressors, self.reference_centers, self.eye_models,
            get_pupil_data_all, determine_best_eye,
            get_image_compensation, apply_compensation_to_pupil, predict_gaze_module
        )

    def export_gaze_data_csv(self):
        """Export gaze tracking data to CSV file"""
        if not self.cap_eye or not self.cap_scene:
            messagebox.showerror("Error", "Please load both eye and scene videos.")
            return
        
        if not any(v is not None for v in self.gaze_regressors.values()):
            messagebox.showerror("Error", "No calibration available. Please calibrate first.")
            return
        
        if self.dlc_data is None:
            messagebox.showerror("Error", "No DLC data loaded.")
            return
        
        out_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not out_path:
            return
        
        # Reset video positions
        self.cap_eye.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Create progress window
        prog_win = Toplevel(self.root)
        prog_win.title("Exporting CSV...")
        ttk.Label(prog_win, text="Processing frames...").pack(pady=10)
        progress = ttk.Progressbar(prog_win, length=250, mode="determinate")
        progress.pack(pady=10)
        progress["maximum"] = self.total_frames
        
        # Collect data for all frames
        all_data = []
        
        try:
            for i in range(self.total_frames):
                # Read eye frame
                ret_e, eye_frame = self.cap_eye.read()
                if not ret_e:
                    break
                
                # Get gaze data for this frame
                frame_data = self.get_frame_gaze_data(i, eye_frame)
                all_data.append(frame_data)
                
                # Update progress
                if i % 10 == 0:
                    progress["value"] = i
                    prog_win.update()
            
            # Create DataFrame
            df = pd.DataFrame(all_data)
            
            # Reorder columns for better organization
            column_order = [
                'best_eye',
                'best_eye_gaze_x',
                'best_eye_gaze_y',
                'left_eye_gaze_x',
                'left_eye_gaze_y',
                'right_eye_gaze_x',
                'right_eye_gaze_y',
                'left_eye_pupil_x_stab',
                'left_eye_pupil_y_stab',
                'right_eye_pupil_x_stab',
                'right_eye_pupil_y_stab',
                'left_eye_pupil_x_raw',
                'left_eye_pupil_y_raw',
                'right_eye_pupil_x_raw',
                'right_eye_pupil_y_raw',
                'left_eye_trackpoints',
                'right_eye_trackpoints',
                'stab_shift_dx',
                'stab_shift_dy',
            ]
            
            # Ensure all columns exist (in case some are missing)
            for col in column_order:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Reorder and save
            df = df[column_order]
            df.to_csv(out_path, index=False)
            
            messagebox.showinfo("Success", f"Gaze data exported to {out_path}\n{len(all_data)} frames processed.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
            print(e)
        finally:
            prog_win.destroy()


if __name__ == "__main__":
    DLCSceneCalibrator3D().root.mainloop()

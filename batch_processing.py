import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import glob
import threading
import cv2
import numpy as np
import pandas as pd
from collections import deque
import json
import time

# Import modules from the provided library
# Ensure eye_tracking_lib folder is in the same directory as this script
try:
    from eye_tracking_lib import (
        EyeModel3D,
        get_image_compensation,
        apply_compensation_to_pupil,
        auto_init_stabilization_from_loaded_calibration,
        load_dlc_csv,
        get_pupil_data_all,
        determine_best_eye,
        predict_gaze as predict_gaze_module,
        load_calibration_data,
        get_frame_gaze_data as get_frame_gaze_data_module
    )
except ImportError:
    # Fallback/Error msg if library is missing
    print("Critical Error: 'eye_tracking_lib' not found. Please ensure the library files are in the same directory.")

class BatchProcessorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DLC Eye-to-Scene: Batch Processor")
        self.root.geometry("1000x800")

        # --- State Variables ---
        self.calibration_file = None
        self.calibration_data = None  # To hold loaded calibration params
        self.file_registry = []  # List of dicts: {'csv': path, 'face': path, 'scene': path, 'status': str}
        self.is_processing = False
        self.stop_event = threading.Event()

        # Calibration Params placeholders
        self.ridge_alpha = 0.1
        self.reference_centers = {}
        self.gaze_regressors = {}
        self.stab_template = None
        self.stab_ref_point = None
        self.template_size = 30

        # UI Variables
        self.output_dir = tk.StringVar(value="")
        self.calib_path_var = tk.StringVar(value="No File Selected")
        
        self.opt_export_csv = tk.BooleanVar(value=True)
        self.opt_export_video = tk.BooleanVar(value=True)
        self.opt_enable_stab = tk.BooleanVar(value=True)
        self.opt_comp_x = tk.BooleanVar(value=True)
        self.opt_comp_y = tk.BooleanVar(value=True)
        self.trail_len_var = tk.IntVar(value=20)
        
        # Eye Models
        self.eye_models = {"left": EyeModel3D(100), "right": EyeModel3D(100)}

        self.create_ui()

    def create_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 1. Configuration Section ---
        config_frame = ttk.LabelFrame(main_frame, text="1. Configuration & Calibration", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # Calibration Loader
        calib_row = ttk.Frame(config_frame)
        calib_row.pack(fill=tk.X, pady=5)
        ttk.Label(calib_row, text="Calibration File (.json):").pack(side=tk.LEFT)
        ttk.Entry(calib_row, textvariable=self.calib_path_var, state='readonly', width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(calib_row, text="Load Calibration", command=self.load_calibration_file).pack(side=tk.LEFT)

        # Output Folder
        out_row = ttk.Frame(config_frame)
        out_row.pack(fill=tk.X, pady=5)
        ttk.Label(out_row, text="Output Directory:").pack(side=tk.LEFT)
        self.ent_out_dir = ttk.Entry(out_row, textvariable=self.output_dir, width=50)
        self.ent_out_dir.pack(side=tk.LEFT, padx=5)
        ttk.Button(out_row, text="Browse...", command=self.browse_output_dir).pack(side=tk.LEFT)
        ttk.Label(out_row, text="(Leave empty to save in source folder)", font=("Arial", 8, "italic")).pack(side=tk.LEFT, padx=5)

        # Settings
        sets_frame = ttk.Frame(config_frame)
        sets_frame.pack(fill=tk.X, pady=5)
        
        # Export Options
        ttk.Label(sets_frame, text="Export:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(sets_frame, text="Gaze CSV", variable=self.opt_export_csv).pack(side=tk.LEFT)
        ttk.Checkbutton(sets_frame, text="Mosaic Video", variable=self.opt_export_video).pack(side=tk.LEFT, padx=10)
        
        ttk.Separator(sets_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Stabilization Options
        ttk.Label(sets_frame, text="Stabilization:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(sets_frame, text="Enable", variable=self.opt_enable_stab).pack(side=tk.LEFT)
        ttk.Checkbutton(sets_frame, text="Comp X", variable=self.opt_comp_x).pack(side=tk.LEFT)
        ttk.Checkbutton(sets_frame, text="Comp Y", variable=self.opt_comp_y).pack(side=tk.LEFT)

        ttk.Separator(sets_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Visual Options
        ttk.Label(sets_frame, text="Video Trail Length:").pack(side=tk.LEFT)
        ttk.Scale(sets_frame, from_=0, to=60, variable=self.trail_len_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, padx=5)
        ttk.Label(sets_frame, textvariable=self.trail_len_var).pack(side=tk.LEFT)

        # --- 2. File Selection Section ---
        files_frame = ttk.LabelFrame(main_frame, text="2. Batch File List", padding=10)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        btn_bar = ttk.Frame(files_frame)
        btn_bar.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_bar, text="Add DLC CSV Files...", command=self.add_files).pack(side=tk.LEFT)
        ttk.Button(btn_bar, text="Clear List", command=self.clear_files).pack(side=tk.LEFT, padx=5)
        ttk.Label(btn_bar, text=" * The system will attempt to auto-match video files based on the CSV filename.", foreground="gray").pack(side=tk.LEFT, padx=10)

        # Treeview for File Matching
        cols = ("CSV File", "Face Video", "Scene Video", "Status")
        self.tree = ttk.Treeview(files_frame, columns=cols, show='headings', selectmode='browse')
        
        self.tree.heading("CSV File", text="DLC CSV File")
        self.tree.heading("Face Video", text="Face/Eye Video")
        self.tree.heading("Scene Video", text="Scene Video")
        self.tree.heading("Status", text="Status")

        self.tree.column("CSV File", width=250)
        self.tree.column("Face Video", width=250)
        self.tree.column("Scene Video", width=250)
        self.tree.column("Status", width=100)

        scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- 3. Execution Section ---
        run_frame = ttk.LabelFrame(main_frame, text="3. Processing", padding=10)
        run_frame.pack(fill=tk.X)

        ctrl_row = ttk.Frame(run_frame)
        ctrl_row.pack(fill=tk.X, pady=5)
        
        self.btn_run = ttk.Button(ctrl_row, text="START BATCH PROCESSING", command=self.start_processing)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.btn_stop = ttk.Button(ctrl_row, text="STOP", command=self.stop_processing, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        # Progress Bars
        progress_frame = ttk.Frame(run_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(progress_frame, text="Total Batch Progress:").pack(anchor=tk.W)
        self.prog_batch = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.prog_batch.pack(fill=tk.X, pady=(2, 10))

        ttk.Label(progress_frame, text="Current File Progress:").pack(anchor=tk.W)
        self.prog_file = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.prog_file.pack(fill=tk.X, pady=(2, 5))

        self.lbl_status = ttk.Label(run_frame, text="Ready.", relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(fill=tk.X)

    # -------------------------------------------------------------------------
    # File Logic
    # -------------------------------------------------------------------------

    def load_calibration_file(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path: return

        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # Use library function to deserialize
            loaded_data = load_calibration_data(data)
            
            # Unpack
            (self.ridge_alpha, self.reference_centers, self.gaze_regressors,
             self.stab_template, self.stab_ref_point, self.template_size) = loaded_data

            self.calib_path_var.set(path)
            self.calibration_file = path
            
            status_msg = "Calibration Loaded."
            if self.stab_template is not None:
                status_msg += " (Includes Stabilization Template)"
            
            messagebox.showinfo("Success", status_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration:\n{e}")

    def browse_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV", "*.csv")])
        if not paths: return

        for p in paths:
            # Check if already added
            if any(entry['csv'] == p for entry in self.file_registry):
                continue
            
            face_vid, scene_vid = self.match_videos(p)
            
            status = "Ready"
            if not face_vid or not scene_vid:
                status = "MISSING VIDEO"

            entry = {
                'csv': p,
                'face': face_vid,
                'scene': scene_vid,
                'status': status
            }
            self.file_registry.append(entry)
            
            # Display just the filenames for readability, but store full paths
            csv_name = os.path.basename(p)
            face_name = os.path.basename(face_vid) if face_vid else "NOT FOUND"
            scene_name = os.path.basename(scene_vid) if scene_vid else "NOT FOUND"
            
            self.tree.insert("", tk.END, values=(csv_name, face_name, scene_name, status))

    def clear_files(self):
        self.file_registry = []
        for item in self.tree.get_children():
            self.tree.delete(item)

    def match_videos(self, csv_path):
        """
        Heuristic to find corresponding videos.
        Assumption based on user input:
        CSV: [SessionID]_faceDLC_....csv
        Scene: [SessionID]_scene.mp4
        Face: [SessionID]_...face...mp4 or similar
        """
        directory = os.path.dirname(csv_path)
        filename = os.path.basename(csv_path)
        
        # 1. Determine Session ID
        # Try splitting by '_faceDLC' as per example
        if "_faceDLC" in filename:
            session_id = filename.split("_faceDLC")[0]
        else:
            # Fallback: take first few segments or entire stem minus extension
            # This is a guess if the naming convention is strictly followed
            session_id = os.path.splitext(filename)[0]

        # Get all video files in directory
        vid_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        all_videos = []
        for ext in vid_extensions:
            all_videos.extend(glob.glob(os.path.join(directory, ext)))
        
        face_candidate = None
        scene_candidate = None

        # 2. Search for matches
        for vid in all_videos:
            bname = os.path.basename(vid)
            
            # Scene Matcher
            if session_id in bname and "scene" in bname:
                scene_candidate = vid
            
            # Face Matcher
            # Priority 1: Contains session_id AND "face"
            # Priority 2: Contains session_id AND "labeled" (as per prompt example)
            elif session_id in bname:
                if "face" in bname:
                    face_candidate = vid
                elif "labeled" in bname and face_candidate is None:
                    # If we haven't found an explicit "face" one, take the labeled one
                    face_candidate = vid
                elif face_candidate is None and "scene" not in bname:
                     # Fallback: if it has session ID and isn't scene, it might be the face one
                     face_candidate = vid

        return face_candidate, scene_candidate

    # -------------------------------------------------------------------------
    # Processing Logic
    # -------------------------------------------------------------------------

    def start_processing(self):
        if not self.calibration_file:
            messagebox.showerror("Error", "Please load a calibration file first.")
            return
        
        if not self.file_registry:
            messagebox.showerror("Error", "No files to process.")
            return

        # Check for missing files
        for entry in self.file_registry:
            if entry['status'] == "MISSING VIDEO":
                if not messagebox.askyesno("Warning", "Some files are missing videos. They will be skipped. Continue?"):
                    return
                break

        self.is_processing = True
        self.stop_event.clear()
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
        # Start Thread
        t = threading.Thread(target=self.process_batch)
        t.daemon = True
        t.start()

    def stop_processing(self):
        if self.is_processing:
            self.stop_event.set()
            self.lbl_status.config(text="Stopping... please wait for current file to finish.")

    def process_batch(self):
        total_files = len(self.file_registry)
        self.prog_batch['maximum'] = total_files
        self.prog_batch['value'] = 0

        # Pre-fetch configuration to avoid thread issues with TK vars
        config = {
            'export_csv': self.opt_export_csv.get(),
            'export_video': self.opt_export_video.get(),
            'enable_stab': self.opt_enable_stab.get(),
            'comp_axes': {'x': tk.BooleanVar(value=self.opt_comp_x.get()), 'y': tk.BooleanVar(value=self.opt_comp_y.get())}, # Wrapper for compatibility
            'trail_len': self.trail_len_var.get(),
            'out_dir': self.output_dir.get()
        }

        # Comp axes needs to mimic the dict of BooleanVars expected by the library
        # We created dummy BooleanVars above, but the library accesses .get(), so that works.

        for idx, entry in enumerate(self.file_registry):
            if self.stop_event.is_set():
                break

            # Update GUI for current file
            csv_name = os.path.basename(entry['csv'])
            self.root.after(0, lambda m=f"Processing {idx+1}/{total_files}: {csv_name}": self.lbl_status.config(text=m))
            
            # Skip invalid entries
            if entry['status'] != "Ready":
                self.root.after(0, lambda i=idx: self.update_tree_status(i, "SKIPPED"))
                self.root.after(0, lambda v=idx+1: self.prog_batch.configure(value=v))
                continue

            try:
                self.process_single_file(entry, config)
                self.root.after(0, lambda i=idx: self.update_tree_status(i, "DONE"))
            except Exception as e:
                print(f"Error processing {csv_name}: {e}")
                self.root.after(0, lambda i=idx: self.update_tree_status(i, "ERROR"))
            
            self.root.after(0, lambda v=idx+1: self.prog_batch.configure(value=v))

        self.is_processing = False
        self.root.after(0, self.processing_complete)

    def process_single_file(self, entry, config):
        """
        Core logic to process a single triplet of files (CSV, Eye Video, Scene Video).
        """
        csv_path = entry['csv']
        eye_path = entry['face']
        scene_path = entry['scene']

        # Determine Output Paths
        base_dir = config['out_dir'] if config['out_dir'] else os.path.dirname(csv_path)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        # Clean up name if it has DLC suffixes (optional aesthetic choice)
        if "_faceDLC" in base_name:
            clean_name = base_name.split("_faceDLC")[0]
        else:
            clean_name = base_name

        csv_out_path = os.path.join(base_dir, f"{clean_name}_GazeData.csv")
        vid_out_path = os.path.join(base_dir, f"{clean_name}_Mosaic.mp4")

        # 1. Load Data
        dlc_data, dlc_bodyparts = load_dlc_csv(csv_path)
        
        cap_eye = cv2.VideoCapture(eye_path)
        cap_scene = cv2.VideoCapture(scene_path)
        
        if not cap_eye.isOpened() or not cap_scene.isOpened():
            raise IOError("Could not open input videos.")

        total_frames = int(min(cap_eye.get(cv2.CAP_PROP_FRAME_COUNT), cap_scene.get(cv2.CAP_PROP_FRAME_COUNT)))
        # Update file progress bar max
        self.root.after(0, lambda m=total_frames: self.prog_file.configure(maximum=m))

        # 2. Auto-Initialize Stabilization for this specific video
        # We use the template from calibration, but we need to find where that feature is in THIS video
        curr_stab_ref = self.stab_ref_point
        curr_stab_curr = self.stab_ref_point # Start assuming it's at ref
        
        if config['enable_stab'] and self.stab_template is not None:
            # Read first frame to find template
            ret, first_frame = cap_eye.read()
            if ret:
                success, new_ref, new_curr = auto_init_stabilization_from_loaded_calibration(
                    first_frame, self.stab_template, self.stab_ref_point, self.stab_ref_point
                )
                if success:
                    curr_stab_ref = new_ref
                    curr_stab_curr = new_curr
                else:
                    # If auto-init fails, we disable stab for this file or warn?
                    # For batch, we'll try to proceed with original ref, but it might jump.
                    # Ideally, log warning.
                    print(f"Warning: Could not auto-locate stabilization template in {os.path.basename(eye_path)}")
                
                # Reset ptr
                cap_eye.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 3. Prepare Video Writer if needed
        video_writer = None
        if config['export_video']:
            w_e = int(cap_eye.get(3))
            h_e = int(cap_eye.get(4))
            w_s = int(cap_scene.get(3))
            h_s = int(cap_scene.get(4))
            fps = cap_scene.get(cv2.CAP_PROP_FPS)
            
            # Mosaic: Eye on left, Scene on right (or stacked? Previous script did Side-by-Side)
            out_size = (w_e + w_s, max(h_e, h_s))
            video_writer = cv2.VideoWriter(vid_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)
        
        # Trail history
        trail_history = deque(maxlen=config['trail_len'] + 1)
        
        # Data accumulation for CSV
        all_gaze_data = []

        # 4. Main Loop
        for i in range(total_frames):
            if self.stop_event.is_set():
                break

            ret_e, frame_e = cap_eye.read()
            ret_s, frame_s = cap_scene.read()
            
            if not ret_e or not ret_s:
                break

            # Get Data (Calls library function which handles prediction, stabilization calc, etc.)
            # Note: get_frame_gaze_data calculates stabilization shift internally if provided params
            frame_data = get_frame_gaze_data_module(
                i, frame_e, dlc_data, dlc_bodyparts,
                tk.BooleanVar(value=config['enable_stab']), # Create temp vars for function sig
                self.stab_template,
                curr_stab_ref,
                curr_stab_curr,
                config['comp_axes'],
                self.gaze_regressors,
                self.reference_centers,
                self.eye_models,
                get_pupil_data_all,
                determine_best_eye,
                get_image_compensation,
                apply_compensation_to_pupil,
                predict_gaze_module
            )

            # Update local stabilization pointer for next frame continuity
            # The library function calculates shift but doesn't persist the 'current point' for the NEXT frame logic 
            # inside itself (it's stateless). We need to update curr_stab_curr based on the shift found.
            # However, get_frame_gaze_data returns a dictionary. It includes 'stab_shift_dx/dy'.
            # We can update curr_stab_curr manually:
            if config['enable_stab'] and self.stab_template is not None and not np.isnan(frame_data['stab_shift_dx']):
                # Recalculate where the point is now based on shift
                # shift = curr - ref  => curr = ref + shift
                dx = frame_data['stab_shift_dx']
                dy = frame_data['stab_shift_dy']
                cx = curr_stab_ref[0] + dx
                cy = curr_stab_ref[1] + dy
                curr_stab_curr = (cx, cy)

            # Collect CSV Data
            if config['export_csv']:
                all_gaze_data.append(frame_data)

            # Draw Video
            if config['export_video'] and video_writer:
                # -- Draw Eye Side --
                # Draw stabilization box
                if config['enable_stab'] and self.stab_template is not None:
                    rx, ry = int(curr_stab_ref[0]), int(curr_stab_ref[1])
                    cx, cy = int(curr_stab_curr[0]), int(curr_stab_curr[1])
                    r = self.template_size
                    cv2.rectangle(frame_e, (rx-r, ry-r), (rx+r, ry+r), (0, 0, 255), 2) # Ref (Red)
                    cv2.rectangle(frame_e, (cx-r, cy-r), (cx+r, cy+r), (0, 255, 0), 2) # Curr (Green)

                # Draw pupil
                best_eye = frame_data['best_eye'] # 'L' or 'R'
                
                # Map 'L'/'R' back to 'left'/'right' keys for data lookup
                # frame_data has flattened keys like 'left_eye_pupil_x_raw'
                for side in ['left', 'right']:
                    px = frame_data.get(f'{side}_eye_pupil_x_raw')
                    py = frame_data.get(f'{side}_eye_pupil_y_raw')
                    if pd.notna(px) and pd.notna(py):
                        cv2.circle(frame_e, (int(px), int(py)), 4, (255, 0, 255), 2)

                # -- Draw Scene Side --
                gaze_pt = None
                if pd.notna(frame_data['best_eye_gaze_x']):
                    gaze_pt = (int(frame_data['best_eye_gaze_x']), int(frame_data['best_eye_gaze_y']))
                
                # Trail
                trail_history.append(gaze_pt)
                
                # Render Trail
                for t_idx, pt in enumerate(trail_history):
                    if pt is None: continue
                    norm_pos = (t_idx + 1) / len(trail_history)
                    is_head = (t_idx == len(trail_history) - 1)
                    
                    if is_head:
                        cv2.circle(frame_s, pt, 15, (0, 255, 0), 3) # Head
                    else:
                        radius = max(3, int(12 * norm_pos))
                        cv2.circle(frame_s, pt, radius, (0, 165, 255), -1) # Tail

                # Combine
                h_e, w_e = frame_e.shape[:2]
                h_s, w_s = frame_s.shape[:2]
                canvas = np.zeros((max(h_e, h_s), w_e + w_s, 3), dtype=np.uint8)
                canvas[:h_e, :w_e] = frame_e
                canvas[:h_s, w_e:] = frame_s
                
                video_writer.write(canvas)

            # Update Progress Bar periodically
            if i % 10 == 0:
                self.root.after(0, lambda v=i: self.prog_file.configure(value=v))

        # Cleanup Single File
        cap_eye.release()
        cap_scene.release()
        if video_writer:
            video_writer.release()
        
        # Save CSV
        if config['export_csv'] and all_gaze_data:
            df = pd.DataFrame(all_gaze_data)
            # Reorder columns same as original tool
            cols = [
                'best_eye', 'best_eye_gaze_x', 'best_eye_gaze_y',
                'left_eye_gaze_x', 'left_eye_gaze_y', 'right_eye_gaze_x', 'right_eye_gaze_y',
                'left_eye_pupil_x_stab', 'left_eye_pupil_y_stab', 'right_eye_pupil_x_stab', 'right_eye_pupil_y_stab',
                'left_eye_pupil_x_raw', 'left_eye_pupil_y_raw', 'right_eye_pupil_x_raw', 'right_eye_pupil_y_raw',
                'left_eye_trackpoints', 'right_eye_trackpoints', 'stab_shift_dx', 'stab_shift_dy',
            ]
            # Filter existing cols
            cols = [c for c in cols if c in df.columns]
            df = df[cols]
            df.to_csv(csv_out_path, index=False)

    def update_tree_status(self, index, status):
        # Update specific item in tree
        # Get item ID by index logic or store IDs. 
        # Treeview items are stored as children.
        children = self.tree.get_children()
        if 0 <= index < len(children):
            item_id = children[index]
            self.tree.set(item_id, "Status", status)

    def processing_complete(self):
        self.btn_run.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_status.config(text="Batch Processing Complete.")
        messagebox.showinfo("Done", "Batch processing finished.")

if __name__ == "__main__":
    app = BatchProcessorGUI()
    app.root.mainloop()

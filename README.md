# Eye-Tracking Gaze Mapper

A GUI-based Python tool that maps where a subject is looking onto a scene recorded by a separate camera, using pupil tracking data produced by [DeepLabCut](https://github.com/deeplabcut/deeplabcut). No commercial eye-tracker hardware required, and no command-line knowledge is needed beyond launching the scripts.

> **Important:** This tool does **not** perform pupil tracking itself. Pupil tracking is a **prerequisite step** that you must carry out independently using DeepLabCut before using this software. DeepLabCut analyzes your eye/face video and exports a CSV file with the pupil coordinates per frame. You must train your own DeepLabCut model and analyze the eye video first following the naming conventions explained below (in Step 1). This tool then reads that CSV and uses it to estimate gaze.

### Designed for

- **Head-fixed or near-fixed subjects** — designed for experimental setups where the head is fixed, with built-in compensation for small residual movements.
- **Real-world scene gaze** — maps gaze onto a real-world scene recorded from a different angle (e.g., a camera pointed at an object or environment the subject is viewing). Gaze targets are treated as flat (2D) surfaces; there is no 3D or depth estimation. Screen-based (display) gaze is also supported.
- **Humans and animals** — works with any subject with a round pupil that can be tracked by DeepLabCut, including non-human primates and other animals.

### Recording tips

**Eye camera field of view:** The closer the eye camera is to the eye (or the more zoomed in), the larger the pupil appears in the frame. More pixels on the pupil means DeepLabCut tracks it more accurately, which directly improves gaze accuracy. Aim to fill as much of the frame as possible with the eye region.

**Frame synchrony:** For guaranteed frame-by-frame synchrony between the eye camera and the scene camera, record both streams together as a **side-by-side mosaic in a single video file** using [OBS Studio](https://obsproject.com/) (free). You can later separate the eye/face and scene videos by cropping them from the mosaic.

**Calibration ground truth — what to record:** This software does not perform real-time calibration. Instead, the calibration is built from a sequence you must include in your recording, in which a small visible object is moved around the gaze target area while the subject is instructed to keep their eyes on it. This sequence provides the ground truth that the software later uses to learn the mapping between pupil position and scene coordinates.

Guidelines for the calibration ground truth sequence:

- Move the object across the **same plane** that is the target of the subject's gaze (e.g., the surface being viewed or a virtual surface close to the view target of interest).
- Cover the **four corners** and **several central points** of the target area for the highest accuracy.
- The sequence can be recorded at **any point** during the session — before the main task, after it, or in a separate block — but the subject's head and all cameras must remain in exactly the same position as during the main task.
- Starting or ending the session with this sequence is usually most practical.

---

## How it works — overview

```
Eye/Face Video (.mp4)
        │
        ▼
  DeepLabCut  ──► DLC CSV file (pupil x, y per frame)
  (you run this)                │
                                │
Scene Camera Video (.mp4) ──────┤
                                ▼
                  scene_camera_calibration.py
                  (interactive GUI — one time per setup)
                                │
                                ▼
                        calibration.json
                                │
        ┌───────────────────────┘
        │  DLC CSV + Face/Scene videos
        ▼
  batch_processing.py  ──► gaze_output.csv  (x, y per frame)
                       ──► mosaic video (optional side-by-side)
```

---

## Third-party software and licenses


| Software                                               | License             | Your responsibility                                                |
| ------------------------------------------------------ | ------------------- | ------------------------------------------------------------------ |
| [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) | LGPL-3.0            | Install and use independently under DeepLabCut's own license terms |
| This tool                                              | MIT (see `LICENSE`) | Free to use, modify, and redistribute                              |


> DeepLabCut must be installed and run independently. This tool only reads the CSV files it exports. You are responsible for complying with DeepLabCut's license.

---

## Requirements

- **Conda** (Anaconda or Miniconda) — used to create an isolated Python environment
- **DeepLabCut** (external, for generating input CSV files) — see [DeepLabCut installation guide](https://deeplabcut.github.io/DeepLabCut/docs/installation.html)
- The Python packages listed in `requirements.txt` (installed automatically during setup below)

---

## Installation

### 1. Install Conda

If you do not have Conda installed, download and install **Miniconda** (lightweight) or **Anaconda** (full distribution). Follow the instructions for your operating system at the [official Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### 2. Create a Conda environment

Open an **Anaconda Prompt** (Windows) or a terminal (macOS/Linux) and create a new environment with Python 3.9:

```bash
conda create -n gaze-mapper python=3.9
conda activate gaze-mapper
```

> Keep this environment active for all remaining steps and whenever you run the software.

### 3. Get the code

```bash
git clone https://github.com/nexflatline/gaze-to-scene-camera-mapper.git
cd gaze-to-scene-camera-mapper
```

Or download and extract the ZIP from the GitHub page (click the green **Code** button → **Download ZIP**), then navigate into the extracted folder in your terminal.

### 4. Install Python dependencies

With the `gaze-mapper` environment active and your terminal inside the project folder, run:

```bash
pip install -r requirements.txt
```

That installs: OpenCV, NumPy, pandas, scikit-learn, and Pillow.

> **tkinter** is required for the GUIs. It is included with the standard Python bundled in Conda on Windows and macOS.  
> On Linux: `conda install -c conda-forge tk`

---

## Quick start (step by step)

### Step 1 — Prepare your data with DeepLabCut

You need DeepLabCut to locate the pupil in each frame of the eye camera video.

**What is DeepLabCut?**  
DeepLabCut is a free AI tool for tracking body parts in video. Here it is used to locate the pupil. You train a model by labeling a small number of frames (marking where the pupil is), and then DeepLabCut can automatically find the pupil in the rest of the video.

**What body-part names must you use in DeepLabCut?**  
The tool finds pupil keypoints by looking for names that contain both `"pupil"` and either `"left"` or `"right"`. You need **at least 5 keypoints per pupil** so the software can fit an ellipse to find the pupil center. Using 8 points (evenly around the pupil rim) gives a more accurate ellipse fit.

**Example: 8-point pupil labeling scheme**

Imagine the pupil as a clock face. Label 8 points around its rim plus the center:


| DeepLabCut body-part name | Position on pupil |
| ------------------------- | ----------------- |
| `left_pupil_12h`          | 12 o'clock        |
| `left_pupil_1h30`         | 1:30              |
| `left_pupil_3h`           | 3 o'clock         |
| `left_pupil_4h30`         | 4:30              |
| `left_pupil_6h`           | 6 o'clock         |
| `left_pupil_7h30`         | 7:30              |
| `left_pupil_9h`           | 9 o'clock         |
| `left_pupil_10h30`        | 10:30             |


Use the exact same names for the right eye, replacing `left` with `right` (e.g., `right_pupil_12h`, `right_pupil_3h`, etc.).

> The names must contain `"pupil"` **and** `"left"` or `"right"` — but the exact naming pattern is flexible as long as those words appear. For example, `left_eye_pupil_top` or `pupil_left_1` also work.

**Exporting from DeepLabCut:**  
After running DeepLabCut analysis on your eye/face video, it will generate a CSV file. In the DeepLabCut GUI, use  
`Analyze Videos` → the CSV appears in the same folder as your video.  
The CSV filename will look like: `VideoName_DLC_resnet50_...csv`

---

### Step 2 — Run the calibration tool

> **Before starting:** The calibration tool does not perform real-time calibration. Instead, it reads frames from your already-recorded video where a calibration ground truth sequence was included (see **Recording tips** above). You will navigate to those frames inside the tool to collect calibration reference points.

Launch the calibration tool:

```bash
python scene_camera_calibration.py
```

The window shows two panels side by side: **Eye Video** (left) and **Scene Camera** (right).

#### 2a. Load your files

In the **"1. Load Data"** panel (bottom left):

1. Click **Load Eye Video** → select the face/eye camera video (`.mp4`).
2. Click **Load Scene Video** → select the scene camera video (`.mp4`).
3. Click **Load DLC CSV** → select the DeepLabCut CSV file for the eye video.

A colored ellipse should appear around the pupil in the Eye Video panel once all three files are loaded.

#### 2b. Define a stabilization landmark

In the **"2. Calibration"** panel:

The stabilization landmark corrects for small movements of the **subject's head** relative to the eye camera. The landmark must be a point that:

- **Moves rigidly with the skull** (so any head shift moves it by the same amount as the eye)
- **Does not move independently of the skull** — this rules out the eyes themselves (which rotate), the eyebrows (which raise), and the mouth (which opens)
- **Is visible in the eye camera frame**, close to the eye region in the image

Good choices are the **bridge of the nose** or the **hairline**. Avoid any facial feature that has its own movement relative to the skull.

1. Click **Define Stab. Landmark**.
2. The button text changes to **CLICK ON EYE VIDEO...**
3. Click on a suitable skull-fixed point (e.g., bridge of the nose) in the Eye Video panel.
4. A colored box appears around the selected region. The system will track this point frame-by-frame.

> Skip this step if the subject's head is rigidly fixed and does not move during the recording. The cameras are always assumed to be fixed; this compensation is only for head movement.

#### 2c. Collect calibration reference points

In the **"2. Calibration"** panel:

The calibration ground truth was established during recording by moving an object across the gaze target area while the subject tracked it with their eyes (see **Recording tips** above). In this step, you navigate to those frames in the video and tell the software where the object was on the scene camera at each moment. This is how the software learns to map pupil position to scene position — it uses only the pre-recorded video; there is no real-time interaction with the subject here.

1. Click **Start Calibration Mode**.
2. Navigate to a frame where the subject was looking at a **known location** on the scene (use the arrow keys or the slider to find the ground truth sequence).
3. **Click on that location** in the Scene Camera panel. The tool pairs the current pupil position (from the DLC CSV) with the point you clicked.
4. Repeat for **about 10 different gaze locations** spread across the scene. Make sure to include corners and central points of the target area.
5. Click **Stop Calibration** when done.

**Keyboard shortcuts:**  
`→` / `←` — jump ±30 frames  
`↑` / `↓` — jump ±5 frames

#### 2d. Compute the calibration model

Click **Compute Calibration**. A message box confirms success and a live gaze dot will appear on the scene camera panel.

#### 2e. Save the calibration

Click **Save Calibration** → choose a filename (e.g., `session01_calibration.json`).

> **Important:** The saved calibration is only valid for the **exact same recording setup** — the same head position and the same camera positions as when the calibration ground truth was recorded. It is designed to be reused across multiple trial recordings within the same session (where nothing has moved between trials). If the subject's head or any camera is repositioned, the calibration must be redone from scratch.

---

### Step 3 — Export gaze data for a single recording

After calibration you can export data directly from the calibration tool:

- **Export Gaze Data CSV** → produces a `.csv` with gaze x, y per frame.
- **Export Video** → produces a side-by-side `.mp4` with the gaze dot overlaid on the scene.

---

### Step 4 — Batch process multiple recordings

If you recorded multiple trials in the same session (same subject, same head and camera positions), you only need to calibrate once. The `.json` file saved in Step 2e is what makes this possible — it stores the full calibration model and can be applied to any number of recordings from the same session without recalibrating. Use the batch processor to apply it across all your trial files automatically:

```bash
python batch_processing.py
```

#### 4a. Load calibration

Click **Load Calibration** → select the `.json` calibration file saved in Step 2e.

#### 4b. Set output directory (optional)

Click **Browse...** to choose where output files will be saved. Leave blank to save in the same folder as the source files.

#### 4c. Add DLC CSV files

Click **Add CSV Files** → select one or more DLC CSV files.  
The tool will **automatically search for the matching videos** in the same folder using the file naming convention below.

#### 4d. File naming convention

For automatic video matching to work, files must follow this pattern:

```
[SessionID]_faceDLC_[anything].csv    ← DLC CSV for the eye/face video
[SessionID]_face[anything].mp4        ← Eye/face camera video
[SessionID]_scene[anything].mp4       ← Scene camera video
```

**Example:**

```
recordings/
├── subject01_session02_faceDLC_resnet50_labeled.csv
├── subject01_session02_face.mp4
└── subject01_session02_scene.mp4
```

The batch processor extracts `subject01_session02` as the session ID from the CSV filename (everything before `_faceDLC`), then finds files containing that ID plus `"face"` or `"scene"`.

If the face or scene video is shown as **NOT FOUND** in the table, you can double-click the row to assign videos manually.

#### 4e. Start processing

1. Select the export options (CSV, mosaic video, stabilization settings).
2. Click **Start Processing**.
  A progress log appears. Output files are written to the chosen output directory.

---

## Output file format (gaze CSV)


| Column                              | Description                                                   |
| ----------------------------------- | ------------------------------------------------------------- |
| `best_eye`                          | Which eye was used (`L` or `R`)                               |
| `best_eye_gaze_x` / `_y`            | Gaze position on scene (pixels) from the best-quality eye     |
| `left_eye_gaze_x` / `_y`            | Gaze position from the left eye (if tracked)                  |
| `right_eye_gaze_x` / `_y`           | Gaze position from the right eye (if tracked)                 |
| `left_eye_pupil_x_raw` / `_y_raw`   | Raw pupil center (pixels) in eye video — left eye             |
| `left_eye_pupil_x_stab` / `_y_stab` | Stabilization-compensated pupil center — left eye             |
| `right_eye_pupil_`*                 | Same columns for right eye                                    |
| `left_eye_trackpoints`              | Number of DLC keypoints above confidence threshold — left eye |
| `right_eye_trackpoints`             | Same for right eye                                            |
| `stab_shift_dx` / `_dy`             | Head movement compensation shift applied (pixels)             |


Missing values (`NaN`) mean the eye was not detectable in that frame.

---

## Calibration JSON

The `.json` file stores the fitted regression models, the stabilization template (as a base64-encoded image patch), and the reference pupil coordinates. It can be reused across all trial recordings within the **same session**, provided the subject's head position and all camera positions are unchanged between trials.

The calibration cannot extrapolate to a new camera or head alignment. If anything has moved — even slightly — between sessions or between a calibration and a new recording, the gaze estimates will be systematically off and the calibration must be redone.

---

## Troubleshooting

**No pupil ellipse visible in the Eye Video panel**  
→ Check that your DLC CSV body-part names contain `"pupil"` and `"left"` or `"right"`.  
→ Make sure the DLC CSV was exported for the correct video.  
→ Check the likelihood threshold: points below 0.4 confidence are discarded.

**Calibration fails / "Need at least 10 points"**  
→ You need at least 10 scene clicks total, with at least 5 per eye if both eyes are tracked.

**Videos not found in batch mode**  
→ Check that your filenames follow the naming convention above.  
→ Make sure the CSV and its matching videos are in the **same folder**.

**Exported video has wrong speed**  
→ The frame rate is read automatically from the scene video. If it still appears wrong, check the FPS of your source video with a media player.

---

## Author

Rafael Bretas  
Center for Information and Neural Networks (CiNet), National Institute of Information and Communications Technology (NICT)

---

## Acknowledgments

Special thanks to Dr. Kazuki Enomoto (Center for Information and Neural Networks) for providing the testing and validation data and for his help with testing the software.

This material is based upon work supported by the JSPS KAKENHI Grant Number JP25K10619.

---

## Citation

If you use this tool in a publication, please consider citing the repository and DeepLabCut.

```
Bretas, R. (2026). Eye-Tracking Gaze Mapper [Software].
National Institute of Information and Communications Technology (NICT).
https://github.com/nexflatline/gaze-to-scene-camera-mapper
```

---

## License

Copyright (c) 2026 National Institute of Information and Communications Technology

This software is released under the [MIT License](LICENSE).  
You are free to use, modify, and redistribute it with attribution.

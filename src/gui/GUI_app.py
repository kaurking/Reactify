import cv2
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from core.obs_bridge import OBSBridge, ObsEffect

from gesture_detector import GestureDetector
from core import MemeLibrary, TriggerEngine, AudioPlayer, VisualRenderer
from core.profile_store import (
    list_profiles,
    save_profile,
    delete_profile,
    copy_asset_to_project,
    sanitize_gesture_name,
)


class ReactifyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reactify")
        self.root.geometry("1100x760")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.detector = None
        self.running = False

        self.library = MemeLibrary()
        self.library.load_defaults()

        self.trigger_engine = TriggerEngine(
            effects=self.library.effects,
            min_confidence=0.15,
        )

        self.audio_player = AudioPlayer()
        self.visual_renderer = VisualRenderer()
        self.obs_bridge = OBSBridge(password="")
        self.obs_bridge.connect()

        self.current_photo = None

        self.last_log_time = 0.0
        self.log_interval = 0.5

        self.loaded_profiles = []

        self.pending_samples = []
        self.sampling_active = False
        self.sample_goal = 5
        self.sample_interval = 2.0
        self.next_sample_time = 0.0
        self.sampling_started_from_profiles = False

        self.profile_sampling_status_var = None
        self.sampling_button = None
        self.cancel_sampling_button = None
        self.notebook = None
        self.camera_tab = None
        self.profiles_tab = None

        self._build_ui()

    def _build_ui(self):
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            root_frame,
            text="Reactify",
            font=("Segoe UI", 20, "bold"),
        )
        title.pack(anchor=tk.W, pady=(0, 10))

        self.notebook = ttk.Notebook(root_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.camera_tab = ttk.Frame(self.notebook, padding=10)
        self.profiles_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.camera_tab, text="Camera")
        self.notebook.add(self.profiles_tab, text="Pose Profiles")

        self._build_camera_tab(self.camera_tab)
        self._build_profiles_tab(self.profiles_tab)

        self._log("GUI ready.")

    def _build_camera_tab(self, parent):
        content = ttk.Frame(parent)
        content.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(content)
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side_panel = ttk.Frame(content, padding=(15, 0))
        side_panel.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_controls(side_panel)
        self._build_status(side_panel)
        self._build_effects(side_panel)
        self._build_logs(side_panel)

    def _build_controls(self, parent):
        box = ttk.LabelFrame(parent, text="Controls", padding=10)
        box.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(box, text="Camera index:").pack(anchor=tk.W)

        self.camera_index_var = tk.StringVar(value="0")

        self.camera_entry = ttk.Entry(
            box,
            textvariable=self.camera_index_var,
            width=10,
        )
        self.camera_entry.pack(anchor=tk.W, pady=(0, 10))

        self.start_button = ttk.Button(
            box,
            text="Start Camera",
            command=self.start_camera,
        )
        self.start_button.pack(fill=tk.X, pady=3)

        self.stop_button = ttk.Button(
            box,
            text="Stop Camera",
            command=self.stop_camera,
            state=tk.DISABLED,
        )
        self.stop_button.pack(fill=tk.X, pady=3)

    def _build_status(self, parent):
        box = ttk.LabelFrame(parent, text="Detection Status", padding=10)
        box.pack(fill=tk.X, pady=(0, 10))

        self.running_var = tk.StringVar(value="Stopped")
        self.gesture_var = tk.StringVar(value="None")
        self.confidence_var = tk.StringVar(value="0.0")
        self.landmarks_var = tk.StringVar(value="No")
        self.effect_var = tk.StringVar(value="None")
        self.hold_progress_var = tk.StringVar(value="0%")

        self._status_row(box, "App:", self.running_var)
        self._status_row(box, "Gesture:", self.gesture_var)
        self._status_row(box, "Confidence:", self.confidence_var)
        self._status_row(box, "Landmarks:", self.landmarks_var)
        self._status_row(box, "Effect:", self.effect_var)
        self._status_row(box, "Hold:", self.hold_progress_var)

    def _build_effects(self, parent):
        box = ttk.LabelFrame(parent, text="Loaded Effects", padding=10)
        box.pack(fill=tk.X, pady=(0, 10))

        self.loaded_effects_var = tk.StringVar()
        self._update_loaded_effects_label()

        ttk.Label(
            box,
            textvariable=self.loaded_effects_var,
            wraplength=280,
        ).pack(anchor=tk.W)

    def _build_logs(self, parent):
        box = ttk.LabelFrame(parent, text="Logs", padding=10)
        box.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(box, height=18, width=40, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_profiles_tab(self, parent):
        left = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        right = ttk.Frame(parent)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(
            left,
            text="Existing Profiles",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 5))

        self.profile_listbox = tk.Listbox(left, width=38, height=22)
        self.profile_listbox.pack(fill=tk.Y, expand=True)

        self.profile_listbox.bind("<<ListboxSelect>>", self.on_profile_selected)

        ttk.Button(
            left,
            text="Refresh",
            command=self.refresh_profiles,
        ).pack(fill=tk.X, pady=(8, 3))

        ttk.Button(
            left,
            text="Delete Selected Profile",
            command=self.delete_selected_profile,
        ).pack(fill=tk.X, pady=3)

        ttk.Label(
            right,
            text="Create / Edit Profile",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 10))

        form = ttk.Frame(right)
        form.pack(fill=tk.X)

        self.profile_display_name_var = tk.StringVar(value="Absolute Cinema")
        self.profile_gesture_var = tk.StringVar(value="absolute_cinema")
        self.profile_threshold_var = tk.StringVar(value="0.55")
        self.profile_cooldown_var = tk.StringVar(value="3.0")
        self.profile_overlay_duration_var = tk.StringVar(value="1.8")
        self.profile_image_var = tk.StringVar(value="assets/images/Absolute_Cinema.png")
        self.profile_sound_var = tk.StringVar(value="assets/sounds/vine-boom.mp3")
        self.profile_samples_var = tk.StringVar(value="Samples recorded: 0")

        self._form_row(form, "Display name:", self.profile_display_name_var)
        self._form_row(form, "Gesture ID:", self.profile_gesture_var)
        self._form_row(form, "Threshold:", self.profile_threshold_var)
        self._form_row(form, "Cooldown:", self.profile_cooldown_var)
        self._form_row(form, "Overlay duration:", self.profile_overlay_duration_var)

        image_row = ttk.Frame(form)
        image_row.pack(fill=tk.X, pady=3)

        ttk.Label(image_row, text="Image:", width=18).pack(side=tk.LEFT)
        ttk.Entry(image_row, textvariable=self.profile_image_var).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )
        ttk.Button(
            image_row,
            text="Browse",
            command=self.browse_profile_image,
        ).pack(side=tk.LEFT, padx=(5, 0))

        sound_row = ttk.Frame(form)
        sound_row.pack(fill=tk.X, pady=3)

        ttk.Label(sound_row, text="Sound:", width=18).pack(side=tk.LEFT)
        ttk.Entry(sound_row, textvariable=self.profile_sound_var).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )
        ttk.Button(
            sound_row,
            text="Browse",
            command=self.browse_profile_sound,
        ).pack(side=tk.LEFT, padx=(5, 0))

        sampling_box = ttk.LabelFrame(right, text="Pose Sampling", padding=10)
        sampling_box.pack(fill=tk.X, pady=(15, 10))

        ttk.Label(
            sampling_box,
            textvariable=self.profile_samples_var,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W, pady=(0, 5))

        self.profile_sampling_status_var = tk.StringVar(
            value="Ready. Click Start Guided Sampling to record pose samples."
        )

        ttk.Label(
            sampling_box,
            textvariable=self.profile_sampling_status_var,
            wraplength=560,
        ).pack(anchor=tk.W, pady=(0, 8))

        self.sampling_button = ttk.Button(
            sampling_box,
            text="Start Guided Sampling",
            command=self.start_profile_sampling,
        )
        self.sampling_button.pack(anchor=tk.W, pady=3)

        self.cancel_sampling_button = ttk.Button(
            sampling_box,
            text="Cancel Sampling",
            command=self.cancel_profile_sampling,
            state=tk.DISABLED,
        )
        self.cancel_sampling_button.pack(anchor=tk.W, pady=3)

        ttk.Button(
            right,
            text="Save Profile",
            command=self.save_current_profile,
        ).pack(anchor=tk.W, pady=5)

        ttk.Button(
            right,
            text="Reload Profiles",
            command=self.reload_profiles_runtime,
        ).pack(anchor=tk.W, pady=5)

        help_text = (
            "Start the camera first, then click the sample button and hold the pose. "
            "The app takes 5 samples, one every 2 seconds. "
            "Use threshold to control strictness: lower = stricter, higher = easier to trigger."
        )

        ttk.Label(
            right,
            text=help_text,
            wraplength=600,
        ).pack(anchor=tk.W, pady=(15, 0))

        self.refresh_profiles()

    def _status_row(self, parent, label_text, variable):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text=label_text, width=12).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=variable).pack(side=tk.LEFT)

    def _form_row(self, parent, label, variable):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)

        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)

        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        return entry

    def start_camera(self):
        if self.running:
            return True

        try:
            camera_index = int(self.camera_index_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid camera index",
                "Camera index must be a number.",
            )
            return False

        try:
            self._log("Starting detector...")
            self.detector = GestureDetector(debug=True)

            self._log("Opening camera...")
            self.cap = cv2.VideoCapture(camera_index)

            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam.")

            self.running = True

            self.running_var.set("Running")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.camera_entry.config(state=tk.DISABLED)

            self._log("Camera started.")
            self.update_frame()

            return True

        except Exception as error:
            self._log(f"Failed to start: {error}")
            messagebox.showerror("Startup error", str(error))
            self.cleanup_camera()
            return False

    def stop_camera(self):
        if not self.running:
            return

        self._log("Stopping camera...")

        self.running = False
        self.sampling_active = False
        self.sampling_started_from_profiles = False
        self.cleanup_camera()

        if self.profile_sampling_status_var is not None:
            self.profile_sampling_status_var.set("Camera stopped. Sampling inactive.")

        if self.sampling_button is not None:
            self.sampling_button.config(state=tk.NORMAL)

        if self.cancel_sampling_button is not None:
            self.cancel_sampling_button.config(state=tk.DISABLED)

        self.running_var.set("Stopped")
        self.gesture_var.set("None")
        self.confidence_var.set("0.0")
        self.landmarks_var.set("No")
        self.effect_var.set("None")
        self.hold_progress_var.set("0%")

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.camera_entry.config(state=tk.NORMAL)

        self._log("Camera stopped.")

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self._log("Could not read frame.")
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)

        detection = self.detector.detect(frame)

        if self.sampling_active:
            effect = None
        else:
            effect = self.trigger_engine.get_triggered_effect(detection)

        hold_progress = self.trigger_engine.get_hold_progress()
        candidate = self.trigger_engine.get_current_candidate()
        self.hold_progress_var.set(
            f"{int(hold_progress * 100)}%" if candidate else "0%"
        )

        now = time.time()
        self.handle_profile_sampling(detection, now)

        has_landmarks = detection.landmarks is not None

        self.gesture_var.set(detection.gesture or "None")
        self.confidence_var.set(str(detection.confidence))
        self.landmarks_var.set("Yes" if has_landmarks else "No")

        if self.sampling_active:
            seconds_left = max(0.0, self.next_sample_time - now)
            self.effect_var.set(f"Sampling in {seconds_left:.1f}s")
        else:
            self.effect_var.set(effect.name if effect else "None")

        if now - self.last_log_time >= self.log_interval:
            self._log(
                f"gesture={detection.gesture}, "
                f"confidence={detection.confidence}, "
                f"landmarks={has_landmarks}, "
                f"effect={effect.name if effect else None}"
            )
            self.last_log_time = now
        '''
        if effect:
            self._log(f"Triggered: {effect.name}")
            self.audio_player.play(effect)
            self.visual_renderer.trigger_overlay(effect)
        '''
        if effect:
            self._log(f"Triggered: {effect.name}")

            # Local preview/debug behavior
            self.audio_player.play(effect)
            self.visual_renderer.trigger_overlay(effect)

            # OBS output behavior
            self.obs_bridge.trigger_effect(
                ObsEffect(
                    scene_name="Reactify",
                    image_source_name="AbsoluteCinemaOverlay",
                    sound_source_name="VineBoomSound",
                    duration=effect.overlay_duration,
                )
            )

        frame = self.visual_renderer.render(frame)

        if self.sampling_active:
            frame = self._draw_sampling_countdown(frame, now)

        preview_frame = self._resize_for_preview(
            frame,
            max_width=740,
            max_height=580,
        )

        self._show_frame(preview_frame)

        self.root.after(15, self.update_frame)

    def _draw_sampling_countdown(self, frame, now):
        seconds_left = max(0.0, self.next_sample_time - now)
        samples_taken = len(self.pending_samples)

        if seconds_left <= 0.15:
            countdown_text = "Capturing..."
        else:
            countdown_text = f"{seconds_left:.1f}s"

        text_lines = [
            "POSE SAMPLING MODE",
            f"Samples: {samples_taken}/{self.sample_goal}",
            f"Next capture: {countdown_text}",
            "Hold your pose clearly",
        ]

        x = 30
        y = 45

        cv2.rectangle(
            frame,
            (20, 15),
            (430, 175),
            (0, 0, 0),
            -1,
        )

        for line in text_lines:
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 35

        return frame

    def _resize_for_preview(self, frame, max_width=740, max_height=580):
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height)

        if scale >= 1:
            return frame

        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv2.resize(frame, (new_width, new_height))

    def _show_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        success, encoded_image = cv2.imencode(".ppm", rgb_frame)

        if not success:
            self._log("Could not encode frame for GUI.")
            return

        self.current_photo = tk.PhotoImage(data=encoded_image.tobytes())
        self.video_label.config(image=self.current_photo)

    def refresh_profiles(self):
        self.profile_listbox.delete(0, tk.END)

        self.loaded_profiles = list_profiles()

        for profile in self.loaded_profiles:
            display_name = profile.get("display_name") or profile.get("gesture", "unknown")
            gesture = profile.get("gesture", "unknown")
            sample_count = len(profile.get("samples", []))

            self.profile_listbox.insert(
                tk.END,
                f"{display_name} ({gesture}) - {sample_count} samples",
            )

    def on_profile_selected(self, event=None):
        selection = self.profile_listbox.curselection()

        if not selection:
            return

        index = selection[0]
        profile = self.loaded_profiles[index]

        self.profile_display_name_var.set(profile.get("display_name", ""))
        self.profile_gesture_var.set(profile.get("gesture", ""))
        self.profile_threshold_var.set(str(profile.get("threshold", 0.55)))
        self.profile_cooldown_var.set(str(profile.get("cooldown", 3.0)))
        self.profile_overlay_duration_var.set(str(profile.get("overlay_duration", 1.8)))
        self.profile_image_var.set(profile.get("image", ""))
        self.profile_sound_var.set(profile.get("sound", ""))

        self.pending_samples = profile.get("samples", [])
        self.profile_samples_var.set(f"Samples recorded: {len(self.pending_samples)}")

    def delete_selected_profile(self):
        selection = self.profile_listbox.curselection()

        if not selection:
            messagebox.showwarning("No profile selected", "Select a profile first.")
            return

        index = selection[0]
        profile = self.loaded_profiles[index]

        gesture = profile.get("gesture")

        if not gesture:
            return

        confirmed = messagebox.askyesno(
            "Delete profile",
            f"Delete profile '{gesture}'?",
        )

        if not confirmed:
            return

        deleted = delete_profile(gesture)

        if deleted:
            self._log(f"Deleted profile: {gesture}")
            self.reload_profiles_runtime()
            self.refresh_profiles()

    def browse_profile_image(self):
        path = filedialog.askopenfilename(
            title="Choose meme image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.webp *.gif"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        try:
            relative_path = copy_asset_to_project(path, "image")
            self.profile_image_var.set(relative_path)
            self._log(f"Selected image: {relative_path}")
        except Exception as error:
            messagebox.showerror("Image error", str(error))

    def browse_profile_sound(self):
        path = filedialog.askopenfilename(
            title="Choose sound effect",
            filetypes=[
                ("Sound files", "*.mp3 *.wav *.ogg"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        try:
            relative_path = copy_asset_to_project(path, "sound")
            self.profile_sound_var.set(relative_path)
            self._log(f"Selected sound: {relative_path}")
        except Exception as error:
            messagebox.showerror("Sound error", str(error))

    def start_profile_sampling(self):
        gesture = sanitize_gesture_name(self.profile_gesture_var.get())

        if not gesture:
            messagebox.showerror(
                "Missing gesture ID",
                "Enter a gesture ID before recording samples.",
            )
            return

        if not self.running:
            started = self.start_camera()

            if not started:
                return

        if self.notebook is not None and self.camera_tab is not None:
            self.notebook.select(self.camera_tab)

        self.pending_samples = []
        self.sampling_active = True
        self.sampling_started_from_profiles = True
        self.sample_goal = 5
        self.sample_interval = 2.0
        self.next_sample_time = time.time() + 3.0

        self.profile_samples_var.set("Samples recorded: 0")
        self.profile_sampling_status_var.set(
            "Sampling started. Move into position. First sample in 3 seconds."
        )

        if self.sampling_button is not None:
            self.sampling_button.config(state=tk.DISABLED)

        if self.cancel_sampling_button is not None:
            self.cancel_sampling_button.config(state=tk.NORMAL)

        self._log("Started guided pose sampling.")

    def cancel_profile_sampling(self):
        if not self.sampling_active:
            return

        self.sampling_active = False
        self.sampling_started_from_profiles = False

        self.profile_sampling_status_var.set(
            f"Sampling cancelled. Kept {len(self.pending_samples)} samples."
        )

        if self.sampling_button is not None:
            self.sampling_button.config(state=tk.NORMAL)

        if self.cancel_sampling_button is not None:
            self.cancel_sampling_button.config(state=tk.DISABLED)

        self.effect_var.set("Sampling cancelled")
        self._log("Sampling cancelled.")

    def handle_profile_sampling(self, detection, now):
        if not self.sampling_active:
            return

        seconds_left = max(0.0, self.next_sample_time - now)
        samples_taken = len(self.pending_samples)

        self.profile_sampling_status_var.set(
            f"Hold the pose. Next sample in {seconds_left:.1f}s. "
            f"Samples: {samples_taken}/{self.sample_goal}"
        )

        if now < self.next_sample_time:
            return

        if detection.sample_data is None:
            if detection.landmarks is None:
                status = (
                    "No body pose detected. Retrying in 2 seconds. "
                    "Step back so your upper body is visible."
                )
                self._log("Sampling skipped: no pose landmarks detected.")
            else:
                status = (
                    "Pose detected, but it could not be normalized. Retrying in 2 seconds. "
                    "Keep both shoulders visible."
                )
                self._log("Sampling skipped: pose could not be normalized.")

            self.profile_sampling_status_var.set(status)
            self.next_sample_time = now + self.sample_interval
            return

        self.pending_samples.append(detection.sample_data)

        samples_taken = len(self.pending_samples)

        self.profile_samples_var.set(f"Samples recorded: {samples_taken}")
        self._log(f"Recorded sample {samples_taken}/{self.sample_goal}")

        if samples_taken >= self.sample_goal:
            self.sampling_active = False
            self.sampling_started_from_profiles = False
            self.profile_sampling_status_var.set(
                "Sampling complete. Review the fields, then click Save Profile."
            )
            self.effect_var.set("Sampling complete")

            if self.sampling_button is not None:
                self.sampling_button.config(state=tk.NORMAL)

            if self.cancel_sampling_button is not None:
                self.cancel_sampling_button.config(state=tk.DISABLED)

            self._log("Sampling complete. Click Save Profile.")
            return

        self.next_sample_time = now + self.sample_interval

    def save_current_profile(self):
        gesture = sanitize_gesture_name(self.profile_gesture_var.get())

        if not gesture:
            messagebox.showerror("Invalid gesture", "Gesture ID cannot be empty.")
            return

        if not self.pending_samples:
            messagebox.showwarning(
                "No samples",
                "Record pose samples before saving this profile.",
            )
            return

        try:
            threshold = float(self.profile_threshold_var.get())
            cooldown = float(self.profile_cooldown_var.get())
            overlay_duration = float(self.profile_overlay_duration_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid number",
                "Threshold, cooldown, and overlay duration must be numbers.",
            )
            return

        profile = {
            "gesture": gesture,
            "display_name": self.profile_display_name_var.get().strip() or gesture,
            "detection_layers": ["pose", "face", "derived"],
            "layer_weights": {
                "pose": 0.25,
                "face": 0.25,
                "derived": 0.50,
            },
            "image": self.profile_image_var.get().strip(),
            "sound": self.profile_sound_var.get().strip(),
            "threshold": threshold,
            "cooldown": cooldown,
            "overlay_duration": overlay_duration,
            "samples": self.pending_samples,
        }

        path = save_profile(profile)

        self._log(f"Saved profile: {path}")
        self.profile_sampling_status_var.set("Profile saved and reloaded.")
        self.reload_profiles_runtime()
        self.refresh_profiles()

    def reload_profiles_runtime(self):
        self.library.reload()

        self.trigger_engine = TriggerEngine(
            effects=self.library.effects,
            min_confidence=0.15,
        )

        if self.detector is not None:
            self.detector.reload_templates()

        self._update_loaded_effects_label()
        self._log(f"Reloaded profiles: {list(self.library.effects.keys())}")

    def _update_loaded_effects_label(self):
        if not hasattr(self, "loaded_effects_var"):
            return

        loaded_effects = ", ".join(self.library.effects.keys())
        self.loaded_effects_var.set(
            loaded_effects if loaded_effects else "No effects loaded"
        )

    def cleanup_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.detector is not None:
            self.detector.close()
            self.detector = None

    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def on_close(self):
        self.running = False
        self.sampling_active = False
        self.cleanup_camera()
        self.audio_player.close()
        self.obs_bridge.disconnect()
        self.root.destroy()


def launch_gui():
    root = tk.Tk()
    ReactifyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()

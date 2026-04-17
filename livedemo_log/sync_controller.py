"""
Multi-Modal Synchronization Controller

Main orchestrator for synchronized data collection from multiple sensing modalities.
Wraps existing collectors (PicoScenes, RealSense) without modifying their code.
"""

import math
import time
import threading
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from multimodal_collection.timestamp_manager import TimestampManager
from multimodal_collection.frame_saver import FrameSaver
from multimodal_collection.visualization import RealtimeVisualizer
from picoscenes_collection.preprocessing import (
    WiFiPreprocessingPipeline,
    WiFiPacketContext,
)
from picoscenes_collection.utils.csi import extract_spatial_streams
from multimodal_collection.gesture_classifier import GestureClassifier


class SyncSession:
    """Represents a synchronized collection session."""

    def __init__(self, session_id: str, output_dir: Path, config: Dict):
        self.session_id = session_id
        self.output_dir = output_dir
        self.config = config
        self.start_time = None
        self.end_time = None
        self.t0 = None

    def to_dict(self) -> Dict:
        """Convert session to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'output_dir': str(self.output_dir),
            'start_time': self.start_time,
            'end_time': self.end_time,
            't0_reference': self.t0,
            'duration_seconds': (self.end_time - self.start_time) if self.end_time else None,
            'config': self.config
        }


class WiFiCollectorWrapper:
    """
    Wraps picoscenes_collection.Dataset_collector without modification.
    Intercepts packet handler to add sync metadata.
    """

    def __init__(self, collector, session_dir: Path, timestamp_manager: TimestampManager,
                 config: Dict, visualizer: Optional[RealtimeVisualizer] = None):
        self.collector = collector
        self.session_dir = session_dir
        self.timestamp_mgr = timestamp_manager
        self.config = config
        self.visualizer = visualizer
        self.running = False
        self.thread = None
        self.preprocessor_pipeline: Optional[WiFiPreprocessingPipeline] = None
        self.gesture_classifier: Optional[GestureClassifier] = None

        # Create output directory for WiFi data
        self.wifi_dir = session_dir / "wifi_csi"
        self.wifi_dir.mkdir(parents=True, exist_ok=True)

        # Redirect collector's save path
        self.original_save_path = self.collector.save_path
        self.collector.save_path = self.wifi_dir

        # Optimize sync metadata saving with batching
        self.sync_metadata_buffer = {}  # device -> list of metadata dicts
        self.sync_metadata_lock = threading.Lock()
        self.sync_metadata_batch_size = 100  # Batch 100 packets before writing

        vis_cfg = self.config.get('visualization', {})
        self.max_wifi_subcarriers = int(vis_cfg.get('max_wifi_subcarriers', 256))
        self.max_wifi_streams = int(vis_cfg.get('max_wifi_streams', 4))

        preprocessing_cfg = self.config.get('wifi', {}).get('preprocessing', {})
        self.preprocessor_pipeline = WiFiPreprocessingPipeline(
            preprocessing_cfg,
            max_streams=self.max_wifi_streams,
            max_subcarriers=self.max_wifi_subcarriers,
        )
        self._sync_preprocessor_descriptors()
        self._init_gesture_classifier()

    def _init_gesture_classifier(self) -> None:
        gesture_cfg = (self.config or {}).get('gesture', {}) or {}
        enable = bool(gesture_cfg.get('enable', False))
        if not enable:
            self.gesture_classifier = None
            return
        model_path = gesture_cfg.get('model_path') or ''
        class_names = None
        classes_file = gesture_cfg.get('class_names_file')
        if classes_file:
            try:
                with open(classes_file, 'r') as f:
                    class_names = [ln.strip() for ln in f if ln.strip()]
            except Exception:
                class_names = None
        try:
            self.gesture_classifier = GestureClassifier(model_path=model_path, class_names=class_names)
            print(f"[Gesture] Loaded model from: {model_path}")
        except Exception as exc:
            print(f"[Gesture] WARNING: Failed to load model '{model_path}': {exc}")
            self.gesture_classifier = None

    def _sync_preprocessor_descriptors(self) -> None:
        if not self.preprocessor_pipeline:
            return
        self.preprocessor_pipeline.set_limits(
            max_streams=self.max_wifi_streams,
            max_subcarriers=self.max_wifi_subcarriers,
        )
        if self.visualizer:
            try:
                self.visualizer.set_wifi_preprocessors(
                    self.preprocessor_pipeline.get_descriptors()
                )
            except AttributeError:
                pass

    def configure_visualization(
        self, visualizer: Optional[RealtimeVisualizer], vis_cfg: Dict[str, Any]
    ) -> None:
        self.visualizer = visualizer
        self.max_wifi_subcarriers = int(vis_cfg.get('max_wifi_subcarriers', 256))
        self.max_wifi_streams = int(vis_cfg.get('max_wifi_streams', 4))
        self._sync_preprocessor_descriptors()

    def wrap_packet_handler(self):
        """Wrap the packet handler to add sync metadata."""
        original_handler = self.collector._packet_handler

        def sync_handler(packet):
            # Extract packet data
            _, packet_data = next(iter(packet.items()))

            # Get timestamps
            timestamp_hw = packet_data['RxSBasic']['systemns']
            timestamp_sys = time.monotonic_ns()

            # Call original handler first
            original_handler(packet)

            # After packet is saved, save sync metadata
            # Find the device and packet ID from the saved packet
            macs = (packet_data['StandardHeader']['Addr1'],
                   packet_data['StandardHeader']['Addr2'],
                   packet_data['StandardHeader']['Addr3'])

            device_name = self.collector._get_device_name(macs)
            if device_name and self.config.get('wifi', {}).get('save_sync_metadata', True):
                # Get current packet count for this device
                packet_id = self.collector.device_counts.get(device_name, 0)

                # Save sync metadata as JSON
                self._save_sync_metadata(device_name, packet_id, timestamp_sys, timestamp_hw)

                # Record in timestamp manager
                self.timestamp_mgr.record_timestamp(
                    'wifi', packet_id, timestamp_sys, timestamp_hw
                )

                if self.visualizer:
                    self._publish_wifi_visual(packet_data, device_name, packet_id,
                                               timestamp_sys, timestamp_hw)

        self.collector._packet_handler = sync_handler

    def _save_sync_metadata(self, device_name: str, packet_id: int,
                           timestamp_sys: int, timestamp_hw: int):
        """
        Save sync metadata as JSON file.

        Uses batching to reduce I/O overhead:
        - Buffers metadata in memory
        - Writes multiple files in one batch
        - Only flushes when buffer is full
        """
        device_run_dir = self.wifi_dir / device_name / self.collector.current_run
        # Lazily create the run directory on first packet for this device
        try:
            device_run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        sync_data = {
            'packet_id': packet_id,
            'device_name': device_name,
            'timestamp_sys': timestamp_sys,
            'timestamp_hw': timestamp_hw,
            't_relative': timestamp_sys - self.timestamp_mgr.t0 if self.timestamp_mgr.t0 else 0,
            'session_id': self.session_dir.name
        }

        # Buffer metadata (add to in-memory buffer)
        with self.sync_metadata_lock:
            if device_name not in self.sync_metadata_buffer:
                self.sync_metadata_buffer[device_name] = []

            self.sync_metadata_buffer[device_name].append({
                'packet_id': packet_id,
                'data': sync_data,
                'device_run_dir': device_run_dir
            })

            # Flush if buffer is full
            if len(self.sync_metadata_buffer[device_name]) >= self.sync_metadata_batch_size:
                self._flush_sync_metadata_buffer(device_name)

    def _flush_sync_metadata_buffer(self, device_name: str):
        """Flush buffered sync metadata to disk."""
        if device_name not in self.sync_metadata_buffer:
            return

        buffer = self.sync_metadata_buffer[device_name]
        if not buffer:
            return

        # Write all buffered metadata files
        for item in buffer:
            packet_id = item['packet_id']
            sync_data = item['data']
            device_run_dir = item['device_run_dir']
            sync_file = device_run_dir / f"{packet_id}_sync.json"

            try:
                with open(sync_file, 'w') as f:
                    json.dump(sync_data, f, indent=2)
            except Exception as e:
                print(f"Error saving sync metadata: {e}")

        # Clear buffer
        self.sync_metadata_buffer[device_name] = []

    def start(self):
        """Start WiFi collection in background thread."""
        if self.running:
            return

        self.running = True
        self.wrap_packet_handler()

        def run():
            try:
                # Initialize the collector but don't call start_collection_run yet
                # We need to set up the collection state first
                # Suppress verbose per-device ASCII table in multimodal scenario
                try:
                    self.collector.print_samples_table = lambda *a, **k: None
                except Exception:
                    pass
                self.collector.set_up_collection_state(
                    samples_per_device=self.collector.samples_per_device
                )
                self.collector.packet_saver = type('PacketSaver', (), {})()


                # Import and create processor
                from picoscenes_collection.parsers.dataset import Pico_Processor, PacketSaver

                # Use batched packet saver if configured
                use_batched = self.config.get('wifi', {}).get('use_batched_packets', False)
                batch_size = self.config.get('wifi', {}).get('batch_size', 1000)

                if use_batched:
                    from picoscenes_collection.parsers.batched import BatchedPacketSaver
                    self.collector.packet_saver = BatchedPacketSaver(
                        self.collector.save_path,
                        self.collector.current_run,
                        batch_size=batch_size
                    )
                else:
                    # Create standard packet saver
                    self.collector.packet_saver = PacketSaver(
                        self.collector.save_path,
                        self.collector.current_run
                    )

                # Create processor and wire it up correctly
                processor = Pico_Processor()
                # Preserve the chosen saver across set_collector (which sets a default saver)
                chosen_saver = self.collector.packet_saver
                processor.set_collector(self.collector)
                # Restore the chosen saver (batched or standard) on the collector
                try:
                    self.collector.packet_saver = chosen_saver
                    # Keep processor in sync with collector saver
                    processor.packet_saver = chosen_saver
                except Exception:
                    pass
                processor.set_packet_handler(self.collector._packet_handler)
                self.collector.processor = processor

                # Start live stream
                processor.live_stream()

            except Exception as e:
                print(f"WiFi collection error: {e}")
                import traceback
                traceback.print_exc()
                self.running = False

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

        # Give it a moment to initialize
        time.sleep(1)

    def stop(self):
        """Stop WiFi collection."""
        if not self.running:
            return

        self.running = False

        # Stop the processor's running loop
        if hasattr(self.collector, 'processor') and self.collector.processor:
            self.collector.processor.running = False

        # Flush any remaining batched packets
        if hasattr(self.collector, 'packet_saver'):
            if hasattr(self.collector.packet_saver, 'flush_all'):
                self.collector.packet_saver.flush_all()

        # Flush any remaining sync metadata
        with self.sync_metadata_lock:
            for device_name in list(self.sync_metadata_buffer.keys()):
                self._flush_sync_metadata_buffer(device_name)

        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=5)

    def _publish_wifi_visual(self, packet_data: Dict[str, Any], device_name: str,
                             packet_id: int, timestamp_sys: int, timestamp_hw: int) -> None:
        if not self.visualizer or not self.visualizer.enabled:
            return

        csi = packet_data.get('CSI') or {}
        streams, stream_matrix = extract_spatial_streams(csi)
        magnitude = csi.get('Mag')
        mag_arr = None

        if stream_matrix is not None:
            try:
                mag_arr = np.asarray(stream_matrix, dtype=np.float32)
            except Exception:
                mag_arr = None
        elif magnitude is not None:
            try:
                mag_arr = np.asarray(magnitude, dtype=np.float32)
            except Exception:
                mag_arr = None

        if mag_arr is None:
            return

        if mag_arr.ndim == 3:
            # Flatten streams x subcarriers
            mag_arr = mag_arr.reshape(-1, mag_arr.shape[-1])
        if mag_arr.ndim == 1:
            mag_arr = mag_arr[np.newaxis, :]
        if mag_arr.ndim == 2 and mag_arr.shape[0] > self.max_wifi_streams:
            mag_arr = mag_arr[:self.max_wifi_streams, :]
        if mag_arr.shape[-1] > self.max_wifi_subcarriers:
            mag_arr = mag_arr[..., :self.max_wifi_subcarriers]

        avg_magnitude = None
        if mag_arr.size:
            try:
                avg_magnitude = float(np.mean(mag_arr))
            except Exception:
                avg_magnitude = None

        processed_results = []
        if self.preprocessor_pipeline:
            hw_timestamp = timestamp_hw
            if not isinstance(hw_timestamp, (int, float)):
                hw_timestamp = 0
            context = WiFiPacketContext(
                device_name=device_name,
                packet_id=packet_id,
                timestamp_sys=timestamp_sys,
                timestamp_hw=int(hw_timestamp),
            )
            processed_results = self.preprocessor_pipeline.apply(packet_data, context)

        stream_payloads = []
        if streams:
            for stream in streams[: self.max_wifi_streams]:
                if stream.magnitude is None:
                    continue
                try:
                    vec = np.asarray(stream.magnitude, dtype=np.float32)
                except Exception:
                    continue
                avg_val = None
                if vec.size:
                    try:
                        avg_val = float(np.mean(vec))
                    except Exception:
                        avg_val = None
                stream_payloads.append(
                    {
                        'id': stream.label,
                        'title': f"{stream.label} (TX{stream.tx}→RX{stream.rx})",
                        'tx': stream.tx,
                        'rx': stream.rx,
                        'csi_magnitude': vec[: self.max_wifi_subcarriers],
                        'average_magnitude': avg_val,
                    }
                )

        payload = {
            'device': device_name,
            'packet_id': packet_id,
            'timestamp_sys': timestamp_sys,
            'timestamp_hw': timestamp_hw,
            'csi_magnitude': mag_arr
        }

        if avg_magnitude is not None:
            payload['average_magnitude'] = avg_magnitude
        if processed_results:
            payload['preprocessed'] = processed_results
        if stream_payloads:
            payload['streams'] = stream_payloads

        rssi = packet_data.get('RxExtraInfo', {}).get('RSSI')
        if rssi is not None:
            payload['rssi'] = rssi

        # Optional: gesture classification
        if self.gesture_classifier is not None:
            try:
                pred = self.gesture_classifier.predict(mag_arr)
                payload['gesture'] = {'label': pred.label, 'score': float(pred.score)}
            except Exception as exc:
                # Soft-fail – do not break the stream
                payload.setdefault('gesture', {'label': 'error', 'score': 0.0})
                if self.config.get('visualization', {}).get('debug', False):
                    print(f"[Gesture] Inference error: {exc}")

        self.visualizer.publish_wifi(payload)


class CameraCollectorWrapper:
    """
    Wraps realsense_collection.RealSenseRecorder without modification.
    Implements custom recording loop with sync metadata.
    """

    def __init__(self, recorder, session_dir: Path, timestamp_manager: TimestampManager,
                 config: Dict, visualizer: Optional[RealtimeVisualizer] = None):
        self.recorder = recorder
        self.session_dir = session_dir
        self.timestamp_mgr = timestamp_manager
        self.config = config
        self.visualizer = visualizer
        self.running = False
        self.thread = None

        # Create output directory for camera data
        self.camera_dir = session_dir / "camera"
        self.camera_dir.mkdir(parents=True, exist_ok=True)

        # Initialize frame saver
        camera_cfg = config.get('camera', {})
        self.frame_saver = FrameSaver(
            output_dir=str(self.camera_dir),
            save_color=camera_cfg.get('save_color', True),
            save_depth=camera_cfg.get('save_depth', True),
            depth_colormap=camera_cfg.get('depth_colormap', False)
        )

        vis_cfg = config.get('visualization', {})
        self.max_vis_width = int(vis_cfg.get('max_camera_width', 640))
        self.max_vis_height = int(vis_cfg.get('max_camera_height', 480))

    def custom_recording_loop(self):
        """Custom recording loop with sync metadata."""
        import pyrealsense2 as rs

        # Use recorder's pipeline initialization
        if not self.recorder._check_device_connected():
            print("Error: No RealSense device detected!")
            return

        try:
            # Initialize pipeline
            pipeline = rs.pipeline()
            config = rs.config()

            # Configure streams
            if self.recorder.enable_depth:
                config.enable_stream(
                    rs.stream.depth,
                    self.recorder.depth_resolution[0],
                    self.recorder.depth_resolution[1],
                    rs.format.z16,
                    self.recorder.fps
                )

            if self.recorder.enable_color:
                config.enable_stream(
                    rs.stream.color,
                    self.recorder.color_resolution[0],
                    self.recorder.color_resolution[1],
                    rs.format.rgb8,
                    self.recorder.fps
                )

            # Start streaming
            pipeline.start(config)

            print(f"Camera recording started at {self.recorder.fps} FPS")

            # Recording loop
            while self.running:
                try:
                    # Wait for frames
                    frames = pipeline.wait_for_frames()

                    # Get timestamp immediately
                    timestamp_sys = time.monotonic_ns()

                    # Extract frames
                    depth_frame = frames.get_depth_frame() if self.recorder.enable_depth else None
                    color_frame = frames.get_color_frame() if self.recorder.enable_color else None

                    # Get hardware timestamp (from depth or color frame)
                    if depth_frame:
                        timestamp_hw = depth_frame.get_timestamp()
                    elif color_frame:
                        timestamp_hw = color_frame.get_timestamp()
                    else:
                        timestamp_hw = 0.0

                    # Save frames with sync metadata
                    frame_id = self.frame_saver.save_frames(
                        color_frame=color_frame,
                        depth_frame=depth_frame,
                        timestamp_sys=timestamp_sys,
                        timestamp_hw=timestamp_hw
                    )

                    # Record in timestamp manager
                    self.timestamp_mgr.record_timestamp(
                        'camera', frame_id, timestamp_sys, timestamp_hw
                    )

                    if self.visualizer:
                        self._publish_camera_visual(
                            color_frame=color_frame,
                            depth_frame=depth_frame,
                            timestamp_sys=timestamp_sys,
                            timestamp_hw=timestamp_hw,
                            frame_id=frame_id
                        )

                except Exception as e:
                    print(f"Error capturing frame: {e}")
                    continue

            # Stop streaming
            pipeline.stop()

            # Save timestamp CSV
            self.frame_saver.save_timestamp_csv()

            print(f"Camera recording stopped. Saved {self.frame_saver.get_frame_count()} frames")

        except Exception as e:
            print(f"Camera recording error: {e}")
            self.running = False

    def start(self):
        """Start camera recording in background thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self.custom_recording_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop camera recording."""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _publish_camera_visual(self, color_frame, depth_frame,
                               timestamp_sys: int, timestamp_hw: float,
                               frame_id: int) -> None:
        if not self.visualizer or not self.visualizer.enabled:
            return

        color_image = None
        depth_map = None

        if color_frame is not None:
            try:
                color_image = np.asanyarray(color_frame.get_data())
                color_image = self._downsample_for_display(color_image)
            except Exception:
                color_image = None

        if depth_frame is not None:
            try:
                depth_map = np.asanyarray(depth_frame.get_data())
                depth_map = self._downsample_for_display(depth_map)
                depth_map = depth_map.astype(np.float32, copy=False)
                max_val = float(depth_map.max()) if depth_map.size else 0.0
                if max_val > 0:
                    depth_map = depth_map / max_val
            except Exception:
                depth_map = None

        payload = {
            'frame_id': frame_id,
            'timestamp_sys': timestamp_sys,
            'timestamp_hw': timestamp_hw,
            'color_image': color_image,
            'depth_map': depth_map
        }

        self.visualizer.publish_camera(payload)

    def _downsample_for_display(self, image: np.ndarray) -> np.ndarray:
        if self.max_vis_width <= 0 or self.max_vis_height <= 0:
            return image

        height, width = image.shape[:2]
        if width <= self.max_vis_width and height <= self.max_vis_height:
            return image

        scale = min(self.max_vis_width / width, self.max_vis_height / height)
        if scale >= 1.0:
            return image

        step = max(int(math.ceil(1.0 / scale)), 1)
        if image.ndim == 2:
            return image[::step, ::step]
        return image[::step, ::step, ...]


class MultiModalController:
    """
    Main controller for synchronized multi-modal data collection.

    Coordinates WiFi CSI and camera collection without modifying existing code.
    """

    def __init__(self, session_name: str, config_path: Optional[str] = None):
        """
        Initialize multi-modal controller.

        Args:
            session_name: Name for this collection session
            config_path: Path to sync_config.json (optional)
        """
        self.session_name = session_name

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "sync_config.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Create output directory
        output_base = Path(self.config.get('output_dir', 'SyncedData'))
        self.session_dir = output_base / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.timestamp_mgr = TimestampManager()
        self.session = SyncSession(session_name, self.session_dir, self.config)

        # Modality wrappers
        self.wifi_wrapper = None
        self.camera_wrapper = None
        self.visualizer: Optional[RealtimeVisualizer] = None

        self.verbose = self.config.get('verbose', True)

        visualization_cfg = self.config.get('visualization', {})
        if visualization_cfg.get('enable'):
            self.visualizer = RealtimeVisualizer(visualization_cfg)

    def register_modality(self, modality_name: str, collector: Any):
        """
        Register a modality collector.

        Args:
            modality_name: "wifi" or "camera"
            collector: Existing collector instance to wrap
        """
        if modality_name == "wifi":
            self.wifi_wrapper = WiFiCollectorWrapper(
                collector, self.session_dir, self.timestamp_mgr, self.config, self.visualizer
            )
            if self.verbose:
                print(f"✓ Registered WiFi collector")

        elif modality_name == "camera":
            self.camera_wrapper = CameraCollectorWrapper(
                collector, self.session_dir, self.timestamp_mgr, self.config, self.visualizer
            )
            if self.verbose:
                print(f"✓ Registered camera collector")

        else:
            raise ValueError(f"Unknown modality: {modality_name}")

    def start_collection(self, duration: Optional[float] = None):
        """
        Start synchronized collection from all registered modalities.

        Args:
            duration: Collection duration in seconds (None = manual stop)
        """
        # Set reference timestamp
        self.timestamp_mgr.reset_reference()
        self.session.t0 = self.timestamp_mgr.t0
        self.session.start_time = time.time()

        if self.verbose:
            print("=" * 60)
            print(f"Starting synchronized collection: {self.session_name}")
            print(f"Output directory: {self.session_dir}")
            print(f"t0 reference: {self.timestamp_mgr.t0}")
            print("=" * 60)

        if self.visualizer:
            if self.verbose:
                print("Starting real-time visualizer...")
            self.visualizer.start()

        # Start all registered modalities
        if self.wifi_wrapper:
            if self.verbose:
                print("Starting WiFi collection...")
            self.wifi_wrapper.start()
            time.sleep(2)  # Give WiFi time to initialize

        if self.camera_wrapper:
            if self.verbose:
                print("Starting camera recording...")
            self.camera_wrapper.start()

        if self.verbose:
            print(f"\n✓ Collection started")
            if duration:
                print(f"  Duration: {duration} seconds")
            else:
                print(f"  Duration: manual stop (call stop_collection())")

        # Wait for duration if specified
        if duration:
            try:
                time.sleep(duration)
            except KeyboardInterrupt:
                print("\nCollection interrupted by user")
            finally:
                self.stop_collection()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def enable_visualization(self, enable: bool = True, **overrides: Any) -> None:
        """Enable or disable the real-time visualization pipeline."""
        visualization_cfg = dict(self.config.get('visualization', {}))
        visualization_cfg.update(overrides)
        visualization_cfg['enable'] = enable
        self.config['visualization'] = visualization_cfg

        self.visualizer = RealtimeVisualizer(visualization_cfg) if enable else None

        if self.wifi_wrapper:
            self.wifi_wrapper.configure_visualization(self.visualizer, visualization_cfg)
        if self.camera_wrapper:
            self.camera_wrapper.visualizer = self.visualizer
            self.camera_wrapper.max_vis_width = int(visualization_cfg.get('max_camera_width', 640))
            self.camera_wrapper.max_vis_height = int(visualization_cfg.get('max_camera_height', 480))

    def stop_collection(self):
        """Stop all data collection."""
        if self.verbose:
            print("\nStopping collection...")

        # Stop all modalities
        if self.wifi_wrapper:
            self.wifi_wrapper.stop()
            if self.verbose:
                print("  ✓ WiFi collection stopped")

        if self.camera_wrapper:
            self.camera_wrapper.stop()
            if self.verbose:
                print("  ✓ Camera recording stopped")

        if self.visualizer:
            self.visualizer.stop()
            if self.verbose:
                print("  ✓ Real-time visualizer stopped")

        self.session.end_time = time.time()

        # Reorganize batch files if using batched packets
        if self.config.get('wifi', {}).get('use_batched_packets', False):
            if self.verbose:
                print("\n  Reorganizing batch files...")
            self._reorganize_wifi_batches()

        # Save session metadata
        if self.config.get('save_session_metadata', True):
            self._save_session_metadata()

        if self.verbose:
            duration = self.session.end_time - self.session.start_time
            print(f"\n✓ Collection complete ({duration:.1f}s)")
            print(f"  Data saved to: {self.session_dir}")

    def _save_session_metadata(self):
        """Save session metadata to JSON."""
        metadata_file = self.session_dir / "metadata.json"

        # Get session info from timestamp manager
        session_info = self.timestamp_mgr.get_session_info()

        # Combine with session data
        metadata = {
            **self.session.to_dict(),
            **session_info
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"  ✓ Session metadata saved")

    def _reorganize_wifi_batches(self):
        """Reorganize batch files into individual packets."""
        from picoscenes_collection.parsers.batched import BatchedPacketSaver

        try:
            # Initialize reorganizer with the save path
            wifi_dir = self.session_dir / "wifi_csi"
            if not wifi_dir.exists():
                if self.verbose:
                    print("    No WiFi data to reorganize")
                return

            # Create a temporary batched saver to access reorganization method
            reorganizer = BatchedPacketSaver(wifi_dir.parent.parent, "temp")
            reorganizer.save_path = wifi_dir.parent.parent

            # Reorganize all batch files
            reorganizer.reorganize_batches()

            if self.verbose:
                print("    ✓ Batch files reorganized into individual packets")

        except Exception as e:
            print(f"    Warning: Could not reorganize batch files: {e}")
            print(f"    You can reorganize manually later with:")
            print(f"    python picoscenes_collection/tools/reorganize_batches.py {self.session_dir}")

    def build_sync_index(self):
        """Build synchronization index for post-hoc alignment."""
        from multimodal_collection.sync_index_builder import SyncIndexBuilder

        if self.verbose:
            print("\nBuilding synchronization index...")

        builder = SyncIndexBuilder(
            session_dir=self.session_dir,
            timestamp_manager=self.timestamp_mgr,
            config=self.config
        )

        builder.build_index()

        if self.verbose:
            print("  ✓ Sync index built: sync_index.h5")

    def generate_quality_report(self):
        """Generate synchronization quality report."""
        if self.verbose:
            print("\nGenerating quality report...")

        # Get sync statistics
        pairs_wifi_to_camera = self.timestamp_mgr.align_nearest_neighbor(
            'wifi', 'camera',
            max_delta_ns=self.config.get('max_time_delta_ms', 50) * 1_000_000
        )

        stats = self.timestamp_mgr.compute_sync_statistics(pairs_wifi_to_camera)

        # Print report
        print("\n" + "=" * 60)
        print("SYNCHRONIZATION QUALITY REPORT")
        print("=" * 60)
        print(f"Session: {self.session_name}")
        print(f"WiFi packets: {len(self.timestamp_mgr.get_timestamps('wifi'))}")
        print(f"Camera frames: {len(self.timestamp_mgr.get_timestamps('camera'))}")
        print(f"Aligned pairs: {stats['n_pairs']}")
        print(f"\nTiming Statistics:")
        print(f"  Mean delta:   {stats['mean_delta_ms']:.2f} ms")
        print(f"  Median delta: {stats['median_delta_ms']:.2f} ms")
        print(f"  Max delta:    {stats['max_delta_ms']:.2f} ms")
        print(f"  Std dev:      {stats['std_delta_ns']/1_000_000:.2f} ms")
        print("=" * 60)

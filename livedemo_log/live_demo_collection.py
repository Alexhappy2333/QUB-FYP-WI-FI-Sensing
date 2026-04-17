#!/usr/bin/env python3
"""
Dedicated live-demo collection entry point.

This script keeps the normal collection workflow untouched by defining a
demo-only controller and WiFi wrapper in this file. The regular
`basic_sync_collection.py` entry point is not modified.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multimodal_collection.live_demo_csi_inference import (  # noqa: E402
    DEFAULT_CLASS_NAMES,
    LiveDemoSklearnGestureClassifier,
    resolve_default_model_path,
)
from multimodal_collection.sync_controller import (  # noqa: E402
    CameraCollectorWrapper,
    MultiModalController,
    WiFiCollectorWrapper,
)
from picoscenes_collection.preprocessing import WiFiPacketContext  # noqa: E402
from picoscenes_collection.parsers import Dataset_collector  # noqa: E402
from picoscenes_collection.utils.csi import extract_spatial_streams  # noqa: E402
from picoscenes_collection.utils.mac import load_addr_mapper  # noqa: E402
from realsense_collection.src.realsense_recorder import RealSenseRecorder  # noqa: E402


class LiveDemoWiFiCollectorWrapper(WiFiCollectorWrapper):
    """Demo-only WiFi wrapper that runs raw CSI -> sklearn gesture inference."""

    def _init_gesture_classifier(self) -> None:
        self.gesture_classifier = None
        self.live_demo_classifier = None

        demo_cfg = (self.config or {}).get("live_demo", {}) or {}
        if not bool(demo_cfg.get("enable", False)):
            return

        model_path = demo_cfg.get("model_path") or resolve_default_model_path()
        class_names = demo_cfg.get("class_names") or DEFAULT_CLASS_NAMES

        try:
            self.live_demo_classifier = LiveDemoSklearnGestureClassifier(
                model_path=model_path,
                class_names=class_names,
                fs=int(demo_cfg.get("fs", 100)),
                n_subcarriers=int(demo_cfg.get("n_subcarriers", 57)),
                n_streams=int(demo_cfg.get("n_streams", 4)),
                hp_cutoff=float(demo_cfg.get("hp_cutoff", 0.6)),
                lp_cutoff=float(demo_cfg.get("lp_cutoff", 25.0)),
                hp_order=int(demo_cfg.get("hp_order", 2)),
                lp_order=int(demo_cfg.get("lp_order", 3)),
                stft_nperseg=int(demo_cfg.get("stft_nperseg", 16)),
                stft_noverlap=int(demo_cfg.get("stft_noverlap", 14)),
                max_doppler_hz=float(demo_cfg.get("max_doppler_hz", 35.0)),
                t_target=int(demo_cfg.get("t_target", 30)),
                window_packets=int(demo_cfg.get("window_packets", 96)),
                min_packets=int(demo_cfg.get("min_packets", 48)),
                infer_stride_packets=int(demo_cfg.get("infer_stride_packets", 8)),
                score_threshold=float(demo_cfg.get("score_threshold", 0.8)),
                vote_window=int(demo_cfg.get("vote_window", 5)),
                min_votes=int(demo_cfg.get("min_votes", 3)),
                hold_seconds=float(demo_cfg.get("hold_seconds", 0.75)),
                unknown_label=str(demo_cfg.get("unknown_label", "unknown")),
                debug=bool(demo_cfg.get("debug", False)),
                debug_dir=demo_cfg.get("debug_dir"),
                debug_every_inferences=int(demo_cfg.get("debug_every_inferences", 10)),
            )
            print(f"[LiveDemo] Loaded sklearn model from: {model_path}")
        except Exception as exc:
            print(f"[LiveDemo] WARNING: Failed to initialize live demo classifier: {exc}")
            self.live_demo_classifier = None

    def _publish_wifi_visual(
        self,
        packet_data: Dict[str, Any],
        device_name: str,
        packet_id: int,
        timestamp_sys: int,
        timestamp_hw: int,
    ) -> None:
        if not self.visualizer or not self.visualizer.enabled:
            return

        csi = packet_data.get("CSI") or {}
        streams, stream_matrix = extract_spatial_streams(csi)
        magnitude = csi.get("Mag")
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
            mag_arr = mag_arr.reshape(-1, mag_arr.shape[-1])
        if mag_arr.ndim == 1:
            mag_arr = mag_arr[np.newaxis, :]
        if mag_arr.ndim == 2 and mag_arr.shape[0] > self.max_wifi_streams:
            mag_arr = mag_arr[: self.max_wifi_streams, :]
        if mag_arr.shape[-1] > self.max_wifi_subcarriers:
            mag_arr = mag_arr[..., : self.max_wifi_subcarriers]

        avg_magnitude = None
        if mag_arr.size:
            try:
                avg_magnitude = float(np.mean(mag_arr))
            except Exception:
                avg_magnitude = None

        processed_results = []
        if self.preprocessor_pipeline:
            hw_timestamp = timestamp_hw if isinstance(timestamp_hw, (int, float)) else 0
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
                        "id": stream.label,
                        "title": f"{stream.label} (TX{stream.tx}→RX{stream.rx})",
                        "tx": stream.tx,
                        "rx": stream.rx,
                        "average_magnitude": avg_val,
                    }
                )

        compact_mag = np.asarray([[avg_magnitude if avg_magnitude is not None else 0.0]], dtype=np.float32)
        payload = {
            "device": device_name,
            "packet_id": packet_id,
            "timestamp_sys": timestamp_sys,
            "timestamp_hw": timestamp_hw,
            "csi_magnitude": compact_mag,
        }

        if avg_magnitude is not None:
            payload["average_magnitude"] = avg_magnitude
        if processed_results:
            payload["preprocessed"] = processed_results
        if stream_payloads:
            payload["streams"] = stream_payloads

        rssi = packet_data.get("RxExtraInfo", {}).get("RSSI")
        if rssi is not None:
            payload["rssi"] = rssi

        if self.live_demo_classifier is not None:
            try:
                pred = self.live_demo_classifier.predict_packet(packet_data)
                if pred is not None:
                    payload["gesture"] = {"label": pred.label, "score": float(pred.score)}
            except Exception as exc:
                if self.config.get("visualization", {}).get("debug", False):
                    print(f"[LiveDemo] Inference error: {exc}")

        self.visualizer.publish_wifi(payload)


class LiveDemoController(MultiModalController):
    """Demo-only controller that swaps in the live demo WiFi wrapper."""

    def register_modality(self, modality_name: str, collector: Any):
        if modality_name == "wifi":
            self.wifi_wrapper = LiveDemoWiFiCollectorWrapper(
                collector, self.session_dir, self.timestamp_mgr, self.config, self.visualizer
            )
            if self.verbose:
                print("✓ Registered WiFi collector (live demo wrapper)")
            return

        if modality_name == "camera":
            self.camera_wrapper = CameraCollectorWrapper(
                collector, self.session_dir, self.timestamp_mgr, self.config, self.visualizer
            )
            if self.verbose:
                print("✓ Registered camera collector")
            return

        raise ValueError(f"Unknown modality: {modality_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Live demo multi-modal collection")
    parser.add_argument("--duration", type=float, default=10.0, help="Collection duration in seconds")
    parser.add_argument("--session", type=str, default=None, help="Session name")
    parser.add_argument(
        "--samples-per-device",
        type=int,
        default=None,
        help="WiFi samples per device (default: unlimited during session)",
    )
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--config", type=str, default=None, help="Path to sync_config.json")
    parser.add_argument("--visualize", action="store_true", help="Enable real-time visualization")
    parser.add_argument(
        "--visualize-mode",
        choices=["matplotlib", "web"],
        default="web",
        help="Visualization backend for the live demo",
    )
    parser.add_argument("--visualize-host", type=str, default=None, help="Web dashboard host")
    parser.add_argument("--visualize-port", type=int, default=None, help="Web dashboard port")
    parser.add_argument(
        "--visualize-origin",
        action="append",
        dest="visualize_origins",
        default=None,
        help="Allowed Origin header(s) for browser access",
    )
    parser.add_argument("--visualize-debug", action="store_true", help="Enable visualization debug logs")
    parser.add_argument(
        "--visualize-camera-rate",
        type=float,
        default=8.0,
        help="Maximum camera publish rate for the web dashboard",
    )
    parser.add_argument(
        "--visualize-wifi-rate",
        type=float,
        default=8.0,
        help="Maximum WiFi publish rate for the web dashboard",
    )
    parser.add_argument(
        "--visualize-camera-width",
        type=int,
        default=480,
        help="Maximum camera width sent to the web dashboard",
    )
    parser.add_argument(
        "--visualize-camera-height",
        type=int,
        default=270,
        help="Maximum camera height sent to the web dashboard",
    )
    parser.add_argument(
        "--visualize-max-streams",
        type=int,
        default=2,
        help="Maximum WiFi stream cards shown in the web dashboard",
    )
    parser.add_argument(
        "--gesture-model-path",
        type=str,
        default=resolve_default_model_path(),
        help="Path to the sklearn joblib model",
    )
    parser.add_argument(
        "--gesture-class-names",
        type=str,
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names in model index order",
    )
    parser.add_argument(
        "--gesture-window-packets",
        type=int,
        default=96,
        help="Rolling CSI packet window size for live inference",
    )
    parser.add_argument(
        "--gesture-min-packets",
        type=int,
        default=48,
        help="Minimum buffered packets before the first live inference",
    )
    parser.add_argument(
        "--gesture-stride-packets",
        type=int,
        default=8,
        help="Run inference every N packets after the buffer is primed",
    )
    parser.add_argument(
        "--gesture-score-threshold",
        type=float,
        default=0.8,
        help="Minimum confidence before a prediction is allowed to become a gesture label",
    )
    parser.add_argument(
        "--gesture-vote-window",
        type=int,
        default=5,
        help="How many recent predictions to keep for majority voting",
    )
    parser.add_argument(
        "--gesture-min-votes",
        type=int,
        default=3,
        help="Minimum votes required within the vote window before switching labels",
    )
    parser.add_argument(
        "--gesture-hold-seconds",
        type=float,
        default=0.75,
        help="Minimum time to hold the current gesture label before allowing a switch",
    )
    parser.add_argument(
        "--gesture-debug",
        action="store_true",
        help="Print raw/stable gesture predictions and save live Doppler snapshots for debugging",
    )
    parser.add_argument(
        "--gesture-debug-dir",
        type=str,
        default=None,
        help="Directory for saving live Doppler debug outputs",
    )
    parser.add_argument(
        "--gesture-debug-every",
        type=int,
        default=10,
        help="Save/print one debug sample every N inferences",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Skip sync index building and quality report after collection",
    )
    return parser.parse_args()


def _parse_class_names(raw: str):
    parts = [item.strip() for item in (raw or "").split(",")]
    return [item for item in parts if item] or list(DEFAULT_CLASS_NAMES)


def main() -> int:
    args = parse_args()

    if args.session is None:
        args.session = f"live_demo_{time.strftime('%Y%m%d_%H%M%S')}"

    print("=" * 70)
    print("LIVE DEMO MULTI-MODAL COLLECTION")
    print("=" * 70)
    print(f"Session: {args.session}")
    print(f"Duration: {args.duration} seconds")
    print(f"Camera FPS: {args.camera_fps}")
    print(f"Model: {args.gesture_model_path}")
    print(f"Gesture classes: {_parse_class_names(args.gesture_class_names)}")
    print(
        "Gesture smoothing: "
        f"threshold={args.gesture_score_threshold}, "
        f"vote_window={args.gesture_vote_window}, "
        f"min_votes={args.gesture_min_votes}, "
        f"hold={args.gesture_hold_seconds}s"
    )
    if args.gesture_debug:
        print(
            "Gesture debug: "
            f"enabled, dir={args.gesture_debug_dir or 'disabled-save-only-print'}, "
            f"every={args.gesture_debug_every}"
        )
    print("=" * 70)

    print("\n1. Setting up WiFi CSI collector...")
    addr_mapper = load_addr_mapper()
    print(f"   Loaded {len(addr_mapper)} device MAC addresses")

    wifi_collector = Dataset_collector(
        addr_mapper,
        receiver="B210-QUB",
        environment="Sync-Test",
        distance="1m",
        date=Path(time.strftime("%j")),
        samples_per_device=args.samples_per_device,
    )
    print("   ✓ WiFi collector ready")

    print("\n2. Setting up RealSense camera...")
    camera_recorder = RealSenseRecorder(
        output_dir="Video",
        fps=args.camera_fps,
        depth_resolution=(1280, 720),
        color_resolution=(1280, 720),
        enable_depth=True,
        enable_color=True,
    )

    if not camera_recorder._check_device_connected():
        print("   ✗ ERROR: RealSense camera not detected!")
        print("   Please connect your Intel RealSense D455 camera.")
        return 1
    print("   ✓ RealSense camera detected")

    print("\n3. Creating live demo controller...")
    sync_ctrl = LiveDemoController(session_name=args.session, config_path=args.config)

    if args.visualize:
        vis_overrides = {
            "mode": args.visualize_mode,
            "max_fps": args.visualize_camera_rate,
            "max_wifi_rate_hz": args.visualize_wifi_rate,
            "max_camera_width": args.visualize_camera_width,
            "max_camera_height": args.visualize_camera_height,
            "max_wifi_streams": args.visualize_max_streams,
        }
        if args.visualize_host:
            vis_overrides["host"] = args.visualize_host
        if args.visualize_port:
            vis_overrides["port"] = args.visualize_port
        if args.visualize_origins:
            vis_overrides["allowed_origins"] = args.visualize_origins
        if args.visualize_debug:
            vis_overrides["debug"] = True
        sync_ctrl.enable_visualization(True, **vis_overrides)

    sync_ctrl.config["live_demo"] = {
        "enable": True,
        "model_path": args.gesture_model_path,
        "class_names": _parse_class_names(args.gesture_class_names),
        "window_packets": args.gesture_window_packets,
        "min_packets": args.gesture_min_packets,
        "infer_stride_packets": args.gesture_stride_packets,
        "score_threshold": args.gesture_score_threshold,
        "vote_window": args.gesture_vote_window,
        "min_votes": args.gesture_min_votes,
        "hold_seconds": args.gesture_hold_seconds,
        "unknown_label": "unknown",
        "debug": args.gesture_debug,
        "debug_dir": args.gesture_debug_dir,
        "debug_every_inferences": args.gesture_debug_every,
    }
    print("   ✓ Live demo controller initialized")

    print("\n4. Registering modalities...")
    sync_ctrl.register_modality("wifi", wifi_collector)
    sync_ctrl.register_modality("camera", camera_recorder)

    print("\n5. Starting live demo collection...\n")
    try:
        sync_ctrl.start_collection(duration=args.duration)
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user (Ctrl+C)")
        sync_ctrl.stop_collection()
    except Exception as exc:
        print(f"\n\nERROR during collection: {exc}")
        sync_ctrl.stop_collection()
        return 1

    if not args.skip_postprocess:
        print("\n6. Building synchronization index...")
        try:
            sync_ctrl.build_sync_index()
        except Exception as exc:
            print(f"   ✗ Error building sync index: {exc}")
            print("   (Sync index can be built manually later)")

        print("\n7. Generating quality report...")
        try:
            sync_ctrl.generate_quality_report()
        except Exception as exc:
            print(f"   ✗ Error generating report: {exc}")

    print("\n" + "=" * 70)
    print("LIVE DEMO COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Session data saved to: SyncedData/{args.session}/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

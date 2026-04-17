"""Real-time visualization utilities for multimodal collection."""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


class RealtimeVisualizer:
    """Real-time dashboard supporting local and remote rendering modes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = bool(self.config.get('enable', False))

        camera_queue_size = int(self.config.get('camera_queue_size', 5))
        wifi_queue_size = int(self.config.get('wifi_queue_size', 20))

        self.camera_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=camera_queue_size)
        self.wifi_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=wifi_queue_size)

        self.max_camera_fps = float(self.config.get('max_fps', 15) or 0)
        self.max_wifi_rate_hz = float(self.config.get('max_wifi_rate_hz', 30) or 0)

        self._camera_interval = 1.0 / self.max_camera_fps if self.max_camera_fps > 0 else 0.0
        self._wifi_interval = 1.0 / self.max_wifi_rate_hz if self.max_wifi_rate_hz > 0 else 0.0

        self._last_camera_publish = 0.0
        self._last_wifi_publish = 0.0

        self.thread: Optional[threading.Thread] = None
        self.running = False
        self._import_error: Optional[Exception] = None

        # Visualization backend selection
        self.mode = (self.config.get('mode') or 'matplotlib').lower()
        if self.mode not in {'matplotlib', 'web'}:
            print(f"[RealtimeVisualizer] Unknown mode '{self.mode}', falling back to matplotlib")
            self.mode = 'matplotlib'

        self._web_server: Optional[_WebVisualizerServer] = None
        if self.mode == 'web':
            host = self.config.get('host', '0.0.0.0')
            port = int(self.config.get('port', 8765))
            allowed_origins = self.config.get('allowed_origins', ['*'])
            self._web_server = _WebVisualizerServer(host, port, allowed_origins)

        self.max_wifi_streams = int(self.config.get('max_wifi_streams', 4) or 1)
        self.wifi_preprocessors: List[Dict[str, Any]] = []

    def start(self) -> None:
        """Start the visualization loop in a background thread."""
        if not self.enabled:
            print("[RealtimeVisualizer] NOT enabled - visualization disabled")
            return
        if self.running:
            print("[RealtimeVisualizer] Already running")
            return

        print(f"[RealtimeVisualizer] Starting in {self.mode} mode...")
        self.running = True
        if self.mode == 'web':
            if not self._web_server:
                print("[RealtimeVisualizer] Web mode unavailable - disabling")
                self.running = False
                return
            if not self._web_server.start():
                print("[RealtimeVisualizer] Failed to start web server - disabling")
                self.running = False
                return
            target = self._run_web
        else:
            target = self._run_matplotlib

        self.thread = threading.Thread(target=target, name="RealtimeVisualizer", daemon=True)
        self.thread.start()
        print(f"[RealtimeVisualizer] ✓ Started successfully in {self.mode} mode")

    def stop(self) -> None:
        """Stop the visualization loop and close plots."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self._web_server:
            self._web_server.stop()
        self.thread = None

    def set_wifi_preprocessors(self, descriptors: Optional[List[Dict[str, Any]]]) -> None:
        """Register preprocessing plots that should appear beneath the WiFi chart."""
        self.wifi_preprocessors = list(descriptors or [])

    def publish_camera(self, payload: Dict[str, Any]) -> None:
        """Publish latest camera frame to the UI."""
        if not self.enabled or not self.running:
            return

        now = time.time()
        if self._camera_interval and now - self._last_camera_publish < self._camera_interval:
            return
        self._last_camera_publish = now

        self._enqueue_latest(self.camera_queue, payload)

    def publish_wifi(self, payload: Dict[str, Any]) -> None:
        """Publish latest WiFi CSI data to the UI."""
        if not self.enabled or not self.running:
            return

        now = time.time()
        if self._wifi_interval and now - self._last_wifi_publish < self._wifi_interval:
            return
        self._last_wifi_publish = now

        self._enqueue_latest(self.wifi_queue, payload)

    @staticmethod
    def _enqueue_latest(q: "queue.Queue[Dict[str, Any]]", payload: Dict[str, Any]) -> None:
        try:
            q.put_nowait(payload)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(payload)

    # ------------------------------------------------------------------
    # Internal visualization loop
    # ------------------------------------------------------------------
    def _run_matplotlib(self) -> None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as exc:  # pragma: no cover - optional dependency
            self._import_error = exc
            print("[RealtimeVisualizer] matplotlib not available - disabling live view:", exc)
            self.enabled = False
            self.running = False
            return

        from collections import deque

        plt.ion()
        extra_count = len(self.wifi_preprocessors)
        stream_slots = max(1, int(self.max_wifi_streams or 1))
        total_rows = 1 + stream_slots + (extra_count if extra_count > 0 else 0)
        figure_height = 6 + stream_slots * 1.8 + extra_count * 1.4
        height_ratios = [2.8] + [1.6] * stream_slots + [1.4] * extra_count
        fig = plt.figure(figsize=(8, figure_height))
        gs = fig.add_gridspec(total_rows, 1, height_ratios=height_ratios)
        ax_camera = fig.add_subplot(gs[0, 0])

        color_cycle = self.config.get('wifi_stream_colors') or [
            "#4af",
            "#fa4",
            "#8f4",
            "#f66",
            "#ba4af4",
            "#47c0f1",
        ]

        wifi_axes_info: List[Dict[str, Any]] = []
        for idx in range(stream_slots):
            ax = fig.add_subplot(gs[1 + idx, 0])
            ax.set_visible(False)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Average amplitude")
            ax.grid(True, alpha=0.3)
            color = color_cycle[idx % len(color_cycle)]
            (line,) = ax.plot([], [], color=color, linewidth=1.5)
            text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va='top')
            wifi_axes_info.append(
                {
                    "ax": ax,
                    "line": line,
                    "text": text,
                    "label": None,
                    "title": None,
                }
            )

        derived_axes: Dict[str, Dict[str, Any]] = {}
        for idx, descriptor in enumerate(self.wifi_preprocessors):
            ax = fig.add_subplot(gs[1 + stream_slots + idx, 0])
            title = descriptor.get('title') or descriptor.get('id') or f"Algorithm {idx + 1}"
            ax.set_title(f"WiFi {title}")
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("Stream index")
            ax.set_aspect('auto')
            im = ax.imshow(np.zeros((1, 1), dtype=np.float32), aspect='auto', cmap='magma')
            text = ax.text(
                0.02,
                0.95,
                "",
                transform=ax.transAxes,
                va='top',
                ha='left',
                color="#ddd",
            )
            derived_axes[descriptor.get('id', str(idx))] = {
                "ax": ax,
                "im": im,
                "title": title,
                "text": text,
                "expected": descriptor.get('expected_shape'),
            }

        try:  # pragma: no cover - depends on backend
            fig.canvas.manager.set_window_title("Multimodal Live View")  # type: ignore[attr-defined]
        except Exception:
            pass

        ax_camera.set_title("Camera (latest frame)")
        ax_camera.axis('off')

        camera_im = None
        depth_im = None

        history_points = self.config.get('wifi_history_points', 600)
        try:
            history_points = int(history_points)
        except (TypeError, ValueError):
            history_points = 600
        if history_points <= 0:
            history_points = 600

        wifi_histories: Dict[str, Dict[str, deque]] = {}
        stream_axis_map: Dict[str, int] = {}
        wifi_t0_ns: Optional[float] = None

        latest_camera = None
        latest_wifi = None

        def assign_axis(label: str, title: Optional[str]) -> Optional[Dict[str, Any]]:
            if label in stream_axis_map:
                idx = stream_axis_map[label]
                info = wifi_axes_info[idx]
                if title and info["title"] != title:
                    info["ax"].set_title(f"WiFi {title}")
                    info["title"] = title
                return info
            for idx, info in enumerate(wifi_axes_info):
                if info["label"] is None:
                    stream_axis_map[label] = idx
                    info["label"] = label
                    info["title"] = title
                    ax_target = info["ax"]
                    ax_target.set_visible(True)
                    ax_target.set_title(f"WiFi {title or label}")
                    return info
            return None

        while self.running:
            updated = False

            try:
                while True:
                    latest_camera = self.camera_queue.get_nowait()
                    updated = True
            except queue.Empty:
                pass

            try:
                while True:
                    latest_wifi = self.wifi_queue.get_nowait()
                    updated = True
            except queue.Empty:
                pass

            if latest_camera:
                color_frame = latest_camera.get('color_image')
                depth_frame = latest_camera.get('depth_map')
                timestamp = latest_camera.get('timestamp_sys')

                if color_frame is not None:
                    color_arr = self._ensure_array(color_frame)
                    if color_arr is not None:
                        if camera_im is None:
                            camera_im = ax_camera.imshow(color_arr)
                        else:
                            camera_im.set_data(color_arr)
                        ax_camera.set_xlabel(f"sys_ns={timestamp}")

                if depth_frame is not None:
                    depth_arr = self._ensure_array(depth_frame)
                    if depth_arr is not None:
                        if depth_im is None:
                            depth_im = ax_camera.imshow(depth_arr, alpha=0.35, cmap='viridis')
                        else:
                            depth_im.set_data(depth_arr)

            if latest_wifi:
                device = latest_wifi.get('device')
                packet_id = latest_wifi.get('packet_id')
                timestamp = latest_wifi.get('timestamp_sys')
                time_ns = timestamp if isinstance(timestamp, (int, float)) else time.time_ns()
                if wifi_t0_ns is None:
                    wifi_t0_ns = float(time_ns)
                time_seconds = (float(time_ns) - wifi_t0_ns) / 1e9 if wifi_t0_ns else 0.0

                stream_entries = latest_wifi.get('streams')
                if stream_entries:
                    stream_entries = stream_entries[:stream_slots]
                else:
                    stream_entries = []
                    mag = latest_wifi.get('csi_magnitude')
                    mag_arr = self._ensure_array(mag)
                    avg_magnitude = latest_wifi.get('average_magnitude')
                    if mag_arr is not None:
                        if mag_arr.ndim == 1:
                            mag_arr = mag_arr[np.newaxis, :]
                        if mag_arr.ndim >= 2:
                            limit = min(mag_arr.shape[0], stream_slots)
                            for idx in range(limit):
                                vec = mag_arr[idx]
                                avg_val = None
                                if vec.size:
                                    try:
                                        avg_val = float(np.mean(vec))
                                    except Exception:
                                        avg_val = None
                                if avg_val is None and avg_magnitude is not None:
                                    avg_val = float(avg_magnitude)
                                stream_entries.append(
                                    {
                                        'id': f"Stream_{idx + 1}",
                                        'title': f"Stream {idx + 1}",
                                        'csi_magnitude': vec,
                                        'average_magnitude': avg_val,
                                    }
                                )
                    elif avg_magnitude is not None:
                        stream_entries.append(
                            {
                                'id': 'Stream_1',
                                'title': 'Stream 1',
                                'average_magnitude': float(avg_magnitude),
                            }
                        )

                for entry in stream_entries or []:
                    label = str(entry.get('id') or entry.get('title') or 'Stream')
                    title = entry.get('title') or label
                    axis_info = assign_axis(label, title)
                    if not axis_info:
                        continue

                    avg_val = entry.get('average_magnitude')
                    if avg_val is None:
                        vec = self._ensure_array(entry.get('csi_magnitude'), np)
                        if vec is not None and vec.size:
                            try:
                                avg_val = float(np.mean(vec))
                            except Exception:
                                avg_val = None
                    if avg_val is None:
                        continue

                    history = wifi_histories.setdefault(
                        label,
                        {
                            "times": deque(maxlen=history_points),
                            "values": deque(maxlen=history_points),
                        },
                    )
                    history["times"].append(time_seconds)
                    history["values"].append(avg_val)

                    times_list = list(history["times"])
                    values_list = list(history["values"])
                    axis = axis_info["ax"]
                    line = axis_info["line"]
                    text = axis_info["text"]

                    line.set_data(times_list, values_list)

                    if times_list:
                        xmin = times_list[0]
                        xmax = times_list[-1]
                        if xmax <= xmin:
                            xmax = xmin + 1.0
                        axis.set_xlim(xmin, xmax)

                    if values_list:
                        y_min = min(values_list)
                        y_max = max(values_list)
                        if y_max - y_min < 1e-9:
                            delta = max(abs(y_min) * 0.05, 1e-3)
                            y_min -= delta
                            y_max += delta
                        else:
                            pad = 0.05 * (y_max - y_min)
                            y_min -= pad
                            y_max += pad
                        axis.set_ylim(y_min, y_max)

                    try:
                        avg_display = f"{float(avg_val):.3f}"
                    except Exception:
                        avg_display = "n/a"
                    text.set_text(
                        f"device={device}\npacket={packet_id}\nstream={label}\nsys_ns={timestamp}\navg={avg_display}"
                    )

                preprocessed = latest_wifi.get('preprocessed') or []
                if preprocessed and derived_axes:
                    for entry in preprocessed:
                        algo_id = entry.get('id') or entry.get('title')
                        if not algo_id:
                            continue
                        target = derived_axes.get(algo_id)
                        if not target:
                            continue
                        matrix = self._ensure_array(entry.get('csi_magnitude'), np)
                        if matrix is None:
                            continue
                        try:
                            matrix = np.asarray(matrix, dtype=np.float32)
                        except Exception:
                            continue
                        if matrix.ndim == 1:
                            matrix = matrix[np.newaxis, :]
                        if matrix.ndim != 2 or matrix.size == 0:
                            continue

                        target['im'].set_data(matrix)
                        target['ax'].set_xlim(-0.5, matrix.shape[1] - 0.5)
                        target['ax'].set_ylim(-0.5, matrix.shape[0] - 0.5)

                        try:
                            vmin = float(np.nanmin(matrix))
                            vmax = float(np.nanmax(matrix))
                        except Exception:
                            vmin, vmax = 0.0, 1.0

                        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-9:
                            center = float(np.nanmean(matrix)) if matrix.size else 0.0
                            span = max(abs(center) * 0.05, 1e-3)
                            vmin, vmax = center - span, center + span
                        target['im'].set_clim(vmin, vmax)
                        title = entry.get('title') or target['title']
                        target['ax'].set_title(f"WiFi {title}")
                        avg_val = entry.get('average_magnitude')
                        if avg_val is None:
                            try:
                                avg_val = float(np.nanmean(matrix)) if matrix.size else None
                            except Exception:
                                avg_val = None
                        shape = entry.get('shape')
                        if not shape:
                            shape = matrix.shape
                        info_lines = []
                        if shape:
                            info_lines.append(f"shape={shape[0]}x{shape[1]}")
                        expected = target.get('expected') or entry.get('expected_shape')
                        if expected:
                            info_lines.append(f"expected {expected}")
                        if avg_val is not None and np.isfinite(avg_val):
                            info_lines.append(f"avg={avg_val:.3f}")
                        target['text'].set_text("\n".join(info_lines))

            if updated:
                fig.canvas.draw_idle()
            plt.pause(0.05)

        plt.ioff()
        plt.close(fig)

    def _run_web(self) -> None:
        import numpy as np

        if not self._web_server:
            return

        last_keepalive = 0.0

        while self.running:
            latest_camera = None
            latest_wifi = None

            try:
                while True:
                    latest_camera = self.camera_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                while True:
                    latest_wifi = self.wifi_queue.get_nowait()
            except queue.Empty:
                pass

            if latest_camera:
                event = self._format_camera_event(latest_camera, np)
                if event:
                    self._web_server.broadcast(event)
                elif self.config.get('debug', False):
                    print("[Web] WARNING: Camera event formatting failed")

            if latest_wifi:
                event = self._format_wifi_event(latest_wifi, np)
                if event:
                    self._web_server.broadcast(event)
                elif self.config.get('debug', False):
                    print("[Web] WARNING: WiFi event formatting failed")

            now = time.time()
            if now - last_keepalive > 10.0:
                self._web_server.broadcast({'type': 'keepalive', 'timestamp': now})
                last_keepalive = now

            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Payload formatting helpers for web mode
    # ------------------------------------------------------------------
    def _format_camera_event(self, payload: Dict[str, Any], np_module) -> Optional[Dict[str, Any]]:
        event: Dict[str, Any] = {
            'type': 'camera',
            'frame_id': payload.get('frame_id'),
            'timestamp_sys': payload.get('timestamp_sys'),
            'timestamp_hw': payload.get('timestamp_hw'),
        }

        color_meta = self._encode_array(payload.get('color_image'), np_module, target_dtype='uint8')
        depth_meta = self._encode_array(payload.get('depth_map'), np_module, target_dtype='float32')

        if color_meta:
            event['color'] = color_meta
        if depth_meta:
            event['depth'] = depth_meta

        return event if ('color' in event or 'depth' in event) else None

    def _format_wifi_event(self, payload: Dict[str, Any], np_module) -> Optional[Dict[str, Any]]:
        csi_mag = payload.get('csi_magnitude')
        if csi_mag is None:
            if self.config.get('debug', False):
                print(f"[Web] WARNING: No csi_magnitude in payload. Keys: {list(payload.keys())}")
            return None

        mag_meta = self._encode_array(csi_mag, np_module, target_dtype='float32')
        if not mag_meta:
            if self.config.get('debug', False):
                print(f"[Web] WARNING: Failed to encode csi_magnitude array")
            return None

        avg_value = payload.get('average_magnitude')
        if avg_value is None:
            arr = self._ensure_array(csi_mag, np_module)
            if arr is not None:
                try:
                    avg_value = float(np_module.mean(arr))
                except Exception:
                    avg_value = None

        event: Dict[str, Any] = {
            'type': 'wifi',
            'device': payload.get('device'),
            'packet_id': payload.get('packet_id'),
            'timestamp_sys': payload.get('timestamp_sys'),
            'timestamp_hw': payload.get('timestamp_hw'),
            'magnitude': mag_meta,
        }

        if avg_value is not None:
            try:
                event['avg_magnitude'] = float(avg_value)
            except (TypeError, ValueError):
                pass

        if 'rssi' in payload:
            event['rssi'] = payload['rssi']

        stream_entries = payload.get('streams') or []
        encoded_streams = []
        for idx, stream in enumerate(stream_entries):
            if idx >= self.max_wifi_streams:
                break
            magnitude = stream.get('csi_magnitude')
            magnitude_meta = None
            if magnitude is not None:
                magnitude_meta = self._encode_array(magnitude, np_module, target_dtype='float32')
            entry: Dict[str, Any] = {
                'id': stream.get('id') or f'stream_{idx + 1}',
                'title': stream.get('title'),
                'tx': stream.get('tx'),
                'rx': stream.get('rx'),
            }
            if magnitude_meta:
                entry['magnitude'] = magnitude_meta
            avg_stream = stream.get('average_magnitude')
            if avg_stream is not None:
                try:
                    entry['avg_magnitude'] = float(avg_stream)
                except (TypeError, ValueError):
                    pass
            encoded_streams.append(entry)
        if encoded_streams:
            event['streams'] = encoded_streams

        preprocessed_payloads = payload.get('preprocessed') or []
        encoded_preprocessed = []
        for idx, preproc in enumerate(preprocessed_payloads):
            magnitude = preproc.get('csi_magnitude')
            mag_meta = self._encode_array(magnitude, np_module, target_dtype='float32')
            if not mag_meta:
                continue
            encoded_entry: Dict[str, Any] = {
                'id': preproc.get('id') or f'preproc_{idx}',
                'title': preproc.get('title'),
                'magnitude': mag_meta,
            }
            avg_override = preproc.get('average_magnitude', preproc.get('avg_magnitude'))
            if avg_override is not None:
                try:
                    encoded_entry['avg_magnitude'] = float(avg_override)
                except (TypeError, ValueError):
                    pass
            expected_shape = preproc.get('expected_shape')
            if expected_shape:
                encoded_entry['expected_shape'] = expected_shape
            shape = preproc.get('shape')
            if shape:
                encoded_entry['shape'] = list(shape)
            encoded_preprocessed.append(encoded_entry)
        if encoded_preprocessed:
            event['preprocessed'] = encoded_preprocessed

        # Optional gesture classification result passthrough
        gesture = payload.get('gesture')
        if isinstance(gesture, dict):
            label = gesture.get('label')
            score = gesture.get('score')
            if isinstance(label, str):
                try:
                    score_val = float(score) if isinstance(score, (int, float)) else None
                except Exception:
                    score_val = None
                event['gesture'] = {'label': label}
                if score_val is not None:
                    event['gesture']['score'] = score_val

        if self.config.get('debug', False):
            print(f"[Web] WiFi event formatted: device={event['device']}, packet={event['packet_id']}, shape={mag_meta['shape']}")

        return event

    def _encode_array(self, value: Any, np_module, target_dtype: Optional[str] = None) -> Optional[Dict[str, Any]]:
        arr = self._ensure_array(value, np_module)
        if arr is None:
            return None

        if target_dtype:
            try:
                arr = arr.astype(target_dtype, copy=False)
            except Exception:
                return None

        try:
            arr = np_module.ascontiguousarray(arr)
        except Exception:
            return None
        data_b64 = base64.b64encode(arr.tobytes()).decode('ascii')

        return {
            'dtype': str(arr.dtype),
            'shape': list(arr.shape),
            'data': data_b64,
        }

    @staticmethod
    def _ensure_array(value: Any, np_module=None):
        if value is None:
            return None

        np = np_module
        if np is None:
            try:
                import numpy as np  # type: ignore
            except Exception:
                return None
        else:
            np = np_module
        if isinstance(value, np.ndarray):
            return value

        try:
            return np.asarray(value)
        except Exception:
            return None


class _WebVisualizerServer:
    """Minimal HTTP + SSE server that streams visualization events."""

    def __init__(self, host: str, port: int, allowed_origins):
        self.host = host
        self.port = port
        if isinstance(allowed_origins, str):
            allowed_origins = [allowed_origins]
        self.allowed_origins = list(allowed_origins or ['*'])
        self._httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.clients: "set[queue.Queue[tuple[str, str]]]" = set()
        self._clients_lock = threading.Lock()
        self._latest_events: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def start(self) -> bool:
        if self.running:
            return True

        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        except OSError as exc:
            print(f"[RealtimeVisualizer] Unable to bind web server on {self.host}:{self.port} - {exc}")
            return False

        self._httpd.timeout = 1.0
        self._httpd.visualizer_server = self  # type: ignore[attr-defined]
        self.running = True

        def serve():
            assert self._httpd is not None
            while self.running:
                self._httpd.handle_request()

        self.thread = threading.Thread(target=serve, name="RealtimeVisualizerHTTP", daemon=True)
        self.thread.start()
        print(f"[RealtimeVisualizer] Web dashboard available at http://{self.host}:{self.port}")
        return True

    def stop(self) -> None:
        self.running = False
        if self._httpd:
            try:
                self._httpd.server_close()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self._httpd = None

    # ------------------------------------------------------------------
    # Event management
    # ------------------------------------------------------------------
    def broadcast(self, event: Dict[str, Any]) -> None:
        if not self.running:
            print(f"[WebServer] DEBUG: Broadcast skipped - server not running")
            return

        event_type = event.get('type')
        if isinstance(event_type, str):
            self._latest_events[event_type] = event
        else:
            event_type = 'message'

        payload = json.dumps(event)

        with self._clients_lock:
            num_clients = len(self.clients)
            if num_clients == 0:
                # Only print this occasionally to avoid spam
                import random
                if random.random() < 0.01:  # 1% of the time
                    print(f"[WebServer] WARNING: No clients connected to receive events!")
            else:
                # DEBUG: Print first few broadcasts
                import random
                if random.random() < 0.05:  # 5% of the time
                    print(f"[WebServer] DEBUG: Broadcasting {event_type} to {num_clients} clients")
            dead_clients = []
            for client in self.clients:
                try:
                    client.put_nowait((event_type, payload))
                except queue.Full:
                    try:
                        client.get_nowait()
                        client.put_nowait((event_type, payload))
                    except queue.Empty:
                        pass
                except Exception:
                    dead_clients.append(client)
            for client in dead_clients:
                self.clients.discard(client)
                print(f"[WebServer] Client disconnected (remaining clients: {len(self.clients)})")

    def _register_client(self) -> "queue.Queue[tuple[str, str]]":
        q: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=16)
        with self._clients_lock:
            self.clients.add(q)
            print(f"[WebServer] Client connected (total clients: {len(self.clients)})")
        for event in self._latest_events.values():
            try:
                event_type = event.get('type', 'message')
                q.put_nowait((str(event_type), json.dumps(event)))
            except queue.Full:
                break
        return q

    def _unregister_client(self, client: "queue.Queue[tuple[str, str]]") -> None:
        with self._clients_lock:
            self.clients.discard(client)

    # ------------------------------------------------------------------
    # HTTP handler factory
    # ------------------------------------------------------------------
    def _make_handler(self):
        allowed_origins = self.allowed_origins
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
                print(f"[WebServer] GET {self.path} from {self.client_address[0]}")
                if self.path in ('/', '/index.html'):
                    self._send_html()
                elif self.path == '/events':
                    print(f"[WebServer] Starting SSE stream for {self.client_address[0]}")
                    self._stream_events()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - match signature
                # Silence default logging to keep console clean
                return

            # --------------------------------------------------
            # Response helpers
            # --------------------------------------------------
            def _send_headers(self, status: HTTPStatus, content_type: str) -> None:
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Cache-Control', 'no-cache')
                origin = self.headers.get('Origin')
                if '*' in allowed_origins:
                    self.send_header('Access-Control-Allow-Origin', '*')
                elif origin and origin in allowed_origins:
                    self.send_header('Access-Control-Allow-Origin', origin)
                self.end_headers()

            def _send_html(self) -> None:
                self._send_headers(HTTPStatus.OK, 'text/html; charset=utf-8')
                self.wfile.write(_VIEWER_HTML.encode('utf-8'))

            def _stream_events(self) -> None:
                self._send_headers(HTTPStatus.OK, 'text/event-stream')
                client_queue = server._register_client()
                try:
                    while server.running:
                        try:
                            event_type, payload = client_queue.get(timeout=1.0)
                        except queue.Empty:
                            keepalive = json.dumps({'type': 'keepalive', 'timestamp': time.time()})
                            event_type, payload = 'keepalive', keepalive
                        message = f"event: {event_type}\ndata: {payload}\n\n"
                        self.wfile.write(message.encode('utf-8'))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    server._unregister_client(client_queue)

        return Handler


_VIEWER_HTML = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Multimodal Live View</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #111; color: #f3f3f3; }
    header { padding: 1rem 1.5rem; background: #1e1e1e; box-shadow: 0 2px 4px rgba(0,0,0,0.4); }
    main { display: flex; flex-wrap: wrap; padding: 1rem; gap: 1.5rem; justify-content: center; }
    section { background: #1a1a1a; border-radius: 8px; padding: 1rem; box-shadow: 0 4px 8px rgba(0,0,0,0.25); }
    canvas { background: #000; border-radius: 4px; }
    #camera canvas { width: 640px; max-width: 90vw; }
    #wifi canvas { width: 640px; max-width: 90vw; height: 256px; }
    .meta { margin-top: 0.5rem; font-size: 0.9rem; color: #bbb; }
    .badge { display: inline-block; padding: 0.25rem 0.5rem; margin-right: 0.5rem; border-radius: 999px; background: #2b7; color: #041; font-weight: 600; }
    #gestureDisplay {
      margin-top: 0.75rem;
      padding: 0.85rem 1rem;
      border-radius: 10px;
      background: linear-gradient(135deg, #1f3b2e, #14251d);
      border: 1px solid rgba(120, 255, 190, 0.18);
      font-size: clamp(1.6rem, 4vw, 2.8rem);
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #cffff1;
      text-align: center;
      min-height: 1.4em;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }
    #gestureDisplay.idle {
      background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
      border-color: rgba(255,255,255,0.08);
      color: #cfcfcf;
    }
    #wifiDerived { margin-top: 1rem; display: flex; flex-direction: column; gap: 1rem; }
    #wifiDerived .derived { background: #141414; border-radius: 6px; padding: 0.75rem; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04); }
    #wifiDerived .derived h3 { margin: 0 0 0.5rem 0; font-size: 1.05rem; }
    #wifiDerived .derived canvas { width: 640px; max-width: 90vw; height: 160px; border-radius: 4px; display: block; }
    #wifiStreams { display: flex; flex-direction: column; gap: 1rem; flex: 1 1 640px; }
    #wifiStreamsContainer { display: flex; flex-direction: column; gap: 1rem; }
    .stream-card { background: #141414; border-radius: 6px; padding: 0.75rem; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04); }
    .stream-card h3 { margin: 0 0 0.5rem 0; font-size: 1.05rem; }
    .stream-card canvas { width: 640px; max-width: 90vw; height: 160px; border-radius: 4px; display: block; }
  </style>
</head>
<body>
  <header>
    <h1>Multimodal Live View</h1>
    <p id=\"status\">Connecting…</p>
  </header>
  <main>
    <section id=\"camera\">
      <h2>Camera</h2>
      <canvas id=\"cameraCanvas\" width=640 height=480></canvas>
      <canvas id=\"depthCanvas\" width=640 height=480 style=\"display:none;margin-top:0.5rem;\"></canvas>
      <div class=\"meta\" id=\"cameraMeta\"></div>
    </section>
    <section id=\"wifi\">
      <h2>WiFi Average CSI Magnitude</h2>
      <canvas id=\"wifiCanvas\" width=640 height=256></canvas>
      <div id=\"gestureDisplay\" class=\"idle\">WAITING</div>
      <div class=\"meta\" id=\"wifiMeta\"></div>
      <div id=\"wifiDerived\"></div>
    </section>
    <section id="wifiStreams">
      <h2>WiFi Spatial Streams</h2>
      <div id="wifiStreamsContainer"></div>
    </section>
  </main>
  <script>
    const cameraCanvas = document.getElementById('cameraCanvas');
    const depthCanvas = document.getElementById('depthCanvas');
    const wifiCanvas = document.getElementById('wifiCanvas');
    const cameraCtx = cameraCanvas.getContext('2d');
    const depthCtx = depthCanvas.getContext('2d');
    const wifiCtx = wifiCanvas.getContext('2d');
    const cameraMeta = document.getElementById('cameraMeta');
    const wifiMeta = document.getElementById('wifiMeta');
    const gestureDisplay = document.getElementById('gestureDisplay');
    const wifiDerived = document.getElementById('wifiDerived');
    const wifiStreamsContainer = document.getElementById('wifiStreamsContainer');
    const derivedCharts = new Map();
    const streamCharts = new Map();
    const statusEl = document.getElementById('status');
    const DERIVED_HISTORY_LIMIT = 360;
    const DERIVED_LINE_THRESHOLD = 8;
    const DERIVED_COLORS = ['#4af', '#fa4', '#8af', '#4f8', '#ff4af2', '#f55', '#4ff2d6', '#ddd'];

    const STREAM_HISTORY_LIMIT = 600;

    const WIFI_MAX_POINTS = 600;
    const wifiHistory = [];
    let wifiT0 = null;
    const dpr = window.devicePixelRatio || 1;

    function pushWifiPoint(timestampNs, avgMagnitude) {
      if (!Number.isFinite(avgMagnitude)) {
        return false;
      }
      let ts = Number(timestampNs);
      if (!Number.isFinite(ts)) {
        ts = Date.now() * 1e6;
      }
      if (wifiT0 === null) {
        wifiT0 = ts;
      }
      const tSeconds = (ts - wifiT0) / 1e9;
      wifiHistory.push({ t: tSeconds, v: avgMagnitude });
      if (wifiHistory.length > WIFI_MAX_POINTS) {
        wifiHistory.splice(0, wifiHistory.length - WIFI_MAX_POINTS);
      }
      return true;
    }

    function renderWifiChart() {
      const displayWidth = wifiCanvas.clientWidth || wifiCanvas.width;
      const displayHeight = wifiCanvas.clientHeight || wifiCanvas.height;
      const width = Math.max(1, Math.floor(displayWidth * dpr));
      const height = Math.max(1, Math.floor(displayHeight * dpr));
      if (wifiCanvas.width !== width || wifiCanvas.height !== height) {
        wifiCanvas.width = width;
        wifiCanvas.height = height;
      }

      wifiCtx.fillStyle = '#000';
      wifiCtx.fillRect(0, 0, width, height);

      if (!wifiHistory.length) {
        wifiCtx.fillStyle = '#666';
        wifiCtx.font = `${14 * dpr}px system-ui`;
        wifiCtx.textAlign = 'center';
        wifiCtx.textBaseline = 'middle';
        wifiCtx.fillText('Waiting for WiFi packets…', width / 2, height / 2);
        return;
      }

      const paddingLeft = 55 * dpr;
      const paddingRight = 25 * dpr;
      const paddingTop = 20 * dpr;
      const paddingBottom = 45 * dpr;
      const plotWidth = Math.max(1, width - paddingLeft - paddingRight);
      const plotHeight = Math.max(1, height - paddingTop - paddingBottom);

      const times = wifiHistory.map(p => p.t);
      const values = wifiHistory.map(p => p.v);
      const tMin = times[0];
      const tMax = times[times.length - 1];
      const timeSpan = Math.max(tMax - tMin, 1e-6);
      const tPad = timeSpan > 1e-6 ? timeSpan * 0.02 : 0.05;
      const tMinPlot = tMin - tPad;
      const tMaxPlot = tMax + tPad;
      let vMin = Math.min(...values);
      let vMax = Math.max(...values);
      let valueSpan = vMax - vMin;
      if (!Number.isFinite(valueSpan) || valueSpan < 1e-9) {
        const pad = Math.max(Math.abs(vMax) * 0.05, 1e-3);
        vMin -= pad;
        vMax += pad;
        valueSpan = vMax - vMin;
      } else {
        const pad = 0.05 * valueSpan;
        vMin -= pad;
        vMax += pad;
        valueSpan = vMax - vMin;
      }

      wifiCtx.strokeStyle = '#333';
      wifiCtx.lineWidth = 1 * dpr;
      wifiCtx.beginPath();
      wifiCtx.moveTo(paddingLeft, paddingTop);
      wifiCtx.lineTo(paddingLeft, paddingTop + plotHeight);
      wifiCtx.lineTo(paddingLeft + plotWidth, paddingTop + plotHeight);
      wifiCtx.stroke();

      wifiCtx.strokeStyle = '#222';
      wifiCtx.lineWidth = 1 * dpr;
      const gridLines = 4;
      for (let i = 1; i < gridLines; i++) {
        const y = paddingTop + (i / gridLines) * plotHeight;
        wifiCtx.beginPath();
        wifiCtx.moveTo(paddingLeft, y);
        wifiCtx.lineTo(paddingLeft + plotWidth, y);
        wifiCtx.stroke();
      }

      wifiCtx.strokeStyle = '#4af';
      wifiCtx.lineWidth = 2 * dpr;
      wifiCtx.beginPath();
      for (let i = 0; i < wifiHistory.length; i++) {
        const x = paddingLeft + ((times[i] - tMinPlot) / (tMaxPlot - tMinPlot)) * plotWidth;
        const norm = (values[i] - vMin) / valueSpan;
        const y = paddingTop + (1 - norm) * plotHeight;
        if (i === 0) {
          wifiCtx.moveTo(x, y);
        } else {
          wifiCtx.lineTo(x, y);
        }
      }
      wifiCtx.stroke();

      const latest = wifiHistory[wifiHistory.length - 1];
      wifiCtx.fillStyle = '#bbb';
      wifiCtx.font = `${12 * dpr}px system-ui`;
      wifiCtx.textAlign = 'left';
      wifiCtx.textBaseline = 'bottom';
      wifiCtx.fillText(`avg = ${latest.v.toFixed(3)}`, paddingLeft, paddingTop - 6 * dpr);

      const axisY = paddingTop + plotHeight;
      wifiCtx.textAlign = 'center';
      wifiCtx.textBaseline = 'top';
      wifiCtx.fillText(`${tMin.toFixed(1)}s`, paddingLeft, axisY + 8 * dpr);
      wifiCtx.fillText(`${tMax.toFixed(1)}s`, paddingLeft + plotWidth, axisY + 8 * dpr);
      wifiCtx.fillText('Time (s)', paddingLeft + plotWidth / 2, axisY + 24 * dpr);

      wifiCtx.save();
      wifiCtx.translate(18 * dpr, paddingTop + plotHeight / 2);
      wifiCtx.rotate(-Math.PI / 2);
      wifiCtx.textAlign = 'center';
      wifiCtx.textBaseline = 'top';
      wifiCtx.fillText('Average amplitude', 0, 0);
      wifiCtx.restore();
    }

    window.addEventListener('resize', () => renderWifiChart());
    renderWifiChart();

    function updateStatus(text, ok=true) {
      statusEl.textContent = text;
      statusEl.style.color = ok ? '#5f5' : '#f55';
    }

    function decodeBase64(str) {
      const binary = atob(str);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      return bytes.buffer;
    }

    function typedArrayFor(meta) {
      const buffer = decodeBase64(meta.data);
      const dtype = meta.dtype.toLowerCase();
      if (dtype.includes('uint8')) return new Uint8Array(buffer);
      if (dtype.includes('int8')) return new Int8Array(buffer);
      if (dtype.includes('uint16')) return new Uint16Array(buffer);
      if (dtype.includes('int16')) return new Int16Array(buffer);
      if (dtype.includes('uint32')) return new Uint32Array(buffer);
      if (dtype.includes('int32')) return new Int32Array(buffer);
      if (dtype.includes('float64')) return new Float64Array(buffer);
      return new Float32Array(buffer);
    }

    function ensureDerivedChart(id, title, expectedShape) {
      const key = id || title || `algo-${derivedCharts.size + 1}`;
      let chart = derivedCharts.get(key);
      if (!chart) {
        const wrapper = document.createElement('div');
        wrapper.className = 'derived';
        wrapper.dataset.algoId = key;

        const heading = document.createElement('h3');
        heading.textContent = title || key;
        wrapper.appendChild(heading);

        const canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 80;
        canvas.style.width = '640px';
        canvas.style.maxWidth = '90vw';
        canvas.style.height = '160px';
        const ctx = canvas.getContext('2d');
        const meta = document.createElement('div');
        meta.className = 'meta';

        wrapper.appendChild(canvas);
        wrapper.appendChild(meta);
        wifiDerived.appendChild(wrapper);

        chart = {
          key,
          wrapper,
          heading,
          canvas,
          ctx,
          meta,
          history: [],
          timestamps: [],
          historyLimit: DERIVED_HISTORY_LIMIT,
          mode: 'heatmap',
          rows: null,
          cols: null,
        };
        derivedCharts.set(key, chart);
      } else if (title && chart.heading.textContent !== title) {
        chart.heading.textContent = title;
      }

      if (expectedShape) {
        chart.wrapper.dataset.expectedShape = expectedShape;
      }
      return chart;
    }

    function heatColor(norm) {
      const value = Math.min(1, Math.max(0, norm));
      const r = Math.floor(255 * value);
      const g = Math.floor(255 * Math.sqrt(value));
      const b = Math.floor(255 * (1 - value));
      return [r, g, b];
    }

    function renderDerivedLineChart(chart) {
      const history = chart.history || [];
      if (!history.length) {
        const ctx = chart.ctx;
        const canvas = chart.canvas;
        const displayWidth = canvas.clientWidth || canvas.width || 640;
        const displayHeight = canvas.clientHeight || canvas.height || 160;
        const width = Math.max(1, Math.floor(displayWidth * dpr));
        const height = Math.max(1, Math.floor(displayHeight * dpr));
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#bbb';
        ctx.font = `${14 * dpr}px system-ui`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Waiting for data…', width / 2, height / 2);
        return;
      }

      const canvas = chart.canvas;
      const ctx = chart.ctx;
      const displayWidth = canvas.clientWidth || canvas.width || 640;
      const displayHeight = canvas.clientHeight || canvas.height || 160;
      const width = Math.max(1, Math.floor(displayWidth * dpr));
      const height = Math.max(1, Math.floor(displayHeight * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.clearRect(0, 0, width, height);
      const paddingLeft = 55 * dpr;
      const paddingRight = 25 * dpr;
      const paddingTop = 22 * dpr;
      const paddingBottom = 42 * dpr;
      const plotWidth = Math.max(1, width - paddingLeft - paddingRight);
      const plotHeight = Math.max(1, height - paddingTop - paddingBottom);

      const firstSample = history[0];
      const seriesCount = Math.max(1, chart.cols || (firstSample ? firstSample.length : 1));
      const firstTs = chart.timestamps[0] ?? Date.now() * 1e6;
      const timesSeconds = chart.timestamps.map((ts) => (ts - firstTs) / 1e9);

      let minVal = Infinity;
      let maxVal = -Infinity;
      for (const sample of history) {
        for (let i = 0; i < seriesCount; i++) {
          const val = sample[i];
          if (!Number.isFinite(val)) {
            continue;
          }
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        }
      }
      if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
        minVal = 0;
        maxVal = 1;
      }
      if (Math.abs(maxVal - minVal) < 1e-9) {
        const pad = Math.max(Math.abs(maxVal) * 0.05, 1e-3);
        minVal -= pad;
        maxVal += pad;
      } else {
        const pad = 0.05 * (maxVal - minVal);
        minVal -= pad;
        maxVal += pad;
      }
      const valueSpan = maxVal - minVal || 1;

      const tMin = timesSeconds[0] ?? 0;
      const tMax = timesSeconds[timesSeconds.length - 1] ?? tMin + 1e-3;
      const tSpan = Math.max(tMax - tMin, 1e-6);
      const tPad = tSpan > 1e-6 ? tSpan * 0.02 : 0.05;
      const tMinPlot = tMin - tPad;
      const tMaxPlot = tMax + tPad;

      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      ctx.moveTo(paddingLeft, paddingTop);
      ctx.lineTo(paddingLeft, paddingTop + plotHeight);
      ctx.lineTo(paddingLeft + plotWidth, paddingTop + plotHeight);
      ctx.stroke();

      ctx.strokeStyle = '#222';
      ctx.lineWidth = 1 * dpr;
      const gridLines = 4;
      for (let i = 1; i < gridLines; i++) {
        const y = paddingTop + (i / gridLines) * plotHeight;
        ctx.beginPath();
        ctx.moveTo(paddingLeft, y);
        ctx.lineTo(paddingLeft + plotWidth, y);
        ctx.stroke();
      }

      for (let series = 0; series < seriesCount; series++) {
        const color = DERIVED_COLORS[series % DERIVED_COLORS.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8 * dpr;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < history.length; i++) {
          const val = history[i][series];
          if (!Number.isFinite(val)) {
            continue;
          }
          const t = timesSeconds[i];
          const x =
            paddingLeft +
            ((t - tMinPlot) / Math.max(tMaxPlot - tMinPlot, 1e-9)) * plotWidth;
          const norm = (val - minVal) / valueSpan;
          const y = paddingTop + (1 - norm) * plotHeight;
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        if (started) {
          ctx.stroke();
        }
      }

      ctx.fillStyle = '#bbb';
      ctx.font = `${12 * dpr}px system-ui`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText('Time (s)', paddingLeft + plotWidth / 2, paddingTop + plotHeight + 24 * dpr);
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${tMin.toFixed(1)}s`, paddingLeft, paddingTop + plotHeight + 8 * dpr);
      ctx.fillText(`${tMax.toFixed(1)}s`, paddingLeft + plotWidth, paddingTop + plotHeight + 8 * dpr);

      ctx.save();
      ctx.translate(18 * dpr, paddingTop + plotHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = '#bbb';
      ctx.fillText('Amplitude', 0, 0);
      ctx.restore();

      if (seriesCount > 1) {
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.font = `${11 * dpr}px system-ui`;
        let legendX = paddingLeft;
        const legendY = paddingTop - 16 * dpr;
        for (let i = 0; i < seriesCount; i++) {
          const color = DERIVED_COLORS[i % DERIVED_COLORS.length];
          ctx.fillStyle = color;
          ctx.fillRect(legendX, legendY, 10 * dpr, 10 * dpr);
          ctx.fillStyle = '#ccc';
          ctx.fillText(`C${i + 1}`, legendX + 14 * dpr, legendY - 2 * dpr);
          legendX += 48 * dpr;
        }
      }
    }

    function renderDerivedHeatmap(chart) {
      const canvas = chart.canvas;
      const ctx = chart.ctx;
      const displayWidth = canvas.clientWidth || canvas.width || 640;
      const displayHeight = canvas.clientHeight || canvas.height || 160;
      const width = Math.max(1, Math.floor(displayWidth * dpr));
      const height = Math.max(1, Math.floor(displayHeight * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, width, height);

      const history = chart.history || [];
      const timestamps = chart.timestamps || [];
      if (!history.length) {
        ctx.fillStyle = '#bbb';
        ctx.font = `${14 * dpr}px system-ui`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Waiting for data…', width / 2, height / 2);
        return;
      }

      const rows = Math.max(1, chart.rows || 1);
      const cols = Math.max(1, chart.cols || 1);
      const featureCount = Math.max(1, rows * cols);
      const timeLen = history.length;

      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < timeLen; i++) {
        const sample = history[i];
        if (!sample) {
          continue;
        }
        for (let j = 0; j < sample.length; j++) {
          const val = sample[j];
          if (!Number.isFinite(val)) {
            continue;
          }
          if (val < min) min = val;
          if (val > max) max = val;
        }
      }
      if (!Number.isFinite(min) || !Number.isFinite(max)) {
        min = 0;
        max = 1;
      }
      if (Math.abs(max - min) < 1e-9) {
        const pad = Math.max(Math.abs(max) * 0.05, 1e-3);
        min -= pad;
        max += pad;
      }
      const range = max - min || 1;

      const paddingLeft = 55 * dpr;
      const paddingRight = 28 * dpr;
      const paddingTop = 22 * dpr;
      const paddingBottom = 46 * dpr;
      const plotWidth = Math.max(1, width - paddingLeft - paddingRight);
      const plotHeight = Math.max(1, height - paddingTop - paddingBottom);

      const image = ctx.createImageData(plotWidth, plotHeight);
      const timeScale = timeLen / plotWidth;
      const featureScale = featureCount / plotHeight;

      for (let y = 0; y < plotHeight; y++) {
        const featureIdx = Math.min(featureCount - 1, Math.floor(y * featureScale));
        for (let x = 0; x < plotWidth; x++) {
          const timeIdx = Math.min(timeLen - 1, Math.floor(x * timeScale));
          const sample = history[timeIdx];
          const val = sample ? sample[featureIdx] : NaN;
          const norm = Number.isFinite(val) ? (val - min) / range : 0;
          const [rCol, gCol, bCol] = heatColor(norm);
          const idx = (y * plotWidth + x) * 4;
          image.data[idx] = rCol;
          image.data[idx + 1] = gCol;
          image.data[idx + 2] = bCol;
          image.data[idx + 3] = 255;
        }
      }
      ctx.putImageData(image, paddingLeft, paddingTop);

      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1 * dpr;
      ctx.strokeRect(paddingLeft, paddingTop, plotWidth, plotHeight);

      if (rows > 1) {
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1 * dpr;
        for (let r = 1; r < rows; r++) {
          const ratio = (r * cols) / featureCount;
          const y = paddingTop + ratio * plotHeight;
          ctx.beginPath();
          ctx.moveTo(paddingLeft, y);
          ctx.lineTo(paddingLeft + plotWidth, y);
          ctx.stroke();
        }
      }

      const firstTs = timestamps[0] ?? Date.now() * 1e6;
      const timesSeconds = [];
      for (let i = 0; i < timestamps.length; i++) {
        const ts = timestamps[i];
        if (typeof ts === 'number') {
          timesSeconds.push((ts - firstTs) / 1e9);
        } else {
          timesSeconds.push(i * 0.001);
        }
      }
      const tMinValue = timesSeconds.length ? timesSeconds[0] : 0;
      let tMaxValue = timesSeconds.length > 1 ? timesSeconds[timesSeconds.length - 1] : tMinValue;
      if (!Number.isFinite(tMaxValue)) {
        tMaxValue = tMinValue;
      }
      if (tMaxValue <= tMinValue) {
        tMaxValue = tMinValue + Math.max((timeLen - 1) * 0.001, 0.1);
      }

      ctx.fillStyle = '#bbb';
      ctx.font = `${12 * dpr}px system-ui`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText('Time (s)', paddingLeft + plotWidth / 2, paddingTop + plotHeight + 24 * dpr);
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${tMinValue.toFixed(1)}s`, paddingLeft, paddingTop + plotHeight + 8 * dpr);
      ctx.fillText(`${tMaxValue.toFixed(1)}s`, paddingLeft + plotWidth, paddingTop + plotHeight + 8 * dpr);

      ctx.save();
      ctx.translate(18 * dpr, paddingTop + plotHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = '#bbb';
      ctx.fillText('Feature index', 0, 0);
      ctx.restore();
    }

    function drawDerived(preproc, packetTimestamp) {
      if (!preproc || !preproc.magnitude) {
        return;
      }
      const chart = ensureDerivedChart(preproc.id, preproc.title, preproc.expected_shape);
      let array;
      try {
        array = typedArrayFor(preproc.magnitude);
      } catch (err) {
        console.error('Derived: failed to decode magnitude', err);
        return;
      }
      if (!array || !array.length) {
        chart.meta.textContent = 'No data';
        chart.history = [];
        chart.timestamps = [];
        chart.ctx.clearRect(0, 0, chart.canvas.width, chart.canvas.height);
        return;
      }

      const shape = Array.isArray(preproc.shape) ? preproc.shape.map((value) => Number(value)) : null;
      let rows = 1;
      let cols = array.length;
      if (shape && shape.length === 2) {
        const sRows = Number(shape[0]);
        const sCols = Number(shape[1]);
        if (Number.isFinite(sRows) && Number.isFinite(sCols) && sRows > 0 && sCols > 0) {
          rows = Math.max(1, Math.round(sRows));
          cols = Math.max(1, Math.round(sCols));
        }
      } else if (Array.isArray(preproc.magnitude.shape) && preproc.magnitude.shape.length === 2) {
        const sRows = Number(preproc.magnitude.shape[0]);
        const sCols = Number(preproc.magnitude.shape[1]);
        if (Number.isFinite(sRows) && Number.isFinite(sCols) && sRows > 0 && sCols > 0) {
          rows = Math.max(1, Math.round(sRows));
          cols = Math.max(1, Math.round(sCols));
        }
      }

      if (rows * cols !== array.length) {
        rows = 1;
        cols = array.length;
      }

      const expectedText = preproc.expected_shape ? ` · expected ${preproc.expected_shape}` : '';
      const timestampValue = Number(preproc.timestamp_sys ?? packetTimestamp);
      const timestampNs = Number.isFinite(timestampValue) ? timestampValue : Date.now() * 1e6;

      const sampleLength = rows * cols;
      const vector = new Float32Array(sampleLength);
      for (let i = 0; i < sampleLength; i++) {
        const val = array[i];
        vector[i] = Number.isFinite(val) ? val : NaN;
      }

      if (!Array.isArray(chart.history)) {
        chart.history = [];
      }
      if (!Array.isArray(chart.timestamps)) {
        chart.timestamps = [];
      }

      const shapeChanged = chart.rows !== rows || chart.cols !== cols;
      if (shapeChanged) {
        chart.history.length = 0;
        chart.timestamps.length = 0;
      }
      chart.rows = rows;
      chart.cols = cols;

      chart.history.push(vector);
      chart.timestamps.push(timestampNs);
      if (chart.history.length > chart.historyLimit) {
        const drop = chart.history.length - chart.historyLimit;
        chart.history.splice(0, drop);
        chart.timestamps.splice(0, drop);
      }

      let avg = preproc.avg_magnitude ?? preproc.average_magnitude;
      if (avg !== undefined && avg !== null) {
        avg = Number(avg);
      }
      if (!Number.isFinite(avg)) {
        let sum = 0;
        let valid = 0;
        for (let i = 0; i < vector.length; i++) {
          const val = vector[i];
          if (!Number.isFinite(val)) {
            continue;
          }
          sum += val;
          valid += 1;
        }
        avg = valid ? sum / valid : null;
      }

      const historyLen = chart.history.length;
      let spanSeconds = 0;
      if (chart.timestamps.length > 1) {
        const tStart = chart.timestamps[0];
        const tEnd = chart.timestamps[chart.timestamps.length - 1];
        spanSeconds = (tEnd - tStart) / 1e9;
      }
      const spanText = spanSeconds > 0 ? ` · span=${spanSeconds.toFixed(2)}s` : '';
      const historyText = historyLen > 1 ? ` · history=${historyLen}` : '';
      const avgText = Number.isFinite(avg) ? ` · latest avg=${Number(avg).toFixed(3)}` : '';

      if (rows === 1) {
        chart.mode = cols <= DERIVED_LINE_THRESHOLD ? 'line' : 'heatmap';
        if (chart.mode === 'line') {
          renderDerivedLineChart(chart);
        } else {
          renderDerivedHeatmap(chart);
        }
        chart.meta.textContent = `shape=${rows}×${cols}${avgText}${historyText}${spanText}${expectedText}`;
        return;
      }

      chart.mode = 'heatmap';
      renderDerivedHeatmap(chart);

      const heatmapAvgText = Number.isFinite(avg) ? ` · latest avg=${Number(avg).toFixed(3)}` : '';
      chart.meta.textContent = `shape=${rows}×${cols}${heatmapAvgText}${historyText}${spanText}${expectedText}`;
    }

    function drawCamera(meta) {
      if (!meta || !meta.color) {
        console.warn('Camera: no color data');
        return;
      }
      const color = meta.color;
      const shape = color.shape;
      if (shape.length < 2) {
        console.error('Camera: unexpected shape', shape);
        return;
      }
      const height = shape[0];
      const width = shape[1];
      if (cameraCanvas.width !== width || cameraCanvas.height !== height) {
        cameraCanvas.width = width;
        cameraCanvas.height = height;
      }

      const array = typedArrayFor(color);
      const imageData = cameraCtx.createImageData(width, height);
      const data = imageData.data;
      for (let src = 0, dst = 0; src < array.length; src += 3, dst += 4) {
        data[dst] = array[src];
        data[dst + 1] = array[src + 1];
        data[dst + 2] = array[src + 2];
        data[dst + 3] = 255;
      }
      cameraCtx.putImageData(imageData, 0, 0);

      if (meta.depth) {
        const depthShape = meta.depth.shape;
        const dWidth = depthShape[1];
        const dHeight = depthShape[0];
        if (depthCanvas.width !== dWidth || depthCanvas.height !== dHeight) {
          depthCanvas.width = dWidth;
          depthCanvas.height = dHeight;
        }
        depthCanvas.style.display = 'block';
        const depthArray = typedArrayFor(meta.depth);
        const depthData = depthCtx.createImageData(dWidth, dHeight);
        const depthPixels = depthData.data;
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < depthArray.length; i++) {
          const val = depthArray[i];
          if (val < min) min = val;
          if (val > max) max = val;
        }
        const range = max - min || 1;
        for (let i = 0, j = 0; i < depthArray.length; i++, j += 4) {
          const norm = (depthArray[i] - min) / range;
          const g = Math.floor(norm * 255);
          depthPixels[j] = g;
          depthPixels[j + 1] = g;
          depthPixels[j + 2] = 255;
          depthPixels[j + 3] = 180;
        }
        depthCtx.putImageData(depthData, 0, 0);
      } else {
        depthCanvas.style.display = 'none';
      }
    }


    function ensureStreamChart(id, title) {
      if (!wifiStreamsContainer) {
        return null;
      }
      let chart = streamCharts.get(id);
      if (!chart) {
        const wrapper = document.createElement('div');
        wrapper.className = 'stream-card';
        const heading = document.createElement('h3');
        heading.textContent = title || id;
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 160;
        const meta = document.createElement('div');
        meta.className = 'meta';
        wrapper.appendChild(heading);
        wrapper.appendChild(canvas);
        wrapper.appendChild(meta);
        wifiStreamsContainer.appendChild(wrapper);
        chart = {
          id,
          wrapper,
          heading,
          canvas,
          ctx: canvas.getContext('2d'),
          meta,
          lastUpdate: 0,
          history: [],
          timestamps: [],
          t0: null,
          historyLimit: STREAM_HISTORY_LIMIT,
        };
        streamCharts.set(id, chart);
      } else if (title && chart.heading.textContent !== title) {
        chart.heading.textContent = title;
      }
      return chart;
    }

    function renderStreamHistory(chart) {
      const canvas = chart.canvas;
      const ctx = chart.ctx;
      const displayWidth = canvas.clientWidth || canvas.width;
      const displayHeight = canvas.clientHeight || canvas.height;
      const width = Math.max(1, Math.floor(displayWidth * dpr));
      const height = Math.max(1, Math.floor(displayHeight * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, width, height);
      const count = chart.history.length;
      if (!count) {
        ctx.fillStyle = '#666';
        ctx.font = `${14 * dpr}px system-ui`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Waiting for samples…', width / 2, height / 2);
        return { min: null, max: null, span: 0 };
      }
      const t0 = chart.t0 ?? chart.timestamps[0];
      const times = chart.timestamps.map((t) => ((t - t0) / 1e9));
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < count; i++) {
        const v = chart.history[i];
        if (!Number.isFinite(v)) {
          continue;
        }
        if (v < min) min = v;
        if (v > max) max = v;
      }
      if (!Number.isFinite(min) || !Number.isFinite(max)) {
        min = 0;
        max = 1;
      }
      if (Math.abs(max - min) < 1e-6) {
        const offset = Math.max(Math.abs(min) * 0.05, 1e-3);
        min -= offset;
        max += offset;
      }
      const paddingLeft = 45 * dpr;
      const paddingRight = 20 * dpr;
      const paddingTop = 18 * dpr;
      const paddingBottom = 40 * dpr;
      const plotWidth = Math.max(1, width - paddingLeft - paddingRight);
      const plotHeight = Math.max(1, height - paddingTop - paddingBottom);
      ctx.strokeStyle = '#333';
      ctx.lineWidth = dpr;
      ctx.beginPath();
      ctx.moveTo(paddingLeft, height - paddingBottom);
      ctx.lineTo(width - paddingRight, height - paddingBottom);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(paddingLeft, paddingTop);
      ctx.lineTo(paddingLeft, height - paddingBottom);
      ctx.stroke();
      ctx.strokeStyle = '#4af';
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      const span = Math.max(times[count - 1] - times[0], 1e-6);
      for (let i = 0; i < count; i++) {
        const x = paddingLeft + (plotWidth * (times[i] - times[0])) / span;
        const val = chart.history[i];
        const norm = (val - min) / (max - min);
        const y = paddingTop + (1 - norm) * plotHeight;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.fillStyle = '#ccc';
      ctx.font = `${12 * dpr}px system-ui`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      const latest = chart.history[count - 1];
      if (Number.isFinite(latest)) {
        ctx.fillText(`${latest.toFixed(3)}`, width - paddingRight, paddingTop + 12 * dpr);
      }
      return { min, max, span: times[count - 1] - times[0] };
    }

    function drawWifiStreams(streams, packetTimestamp) {
      if (!wifiStreamsContainer) {
        return;
      }
      const seen = new Set();
      const ts = Number(packetTimestamp);
      const nowNs = Number.isFinite(ts) ? ts : Date.now() * 1e6;
      if (Array.isArray(streams)) {
        streams.forEach((stream, index) => {
          const id = stream.id || stream.title || `stream_${index + 1}`;
          const chart = ensureStreamChart(id, stream.title || id);
          if (!chart) {
            return;
          }
          let avg = Number(stream.avg_magnitude);
          if (!Number.isFinite(avg)) {
            const vector = stream.magnitude ? typedArrayFor(stream.magnitude) : null;
            if (vector && vector.length) {
              let sum = 0;
              let valid = 0;
              for (let i = 0; i < vector.length; i++) {
                const value = vector[i];
                if (!Number.isFinite(value)) {
                  continue;
                }
                sum += value;
                valid += 1;
              }
              avg = valid ? sum / valid : NaN;
            }
          }
          if (!Number.isFinite(avg)) {
            return;
          }
          if (chart.t0 === null || !Number.isFinite(chart.t0)) {
            chart.t0 = nowNs;
          }
          chart.timestamps.push(nowNs);
          chart.history.push(avg);
          if (chart.history.length > chart.historyLimit) {
            const drop = chart.history.length - chart.historyLimit;
            chart.history.splice(0, drop);
            chart.timestamps.splice(0, drop);
            chart.t0 = chart.timestamps.length ? chart.timestamps[0] : nowNs;
          }
          const stats = renderStreamHistory(chart);
          const spanSeconds = chart.timestamps.length > 1
            ? (chart.timestamps[chart.timestamps.length - 1] - chart.timestamps[0]) / 1e9
            : 0;
          const parts = [`avg=${avg.toFixed(3)}`, `history=${chart.history.length}`];
          if (spanSeconds > 0) {
            parts.push(`span=${spanSeconds.toFixed(2)}s`);
          }
          if (stats && Number.isFinite(stats.min) && Number.isFinite(stats.max)) {
            parts.push(`range=[${stats.min.toFixed(2)}, ${stats.max.toFixed(2)}]`);
          }
          chart.meta.textContent = parts.join(' · ');
          chart.lastUpdate = nowNs;
          seen.add(id);
        });
      }
      streamCharts.forEach((chart, id) => {
        if (seen.has(id)) {
          return;
        }
        if (nowNs - chart.lastUpdate > 5e9) {
          if (chart.wrapper && chart.wrapper.parentNode) {
            chart.wrapper.parentNode.removeChild(chart.wrapper);
          }
          streamCharts.delete(id);
        }
      });
    }

    function drawWifi(meta) {
      if (!meta) {
        console.warn('WiFi: missing payload');
        return null;
      }

      let avg = meta.avg_magnitude ?? meta.average_magnitude;
      if (avg !== undefined && avg !== null) {
        avg = Number(avg);
      }
      if (!Number.isFinite(avg) && meta.magnitude) {
        try {
          const array = typedArrayFor(meta.magnitude);
          if (array && array.length) {
            let sum = 0;
            for (let i = 0; i < array.length; i++) {
              sum += array[i];
            }
            avg = sum / array.length;
          }
        } catch (err) {
          console.error('WiFi: failed to decode magnitude', err);
        }
      }

      if (!Number.isFinite(avg)) {
        console.warn('WiFi: unable to compute average magnitude');
        return null;
      }

      const ts = typeof meta.timestamp_sys === 'number' ? meta.timestamp_sys : null;
      if (!pushWifiPoint(ts, avg)) {
        return null;
      }

      drawWifiStreams(meta.streams || null, meta.timestamp_sys);
      renderWifiChart();
      return avg;
    }

    const source = new EventSource('events');
    source.onopen = () => updateStatus('Connected');
    source.onerror = () => updateStatus('Disconnected – retrying…', false);

    // Listen for camera events
    source.addEventListener('camera', (event) => {
      try {
        const payload = JSON.parse(event.data);
        drawCamera(payload);
        cameraMeta.innerHTML = `<span class="badge">Frame ${payload.frame_id ?? '–'}</span>sys_ns=${payload.timestamp_sys ?? '–'}`;
      } catch (err) {
        console.error('Failed to process camera event', err, event.data);
      }
    });

    // Listen for wifi events
    source.addEventListener('wifi', (event) => {
      try {
        const payload = JSON.parse(event.data);
        const avg = drawWifi(payload);
        const avgText = Number.isFinite(avg) ? ` · avg=${avg.toFixed(3)}` : '';
        let gestureText = '';
        if (payload.gesture && typeof payload.gesture.label === 'string') {
          const score = typeof payload.gesture.score === 'number' ? ` (${(payload.gesture.score * 100).toFixed(1)}%)` : '';
          gestureText = ` · gesture=${payload.gesture.label}${score}`;
          if (gestureDisplay) {
            gestureDisplay.textContent = payload.gesture.label;
            gestureDisplay.classList.toggle('idle', payload.gesture.label === 'unknown');
          }
        } else if (gestureDisplay) {
          gestureDisplay.textContent = 'WAITING';
          gestureDisplay.classList.add('idle');
        }
        wifiMeta.innerHTML = `<span class=\"badge\">${payload.device ?? 'device'}</span>packet=${payload.packet_id ?? '–'} · sys_ns=${payload.timestamp_sys ?? '–'}${payload.rssi !== undefined ? ` · RSSI=${payload.rssi}` : ''}${avgText}${gestureText}`;
        if (Array.isArray(payload.preprocessed)) {
          payload.preprocessed.forEach((entry) => drawDerived(entry, payload.timestamp_sys));
        }
      } catch (err) {
        console.error('Failed to process wifi event', err, event.data);
      }
    });

    // Listen for keepalive events (optional, just to avoid warnings)
    source.addEventListener('keepalive', (event) => {
      // Keepalive - do nothing, just prevents "unhandled event" warnings
    });
  </script>
</body>
</html>
"""

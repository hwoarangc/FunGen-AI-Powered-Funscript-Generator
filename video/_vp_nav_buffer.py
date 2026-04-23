"""VideoProcessor NavBufferMixin — arrow-key navigation backed by a deque
of recently-played frames and the PyAV source for misses.

The buffer is populated by the PyAV processing loop as frames are decoded,
so sequential forward/backward arrow-key presses land on instant O(1)
lookups. Misses go through a dedicated async worker so the imgui main
thread never blocks on get_frame (one hot keyframe seek can be 300+ ms
on 8K VR).
"""

import threading
from typing import Optional

import numpy as np


POINT_NAV_PREFETCH_MARGIN = 15


class NavBufferMixin:
    """Mixin fragment for VideoProcessor."""

    def _compute_nav_buffer_size(self) -> int:
        """Compute arrow-nav buffer size based on available RAM.

        Each frame is yolo_input_size^2 * 3 bytes. Cap at ~10% of free RAM,
        floor 120 frames, ceiling 1800 (~1 minute at 30fps).
        """
        frame_bytes = self.yolo_input_size * self.yolo_input_size * 3
        try:
            import psutil
            avail = psutil.virtual_memory().available
            budget = int(avail * 0.10)
            max_frames = max(120, budget // frame_bytes)
            max_frames = min(max_frames, 1800)
        except (ImportError, OSError):
            max_frames = 300
        self.logger.debug(
            f"Arrow-nav buffer: {max_frames} frames "
            f"({max_frames * frame_bytes / (1024*1024):.0f} MB)"
        )
        return max_frames

    # ------------------------------------------------------------------ buffer

    def _buffer_lookup(self, target_frame: int) -> Optional[np.ndarray]:
        """O(1) lookup in the frame buffer by offset from first entry."""
        with self._frame_buffer_lock:
            if not self._frame_buffer:
                return None
            first_idx = self._frame_buffer[0][0]
            offset = target_frame - first_idx
            if 0 <= offset < len(self._frame_buffer):
                stored_idx, frame_data = self._frame_buffer[offset]
                if stored_idx == target_frame:
                    return frame_data
            return None

    def _buffer_append(self, frame_index: int, frame_data: np.ndarray):
        """Append a frame to the buffer; clear on non-contiguous append."""
        with self._frame_buffer_lock:
            if self._frame_buffer:
                last_idx = self._frame_buffer[-1][0]
                if frame_index != last_idx + 1:
                    self._frame_buffer.clear()
            self._frame_buffer.append((frame_index, frame_data))

    def _clear_nav_state(self):
        """Clear buffered nav frames. Called on seeks and teardown."""
        with self._frame_buffer_lock:
            self._frame_buffer.clear()

    @property
    def buffer_info(self) -> dict:
        """Return buffer stats for status strip display."""
        with self._frame_buffer_lock:
            size = len(self._frame_buffer)
            capacity = self._frame_buffer.maxlen or 0
            start = self._frame_buffer[0][0] if self._frame_buffer else -1
            end = self._frame_buffer[-1][0] if self._frame_buffer else -1
        return {'size': size, 'capacity': capacity, 'start': start, 'end': end,
                'current': self.current_frame_index}

    def _clear_cache(self):
        with self.frame_cache_lock:
            if self.frame_cache is not None:
                cache_len = 0
                try:
                    cache_len = len(self.frame_cache)
                except (TypeError, AttributeError):
                    pass
                if cache_len > 0:
                    self.logger.debug(f"Clearing frame cache (had {cache_len} items).")
                    self.frame_cache.clear()
        self._clear_nav_state()

    # ----------------------------------------------------------- async fetch

    def _init_arrow_async(self) -> None:
        """Spawn the background arrow-nav fetch worker. Called once from
        VideoProcessor.__init__. Safe to call twice (second call no-ops)."""
        if getattr(self, "_arrow_thread", None) is not None:
            return
        self._arrow_target: Optional[int] = None
        self._arrow_target_lock = threading.Lock()
        self._arrow_wake = threading.Event()
        self._arrow_stop = threading.Event()
        self._arrow_epoch = 0
        self._arrow_thread = threading.Thread(
            target=self._arrow_async_loop, daemon=True, name="ArrowNavFetch")
        self._arrow_thread.start()

    def _stop_arrow_async(self) -> None:
        ev = getattr(self, "_arrow_stop", None)
        if ev is None:
            return
        ev.set()
        self._arrow_wake.set()
        try:
            self._arrow_thread.join(timeout=1.0)
        except Exception:
            pass

    def _enqueue_arrow_fetch(self, target: int) -> int:
        """Request an async fetch of ``target``. Returns an epoch; the worker
        only commits if the epoch still matches at completion time (coalesces
        rapid keypresses so only the final landing target blocks the frame)."""
        with self._arrow_target_lock:
            self._arrow_epoch += 1
            self._arrow_target = int(target)
            my_epoch = self._arrow_epoch
        self._arrow_wake.set()
        return my_epoch

    def _arrow_async_loop(self) -> None:
        while not self._arrow_stop.is_set():
            if not self._arrow_wake.wait(timeout=0.5):
                continue
            self._arrow_wake.clear()
            with self._arrow_target_lock:
                target = self._arrow_target
                my_epoch = self._arrow_epoch
            if target is None or self.pyav_source is None:
                continue
            frame = self.pyav_source.get_frame(int(target), timeout=2.0, accurate=False)
            if frame is None:
                continue
            self._buffer_append(int(target), frame)
            with self._arrow_target_lock:
                stale = (self._arrow_epoch != my_epoch)
            if stale:
                continue
            with self.frame_lock:
                self.current_frame = frame
                self._frame_version += 1

    # --------------------------------------------------------------- arrow nav

    def _nav_to_target(self, target_frame: int) -> Optional[np.ndarray]:
        """Shared path for both forward and backward arrow nav. Cache hit
        returns the frame synchronously; cache miss advances the cursor and
        enqueues an async fetch so imgui never blocks on a slow seek."""
        frame = self._buffer_lookup(target_frame)
        if frame is not None:
            self.current_frame_index = target_frame
            return frame
        if self.pyav_source is None:
            self.logger.warning(f"Nav miss and no PyAV source for frame {target_frame}")
            return None
        # Cache miss: advance cursor sync (timeline tracks the keypress) and
        # enqueue an async fetch. Background worker commits the frame via
        # current_frame + _frame_version bump when decode lands.
        self.current_frame_index = target_frame
        self._enqueue_arrow_fetch(target_frame)
        return None

    def arrow_nav_forward(self, target_frame: int) -> Optional[np.ndarray]:
        """Navigate forward: buffer hit, otherwise PyAV get_frame."""
        if self.arrow_nav_in_progress:
            return None
        self.arrow_nav_in_progress = True
        try:
            return self._nav_to_target(target_frame)
        finally:
            self.arrow_nav_in_progress = False

    def arrow_nav_backward(self, target_frame: int) -> Optional[np.ndarray]:
        """Navigate backward: buffer hit, otherwise PyAV get_frame."""
        return self._nav_to_target(target_frame)

    def prefetch_around(self, center_frame: int, margin: int = POINT_NAV_PREFETCH_MARGIN):
        """Warm ±margin frames around ``center_frame`` into the nav buffer
        using the PyAV source. Runs in a background thread so the caller
        (usually a UI click) returns immediately.

        Debounced + single-flight: if called again while a prior prefetch is
        still running, the prior thread is signaled to stop and a new one
        starts at the new center. Point-mashing (Up/Down repeatedly) never
        stacks up multiple 15-frame fetch threads fighting the same decoder.
        """
        total = getattr(self, 'total_frames', 0) or 0
        if total <= 0 or not self.video_path or self.pyav_source is None:
            return

        target_end = min(total - 1, center_frame + margin)
        with self._frame_buffer_lock:
            if self._frame_buffer:
                buf_end = self._frame_buffer[-1][0]
                if buf_end >= target_end:
                    return

        # Lazy-init single-flight state on first call.
        if not hasattr(self, '_prefetch_lock'):
            self._prefetch_lock = threading.Lock()
            self._prefetch_stop_event = threading.Event()
            self._prefetch_thread = None

        # Cancel any in-flight prefetch and wait briefly for it to exit so
        # we don't have two threads hammering get_frame simultaneously.
        with self._prefetch_lock:
            prior = self._prefetch_thread
            if prior is not None and prior.is_alive():
                self._prefetch_stop_event.set()
            self._prefetch_stop_event = threading.Event()
            stop_event = self._prefetch_stop_event

        if prior is not None and prior.is_alive():
            prior.join(timeout=0.1)  # best-effort; don't block the GUI

        def _fill():
            src = self.pyav_source
            if src is None: return
            # Start at center+1: the main thread already seeked to center,
            # so get_frame(center) would force a redundant full seek instead
            # of hitting the +1 pump fast-path. Beginning at center+1 makes
            # every prefetch call a cheap pump.
            start = max(0, center_frame + 1)
            read = 0
            for idx in range(start, target_end + 1):
                if stop_event.is_set():
                    return
                frame = self._buffer_lookup(idx)
                if frame is not None:
                    continue
                frame = src.get_frame(idx, timeout=2.0, accurate=False)
                if frame is None:
                    break
                if stop_event.is_set():
                    return
                self._buffer_append(idx, frame)
                read += 1
            if read > 0:
                self.logger.debug(f"Prefetch: warmed {read} frames near {center_frame}")

        t = threading.Thread(target=_fill, daemon=True, name="NavPrefetch")
        with self._prefetch_lock:
            self._prefetch_thread = t
        t.start()

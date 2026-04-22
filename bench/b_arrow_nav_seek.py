"""Arrow-nav seek latency harness.

Measures wall-time of PyAVFrameSource.get_frame() across a spread of frame
indices that defeat the in-process buffer (every call is a hard miss). The
goal is to characterise the cache-miss latency that gates the imgui main
thread on held-arrow scrubbing.

Usage:
    python -m bench.b_arrow_nav_seek <video.mp4> <label> [N]

Defaults to N=100 samples. Prints CSV: label,phase,p50,p90,p99,max,timeouts.
"""
import logging
import sys
import time
from pathlib import Path


def _bootstrap_path() -> None:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))


def _percentiles(samples_ms: list[float]) -> tuple[float, float, float, float]:
    s = sorted(samples_ms)
    n = len(s)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (s[n // 2],
            s[min(n - 1, int(n * 0.90))],
            s[min(n - 1, int(n * 0.99))],
            s[-1])


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    video_path = sys.argv[1]
    label = sys.argv[2]
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    if not Path(video_path).exists():
        print(f"video not found: {video_path}", file=sys.stderr)
        return 1

    _bootstrap_path()
    # main uses PyAVFrameSource; v0.9.0 dropped it for FFmpegFrameSource. Try both.
    SourceCls = None
    try:
        from video.frame_source.pyav_source import PyAVFrameSource, SourceConfig
        SourceCls = PyAVFrameSource
    except ImportError:
        from video.frame_source.ffmpeg_source import FFmpegFrameSource
        from video.frame_source._types import SourceConfig
        SourceCls = FFmpegFrameSource

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    log = logging.getLogger("bench")

    cfg = SourceConfig(
        video_path=video_path,
        filter_chain="format=bgr24",
        output_w=640,
        output_h=640,
    )
    src = SourceCls(cfg)
    if not src.open():
        print("open failed", file=sys.stderr)
        return 1

    total = src.total_frames
    fps = src.fps or 30.0
    if total <= 10:
        print(f"video too short ({total} frames)", file=sys.stderr)
        return 1
    log.warning(f"video: {Path(video_path).name} | total={total} fps={fps:.2f}")

    # Cache-miss pass: jump across the file with a stride that always lands in
    # a fresh GOP. Pick targets evenly distributed; PyAVFrameSource has no
    # built-in cache so each call requires a real decode.
    stride = max(1, total // (n_samples + 2))
    targets = [stride * i for i in range(1, n_samples + 1)]
    miss_samples: list[float] = []
    timeouts = 0
    for t in targets:
        t0 = time.perf_counter()
        frame = src.get_frame(t, timeout=2.0, accurate=True)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        miss_samples.append(dt_ms)
        if frame is None:
            timeouts += 1

    # Cache-hit-ish baseline: re-fetch the SAME indices we just decoded. PyAV
    # does not have an internal frame cache but the recently-touched GOPs are
    # in OS page cache so this still skews to the lower bound of "decode and
    # produce a frame".
    hit_samples: list[float] = []
    for t in targets[:20]:
        t0 = time.perf_counter()
        src.get_frame(t, timeout=2.0, accurate=True)
        hit_samples.append((time.perf_counter() - t0) * 1000.0)

    # accurate=False variant: land on the GOP keyframe instead of draining to
    # the exact target. This is what Tier A intends to use for arrow nav.
    inacc_samples: list[float] = []
    for t in targets:
        t0 = time.perf_counter()
        src.get_frame(t, timeout=2.0, accurate=False)
        inacc_samples.append((time.perf_counter() - t0) * 1000.0)

    src.close()

    print("label,phase,p50_ms,p90_ms,p99_ms,max_ms,timeouts,n")
    for phase, samples in (("miss_accurate", miss_samples),
                           ("hit_repeat_accurate", hit_samples),
                           ("miss_inaccurate", inacc_samples)):
        p50, p90, p99, mx = _percentiles(samples)
        to = timeouts if phase == "miss_accurate" else 0
        print(f"{label},{phase},{p50:.1f},{p90:.1f},{p99:.1f},{mx:.1f},{to},{len(samples)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

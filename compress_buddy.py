#!/usr/bin/env python3
"""
compress_buddy.py - Compress video files using hardware acceleration or CRF encoding.

Usage:
    python3 compress_buddy.py [options] files...

Options:
    --mode hardware|crf|software   Encode mode (hardware or CRF/software)
    --quality N                    Quality value, 0 being worst and 100 being best (default None)
    --codec h264|h265              Codec to use (h264 for compatibility, h265 for better compression)
    --workers N                    Parallel workers (only for CRF mode)
    --min-kbps N                   Minimum target bitrate in kbps (default 3000)
    --max-kbps N                   Maximum target bitrate in kbps (default 8000)
    --dry-run                      Show actions but don't run final ffmpeg commands (will run some analysis)
    --overwrite                    Overwrite existing files
    --copy-audio                   Copy AAC audio instead of re-encoding
    --suffix mp4|mov|mkv|avi       Output file suffix (default mov)
    --max-width N                  Maximum output width in pixels (preserves aspect)
    --max-height N                 Maximum output height in pixels (preserves aspect)
    --delete-original              Delete original after successful encode (use with caution)
    --output /path/to/outdir       Place converted files into this directory
    --chunk-minutes N              Split output into N-minute chunks
    --log-level LEVEL              Set log level (DEBUG, INFO, WARNING, ERROR)
    --threads N                    Pass `-threads N` to ffmpeg to limit encoder threads
    --force-encoder ENCODER        Force exact ffmpeg encoder token to use (e.g. hevc_videotoolbox, h264_nvenc)
    --target-factor FACTOR         Target size factor relative to source bitrate (0.0 < FACTOR <= 1.0, default 0.7)
    --motion-multiplier MULT       Motion multiplier to adjust bitrate (default 1.0)
    --motion-threshold-seconds N   Only run motion analysis if video is at least N seconds long (default 0)


Notes:
    - Requires ffmpeg and ffprobe in PATH.
    - On macOS, default hardware encoder uses 'hevc_videotoolbox'.
    - On Windows/Linux, attempts to auto-select best available hardware encoder.

Examples:
    # process specific files in place:
    python3 compress_buddy.py video1.mov video2.mov

    # place converted files into /tmp/outdir:
    python3 compress_buddy.py -o /tmp/outdir *.mov

    # dry-run to preview actions without running ffmpeg:
    python3 compress_buddy.py --dry-run -o /tmp/outdir myvideo.mov

    # run with 4 workers using CRF mode:
    python3 compress_buddy.py --workers 4 --mode crf --quality 28 *.mp4

    # split into 15-minute parts for sharing in Apple Photos:
    python3 compress_buddy.py --chunk-minutes 15 -o /tmp/outdir mylongvideo.mov

    # output as AVI container:
    python3 compress_buddy.py --suffix avi myvideo.mov

"""
import argparse
import ctypes
import ctypes.util
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def setup_logging():
    fmt = "[%(asctime)s] %(levelname)s %(message)s"
    datefmt = "%Y/%m/%d %H:%M:%S %z"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


LOG = logging.getLogger("compress_buddy")


def format_cmd_for_logging(cmd):
    """Return a human-readable command string for logs, using platform-appropriate quoting."""
    if os.name == "nt":
        try:
            return subprocess.list2cmdline(cmd)
        except Exception:
            return " ".join(shlex.quote(x) for x in cmd)
    return " ".join(shlex.quote(x) for x in cmd)


def ensure_ffmpeg_available(dry_run: bool):
    """Fail early if ffmpeg/ffprobe are missing, unless this is a dry-run."""
    if dry_run:
        return
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        LOG.error(
            "ffmpeg and ffprobe must be available on PATH, see README for install instructions."
        )
        sys.exit(1)


def get_ffmpeg_hwaccels():
    """Return a set of hwaccel names reported by `ffmpeg -hwaccels`.

    Returns empty set on error.
    """
    try:
        command = ["ffmpeg", "-hide_banner", "-hwaccels"]
        res = run_cmd(command)
        if res.returncode != 0:
            return set()
        lines = res.stdout.splitlines()
        hw = set()
        for i in lines:
            s = i.strip()
            if not s or s.lower().startswith("hardware"):
                continue
            hw.add(s)
        return hw
    except Exception:
        return set()


def get_ffmpeg_encoders():
    """Return a set of encoder names reported by `ffmpeg -encoders`.

    Returns empty set on error.
    """
    try:
        command = ["ffmpeg", "-hide_banner", "-encoders"]
        res = run_cmd(command)
        if res.returncode != 0:
            return set()
        enc = set()
        for line in res.stdout.splitlines():
            # parse lines like: " V..... h264_nvenc           NVIDIA NVENC H.264 encoder"
            m = re.match(r"^\s*[A-Z\.]+\s+(\S+)\s+", line)
            if m:
                enc.add(m.group(1))
        return enc
    except Exception:
        return set()


def get_ffmpeg_decoders():
    """Return a set of decoder names reported by `ffmpeg -decoders`. Returns empty set on error."""
    try:
        command = ["ffmpeg", "-hide_banner", "-decoders"]
        res = run_cmd(command)
        if res.returncode != 0:
            return set()
        dec = set()
        for line in res.stdout.splitlines():
            m = re.match(r"^\s*[A-Z\.]+\s+(\S+)\s+", line)
            if m:
                dec.add(m.group(1))
        return dec
    except Exception:
        return set()


def choose_best_hw_encoder(preferred: str):
    """Given preferred codec ('h264' or 'h265'/'hevc'), return (encoder_name, hwaccel_or_None).

    Returns (None, None) if no suitable hardware encoder found.
    """
    pref = preferred.lower()
    if pref == "hevc":
        pref = "h265"
    # candidate lists ordered by preference
    candidates = []
    if pref.startswith("h264"):
        candidates = [
            ("h264_nvenc", None),
            ("h264_qsv", "qsv"),
            ("h264_d3d11va", "d3d11va"),
            ("h264_dxva2", "dxva2"),
            ("h264_videotoolbox", "videotoolbox"),
        ]
    else:
        # assume h265
        candidates = [
            ("hevc_nvenc", None),
            ("hevc_qsv", "qsv"),
            ("hevc_d3d11va", "d3d11va"),
            ("hevc_dxva2", "dxva2"),
            ("hevc_videotoolbox", "videotoolbox"),
        ]

    encoders = get_ffmpeg_encoders()
    hwaccels = get_ffmpeg_hwaccels()

    def nvenc_available():
        # try to find and load libcuda; libcuda.so.1 is common
        try:
            lib = ctypes.util.find_library("cuda")
            if lib:
                ctypes.CDLL(lib)
                return True
            # try common soname
            ctypes.CDLL("libcuda.so.1")
            return True
        except Exception:
            # log debug detail about why libcuda couldn't be loaded
            LOG.debug("nvenc runtime check: libcuda not loadable, will skip nvenc")
            return False

    have_nvenc = nvenc_available()

    for name, hw in candidates:
        if name in encoders:
            # skip nvenc if CUDA / libcuda not available at runtime
            if "nvenc" in name and not have_nvenc:
                LOG.info(
                    "Found %s encoder but libcuda not available at runtime, skipping nvenc",
                    name,
                )
                continue
            # if encoder requires a hwaccel, ensure it's present (nvenc typically doesn't need -hwaccel)
            if hw and hw not in hwaccels:
                continue
            return name, hw
    return None, None


def nvenc_runtime_available():
    """Return True if libcuda appears loadable at runtime, False otherwise."""
    try:
        lib = ctypes.util.find_library("cuda")
        if lib:
            ctypes.CDLL(lib)
            return True
        ctypes.CDLL("libcuda.so.1")
        return True
    except Exception:
        return False


def run_cmd(cmd):
    LOG.info(f"Running command: {format_cmd_for_logging(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res


def ffprobe_json(path):
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    res = run_cmd(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()}")
    return json.loads(res.stdout)


def motion_analysis(
    video_path: Path,
    sample_rate: int = 10,
) -> dict:
    """
    Analyze video motion using ffmpeg's signalstats filter to measure frame differences.

    Args:
        video_path: Path to input video file
        sample_rate: Analyze every Nth frame (10 = analyze ~10% of frames for speed)

    Returns:
        dict with keys:
            - 'multiplier': float between 1.0 and 1.6 (bitrate scaling factor)
            - 'motion_level': str ('low', 'medium', 'high')
            - 'avg_motion': float (average frame-to-frame difference 0-255)
            - 'peak_motion': float (maximum frame-to-frame difference)
            - 'frame_count': int (total frames analyzed)
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    LOG.info(f"Analyzing motion in: {video_path}")
    LOG.info(f"Sample rate: every {sample_rate}th frame")

    import re

    slash_comma = r"""\,"""
    filters_to_try = [f"select='not(mod(n{slash_comma}{sample_rate}))',signalstats", f"fps=1/{sample_rate},signalstats"]
    ydif_re = re.compile(r"YDIF\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)

    motion_values = []
    frame_count = 0
    stderr_lines = []

    try:
        # Try signalstats with a couple of sampling filter variants
        for vf in filters_to_try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-i",
                str(video_path),
                "-vf",
                vf,
                "-frames:v",
                "60",
                "-f",
                "null",
                "-",
            ]
            LOG.debug("Running ffmpeg for motion sampling with filter: %s", vf)
            result = run_cmd(ffmpeg_cmd)
            stderr_lines = result.stderr.splitlines()

            for line in stderr_lines:
                m = ydif_re.search(line)
                if m:
                    try:
                        ydif = float(m.group(1))
                        motion_values.append(ydif)
                        frame_count += 1
                        LOG.debug("Frame %d: YDIF=%.2f", frame_count, ydif)
                    except ValueError:
                        LOG.warning("Could not parse numeric YDIF from line: %s", line)

            if motion_values:
                break

        # Loose heuristic: look for any Y*DIF-like token then any number on the same line
        if not motion_values:
            loose_re = re.compile(r"\bY[A-Z0-9_]*DIF[A-Z0-9_]*\b", re.IGNORECASE)
            num_re = re.compile(r"([+-]?\d+(?:\.\d+)?)")
            for line in stderr_lines:
                if loose_re.search(line):
                    num_m = num_re.search(line)
                    if num_m:
                        try:
                            ydif = float(num_m.group(1))
                            motion_values.append(ydif)
                            frame_count += 1
                            LOG.debug("(loose) Frame %d: YDIF=%.2f", frame_count, ydif)
                        except ValueError:
                            continue

        # Fallback: try an ffmpeg-only pipeline that computes per-sample-frame differences
        # using tblend=all_mode=difference then signalstats on the resulting diff frames.
        if not motion_values:
            LOG.info("signalstats produced no YDIF tokens; attempting ffmpeg-only tblend+signalstats fallback")
            ffmpeg_diff_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-i",
                str(video_path),
                "-vf",
                f"fps=1/{sample_rate},tblend=all_mode=difference,signalstats",
                "-frames:v",
                "60",
                "-f",
                "null",
                "-",
            ]
            LOG.debug("Running ffmpeg diff pipeline: %s", format_cmd_for_logging(ffmpeg_diff_cmd))
            res = run_cmd(ffmpeg_diff_cmd)
            stderr_lines = res.stderr.splitlines()

            # Parse any Y* metrics emitted by signalstats on the diff frames (YAVG, YMAX, YDIF etc.)
            y_any_re = re.compile(r"\bY[A-Z0-9_]*\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
            for line in stderr_lines:
                m = y_any_re.search(line)
                if m:
                    try:
                        val = float(m.group(1))
                        motion_values.append(val)
                        frame_count += 1
                        LOG.debug("(ffmpeg-diff) Frame %d: Y-val=%.2f", frame_count, val)
                    except ValueError:
                        continue

        if not motion_values:
            sample = "\n".join(stderr_lines[-60:]) if stderr_lines else "(no stderr)"
            LOG.warning("No motion metrics extracted from ffmpeg signalstats or fallback, using default multiplier")
            LOG.debug("stderr sample:\n%s", sample)
            return 1.0

        # Calculate motion statistics
        avg_motion = sum(motion_values) / len(motion_values)
        peak_motion = max(motion_values)

        LOG.info(f"Motion stats - Avg: {avg_motion:.2f}, Peak: {peak_motion:.2f}")

        multiplier, motion_level = _calculate_multiplier(avg_motion, peak_motion)

        LOG.info(
            f"Motion analysis complete: {motion_level.upper()} motion "
            f"(avg={avg_motion:.2f}, peak={peak_motion:.2f}) -> multiplier={multiplier:.2f}x"
        )

        return multiplier
    except Exception as e:
        LOG.warning(f"Unable to complete motion analysis: {e}")
        raise


def _calculate_multiplier(avg_motion: float, peak_motion: float) -> tuple[float, str]:
    """
    Calculate bitrate multiplier based on motion metrics.

    Args:
        avg_motion: Average frame-to-frame difference (0-255)
        peak_motion: Peak frame-to-frame difference (0-255)

    Returns:
        tuple of (multiplier: float, motion_level: str)
    """

    # Use weighted combination of average and peak motion
    # Peak motion weighted slightly higher to catch bursts
    motion_score = (avg_motion * 0.6) + (peak_motion * 0.4)

    # Thresholds (empirically calibrated for sports footage)
    # Low: slow-moving or static content (talking heads, slideshows)
    # Medium: moderate motion (typical sports, conversations with movement)
    # High: fast motion (basketball, soccer, action scenes)

    LOW_THRESHOLD = 8.0
    MEDIUM_THRESHOLD = 20.0

    # Linear interpolation for smooth scaling 1.0 to 1.6
    MIN_MULTIPLIER = 1.0
    MAX_MULTIPLIER = 1.6

    if motion_score < LOW_THRESHOLD:
        motion_level = "low"
        multiplier = MIN_MULTIPLIER
    elif motion_score < MEDIUM_THRESHOLD:
        motion_level = "medium"
        # Linear interpolation between low and medium
        progress = (motion_score - LOW_THRESHOLD) / (MEDIUM_THRESHOLD - LOW_THRESHOLD)
        multiplier = MIN_MULTIPLIER + (1.3 - MIN_MULTIPLIER) * progress
    else:
        motion_level = "high"
        # Linear interpolation for high motion
        if motion_score > MEDIUM_THRESHOLD * 1.5:
            multiplier = MAX_MULTIPLIER
        else:
            progress = (motion_score - MEDIUM_THRESHOLD) / (MEDIUM_THRESHOLD * 0.5)
            multiplier = 1.3 + (MAX_MULTIPLIER - 1.3) * progress

    # Clamp to valid range
    multiplier = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, multiplier))

    return multiplier, motion_level


def compute_bitrate_and_duration(path):
    info = ffprobe_json(path)
    duration = float(info.get("format", {}).get("duration") or 0.0)
    # prefer first video stream bit_rate
    bitrate = None
    bitrate_source = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and s.get("bit_rate"):
            try:
                bitrate = int(s["bit_rate"])
                bitrate_source = "stream"
                break
            except Exception:
                bitrate = None
    # Fall back to container-level bit_rate (some formats like WebM/mkv store bitrate there)
    if not bitrate:
        fmt_br = info.get("format", {}).get("bit_rate")
        if fmt_br:
            try:
                bitrate = int(fmt_br)
                bitrate_source = "format"
            except Exception:
                bitrate = None
    # If still no bitrate, return None so caller can decide (we intentionally do not estimate from size/duration).
    return bitrate, duration, info, bitrate_source


def build_common_base(inp, hardware_accel=None, error_level="error"):
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", error_level]
    if hardware_accel:
        base += ["-hwaccel", hardware_accel]
    base += ["-i", str(inp)]
    return base


def build_reencode_cmd_for_concat(concat_path: str, out_path: Path, args) -> list:
    """Build an ffmpeg command to re-encode a concat list using project defaults.

    This reuses the project's default decisions where possible: CRF mapping via
    `map_quality_to_crf()`, `--threads`, audio handling (`--copy-audio`),
    `--max-width`/`--max-height` scaling, pixel format and container flags.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        args.log_level.lower(),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_path,
    ]

    # Video codec selection: prefer software x264/x265 for fallback
    codec_hint = str(getattr(args, "codec", "") or "").lower()
    if "265" in codec_hint or "hevc" in codec_hint:
        vcodec = "libx265"
    else:
        vcodec = "libx264"

    # Use CRF if quality provided (or default), otherwise fall back to a reasonable CRF
    crf_val = map_quality_to_crf(getattr(args, "quality", None) or 45)

    # Reuse the shared tail builder so container flags, audio handling,
    # scaling, threads and progress behavior remain consistent.
    tail = build_encode_tail(
        args,
        has_video=True,
        has_audio=True,
        vcodec=vcodec,
        crf=crf_val,
        out_path=out_path,
        preset="fast",
    )

    cmd += tail
    return cmd


def build_encode_tail(args, has_video: bool, has_audio: bool, *, vcodec=None, crf=None, bitrate_kbps=None, tmp_path=None, chunking=False, tmp_pattern=None, out_path=None, preset=None):
    """Build the tail (options + output) for an ffmpeg encode command.

    This is the shared logic used by `process_file()` and the concat re-encode
    fallback so both use the same decisions for threads, scaling, audio, pixel
    format and container flags.
    """
    tail = []

    # map video/audio/subs explicitly. When any -map is used, ffmpeg
    # disables automatic mapping so we must include video when present.
    if has_video:
        tail += ["-map", "0:v?"]
    tail += ["-map", "0:a?"]
    tail += ["-map", "0:s?"]

    # Video encoding selection
    if has_video:
        if crf is not None and vcodec is not None:
            preset_val = preset or "veryslow"
            tail += ["-c:v", vcodec, "-preset", preset_val, "-crf", str(crf)]
        elif bitrate_kbps is not None:
            tail += ["-c:v", "libx264", "-b:v", f"{int(bitrate_kbps)}k"]
        elif vcodec is not None:
            tail += ["-c:v", vcodec]

    # threads
    if getattr(args, "threads", None) is not None:
        try:
            tail += ["-threads", str(int(args.threads))]
        except Exception:
            pass

    # Scaling filter
    if getattr(args, "max_width", None) is not None or getattr(args, "max_height", None) is not None:
        mw = args.max_width or "iw"
        mh = args.max_height or "ih"
        vf = f"scale='min({mw},iw)':'min({mh},ih)':force_original_aspect_ratio=decrease"
        tail += ["-vf", vf]

    # Audio handling
    if has_audio:
        if getattr(args, "copy_audio", False):
            tail += ["-c:a", "copy"]
        else:
            tail += ["-c:a", "aac", "-b:a", "128k"]

    tail += ["-pix_fmt", "yuv420p"]

    # container flags for mp4/mov
    if getattr(args, "suffix", None) in ("mp4", "mov"):
        tail += ["-movflags", "+faststart"]

    # progress reporting
    tail += ["-progress", "pipe:1", "-nostats"]

    # chunking vs single output
    if chunking and tmp_pattern:
        # force key frames at boundaries for cleaner cuts
        try:
            seg_seconds = int(getattr(args, "chunk_minutes", 0)) * 60
        except Exception:
            seg_seconds = 0
        seg_seconds = max(1, seg_seconds)
        tail += ["-force_key_frames", f"expr:gte(t,n_forced*{seg_seconds})"]
        tail += [
            "-f",
            "segment",
            "-segment_time",
            str(seg_seconds),
            "-reset_timestamps",
            "1",
            tmp_pattern,
        ]
    else:
        # use provided tmp_path or final out_path
        if tmp_path is not None:
            tail += [str(tmp_path)]
        elif out_path is not None:
            tail += [str(out_path)]

    return tail


def parse_out_time(t):
    # parse "HH:MM:SS[.ms]" into seconds (float)
    try:
        parts = t.split(":")
        if len(parts) != 3:
            return 0.0
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0


def format_bytes(num_bytes):
    """Format bytes into human-friendly string using KB/MB/GB with one decimal when appropriate.

    If the size is >= 1.0 of the next unit, show in that unit (e.g., 1.5 GB instead of 1536.0 MB).
    """
    try:
        num = float(num_bytes)
    except Exception:
        return "0 B"
    # Use IEC binary units to match 1024-based math but make units explicit
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while num >= 1024.0 and idx < len(units) - 1:
        num /= 1024.0
        idx += 1
    # bytes as integer, higher units with two decimals
    if idx == 0:
        return f"{int(num)} {units[idx]}"
    else:
        return f"{num:.2f} {units[idx]}"


def ring_bell():
    """Write a terminal bell to stderr, ignoring errors."""
    try:
        import sys

        sys.stderr.write("\a")
        sys.stderr.flush()
    except Exception:
        pass


def log_created_single(name, orig_size, new_size):
    """Log creation of a single-file output using format_bytes."""
    try:
        if orig_size and new_size is not None:
            LOG.info(
                f"Created {name} ({format_bytes(orig_size)} -> {format_bytes(new_size)})"
            )
        elif new_size is not None:
            LOG.info(f"Created {name} ({format_bytes(new_size)})")
        else:
            LOG.info(f"Created {name}")
    except Exception:
        LOG.info(f"Created {name}")


def log_created_segment(name, new_size, orig_size, total_new_size):
    """Log creation of a segment and an optional running total for chunked outputs."""
    try:
        if orig_size:
            LOG.info(
                f"Created {name} ({format_bytes(new_size)}) ({format_bytes(orig_size)} -> {format_bytes(total_new_size)})"
            )
        else:
            LOG.info(f"Created {name} ({format_bytes(new_size)})")
    except Exception:
        LOG.info(f"Created {name}")


def log_total_segments(orig_size, out, total_new_size, created_count):
    """Log total segments created for chunked outputs."""
    try:
        if orig_size:
            LOG.info(
                f"Created {created_count} segment(s) for {out.name} ({format_bytes(orig_size)} -> {format_bytes(total_new_size)})"
            )
        else:
            LOG.info(
                f"Created {created_count} segment(s) for {out.name} (total {format_bytes(total_new_size)})"
            )
    except Exception:
        LOG.info(f"Created {created_count} segment(s) for {out.name}")


def map_quality_to_crf(q):
    """Map a user-facing quality in 0-100 (100=best) to ffmpeg CRF 0-51 (0=best).

    This performs a linear, inverted mapping where 100 -> 0 and 0 -> 51.
    Values are clamped to the valid ranges.
    """
    try:
        qv = int(q)
    except Exception:
        return 28
    qv = max(0, min(100, qv))
    crf = int(round((100 - qv) * 51.0 / 100.0))
    return max(0, min(51, crf))


def run_ffmpeg_with_progress(cmd, total_duration=None):
    """
    Run ffmpeg with '-progress pipe:1 -nostats' already present in cmd.
    Streams stdout, parses 'out_time' progress keys, and computes speed = out_time_secs / elapsed_secs.
    Returns (returncode, stdout_text, stderr_text, final_speed)
    """
    start = time.time()
    full_cmd = list(cmd)
    preexec = None

    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=preexec,
    )

    final_out_time = 0.0
    stdout_lines = []
    stderr_lines = []
    # last time we wrote a live status line (throttle updates)
    last_status_write = 0.0
    # last percent value printed (float 0.0-100.0), used to avoid regressing progress display
    last_pct_printed = -1.0
    current_speed = 0.0

    try:

        def _compute_progress_fields(out_time_val, start_ts, total_dur):
            """Return (pct_str, eta_text, current_speed, pct_val) for given out_time in seconds.

            pct_val is a float 0.0-100.0 when total_dur is known, otherwise None.
            """
            try:
                elapsed_now = max(1e-6, time.time() - start_ts)
                current_speed_local = (
                    out_time_val / elapsed_now if elapsed_now > 0 else 0.0
                )
            except Exception:
                current_speed_local = 0.0
            pct = "?"
            eta_txt = "?"
            pct_val = None
            try:
                if total_dur and total_dur > 0:
                    pct_val = min(100.0, (out_time_val / total_dur) * 100.0)
                    pct = f"{pct_val:5.1f}%"
                    if current_speed_local > 0:
                        remaining = max(0.0, total_dur - out_time_val)
                        eta_seconds = remaining / current_speed_local
                        hrs = int(eta_seconds // 3600)
                        mins = int((eta_seconds % 3600) // 60)
                        secs = int(eta_seconds % 60)
                        if hrs:
                            eta_txt = f"{hrs}:{mins:02d}:{secs:02d}"
                        else:
                            eta_txt = f"{mins:02d}:{secs:02d}"
            except Exception:
                pass
            return pct, eta_txt, current_speed_local, pct_val

        def _write_progress_line(pct_str, eta_str, speed_val):
            try:
                import sys

                sys.stderr.write(
                    f"\r{pct_str} ETA {eta_str} | Encode speed: {speed_val:.2f}x    "
                )
                sys.stderr.flush()
            except Exception:
                pass

        def _handle_progress_line(out_line):
            nonlocal final_out_time, last_status_write, last_pct_printed, current_speed
            LOG.debug(f"ffmpeg stdout: {out_line.strip()}")
            stdout_lines.append(out_line)
            line = out_line.strip()
            if "=" in line:
                key, val = line.split("=", 1)
                if key == "out_time":
                    try:
                        final_out_time = parse_out_time(val.strip())
                        # compute current speed and update a single-line status (throttled)
                        if time.time() - last_status_write > 0.5:
                            pct, eta_txt, current_speed, pct_val = (
                                _compute_progress_fields(
                                    final_out_time, start, total_duration
                                )
                            )
                            try:
                                if pct_val is None or pct_val >= last_pct_printed:
                                    _write_progress_line(pct, eta_txt, current_speed)
                                    if pct_val is not None:
                                        last_pct_printed = pct_val
                                    last_status_write = time.time()
                            except Exception:
                                pass
                    except Exception:
                        pass

        while True:
            out_line = proc.stdout.readline()
            if out_line:
                _handle_progress_line(out_line)
                continue
            if proc.poll() is not None:
                # drain remaining stdout
                for out_line in proc.stdout:
                    _handle_progress_line(out_line)
                break
                # drain a bit of stderr to avoid blocking if ffmpeg writes a lot
            err_chunk = proc.stderr.readline()
            if err_chunk:
                stderr_lines.append(err_chunk)
                LOG.debug(f"ffmpeg stderr: {err_chunk.strip()}")
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise

    elapsed = max(1e-6, time.time() - start)
    final_speed = final_out_time / elapsed if elapsed > 0 else 0.0

    # collect remaining stderr
    try:
        stderr_rest = proc.stderr.read()
        if stderr_rest:
            stderr_lines.append(stderr_rest)
    except Exception:
        pass

    # clear the live status line we wrote to stderr
    try:
        import sys

        sys.stderr.write("\r" + " " * 80 + "\r")
        sys.stderr.flush()
    except Exception:
        pass

    return (
        proc.returncode,
        "".join(stdout_lines),
        "".join(stderr_lines),
        final_speed,
        elapsed,
        final_out_time,
    )


def process_concat_list(concat_path: str, inputs: list, args):
    """Encode directly from a concat list (single ffmpeg invocation).

    This function builds an ffmpeg command that reads the concat demuxer list
    and then appends the shared encode tail produced by `build_encode_tail()`.
    It handles chunking and moving segments (if requested) just like `process_file()`.
    """
    try:
        # Determine output directory and name
        if getattr(args, "output", None):
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / ("joined." + args.suffix)
        else:
            out_dir = Path(inputs[0]).parent
            out = out_dir / ("joined." + args.suffix)

        chunk_minutes = getattr(args, "chunk_minutes", 0) or 0
        chunking = int(chunk_minutes) > 0

        tmp_dir = None
        tmp_path = None
        tmp_pattern = None
        if chunking:
            tmp_dir = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=str(out.parent)))
            tmp_pattern = str(tmp_dir / (Path(inputs[0]).stem + ".%03d" + out.suffix))
        else:
            with tempfile.NamedTemporaryFile(prefix=out.stem + ".", suffix=out.suffix, dir=str(out.parent), delete=False) as tf:
                tmp_path = Path(tf.name)

        # build initial concat input command
        top_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            args.log_level.lower(),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
        ]

        # choose codec options
        vcodec = None
        crf_val = None
        bitrate_kbps = None
        if args.mode == "crf":
            # pick software encoder token
            if args.codec == "h265" or str(args.codec).lower() == "hevc":
                vcodec = "libx265"
            else:
                vcodec = "libx264"
            crf_val = map_quality_to_crf(getattr(args, "quality", None) or 45)
        else:
            bitrate_kbps = None

        tail = build_encode_tail(
            args,
            has_video=True,
            has_audio=True,
            vcodec=vcodec,
            crf=crf_val,
            bitrate_kbps=bitrate_kbps,
            tmp_path=tmp_path,
            chunking=chunking,
            tmp_pattern=tmp_pattern,
            out_path=out,
        )

        cmd = top_cmd + tail
        LOG.info("Running concat-encode...")
        # Honor dry-run: log the final ffmpeg command and do not execute it.
        if getattr(args, "dry_run", False):
            LOG.info("(dry-run) Would run: %s", format_cmd_for_logging(cmd))
            # In dry-run mode we don't produce files; return early
            return
        res = run_cmd(cmd)
        if res.returncode != 0:
            LOG.error(f"ffmpeg failed for joined input: {res.stderr}")
            try:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                if tmp_dir and tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            sys.exit(1)

        # handle outputs (segments or single file)
        if chunking:
            seg_files = sorted(Path(tmp_dir).glob(Path(inputs[0]).stem + ".*" + out.suffix))
            if not seg_files:
                LOG.error("No segments produced by ffmpeg concat-encode.")
                return
            total_new_size = 0
            created_count = 0
            for idx, sf in enumerate(seg_files, start=1):
                target_name = out.parent / f"{out.stem}_part{idx:03d}{out.suffix}"
                if target_name.exists() and not args.overwrite:
                    LOG.warning(f"{target_name.name} exists, skipping... (use --overwrite to replace)")
                    continue
                os.replace(sf, target_name)
                new_sz = target_name.stat().st_size if target_name.exists() else None
                total_new_size += new_sz or 0
                created_count += 1
                log_created_segment(target_name.name, new_sz, None, total_new_size)
            if created_count:
                log_total_segments(None, out, total_new_size, created_count)
        else:
            try:
                if out.exists() and not args.overwrite:
                    LOG.error(f"{out.name} exists and --overwrite not set. Skipping moving joined output.")
                else:
                    os.replace(tmp_path, out)
                new_sz = out.stat().st_size if out.exists() else None
                log_created_single(out.name, None, new_sz)
            except Exception:
                LOG.error("Failed to move joined output into final location.")

        return
    finally:
        try:
            Path(concat_path).unlink(missing_ok=True)
        except Exception:
            pass


def write_symlinked_concat(files: list, concat_path: str):
    """Create a temporary directory containing safe symlinked names for each
    input file and write a concat demuxer list that references those symlinks.

    Returns (concat_path_str, symlink_dir_path) where symlink_dir_path should be
    removed by the caller when done. On platforms where symlinks are not
    available or fail, the function falls back to copying the files into the
    temp dir (copying preserves safe names but may be expensive).
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="compress_buddy_join_"))
    try:
        with open(concat_path, "w") as cf:
            for idx, f in enumerate(files, start=1):
                src = Path(f).resolve()
                # create a safe name like 0001.mov, preserving suffix
                safe_name = f"{idx:04d}{src.suffix}"
                dest = tmpdir / safe_name
                try:
                    # attempt to create a relative symlink for portability
                    rel = os.path.relpath(src, start=tmpdir)
                    os.symlink(rel, dest)
                except Exception:
                    # fallback to copying if symlink not allowed
                    try:
                        shutil.copy2(src, dest)
                    except Exception:
                        # last resort: write absolute path line directly
                        cf.write(f"file '{str(src)}'\n")
                        continue
                # write concat entry pointing at the safe filename in the tmpdir
                cf.write(f"file '{str(dest)}'\n")
    except Exception:
        # cleanup on failure
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        raise
    return concat_path, tmpdir


def process_file(path, args):
    inp = Path(path)
    # capture original size early for logging
    try:
        orig_size = inp.stat().st_size
    except Exception:
        orig_size = None
    # determine output path: if args.output is set and is a directory, place file there
    if getattr(args, "output", None):
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / inp.with_suffix(f".{args.suffix}").name
    else:
        out = inp.with_suffix(f".{args.suffix}")
    if out.exists() and not args.overwrite:
        LOG.warning(f"{out.name} exists, skipping... (use --overwrite to replace)")
        return

    bitrate, duration, probe, bitrate_source = compute_bitrate_and_duration(inp)
    if bitrate is None:
        LOG.error(
            f"{inp.name}: ffprobe did not provide a source bitrate. Please re-run with --min-kbps and/or --max-kbps set to explicit values or check the ffprobe output."
        )
        return
    LOG.debug(f"Using source bitrate from: {bitrate_source or 'unknown'}")
    source_kbps = bitrate / 1000.0
    # compute raw target from the source bitrate and the target factor
    base_target = source_kbps * float(getattr(args, "target_factor", 0.7))

    # Compute scale multiplier (default 1.0). We'll compute motion multiplier separately
    # so motion analysis can run even when the user is not requesting scaling.
    scale_multiplier = 1.0
    try:
        video_stream = None
        for s in probe.get("streams", []):
            if s.get("codec_type") == "video":
                video_stream = s
                break
        if (args.max_width is not None or args.max_height is not None) and video_stream:
            in_w = int(video_stream.get("width") or 0)
            in_h = int(video_stream.get("height") or 0)
            # determine output dims based on provided max dimensions; infer missing dimension from input aspect
            mw = args.max_width
            mh = args.max_height
            if in_w and in_h:
                in_aspect = in_w / in_h
                if mw is None and mh is not None:
                    mw = int(mh * in_aspect)
                elif mh is None and mw is not None:
                    mh = int(mw / in_aspect)
            mw = mw or 1920
            mh = mh or 1080
            out_w = min(in_w or mw, mw)
            out_h = min(in_h or mh, mh)

            area_in = max(1, in_w * in_h)
            area_out = max(1, out_w * out_h)
            r_spatial = area_out / area_in

            # frame rate scaling (attempt to read fps)
            def _parse_rational(r):
                try:
                    if not r:
                        return 0.0
                    if isinstance(r, (int, float)):
                        return float(r)
                    if "/" in str(r):
                        num, den = str(r).split("/", 1)
                        return float(num) / float(den) if float(den) != 0 else 0.0
                    return float(r)
                except Exception:
                    return 0.0

            f_in = _parse_rational(
                video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
            )
            f_out = f_in

            spatial_exp = 0.9
            temporal_exp = 1.0

            scale_multiplier = (r_spatial**spatial_exp) * (
                (f_out / f_in) ** temporal_exp if f_in and f_out else 1.0
            )

    except Exception:
        scale_multiplier = 1.0

    # motion multiplier: sample the file to detect high motion (sports)
    try:
        # Decide whether to run motion analysis.
        if args.mode == "hardware" and getattr(args, "quality", None) is not None:
            LOG.info(
                "Hardware mode with explicit quality provided; skipping motion-based bitrate adjustment"
            )
            motion_mult = 1.0
        elif args.motion_multiplier is not None:
            motion_mult = float(args.motion_multiplier)
        elif not getattr(args, "skip_motion_analysis", False) and (
            duration >= int(getattr(args, "motion_threshold_seconds", 120))
        ):
            motion_mult = motion_analysis(inp, args.sample_rate)
        else:
            motion_mult = 1.0
    except Exception:
        motion_mult = 1.0

    # Log the final motion multiplier for visibility
    try:
        LOG.info("Motion multiplier: %.2f", motion_mult)
    except Exception:
        pass

    scale_multiplier *= motion_mult

    # Prevent scale multiplier from becoming too small (aggressive downscale -> tiny target bitrate)
    try:
        # floor at 0.5 to avoid producing unusably low target bitrates
        if scale_multiplier < 0.5:
            LOG.debug(
                "scale_multiplier %.3f is below minimum; clamping to 0.5",
                scale_multiplier,
            )
            scale_multiplier = 0.5
    except Exception:
        pass

    target_kbps = int(max(300, base_target * scale_multiplier))
    if args.min_kbps is not None:
        target_kbps = max(int(args.min_kbps), target_kbps)
    if args.max_kbps is not None:
        target_kbps = min(int(args.max_kbps), target_kbps)
    # Ensure we never target a bitrate higher than the original source bitrate
    try:
        if bitrate is not None:
            target_kbps = min(int(bitrate / 1000.0), target_kbps)
    except Exception:
        # If anything goes wrong here, leave target_kbps as-is
        pass
    LOG.info(
        "%s: duration=%.1fs source=%.0f kbps -> target=%d kbps",
        inp.name,
        duration,
        source_kbps,
        target_kbps,
    )

    # Inspect streams
    streams = probe.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    has_subs = any(s.get("codec_type") == "subtitle" for s in streams)


    # Check whether the input video codec has any hardware decoder available
    if has_video:
        # determine codec_name for video
        in_video_codec = None
        for s in streams:
            if s.get("codec_type") == "video":
                in_video_codec = s.get("codec_name")
                break
        if in_video_codec:
            # ffmpeg decoders are names like 'vp9', 'libvpx-vp9', 'vp9_qsv', 'vp9_cuvid'
            decoders = get_ffmpeg_decoders()
            hw_decoder_present = any(
                d
                for d in decoders
                if in_video_codec in d
                and any(
                    x in d
                    for x in (
                        "qsv",
                        "cuvid",
                        "nvdec",
                        "v4l2m2m",
                        "videotoolbox",
                        "vaapi",
                    )
                )
            )
            if not hw_decoder_present:
                LOG.warning(
                    f"Input codec '{in_video_codec}' appears to lack a hardware decoder on this system, falling back to software decoding..."
                )

    if args.dry_run:
        LOG.info(
            "   (dry-run) mode=%s video=%s audio=%s subs=%s",
            args.mode,
            has_video,
            has_audio,
            has_subs,
        )
        return

    # prepare output parent and chunking parameters
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk_minutes = getattr(args, "chunk_minutes", 0) or 0
    chunking = int(chunk_minutes) > 0
    tmp_dir = None
    tmp_path = None
    if chunking:
        # create a temporary directory next to the output so moves are atomic
        tmp_dir = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=str(out.parent)))
        # ffmpeg will write segments into this directory
        tmp_pattern = str(tmp_dir / (inp.stem + ".%03d" + out.suffix))
        LOG.debug("Segmenting into %s (chunk %dm)", tmp_pattern, chunk_minutes)
    else:
        # create a named temp file in the same directory as the output so atomic replace works across filesystems
        # Use the output stem (name without suffix) as the prefix so the suffix is not duplicated
        with tempfile.NamedTemporaryFile(
            prefix=out.stem + ".", suffix=out.suffix, dir=str(out.parent), delete=False
        ) as tf:
            tmp_path = Path(tf.name)
    try:
        hwaccel = getattr(args, "_hwaccel", None)
        cmd = build_common_base(
            inp,
            hardware_accel=hwaccel if args.mode == "hardware" else None,
            error_level=args.log_level.lower(),
        )

        # keep video map if present
        if has_video:
            cmd += ["-map", "0:v:0"]

        # Determine codec/bitrate decisions and let shared tail append the remaining flags
        tail_vcodec = None
        tail_crf = None
        tail_bitrate = None

        if args.mode == "crf":
            req_enc = f"lib{args.codec.replace('h', 'x')}"
            available_encs = get_ffmpeg_encoders()
            if req_enc not in available_encs:
                LOG.warning(
                    f"{req_enc} requested but not present in ffmpeg build, falling back to libx264...",
                )
                req_enc = "libx264"
                if req_enc not in available_encs:
                    LOG.error(
                        f"Neither libx265 nor libx264 available in ffmpeg encoders: {', '.join(sorted(list(available_encs))[:40]) or '<none>'}",
                    )
                    LOG.error(
                        "no suitable software encoder found. Install x264/x265 or run in hardware mode. Aborting.",
                    )
                    try:
                        if tmp_path and tmp_path.exists():
                            tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return
            crf_val = map_quality_to_crf(args.quality)
            LOG.info("Using CRF=%s (mapped from quality=%s)", crf_val, args.quality)

        else:
            # hardware or bitrate-based mode: keep encoder tokens as before
            enc = args.codec
            if any(k in enc for k in ("nvenc", "qsv", "videotoolbox", "d3d11va", "dxva2")):
                cmd += ["-c:v", enc]
            else:
                cmd += ["-c:v", f"{enc}_videotoolbox"]
            if args.quality:
                cmd += ["-q:v", str(args.quality)]
            else:
                cmd += [
                    "-b:v",
                    f"{target_kbps}k",
                    "-maxrate",
                    f"{target_kbps}k",
                    "-bufsize",
                    f"{max(1000, target_kbps*2)}k",
                ]

        # Build shared tail and append
        tail = build_encode_tail(
            args,
            has_video=has_video,
            has_audio=has_audio,
            vcodec=tail_vcodec,
            crf=tail_crf,
            bitrate_kbps=tail_bitrate,
            tmp_path=tmp_path,
            chunking=chunking,
            tmp_pattern=tmp_pattern,
            out_path=out,
        )

        cmd += tail

        ffmpeg_command_message = "Running ffmpeg command: "
        cmd_str = format_cmd_for_logging(cmd)
        ffmpeg_command_message += f"{cmd_str}"
        LOG.info(ffmpeg_command_message)

        rc, _, stderr_text, speed, elapsed_sec, final_out_time = (
            run_ffmpeg_with_progress(cmd, total_duration=duration)
        )
        if rc != 0:
            err = stderr_text.strip().splitlines()
            tail = err[-10:] if err else ["<no stderr>"]
            tail_text = "\n".join(tail)
            LOG.error(f"ffmpeg failed for {inp.name}: {tail_text}")
            try:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            LOG.debug(f"Full ffmpeg stderr:\n{stderr_text}")
            LOG.debug(f"Full ffmpeg cmd: {' '.join(shlex.quote(x) for x in cmd)}")
            return

        if chunking:
            # move all generated segments from tmp_dir to final names in out.parent
            seg_files = sorted(tmp_dir.glob(inp.stem + ".*" + out.suffix))
            if not seg_files:
                LOG.error(f"No segments produced for {inp.name}. Aborting.")
            total_new_size = 0
            created_count = 0
            for idx, sf in enumerate(seg_files, start=1):
                final_name = f"{inp.stem}_part{idx:03d}{out.suffix}"
                final_path = out.parent / final_name
                if final_path.exists() and not args.overwrite:
                    LOG.warning(
                        f"{final_path.name} exists, skipping... (use --overwrite to replace)"
                    )
                    continue
                os.replace(sf, final_path)
                created_count += 1
                new_sz = 0
                try:
                    new_sz = final_path.stat().st_size
                except Exception:
                    pass
                total_new_size += new_sz
                log_created_segment(final_path.name, new_sz, orig_size, total_new_size)
            # summary for chunked output
            if created_count:
                log_total_segments(orig_size, out, total_new_size, created_count)
        else:
            # Atomic replace
            os.replace(tmp_path, out)
            new_sz = None
            try:
                new_sz = out.stat().st_size
            except Exception:
                new_sz = None
            log_created_single(out.name, orig_size, new_sz)
        LOG.info(f"Encode speed: {speed:.2f}x realtime")
        try:
            hrs = int(elapsed_sec // 3600)
            mins = int((elapsed_sec % 3600) // 60)
            secs = int(elapsed_sec % 60)
            if hrs:
                elapsed_txt = f"{hrs}:{mins:02d}:{secs:02d}"
            else:
                elapsed_txt = f"{mins:02d}:{secs:02d}"
            LOG.info(
                f"Total encode time: {elapsed_txt} (encoded {final_out_time:.1f}s of source)"
            )
        except Exception:
            pass
        if args.delete_original:
            inp.unlink(missing_ok=True)
            LOG.info(f"Deleted original file {inp.name}")
    finally:
        # ring terminal bell to notify completion of encode unless disabled
        if not args.no_bell:
            ring_bell()

        # cleanup temp artifacts
        try:
            if chunking and tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            elif not chunking and tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def main(argv):
    args = arg_parse(argv)

    # Normalize encoder synonyms to canonical 'h264'/'h265'
    if args.codec:
        enc_map = {
            "size": "h265",
            "compatibility": "h264",
            "avc": "h264",
            "x264": "h264",
            "h264": "h264",
            "hevc": "h265",
            "x265": "h265",
            "h265": "h265",
        }
        args.codec = enc_map.get(str(args.codec).lower(), args.codec)

    if args.mode == "software":
        args.mode = "crf"

    if args.mode == "hardware":
        if platform.system() == "Darwin":
            if args.workers != 1:
                LOG.info(
                    "Hardware mode on macOS detected — capping workers to 1 for VideoToolbox stability"
                )
            args.workers = 1
    if args.mode == "hardware" and args.codec == "h265":
        args.codec = "hevc"  # ffmpeg uses 'hevc' for h265 when using hwaccel
    # If hardware mode requested, attempt to pick a suitable hw encoder if available
    if args.mode == "hardware":
        # try to auto-select best available hw encoder
        # If a user forced an exact encoder token, validate and use it
        if getattr(args, "force_encoder", None):
            forced = args.force_encoder
            encs = get_ffmpeg_encoders()
            if forced not in encs:
                LOG.error(
                    f"Requested forced encoder {forced} not available in this ffmpeg build."
                )
                LOG.error(
                    f"Available encoders: {', '.join(sorted(list(encs))[:40]) or '<none>'}"
                )
                sys.exit(1)
            # if forcing nvenc, ensure runtime libcuda is present
            if "nvenc" in forced and not nvenc_runtime_available():
                LOG.error(
                    f"Forced encoder {forced} requires NVENC but libcuda cannot be loaded at runtime. Aborting."
                )
                sys.exit(1)
            LOG.info(f"Using forced hardware encoder {forced}.")
            args.codec = forced
            # try to infer a hwaccel from encoder token
            if "qsv" in forced:
                setattr(args, "_hwaccel", "qsv")
            elif "videotoolbox" in forced:
                setattr(args, "_hwaccel", "videotoolbox")
            elif "vaapi" in forced:
                setattr(args, "_hwaccel", "vaapi")
        else:
            chosen, hwaccel = choose_best_hw_encoder(args.codec)
            if chosen:
                LOG.info(
                    f"Auto-selected hardware encoder {chosen} (hwaccel={hwaccel})."
                )
                args.codec = chosen
                setattr(args, "_hwaccel", hwaccel)
            else:
                LOG.error(
                    "No suitable hardware encoder found; no override provided. Aborting. Run with --mode crf for software encoding or --force-encoder to pick one."
                )
                encs = sorted(get_ffmpeg_encoders())
                hw = sorted(get_ffmpeg_hwaccels())
                decs = sorted(get_ffmpeg_decoders())
                LOG.error(
                    f"Detected encoders (sample): {', '.join(encs[:40]) or '<none>'}"
                )
                LOG.error(f"Detected hwaccels: {', '.join(hw) or '<none>'}")
                LOG.error(
                    f"Detected decoders (sample): {', '.join(decs[:40]) or '<none>'}"
                )
                sys.exit(1)
    # For CRF mode, map 0-100 user quality to ffmpeg CRF (0-51, inverted mapping)
    if args.mode == "crf":
        if args.quality is None:
            # default user-facing quality chosen to correspond roughly to CRF~28
            args.quality = 45
            LOG.info(
                f"No quality provided, defaulting to quality {args.quality} (CRF = {map_quality_to_crf(args.quality)})"
            )
        else:
            LOG.debug(f"User provided quality {args.quality} for CRF mode.")
    if args.quality and (args.quality < 0 or args.quality > 100):
        LOG.error("Quality must be between 0 and 100, aborting.")
        sys.exit(1)

    # Auto-calc `--threads` per worker when not explicitly provided.
    # If multiple workers are used, divide available logical CPUs across workers.
    if getattr(args, "threads", None) is None and getattr(args, "workers", 1) > 1:
        try:
            cores = os.cpu_count() or 1
            per = max(1, cores // int(args.workers))
            args.threads = per
            # Defer informational logging until logging is configured; store a flag
            setattr(args, "_threads_auto", True)
            LOG.debug(
                f"Auto-setting --threads {per} per worker (total CPU cores: {cores})"
            )
        except Exception:
            pass

    # configure logging
    setup_logging()
    LOG.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    ensure_ffmpeg_available(getattr(args, "dry_run", False))

    files = args.files
    quick_joined_path = None

    # Handle --quick_join: concatenate all inputs into a single temp file first
    if getattr(args, "quick_join", False):
        if len(files) < 2:
            LOG.warning("--quick-join requires at least 2 input files, processing normally...")
        else:
            LOG.info(f"Quick-joining {len(files)} input files into a single video...")
            # Create a temporary concat file
            concat_path = None
            joined_tmp = None
            concat_path = None
            try:
                # Try to create the concat list next to the first input so users
                # can inspect it while the job runs. Fall back to system temp
                # directory on any failure (permissions, non-existent parent, etc.).
                try:
                    first_parent = Path(files[0]).parent
                    first_parent.mkdir(parents=False, exist_ok=True)
                    fd_path = first_parent / ("compress_buddy_concat_" + next(tempfile._get_candidate_names()) + ".txt")
                    # Use low-level open to mimic mkstemp behavior (descriptor + path)
                    concat_fd = os.open(str(fd_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    concat_path = str(fd_path)
                except Exception:
                    concat_fd, concat_path = tempfile.mkstemp(suffix=".txt", prefix="compress_buddy_concat_")
                with os.fdopen(concat_fd, "w") as cf:
                    for f in files:
                        # ffmpeg concat demuxer requires absolute paths; quote when needed
                        abs_path = Path(f).resolve()
                        ap = str(abs_path)
                        # If the path contains a single quote, use double quotes around it
                        if "'" in ap:
                            cf.write(f'file "{ap}"\n')
                        else:
                            cf.write(f"file '{ap}'\n")
                
                # Determine output name for joined file
                if getattr(args, "output", None):
                    out_dir = Path(args.output)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    joined_name = "joined" + "." + args.suffix
                else:
                    # Use parent of first input file
                    out_dir = Path(files[0]).parent
                    joined_name = "joined" + "." + args.suffix
                
                joined_path = out_dir / joined_name
                
                # Create temp file for joined output
                with tempfile.NamedTemporaryFile(
                    prefix=joined_path.stem + ".",
                    suffix=joined_path.suffix,
                    dir=str(out_dir),
                    delete=False
                ) as tf:
                    joined_tmp = Path(tf.name)
                
                LOG.info(f"Concatenating to: {joined_path}")
                
                if not args.dry_run:
                    # Run ffmpeg concat using run_cmd() to ensure standardized logging
                    concat_cmd = [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        args.log_level.lower(),
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        concat_path,
                        "-c",
                        "copy",
                        str(joined_tmp),
                    ]

                    result = run_cmd(concat_cmd)


                    if result.returncode != 0:
                        # Copy-based concat failed, inform user and abort
                        LOG.error(
                            "Copy-based concat failed, aborting quick-join..."
                        )
                        sys.exit(1)

                        # Move to final location (respect --overwrite)
                    if joined_path.exists() and not args.overwrite:
                        LOG.error(f"{joined_path.name} exists and --overwrite not set. Aborting quick-join.")
                        try:
                            if joined_tmp and joined_tmp.exists():
                                joined_tmp.unlink(missing_ok=True)
                        except Exception:
                            pass
                        sys.exit(1)
                    os.replace(joined_tmp, joined_path)
                    LOG.info(f"Successfully joined videos into: {joined_path}")

                    # Now process the joined file
                    files = [str(joined_path)]
                    quick_joined_path = Path(joined_path)
                else:
                    LOG.info(f"(dry-run) Would join {len(files)} files into {joined_path}")
                    # In dry-run, just process the first file as a placeholder
                    files = files[:1]
                
            except Exception as e:
                LOG.error(f"Failed to join videos: {e}")
                sys.exit(1)
            finally:
                # Clean up concat file
                try:
                    if concat_path:
                        Path(concat_path).unlink(missing_ok=True)
                except Exception:
                    pass

    try:
        if args.workers > 1 and not getattr(args, "join", False):
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(process_file, f, args): f for f in files}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        LOG.error(
                            f"Exception processing [bold]{futures[fut]}[/bold]: {e}."
                        )
        else:
            if getattr(args, "join", False):
                # Create a temporary concat file
                try:
                    # Try to place the concat list in the same directory as
                    # the first input so it's easy to inspect while running.
                    first_parent = Path(files[0]).parent
                    first_parent.mkdir(parents=False, exist_ok=True)
                    fd_path = first_parent / ("compress_buddy_concat_" + next(tempfile._get_candidate_names()) + ".txt")
                    concat_fd = os.open(str(fd_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    concat_path = str(fd_path)
                except Exception:
                    concat_fd, concat_path = tempfile.mkstemp(suffix=".txt", prefix="compress_buddy_concat_")
                try:
                    with os.fdopen(concat_fd, "w") as cf:
                        for f in files:
                            # ffmpeg concat demuxer requires absolute paths; quote when needed
                            abs_path = Path(f).resolve()
                            ap = str(abs_path)
                            if "'" in ap:
                                cf.write(f'file "{ap}"\n')
                            else:
                                cf.write(f"file '{ap}'\n")

                    process_concat_list(concat_path, files, args)
                finally:
                    # Clean up concat file
                    try:
                        if concat_path:
                            Path(concat_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                for f in files:
                    try:
                        process_file(f, args)
                    except Exception as e:
                        LOG.error(f"Exception processing {f}: {e}.")
    except KeyboardInterrupt:
        LOG.error("\nKeyboard interrupt received, stopping program.")
        sys.exit(130)
    finally:
        # Clean up quick-joined intermediate if created (it's an internal temp)
        try:
            if quick_joined_path and quick_joined_path.exists():
                quick_joined_path.unlink(missing_ok=True)
                LOG.debug(f"Removed temporary quick-joined file {quick_joined_path}")
        except Exception:
            pass


def arg_parse(argv):
    p = argparse.ArgumentParser(description="Compress Buddy helps you compress videos")
    p.add_argument("files", nargs="+", help="input files")
    p.add_argument(
        "--mode",
        choices=("hardware", "crf", "software"),
        default="hardware",
        help="encode mode",
    )
    p.add_argument(
        "--quality",
        type=int,
        default=None,
        help="Quality value, 0 being worst and 100 being best",
    )
    p.add_argument(
        "--codec",
        choices=(
            "h264",
            "h265",
            "avc",
            "hevc",
            "x264",
            "x265",
            "size",
            "compatibility",
        ),
        default="h265",
        help="codec to target (h264/avc/compatibility or h265/hevc/size). 'h264' for compatibility, 'h265' for better compression.",
    )
    p.add_argument(
        "--force-encoder",
        type=str,
        default=None,
        help="Force exact ffmpeg encoder token to use (e.g. hevc_videotoolbox, h264_nvenc). Overrides auto-selection.",
    )
    p.add_argument(
        "--min-kbps",
        type=int,
        default=None,
        help="min target kbps (optional). If not provided and ffprobe cannot determine source bitrate, the run will abort.",
    )
    p.add_argument(
        "--max-kbps",
        type=int,
        default=None,
        help="max target kbps (optional)",
    )
    p.add_argument(
        "--target-factor",
        type=float,
        default=0.7,
        help="Fraction of source bitrate to target (0.0 < factor <= 1.0). Default 0.7",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="show actions but don't run ffmpeg"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="overwrite existing outputs"
    )
    p.add_argument("--workers", type=int, default=1, help="parallel workers")
    p.add_argument(
        "--output",
        "-o",
        dest="output",
        help="output directory (place converted files here)",
        default=None,
    )
    p.add_argument(
        "--chunk-minutes",
        type=int,
        default=0,
        help="split output into N-minute chunks",
    )
    p.add_argument(
        "--join",
        action="store_true",
        help="join all input videos into a single file before processing (works with --chunk-minutes)",
    )
    p.add_argument(
        "--quick-join",
        action="store_true",
        help="Quickly join inputs with a copy-based concat (-c copy) and then process the result; fails if inputs incompatible",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="set log level (DEBUG, INFO, WARNING, ERROR)",
    )
    p.add_argument(
        "--copy-audio",
        action="store_true",
        help="don't re-encode AAC audio streams, just copy them",
    )
    p.add_argument(
        "--suffix",
        choices=("mp4", "mov", "mkv", "avi"),
        default="mov",
        help="output file suffix (default mov)",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Maximum output width in pixels (preserves aspect ratio).",
    )
    p.add_argument(
        "--max-height",
        type=int,
        default=None,
        help="Maximum output height in pixels (preserves aspect ratio).",
    )
    p.add_argument(
        "--delete-original",
        action="store_true",
        help="delete original file after successful compression",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Pass `-threads N` to ffmpeg to limit encoder threads (helps bound CPU usage).",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=10,
        help="Sample the video every N seconds for motion analysis (default 10)",
    )
    p.add_argument(
        "--skip-motion-analysis",
        action="store_true",
        help="Do not run motion analysis even when auto-enabled for long videos",
    )
    p.add_argument(
        "--motion-threshold-seconds",
        type=int,
        default=120,
        help="Auto-run motion analysis for videos with duration >= this many seconds (default 120s)",
    )
    p.add_argument(
        "--no-bell",
        action="store_true",
        help="Do not ring terminal bell on completion",
    )
    p.add_argument(
        "--motion-multiplier",
        type=float,
        default=None,
        help="Manually specify motion multiplier (overrides automatic analysis). Example: 1.2",
    )

    args = p.parse_args(argv)

    # Validate target factor
    try:
        tf = float(args.target_factor)
        if not (0.0 < tf <= 1.0):
            raise ValueError()
    except Exception:
        p.error("--target-factor must be a number > 0 and <= 1.0")

    # Validate min/max if both provided
    if args.min_kbps is not None and args.max_kbps is not None:
        try:
            if int(args.min_kbps) > int(args.max_kbps):
                p.error("--min-kbps must be <= --max-kbps")
        except Exception:
            p.error("Invalid --min-kbps or --max-kbps value")

    # Validate max width/height if provided
    if args.max_width is not None:
        try:
            if int(args.max_width) <= 0:
                p.error("--max-width must be a positive integer")
        except Exception:
            p.error("Invalid --max-width value")
    if args.max_height is not None:
        try:
            if int(args.max_height) <= 0:
                p.error("--max-height must be a positive integer")
        except Exception:
            p.error("Invalid --max-height value")

    if args.sample_rate <= 0:
        p.error("--sample-rate must be a positive integer")

    if args.motion_multiplier is not None:
        try:
            mm = float(args.motion_multiplier)
            if mm <= 0:
                p.error("--motion-multiplier must be > 0")
        except Exception:
            p.error("--motion-multiplier must be a number")

    if args.motion_threshold_seconds is not None and args.motion_threshold_seconds < 0:
        p.error("--motion-threshold-seconds must be >= 0")

    if args.workers <= 0:
        p.error("--workers must be a positive integer")

    # Prevent using both join modes simultaneously
    if args.join and args.quick_join:
        p.error("--join and --quick-join are mutually exclusive; choose one")

    if args.quality is not None:
        try:
            q = int(args.quality)
            if q < 0 or q > 100:
                p.error("--quality must be between 0 and 100")
        except Exception:
            p.error("Invalid --quality value")

    return args


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        LOG.error("Interrupted by user, exiting program.")
        sys.exit(130)

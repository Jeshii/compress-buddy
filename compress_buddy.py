#!/usr/bin/env python3
"""
compress_buddy.py - Compress video files using hardware acceleration or CRF encoding.

Usage:
    python3 compress_buddy.py [options] files...

Options:
    --mode hardware|crf            Encode mode (hardware or CRF)
    --quality N                    Quality value, 0 being worst and 100 being best (default None)
    --codec h264|h265              Codec to use (h264 for compatibility, h265 for better compression)
    --workers N                    Parallel workers (only for CRF mode)
    --min-kbps N                   Minimum target bitrate in kbps (default 3000)
    --max-kbps N                   Maximum target bitrate in kbps (default 8000)
    --dry-run                      Show actions but don't run ffmpeg
    --overwrite                    Overwrite existing files
    --copy-audio                   Copy AAC audio instead of re-encoding
    --suffix mp4|mov|mkv           Output file suffix (default mov)
    --max-width N                  Maximum output width in pixels (preserves aspect)
    --max-height N                 Maximum output height in pixels (preserves aspect)
    --delete-original              Delete original after successful encode
    --output /path/to/outdir       Place converted files into this directory
    --chunk-minutes N              Split output into N-minute chunks
    --log-level LEVEL              Set log level (DEBUG, INFO, WARNING, ERROR)
    --nice N                       Start ffmpeg with this niceness (POSIX)
    --threads N                    Pass `-threads N` to ffmpeg to limit encoder threads
    --force-encoder ENCODER        Force exact ffmpeg encoder token to use (e.g. hevc_videotoolbox, h264_nvenc)
    --target-factor FACTOR         Target size factor relative to source bitrate (0.0 < FACTOR <= 1.0, default 0.7)

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


def _parse_signalstats_text(text, key="YAVG"):
    """Parse signalstats output text and return list of floats for the given key per-frame."""
    vals = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # lines look like: n:0 YMIN:0 YMAX:255 YAVG:0.123 ...
        parts = line.split()
        for p in parts:
            if p.startswith(key + ":"):
                try:
                    v = float(p.split(":", 1)[1])
                    vals.append(v)
                except Exception:
                    pass
    return vals


def compute_motion_multiplier(path, args):
    """
    Compute a motion multiplier based on per-frame absolute differences.

    Returns a multiplier >= 1.0 (1.0 = normal), where higher values indicate more motion.
    This runs a short ffmpeg pass that computes frame diffs and extracts YAVG via signalstats.
    """
    try:
        # build ffmpeg command sampling up to sample_seconds (use -t to limit runtime)
        # Downscale to a small width to speed up analysis while preserving motion characteristics
        # Use scale=320:-2 to preserve aspect ratio
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vf",
            "scale=320:-2,tblend=all_expr=abs(A-B),signalstats",
            "-t",
            str(int(args.sample_seconds)),
            "-f",
            "null",
            "-",
        ]
        LOG.info(
            f"Analyzing motion for {Path(path).name} (sampling {int(args.sample_seconds)}s)..."
        )
        LOG.debug(f"Running motion analysis command: {format_cmd_for_logging(cmd)}")
        # run and capture stderr; signalstats writes stats to stderr
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stats_text = proc.stderr or ""
        vals = _parse_signalstats_text(stats_text, key="YAVG")
        if not vals:
            return 1.0
        # normalize by 255 (YAVG on 0-255 scale)
        norm_vals = [v / 255.0 for v in vals]
        # use median-ish summary (robust to spikes)
        try:
            import statistics

            m = statistics.median(norm_vals)
        except Exception:
            # fallback to simple average
            m = sum(norm_vals) / len(norm_vals)

        # map median to multiplier: thresholds tuned heuristically
        if m < 0.02:
            mult = 1.0
        elif m < 0.06:
            mult = 1.3
        else:
            mult = 1.6
        LOG.debug("Motion analysis: median_norm=%.4f -> multiplier=%.2f", m, mult)
        return mult
    except Exception:
        LOG.debug("Motion analysis failed; defaulting multiplier to 1.0")
        return 1.0


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


def run_ffmpeg_with_progress(cmd, args, total_duration=None):
    """
    Run ffmpeg with '-progress pipe:1 -nostats' already present in cmd.
    Streams stdout, parses 'out_time' progress keys, and computes speed = out_time_secs / elapsed_secs.
    Returns (returncode, stdout_text, stderr_text, final_speed)
    """
    start = time.time()
    full_cmd = list(cmd)
    preexec = None
    # On POSIX, we can set an exact niceness using preexec_fn so the child starts with lower priority.
    if os.name != "nt" and getattr(args, "nice", None) is not None:
        try:
            nice_val = int(args.nice)

            def _set_nice():
                try:
                    os.setpriority(os.PRIO_PROCESS, 0, nice_val)
                except Exception:
                    try:
                        os.nice(nice_val)
                    except Exception:
                        pass

            preexec = _set_nice
            LOG.info(f"ffmpeg will be started with POSIX nice={nice_val}")
        except Exception:
            LOG.debug("Invalid nice value provided, ignoring...")

    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=preexec,
    )

    # If psutil is available, prefer to set niceness via psutil for more platform support (Windows)
    if getattr(args, "nice", None) is not None:
        try:
            import psutil

            p = psutil.Process(proc.pid)
            try:
                # On Unix this sets niceness; on Windows this accepts priority class constants.
                p.nice(int(args.nice))
                LOG.debug(f"Set process niceness via psutil to {args.nice}")
            except Exception:
                # On Windows, map positive niceness to BELOW_NORMAL/IDLE classes
                try:
                    if platform.system() == "Windows":
                        if int(args.nice) >= 10:
                            p.nice(psutil.IDLE_PRIORITY_CLASS)
                        else:
                            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                        LOG.debug(
                            f"Set Windows process priority via psutil (mapped from nice {args.nice})"
                        )
                except Exception:
                    LOG.debug(
                        f"psutil could not set niceness/priority for pid {proc.pid}"
                    )
        except Exception:
            # psutil not available — we've already attempted preexec_fn on POSIX; on Windows we can't set priority.
            if platform.system() == "Windows":
                LOG.warning(
                    "psutil not installed, cannot lower ffmpeg process priority on Windows, install 'psutil' to enable this feature, skipping..."
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
                            pct, eta_txt, current_speed, pct_val = _compute_progress_fields(
                            final_out_time, start, total_duration
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


def process_file(path, args):
    inp = Path(path)
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
        run_motion = False
        # Always consider motion when scaling is requested (we already adapt bitrate for scaling)
        if args.max_width is not None or args.max_height is not None:
            run_motion = True
        else:
            if not getattr(args, "skip_motion_analysis", False) and (
                duration >= int(getattr(args, "motion_threshold_seconds", 120))
            ):
                run_motion = True

        if (
            args.mode == "hardware"
            and getattr(args, "quality", None) is not None
        ):
            LOG.debug(
                "Hardware mode with explicit quality provided; skipping motion-based bitrate adjustment"
            )
            motion_mult = 1.0
        elif run_motion:
            motion_mult = compute_motion_multiplier(inp, args)
        else:
            motion_mult = 1.0
    except Exception:
        motion_mult = 1.0

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

    # pick first audio codec name
    audio_codec_name = None
    for s in streams:
        if s.get("codec_type") == "audio":
            audio_codec_name = s.get("codec_name")
            break

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
        with tempfile.NamedTemporaryFile(
            prefix=out.name + ".", suffix=out.suffix, dir=str(out.parent), delete=False
        ) as tf:
            tmp_path = Path(tf.name)
    try:
        hwaccel = getattr(args, "_hwaccel", None)
        cmd = build_common_base(
            inp,
            hardware_accel=hwaccel if args.mode == "hardware" else None,
            error_level=args.log_level.lower(),
        )

        if has_video:
            cmd += ["-map", "0:v:0"]

        cmd += ["-map", "0:a?"]
        cmd += ["-map", "0:s?"]

        # video settings: apply scaling if max width/height provided
        if args.max_width is not None or args.max_height is not None:
            # try to pick sensible partner dimension from probe to preserve aspect ratio
            mw = args.max_width
            mh = args.max_height
            # find first video stream from probe
            video_stream = None
            for s in probe.get("streams", []):
                if s.get("codec_type") == "video":
                    video_stream = s
                    break
            if video_stream and (mw is None or mh is None):
                try:
                    in_w = int(video_stream.get("width") or 0)
                    in_h = int(video_stream.get("height") or 0)
                    if in_w and in_h:
                        if mw is None and mh is not None:
                            # compute mw from mh using input aspect
                            mw = int(mh * (in_w / in_h))
                        elif mh is None and mw is not None:
                            mh = int(mw / (in_w / in_h))
                except Exception:
                    pass
            mw = mw or 1920
            mh = mh or 1080
            vf = f"scale='min({mw},iw)':'min({mh},ih)':force_original_aspect_ratio=decrease"
            cmd += ["-vf", vf]

        if args.mode == "crf":
            # choose encoder token for CRF mode: prefer requested (libx265/libx264), but
            # fall back gracefully if encoder not present in this ffmpeg build.
            req_enc = f"lib{args.codec.replace('h', 'x')}"
            available_encs = get_ffmpeg_encoders()
            if req_enc not in available_encs:
                LOG.warning(
                    f"{req_enc} requested but not present in ffmpeg build, falling back to libx264...",
                )
                # fall back to libx264
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
            cmd += [
                "-c:v",
                req_enc,
                "-preset",
                "veryslow",
                "-crf",
                str(crf_val),
            ]
        else:
            # If args.codec already contains a concrete encoder name (e.g., nvenc, qsv), use it as-is;
            # otherwise, attempt to use platform-specific hardware encoder naming.
            enc = args.codec
            # common encoder indicators
            if any(
                k in enc for k in ("nvenc", "qsv", "videotoolbox", "d3d11va", "dxva2")
            ):
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
        cmd += ["-pix_fmt", "yuv420p"]

        # allow limiting encoder threads
        if getattr(args, "threads", None) is not None:
            try:
                t = int(args.threads)
                if t > 0:
                    cmd += ["-threads", str(t)]
                    LOG.info("Passing -threads %s to ffmpeg", t)
            except Exception:
                LOG.warning("Invalid threads value, ignoring...")

        # audio handling: prefer copying AAC if requested/available, otherwise encode to AAC
        if has_audio:
            if args.copy_audio or (
                audio_codec_name and audio_codec_name.lower() == "aac"
            ):
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "128k"]

        # request faststart for MP4/MOV so progressive download compatibility is set
        if getattr(args, "suffix", None) in ("mp4", "mov"):
            cmd += ["-movflags", "+faststart"]
        # Request machine-parseable progress on stdout and suppress default tty stats
        cmd += ["-progress", "pipe:1", "-nostats"]

        if chunking:
            # segmenting: use ffmpeg segment muxer
            seg_seconds = int(chunk_minutes) * 60
            # ensure segment_time is at least 1
            seg_seconds = max(1, seg_seconds)
            # attempt to force key-frames at segment boundaries for clean cuts
            cmd += ["-force_key_frames", f"expr:gte(t,n_forced*{seg_seconds})"]
            cmd += [
                "-f",
                "segment",
                "-segment_time",
                str(seg_seconds),
                "-reset_timestamps",
                "1",
                tmp_pattern,
            ]
        else:
            # tell ffmpeg to write output to the tmp file (tmp has correct suffix)
            cmd += [str(tmp_path)]

        ffmpeg_command_message = "Running ffmpeg command: "
        cmd_str = format_cmd_for_logging(cmd)
        ffmpeg_command_message += f"{cmd_str}"
        LOG.info(ffmpeg_command_message)

        rc, _, stderr_text, speed, elapsed_sec, final_out_time = (
            run_ffmpeg_with_progress(cmd, args, total_duration=duration)
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
            for idx, sf in enumerate(seg_files, start=1):
                final_name = f"{inp.stem}_part{idx:03d}{out.suffix}"
                final_path = out.parent / final_name
                if final_path.exists() and not args.overwrite:
                    LOG.warning(f"{final_path.name} exists, skipping... (use --overwrite to replace)")
                    continue
                os.replace(sf, final_path)
                LOG.info(
                    f"Created {final_path.name} ({final_path.stat().st_size / 1024 / 1024:.1f} MB)"
                )
        else:
            # Atomic replace
            os.replace(tmp_path, out)
            LOG.info(f"Created {out.name} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
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
                "Total encode time: %s (encoded %.1fs of source)",
                elapsed_txt,
                final_out_time,
            )
        except Exception:
            pass
        if args.delete_original:
            inp.unlink(missing_ok=True)
            LOG.info(f"Deleted original file {inp.name}")
    finally:
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
                LOG.info(f"Auto-selected hardware encoder {chosen} (hwaccel={hwaccel}).")
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
            args.quality = 44
            LOG.debug(
                f"No quality provided for CRF; defaulting user-quality {args.quality}."
            )
        else:
            LOG.debug(f"User provided quality {args.quality} for CRF mode.")
    if args.quality and (args.quality < 0 or args.quality > 100):
        LOG.error("Quality must be between 0 and 100. Aborting.")
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
    try:
        if args.workers > 1:
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
            for f in files:
                try:
                    process_file(f, args)
                except Exception as e:
                    LOG.error(f"Exception processing {f}: {e}.")
    except KeyboardInterrupt:
        LOG.error("Keyboard interrupt received, stopping program.")
        sys.exit(130)


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
        "--nice",
        type=int,
        default=None,
        help=(
            "Start ffmpeg with this niceness (POSIX). Higher values are lower priority. "
            "Suggested values: 5 (light background), 10 (background), 15 (very low). "
            "On Windows, lowering priority requires installing 'psutil'."
        ),
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Pass `-threads N` to ffmpeg to limit encoder threads (helps bound CPU usage).",
    )
    p.add_argument(
        "--sample-seconds",
        type=int,
        default=15,
        help="Number of seconds to sample for motion analysis (default 15s).",
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

    if args.sample_seconds <= 0:
        p.error("--sample-seconds must be a positive integer")

    if args.motion_threshold_seconds is not None and args.motion_threshold_seconds < 0:
        p.error("--motion-threshold-seconds must be >= 0")

    if args.workers <= 0:
        p.error("--workers must be a positive integer")

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

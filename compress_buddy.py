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
    --min-kbps N                   Minimum target bitrate in kbps
    --max-kbps N                   Maximum target bitrate in kbps
    --dry-run                      Show actions but don't run final ffmpeg commands (will run some analysis)
    --overwrite                    Overwrite existing files
    --copy-audio                   Copy AAC audio instead of re-encoding
    --suffix mp4|mov|mkv|avi       Output file suffix
    --max-width N                  Maximum output width in pixels (preserves aspect)
    --max-height N                 Maximum output height in pixels (preserves aspect)
    --delete-original              Delete original after successful encode (use with caution)
    --output /path/to/outdir       Place converted files into this directory
    --chunk-minutes N              Split output into N-minute chunks
    --chunk-seconds N              Split output into N-second chunks
    --log-level LEVEL              Set log level (DEBUG, INFO, WARNING, ERROR)
    --threads N                    Pass `-threads N` to ffmpeg to limit encoder threads
    --force-encoder ENCODER        Force exact ffmpeg encoder token to use (e.g. hevc_videotoolbox, h264_nvenc)
    --target-factor FACTOR         Target size factor relative to source bitrate (0.0 < FACTOR <= 1.0, default 0.7)
    --motion-multiplier MULT       Motion multiplier to adjust bitrate (default 1.0)
    --join                         Join all input videos into a single file before processing
    --skip-processing              Skip processing of files (useful with --join to only join files)


Notes:
    - Requires ffmpeg and ffprobe in PATH.
    - On macOS, default hardware encoder uses 'hevc_videotoolbox'.
    - On Windows/Linux, attempts to auto-select best available hardware encoder.

Examples:
    # process specific files in place:
    python3 compress_buddy.py video1.mp4 video2.mp4

    # place converted files into /tmp/outdir:
    python3 compress_buddy.py -o /tmp/outdir *.mp4

    # dry-run to preview actions without running ffmpeg:
    python3 compress_buddy.py --dry-run -o /tmp/outdir myvideo.mp4

    # run with 4 workers using CRF mode:
    python3 compress_buddy.py --workers 4 --mode crf --quality 28 *.mp4

    # split into 15-minute parts for sharing in Apple Photos:
    python3 compress_buddy.py --chunk-minutes 15 -o /tmp/outdir mylongvideo.mov

    # output as AVI container:
    python3 compress_buddy.py --suffix avi myvideo.mp4

"""
import argparse
import configparser
import glob
import json
import logging
import os
import platform
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def load_user_config():
    """Load user configuration from standard locations and return a dict of values.

    Search order (first found wins):
      - ./compress_buddy.ini (script directory)
      - $XDG_CONFIG_HOME/compress_buddy/config.ini
      - ~/.config/compress_buddy/config.ini
      - ~/.compress_buddy.ini

    Uses safe defaults if no config is present.
    """
    cfg_paths = []
    script_dir = Path(__file__).resolve().parent
    cfg_paths.append(script_dir / "compress_buddy.ini")
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        cfg_paths.append(Path(xdg) / "compress_buddy" / "config.ini")
    cfg_paths.append(Path.home() / ".config" / "compress_buddy" / "config.ini")
    cfg_paths.append(Path.home() / ".compress_buddy.ini")

    defaults = {
        "target_factor": 0.7,
        "baseline_1080p_kbps": 8000.0,
        "hevc_multiplier": 0.7,
        "default_bit_depth": 10,
        "judgement_upper": 1.15,
        "judgement_lower": 0.85,
        "res_factor_4k": 1.8,
        "res_factor_1080p": 1.0,
        "res_factor_720p": 0.6,
        "res_factor_sd": 0.35,
        "suffix": "mp4",
        # config values we also want available to the runtime
        "preferred_codec": "size",
        "preferred_mode": "hardware",
        "default_quality": 50,
        "fps_baseline": 30.0,
    }

    cp = configparser.ConfigParser()
    found = None
    for p in cfg_paths:
        try:
            if p.exists():
                cp.read(p)
                found = p
                break
        except Exception:
            continue

    out = defaults.copy()
    if found and cp.has_section("defaults"):
        sec = cp["defaults"]

        def _getfloat(k):
            try:
                return float(sec.get(k, out[k]))
            except Exception:
                return out[k]

        def _getbool(k):
            try:
                return sec.getboolean(k, fallback=out[k])
            except Exception:
                return out[k]

        for k in (
            "target_factor",
            "baseline_1080p_kbps",
            "hevc_multiplier",
            "judgement_upper",
            "judgement_lower",
            "res_factor_4k",
            "res_factor_1080p",
            "res_factor_720p",
            "res_factor_sd",
            "default_bit_depth",
            "suffix",
        ):
            out[k] = _getfloat(k)

        # Read some string / integer preferences that aren't floats
        try:
            out["preferred_codec"] = sec.get("preferred_codec", out.get("preferred_codec"))
        except Exception:
            out["preferred_codec"] = out.get("preferred_codec")

        try:
            out["preferred_mode"] = sec.get("preferred_mode", out.get("preferred_mode"))
        except Exception:
            out["preferred_mode"] = out.get("preferred_mode")

        try:
            out["default_quality"] = int(sec.get("default_quality", out.get("default_quality")))
        except Exception:
            out["default_quality"] = out.get("default_quality")

        try:
            out["fps_baseline"] = float(sec.get("fps_baseline", out.get("fps_baseline")))
        except Exception:
            out["fps_baseline"] = out.get("fps_baseline")

    return out

# Globals
LOG = logging.getLogger("compress_buddy")
USER_CONFIG = load_user_config()

def setup_logging():
    fmt = "[%(asctime)s] %(levelname)s %(message)s"
    datefmt = "%Y/%m/%d %H:%M:%S %z"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

def format_cmd_for_logging(cmd):
    """Return a human-readable command string for logs, using platform-appropriate quoting."""
    if os.name == "nt":
        try:
            return subprocess.list2cmdline(cmd)
        except Exception:
            return " ".join(shlex.quote(x) for x in cmd)
    return " ".join(shlex.quote(x) for x in cmd)


def ensure_ffmpeg_available():
    """Fail early if ffmpeg/ffprobe are missing"""
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
        cmd = ["ffmpeg", "-hide_banner", "-hwaccels"]
        res = run_cmd(
            cmd=cmd, dry_run=False  # always run hwaccels check, even in dry-run mode
        )
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
        cmd = ["ffmpeg", "-hide_banner", "-encoders"]
        res = run_cmd(
            cmd=cmd, dry_run=False  # always run encoders check, even in dry-run mode
        )
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
        cmd = ["ffmpeg", "-hide_banner", "-decoders"]
        res = run_cmd(
            cmd=cmd, dry_run=False  # always run decoders check, even in dry-run mode
        )
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


def choose_best_hw_encoder(preferred_codec: str) -> tuple[str, bool]:
    """Given preferred codec ('h264' or 'h265'/'hevc'), return (encoder_name, hwaccel_or_None).

    Returns (None, None) if no suitable hardware encoder found.
    """
    pref = (preferred_codec or "").lower()
    # candidate lists ordered by preference

    encoders = get_ffmpeg_encoders()
    hwaccels = get_ffmpeg_hwaccels()

    # Normalize sets to lowercase for matching
    enc_l = {e: e.lower() for e in encoders}
    hw_l = {h: h.lower() for h in hwaccels}

    # Known hw tokens to look for in encoder names
    known_hw_tokens = ["qsv", "cuda", "cuvid", "nvenc", "nvdec", "vaapi", "videotoolbox", "dxva2", "d3d11va", "amf"]

    def _match_pref(pref_list):
        # prefer encoders that contain both a hw token and the codec prefix
        for hw in hw_l.values():
            for enc, el in enc_l.items():
                if hw in el and any(p in el for p in pref_list):
                    return enc, hw
        # fallback: look for known hw tokens embedded in encoder name
        for token in known_hw_tokens:
            for enc, el in enc_l.items():
                if token in el and any(p in el for p in pref_list):
                    # choose the most likely hwaccel name if available
                    hw_choice = token if token in hw_l.values() else None
                    return enc, hw_choice
        # final fallback: return any software encoder matching pref
        for enc, el in enc_l.items():
            if any(p in el for p in pref_list):
                return enc, None
        return None, None

    if pref == "h264":
        return _match_pref(["h264", "avc", "x264"]) or (None, None)
    if pref in ("h265", "hevc"):
        return _match_pref(["h265", "hevc", "x265"]) or (None, None)
    return None, None

def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"

def ffmpeg_version_tuple() -> tuple[int, int, int]:
    res = run_cmd(["ffmpeg", "-version"], dry_run=False)
    if res.returncode != 0 or not res.stdout:
        return (0, 0, 0)
    m = re.search(r"ffmpeg version\s+(\d+)\.(\d+)(?:\.(\d+))?", res.stdout)
    if not m:
        return (0, 0, 0)
    major = int(m.group(1))
    minor = int(m.group(2))
    patch = int(m.group(3) or 0)
    return (major, minor, patch)

def ffmpeg_is_at_least(major:int, minor:int) -> bool:
    v = ffmpeg_version_tuple()
    return (v[0], v[1]) >= (major, minor)

def supports_10bit_hw(encoder_name: str) -> bool:
    """Return True if ffmpeg accepts encoding with the given encoder name and 10-bit pixel format.

    This runs a very short ffmpeg lavfi test encode using `-pix_fmt yuv420p10le` and checks the returncode.
    """
    try:
        if not encoder_name:
            return False
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.1:size=1280x720:rate=1",
            "-c:v",
            encoder_name,
            "-pix_fmt",
            "yuv420p10le",
            "-t",
            "0.1",
            "-f",
            "null",
            "-",
        ]
        res = run_cmd(cmd=cmd, dry_run=False)
        return res.returncode == 0
    except Exception:
        return False


def select_encoder_settings(args: argparse.Namespace) -> str:
    """Centralize encoder selection decisions.

    Returns encoder

    Raises RuntimeError on unrecoverable selection failures (missing encoders).
    """
    # prefer the requested software encoder
    req_enc = getattr(args, "codec", None)
    available_encs = get_ffmpeg_encoders()
    if req_enc not in available_encs:
        raise RuntimeError(
            f"{req_enc} requested but not present in list of available encoders: {', '.join(sorted(list(available_encs))[:40]) or '<none>'}"
        )
    return req_enc


def run_cmd(cmd: list, dry_run: bool = False) -> subprocess.CompletedProcess:
    """Run an external command and return a CompletedProcess."""
    LOG.info(f"Running command: {format_cmd_for_logging(cmd)}")
    if dry_run:
        LOG.debug("Dry-run: skipping execution of external command.")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res


def ffprobe_json(path, args=None):
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
    res = run_cmd(
        cmd=cmd,
        dry_run=False,  # always run ffprobe, even in dry-run mode
    )
    if res.returncode != 0 or not res.stdout:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()}")
    return json.loads(res.stdout)


def compute_bitrate_and_duration(path, args=None):
    info = ffprobe_json(path, args=args)
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
    # Only append hwaccel if a sensible string token was provided
    if hardware_accel and isinstance(hardware_accel, str):
        base += ["-hwaccel", hardware_accel]
    base += ["-i", str(inp)]
    return base


def build_encode_tail(
    args,
    has_video: bool,
    has_audio: bool,
    *,
    vcodec=None,
    quality=None,
    mode=None,
    bitrate_kbps=None,
    tmp_path=None,
    chunking=False,
    tmp_pattern=None,
    out_path=None,
    preset=None,
):
    """Build the tail (options + output) for an ffmpeg encode command.

    This is the shared logic used by `process_file()` and the concat re-encode
    fallback so both use the same decisions for threads, scaling, audio, pixel
    format and container flags.
    """
    tail = []

    # mapping: by default we explicitly map video/audio/subs. When any
    # -map is used, ffmpeg disables automatic mapping so we must include
    # the streams we want. If the user requests `--auto-map`, skip adding
    # explicit -map flags and let ffmpeg perform its automatic selection.
    if not getattr(args, "auto_map", False):
        if has_video:
            tail += ["-map", "0:v?"]
        if has_audio:
            tail += ["-map", "0:a?"]
        tail += ["-map", "0:s?"]

    # Video encoding selection - rely on caller-provided tokens/values
    if has_video:
        if vcodec is not None:
            tail += ["-c:v", vcodec]
        if mode == "crf":
            if quality is not None:
                preset_val = preset or "veryslow"
                tail += ["-preset", preset_val, "-crf", map_quality_to_crf(quality)]
            elif bitrate_kbps is not None:
                tail += ["-b:v", f"{int(bitrate_kbps)}k"]
        else:
            if bitrate_kbps is not None:
                tail += ["-b:v", f"{int(bitrate_kbps)}k"]
                if "videotoolbox" in (vcodec or "").lower():
                    tail += ["-constant_bit_rate", "1"]
            if quality is not None:
                vcodec_l = (vcodec or "").lower()
                if "videotoolbox" in vcodec_l:
                    if is_apple_silicon() and ffmpeg_is_at_least(4, 4):
                        tail += ["-q:v", str(int(quality))]
                    else:
                        raise RuntimeError(
                            "VideoToolbox constant-quality (-q:v) requires Apple Silicon and ffmpeg >= 4.4."
                        )
                else:
                    tail += ["-q:v", str(int(quality))]

    # threads
    if getattr(args, "threads", None) is not None:
        try:
            tail += ["-threads", str(int(args.threads))]
        except Exception:
            pass

    # Scaling filter
    if (
        getattr(args, "max_width", None) is not None
        or getattr(args, "max_height", None) is not None
    ):
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

    # honor requested bit depth (8 or 10). Default comes from USER_CONFIG.
    try:
        bit_depth = int(getattr(args, "bit", USER_CONFIG.get("default_bit_depth", 10)))
    except Exception:
        bit_depth = int(USER_CONFIG.get("default_bit_depth", 10))

    if "videotoolbox" in (vcodec or "").lower():
        pix = "yuv420p10le" if bit_depth == 10 else "yuv420p"
        tail += ["-pix_fmt", pix]

    # If 10-bit requested and output codec is HEVC/x265, add profile hints
    vcodec_l = (vcodec or "").lower() if vcodec is not None else ""
    if bit_depth == 10 and vcodec_l:
        if "hvec_videotoolbox" in vcodec_l:
            tail += ["-profile", "2"]
        elif "libx265" in vcodec_l or "x265" in vcodec_l:
            tail += ["-x265-params", "profile=main10"]
        elif "hevc" in vcodec_l or "h265" in vcodec_l:
            tail += ["-profile:v", "main10"]

    # container flags for mp4/mov
    if getattr(args, "suffix", None) in ("mp4", "mov"):
        tail += ["-movflags", "+faststart"]

    # tags
    if "hvec_videotoolbox" in vcodec_l:
        tail += ["-tag:v", "hvc1"]

    # progress reporting
    tail += ["-progress", "pipe:1", "-nostats"]

    # chunking vs single output
    if chunking and tmp_pattern:
        # force key frames at boundaries for cleaner cuts
        # prefer explicit seconds if provided, fall back to minutes for backward compatibility
        try:
            seg_seconds = int(getattr(args, "chunk_seconds", 0) or 0)
        except Exception:
            seg_seconds = 0
        if not seg_seconds:
            try:
                seg_seconds = int(getattr(args, "chunk_minutes", 0) or 0) * 60
            except Exception:
                seg_seconds = 0
        seg_seconds = max(1, seg_seconds)
        tail += ["-force_key_frames", f"expr:gte(t,n_forced*{seg_seconds})"]
        # Ensure per-segment MP4/MOV have moov atom fronted so importers (Photos) can read them.
        if getattr(args, "suffix", None) in ("mp4", "mov"):
            seg_fmt = getattr(args, "suffix")
            tail += [
                "-segment_format",
                seg_fmt,
                "-segment_format_options",
                "movflags=+faststart",
            ]
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
    """Map a user-facing quality in 1-100 (100=best) to ffmpeg CRF 0-51 (0=best).

    This performs a linear, inverted mapping where 100 -> 0 and 1 -> 51.
    Values are clamped to the valid ranges.
    """
    if q is None:
        return None
    try:
        qv = int(q)
    except Exception:
        return 28
    qv = max(1, min(100, qv))
    crf = int(round((100 - qv) * 51.0 / 100.0))
    return max(0, min(51, crf))


def run_ffmpeg_with_progress(cmd: list, total_duration: float = None) -> tuple:
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
    last_status_write = 0.0
    last_pct_printed = -1.0
    current_speed = 0.0

    # Use threads to read stdout and stderr concurrently to avoid deadlocks
    q_out = queue.Queue()
    q_err = queue.Queue()

    def _reader_thread(stream, q):
        try:
            for line in iter(stream.readline, ""):
                q.put(line)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(
        target=_reader_thread, args=(proc.stdout, q_out), daemon=True
    )
    t_err = threading.Thread(
        target=_reader_thread, args=(proc.stderr, q_err), daemon=True
    )
    t_out.start()
    t_err.start()

    def _compute_progress_fields(out_time_val, start_ts, total_dur):
        try:
            elapsed_now = max(1e-6, time.time() - start_ts)
            current_speed_local = out_time_val / elapsed_now if elapsed_now > 0 else 0.0
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
        LOG.debug(f"\nffmpeg stdout: {out_line.strip()}")
        stdout_lines.append(out_line)
        line = out_line.strip()
        if "=" in line:
            key, val = line.split("=", 1)
            if key == "out_time":
                try:
                    final_out_time = parse_out_time(val.strip())
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

    try:
        # main loop: drain both queues until process exits and queues are empty
        while True:
            try:
                # prefer stdout lines first (progress uses stdout)
                out_line = q_out.get(timeout=0.1)
            except queue.Empty:
                out_line = None

            if out_line is not None:
                _handle_progress_line(out_line)

            # drain any stderr available
            while True:
                try:
                    err_line = q_err.get_nowait()
                    stderr_lines.append(err_line)
                    LOG.debug(f"ffmpeg stderr: {err_line.strip()}")
                except queue.Empty:
                    break

            # break when process has exited and both queues are empty
            if proc.poll() is not None:
                # drain remaining queued lines
                while not q_out.empty():
                    _handle_progress_line(q_out.get())
                while not q_err.empty():
                    line = q_err.get()
                    stderr_lines.append(line)
                    LOG.debug(f"ffmpeg stderr: {line.strip()}")
                break
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise

    elapsed = max(1e-6, time.time() - start)
    final_speed = final_out_time / elapsed if elapsed > 0 else 0.0

    # join reader threads (they will exit once streams are closed)
    try:
        t_out.join(timeout=0.1)
    except Exception:
        pass
    try:
        t_err.join(timeout=0.1)
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


def encode_and_handle_output(
    cmd,
    out,
    inputs,
    args,
    tmp_path=None,
    tmp_dir=None,
    chunking=False,
    total_duration=None,
    orig_size=None,
):
    """Run ffmpeg via `run_ffmpeg_with_progress`, then move segments or atomically replace output.

    Returns (success: bool, final_speed, elapsed_sec, final_out_time).
    """
    LOG.info("Running ffmpeg command: %s", format_cmd_for_logging(cmd))

    rc, _, stderr_text, speed, elapsed_sec, final_out_time = run_ffmpeg_with_progress(
        cmd, total_duration=total_duration
    )
    if rc != 0:
        err = stderr_text.strip().splitlines()
        tail = err[-10:] if err else ["<no stderr>"]
        tail_text = "\n".join(tail)
        LOG.error(
            f"ffmpeg failed for {out.name if out is not None else 'output'}: {tail_text}"
        )
        # attempt cleanup of tmp artifacts
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        LOG.debug(f"Full ffmpeg stderr:\n{stderr_text}")
        LOG.debug(f"Full ffmpeg cmd: {' '.join(shlex.quote(x) for x in cmd)}")
        return False, speed, elapsed_sec, final_out_time

    # handle outputs (segments or single file)
    if chunking:
        seg_files = sorted(Path(tmp_dir).glob(Path(inputs[0]).stem + ".*" + out.suffix))
        if not seg_files:
            LOG.error("No segments produced by ffmpeg.")
            return False, speed, elapsed_sec, final_out_time
        total_new_size = 0
        created_count = 0
        for idx, sf in enumerate(seg_files, start=1):
            target_name = out.parent / f"{out.stem}_part{idx:03d}{out.suffix}"
            if target_name.exists() and not args.overwrite:
                LOG.warning(
                    f"{target_name.name} exists, skipping... (use --overwrite to replace)"
                )
                continue
            os.replace(sf, target_name)
            new_sz = target_name.stat().st_size if target_name.exists() else None
            total_new_size += new_sz or 0
            created_count += 1
            log_created_segment(target_name.name, new_sz, orig_size, total_new_size)
        if created_count:
            log_total_segments(orig_size, out, total_new_size, created_count)
    else:
        try:
            if out.exists() and not args.overwrite:
                LOG.error(
                    f"{out.name} exists and --overwrite not set. Skipping moving output."
                )
            else:
                os.replace(tmp_path, out)
            new_sz = out.stat().st_size if out.exists() else None
            log_created_single(out.name, orig_size, new_sz)
        except Exception:
            LOG.error("Failed to move output into final location.")

    return True, speed, elapsed_sec, final_out_time


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
    encoding_completed = False

    # capture original size early for logging
    try:
        orig_size = inp.stat().st_size
    except Exception:
        orig_size = None
    # determine output path. Support three modes:
    # - args.output_file: explicit single-file target (set in arg_parse)
    # - args.output_is_dir: explicit output directory
    # - default: same directory as input, with suffix
    if getattr(args, "output_file", None):
        out = Path(args.output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
    elif getattr(args, "output_is_dir", False):
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / inp.with_suffix(f".{args.suffix}").name
    else:
        out = inp.with_suffix(f".{args.suffix}")
    try:
        # If the output exists and --overwrite not requested, normally skip.
        # If the output path resolves to the same path as the input, require
        # explicit confirmation via `--overwrite` to avoid accidentally
        # deleting the only copy (we used to continue here which could lead
        # to deleting the original after skipping the move).
        if out.exists() and not args.overwrite:
            try:
                if out.resolve() == inp.resolve():
                    LOG.error(
                        f"Output path {out} equals input {inp}. Refusing to proceed without --overwrite."
                    )
                    return
                else:
                    LOG.warning(
                        f"{out.name} exists, skipping... (use --overwrite to replace)"
                    )
                    return

            except Exception:
                # If resolve() fails for any reason, fall back to conservative skip
                LOG.warning(
                    f"{out.name} exists, skipping... (use --overwrite to replace)"
                )
                return
    except Exception:
        # If any filesystem check fails, continue and let later operations surface errors
        pass

    bitrate, duration, probe, bitrate_source = compute_bitrate_and_duration(inp, args)
    if bitrate is None:
        LOG.error(
            f"{inp.name}: ffprobe did not provide a source bitrate. Please re-run with --min-kbps and/or --max-kbps set to explicit values or check the ffprobe output."
        )
        return
    LOG.debug(f"Using source bitrate from: {bitrate_source or 'unknown'}")
    source_kbps = bitrate / 1000.0
    # compute raw target from the source bitrate and the target factor
    try:
        tf_val = args.target_factor if getattr(args, "target_factor", None) is not None else float(USER_CONFIG.get("target_factor", 0.7))
        base_target = source_kbps * float(tf_val)
    except Exception:
        base_target = source_kbps * float(USER_CONFIG.get("target_factor", 0.7))

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
            # determine output dims based on provided max dimensions
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
            # Simple FPS-aware adjustment: pull baseline from config. Higher
            # source framerates will proportionally increase target bitrate
            # to preserve per-frame quality. This is a lightweight heuristic
            # rather than a full motion analysis.
            fps_baseline = float(getattr(args, "fps_baseline", 30.0))
            try:
                fps_multiplier = (
                    (float(f_in) / fps_baseline) if f_in and fps_baseline > 0 else 1.0
                )
            except Exception:
                fps_multiplier = 1.0

            spatial_exp = 0.9
            temporal_exp = 1.0

            scale_multiplier = (r_spatial**spatial_exp) * (fps_multiplier**temporal_exp)

    except Exception:
        scale_multiplier = 1.0

    try:
        if args.mode == "hardware" and getattr(args, "quality", None) is not None:
            LOG.info(
                "Hardware mode with explicit quality provided, skipping motion-based bitrate adjustment..."
            )
            motion_mult = 1.0
        elif args.motion_multiplier is not None:
            motion_mult = float(args.motion_multiplier)
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
                "scale_multiplier %.3f is below minimum so clamping to 0.5",
                scale_multiplier,
            )
            scale_multiplier = 0.5
    except Exception:
        pass

    # Compute a base target from the source and scale multiplier. The final
    # decision (prefer suggested_kbps vs. target_factor) is applied later
    # after the suggested_kbps heuristic is computed.
    try:
        target_kbps = int(max(300, base_target * scale_multiplier))
    except Exception:
        target_kbps = int(max(300, base_target * scale_multiplier))

    # Bitrate quality suggestion
    try:
        # find first video stream for resolution/codec hints
        vid_stream = None
        for s in probe.get("streams", []):
            if s.get("codec_type") == "video":
                vid_stream = s
                break

        width = int(vid_stream.get("width") or 0) if vid_stream else 0
        height = int(vid_stream.get("height") or 0) if vid_stream else 0
        # Respect user-requested max dimensions when suggesting bitrate: compute
        # the effective output resolution and use that for heuristics so that
        # setting --max-width/--max-height lowers suggested bitrate appropriately.
        try:
            eff_w = width
            eff_h = height
            if (getattr(args, "max_width", None) is not None or getattr(args, "max_height", None) is not None) and width and height:
                in_aspect = float(width) / float(height) if height else 0
                mw = args.max_width
                mh = args.max_height
                if mw is None and mh is not None:
                    mw = int(mh * in_aspect)
                elif mh is None and mw is not None:
                    mh = int(mw / in_aspect)
                mw = mw or 1920
                mh = mh or 1080
                eff_w = min(width, mw)
                eff_h = min(height, mh)
        except Exception:
            eff_w = width
            eff_h = height
        codec_name = (
            vid_stream.get("codec_name")
            if vid_stream and vid_stream.get("codec_name")
            else (getattr(args, "codec", None) or "unknown")
        )

        # resolution category factors (relative to a baseline 1080p expectation)
        # Use effective (possibly scaled) resolution for classification
        if eff_w >= 7680 or eff_h >= 4320:
            res_factor = float(USER_CONFIG.get("res_factor_8k", 3.5))
            res_label = "8k"
        elif eff_w >= 3840 or eff_h >= 2160:
            res_factor = float(USER_CONFIG.get("res_factor_4k", 1.8))
            res_label = "4k"
        elif eff_w >= 1920 or eff_h >= 1080:
            res_factor = float(USER_CONFIG.get("res_factor_1080p", 1.0))
            res_label = "1080p"
        elif eff_w >= 1280 or eff_h >= 720:
            res_factor = float(USER_CONFIG.get("res_factor_720p", 0.6))
            res_label = "720p"
        else:
            res_factor = float(USER_CONFIG.get("res_factor_sd", 0.35))
            res_label = "SD"

        # baseline for 1080p in kbps comes from configuration
        baseline_1080p_kbps = float(USER_CONFIG.get("baseline_1080p_kbps", 8000.0))

        expected_kbps = baseline_1080p_kbps * res_factor

        # codec advantage: HEVC typically needs less bitrate for similar quality
        hevc_mult = float(USER_CONFIG.get("hevc_multiplier", 0.7))
        if codec_name and (
            "265" in str(codec_name).lower() or "hevc" in str(codec_name).lower()
        ):
            expected_kbps *= hevc_mult

        # incorporate motion multiplier if available (fall back to 1.0)
        mm = (
            float(motion_mult)
            if ("motion_mult" in locals() and motion_mult is not None)
            else float(getattr(args, "motion_multiplier", 1.0) or 1.0)
        )
        expected_kbps *= mm

        # Decide label with configurable hysteresis thresholds
        upper = float(USER_CONFIG.get("judgement_upper", 1.15))
        lower = float(USER_CONFIG.get("judgement_lower", 0.85))
        if target_kbps >= expected_kbps * upper:
            level = "high"
        elif target_kbps <= expected_kbps * lower:
            level = "low"
        else:
            level = "standard"

        LOG.debug(
            f"{target_kbps} is {level} for the motion multiplier of {mm} based on the resolution {width}x{height} ({res_label}) and the {codec_name} codec"
        )
        try:
            suggested_kbps = int(round(expected_kbps))
            LOG.info(
                f"Suggested target bitrate: {suggested_kbps} kbps (for {res_label}, motion_mult={mm:.2f}, codec={codec_name})"
            )
        except Exception:
            LOG.warning("Failed to compute suggested bitrate", exc_info=True)
    except Exception:
        # Never fail the run due to the judgement helper
        LOG.warning("Failed to compute bitrate judgement", exc_info=True)

    # Finalize target_kbps: prefer suggested_kbps unless user supplied --target-factor
    try:
        if getattr(args, "target_factor", None) is None and "suggested_kbps" in locals() and suggested_kbps:
            target_kbps = int(max(300, int(suggested_kbps)))
        else:
            target_kbps = int(max(300, base_target * scale_multiplier))
    except Exception:
        target_kbps = int(max(300, base_target * scale_multiplier))

    # Apply min/max caps and ensure we never target higher than source bitrate
    if args.min_kbps is not None:
        target_kbps = max(int(args.min_kbps), target_kbps)
    if args.max_kbps is not None:
        target_kbps = min(int(args.max_kbps), target_kbps)
    try:
        if bitrate is not None:
            target_kbps = min(int(bitrate / 1000.0), target_kbps)
    except Exception:
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
        # Report dry-run summary plus the bitrate judgement (high/standard/low)
        level = locals().get("level", "unknown")
        suggested_kbps = locals().get("suggested_kbps", None)
        try:
            LOG.info(
                "   (dry-run) mode=%s video=%s audio=%s subs=%s",
                args.mode,
                has_video,
                has_audio,
                has_subs,
            )
            if suggested_kbps:
                LOG.info(
                    "   judgement=%s suggested_kbps=%d target_kbps=%d",
                    level,
                    int(suggested_kbps),
                    int(target_kbps),
                )
            else:
                LOG.info("   judgement=%s target_kbps=%s", level, str(target_kbps))
        except Exception:
            LOG.info("   (dry-run) mode=%s video=%s audio=%s subs=%s", args.mode, has_video, has_audio, has_subs)
        return

    # prepare output parent and chunking parameters
    out.parent.mkdir(parents=True, exist_ok=True)
    # If the user requested high-only behavior, skip files not judged 'high'
    try:
        if getattr(args, "high_only", False):
            level = locals().get("level", None)
            if level is None:
                LOG.warning("--high-only was requested but judgement could not be determined, skipping %s...", inp.name)
                return
            if level != "high":
                LOG.info("Skipping %s: judged %s (only compressing 'high')", inp.name, level)
                return
    except Exception:
        pass
    # prefer explicit seconds flag when present
    chunk_seconds = int(getattr(args, "chunk_seconds", 0) or 0)
    if not chunk_seconds:
        chunk_minutes = getattr(args, "chunk_minutes", 0) or 0
        chunk_seconds = int(chunk_minutes) * 60
    chunking = int(chunk_seconds) > 0
    tmp_dir = None
    tmp_path = None
    tmp_pattern = None
    if chunking:
        # create a temporary directory next to the output so moves are atomic
        tmp_dir = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=str(out.parent)))
        # ffmpeg will write segments into this directory
        tmp_pattern = str(tmp_dir / (inp.stem + ".%03d" + out.suffix))
        LOG.debug("Segmenting into %s (chunk %ds)", tmp_pattern, chunk_seconds)
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

        # Let the shared tail handle mapping and output placement. Only compute
        # which encoder / mode values to pass into build_encode_tail so we avoid
        # duplicated flags here.
        # Determine codec/bitrate decisions and let shared tail append the remaining flags
        tail_vcodec = None
        tail_bitrate = None
        tail_preset = None

        # Centralized encoder selection
        try:
            sel_vcodec = select_encoder_settings(args)
            sel_bitrate = target_kbps

        except RuntimeError as e:
            LOG.error(str(e))
            try:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        if args.mode == "crf":
            # software encoder token and CRF mapped
            tail_vcodec = sel_vcodec
            tail_preset = "veryslow"
            tail_crf = map_quality_to_crf(args.quality)
            LOG.info("Using CRF=%s (mapped from quality=%s)", tail_crf, args.quality)
        else:
            # hardware or bitrate-based mode: prefer bitrate targeting
            tail_vcodec = sel_vcodec or args.codec
            tail_bitrate = sel_bitrate

        tail_kwargs = {
            "has_video": has_video,
            "has_audio": has_audio,
            "vcodec": tail_vcodec,
            "quality": getattr(args, "quality", None),
            "bitrate_kbps": tail_bitrate,
            "tmp_path": tmp_path,
            "chunking": chunking,
            "tmp_pattern": tmp_pattern,
            "out_path": out,
            "preset": tail_preset,
        }

        if getattr(args, "quality", None) is not None:
            tail_kwargs["quality"] = getattr(args, "quality")

        try:
            tail = build_encode_tail(args, **tail_kwargs)
        except RuntimeError as e:
            LOG.error(str(e))
            LOG.error("failed to build encode tail. Aborting.")
            return

        cmd += tail

        success, speed, elapsed_sec, final_out_time = encode_and_handle_output(
            cmd,
            out,
            [str(inp)],
            args,
            tmp_path=tmp_path,
            tmp_dir=tmp_dir,
            chunking=chunking,
            total_duration=duration,
            orig_size=orig_size,
        )
        if not success:
            return
        encoding_completed = True  # Set flag to True upon successful encoding
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
    finally:
        # ring terminal bell to notify completion of encode unless disabled
        if not args.no_bell:
            ring_bell()

        # Only delete original if encoding actually completed successfully
        if encoding_completed and args.delete_original:
            try:
                if out.resolve() != inp.resolve():
                    inp.unlink(missing_ok=True)
                    LOG.info(f"Deleted original file {inp.name}")
                else:
                    LOG.warning(f"Not deleting {inp.name} because output overwrote it in-place")
            except Exception as e:
                LOG.warning(f"Failed to delete original: {e}")

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

    if args.mode == "software":
        args.mode = "crf"

    if args.mode == "hardware":
        if platform.system() == "Darwin":
            if args.workers != 1:
                LOG.info(
                    "Hardware mode on macOS detected — capping workers to 1 for VideoToolbox stability"
                )
            args.workers = 1

    # Normalize encoder synonyms to canonical 'h264'/'h265'
    if args.codec:
        enc_map = {
            "software": {
                "size": "h265",
                "compatibility": "h264",
                "avc": "h264",
                "x264": "h264",
                "h264": "h264",
                "hevc": "h265",
                "x265": "h265",
                "h265": "h265",
            },
            "hardware": {
                "size": "h265",
                "compatibility": "h264",
                "avc": "h264",
                "h264": "h264",
                "h265": "h265",
            },
        }
        args.codec = enc_map.get(str(args.codec).lower(), args.codec)
    if args.mode == "hardware" and args.codec == "h265":
        args.codec = "hevc"  # ffmpeg uses 'hevc' for h265 when using hwaccel

    # If hardware mode requested, attempt to pick a suitable hw encoder if available
    if args.mode == "hardware":
        # try to auto-select best available hw encoder
        # If a user forced an exact encoder token, validate and use it
        if getattr(args, "encoder", None):
            encs = get_ffmpeg_encoders()
            if args.encoder not in encs:
                LOG.error(
                    f"Requested encoder {args.encoder} not available in this ffmpeg build."
                )
                LOG.error(
                    f"Available encoders: {', '.join(sorted(list(encs))[:40]) or '<none>'}"
                )
                sys.exit(1)
            LOG.info(f"Using encoder {args.encoder}.")
            args.codec = args.encoder
            # try to infer a hwaccel from encoder token
            enc_low = args.encoder.lower()
            inferred = None
            if "qsv" in enc_low:
                inferred = "qsv"
            elif "videotoolbox" in enc_low:
                inferred = "videotoolbox"
            elif "vaapi" in enc_low:
                inferred = "vaapi"
            elif "nvenc" in enc_low or "cuvid" in enc_low or "nvdec" in enc_low:
                # NVENC/CUDA family
                inferred = "cuda"
            elif "amf" in enc_low:
                # AMD AMF typically pairs with dxva/d3d11 on Windows
                inferred = "d3d11va"
            if inferred:
                setattr(args, "_hwaccel", inferred)
            # If user requested 10-bit output, probe whether this encoder accepts 10-bit
            try:
                if getattr(args, "bit", USER_CONFIG.get("default_bit_depth", 10)) == 10:
                    if not supports_10bit_hw(args.codec):
                        LOG.warning(
                            f"Requested 10-bit output but hardware encoder {args.codec} does not appear to support 10-bit, falling back to 8-bit...",
                        )
                        args.bit = 8
            except Exception:
                pass
        else:
            chosen, hwaccel = choose_best_hw_encoder(args.codec)
            if chosen:
                LOG.info(
                    f"Auto-selected hardware encoder {chosen} (hwaccel={hwaccel})."
                )
                args.codec = chosen

                if hwaccel:
                    setattr(args, "_hwaccel", hwaccel)
                else:
                    # Try to infer a sensible hwaccel on Windows from encoder token
                    try:
                        if platform.system() == "Windows":
                            hwaccels = get_ffmpeg_hwaccels()
                            cl = chosen.lower()
                            if "nvenc" in cl or "cuvid" in cl or "nvdec" in cl:
                                if "cuda" in hwaccels:
                                    setattr(args, "_hwaccel", "cuda")
                                elif "d3d11va" in hwaccels:
                                    setattr(args, "_hwaccel", "d3d11va")
                            elif "qsv" in cl:
                                if "qsv" in hwaccels:
                                    setattr(args, "_hwaccel", "qsv")
                            elif "amf" in cl:
                                if "d3d11va" in hwaccels:
                                    setattr(args, "_hwaccel", "d3d11va")
                                elif "dxva2" in hwaccels:
                                    setattr(args, "_hwaccel", "dxva2")
                    except Exception:
                        pass
                # If user requested 10-bit output, probe whether this encoder accepts 10-bit
                try:
                    if (
                        getattr(args, "bit", USER_CONFIG.get("default_bit_depth", 10))
                        == 10
                    ):
                        if not supports_10bit_hw(args.codec):
                            LOG.warning(
                                f"Requested 10-bit output but auto-selected hardware encoder {args.codec} does not appear to support 10-bit, falling back to 8-bit...",
                            )
                            args.bit = 8
                except Exception:
                    pass
            else:
                LOG.error(
                    "No suitable hardware encoder found. Aborting. Run with --mode crf for software encoding or --force-encoder to pick one."
                )
                encs = sorted(get_ffmpeg_encoders())
                hw = sorted(get_ffmpeg_hwaccels())
                decs = sorted(get_ffmpeg_decoders())
                LOG.error(
                    f"Detected encoders: {', '.join(encs) or '<none>'}"
                )
                LOG.error(f"Detected hwaccels: {', '.join(hw) or '<none>'}")
                LOG.error(
                    f"Detected decoders: {', '.join(decs) or '<none>'}"
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
            # Defer informational logging until logging is configured
            setattr(args, "_threads_auto", True)
            LOG.debug(
                f"Auto-setting --threads {per} per worker (total CPU cores: {cores})"
            )
        except Exception:
            pass

    # configure logging
    setup_logging()
    LOG.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    ensure_ffmpeg_available()

    # Expand inputs: support globbing, directories and recursive search filtered by extensions
    exts_raw = getattr(args, "extensions", None)
    if exts_raw:
        exts = set(
            "." + e.strip().lstrip(".").lower()
            for e in exts_raw.split(",")
            if e.strip()
        )
    else:
        exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".ts"}

    files = []
    for entry in args.files:
        # expand shell-style globs if present
        if any(ch in entry for ch in ["*", "?", "["]):
            matches = glob.glob(entry, recursive=True)
            matches.sort()
            for m in matches:
                p = Path(m)
                if p.is_dir():
                    if args.recursive:
                        for pp in p.rglob("*"):
                            if pp.is_file() and pp.suffix.lower() in exts:
                                files.append(str(pp))
                else:
                    if p.suffix.lower() in exts:
                        files.append(str(p))
            continue

        p = Path(entry)
        if p.exists():
            if p.is_dir():
                if args.recursive:
                    for pp in p.rglob("*"):
                        if pp.is_file() and pp.suffix.lower() in exts:
                            files.append(str(pp))
                else:
                    LOG.warning(
                        "Directory passed without --recursive, skipping: %s", entry
                    )
            elif p.is_file():
                if p.suffix.lower() in exts or not args.recursive:
                    files.append(str(p))
        else:
            LOG.warning("Input path does not exist: %s", entry)

    # deduplicate while preserving order. Resolve paths (non-strict) first so
    # different textual forms (./x.mp4, ../dir/x.mp4, /abs/path/x.mp4) collapse.
    seen = set()
    unique_files = []
    for f in files:
        try:
            rf = str(Path(f).resolve(strict=False))
        except Exception:
            rf = os.path.abspath(f)
        if rf not in seen:
            seen.add(rf)
            unique_files.append(rf)
    files = unique_files

    if not files:
        LOG.error("No input files found after expanding inputs. Exiting.")
        sys.exit(1)

    quick_joined_path = None
    quick_join_orig_files = None
    quick_join_delete_on_exit = False

    # Handle --join: concatenate all inputs into a single temp file first
    if getattr(args, "join", False):
        if len(files) < 2:
            LOG.warning(
                "--join requires at least 2 input files, processing normally..."
            )
        else:
            LOG.info(f"Joining {len(files)} input files into a single video...")
            # Create a temporary concat file
            concat_path = None
            joined_tmp = None
            concat_path = None
            try:
                # Remember original inputs so we can delete them later if
                # requested and we decide to skip further processing.
                quick_join_orig_files = list(files)
                # Try to create the concat list next to the first input so users
                # can inspect it while the job runs. Fall back to system temp
                # directory on any failure (permissions, non-existent parent, etc.).
                try:
                    first_parent = Path(files[0]).parent
                    first_parent.mkdir(parents=False, exist_ok=True)
                    fd_path = first_parent / (
                        "compress_buddy_concat_"
                        + next(tempfile._get_candidate_names())
                        + ".txt"
                    )
                    # Use low-level open to mimic mkstemp behavior (descriptor + path)
                    concat_fd = os.open(
                        str(fd_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                    )
                    concat_path = str(fd_path)
                except Exception:
                    concat_fd, concat_path = tempfile.mkstemp(
                        suffix=".txt", prefix="compress_buddy_concat_"
                    )
                LOG.info(f"Writing concat list to: {concat_path}")
                with os.fdopen(concat_fd, "w") as cf:
                    for f in files:
                        # ffmpeg concat demuxer requires absolute paths
                        abs_path = Path(f).resolve()
                        ap = str(abs_path)
                        # If the path contains a single quote, use double quotes around it
                        if "'" in ap:
                            cf.write(f'file "{ap}"\n')
                        else:
                            cf.write(f"file '{ap}'\n")

                # Determine output name for joined file
                if getattr(args, "output_file", None):
                    joined_path = Path(args.output_file)
                    out_dir = joined_path.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                elif getattr(args, "output_is_dir", False):
                    out_dir = Path(args.output)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    joined_name = "joined" + "." + args.suffix
                    joined_path = out_dir / joined_name
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
                    delete=False,
                ) as tf:
                    joined_tmp = Path(tf.name)

                LOG.info(f"Concatenating to: {joined_path}")

                if not args.dry_run:
                    # Compute total duration (sum of input durations) so progress can be shown
                    total_duration = 0.0
                    try:
                        for f in quick_join_orig_files:
                            try:
                                _, dur, _, _ = compute_bitrate_and_duration(
                                    Path(f), args
                                )
                                total_duration += float(dur or 0.0)
                            except Exception:
                                LOG.debug(f"Could not determine duration for {f}")
                    except Exception:
                        total_duration = None

                    # Run ffmpeg concat with progress so we can show ETA/%%/speed
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
                        "-progress",
                        "pipe:1",
                        "-nostats",
                        str(joined_tmp),
                    ]

                    rc, _, stderr_text, _, _, _ = run_ffmpeg_with_progress(
                        concat_cmd, total_duration=total_duration
                    )

                    if rc != 0:
                        LOG.error(
                            "Copy-based concat failed, aborting join. You may want to try re-encoding the files first."
                        )
                        LOG.debug(f"ffmpeg stderr:\n{stderr_text}")
                        try:
                            if joined_tmp and joined_tmp.exists():
                                joined_tmp.unlink(missing_ok=True)
                        except Exception:
                            pass
                        sys.exit(1)

                    # Move to final location (respect --overwrite)
                    if joined_path.exists() and not args.overwrite:
                        LOG.error(
                            f"{joined_path.name} exists and --overwrite not set. Aborting join."
                        )
                        try:
                            if joined_tmp and joined_tmp.exists():
                                joined_tmp.unlink(missing_ok=True)
                        except Exception:
                            pass
                        sys.exit(1)
                    os.replace(joined_tmp, joined_path)
                    LOG.info(f"Successfully joined videos into: {joined_path}")

                    # Now process the joined file (by default we plan to process it)
                    files = [str(joined_path)]
                    quick_joined_path = Path(joined_path)
                    # By default, mark this as an intermediate we will delete on exit
                    quick_join_delete_on_exit = True
                else:
                    LOG.info(
                        f"(dry-run) Would join {len(files)} files into {joined_path}"
                    )
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
                if getattr(args, "skip_processing", False):
                    continue
                try:
                    process_file(f, args)
                except Exception as e:
                    LOG.error(f"Exception processing {f}: {e}.")
    except KeyboardInterrupt:
        LOG.error("\nKeyboard interrupt received, stopping program.")
        sys.exit(130)
    finally:
        # Clean up joined intermediate if created (it's an internal temp)
        try:
            if (
                quick_joined_path
                and quick_joined_path.exists()
                and quick_join_delete_on_exit
            ):
                quick_joined_path.unlink(missing_ok=True)
                LOG.info(f"Removed temporary joined file: {quick_joined_path}")
        except Exception:
            pass


def arg_parse(argv):
    p = argparse.ArgumentParser(description="Compress Buddy helps you compress videos")
    p.add_argument("files", nargs="+", help="input files")
    p.add_argument(
        "--mode",
        choices=("hardware", "crf", "software"),
        default=None,
        help="encode mode",
    )
    p.add_argument(
        "--quality",
        type=int,
        default=None,
        help="Quality value, 1 being worst and 100 being best",
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
        default=None,
        help="codec to target (h264/avc/compatibility or h265/hevc/size). 'h264' for compatibility, 'h265' for better compression.",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Force exact ffmpeg encoder token to use.",
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
        default=None,  # Default loaded from USER_CONFIG in process_file
        help="Fraction of source bitrate to target (0.0 < factor <= 1.0)",
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
        "--chunk-seconds",
        type=int,
        default=0,
        help="split output into N-second chunks (preferred over minutes)",
    )
    p.add_argument(
        "--join",
        action="store_true",
        help="join all input videos into a single file before processing",
    )
    p.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip processing of files (useful with --join to only join files)",
    )
    p.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recurse into directories to find input files",
    )
    p.add_argument(
        "--extensions",
        type=str,
        default="mp4,mov,mkv,webm,avi,m4v,mpg,mpeg,ts",
        help="Comma-separated file extensions to include when recursing (no dots).",
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
        default=None,
        help="output file suffix (default in config)",
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
        "--no-bell",
        action="store_true",
        help="Do not ring terminal bell on completion",
    )
    p.add_argument(
        "--auto-map",
        action="store_true",
        help="Use ffmpeg's automatic stream selection (do not add explicit -map flags)",
    )
    p.add_argument(
        "--motion-multiplier",
        type=float,
        default=None,
        help="Specify motion multiplier. >1.0 increases bitrate for high-motion videos, <1.0 decreases for low-motion.",
    )

    p.add_argument(
        "--high-only",
        action="store_true",
        help="Only compress files judged as 'high' bitrate",
    )

    p.add_argument(
        "--bit",
        type=int,
        choices=(8, 10),
        default=None,
        help="Output bit depth, 8 or 10. If omitted, defaults from config (default_bit_depth).",
    )

    args = p.parse_args(argv)

    # Set bit depth default from USER_CONFIG if not provided on CLI
    try:
        if getattr(args, "bit", None) is None:
            args.bit = int(USER_CONFIG.get("default_bit_depth", 10))
        if args.bit not in (8, 10):
            raise ValueError("--bit must be 8 or 10")
    except Exception:
        p.error("--bit must be 8 or 10")

    try:
        if getattr(args, "mode", None) is None:
            args.mode = str(USER_CONFIG.get("preferred_mode", "hardware")).lower()
            if args.mode not in ("hardware", "crf", "software"):
                raise ValueError()
    except Exception:
        p.error("Invalid default_mode in config, must be hardware, crf, or software")

    try:
        if getattr(args, "codec", None) is None:
            args.codec = str(USER_CONFIG.get("preferred_codec", "size")).lower()
            if args.codec not in (
                "h264",
                "h265",
                "avc",
                "hevc",
                "x264",
                "x265",
                "size",
                "compatibility",
            ):
                raise ValueError()
    except Exception:
        p.error("Invalid preferred_codec in config")

    # Validate target factor only if provided on the CLI. We intentionally do
    # not populate it from config here so the code can prefer the heuristic
    # `suggested_kbps` when the user did not explicitly pass --target-factor.
    if args.target_factor is not None:
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

    if args.motion_multiplier is not None:
        try:
            mm = float(args.motion_multiplier)
            if mm <= 0:
                p.error("--motion-multiplier must be > 0")
        except Exception:
            p.error("--motion-multiplier must be a number")

    if args.workers <= 0:
        p.error("--workers must be a positive integer")

    if args.quality is not None:
        try:
            q = int(args.quality)
            if q < 0 or q > 100:
                p.error("--quality must be between 0 and 100")
        except Exception:
            p.error("Invalid --quality value")

    # Pull default from config if suffix not provided
    try:
        if getattr(args, "suffix", None) is None:
            args.suffix = str(USER_CONFIG.get("suffix", "mp4"))
    except Exception:
        args.suffix = USER_CONFIG.get("suffix", "mp4")

    # Normalize and validate `--output` semantics:
    # Rules implemented:
    # 1) Trailing slash -> treat as directory
    # 2) If an extension is present (Path.suffix) -> treat as a file
    # 3) If an extension is provided and --suffix is provided, they must match
    # 4) If no extension, no --suffix, and no trailing slash -> assume folder
    if getattr(args, "output", None):
        out_str = str(args.output)
        is_trailing = out_str.endswith(os.sep) or out_str.endswith("/")
        out_path = Path(out_str)
        has_ext = bool(out_path.suffix)

        if is_trailing:
            # explicit directory
            args.output_is_dir = True
            # keep args.output as provided (string)
        elif has_ext:
            # explicit file
            ext = out_path.suffix.lstrip(".").lower()
            if getattr(args, "suffix", None) and ext != str(args.suffix).lower():
                p.error(
                    f"--output extension .{ext} conflicts with --suffix {args.suffix}"
                )
            args.output_file = str(out_path)
            args.output_is_dir = False
        else:
            # no suffix in provided output
            if getattr(args, "suffix", None):
                # treat as a filename without extension: append suffix
                out_with_ext = out_path.with_suffix("." + str(args.suffix))
                args.output_file = str(out_with_ext)
                args.output_is_dir = False
            else:
                # ambiguous: default to directory per rule (4)
                args.output_is_dir = True

    # If user provided multiple input files but specified a single output file
    # (not a directory) and did not request a join, that's an error.
    if getattr(args, "output_file", None) and len(getattr(args, "files", [])) > 1:
        if not getattr(args, "join", False):
            p.error(
                "--output targets a single file but multiple input files were provided. Use --join or specify an output directory."
            )

    return args


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        LOG.error("Interrupted by user, exiting program.")
        sys.exit(130)

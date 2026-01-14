#!/usr/bin/env python3
"""
compress_buddy.py - Compress video files using hardware acceleration or CRF encoding.

Usage:
    python3 compress_buddy.py [options] files...

Options:
    --mode hardware|crf            Encode mode (hardware or CRF)
    --quality N                    Quality value (0-100)
    --encoder h264|h265            Encoder to use (h264 for compatibility, h265 for better compression)
    --workers N                    Parallel workers (only for CRF mode)
    --min-kbps N                   Minimum target bitrate in kbps (default 3000)
    --max-kbps N                   Maximum target bitrate in kbps (default 8000)
    --dry-run                      Show actions but don't run ffmpeg
    --overwrite                    Overwrite existing files
    --copy-audio                   Copy AAC audio instead of re-encoding
    --suffix mp4|mov|mkv           Output file suffix (default mov)
    --scale                        Scale video down to max 1920x1080
    --delete-original              Delete original after successful encode
    --output /path/to/outdir       Place converted files into this directory
    --chunk-minutes N              Split output into N-minute chunks

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

"""
import argparse
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

# logging setup: try to use rich for pretty output, fall back to basic logging
try:
    from rich.logging import RichHandler

    def setup_logging(use_rich=True):
        fmt = "[%(asctime)s] %(levelname)s %(message)s"
        # include numeric timezone offset
        datefmt = "%Y/%m/%d %H:%M:%S %z"
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        global RICH_MARKUP_ENABLED
        if use_rich:
            handler = RichHandler(rich_tracebacks=True)
            try:
                # RichHandler may manage formatting internally; try to set formatter
                handler.setFormatter(formatter)
            except Exception:
                pass
            # Rich supports markup; enable tag passthrough
            try:
                setattr(handler, "rich_markup", True)
                RICH_MARKUP_ENABLED = True
            except Exception:
                RICH_MARKUP_ENABLED = True
            logging.basicConfig(level=logging.INFO, handlers=[handler])
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            RICH_MARKUP_ENABLED = False
            logging.basicConfig(level=logging.INFO, handlers=[handler])

except Exception:
    RichHandler = None

    def setup_logging(use_rich=False):
        fmt = "[%(asctime)s] %(levelname)s %(message)s"
        datefmt = "%Y/%m/%d %H:%M:%S %z"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        global RICH_MARKUP_ENABLED
        RICH_MARKUP_ENABLED = False
        logging.basicConfig(level=logging.INFO, handlers=[handler])


LOG = logging.getLogger("compress_buddy")

# whether the configured logging handler supports Rich markup rendering
RICH_MARKUP_ENABLED = False


def format_rich(text: str) -> str:
    """Return `text` unchanged when the configured logging handler supports Rich markup.

    Otherwise strip Rich-style tags so raw logs don't contain markup markers.
    """
    if RICH_MARKUP_ENABLED:
        return text
    return re.sub(r"\[/?[^\]]+\]", "", text)


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
            "ffmpeg and ffprobe must be available on PATH. See README for install instructions."
        )
        sys.exit(1)


def get_ffmpeg_hwaccels():
    """Return a set of hwaccel names reported by `ffmpeg -hwaccels`.

    Returns empty set on error.
    """
    try:
        res = run_cmd(["ffmpeg", "-hide_banner", "-hwaccels"])
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
        res = run_cmd(["ffmpeg", "-hide_banner", "-encoders"])
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

    for name, hw in candidates:
        if name in encoders:
            # if encoder requires a hwaccel, ensure it's present (except nvenc which doesn't need -hwaccel)
            if hw and hw not in hwaccels:
                continue
            return name, hw
    return None, None


def run_cmd(cmd):
    LOG.debug(format_rich(f"CMD: {format_cmd_for_logging(cmd)}"))
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


def compute_bitrate_and_duration(path):
    info = ffprobe_json(path)
    duration = float(info.get("format", {}).get("duration") or 0.0)
    # prefer first video stream bit_rate
    bitrate = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and s.get("bit_rate"):
            bitrate = int(s["bit_rate"])
            break
    if not bitrate:
        # fallback: use file size * 8 / duration (bps)
        size_bytes = os.path.getsize(path)
        if duration > 0:
            bitrate = int(size_bytes * 8 / duration)
        else:
            bitrate = 5_000_000  # arbitrary fallback 5Mbps
    return bitrate, duration, info


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


def run_ffmpeg_with_progress(cmd, args):
    """
    Run ffmpeg with '-progress pipe:1 -nostats' already present in cmd.
    Streams stdout, parses 'out_time' progress keys, and computes speed = out_time_secs / elapsed_secs.
    Returns (returncode, stdout_text, stderr_text, final_speed)
    """
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    final_out_time = 0.0
    stdout_lines = []
    stderr_lines = []

    try:
        while True:
            out_line = proc.stdout.readline()
            LOG.debug(format_rich(f"ffmpeg stdout: [bold]{out_line.strip()}[/bold]"))
            if out_line:
                stdout_lines.append(out_line)
                line = out_line.strip()
                if "=" in line:
                    key, val = line.split("=", 1)
                    if key == "out_time":
                        final_out_time = parse_out_time(val.strip())
                continue
            if proc.poll() is not None:
                # drain remaining stdout
                for out_line in proc.stdout:
                    stdout_lines.append(out_line)
                    line = out_line.strip()
                    if "=" in line:
                        key, val = line.split("=", 1)
                        if key == "out_time":
                            final_out_time = parse_out_time(val.strip())
                break
            # drain a bit of stderr to avoid blocking if ffmpeg writes a lot
            err_chunk = proc.stderr.readline()
            if err_chunk:
                stderr_lines.append(err_chunk)
                LOG.debug(
                    format_rich(f"ffmpeg stderr: [bold]{err_chunk.strip()}[/bold]")
                )
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

    return proc.returncode, "".join(stdout_lines), "".join(stderr_lines), final_speed


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
        LOG.warning("%s exists, skipping", out.name)
        return

    bitrate, duration, probe = compute_bitrate_and_duration(inp)
    source_kbps = bitrate / 1000.0
    target_kbps = int(max(args.min_kbps, min(args.max_kbps, source_kbps * 0.65)))
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

        # video settings
        if args.scale:
            vf = "scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease"
            cmd += ["-vf", vf]

        if args.mode == "crf":
            cmd += [
                "-c:v",
                f"lib{args.encoder.replace('h', 'x')}",
                "-preset",
                "veryslow",
                "-crf",
                str(args.quality // 2),
            ]
        else:
            # If args.encoder already contains a concrete encoder name (e.g., nvenc, qsv), use it as-is;
            # otherwise, attempt to use platform-specific hardware encoder naming.
            enc = args.encoder
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

        # audio handling: prefer copying AAC if requested/available, otherwise encode to AAC
        if has_audio:
            if args.copy_audio or (
                audio_codec_name and audio_codec_name.lower() == "aac"
            ):
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "128k"]

        # always request faststart so progressive download / MOV compatibility is set
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
        ffmpeg_command_message += f"[bold]{cmd_str}[/bold]"
        LOG.info(format_rich(ffmpeg_command_message))

        rc, _, stderr_text, speed = run_ffmpeg_with_progress(cmd, args)
        if rc != 0:
            err = stderr_text.strip().splitlines()
            tail = err[-10:] if err else ["<no stderr>"]
            LOG.error(format_rich(f"ffmpeg failed for {inp.name}: {'\\n'.join(tail)}"))
            try:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            LOG.debug(format_rich(f"Full ffmpeg stderr:\n{stderr_text}"))
            LOG.debug(
                format_rich(
                    f"Full ffmpeg cmd: [bold]{' '.join(shlex.quote(x) for x in cmd)}[/bold]"
                )
            )
            return

        if chunking:
            # move all generated segments from tmp_dir to final names in out.parent
            seg_files = sorted(tmp_dir.glob(inp.stem + ".*" + out.suffix))
            if not seg_files:
                LOG.error(
                    format_rich(f"No segments produced for [bold]{inp.name}[/bold]")
                )
            for idx, sf in enumerate(seg_files, start=1):
                final_name = f"{inp.stem}_part{idx:03d}{out.suffix}"
                final_path = out.parent / final_name
                if final_path.exists() and not args.overwrite:
                    LOG.warning(
                        format_rich(f"[bold]{final_path.name}[/bold] exists, skipping")
                    )
                    continue
                os.replace(sf, final_path)
                LOG.info(
                    format_rich(
                        f"Created [bold]{final_path.name}[/bold] ({final_path.stat().st_size / 1024 / 1024:.1f} MB)"
                    )
                )
        else:
            # Atomic replace
            os.replace(tmp_path, out)
            LOG.info(
                format_rich(
                    f"Created [bold]{out.name}[/bold] ({out.stat().st_size / 1024 / 1024:.1f} MB)"
                )
            )
        LOG.info(format_rich(f"Encode speed: [bold]{speed:.2f}x realtime[/bold]"))
        if args.delete_original:
            inp.unlink(missing_ok=True)
            LOG.info(format_rich(f"Deleted original file [bold]{inp.name}[/bold]"))
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

    if args.mode == "hardware":
        if platform.system() == "Darwin":
            if args.workers != 1:
                LOG.info(
                    "Hardware mode on macOS detected — capping workers to 1 for VideoToolbox stability"
                )
            args.workers = 1
    if args.mode == "hardware" and args.encoder == "h265":
        args.encoder = "hevc"  # ffmpeg uses 'hevc' for h265 when using hwaccel
    # If hardware mode requested, attempt to pick a suitable hw encoder if available
    if args.mode == "hardware":
        # try to auto-select best available hw encoder
        chosen, hwaccel = choose_best_hw_encoder(args.encoder)
        if chosen:
            LOG.info("Auto-selected hardware encoder %s (hwaccel=%s)", chosen, hwaccel)
            # Use the exact encoder token returned by ffmpeg (e.g. 'hevc_nvenc').
            args.encoder = chosen
            # if a hwaccel is required, store it on args for later use
            setattr(args, "_hwaccel", hwaccel)
        else:
            LOG.warning(
                "No suitable hardware encoder found; falling back to software/CRF if requested"
            )
    if args.quality and args.mode == "crf":
        LOG.info(f"Dividing quality {args.quality} by 2 for CRF")
    if args.mode == "crf" and not args.quality:
        args.quality = 28 * 2  # default CRF value doubled
    if args.quality and (args.quality < 0 or args.quality > 100):
        LOG.error("Quality must be between 0 and 100")
        sys.exit(1)

    # configure logging
    setup_logging()
    LOG.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    ensure_ffmpeg_available(getattr(args, "dry_run", False))

    files = args.files
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_file, f, args): f for f in files}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    LOG.error(
                        format_rich(
                            f"Exception processing [bold]{futures[fut]}[/bold]: {e}"
                        )
                    )
    else:
        for f in files:
            try:
                process_file(f, args)
            except Exception as e:
                LOG.error(format_rich(f"Exception processing [bold]{f}[/bold]: {e}"))


def arg_parse(argv):
    p = argparse.ArgumentParser(description="Compress Buddy helps you compress videos")
    p.add_argument("files", nargs="+", help="input files")
    p.add_argument(
        "--mode", choices=("hardware", "crf"), default="hardware", help="encode mode"
    )
    p.add_argument("--quality", type=int, default=None, help="Quality value (0-100)")
    p.add_argument(
        "--encoder", choices=("h264", "h265"), default="h265", help="encoder name"
    )
    p.add_argument("--min-kbps", type=int, default=3000, help="min target kbps")
    p.add_argument("--max-kbps", type=int, default=8000, help="max target kbps")
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
        choices=("mp4", "mov", "mkv"),
        default="mov",
        help="output file suffix (default mov)",
    )
    p.add_argument("--scale", action="store_true", help="scale video to max 1920x1080")
    p.add_argument(
        "--delete-original",
        action="store_true",
        help="delete original file after successful compression",
    )
    args = p.parse_args(argv)
    return args


if __name__ == "__main__":
    main(sys.argv[1:])

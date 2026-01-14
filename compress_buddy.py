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


def get_ffmpeg_decoders():
    """Return a set of decoder names reported by `ffmpeg -decoders`. Returns empty set on error."""
    try:
        res = run_cmd(["ffmpeg", "-hide_banner", "-decoders"])
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
                    "Found %s encoder but libcuda not available at runtime; skipping nvenc",
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
    LOG.debug(f"CMD: {format_cmd_for_logging(cmd)}")
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
            LOG.info("ffmpeg will be started with POSIX nice=%s", nice_val)
        except Exception:
            LOG.debug("Invalid nice value provided; ignoring")

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
                LOG.debug("Set process niceness via psutil to %s", args.nice)
            except Exception:
                # On Windows, map positive niceness to BELOW_NORMAL/IDLE classes
                try:
                    if platform.system() == "Windows":
                        if int(args.nice) >= 10:
                            p.nice(psutil.IDLE_PRIORITY_CLASS)
                        else:
                            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                        LOG.debug(
                            "Set Windows process priority via psutil (mapped from nice %s)",
                            args.nice,
                        )
                except Exception:
                    LOG.debug(
                        "psutil could not set niceness/priority for pid %s", proc.pid
                    )
        except Exception:
            # psutil not available — we've already attempted preexec_fn on POSIX; on Windows we can't set priority.
            if platform.system() == "Windows":
                LOG.warning(
                    "psutil not installed; cannot lower ffmpeg process priority on Windows. Install 'psutil' to enable this feature."
                )
    final_out_time = 0.0
    stdout_lines = []
    stderr_lines = []

    try:
        while True:
            out_line = proc.stdout.readline()
            LOG.debug(f"ffmpeg stdout: {out_line.strip()}")
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
                    "Input codec '%s' appears to lack a hardware decoder on this system; this may limit encode speed (software decode).",
                    in_video_codec,
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

        # video settings
        if args.scale:
            vf = "scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease"
            cmd += ["-vf", vf]

        if args.mode == "crf":
            cmd += [
                "-c:v",
                f"lib{args.codec.replace('h', 'x')}",
                "-preset",
                "veryslow",
                "-crf",
                str(args.quality // 2),
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

        rc, _, stderr_text, speed = run_ffmpeg_with_progress(cmd, args)
        if rc != 0:
            err = stderr_text.strip().splitlines()
            tail = err[-10:] if err else ["<no stderr>"]
            LOG.error(f"ffmpeg failed for {inp.name}: {'\\n'.join(tail)}")
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
                LOG.error(f"No segments produced for {inp.name}")
            for idx, sf in enumerate(seg_files, start=1):
                final_name = f"{inp.stem}_part{idx:03d}{out.suffix}"
                final_path = out.parent / final_name
                if final_path.exists() and not args.overwrite:
                    LOG.warning(f"{final_path.name} exists, skipping")
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
                    "Requested forced encoder '%s' not available in this ffmpeg build.",
                    forced,
                )
                LOG.error(
                    "Available encoders: %s",
                    ", ".join(sorted(list(encs))[:40]) or "<none>",
                )
                sys.exit(1)
            # if forcing nvenc, ensure runtime libcuda is present
            if "nvenc" in forced and not nvenc_runtime_available():
                LOG.error(
                    "Forced encoder '%s' requires NVENC but libcuda cannot be loaded at runtime. Aborting.",
                    forced,
                )
                sys.exit(1)
            LOG.info("Using forced hardware encoder %s", forced)
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
                    "Auto-selected hardware encoder %s (hwaccel=%s)", chosen, hwaccel
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
                    "Detected encoders (sample): %s", ", ".join(encs[:40]) or "<none>"
                )
                LOG.error("Detected hwaccels: %s", ", ".join(hw) or "<none>")
                LOG.error(
                    "Detected decoders (sample): %s", ", ".join(decs[:40]) or "<none>"
                )
                sys.exit(1)
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
                    LOG.error(f"Exception processing [bold]{futures[fut]}[/bold]: {e}")
    else:
        for f in files:
            try:
                process_file(f, args)
            except Exception as e:
                LOG.error(f"Exception processing {f}: {e}")


def arg_parse(argv):
    p = argparse.ArgumentParser(description="Compress Buddy helps you compress videos")
    p.add_argument("files", nargs="+", help="input files")
    p.add_argument(
        "--mode", choices=("hardware", "crf"), default="hardware", help="encode mode"
    )
    p.add_argument("--quality", type=int, default=None, help="Quality value (0-100)")
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
        help="codec to target (h264/avc or h265/hevc). Accepts common synonyms",
    )
    p.add_argument(
        "--force-encoder",
        type=str,
        default=None,
        help="Force exact ffmpeg encoder token to use (e.g. hevc_videotoolbox, h264_nvenc). Overrides auto-selection.",
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
        choices=("mp4", "mov", "mkv", "avi"),
        default="mov",
        help="output file suffix (default mov)",
    )
    p.add_argument("--scale", action="store_true", help="scale video to max 1920x1080")
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
    args = p.parse_args(argv)
    return args


if __name__ == "__main__":
    main(sys.argv[1:])

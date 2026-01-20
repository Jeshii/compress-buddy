# compress-buddy
An ffmpeg wrapper that helps you navigate the flags to save some space by compressing video.

## Requirements

- Python 3.8+ (uses only the standard library)
- ffmpeg and ffprobe installed and available on `PATH`

On macOS, hardware acceleration uses VideoToolbox (`videotoolbox`) which requires an ffmpeg build with VideoToolbox support (brew-installed ffmpeg typically includes this). I also noticed that ffmpeg has started requiring ffmpeg-full formula to be installed to get h265 support.

## Quick Setup

1. Clone the repo:

```bash
git clone https://github.com/Jeshii/compress-buddy.git
cd compress-buddy
```

2. Verify `ffmpeg` and `ffprobe` are available and list encoders you may need:

```bash
command -v ffmpeg && command -v ffprobe
ffmpeg -encoders | grep -E "libx264|libx265|hevc_videotoolbox|h264_videotoolbox|videotoolbox"
```

## Usage

Basic:

```bash
python3 compress_buddy.py video.mov
```

Place converted files into an output directory:

```bash
python3 compress_buddy.py -o /tmp/outdir video.mov
```

Dry run (no ffmpeg executed, just prints decisions):

```bash
python3 compress_buddy.py --dry-run -o /tmp/outdir video.mov
```

CRF mode with 4 workers (when you hate your computer):

```bash
python3 compress_buddy.py --mode crf --quality 28 --workers 4 *.mp4
```

Chunk output into 15-minute parts:

```bash
python3 compress_buddy.py --chunk-minutes 15 -o /tmp/outdir long_video.mov
```

Join multiple inputs:

```bash
# process multiple inputs as a single logical input (one encode pass)
python3 compress_buddy.py --join -o /tmp/outdir part1.mov part2.mov

# quick-join: perform a fast copy-based concat (-c copy), fail if incompatible, then process the joined result
python3 compress_buddy.py --quick-join -o /tmp/outdir part1.mov part2.mov
```

## Options

General
- `-h`, `--help` — show this help message and exit.
- `--dry-run` — show decisions and planned actions, do not run ffmpeg.
- `--log-level <DEBUG|INFO|WARNING|ERROR>` — set logging verbosity (default: INFO).
- `--no-bell` — suppress terminal bell on completion.

Encoding / codec selection
- `--mode {hardware,crf,software}` — encode mode. `hardware` uses platform HW accel, `crf` uses constant-rate-factor software encoding.
- `--quality <0-100>` — user-facing quality (0 worst, 100 best). In CRF mode we map this to ffmpeg's CRF scale.
- `--codec {h264,h265,avc,hevc,x264,x265,size,compatibility}` — high-level codec choice (we normalize to `h264` / `h265`).
- `--force-encoder <TOKEN>` — force an exact ffmpeg encoder token (e.g. `hevc_videotoolbox`, `h264_nvenc`). We validate the token against `ffmpeg -encoders`.
- `--suffix {mp4,mov,mkv,avi}` — output file suffix (default: `mov`).

Bitrate / quality targeting
- `--min-kbps <N>` — minimum target bitrate in kbps (optional). If ffprobe cannot determine source bitrate, we will ask for bounds or abort.
- `--max-kbps <N>` — maximum target bitrate in kbps (optional).
- `--target-factor <float>` — fraction of source bitrate to target (0.0 < factor <= 1.0). Default: `0.7`.

Audio
- `--copy-audio` — copy AAC audio streams instead of re-encoding.
- If not copying and input audio is not AAC, we encode audio to AAC at 128k.

Scaling / dimensions
- `--max-width <px>` — maximum output width in pixels (preserves aspect ratio).
- `--max-height <px>` — maximum output height in pixels (preserves aspect ratio).
- If only one of `--max-width`/`--max-height` is provided, we infer the other from the input aspect ratio.

Chunking / output
- `--output, -o <DIR>` — output directory (place converted files here).
- `--chunk-minutes <N>` — split output into N-minute segments. Segments are written to a temporary directory next to the output and then moved into place after encoding.
- `--overwrite` — overwrite existing outputs.
- `--delete-original` — delete original file after successful compression.

Performance / resource control
- `--workers <N>` — number of parallel workers (we force `1` for macOS hardware mode for stability).
- `--threads <N>` — pass `-threads N` to ffmpeg to bound encoder threads.

Motion analysis (optional)
- `--sample-rate <seconds>` — sample the video every N seconds for motion analysis (default: `10`).
- `--skip-motion-analysis` — do not run motion analysis even when auto-enabled.
- `--motion-threshold-seconds <seconds>` — only run motion analysis for videos with duration >= this value (default: `120`).
- `--motion-multiplier <float>` — manually specify motion multiplier to override automatic analysis (e.g. `1.2`).

Other
- `--target-factor` — see Bitrate / quality targeting above.
- `--copy-audio` — see Audio above.

Notes
- We clamp computed target bitrates to `--min-kbps` / `--max-kbps` if provided, and never intentionally target a bitrate higher than the source (as reported by ffprobe).
- If ffprobe cannot determine a source bitrate and you do not provide bounds, the run will abort rather than guessing from file size.
- Use `--dry-run` to validate decisions without running ffmpeg.
- For Windows specifics (path handling, AV interference, long paths), see the Windows Notes section below.

## Other Notes & Gotchas

- ffprobe is used to check for bitrates on the source. We prefer per-stream `bit_rate` when available and fall back to the container-level `format.bit_rate` for formats like WebM/Matroska. The target bitrate is computed as `source_kbps * --target-factor` (default 0.7). `--min-kbps` and `--max-kbps` are optional and, if provided, clamp the computed target. If ffprobe cannot determine a bitrate at all the tool will ask you to provide `--min-kbps` or `--max-kbps` explicitly rather than guessing from file size.
- Quality mapping: `--quality` is a user-facing 0–100 scale, with 0 being worst and 100 being best. This is inverted in CRF mode automatically to keep things simple.
- macOS + VideoToolbox: hardware mode uses VideoToolbox and we force `--workers 1` on Darwin for stability.
- Audio handling: if the input audio is AAC or you pass `--copy-audio`, audio will be copied. Otherwise it will be re-encoded to AAC at 128k.
- Byte display: sizes shown by the tool use IEC binary units (`KiB`, `MiB`, `GiB`) which are based on 1024 bytes. This makes it explicit that values like `2.59 GiB` are using 1024-based math and explains apparent differences with Finder which often reports decimal GB.
- Bitrate cap: the computed target bitrate is capped to the original source bitrate (as reported by ffprobe) so we will never intentionally target a bitrate higher than the source. If ffprobe cannot determine the source bitrate, the tool will continue to ask you to provide explicit `--min-kbps`/`--max-kbps` or accept defaults.
- Chunking: segments are written to a temporary directory next to the output and moved into place after encoding. If the process fails mid-run, partial files may need manual cleanup.
- Destructive flags: `--overwrite` and `--delete-original` will replace or remove files — use with caution.

## Codec / Encoder selection

- The flag `--codec` accepts codec names and common synonyms. Use `--codec h264` or `--codec hevc`, the aliases `avc`, `x264`, `x265`, or even simply `size` (h265) or `compatibility` (h264).
- If you need to force a specific ffmpeg encoder token (for example, `hevc_videotoolbox`, `h264_nvenc`, or `hevc_qsv`), use `--force-encoder "<token>"`. We will validate the token exists in your `ffmpeg -encoders` output before using it.

Examples:

```bash
# use the codec name
python3 compress_buddy.py --codec hevc myvideo.mov

# force a specific ffmpeg encoder token
python3 compress_buddy.py --mode hardware --force-encoder hevc_videotoolbox myvideo.mov
```

## Windows Notes

- Verify ffmpeg/ffprobe availability:

```powershell
where.exe ffmpeg
where.exe ffprobe
```

- Check available hardware acceleration methods and encoders:

```powershell
ffmpeg -hwaccels
ffmpeg -encoders | findstr /R /C:"nvenc" /C:"qsv" /C:"d3d11va" /C:"dxva2"
```

- Are these some Windows hardware encoders names? Not sure!
	- NVIDIA: `h264_nvenc`, `hevc_nvenc`
	- Intel QSV: `h264_qsv`, `hevc_qsv`
	- Direct3D/DXVA: `h264_d3d11va`, `h264_dxva2`

- Path and file handling:
	- Windows prevents deleting files that are still open
	- Stuff may fail across different drives
	- Very long paths (>260 chars) can be problematic


- Quick test (PowerShell):
```powershell
# dry-run to confirm behavior without running ffmpeg
python compress_buddy.py --dry-run -o C:\tmp test.mp4

# real run (ensure ffmpeg.exe is on PATH)
python compress_buddy.py -o C:\tmp test.mp4
```

## Troubleshooting

- If you see errors about unknown encoders, try installing `ffmpeg-full`, install an ffmpeg build with the required encoders (e.g., `libx264`, `libx265`)
- To get more visibility into ffmpeg progress and script decisions, run with `--log-level DEBUG` (it is very VERBOSE tho)


## Limiting CPU usage and threads

If an encode still uses too much CPU even with `--nice`, you can further limit ffmpeg:

- `--threads N` passes `-threads N` to ffmpeg directly
- If you run with `--workers > 1` and do not specify `--threads`, we will auto-calculate a `threads` value per worker by dividing the logical CPU count by the number of workers (minimum 1)

Examples:

```bash
# limit encoder threads to 4
python3 compress_buddy.py --threads 4 myvideo.mov

```

## Contributing / Fixes

If you run into a bug, please open an issue with the command you ran and the `ffmpeg` version (output from `ffmpeg -version`). Or feel free to open a PR!
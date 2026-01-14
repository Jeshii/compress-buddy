# compress-buddy
An ffmpeg wrapper that helps you navigate the flags to save some space by compressing video.

## Requirements

- Python 3.8+ (script uses only the standard library)
- ffmpeg and ffprobe installed and available on `PATH`

On macOS, hardware acceleration uses VideoToolbox (`videotoolbox`) which requires an ffmpeg build with VideoToolbox support (brew-installed ffmpeg typically includes this).

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

Basic (in-place):

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

CRF mode with 4 workers (software encoding):

```bash
python3 compress_buddy.py --mode crf --quality 28 --workers 4 *.mp4
```

Chunk output into 15-minute parts:

```bash
python3 compress_buddy.py --chunk-minutes 15 -o /tmp/outdir long_video.mov
```

## Important Notes & Gotchas

- Quality mapping: `--quality` is a user-facing 0–100 scale. When using `--mode crf`, the script divides the provided `--quality` by 2 and uses that result as the CRF value passed to ffmpeg. Internally, the *default* CRF is stored as a doubled value (e.g., `56` represents a user quality of `28`), but you should always pass normal 0–100 values. For example, `--quality 28` results in `-crf 14` being sent to ffmpeg.
- macOS + VideoToolbox: hardware mode uses VideoToolbox and the script forces `--workers 1` on Darwin for stability.
- Audio handling: if the input audio is AAC or you pass `--copy-audio`, audio will be copied. Otherwise it will be re-encoded to AAC at 128k.
- Chunking: segments are written to a temporary directory next to the output and moved into place after encoding. If the process fails mid-run, partial files may need manual cleanup.
- Destructive flags: `--overwrite` and `--delete-original` will replace or remove files — use with caution.

## Codec / Encoder selection

- The flag `--codec` accepts high-level codec names and common synonyms. Use `--codec h264` or `--codec hevc` (or the familiar aliases `avc`, `x264`, `x265`). The script normalizes these to the internal canonical values `h264` or `h265`.
- If you need to force a specific ffmpeg encoder token (for example, `hevc_videotoolbox`, `h264_nvenc`, or `hevc_qsv`), use `--force-encoder "<token>"`. The script will validate the token exists in your `ffmpeg -encoders` output before using it.

Examples:

```bash
# use the high-level codec name (accepts synonyms)
python3 compress_buddy.py --codec hevc myvideo.mov

# force a specific ffmpeg encoder token (non-interactive)
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

- Common Windows hardware encoder names:
	- NVIDIA: `h264_nvenc`, `hevc_nvenc`
	- Intel QSV: `h264_qsv`, `hevc_qsv`
	- Direct3D/DXVA: `h264_d3d11va`, `h264_dxva2` (availability varies)

- Path and file handling:
	- Windows prevents deleting files that are still open; ensure no other process is holding files created by the script.
	- `os.replace()` may fail across different drives; the script creates temp files next to the target output to avoid this, but if a move fails it will fall back to `shutil.move()` with a warning.
	- Very long paths (>260 chars) can be problematic; consider enabling long path support in Windows or using UNC `\\?\` prefixes.

- Antivirus / real-time scanning:
	- AV can block or slow temp files during heavy ffmpeg I/O. If encodes fail or are very slow, try temporarily excluding the working/output folder from real-time scanning.

- WSL (Windows Subsystem for Linux):
	- If you run the script inside WSL, prefer operating entirely inside the WSL filesystem (e.g., `/home`) rather than crossing into `C:\` mounts for heavy I/O.
	- Alternatively, install a native Windows `ffmpeg.exe` and run the script from PowerShell for best performance.

- Quick test (PowerShell):
```powershell
# dry-run to confirm behavior without running ffmpeg
python compress_buddy.py --dry-run -o C:\tmp test.mp4

# real run (ensure ffmpeg.exe is on PATH)
python compress_buddy.py -o C:\tmp test.mp4
```

## Troubleshooting

- If you see errors about unknown encoders, install an ffmpeg build with the required encoders (e.g., `libx264`, `libx265`) or use your system package manager to get a fuller ffmpeg build.
- To get more visibility into ffmpeg progress and script decisions, run with `--log-level DEBUG`.

## Running ffmpeg at lower priority (`--nice`)

You can request the child ffmpeg process be started with a lower CPU priority using `--nice N`. This helps keep encodes from interfering with interactive work.

- Suggested niceness values:
	- `5` — light background priority (small deprioritization)
	- `10` — typical background priority
	- `15` — very low priority (best for long unattended batch jobs)

- Notes:
	- On POSIX systems (macOS, Linux), `--nice` attempts to start ffmpeg with the requested niceness. Higher numbers mean lower priority.
	- On Windows lowering process priority requires the optional `psutil` Python package. Install it with `pip install psutil` to enable priority lowering on Windows.

Verification:

On macOS/Linux you can verify niceness with:

```bash
# find the ffmpeg PID (example) and show niceness
ps -o pid,ni,cmd -C ffmpeg

# or watch during run
top -o %CPU
```

On Windows, use Task Manager or PowerShell to inspect process priority:

```powershell
# list ffmpeg processes
Get-Process ffmpeg | Select-Object Id,ProcessName,PriorityClass
```


## Contributing / Fixes

If you run into a bug, please open an issue with the command you ran and the `ffmpeg` version (output from `ffmpeg -version`).
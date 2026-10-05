# Agent Instructions — compress-buddy

This repository is a single-file CLI tool that wraps ffmpeg/ffprobe to compress video files. The idea is to save space on the user's device by efficiently encoding videos. The instructions below highlight repository-specific patterns, conventions, and developer workflows that help an AI coding agent make productive, safe edits.

## Quick Overview
- **Entry point**: compress_buddy.py - single Python script implementing the entire CLI.
- **External dependencies**: `ffmpeg` and `ffprobe` must be available in `PATH` (required at runtime).
- **Primary responsibilities**: probe input files, choose encoding parameters, spawn `ffmpeg` with `-progress pipe:1`, parse progress, and atomically move outputs into place.

## Important Implementation Details
- **Modes**: `--mode` supports `hardware` and `crf`. Hardware uses platform HW accel (macOS: VideoToolbox). CRF uses software libx264/libx265.
  - CRF mapping: the code uses `f"lib{args.encoder.replace('h','x')}"` (see where `-c:v` is set).
  - When `--mode hardware` and `--encoder h265`, the script maps to `hevc` for ffmpeg (see `main()` mapping).
- **Quality handling**: For CRF the script divides provided `--quality` by 2 before passing to ffmpeg; default CRF is `28` (the script stores it as `28*2` initially). Quality must be 0–100.
- **Bitrate targeting**: When not using CRF and no `--quality`, the script computes target bitrate from source (`source_kbps * 0.65`) bounded by `--min-kbps` and `--max-kbps`.
- **Audio handling**: If input audio codec is AAC or `--copy-audio` is passed, audio is copied; otherwise audio is encoded to AAC at `128k`.
- **Scaling**: `--scale` adds a `-vf scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease` filter.
- **Chunking**: `--chunk-minutes N` uses ffmpeg's `segment` muxer and writes parts to a temporary directory before moving them into the final output directory. Final names are `STEM_partNNN.suffix`.
- **Atomic output**: Single-file outputs are written to a named temp file in the same directory then `os.replace()`-ed into final name for atomicity. For chunking, segments are generated in a tmp dir and moved.
- **Progress parsing**: `-progress pipe:1 -nostats` is used; progress is parsed in `run_ffmpeg_with_progress()` by reading `out_time` and computing speed. See `parse_out_time()` and `run_ffmpeg_with_progress()`.
- **Concurrency**: `--workers N` uses `ThreadPoolExecutor` for parallel processing, but `main()` forces `args.workers = 1` for hardware mode on macOS (VideoToolbox instability).

## Where To Make Common Changes
- Change encoder behavior (mappings, added encoders): edit `main()` and the video settings block in `process_file()` around `-c:v` selection (search for `-c:v` and `videotoolbox`).
- Change progress or performance reporting: edit `run_ffmpeg_with_progress()` and `parse_out_time()`.
- Change chunk naming or behavior: edit the chunking block in `process_file()` where `tmp_pattern` and segment move logic live.
- Add extra ffmpeg args: `process_file()` builds the command in `cmd` — modify there.

## Developer Commands / Common Workflows
- Run (process files): `python3 compress_buddy.py [options] file1.mov file2.mp4`
- Dry run (validate decisions, no ffmpeg calls): `python3 compress_buddy.py --dry-run -o /tmp/outdir myvideo.mov`
- Debug verbose output: add `--log-level DEBUG` to see debug logs and ffmpeg progress parsing.
- Example from README/header: `python3 compress_buddy.py --workers 4 --mode crf --quality 28 *.mp4`
- Recurse a directory tree: `python3 compress_buddy.py -r -o /tmp/outdir /path/to/videos`. `-r` is required for directories; without it the directory is skipped with a warning.
- Input expansion lives in `main()`: shell globs are expanded with `glob.glob(..., recursive=True)` and directories are walked with `Path.rglob("*")`, then filtered by `--extensions`. Change traversal behavior or the default extension set there, not in `process_file()`.
- `-o` flattens the tree: outputs use the source basename, so same-named files in different subdirectories collide and the later one is skipped. Omit `-o` to keep the tree structure.

## Safety, Tests, and Limitations
- Be careful when modifying atomic replacement and deletion logic — `process_file()` uses `os.replace()` and `unlink(missing_ok=True)` which are destructive when `--delete-original` or `--overwrite` are used.
- Recursive runs touch every matching file under the given path. Prefer verifying with `--dry-run` before a real run, especially alongside `--delete-original`.

## Readme tone
- When generating or updating README.md content, maintain a clear, concise, and user-friendly tone. Focus on practical examples and important notes to help users understand how to effectively use the tool. Always say "we" instead of "the tool" or "the script" to create a more engaging and inclusive experience.

## Logging
- Use the `LOG` logger for all logging. Follow existing patterns for log levels: use `LOG.info()` for general information, `LOG.warning()` for recoverable issues, and `LOG.error()` for critical problems. Avoid using print statements.
- Use commas to separate items in lists within log messages for clarity. Avoid dashes `-` and semicolons `;` in log messages.
- Warning messages should end with three dots (...) to indicate continuation.
- Error messaged should end with a period (.) to maintain consistency and signify finality.

## Subprocess Calls
- Prefer using the repository helper `run_cmd()` when invoking external commands instead of calling `subprocess.run()` directly. `run_cmd()` centralizes logging, quoting, and return handling so behavior is consistent across the script. Example:

- Bad: using `subprocess.run()` directly:

  ```python
  res = subprocess.run(["ffmpeg", "-i", "in.mp4", "out.mp4"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  if res.returncode != 0:
      LOG.error(f"ffmpeg failed: {res.stderr}")
  ```

- Good: use `run_cmd()`:

  ```python
  res = run_cmd(["ffmpeg", "-i", "in.mp4", "out.mp4"])
  if res.returncode != 0:
      LOG.error(f"ffmpeg failed: {res.stderr}")
  ```

This keeps command execution logs uniform and ensures `format_cmd_for_logging()` is used for readable command strings.
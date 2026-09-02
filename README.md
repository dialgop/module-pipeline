# module_pipe_project

A modular C++ processing pipeline showcasing interfaces, inheritance, templates, and RAII — built as an OpenCV vision pipeline: capture, motion detection, segmentation, and color analysis.

The project started as a simulated exercise (every module just logged a fake message) and is being converted, module by module, into a real pipeline operating on real video frames.

## Architecture

- **`IModule`** — the interface every pipeline stage implements: `run(FrameContext&)` and `name()`.
- **`FrameContext`** — a plain struct passed by reference through every module's `run()`. It's the shared state for one frame: the captured image, the motion mask/ROI/flag, and (once `SegmentationProcessor` exists) the segmented mask. Each module reads what earlier modules wrote and writes what later modules will read.
- **`Pipeline<ModuleT>`** — owns an ordered list of modules and runs them, taking `should_run`/`on_run` callables so the caller controls which modules run for a given frame (e.g. "only run segmentation if motion was found") without the pipeline itself knowing why.
- **`LoggerBase`** — a small timestamped logging mixin used by every module.

## Modules

| Module | Status | What it does |
|---|---|---|
| `CameraSensor` | done | Wraps `cv::VideoCapture` (RAII: opens on construct, releases on destruct). Supports either a webcam index or a video file path. |
| `MovementDetector` | done | Frame-differencing motion detection (see [Motion detection strategy](#motion-detection-strategy) below). |
| `SegmentationProcessor` | planned | Will take the region `MovementDetector` flagged and segment the moving object's precise shape (contours, later possibly an ONNX model via `cv::dnn`). |
| `ColorProcessor` | placeholder | Currently a no-op logger; will analyze color only within the segmented region once that exists. |

`main.cpp` still wires things up in the original "simulated" shape (single pass, no capture loop) — that gets rebuilt once every module above is real, so the whole thing runs as a live loop over captured frames.

## Motion detection strategy

`MovementDetector` compares each frame to the previous one:

1. Convert to grayscale and blur (`GaussianBlur`, 21×21) — smooths out sensor/lighting noise before it can register as a false "changed" pixel.
2. `absdiff` against the previous (also blurred) grayscale frame.
3. `threshold` the difference — any pixel that changed by more than `kDiffThreshold` (25) counts as "changed".
4. `dilate` the result to merge nearby changed pixels into fewer, larger regions.

**Detection gate — why total changed pixels, not the largest contour:** the first version of this gated motion on the area of the *single largest contour* in the diff mask (a common pattern for "is there one clearly moving object"). Verified against a real test clip, that approach reported **zero** motion frames out of 367 — even in stretches where the video clearly had motion. The diff mask *did* light up (hundreds to ~4000 changed pixels in busy stretches), but as many small scattered regions rather than one solid blob, so no single contour ever crossed the area threshold.

The fix: gate on `cv::countNonZero(mask) >= kMinMotionPixels` (currently 200) — the *total* number of changed pixels, regardless of how they're distributed — and set the reported ROI to `cv::boundingRect(mask)`, the bounding box over every changed pixel (OpenCV's `boundingRect` accepts a mask directly and treats its nonzero pixels as a point set). Re-verified against the same clip: 253/367 frames correctly flagged as motion, tracking the video's actual quiet/active stretches.

This also cleanly splits responsibility from the upcoming `SegmentationProcessor`: `MovementDetector` answers "did anything change, and roughly where" (cheap, coarse), while precise shape extraction via contours belongs to segmentation, which needs it for real.

## Building

Requires OpenCV (found via `find_package(OpenCV REQUIRED)` — developed against 4.6.0) and a C++20 compiler.

```
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug
```

## Testing without a webcam

This project has been developed under WSL2, which doesn't pass USB devices (like a laptop webcam) through to Linux by default — so `CameraSensor`'s webcam-index constructor won't find a device here. `external/example.mp4` is used as a stand-in video source for local testing via `CameraSensor`'s video-file constructor; real webcam testing is expected to happen outside WSL (or via `usbipd-win` passthrough).
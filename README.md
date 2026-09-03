# module_pipe_project

A modular C++ processing pipeline showcasing interfaces, inheritance, templates, and RAII — built as an OpenCV vision pipeline: capture, motion detection, real person detection, segmentation, and color analysis.

The project started as a simulated exercise (every module just logged a fake message) and has since been converted, module by module, into a real pipeline operating on real video frames.

![Pipeline output: detected people outlined in red, each with a real segmentation model's silhouette highlighted in green, and the sampled average color shown bottom-left](docs/images/pipeline_output.png)

*Real output from the pipeline running on `external/park.mp4`: `MovementDetector` flags moving regions, `ObjectDetector` verifies which ones are actually people (red boxes), `SegmentationProcessor` runs a real segmentation model within each person's box (green), and `ColorProcessor` samples its average color (swatch, bottom-left).*

## Architecture

- **`IModule`** — the interface every pipeline stage implements: `run(FrameContext&)` and `name()`.
- **`FrameContext`** — a plain struct passed by reference through every module's `run()`. It's the shared state for one frame: the captured image, the motion mask/regions/flag, the verified person boxes, and the segmented mask. Each module reads what earlier modules wrote and writes what later modules will read.
- **`Pipeline<ModuleT>`** — owns an ordered list of modules and runs them, taking `should_run`/`on_run` callables so the caller controls which modules run for a given frame (e.g. "only run detection if motion was found") without the pipeline itself knowing why.
- **`LoggerBase`** — a small timestamped logging mixin used by every module.

## Modules

| Module | What it does |
|---|---|
| `CameraSensor` | Wraps `cv::VideoCapture` (RAII: opens on construct, releases on destruct). Supports either a webcam index or a video file path. |
| `MovementDetector` | Frame-differencing motion detection, producing one bounding box per distinct motion cluster (see [Motion detection strategy](#motion-detection-strategy)). |
| `ObjectDetector` | Runs a real, pretrained object detector on each motion cluster and keeps only the ones actually classified as a person (see [Person detection strategy](#person-detection-strategy)). The "verify" stage of the coarse-to-fine cascade. |
| `SegmentationProcessor` | For each `ObjectDetector`-confirmed person, finds a silhouette within their own box via contours (see [Segmentation and color strategy](#segmentation-and-color-strategy)). Classical CV, not a trained model — see that section for why. |
| `ColorProcessor` | Computes the average color within the segmented silhouette only — not the whole frame. |

`main.cpp` runs these as a live loop: capture a frame, run motion/detection/segmentation/color analysis (each gated by `should_run` on what the previous stages found), display it, repeat until the source is exhausted or Esc is pressed.

## Pipeline stages

![Five stages: captured frame, motion mask, object detection, segmented silhouette, and the overlay with sampled color](docs/images/pipeline_stages.png)

1. **Captured frame** — straight from `CameraSensor`.
2. **Motion mask** — `MovementDetector`'s raw diff mask: every pixel that changed, however scattered.
3. **Object detection** — `MovementDetector`'s candidate regions (yellow) narrowed down by `ObjectDetector` to only the ones confirmed to actually contain a person (red).
4. **Per-person silhouettes** — `SegmentationProcessor`'s refinement: for *each* confirmed person box from stage 3, the largest contour found within just that box, filled and cleaned up — one silhouette per person, not one blob for the whole frame.
5. **Overlay** — every person's segmentation contour and detection box drawn back onto the original frame, plus the color `ColorProcessor` sampled from those silhouettes.

## Motion detection strategy

`MovementDetector` compares each frame to the previous one:

1. Convert to grayscale and blur (`GaussianBlur`, 21×21) — smooths out sensor/lighting noise before it can register as a false "changed" pixel.
2. `absdiff` against the previous (also blurred) grayscale frame.
3. `threshold` the difference — any pixel that changed by more than `kDiffThreshold` (25) counts as "changed".
4. `dilate` the result to merge nearby changed pixels into fewer, larger regions.

**Detection gate — why total changed pixels, not the largest contour:** the first version of this gated motion on the area of the *single largest contour* in the diff mask (a common pattern for "is there one clearly moving object"). Verified against a real test clip, that approach reported **zero** motion frames out of 367 — even in stretches where the video clearly had motion. The diff mask *did* light up (hundreds to ~4000 changed pixels in busy stretches), but as many small scattered regions rather than one solid blob, so no single contour ever crossed the area threshold.

The fix (v1): gate on `cv::countNonZero(mask) >= kMinMotionPixels` (currently 200) — the *total* number of changed pixels, regardless of how they're distributed. Re-verified against the same clip: 253/367 frames correctly flagged as motion, tracking the video's actual quiet/active stretches.

**Evolving to per-cluster regions:** a single combined bounding box over all changed pixels isn't useful once you want to know *how many* things moved and *where each one* is — needed so `ObjectDetector` can check each candidate separately instead of one giant box spanning the whole scene. `MovementDetector` now uses `cv::connectedComponentsWithStats` to label every separate blob individually, keeping only clusters whose own pixel count clears `kMinMotionPixels` (the same noise-filter idea, just applied per-cluster instead of to the sum) — this also replaced the old total-pixel `motionDetected` gate with a more principled one: motion is only "real" once at least one *cluster* clears the threshold, not just the sum of scattered noise. Verified on `park.mp4` (1920×1080): a walking person's motion fragments into several small blobs (moving limbs vs. a nearly-static torso), so a large elliptical dilation kernel (25×25, up from the original default 3×3/2-iterations) plus an explicit region-merging pass (repeatedly merge any two clusters whose margin-expanded boxes overlap) were both needed to consolidate one person's fragments back into a single region — dropped a worst case of 27 fragmented boxes down to 5 clean ones. Known limitation: two people standing/walking close enough together can still merge into one combined region — motion alone can't tell them apart, which is exactly the gap `ObjectDetector` (next section) closes by looking at appearance instead of just pixel change. The merge step is also capped (`kMaxMergedFraction`, currently half the frame's width/height): without a cap, a long chain of smaller in-between merges could in principle glue two genuinely distant, unrelated people into one nonsensical region — the cap rejects any merge whose *result* would exceed that size, even if the pairwise overlap check alone would have allowed it.

This also cleanly splits responsibility from `SegmentationProcessor`: `MovementDetector` answers "did anything change, and roughly where, as how many distinct things" (cheap, coarse), while precise shape extraction via contours belongs to segmentation, which needs it for real.

## Person detection strategy

`MovementDetector`'s clusters are motion-shaped, not person-shaped — a swaying branch and a walking person produce the same kind of output. `ObjectDetector` is the "verify" stage of a coarse-to-fine cascade (motion → detection → optional segmentation), spending real model inference only on `ctx.motionRegions`, never the whole frame: crop each region (padded so a person isn't clipped at the edge), run a real pretrained detector, and keep only detections classified as `"person"` above a confidence threshold.

**The actual hypothesis being tested here** isn't "this specific model is highly accurate" — the *quality* of detection (how often a real person gets missed or misclassified) is almost entirely a property of whichever model sits inside `ObjectDetector`, swappable independently of everything else. What this cascade is testing is whether **narrowing the search area first** (motion → only the regions something actually changed) makes person detection more *effective* than running a detector or segmentation model over the entire frame on every tick: less area to search means less room for a background object to get misclassified, and inference cost scales with the (usually small) motion-region area instead of the whole image. Swapping in a stronger model later would raise per-region accuracy without touching this architecture at all; the cascade's job is efficiency and focus, not accuracy — that's the model's job.

**A real compatibility wall, hit by actually testing, not assumed:** the first model chosen was `person_detection_mediapipe` (Apache-2.0, from OpenCV's own model zoo, purpose-built for people). It loaded and even matched a verified-exact port of its anchor decoding — but `cv::dnn` failed to import it on our OpenCV 4.6.0, with a `Clip` node parse error: the model uses ONNX opset 13's 3-input `Clip` (implementing `Relu6`), which this OpenCV version's importer doesn't support. Two targeted patch attempts (rewriting `Clip` as `Relu`+`Min`; then also reshaping the scalar constants to `[1]`-shaped tensors) both failed at the same `eltwise_layer.cpp` assertion — a real gap in this OpenCV version's `Min`-layer implementation for constant operands, not something worth patching further.

**The fix:** switched to `mobilenet_ssd` (chuanqi305/MobileNet-SSD, Caffe, MIT license) — an older architecture that sidesteps ONNX entirely (Caffe's own format), and whose `DetectionOutput` layer decodes anchors, runs NMS, and produces ready-to-use boxes *inside* the network graph itself, so `ObjectDetector` doesn't need to hand-decode anything. Trade-off: it's a **general-purpose** 20-class Pascal VOC detector (aeroplane, bicycle, ..., person, ...), not a person-specific model — `ObjectDetector` filters its output to class index 15 ("person") itself; the weights know about 20 unrelated categories it never uses.

**Verified, with honest limitations:** visually confirmed on `park.mp4` — every detected box lands on a real person, zero false positives on background clutter, across a full 858-frame run (457 frames with at least one confirmed person, out of 712 with any motion at all — real filtering, not a rubber stamp). But: CPU inference is well below real-time (~80–100s of processing for ~30s of footage), and false-negative rate is unmeasured (only spot-checked visually, no ground-truth labels) — this is a verified proof of concept, not a "trust it unattended" system.

## Segmentation and color strategy

**No trained model here, deliberately.** Unlike `ObjectDetector`, `SegmentationProcessor` is classical CV, not a neural network. It was deprioritized on purpose: the task is "detect people from movement," not "produce precise body outlines," so a real segmentation model would spend real integration effort (and, based on what happened sourcing `ObjectDetector`'s model — see [Person detection strategy](#person-detection-strategy) — likely real compatibility risk too) on a capability the task doesn't actually need.

**Per-person, not per-frame:** for each `ObjectDetector`-confirmed box in `ctx.detectedPeople`, `SegmentationProcessor` runs `findContours` *within just that person's crop* of `ctx.motionMask`, keeps the largest contour, and draws it filled into the matching region of a shared full-frame canvas (a `cv::Mat` ROI view, so no manual coordinate offsetting is needed — points found in the crop already land in the right place when drawn into the equivalently-offset canvas region). A morphological close (`MORPH_CLOSE`) then patches small internal holes across the whole canvas. `ColorProcessor` computes `cv::mean(frame, segmentedMask)` — the average color across every person's silhouette combined.

This replaced an earlier whole-frame version that searched for the *single* largest contour across the entire motion mask, with no awareness of `ObjectDetector`'s boxes at all — visibly wrong once multiple people stood close together (their fragmented, merged motion blobs produced one nonsensical shape spanning unrelated body parts, not a real silhouette). Segmenting inside each already-disambiguated person box fixed this immediately, since the hard "which pixels belong to which person" problem was already solved upstream by a real model, not by the contour search getting smarter.

**Why a partial silhouette is fine, not a flaw:** because this only sees pixels that *changed* between frames, a person's mostly-static torso often contributes little or nothing to their silhouette — moving limb edges dominate, so the result is frequently a partial outline, not a filled body shape. For the actual goal — detect that a person is there, roughly where, roughly what they look like — a partial-but-correctly-placed silhouette works just as well as a complete one; a full body outline was never a requirement, and reflects real footage rather than an idealized one. The one property that *would* matter is false positives (a wrong silhouette drawn where no person is), and structurally the design is biased against that: segmentation only ever runs inside a box a real model already confirmed as a person, so a false silhouette would require `ObjectDetector` to have been wrong first. Visually spot-checked across two frames (4 and 5 confirmed people) with zero false positives — a small sample, not an exhaustively measured rate, but consistent with the design's structural bias toward avoiding them.

**Gating, and a lesson applied proactively this time:** `SegmentationProcessor` and `ColorProcessor` are both gated externally in `main.cpp`'s `should_run`, on `ctx.frameValid && ctx.motionDetected && !ctx.detectedPeople.empty()` for both. That `ctx.motionDetected` check matters for a subtle reason beyond correctness: `ctx.detectedPeople` is only *repopulated* when `ObjectDetector` runs, which itself only happens when `motionDetected` is true — so a motion-free tick would otherwise leave `detectedPeople` (and therefore `segmentedMask`) holding stale data from an earlier tick. This is the exact same category of staleness bug an earlier version of this pipeline actually shipped with (`ColorProcessor` once sampled a stale `segmentedMask` against an unrelated frame — only caught by running the full pipeline against a video with genuine quiet stretches). This time, the same shape of bug was anticipated and designed around from the start, rather than rediscovered by testing.

## Building

Requires OpenCV (found via `find_package(OpenCV REQUIRED)` — developed against 4.6.0) and a C++20 compiler.

```
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug
```

## Test videos

This project has been developed under WSL2, which doesn't pass USB devices (like a laptop webcam) through to Linux by default — so `CameraSensor`'s webcam-index constructor won't find a device here. Two video files stand in for a live camera via `CameraSensor`'s video-file constructor:

- **`external/park.mp4`** (default, used by `main.cpp`) — people walking in a park. [Source: Pexels, "A Person Walking Down A Road In The Fall"](https://www.pexels.com/video/a-person-walking-down-a-road-in-the-fall-19341635/), free to use under the [Pexels License](https://www.pexels.com/license/).
- **`external/example.mp4`** — an aerial coastline clip used during earlier development (see the motion detection strategy notes above); kept as a secondary test asset.

Real webcam testing is expected to happen outside WSL (or via `usbipd-win` passthrough).
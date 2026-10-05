# Metal Fork Changelog

Fixes to the Metal renderer and shared RSX paths, newest first. All entries were
verified by TU compile (build tree was later wiped; reconfigure before rebuilding),
focused unit tests where runnable, and log evidence from on-device runs.















## 2026-10-05 — Attachment retention: RSX depth/colour toggles no longer end the draw pass (X-Men Origins: Wolverine stutter)

- Wolverine log (steady state, 60 fps, 200% scale): 68 draw passes per frame, 46 ended by "framebuffer change", 2070 MiB
  of attachments loaded and 2243 MiB stored per frame; GPU 11 ms per frame, and the RSX thread waited for it 4.6 times
  per frame (6.3 ms, up to 10.7 ms) at "zcull report read: RSX copy/blit reads report memory", i.e. every frame that
  waited longer than the budget was a visible hitch. The RSX layout drops the depth buffer for draws that do not test
  depth (and the colour buffers for depth-only clears), so each toggle was a pass end: every attachment stored and
  loaded again (~33 MiB at 1440p).
- The draw pass now keeps its attachments when the new RSX set is a subset of them (same area/samples, colour surfaces
  a prefix, the pass's depth or none). Retained attachments are never written (write mask 0 in the pipeline,
  depth/stencil state "always, no write"), re-validated at every layout change, protected from spilling, and dropped
  (pass ended) before a draw that samples one of them. Telemetry: "attachments retained across N RSX layout changes
  (M dropped again for a draw sampling them)" in the render passes line. DESIGN.md §4.

## 2026-10-05 — Performance audit: upstream video defaults, RSX thread hot path, Metal command ordering, glitch fixes

Configuration (the largest part of the low frame rates):
- The fork's global defaults turned the most expensive RSX options on for every game: forced 2x host MSAA on every
  render target (G-buffers, bloom chains and shadow maps included, each resolved whenever it is sampled) at 200%
  resolution scale, i.e. 8x the fill rate of native, plus Write Color Buffers, Write Depth Buffer, Read Color Buffers
  and Read Depth Buffer (per-title workarounds upstream: readbacks the RSX thread waits for, and surfaces initialised
  from guest memory that show stale frames through). Defaults are the upstream ones again (forced MSAA off, the four
  memory options off, 100% scale). metal-fork-defaults-v11 moves installs still on exactly those values; anything else
  is kept. Titles that need any of them keep getting them from their per-title config (GTA IV, God of War: Ascension,
  Far Cry 3, ...).

RSX thread (CPU):
- NV406E semaphore acquire: the flip semaphore waited in a `sched_yield()` loop, i.e. a full core of syscalls for most
  of every vblank period on a vsync-limited game; it waits on the cache line again (WFE on arm64, the upstream path).
  Timed 50 us sleeps only start after 1 ms of waiting (macOS timed waits oversleep by far more than 50 us) and no
  longer wake on unrelated thread notifications. An already-satisfied acquire no longer drops the FIFO prefetch cache
  (`flush_fifo()` on every acquire, hundreds per frame in SPURS games; upstream never did).
- seq-cst fences restored in `write_gcm_label` and `semaphore_acquire` (upstream): PPU/SPU threads that observe GET
  also see every label the RSX wrote before publishing it (arm64 is weakly ordered).
- The per-stage draw profiler (`m_profiler`) was forced on: ~10 clock reads per draw for timings nothing displays
  unless the debug overlay is on. Upstream gating restored.
- The stall tripwire ran every 1024 loop turns (thousands of clock reads per second with an empty FIFO); now in the
  64-cycle sub-unit slice, at most every 250 ms.

Metal backend:
- Compute encoders: every blit/dispatch after the first waited for every earlier command of its encoder, whatever it
  touched (`needs.encoder = compute_classes` unconditionally), so N texture uploads or readback copies in one encoder
  ran strictly one after another. Commands within an encoder are fully declared, so they are ordered by the hazard
  check alone now; the all-stages barrier each encoder begins with (render passes do not declare everything) stays.
- Render pass begin walked the descriptor twice (hazard declarations + attachment traffic telemetry: ~100 objc_msgSends
  per pass, tens to hundreds of passes per frame); one walk, stopping at the first unused colour slot.
- A full-frame RSX clear of some planes only (depth between passes over the same colour targets, colour with depth
  kept) ended the draw pass, storing every attachment and reloading the untouched ones in the next pass. It is drawn
  as the in-pass quad instead; a clear of every attached plane still ends the pass (free loadAction=Clear). New
  counter in the passes telemetry line.
- 16-bit multi-range indexed draws (strips joined by primitive restart): a range starting after an odd index count is
  misaligned for Metal and was re-copied on the CPU per range, after a blocking `dma_manager::sync()` on the RSX
  thread per draw. Such uploads are widened to 32-bit once at upload time; the realignment path remains only for the
  offloader-generated indices of emulated primitives.
- `program::bind()`: the process-wide sampler liveness mutex + hash lookup ran for every sampler slot of every draw
  before checking whether the table already held the ID; it is skipped when it does. Binding location -> slot lookups
  (12-20 per draw) use a flat per-stage index instead of the `unordered_map`. Depth-stencil state lookups cache the
  last key.
- Draws the interpreter cannot run wait at most 16 ms (was 40) for their pipeline, 24 ms per frame (was 60).
- Graphics self-check (MTLFrameInspector: compute dispatches and a surface-store walk on the present path every 15th /
  30th frame) is opt-in: "Graphics Self-Check" (off by default).

Glitch fixes:
- MSL libraries compile with precise math functions (`MathFloatingPointFunctionsPrecise`): the fast:: variants have
  undefined results outside their domain (pow(0, x), log2(0), rsqrt(0)) and implementation-defined precision, which the
  shared GLSL relies on for exponential fog, sRGB encode/decode and LIT/POW/LG2/RSQ (NaN pixels, fog banding). The
  math mode (reassociation, contraction) is unchanged. Archive format version 8.
- Fragment programs declared each stencil mirror right after its texture unit; sampler slots are handed out in input
  order, so programs with many texture units plus a depth-redirected one sampled their last real textures with the
  constant fallback sampler (nearest, clamp-to-border, LOD 0). Mirrors are declared after every texture unit.
- Texture cache: a scaled copy of a typeless-converted source into a 3D texture requested a second helper with the same
  (format, class) key and a larger size, which disposed of the helper holding the converted source before it was read
  (garbage/black slices). The first request now covers both.
- `dma_transfer`: the non-tiled readback copy to guest memory is bounded by the converted byte count, never copying
  scratch bytes (shared with texture uploads) past it.

## 2026-10-04 — cellAudio: one-block lag reverted; write-ahead probe; 10 s audio reports

- The one-block lag gave 98% delivery in GTA IV's menus but sounded much worse in play (no gameplay report: the session ended 25 s into gameplay). Reverted to upstream's mix position.
- New per-10 s line "Audio write-ahead": how many blocks the game had already written past the read position at each MIX event, and how long after the event the mixed block was complete (average/longest vs. the 5.33 ms period). Decides whether a lag is safe for a game.
- Audio report every 10 s (was 30 s), printed whenever the game delivered audio.

## 2026-10-04 — cellAudio: mix one block behind the read position (game keeps real-time rate)

- GTA IV log after the faded-gap change: 240 gaps (7.6 s of filler) in 30 s, queue 37 ms. Not a slow cellAudio thread and not the output: the game delivered ~70% of real-time audio. With buffering, this thread waits for the block at the read position before sending the next MIX event, so the event rate equals the game's response time; GTA IV's SPURS mixer answers in ~6.7 ms (> one 5.3 ms period): ~148 events/s instead of 187.5.
- The block mixed at the end of a period is now the one behind the read position the game is told (tag check, mix and clear all use mix_offset() = -1 with buffering). The game gets a full extra period and its pipeline keeps up; +5.3 ms latency. All games.
- The 30 s audio line now says how much audio the game delivered (% of real time, and % while not idle).

## 2026-10-04 — cellAudio: no more crackling from a late game; faded gaps

- GTA IV log (AirPods): 1017 silent and 135 skipped periods in 30 s, 29 output underruns (843 ms), queue 20 ms on average (desired 100 ms), cellAudio thread never late. The wait-for-a-late-game rule needed more than 50 ms queued, more than a real-time-paced game ever keeps, so each period the game was slightly late became 5.3 ms of hard-cut silence (crackling, stutter); skipped periods queued nothing and ran the output dry.
- Wait floor = the output's largest measured pull (callbacks within 2 ms count as one burst, so Bluetooth bursts are covered) + one period + 1 ms, capped at half the desired buffer; wait for at most the desired buffer duration.
- When the queue does run low: one faded gap that rebuilds a reserve (floor + max(4 periods, desired/4)), for skips too, instead of a gap every period.
- A game that wrote audio a moment ago is late, not idle: wait for it. After 8 untouched periods in a row it is idle and gets silence at the normal rate (upstream behaviour).
- Every edge between game audio and inserted silence fades (2.7 ms decay from the last frame, 2.7 ms fade-in).
- Log line now separates gaps for a late game from silence while idle, and prints the output's largest pull. Applies to all games.

## 2026-10-04 — Accurate RSX reservation access for every game; stall report on close

- Assassin's Creed II (BLES00669) hung on a loading screen ~6 s after its SPUs began reading the RSX report area in main memory; one SPURS kernel never stopped on exit (37 s). Same signature as the God of War: Ascension freeze cured by Accurate RSX reservation access: without it RSX writes to main memory do not take the 128-byte reservations, and an SPU atomic on the same line (SPURS state) can be lost. Now forced on at boot for every game instead of a per-title list.
- RSX stall tripwire fires after 8 s without a flip (was 15 s; the AC2 session was closed after 13 s with no report), and closing a game that stopped flipping 4+ s ago writes the stall report (thread wait sites) to RPCS3.log and RSXStallReports.log.

## 2026-10-04 — NV406E semaphore acquire on the RSX's own held-back release

- GTA IV log: with all labels deferred in order, the wait moved to "zcull report read: semaphore acquire flushes held-back labels" (5.8-9.8 ms per frame), and the acquire's own wait was then 0 ms: the game makes the RSX acquire a semaphore the RSX released itself just before (held back only for the CPU's view of the zcull reports).
- semaphore_acquire: when the newest held-back label for the address carries the awaited value, the acquire is satisfied without forcing the labels out (later commands reach the GPU after the work the release followed, by queue order). The memory still gets the value when the reports land. Otherwise unchanged (flush, then wait). Applies to all games.

## 2026-10-04 — Labels join held-back labels instead of forcing them out

- GTA IV log (approximate ZCULL now on): the RSX thread waited 7.4-10.1 ms per frame at "zcull report read: labels held back for reports forced out". write_gcm_label deferred only pipeline-flushing labels and flip semaphores; any other label (e.g. texture read semaphores, written without a pipeline flush) flushed the held-back labels first, i.e. waited for the GPU to write their reports. Now every label is deferred behind held-back ones, in order (nv47_sync.hpp). Semaphore acquires still flush them before waiting (tagged separately in the log).

## 2026-10-04 — GTA IV: zcull report reads named by cause, approximate ZCULL

- GTA IV log: the named wait site was "zcull report read", 5.3-6.9 ms per frame (GPU 66-80% busy, CPU 70-87% on every core). The reason for a report read is now recorded (rsx::reports::read_reason_scope, Common/zcull_read_reason.hpp): conditional rendering evaluated on the CPU, RSX copy/blit reading report memory (NV0039/NV3089), full sync (semaphore/notify/reference/user command), labels forced out; the Metal wait site carries it.
- GTA IV built-in settings: Accurate ZCULL stats off (approximate ZCULL), so most report reads stop at the first visible sample instead of waiting for every query.

## 2026-10-04 — Performance overlay: CPU and GPU utilization measured like Redline

- CPU (utils::cpu_stats::get_usage, macOS): whole-system load from host_statistics(HOST_CPU_LOAD_INFO) tick deltas, (user + system + nice) / (those + idle), wrapping 32-bit deltas, time-scaled EMA (gain 0.4 per second). Replaces the process-time estimate (times(), 10 ms resolution, divided by the logical CPU count). Same method as mac-resource-monitor (Redline) Sources/Metrics.swift readCPU.
- GPU (MTLGSRender::get_gpu_utilization_pct): the driver's IOAccelerator "PerformanceStatistics" ("Device/Renderer/Tiler Utilization %", busiest key across all accelerators, services re-resolved every 30 s), same EMA. New mtlutils/iokit_gpu_stats.{h,cpp}. The renderer's own command-buffer busy time remains the fallback when no accelerator exposes the statistics.

## 2026-10-04 — Named GPU wait sites in the log

- GTA IV (city): the RSX thread still waited 5.5-8.6 ms per frame for the GPU, attributed only to the last FIFO method (0x0068, NV406E semaphore release), which did not identify the wait. Waits are now named where they happen (wait_site_scope): zcull report read, occlusion query owner list, surface flush to guest memory, DMA flush submission, frame context, hard sync, thread scratch ring recycle. The log line is now "RSX thread waits for GPU work by site"; unnamed waits still show the FIFO method.

## 2026-10-04 — GTA IV hitches: idle zcull drain no longer hard-syncs at 1 ms, shorter pipeline waits

- GTA IV log (gameplay, ~38 fps, frame 26.4 ms): the RSX thread waited 6.43 ms per frame for the GPU, attributed to method 0x0068 (the last command before the FIFO ran dry). The idle zcull drain (RSXThread on_task loop) called zcull sync after 1 ms of empty FIFO, i.e. waited for all queued GPU work, so the commands arriving next waited behind it. Now: after 1 ms idle the work is only submitted (update with hint) and the reports/labels retire through the regular update; the hard sync runs only after 20 ms of idle.
- Draws whose program the shader interpreter cannot run wait at most 40 ms (was 250 ms) for their recompiled pipeline, 60 ms per frame (was 600 ms): pipeline builds took 102 ms on average without the pipeline archive (first run of new shaders), and those waits showed as hitches.

## 2026-10-04 — Reverted: macOS SPU lock-line wait change and God of War: Ascension readback settings

- God of War: Ascension (update 01.04, Disable MLAA / Disable Motion Blur / Skip intro patches) froze ~35 s into play: RSX parked on a jump-to-self, SPURS kernel 1 running at 100% (PC 0x6a54, LR event pending, no reservation) and not reacting to the stop request, the other kernels waiting on the SPURS lock line. Same signature as the freeze fixed by Accurate RSX reservation access, which was on.
- The two changes of the previous entry that touch exactly this hand-off are reverted to the last build that ran gameplay without freezing: the macOS SPU RdEventStat wait is the upstream one-time-callback path again, and Write Depth Buffer / Read Color Buffers / Read Depth Buffer follow the global config again. The NV406E semaphore sleep (RSX side) stays.

## 2026-10-04 — macOS SPU lock-line wait storm, RSX semaphore sleep, God of War: Ascension readback settings

- SPU thread exit stats (God of War: Ascension): each SPURS kernel logged ~62 million spurious wake-ups in 85 s (~730,000/s) with 51 s of "wait" time, i.e. the RdEventStat lock-line reservation wait spun. The non-Linux path re-waits through a one-time callback that overrides the timeout and calls lv2_obj::notify_all() on every wake-up; on macOS (condition-variable wait engine) the six kernels kept waking each other. macOS now uses the Linux path: a plain wait notified by the writer, 50 us timeout.
- NV406E semaphore acquire: spins on the cache line for the first 200 us, then sleeps in 50 us steps (the RSX thread used 65% of a core while busy ~25% of each frame, waiting ~4.6 ms per frame on semaphores the SPUs release).
- God of War: Ascension built-in settings: Write Depth Buffer, Read Color Buffers, Read Depth Buffer off (upstream defaults, not needed by the game).
- Note: the game in the log is APP_VER 01.00 (PPU-5842e8bc...). The official Disable MLAA patch only exists for updates 01.04 and 01.12.

## 2026-10-04 — NV406E SET_REFERENCE no longer stalls on zcull reports

- God of War: Ascension log ("RSX thread waits outside a known cause, by guest command"): method 0x0050 (NV406E_SET_REFERENCE) made the RSX thread wait 4.7-6.6 ms per frame for the GPU: set_reference calls sync(), which reads every queued zcull report while the CPU reads reports, i.e. waits for all queued GPU work. GPU 62-66% busy, ~46-48 fps.
- REF is now written as a deferred label (like the semaphores, see ZCULL_control::defer_label_write) when reports are pending for the CPU: in order, once the reports before it have landed, while the RSX keeps processing. GET is still written immediately. Without pending reports the old path (sync + immediate write) runs.

## 2026-10-04 — Surface budget overflow fix, early submits at pass boundaries

- Bug in the previous entry: the 4096 MiB surface soft budget was multiplied to bytes in u32 inside surface_store::run_cleanup_internal and wrapped to 0, so the cache counted as over budget on every trim ("Surface cache is using too much memory! (48M)") and collapsed all dirty surfaces every frame (forced memory barriers, MSAA resolves). Now computed in u64.
- Submissions at render pass boundaries: when the draw pass ends for a framebuffer change anyway (no extra attachment store/load), the recorded work is submitted if >= 1.5 ms and >= 64 draws passed since the last submit, so the GPU starts earlier and readbacks/zcull reads wait for less (God of War: Ascension: GPU 74% busy, RSX thread waiting ~6 ms per frame in ~2 waits).
- Log: "RSX thread waits outside a known cause, by guest command" (FIFO method of the waits that have no pass context).
- Graphics check: black / partly black screens only count while the game draws (> 50 draws per frame); loading screens and fades no longer make the verdict SUSPECT.

## 2026-10-04 — Graphics self-check in the log (MTLFrameInspector)

- The log now checks what the GPU actually produced, not only what the backend did. Every 15th presented frame a compute kernel samples a 128x72 grid of the image about to be shown; every 30th, the float render targets (RGBA16F/RGBA32F/R32F/RG16F/R16F) drawn to since the last check are checked for NaN/Inf (bit test, safe under fast math) and fp16 overflow. Results are read back once the GPU finished the frame; nothing waits, nothing changes rendering.
- Warnings ("Graphics check: ...", at most one per kind per 10 s): black screen while the game draws (> 50 draws per frame, 3 checks in a row), parts of the screen turned black (>= 10% of a 32x18 map went from lit to black), NaN/Inf from shader math in a render target (address, format, size, MSAA, share of samples), fp16 overflow.
- Every 30 s: "graphics check over 30s" with counts and a verdict (no problems found / SUSPECT: ...), plus a 32x18 text brightness map of the screen, so the log carries a picture of what was on screen. Findings that concern an image print its map too.

## 2026-10-04 — Surface cache soft budget (Far Cry 3 earlier frames showing through)

- Far Cry 3 log (200% scale, forced 2x MSAA): "Surface cache is using too much memory! (451M)" ~3400 times; the Metal trim still used the upstream 300 MiB soft budget, so every frame collapsed all dirty surfaces and dropped surfaces whose memory tag changed (recreated, with Read Color Buffers loading stale guest memory). God of War: Ascension sat at 633 MiB, also above it.
- Soft budget is now a quarter of the device surface quota (4 GiB on a 64 GB Mac), at least 300 MiB. Real memory pressure handling is unchanged.
- Far Cry 3: built-in settings Write Color Buffers on and Accurate RSX reservation access on (forced, like God of War: Ascension), from the official recommendations.

## 2026-10-04 — Fewer submissions from zcull reports (all games)

- God of War: Ascension log (heavy gameplay, ~27 fps): the RSX thread was busy 100% of each frame while the GPU was 54% busy; 83 submissions per frame, 75 of them early submits for zcull reports (each semaphore deferred behind reports submitted immediately, and the periodic check every ~300 us), which split each frame into 129 render passes instead of ~50 (8.4 GB of attachment loads/stores per frame at 200% with forced 2x MSAA).
- Early submits for reports the CPU reads, and for the labels waiting on them, now happen at most once per millisecond (RSXZCULL min_soft_sync_interval_us). Labels deferred in between are submitted by the periodic check. Reads of reports that are not ready still sync on their own, so results stay exact.
- Invalidated surface pool limit 256 -> 1024: above it every cleanup collapsed dirty surfaces (forced copies/MSAA resolves of surfaces nothing read yet). Memory is still bounded by the surface cache quota.
- New log line "RSX thread waits for GPU work per frame by cause" (readback / query / texture setup / ...), to pin down the remaining ~10 ms of RSX thread waits per frame.

## 2026-10-04 — God of War: Ascension menu freeze and silent audio

- Log: after the main menu came up one SPURS kernel (SPU 5) spun at 87–100% of a core, every audio period from then on was silent (the game's SPU mixer stopped writing its port; no cellAudioOutConfigure happened), and at 1:24 the RSX parked on a jump-to-self with no flip for 15 s. cellAudio was not the cause.
- Built-in settings for the title (BCES01741 and other regions): Accurate RSX reservation access on (also forced over a custom config), Write Color Buffers on, approximate ZCULL (precise stats cost ~43 query submissions per frame), SPU Block Size Mega, Sleep Timers As Host.

## 2026-10-04 — Depth-bounds emulation tests STORED depth again (reverts the 2026-10-03 entry below)

The 2026-10-03 change below rests on a wrong premise. EXT_depth_bounds_test: "The depth bounds test determines
whether the depth value (Zpixel) stored at the location given by the incoming fragment's (xw,yw) location lies within
the depth bounds range" and "Unlike the depth test, the depth bounds test has NO dependency on the fragment's
window-space depth value" (Vulkan's depthBoundsTestEnable is defined the same way). Deferred renderers rely on this to
limit light volumes to pixels whose scene depth is in range; testing the light volume's own depth lit the wrong pixels
(GTA IV: flickering light patches in gameplay and cutscenes). The grids of squares once seen in Wolverine came from
reading the depth attachment inside its own pass, already fixed by the copy made outside the pass.

- `MTLDepthBounds.h`: the test reads `texelFetch(depth_bounds_texture, ivec2(gl_FragCoord.xy), 0)`, first thing in
  `main()` (no dependency on exported depth), and discards outside `[min, max]`.
- `MTLFragmentProgram.cpp`: `depth_bounds_texture` declared at `depth_bounds_location` (`sampler2DMS` when the program
  key carries `ROP_MULTISAMPLED`, i.e. the depth buffer is multisampled: sample 0 is read).
- `MTLDraw.cpp`: binds the depth buffer's copy (`redirect_depth_attachment_read`, like depth-compare emulation).
- `tests/test_mtl_depth_bounds.cpp` updated. GLSL for both sampler types validated with glslang/spirv-val/spirv-cross.

Also for GTA IV (built-in title config): MSAA Auto (forced host MSAA anti-aliases it) and time stretching off (its
audio queue sat at the 75 ms stretch threshold of the 100 ms buffer, so playback kept slowing down).

## 2026-10-03 — Depth-bounds emulation tested the wrong depth (stored, not fragment)

On GPUs without a hardware depth bounds test (pre-Apple10, e.g. M1 Max), the
29 draws/frame Wolverine runs through the test had the fragment shader fetch
the depth the buffer held *before* the draw (a per-draw copy made outside the
pass) and discard on that. Stored depth belongs to earlier draws: the first
depth-bounds draw after a clear to 1.0 with bounds [0, 0.5] discarded
everything, and every other draw culled on stale tiles — the grids of squares
and wrong-discarded pixels in the log triage. Real RSX tests the fragment's
*own* depth (GL/VK backends run it natively; same binary on Apple10 hardware
tested fragment depth via `setDepthTestBounds`, so M1 and M4 disagreed per
pixel). The shader now tests `gl_FragCoord.z` (the exported `r1.z` after
`fs_main()` for depth-exporting programs, which is what hardware tests),
pixel-rate for MSAA too; the `depth_bounds_texture` declaration, the per-draw
depth copy and the binding are gone (slot kept as the test marker for the
per-draw bounds push). Program keys unchanged, pipeline-archive digest keys on
translated shaders so caches re-record automatically. NOT YET COMPILE-VERIFIED
(agent shell EMFILE-down; user rebuild required) and no on-device run yet.

## 2026-10-03 — Granular graphics-flow logging (per-kernel trace + issue census)

Every GPU flow outside the draw passes is now countable from RPCS3.log alone:
new `Emu/RSX/Metal/MTLGraphicsLog.{h,cpp}` records MSAA resolve/unresolve (by
aspect), blit/scale passes, compute kernels (detile, gather/scatter, shuffle,
fconvert, deswizzle), uploads, DMA readbacks (+bytes), surface read/clear init,
spills, depth copies, gather reuse and presents — one free-when-disabled trace
line each, summarized per frame in a new `Metal: graphics flow per frame` notice
in the existing stats block. A log listener collects RSX warnings and worse
(deduplicated, frame-tagged, top-8 reported as `Metal: graphics issues`), so
the 100+ error sites need no per-site changes, and renderer init logs a
`Metal: graphics config` snapshot (MSAA, the four buffer/resolve settings,
resolution scale, output scaling) for bug reports.

## 2026-10-03 — SVR 2011 logo stall: boot barrier phase 2 proceeds on sustained execution

On-device log (BLUS30621, 5 min session): the SPU group-start boot barrier fired
12,925 times, accumulating ~126 s of PPU stall, including 56 hits of the full
2 s phase-2 timeout — one every ~2.2 s for the whole logo phase
(`SpursMan1SpursHdlr0`, group `0x4000200`, `threads executing but not yet
waiting`). Compute-bound kernels (Bink video decode at 99.9% of a core for
40 s straight) execute for seconds without touching a wait state, so every
group start froze the game's SPURS dispatcher for 2 s: the movie stuttered at
~10 flips/s with 0 draws and the title/login screen never arrived (user
rebooted once, then quit). Phase 1 (first-block execution, 30 s cap) is
untouched; phase 2 now proceeds early once every live thread shows sustained
execution past a 100 ms grace (block_counter is JIT-emitted, execution-only,
so advancement is genuine execution, never compiler activity). Stalled threads
(still compiling/booting: no counter movement, the real GoW hazard) still wait
out the full 2 s, so the Ascension wedge protection is intact.

- New `Emu/Cell/lv2/spu_boot_settle.h`: dependency-free settle tracker with the
  grace/recency/timeout decision; `sys_spu_thread_group_start` phase 2 drives
  it instead of the bare wait-flag loop. Fast-path behavior is identical
  (all-waiting still breaks immediately; only full-timeout cases return early).
- Sub-2 ms all-polling starts demoted to trace (8.6k notice lines/session
  gone); every slow or unusual start still logs at notice with a new
  `threads executing (sustained), proceeding early` outcome.
- New `tests/test_spu_boot_settle.cpp` (7 tests: immediate settle, stopped
  ignored, sustained proceeds at ~100 ms not 2000 ms, stalled waits out the
  timeout, brief-advance-then-stall never proceeds early, mixed, late wait).
- Verified: `sys_spu.cpp` TU compiles under the recorded build flags; all 7
  tests execute green (standalone runner: repo GTest is uninstallable offline,
  `rpcs3_test` still needs `brew install googletest` with fixed perms).
- In-game confirmation still needed: rebuild, boot SVR 2011, logo should play
  through to title without the 2 s hitches (`proceeding early` lines, none at
  the full timeout, in the log).
- Follow-up (same day, new 82 s log on the fixed build): zero full timeouts,
  59 early-outs all breaking at 101–102 ms, all inside the 15 s logo phase
  (~4×100 ms stalls/s — still the dominant video stutter; nothing stalls
  after 0:00:44). Grace tightened 100 ms → 10 ms with a 3 ms recency window,
  plus a seen-flag so threads that never advance inside phase 2 can no longer
  count as executing on recency alone (required now that grace < recency).
  Worst case per start drops ~100 ms → ~10 ms; stalled (compiling) threads
  still keep the full 2 s. 7/7 tests updated and green.
- Follow-up 2 (in-match log, 543 draws, ~17 ms frames): barrier fires at
  exactly 60/s (one group restart per frame on `0x4000200`, 2459 early-outs) —
  60 × 11 ms = 66% PPU stall, the whole in-game stutter. Grace 10 ms →
  500 us with a 250 us recency window and 200 us poll cadence: phase 1
  already proves the thread awake and executing, so the grace only covers a
  dispatcher-prologue tail (SPURS kernels poll first by design); compile or
  deschedule gaps show as no advancement and still wait out the 2 s, and
  sleep overshoot only lengthens the grace (the safe direction). Per-frame
  cost drops ~11 ms → sub-ms. Tests moved to matching 200 us ticks (blip
  test is now a single block + freeze; late-wait polls at 400 us); 7/7 green,
  `sys_spu.cpp` TU clean, format clean.

## 2026-09-29 — Sampler crash verified fixed; prologue freeze is a distinct quiet-transition class

On-device run with the LOD-quantization build (BCES01741, 3.5 min): sampler creations
fell 1084 -> 82 (IDs top out at 0x46, LODs snapped to 0.5 steps), NO segfault, session
ended in an orderly user stop. The sampler table-overflow class is closed. The remaining
prologue freeze is a different class: frames present at ~45 fps until ~0:02:40, then the
game parks RSX itself (jump-to-self, PUT static) for a load that never finishes, with
zero errors. Audited every layer and all are healthy-but-parked: GPU completes (no fence
timeouts, presents until onset), the offloader drains (no 2 s give-ups), the SPU LR waits
are timeout-bounded (100-200 ms re-poll, so no lost wakeup can park them — the
reservation-line write genuinely never comes), one persistent SPU group, one boot-time
join pending 180 s (background by design). Missing piece is PPU-side: which load driver
(main/usleep poll target, SPURS submitter) never advances.

- Stall PPU lines now name the joined group (`group=0x...` from the parked thread's
  GPR3) when the wait is `sys_spu_thread_group_join`: next freeze distinguishes a dead
  group from the live kernel group in-report. Pure `format_stall_thread_line` extra +
  2 formatter tests; existing lines render byte-identical.
- CB-chain "out of free entries" is rate-limited (`mtl::spam_meter`: first 32 pass,
  then one summary per 1024 with a running total): the wedge fired it 1126x in 60 s and
  buried every other signal. New `tests/test_metal_spam_meter.cpp`, 3/3 executed green.
- Verified: RSXThread/MTLGSRender TUs recompile, fresh `rpcs3.app` built 00:20.
- Follow-up forensics: the stall dump now decodes WHAT each parked PPU thread waits
  for from its syscall argument registers (`format_stall_wait_args`: join group,
  usleep duration, queue/cond/mutex/flag ids, timeouts; unknown waits decode empty).
  The GoW wedge's `group=0x4000100` line already confirmed Hdlr0 joins the live kernel
  group. 4 new formatter tests in `tests/test_rsx_stall_report.cpp` (TU compiles;
  execution blocked by the sandbox Qt-init hang that stops all suites equally).
  Fresh `rpcs3.app` built 00:43.
- Poll-loop context in the stall dump: threads parked in `sys_timer_usleep` now carry
  the raw guest code words around the sleep call site plus the GPR file, so the next
  report names the exact polled-flag address (loop load displacement + base register)
  instead of just `sleep=0x1e`. The wedge's new data: main polls every 30 us
  (spinwait, not frame pacing), the voice 160 ms anchor and the 60 s NpService timeout
  validate the register decoding, and GPR3 is confirmed clobbered (HLE pre-writes
  CELL_OK on entry: `sys_lwcond.cpp:443`) so queue/cond IDs read 0 — the decoder keeps
  the intact GPR4+ fields. 2 new formatter tests; TU + test TU compile. Fresh
  `rpcs3.app` built 01:07.
- Poll-loop capture decoded by hand: main's 30 us loop ends in `lhz r3,0(r28)` with
  r28=0x99aac4 (halfword flag, want nonzero; computed as 0x99AAC0+r31 with r31=4),
  EdgeWatchdog's 800 us loop ends in `lwz r3,0(r30)` with r30=0x99ac18 (progress
  counter, must pass 3) after a per-iteration kick call, and stores r28 to 0x9AAC14
  on progress. One loader block (~0x99AAC0, thread struct at 0x99AB90): main's flag
  (+4), Edge's store (+0x54), Edge's counter (+0x158). Nothing in the log writes it;
  all produce-paths idle.
- The dump now resolves the poll load itself (`decode_stall_poll_addr`: last
  lwz/lbz/lhz/lha/lwarx wins, EA from GPRs) and prints the flag's live value
  (`pollval=[ea]=value`): stuck-at-zero across freezes = producer never ran,
  advancing = stalls partway. 3 new tests built from the live captured words
  (TU + test TU compile). Fresh `rpcs3.app` built 01:25.
- Flag values are in: main's slot flag reads word 0x00011404 (halfword 0x0001, stuck
  SET — main waits for CLEAR), Edge's counter reads 3 (needs 4). Same block, new
  run: base stable at 0x99AAC0, slot moved 4 -> 8. Producer of unit 4 never ran.
- The dump now lists LV2 event-queue backlogs (bounded, try-lock, skipped when
  contended): backlog>0 under a parked receiver = lost wakeup (directly fixable);
  all-empty = never posted (producer-side). 1 new formatter test; TU + test TU
  compile. Fresh `rpcs3.app` built 01:42.
- Correction: the queue pass used single-param `idm::select` and reported a bogus
  "0 event queue(s)" (a live game always has queues); fixed to
  `idm::select<lv2_obj, lv2_event_queue>`. SPU lines also carry outbox state
  (`outmb`/`outintr` = posted-unread SPU->PPU mailbox/interrupt messages): pending
  outintr under an empty queue names the interrupt-delivery bridge as the break.
  1 new mailbox test; TU + test TU compile. Fresh `rpcs3.app` built 01:52.

## 2026-09-28 — RSX 0x448/0x449 root cause: per-frame LOD jitter mints sampler states past the driver table

The always-on sampler-creation log (1084 lines, 996 distinct IDs) cracked it: driver
GPU resource IDs run as a perfect monotonic sequence 0x1d-0x400, the killer (0x400,
then 0x401 in the prior session) is always the max-created ID, and IDs <= 0x3fe bind
fine. The pool dedupes by exact float equality while GoW varies min_lod continuously
(40 distinct values in the last 60 creations: 0.031250, 0.062500, 0.066406, ...), so
every frame mints new MTLSamplerStates until ~1024 simultaneously-live
argument-buffer samplers overflow the driver-side table and setSamplerState faults
resolving the newest ID. The killer 0x400 differs from healthy 0x3fa only in min_lod
(1.457031 vs 1.523438). The old trim() never fired: its ceiling was 1024, the cliff.

- `mtl::quantize_sampler_lod()` (`mtlutils/sampler.h`): snap min/max LOD and bias to
  0.5 steps. Killer and healthy merge to 1.5; the observed jitter cluster collapses to
  <= 2 buckets; half a mip step is invisible (the Nearest path already rounded bias).
- `MTLDraw.cpp` fragment path: quantize min_lod/max_lod/lod_bias on every mip path,
  replacing the Nearest-only bias rounding (same result, wider coverage). Vertex path:
  quantize min/max LOD (it had no rounding at all).
- `resource_manager::trim()`: ceiling 1024 -> 512, so the pool sits well below the
  driver cliff (disposed states stay live until the GPU drains the eid queue).
- New `tests/test_metal_sampler_lod.cpp` (MetalSamplerLOD, 4 tests) pins the snap
  contract against the exact killer/healthy values from the crash log.
- Verified: MTLDraw/ResourceManager/sampler TUs compile, fresh `rpcs3.app` built
  23:46, LOD assertions pass against the real header + real sampler.cpp.o via a
  standalone harness (the Qt test binary hangs at startup in this sandbox for all
  suites including pre-existing ones — environmental, gtest file is in-tree for CI).

## 2026-09-28 — GoW Ascension wedge (parked RSX) + sampler-ID flood: diagnose, don't guess

On-device log (BCES01741): healthy ~60 fps output until 2:04, heavy scene at
2:04-2:34 (892 draws/frame, GPU 69% at 200% resolution scale), then from 2:34 a
flood of 204 dead-sampler substitutions with tiny never-live IDs (0x25-0x35,
fragment stage, dozens of programs), draws stopping at ~2:47, and the 15 s
no-flip stall tripwire at 3:02 (GET=0x374170 PUT=0xa80000 static, jump-to-self).
The process never segfaulted; the game parked RSX waiting on an unnamed event.

- `report_dead_sampler` (`MTLProgramPipeline.cpp/.h`) now logs the slot's texture
  slot and array size with each substitution, so the next log says directly whether
  the index is out of range (layout bug) or the ID is stale in a valid slot
  (lifetime bug).
- The stall tripwire (`RSXThread.cpp/.h`) now dumps what every PPU thread is
  blocked in when it fires (`dump_stall_thread_states`, bounded to 32 threads,
  read-only): name, waiting/running, HLE function and CIA, via the pure tested
  `format_stall_thread_line()`. The parked FIFO alone cannot name the wait.
- Follow-up from the first wedge dump: it showed `main_thread` in
  `sys_timer_usleep` and `BigSpursHdlr0` stuck 85 s in
  `sys_spu_thread_group_join`, i.e. an SPU-side wedge, while the dump only
  covered PPU threads. The dump now covers SPU threads too (bounded to 16,
  `SPU[id] "name": waiting/running at PC 0x...`), and the line formatter takes
  a pc label ("CIA"/"PC"). Focused suites pass 20/20.

## 2026-09-28 — GoW gameplay segfault 0x448: killer passed every guard, trace run needed

On-device crash report (new pipeline worked end to end): `program::bind` fault in
`setSamplerState` on `rsx::thread`, FIFO GET=0x6c4b24, with the enriched
substitution log proving the 8 prior dead IDs sat in *valid* slots
(`texture slot 7+1`, array 1) — stale small IDs, not out-of-range indices — and
no layout-validation abort anywhere, so the crashing write carried a live-checked
ID at a valid index into a valid table. Static elimination is exhausted (single
write site, uniform table capacities, airtight single-threaded registry); what
remains is driver-internal (sampler config the G13 driver rejects) or heap
corruption, which only the per-write trace can separate.
- Raised the sampler creation-log cap 4096 -> 16384 (`mtlutils/sampler.cpp`) so a
  traced run still shows the killer's `created id=` configuration line in a long
  session. (An attempt to add the shader MSL hash to the trace line was reverted:
  `program` holds no shader refs; the FIFO tag on each trace line plus the
  creation log already identify the killer.)
- Next step is one traced run (`RPCS3_METAL_SAMPLER_TRACE=1`, see reply): the last
  trace line names prog/stage/slot/ID/liveness, the creation line names the exact
  sampler state to quarantine.
- Tests: new `RSXStallReport` suite in `rpcs3/tests/test_rsx_stall_report.cpp`.
  Focused suites pass 19/19.
- FPS: no code defect found on the hot path for this scene; the measurable walls
  are GPU time at 200% scale + MSAA Auto + 16x AF (12.9 ms/frame) on an M1 Max.
  Turn the resolution scale down first (see reply).

## 2026-09-28 — setSamplerState segfault (RSX 0x03c4da4, fault 0x449): harden the write path

The fault address equals the offset, so the driver dereferenced a bad sampler ID,
index, or table in `mtl::glsl::program::bind` via `MTLGSRender::emit_geometry`.
Prior evidence showed the liveness substitution firing yet the crash persisting, so
the fix closes the whole write path, not just ID lifetime (`MTLProgramPipeline.cpp`,
`MTLProgramPipeline.h`, `MTLGSRender.cpp`):

- `program::init_layouts()` validates every slot against the table capacities (31
  buffers, 64 textures, 16 samplers) with the same overflow-safe check the builder
  uses, plus the push-constant slot and buffer count. A bad layout now fails at
  pipeline creation with a named error instead of segfaulting in the driver.
- `take_slots()` and the combined sampler/texture condition use subtraction-form
  range checks, so an absurd `array_size` degrades to `umax` instead of wrapping
  into an in-range index that `bind()` would write out of bounds.
- `bind()` fails fast with a named error on a null argument table instead of
  faulting inside `setSamplerState`.
- Sampler fallbacks are pre-created once at renderer init
  (`precache_fallback_samplers()` in `on_init_thread`), so no draw creates them
  lazily on the RSX thread.
- Teardown releases the fragment/vertex sampler-handle references before
  `resource_manager::flush()` destroys the pooled samplers (previously the handles
  dangled across the flush).
- The dead-ID liveness substitution and `RPCS3_METAL_SAMPLER_TRACE=1` diagnostics
  are unchanged; run with the trace if a fault ever recurs to name the exact
  program/stage/slot/ID.

Tests: new `MetalBindingLayout` suite in `rpcs3/tests/test_metal_binding_layout.cpp`
(registered in `rpcs3/CMakeLists.txt`) pins the builder inside the table capacities,
including overflow and exhaustion shapes. Verified: both touched TUs compile, the
`rpcs3_test` binary links, and `MetalSamplerLiveness.*:MetalBindingLayout.*` passes
10/10 on-device (build-metal/, gtest built locally; the post-build discovery step
needs an unsandboxed run because the sandbox denies shm at startup).

## 2026-09-28 — Crash-reporting revamp: detailed log before dialog, detailed dialog

Every crash now lands a full report in RPCS3.log before any message is shown, and
the message itself says what failed, where, and where the report lives.

- New `Utilities/crash_report.h/.cpp` (in `rpcs3_emu`): pure `format_report()` /
  `format_summary()` builders plus the signal-safe one-page summary writer and the
  next-launch summary-file helpers. No competing signal handlers: the installed
  POSIX/Windows handlers in `Utilities/Thread.cpp` keep owning register dumps and
  JIT recovery.
- `thread_ctrl::emergency_exit()`: always captures the native backtrace (no longer
  local-builds-only), logs thread name/id, PPU function, RAM usage, and the new
  emulator context hook (`add_fatal_context_provider`, fed by an RSX FIFO
  GET/PUT + last-method provider registered in `Emu::Init`), then flushes with
  `sync_all()` before the dialog path.
- Native handlers (`signal_handler`, `sigill_handler`, Windows filter): full detail
  goes to the log at error level first, a one-page summary is recorded for the next
  launch (skipped for orderly fatals and recovered JIT faults), and the dialog gets
  a short reason instead of the register dump. New coverage: SIGFPE/SIGABRT and
  SIGBUS everywhere; orderly-path signals re-raise instead of looping.
- `report_fatal_error` dialog: syncs the log first, then shows thread name/id and
  the exact `RPCS3.log` path alongside the existing title/build/date/RAM lines.
  The "Emulation stopped" box names the thread and log path too.
- Next launch after a native crash shows a "Previous Session Crashed" dialog (GUI)
  or stderr note (headless) with the summary and log location; the summary file is
  consumed on display.

Tests: new `CrashReport` suite in `rpcs3/tests/test_crash_report.cpp` (6 tests:
report sections, empty-context safety, summary variants, summary-file round trip,
and the real recorder seam). Verified: all touched TUs compile, `rpcs3_test` links,
and `MetalSamplerLiveness.*:MetalBindingLayout.*:CrashReport.*` passes 16/16.

## 2026-09-28 — Sampler trace is now always on (ring, dumped on crash)

Per-write `rsx_log.notice` spam can't be the default (log flood + RSX-thread cost),
but needing a foresight-enabled repro run for crash forensics is equally wrong. So
capture is now unconditional: `program::bind()` records every driver-visible
`setSamplerState` (prog uid, stage, slot, ID, liveness) into a bounded 128-entry
ring (`sampler_write_ring`, one relaxed atomic increment + stores per actual table
write, ~zero steady-state cost), and the crash report dumps the recent entries.
`thread_ctrl::set_fatal_context_provider` became `add_fatal_context_provider`
(duplicate-registration safe) so the Metal backend registers its
`sampler_write_section` next to the RSX FIFO provider; `RPCS3_METAL_SAMPLER_TRACE=1`
remains as the verbose per-write + creation-log deep-dive mode.

Tests: new `MetalSamplerTrace` suite in `rpcs3/tests/test_metal_sampler_trace.cpp`
(order/fields, newest-wins truncation, wrap-around, crash-section empty/non-empty).

## 2026-09-28 — Loading freeze left no trace: stall reports now survive force-quit + relaunch

A GoW loading freeze produced no usable log: the frozen session's `RPCS3.log` was
gone after the user relaunched (each launch rotates the previous log away, and
rapid relaunching churns even the `.gz`). Two gaps made that possible, both closed:
(1) the tripwire never flushed — a force-kill could lose the report from the log
buffer; (2) the report lived only in `RPCS3.log`, which rotation destroys.
`check_stall_tripwire()` now mirrors each fire (tripwire line + PPU/SPU thread
dump) to an append-only `RSXStallReports.log` next to `RPCS3.log` (own wall-clock
header per entry, fsync'd) and calls `logs::listener::sync_all()` so the log file
holds it too. Verified the RSX task loop turns while emulation runs (bounded
in-loop waits, semaphore-acquire timeout), so no separate watchdog thread was
needed. `dump_stall_thread_states()` now also returns its text (still logs it);
log message text is byte-identical to before.

Tests: `RSXStallReport` gains sidecar round-trip (append twice, both timestamped
entries present), empty/bad-path rejection, and sidecar-path placement. 27/27 pass
across `MetalSamplerTrace/Liveness/BindingLayout`, `CrashReport`, `RSXStallReport`.

## 2026-09-28 — Ascension loading wedge diagnosed via sidecar; SPU wait detail added

The sidecar caught its first freeze the same night: GoW Ascension EU, FIFO
spinning (game parked RSX, PUT static), `BigSpursHdlr0` 81 s in
`sys_spu_thread_group_join`, all 6 `BigCellSpursKernel` SPUs in `SPU_RdEventStat`
wait at PC 0x11a8 for ~72-78 s. Ruled out: fork regression (SPU/event code is
untouched vs HEAD), settings (all SPU knobs are upstream defaults), tripwire
blindness (loop turns while running), delivery races in the wait path (100 us
self-wake + re-check; nobody writes what they wait for). Remaining decider is
WHAT the SPUs wait on (DMA tags vs lock line vs PPU signal), so each SPU dump
line now carries `evmask/evpend/mfc` (event wait mask, sticky pending events,
MFC queue depth) via the pure tested `format_spu_wait_detail()`; the live read
is one atomic load plus the same best-effort PC-style read. Next freeze report
selects the fix: MFC completion bug, stale reservation baseline, or lost signal.

## 2026-09-28 — 0x448 setSamplerState segfault fixed: guard the ID across the driver call

The always-on ring named the killer on the first live crash: `prog=137 stage=1
slot=1 id=0x400 live=1` — every pre-crash write, including the faulting one,
passed the liveness check. The ID-lifetime theory is dead; the bug is a
check-then-use race: an ID live at `contains()` time is destroyed (pool eviction
under streaming pressure — 452 to 721 samplers in 30 s, with 8 dead-ID
substitutions firing 1.1 s before the crash) on another thread
(GPU-completion/eid-scope reclamation of ref-counted sampler objects) before
`setSamplerState` runs, and the driver faults resolving the freed ID (nil+0x448).
`sampler_liveness` gains `retain_if_live()`, and `program::bind()` holds the
returned guard across the driver call — removal of that ID blocks until the call
returns, making check and use atomic. Same-thread reentry audited (no liveness
calls while a guard is held; the mutex stays a leaf), and the trace/ring live
flags now come from the held guard instead of a re-check.

Tests: `MetalSamplerLiveness` gains guard-owned/guard-released cases plus a
threaded removal-blocks-until-release test (deterministic choreography, always
joined). 33/33 pass across all focused suites.

## 2026-09-28 — Ascension loading wedge eliminated: SPU group-start boot barrier

The third wedge report (same sync point, now with wait detail) gave the decider:
every kernel `evmask=0x400 evpend=0x0 mfc=0` — pure lock-line wait, nothing
pending, MFC idle. No lost signal, no stuck DMA: the kernels' first poll landed
on already-written state (stale reservation baseline). Timeline proof: fresh SPU
LLVM compiles overlap group create/start/join, so kernels boot seconds after the
PPU has submitted and joined. `sys_spu_thread_group_start` now waits (after
releasing the group mutex, kicking each thread) until every started thread has
executed its first block, making PPU submissions postdate SPU first-poll as on
hardware. Bounded by a 30 s timeout (falls back to immediate return), aborted on
stop, skipped for SPU-proxied starts. No headless unit test exists for an LV2
timing barrier (needs live SPU groups); verification is the repro itself — the
next boot either loads (fixed) or lands a fourth sidecar report (curious new
data), and all 33 focused tests still pass.

## 2026-09-28 — Barrier phase 2: first execution is not first poll

The fourth wedge fired with phase 1 (first-execution) in place: the kernel init
path between first block and first poll still loses to a fast PPU submitter. The
barrier is now two-phase: (1) first execution, 30 s timeout (covers LLVM entry
compile); (2) first wait-state-or-stop, 2 s timeout (covers init-to-poll;
compute-bound kernels that never wait fall through instead of stalling group
start). The barrier also logs one `sys_spu.notice` per group start with thread
count, milliseconds, and outcome (`all threads polling or stopped` /
`threads executing but not yet waiting` / `timeout` / `emulation stopping`) —
this both proves engagement in the next log and shows SPU boot latency outright.

## 2026-09-28 — Prologue wedge: boot timing eliminated, capturing the lock address

The fifth wedge (prologue screen) landed with the barrier proven engaged: 1 ms,
all threads polling. SPU boot latency is eliminated as the trigger — the kernels
were polling before the PPU proceeded, yet still ended idle. What remains unknown
is WHICH lock they poll, so each SPU dump line now carries `raddr=` (the
reservation address recorded at RdEventStat wait entry; zero unless waiting).
That address names the protocol (SPURS area, taskset queue, peer LS) and
therefore the writer that never wrote — the final datum before the fix.

## 2026-09-28 — 0x449 crash with guard held: lifetime eliminated, creation log always on

The prologue crash recurred with the liveness guard active (`live=1` from the
held guard, not a re-check): a live-registered, live-object ID still faults the
driver. Lifetime is eliminated as a cause. Both killers (`0x400`, `0x401`) sit at
the session's ID high-water mark while everything ≤ `0x3fe` binds fine — either a
driver-side ID-table limit reached under pool churn, or a killer sampler config.
The decider is the ID->config map, so sampler creation lines are now always on
(bounded 8192/process via the tested saturating `sampler_creation_trace_budget()`;
verbose mode keeps its own cap). The next crash log names the killer's config
outright. 34/34 pass.

## 2026-09-28 — Sampler crash persists past the liveness fix: tracing the killer write

The liveness substitution fired 8 times (`stage 1, slots 5/6/7/13, stored id=0x31`,
a never-live ID) but the session still segfaulted in `setSamplerState` 130 ms later,
so the crashing write passed a live-checked ID: the fault is in the write itself
(slot/index/table/sampler-state), not ID lifetime. Pool stayed ≤467 (trim never runs),
program cache is never cleared mid-game, warnings cluster benign.
New diagnostic (`sampler_liveness.h`, `sampler.cpp`, `MTLProgramPipeline.cpp`):
`RPCS3_METAL_SAMPLER_TRACE=1` logs every sampler creation (ID + full state) and every
`setSamplerState` write (program/stage/slot/ID/liveness/table). Off by default.
Next run with the trace names the exact killer; correlate its ID with the creation log
to see the sampler state (suspect: exotic state on G13, or a bad slot index).

## 2026-09-28 — GoW sampler crash in `setSamplerState:atIndex:` (fault at 0x448)

Symptom: `Segfault reading location 0x448` from `mtl::glsl::program::bind` via
`MTLGSRender::emit_geometry`, ~3 s after the post-video scene resumes
(`GET=0x0a08aec PUT=0xa80000`, game running). The fault address equals the offset,
so the ID was dead (or zero), and the session's binary already contains the earlier
zero-ID fallback — the crashing ID was non-zero but no longer backed by a sampler.
Root cause class: `bind()` records sampler slots by numeric resource ID while sampler
objects can be destroyed afterwards (pool trim, teardown), leaving the ID lingering in
a cached program's bindings; the driver faults with no context.
Fix (`mtlutils/sampler_liveness.h`, `mtlutils/sampler.cpp`,
`MTLProgramPipeline.cpp/.h`): every sampler registers its ID on creation and
withdraws it on destruction; `bind()` resolves each slot through the live set and
substitutes a live fallback (default, then null), logging stage/slot/stored-ID once
per program/slot with a running total. If no sampler is alive at all (teardown) the
write is skipped instead of faulting. The old fatal throw on this path is gone.
Test: `MetalSamplerLiveness` in `rpcs3/tests/test_metal_sampler_liveness.cpp`
(registered in `rpcs3/CMakeLists.txt`).

## 2026-09-28 — Prologue freeze, second episode (same signature)

Repeat tripwire from the fresh log (`~/Library/Caches/rpcs3-metal/RPCS3.log`):
`GET=0x8b0450 PUT=0xa80000, PUT static, FIFO state=spinning (jump-to-self)`.
Sequence right before the stall: `cellVdecClose` aborts the in-flight AU,
decoder thread exits, main thread does `sys_rsx_context_iounmap(io=0x5f00000)`
with `RSX is not idle while unmapping io`, unmaps the video shared memory, then
goes silent (no further main-thread syscalls; only `voice thread`
`cellVoiceGetPortInfo` spam continues). Backend keeps turning per protocol, so
the wedge is game-side: after video teardown the game parks RSX and waits for
something that never arrives. Still needs a native backtrace of the guest
threads during the freeze (see verify step) to name the wait object.

## 2026-09-28 — Prologue freeze diagnosed: game parks RSX on jump-to-self

Symptom: freeze right after a video ends (Kratos-face close-up): frozen frame, audio
continues, `rsx::thread` hot with no RSX output.
Diagnosis: the new stall tripwire fired and proved the state —
`GET=0x828ad0 PUT=0xa80000, FIFO state=spinning` (jump-to-self: the game parked RSX and
never released it). The backend spins per protocol here and must not break the spin;
the game waits for something else after video teardown (decoder closed, video memory
unmapped at the freeze point) and never patches the jump. Next step needs a native
sample of the guest threads during the freeze (`sample rpcs3 5` in Terminal) to name
the game-side wait object.
Tripwire improvement (`rpcs3/Emu/RSX/RSXThread.h`, `rpcs3/Emu/RSX/RSXThread.cpp`): the
report now decodes the FIFO state name and says whether PUT is static (game dead) or
advancing (game writing but not releasing).

## 2026-09-28 — Freeze audit: bounded waits, stall tripwire

Symptoms: God of War: Ascension froze in cutscenes and in the main menu (also seen
in other games): frozen frame, audio continues, `rsx::thread` at ~88% CPU with zero
RSX log output for 50–70 s, exit request ignored.

Findings and fixes:

- `dma_manager::sync()` spun unboundedly (`utils::pause()`, no timeout, no log, no
  stop-check) whenever the offloader stopped making progress, freezing the RSX thread
  at full CPU and ignoring exit requests.
  Fix (`rpcs3/Emu/RSX/RSXOffload.h`, `rpcs3/Emu/RSX/RSXOffload.cpp`): 2 s bounded wait
  with an error log (queued/processed counts), stop-check on the RSX path, and a wedge
  latch so later calls fail fast to the existing fallback paths (e.g. inline submit)
  until the offloader moves again. Callers already tolerate `false`.
  Test: `rpcs3/tests/test_rsx_offload_sync.cpp` (registered in `rpcs3/CMakeLists.txt`).
- Renderer destructors retried that drain loop unboundedly; a wedged offloader would hang
  shutdown. Fix: 5 s bound with an error in `MTLGSRender::~MTLGSRender`
  (`rpcs3/Emu/RSX/Metal/MTLGSRender.cpp`) and `VKGSRender::~VKGSRender`
  (`rpcs3/Emu/RSX/VK/VKGSRender.cpp`).
- Unspecialized-pipeline wait hung forever if a builder died without publishing.
  Fix (`rpcs3/Emu/RSX/Metal/MTLPipelineCompiler.cpp`): 30 s deadline, error log,
  `nullptr` fallback to full-state compile (an already-handled outcome).
- New RSX stall tripwire (`rpcs3/Emu/RSX/RSXThread.h`, `rpcs3/Emu/RSX/RSXThread.cpp`):
  when the task loop keeps turning with no display flip for 15 s while running, it logs
  one error per episode with FIFO GET/PUT, FIFO state and last method. Diagnostic only,
  never blocks; armed by the first flip, skips paused sessions. Any future freeze of this
  class now names its state instead of going silent.

Audit notes (checked, no change needed): all GPU timeline waits funnel into
`timeline::wait`, which logs every 10 s when stuck; `wait_for_queued_submits`,
present waits (`FRAME_PRESENT_TIMEOUT`), `frame_context_cleanup`, pipeline-archive and
async-scheduler waits are all bounded; `semaphore_acquire` has a recovery timeout and
stop-check; FIFO fetch sleeps when empty; compile-budget loops are bounded/stoppable.

## 2026-09-27 — GoW sampler segfault in `setSamplerState:atIndex:`

Symptom: native segfault (fault at 0x449) from `mtl::glsl::program::bind` via
`MTLGSRender::emit_geometry`.
Fix (`rpcs3/Emu/RSX/Metal/MTLProgramPipeline.cpp`,
`rpcs3/Emu/RSX/Metal/mtlutils/sampler.cpp`): missing-sampler fallback chain
(default sampler, then null sampler) so an invalid ID never reaches the driver; a
fail-fast check at sampler creation; a fatal naming stage/slot/IDs if no valid sampler
exists instead of a context-free native crash.

## 2026-09-27 — Host GPU labels default back off

The Metal label path (blit copies + `RSXDMAWriter` queue, upload-heap staging, no extra
compute kernel) hung GoW-class games, so `metal-fork-defaults-v8`
(`rpcs3/Emu/System.cpp`) no longer enables `Allow Host GPU Labels`; fresh installs keep
the config default `false`. The implementation stays available as opt-in.
Test: `HostGPULabelsStayOff` in `rpcs3/tests/test_config_defaults.cpp`.

## 2026-09-27 — Fork defaults on (v8)

`metal-fork-defaults-v8` (`rpcs3/Emu/System.cpp`) enables for existing installs: Read
Color Buffers, Read Depth Buffer, Write Depth Buffer, Force Hardware MSAA Resolve.

## 2026-09-27 — Checkerboard/particle artifacts (e.g. Wolverine BLES00150)

Bound depth attachments sampled as textures faulted on Metal. Fix: redirect
bound-depth reads through an out-of-pass depth copy
(`MTLGSRender::redirect_depth_attachment_read`, `MTLDraw.cpp`).

## How to verify on device

Reconfigure (the old `build-metal/` tree is gone) and rebuild, then:

1. GoW menu/cutscene: previously froze with a silent hot RSX thread; a recurrence now
   logs either `RSX offloader sync timed out` (with counts) or the `RSX stall tripwire`
   line (with GET/PUT) — send `~/Library/Caches/rpcs3-metal/RPCS3.log`.
   During the freeze, run `sample rpcs3 5 -file /tmp/rpcs3-sample.txt`, then send
   the 30-line blocks headed `rsx::thread` and `PPU[0x1000000] Thread (main_thread)` —
   the top-level `__CFRunLoopDoTimers` block is only the UI thread and cannot name
   the guest wait. Alternatively pause in View > Debugger and copy main_thread's
   stack.
2. Re-run the focused tests once GTest is available (`rpcs3_test` needs
   `-DBUILD_RPCS3_TESTS=ON`).

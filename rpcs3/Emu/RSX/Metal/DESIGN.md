# RPCS3 native Metal 4 backend — design & team contract

Target: macOS 26+, Apple silicon (M1 = Apple7 = Metal 4 family) only. Metal is the ONLY renderer (Vulkan/OpenGL are
not built). Bindings: metal-cpp 381.0.0 (`3rdparty/metal-cpp`). Namespace: `mtl` (lowercase; `MTL`/`MTL4`/`NS`/`CA`
are metal-cpp). Files: `rpcs3/Emu/RSX/Metal/`, utilities in `Metal/mtlutils/`. The CMake globs every `*.cpp`/`*.mm`
here recursively — adding a file is enough.

Contents: 1. Porting rule · 2. Ownership & lifetime · 3. Synchronization (hazard-tracked barriers) · 4. Render passes
(tile memory) · 5. Binding model · 6. Shader path · 7. Render pipelines (flexible pipeline states) · 8. Shader
interpreter · 9. Pipeline binary archive · 10. CPU encoding cost · 11. Presentation, pacing and upscaling · 12. Metal
gaps · 13. Component owners · 14. Compile checking.

## 1. Porting rule: mirror the VK backend

Every component is a port of its `rpcs3/Emu/RSX/VK/` counterpart. Keep the same structure, class names (in namespace
`mtl`), method names and algorithms, so upstream VK fixes can be carried over. Rename file prefixes `VK` -> `MTL`
(e.g. `VKTextureCache.cpp` -> `MTLTextureCache.cpp`, class `VKGSRender` -> `MTLGSRender`).

### Type substitution table (applies to ALL public signatures)

| Vulkan backend                                   | Metal backend                                                  |
|--------------------------------------------------|----------------------------------------------------------------|
| `const vk::command_buffer&` / `vk::command_buffer&` | `mtl::command_list&` (always non-const)                     |
| `vk::command_buffer_chunk`                        | `mtl::command_buffer_chunk` (MTLGSRenderTypes.hpp, derives command_list) |
| `vk::render_device`, `vk::g_render_device`         | `mtl::render_device`, `mtl::g_render_device`                   |
| `vk::buffer`, `vk::buffer_view`                    | `mtl::buffer`, `mtl::buffer_view` (mtlutils/buffer_object.h)   |
| `vk::image`, `vk::viewable_image`, `vk::image_view`| `mtl::image`, `mtl::viewable_image`, `mtl::image_view` (mtlutils/image.h) |
| `VkFormat`                                         | `MTL::PixelFormat`                                             |
| `VkComponentMapping`                               | `MTL::TextureSwizzleChannels`                                  |
| `VkImageAspectFlags` / `VK_IMAGE_ASPECT_*_BIT`     | `u32` / `mtl::aspect_color`, `aspect_depth`, `aspect_stencil`  |
| `VkBufferImageCopy`                                | `mtl::buffer_image_copy` (MTLHelpers.h)                        |
| `vk::data_heap`                                    | `mtl::data_heap` (mtlutils/data_heap.h)                        |
| `vk::glsl::program`, `program_input`, `shader`     | `mtl::glsl::program`, `program_input`, `shader` (MTLProgramPipeline.h) |
| `vk::pipeline_props`                               | `mtl::pipeline_props` (MTLProgramPipeline.h)                   |
| `vk::sampler`, `border_color_t`                    | `mtl::sampler`, `mtl::border_color_t` (mtlutils/sampler.h)     |
| `vk::fence` / `VkSemaphore` / timeline             | `mtl::fence` / `mtl::timeline` (MTLSharedEvent)                |
| `VkFilter`                                         | `bool linear` (or `MTL::SamplerMinMagFilter`)                  |
| `vk::framebuffer*` / `VkRenderPass` params          | `mtl::image* target` (+ optional depth target); passes build their own `MTL4::RenderPassDescriptor` |
| `vk::get_resource_manager()`, `vk::get_gc()`       | `mtl::get_resource_manager()`, `mtl::get_gc()` (MTLResourceManager.h) |
| `vk::get_event_id()` etc.                          | same names in `mtl::` (MTLResourceManager.h)                   |
| `vk::raise_status_interrupt` etc.                  | same names in `mtl::` (MTLHelpers.h)                           |
| `vk::get_compute_task<T>()`                        | `mtl::get_compute_task<T>()` (MTLCompute.h)                    |
| `vk::begin_renderpass/end_renderpass/is_renderpass_open(cmd)` | `cmd.begin_render_pass(desc)` / `cmd.end_render_pass(reason)` / `cmd.is_render_pass_open()` |
| `vkCmdCopy*`, `vkCmdFillBuffer`                    | `cmd.blit({ mtl::read_*(src), mtl::write_*(dst) })->copyFromBuffer/copyFromTexture/fillBuffer(...)` |
| `vkCmdClearColorImage` / `vkCmdClearDepthStencilImage` | `mtl::clear_image()` (a deferred clear when possible, §4)  |
| image layouts, `change_layout`, `push_layout`       | **Delete.** Metal has no layouts.                              |
| `vk::insert_*_barrier(...)`                        | Delete: every command declares its accesses and the command list encodes the barriers they need (§3) |

## 2. Ownership & lifetime

* metal-cpp: `new*`/`alloc`/`copy` return +1 references; release them. Use `mtl::ref<T>` for RAII. Everything else is
  autoreleased: wrap entry points on every thread in `mtl::autorelease_scope` (RSX thread: per draw/flip/task;
  compiler threads: per job).
* **Metal 4 command buffers do NOT retain resources.** Never destroy a buffer/texture/view/pipeline/sampler that
  in-flight work may use: hand it to `mtl::get_gc()->dispose(unique_ptr)` (deferred until the current event id
  completes), exactly as the VK backend does with `vk::get_gc()`.
* Residency: `mtl::buffer` and `mtl::image` register themselves in the device residency set on creation and evict on
  destruction. Raw `MTL::Buffer`/`MTL::Texture` you create yourself must call `g_render_device->make_resident()` /
  `evict()`. Views and texture-buffers created from a registered parent need nothing. Removals only take effect at
  the set's `commit()`, so `evict()` keeps the allocation retained until a commit includes the removal (see §10).
* Lossless texture compression: Apple GPUs compress Private textures transparently unless their usage forbids it.
  `MTLTextureUsagePixelFormatView` is one of the things that forbids it, so `mtl::image` sets it only for combined
  depth-stencil formats (stencil sampled through an `X32_Stencil8` view) and when the creator asks for it (texture
  cache sections that may be sampled through an snorm/sRGB view). Swizzle, texture-type and subresource-range views
  do not need it. Creating a format-changing view of a texture without the flag logs an error.
  Trade-off: every `shader_read` section whose format has an snorm twin gets the flag, since signedness is sampler
  state the cache does not know at upload. That includes BGRA8Unorm (A8R8G8B8/D8R8G8B8, viewed as RGBA8Snorm with R/B
  swizzles exchanged by `image_view::as()`), i.e. most uncompressed game textures sample uncompressed; DXT/BC
  textures, render targets and blit destinations keep compression. `load_texture_env()` falls back to shader-side
  sign extension (point sampled) for an image without the flag (`image::supports_view_format()`).

## 3. Synchronization model (Metal 4: all resources untracked): hazard-tracked barriers

Every GPU command recorded through `mtl::command_list` (mtlutils/commands.h) declares what it reads and writes
(`mtl::gpu_access`). Compute encoder work is ordered conservatively: every compute encoder begins with a consumer
barrier on every stage of all earlier work of the queue, and every command after the first (`blit()`, `dispatch()`)
waits for the earlier commands of its encoder; only the parts of one command (`blit_concurrent()`) run concurrently.
Letting compute commands overlap earlier work (barriers only for declared conflicts) is only correct if every one of
the backend's GPU operations declares exactly the memory it touches, which nothing verifies; a copy racing a render
pass shows as tile-shaped stale content (GoldenEye 007: Reloaded showed stale 32x32-pixel tiles of its scene target).
The declarations of compute commands are still recorded for what is checked against them (later render passes'
vertex-stage reads, submissions of draws and external work). Render passes are ordered when they begin, because of the
rule below.

**The TBDR rule: no ordering can be added to a render pass after its first draw.** An Apple GPU bins every draw of a
pass (vertex work) before it shades any of them tile by tile (fragment/tile work, attachment loads and stores). A
barrier recorded between two draws of a pass therefore cannot hold back the pass's fragment work for the later draw
only, and is not relied upon to hold it back at all: the pass's order against earlier work is established by barriers
encoded when it begins, before any draw. (Apple documents that a consumer barrier applies to the whole pass it is
encoded in; the rule does not rely on that for a barrier recorded after draws. Such barriers were the leading suspect,
not confirmed on hardware, for the tile-shaped corruption of the build before this rule where a pass sampled what the
previous pass rendered, e.g. deferred lighting reading its G-buffer. The pass barriers are the ordering of the
pre-overhaul renderer, which did not show it.)

* Accesses: buffers by `MTL::Buffer` and byte range; textures by their root `MTL::Texture` with a 64-bit subresource
  mask (bit `(level % 8) * 8 + layer % 8`, aliases only add overlaps; a 3D texture has one layer). Views count as the
  texture they were made from (`image_view::parent_texture()`, also for transient texture-view-pool views, which have no
  `MTL::Texture`), textures made from a buffer (texel buffers) as that buffer. Helpers: `read_buffer/write_buffer`,
  `read_image/write_image` (levels/layers of an `mtl::image`, or an `image_view`), `read_texture/write_texture` (any
  `MTL::Texture`, resolved with Metal queries), `write_object` (opaque state, e.g. a MetalFX scaler).
  `gpu_access::undeclared()` is the fallback for an access that cannot be described: the command is ordered after
  everything earlier (a compute command gets a full barrier, a draw what the pass barriers leave out) and counts as a
  write of everything (telemetry: full-barrier fallbacks). Nothing reaches it in steady state; a non-zero
  count means a binding path without an access description (a programming error).
* Where encoders begin: every render encoder in `command_list::open_render_encoder()` (called by `begin_render_pass()`
  after it folded deferred clears, §4, and by the clear-only passes of `flush_deferred_clears()`), every compute encoder
  command in `compute_command()` (`blit()`, `dispatch()`, `blit_concurrent()`), which records pending deferred clears
  before it opens a compute encoder. Nothing else creates encoders except frameworks through `handle()` (below).
* Compute encoder commands: `cmd.blit(accesses)` (copies, fills; returns the compute encoder to record the operations the
  declaration covers, e.g. all rows/layers/levels of one transfer), `cmd.dispatch(accesses)` (called by
  `glsl::program::bind()` for compute programs from the bound slots; the caller then dispatches on
  `cmd.compute_encoder()`), `cmd.blit_concurrent(accesses)` (joins the command begun by the last blit/dispatch: not
  ordered after that command, for independent parts such as copies into disjoint rectangles of one level). The
  encoder's first command follows the full consumer queue barrier (every stage -> blit|dispatch) the encoder begins
  with, every later `blit()`/`dispatch()` an intra-encoder barrier on the earlier commands (conservative ordering, see
  above).
* Render passes: every render encoder (`open_render_encoder()`) begins with the two **pass barriers**, encoded before
  anything else: `barrierAfterQueueStages(every stage except fragment|tile -> vertex|fragment|tile)` and
  `barrierAfterQueueStages(fragment|tile -> fragment|tile)`. The pass's fragment and tile work (attachment loads,
  shading, stores) waits for every stage of all earlier work of the queue, earlier command buffers included; its vertex
  work (binning) waits for everything except earlier fragment/tile work, so it can overlap the shading of the previous
  pass. They cover the attachments (read after write, write after write/read, clears folded into load actions) and
  everything the fragment stage reads, so they are encoded unconditionally: what the draws of a pass read is only
  known after it began. The attachments and resolve targets of the descriptor as begun are recorded as written by the
  pass (for later compute commands and submissions); in-pass clear quads only write attachments of the open pass.
* What the draws read is declared when it is bound, before the draw's pipeline state is set: `glsl::program::bind()`
  declares every slot of the vertex and fragment tables (`cmd.table_access()`, skipped when the slot did not change in
  the pass; arrays element by element, e.g. the 64 texture slots of the shader interpreter), then sets the pipeline;
  index buffers (`cmd.draw_access(read, StageVertex)`, before `bind()`) and visibility result slots
  (`query_pool_manager::resume_query`: a write of the slot by `stages_attachment`) are declared by their users. The
  only conflicts the pass barriers leave open are vertex-stage accesses after earlier fragment/tile work (a vertex
  texture that an earlier pass rendered, a vertex read of a visibility or storage result) and undeclared accesses. Such a declaration gets a
  queue barrier (-> vertex) if the pass has no draw yet (no pipeline state set on the encoder,
  `render_encoder_bindings::program_uid == 0`: Metal draws nothing without one); once the pass has draws, no barrier
  is encoded and `cmd.pass_split_required()` becomes true: the renderer ends the draw pass (`pass_split_reason::
  vertex_read`), begins a new one and binds again, so the barrier precedes that pass's first draw (MTLDraw.cpp
  `emit_geometry`). Other passes (overlays, copies, clears) bind nothing to their vertex stage that fragment work
  writes; if one did, the barrier is encoded anyway, an error is logged once and the telemetry counts it ("after it",
  must stay 0).
* Stages: attachment accesses count as written by vertex|fragment|tile work (what later work waits for), visibility
  writes by fragment|tile work; the pass barriers order both. Uniform buffers, sampled textures, separate images and
  texel buffers are reads (samplers are no memory access); storage buffers/images are writes unless the stage's shader
  cannot write any (SPIR-V `NonWritable` on all of them, reflected at translation: `shader::writes_storage()`, a
  specialization answering for its base shader -> `program::set_storage_writes()`; RSX programs, the shader interpreter
  and overlays only read). A buffer bound with an empty range gives the shader no bytes and declares nothing. Push
  constants and buffer-size tables are scratch-heap blocks the CPU writes for one bind only and are not declared
  (nothing on the GPU writes them).
* Coverage: a consumer queue barrier (A -> B) blocks B-work of its pass and of every later pass of the queue until
  A-work of every earlier pass completed (Metal docs, "Synchronizing passes with consumer barriers"), so once one barrier
  covered an access for stage B, later B-work needs no barrier for it (the pass barriers cover everything but
  fragment|tile -> vertex for the pass and all later ones). Intra-encoder barriers only order their encoder.
  The tracker keeps, per resource, the merged accesses of earlier encoders / earlier commands of the open encoder / the
  current command, and per (producer, consumer) stage class the serial of the last barrier. State is per recording.
  Every render encoder is a pass of its own (passes are never suspended/resumed, §4).
* Resource identity (why compute commands may begin without a barrier): a conflict is only found when both accesses
  name the same resource, so every path names memory the same way: a texture by its root `MTL::Texture` (views through
  `parentTexture()`, `image_view::parent_texture()` for views and transient view-pool views; `mtl::image::value` is
  always a root texture), a texture made from a buffer (texel buffers, `buffer_view`) by its `MTL::Buffer`, a buffer by
  its `MTL::Buffer` (`mtl::buffer::value()`; rings and the scratch heap are sub-ranges of one buffer and grow into new
  buffers; a DMA block that inherits another uses its head's buffer). No two resources alias memory: there are no
  heaps or aliasable resources, and passthrough DMA blocks (`newBufferWithBytesNoCopy` over guest memory, which a
  re-created block would alias) are disabled (`supports_passthrough_dma` stays false). Render work writes only what it
  declares by construction: attachments and resolve targets from the descriptor, visibility slots in `resume_query()`
  (the only code enabling visibility counting), storage slots in `bind()` (RSX programs, the interpreter and overlays
  never write storage). Enabling passthrough DMA or heap placement requires identity by memory range first.
* Inside a render pass nothing can wait on the pass itself: Apple GPUs do not support fragment->fragment barriers in a
  pass, and the tracker ignores conflicts between accesses of the open pass. To sample a surface that is bound as an
  attachment either (a) use framebuffer fetch (`[[color(n)]]`, same pixel only) or (b) end the pass: the sampling draw
  then runs in the next pass, whose pass barriers order its fragment-stage read after the ended pass's writes (a
  vertex-stage read gets its barrier before the new pass's first draw). The pass only ends when the
  sampled surface was written by the open pass (`render_target::written_in_pass`) and not only by earlier draws of the
  current draw's feedback streak (same program instructions, shader control, fragment textures, blending, viewport and
  scissor, no texture cache invalidation or wait-for-idle in between; `render_target::feedback_read_in_pass_allowed`): a
  draw of the run reads the surface without the writes of the other draws of the run, except for reads that cross into
  a tile already stored (the same race a single feedback draw has with its own writes). The depth attachment is the
  exception: no shader reads it during its pass, readers get a copy made outside the pass (§4, shader reads of the
  draw's depth buffer), so it never needs this split. A vertex texture always ends
  the pass when the open pass wrote it (vertex work runs before fragment work). Write-after-read: depth writes need no
  split (depth reads count as same-pixel reads, ordered before later writes of the tile); a colour write after reads
  of the surface in the open pass (`render_target::read_in_pass`) splits, unless the readers and the writer are draws
  of one feedback streak (a colour clear quad after such reads always splits). Strict Rendering Mode restores a split
  for every read-after-write and write-after-read. A draw pass attaches exactly the RSX surfaces it was begun for and
  ends when they change (§4): an attachment of the open pass that a draw samples is always a bound surface, i.e. a
  feedback read. Deferred clears the next pass would fold are recorded on their own first when a draw samples the
  cleared texture (§4).
* Across submissions: the tracker of a recording only sees that recording. `submit()` checks what the recording and
  its prologue accessed against the accesses of earlier submissions to the same queue that may still run (per-queue
  records, dropped when the submission's timeline value is reached; a queue barrier of a submission in between covers
  them like inside a list), for the (producer, consumer) stage pairs the recording's own barriers do not order
  (`hazard_entry::external_read/external_write`): external work in full, draws only for their vertex stage after
  earlier fragment/tile work. Attachments, fragment-stage reads and compute commands need no check: queue barriers
  order against every earlier pass of the queue, so the pass barriers and the barrier every compute encoder begins with
  already order them after all earlier submissions.
  A conflict makes the queue wait (`waitForEvent`) for the latest conflicting submission before the commit; without one
  the submission starts while earlier work still runs (e.g. the next frame's first passes during this frame's MetalFX
  and present passes). Checked under the submit lock, i.e. in commit order, whichever thread recorded the list. The
  list's pending deferred clears are recorded when it ends, before the check.
* Prologue (`command_list::prologue()`): committed right before its list; every prologue encoder (its clear-only passes
  included) ends with a producer barrier (`barrierAfterStages(its stages, all stages)`), so all work of the list,
  including work recorded before the prologue work, runs after it.
* Contract: every GPU operation is covered by a declaration. Record compute operations only on the encoder returned by
  `blit()`/`dispatch()` (or `compute_encoder()` right after one, for the operations that declaration covers); bind
  resources for draws and dispatches only through `glsl::program::bind()` (every `bind_uniform*()` records the access
  of what it binds); a draw that reads anything outside the argument tables declares it with `draw_access()`. An access
  that cannot be described is declared as `gpu_access::undeclared()`, never left out.
* Work a framework encodes into our command buffer (MetalFX, through `handle()`, which records pending deferred clears
  first): the command that updates the fence the framework waits on is declared with the framework's accesses (ordered
  after what they conflict with), and `cmd.external_work(accesses)` records them with all stages, so later readers of
  the output get barriers on it (and nothing else does). The fence the framework updates when its output is written is
  waited on by the first command of a fresh compute encoder right after it (`metalfx_upscale_pass::wait_fence`, RCAS
  dispatches into that encoder); the present passes begin with the pass barriers.
* CPU/GPU sync: `mtl::timeline` (MTLSharedEvent). `command_list::submit()` commits and signals the next timeline
  value (signaled after all earlier work of the queue); `poke()`/`wait()` poll/wait. The renderer's
  `command_buffer_chunk` calls `mtl::on_event_completed(eid_tag)`.
* Commit order is GPU order. With Multithreaded RSX the renderer's lists are committed by the offloader thread
  (`mtl::queue_submit`); lists committed directly (`queue_submit_now`) that read what those lists render, i.e. the texture
  cache's readbacks (`cleanup_after_dma_transfers`), first wait until the queued ones are committed
  (`mtl::wait_for_queued_submits`). A flush request queues the primary list and the faulting thread commits its readbacks
  right after; without the wait they could run before the rendering they copy and flush stale data into guest memory.
* Shared (unified memory) buffers written by the CPU before `submit()` are visible to that submission. GPU writes are
  visible to the CPU after the submission's timeline value is signaled.
* Telemetry (30 s stats line "barriers per frame"): compute commands that began without a barrier (0: conservative
  ordering), queue barriers (of
  which render pass barriers, barriers before a pass's first draw, and after it: must stay 0), pass splits for vertex
  reads, intra-encoder barriers, full-barrier fallbacks, submissions that waited for / overlapped earlier ones.

## 4. Render passes (tile memory)

On a TBDR GPU every render pass loads its attachments into tile memory when it begins (unless it clears them) and
stores them when it ends; at 225 % resolution scale that is tens of MiB per attachment set and pass. The draw pass (the
renderer's pass of the RSX surfaces, `render_pass_tracker`) therefore stays open across draws until one of the events
below ends it. Its order against earlier work is fixed when it begins (the pass barriers and the TBDR rule of §3):
* It opens lazily at the first draw (or scissored/masked clear) and ends at an RSX surface change, a full-frame RSX
  clear, a feedback read (§3), a draw that needs a barrier after the pass's first draw (§3: its vertex stage reads what
  an earlier pass rendered, `pass_split_reason::vertex_read`), a copy/compute command, another pass, a query pool
  replacement or a submit. `command_list::end_render_pass(reason)` records which (telemetry).
* Surface changes: every change of the RSX surface set (`prepare_rtts`: other colour or depth-stencil surfaces, their
  count, the area or the sample count) ends the draw pass (`pass_end_reason::framebuffer_change`); the next draw opens
  a pass that attaches exactly the new surfaces. A pass never attaches a surface the RSX does not bind, and never
  outlives the surface set it was begun for. The RSX layout of a draw drops the depth buffer when the draw does not
  test depth or stencil, and the colour buffers when it writes no colour, and clears bind only what they clear: each
  such toggle is a surface change.
* Colour attachment N is RSX surface N in every pass: render passes do not enable colour attachment mapping, and
  every render pipeline uses the identity mapping (`colorAttachmentMappingState = Identity`, set in one place for every
  pipeline descriptor, §7). Framebuffer fetch (`[[color(n)]]`, programmable blending) reads surface n.
  `RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING` (the only programs with `input_type_attachment` inputs) gives full-state
  render pipelines (never specializations, §7) and is never run by the shader interpreter (§8).
* Clears. `command_list::defer_clear()` records a whole-area clear of level 0 of a 2D render target as a pending load
  action: the next pass that attaches the texture over exactly that area begins with `loadAction=Clear` for it
  (`begin_render_pass` folds it and restores the descriptor afterwards); anything else recorded first (a compute
  command, a pass not attaching it that way, `handle()`, the end of the list) gets the pending clears first as
  clear-only passes (one per area and sample count). Pending clears imply no open encoder. The hazard tracker sees the
  clear where it is recorded: as a write of the attachment by the pass that folds it, or by its clear-only pass (§3).
  `mtl::clear_image` (surface initialization, most texture cache clears), `clear_color_texture` and full-frame RSX
  clears (every channel of the bound surfaces) use it. A full-frame RSX clear ends an open draw pass first
  (`pass_end_reason::clear`), so it becomes the load action of the next pass that attaches the surfaces: a full-frame
  clear is never drawn over earlier draws of a pass. Scissored and channel-masked RSX clears (and the partial-stencil
  part of a clear) are drawn as a quad in the draw pass (`inpass_clear`, opening the pass if needed, which folds the
  full-frame parts of the same clear); a colour clear quad after feedback reads of the surface in the open pass splits
  it first (§3), and the quad invalidates the encoder state it overwrote (§10). A draw that samples a texture with a
  pending clear flushes the clears before its pass opens (the pass would fold them and the draw would sample stale
  memory); folded attachments count as written by the pass (`written_in_pass`, feedback streak reset), like a clear
  quad drawn in it.
* Depth bounds test (RSX: a fragment whose own depth is outside [min, max] is not rendered; the GL and VK backends
  run exactly this natively, `GL_DEPTH_BOUNDS_TEST_EXT` / `depthBoundsTestEnable`). Apple10 GPUs test in hardware
  (`setDepthTestBounds` when `caps().depth_bounds`). Before Apple10 the fragment program performs it:
  - Key: `MTLGSRender::get_backend_fragment_program_export_config()` (RSXThread's per-draw export configuration, bits in
    `fs_export_config_mask`, so a change invalidates the fragment program and the program-cache hint) adds
    `RSX_SHADER_CONTROL_DEPTH_BOUNDS_TEST` for draws with the test enabled, bounds (clamped to [0, 1] like the hardware
    test) other than [0, 1] and a bound depth buffer, and `ROP_MULTISAMPLED` when that buffer is multisampled (it marks
    multisampled ROP output; the test itself is pixel-rate). No other program changes. With no depth buffer bound the
    test does nothing. The RSX layout keeps the depth buffer for an active test but is not re-evaluated when the test
    becomes active: `begin()` re-evaluates it then (once while the buffer stays out).
  - Shader (MTLFragmentProgram): `main()` begins with the test, `gl_FragCoord.z` outside the pushed bounds ->
    `discard` (MSL `discard_fragment()`, demote semantics: derivatives of the neighbours stay defined). Programs that
    export depth test the exported `r1.z` (saturated, the value written to the attachment) after `fs_main()` instead,
    which is what the hardware tests. Multisampled draws test the same fragment depth once per pixel: every sample
    shares it, and MSL exposes no per-sample fragment depth. Bounds: push constants 112..120, only in these programs'
    push block. Binding: nothing (`depth_bounds_location`, after every other texture, is only the test marker). The
    test reads no depth buffer, so these draws make no depth copy and open no extra pass.
  - History: the test used to fetch the depth stored in the depth buffer before the draw (a copy made outside the
    pass) and test that. Stored depth belongs to earlier draws, so it discarded or kept the wrong pixels (X-Men
    Origins: Wolverine's lights drawn through the test came out as grids of squares, one per tile).
  - The shader interpreter does not run these programs (their draws wait for the recompiled pipeline).
  - Telemetry: the "render passes" line has "depth bounds test: N draws per frame emulated in the fragment shader"; the
    first draw logs a notice.
* Shader reads of the draw's depth buffer (texture units sampling the depth buffer the draw pass attaches: soft
  particles, fog, lights reconstructing positions, and depth compare emulation) read a copy of
  it, `m_depth_copy`, never the attachment. Metal does not define shader reads of a texture that is an attachment of
  the running render pass, and Apple GPUs do return wrong depth for some pixels of every tile then, even when the pass
  never changed depth.
  - The copy has the buffer's format, size, sample count and usage (so every view a unit takes of the buffer, stencil
    and format views included, exists on the copy: `redirect_depth_attachment_read` binds the equivalent view, the
    interpreter's units too). It is made by one whole-texture blit (`mtl::copy_image`) outside the pass: an open draw
    pass ends first (`pass_split_reason::depth_copy`; it resumes with the draw and loads what it stored), pending clears
    are recorded before the blit (`compute_command`), and every compute encoder begins with a barrier after the queue's
    earlier stages, which orders the blit after the writes to the buffer and after the draws that read the previous
    copy (whose texture bindings are declared reads as well). The draw reads the copy like any other sampled texture.
  - Validity: the copy holds the planes of `source` as of `render_target::content_tag` == `depth_tag` (depth) /
    `stencil_tag` (stencil), the same source. Every recorded write to a surface replaces its content tag (draws and clears through
    `mark_attachment_writes`, memory initialization, inheritance, unspill, unresolve, blit engine transfers,
    recycling); `mark_attachment_writes` carries each plane's tag along when the write leaves that plane alone (it is
    given the planes the write may change: a draw changes depth exactly when a depth buffer is attached, depth test and
    writes are on and the depth function can change the stored value (not equal / never), and stencil when the stencil
    test is on with a non-zero front mask or, two-sided, back mask; clears exactly). Anything else makes the next read
    copy again. Typical frames copy once: lights and particles come after the last depth writes and only interleave
    stencil and colour work. The draw's own depth writes do not matter (reads see the depth before the draw; its
    sub-draws count as one draw, overlapping primitives of one draw see the depth from before the draw).
  - Feedback streaks (§3) that read the depth buffer and write depth keep what reading the attachment memory gave them:
    after a draw that read the copy and changed depth, the copy stays current for the next draws of the same material
    key (`streak_key` / `streak_tag`, carried like the plane tags, not in Strict Rendering Mode) while only they changed
    depth: they see the depth from before the streak, any other reader gets a new copy with the streak's writes.
  - No other split is needed for these reads: `render_target::texture_barrier` does not split for depth surfaces (one
    written by the open pass is the draw pass's depth attachment), `feedback_read_needs_split` skips the depth
    attachment, and depth compare emulation has no split of its own any more. A pending clear of the depth attachment
    does not make a draw sampling it record its clears first (`draw_reads_deferred_clear`): the copy is made after them.
  - Telemetry: the "render passes" line ends with "shader reads of the depth buffer during its pass: C copies of it per
    frame (M pass splits)".
* Load/store actions: attachments load unless a deferred clear folds into them; overlay passes use DontCare only when
  their draw overwrites the whole target. Everything stores: RSX surfaces are guest memory, read later by textures,
  readbacks or the next pass (there are no in-pass MSAA resolves).
* Write Color/Depth Buffers: `prepare_rtts` records no speculative readback (`flush_if_cache_miss_likely`) of a
  previously bound surface that the new layout binds again over the same memory: `lock_memory_region` locks its
  section again in the same call, and the section's `create()` drops a synchronized readback (its data would be stale:
  the surface keeps being rendered to), so the copy would only end the draw pass and flag the list for an early
  submit. Skipping it cannot expose stale data: a CPU/SPU access to an unsynchronized locked section faults and
  flushes the current contents synchronously.
* No suspend/resume (`begin_render_pass` takes no encoder options). MTL4 merges a suspended pass with its resumption
  only when the command buffers are committed in one commit call. Every command buffer boundary of the renderer is a
  submission that has to happen then: the CPU waits for its results (Write Color Buffers and DMA readbacks, zcull
  reports, hard syncs), the GPU has to produce them (visibility results are written when a pass ends), or the frame is
  handed to the GPU (flip). Merging would delay the submission until the pass ends, so no split is removable this way;
  RSX commands are recorded serially into one list, so there are no parallel encoders to merge either.
* Telemetry (30 s stats line "render passes"): passes per frame (draw passes, other passes by what the RSX thread was
  doing, `pass_context_scope`), why draw passes ended, feedback splits by reason and feedback reads kept in the pass,
  readbacks not speculated, clears drawn as quads in the draw pass (scissored or masked) / folded into a load action /
  as clear-only passes, the attachment bytes loaded and stored per frame, and the depth bounds test (in hardware, or
  draws per frame that performed it in the fragment program), and the copies of the depth buffer made for shader reads
  during its pass with the pass splits they caused.

## 5. Binding model

One `MTL4::ArgumentTable` per stage slot lives in each `command_list` (`argument_table(table_vertex|table_fragment|
table_compute)`). `glsl::program::bind()` writes GPU addresses / resource IDs into them and calls `setArgumentTable`.
`glsl::build_binding_layout()` is the single source of truth for GLSL binding -> Metal index; the shader translator
uses the same layout for SPIRV-Cross `add_msl_resource_binding`. There is no `setBytes` in Metal 4: push constants and
small uniforms go to `mtl::get_scratch_heap()` and are bound by address.

Samplers must be created with `supportArgumentBuffers(true)` (mtl::sampler does). Stage limits: 31 buffers, 64
textures (we size tables to 64), 16 samplers. Inputs are uniform/storage buffers, texel buffers, combined image samplers
(`input_type_texture`: a texture slot and, for the first 16, the sampler slot of the same index), separate images
(`input_type_separate_image`, texture slots only) and separate samplers (`input_type_sampler`, sampler slots after the
combined ones); any of them may be an array (`array_size`, bound with `bind_uniform_array`). Every `bind_uniform*()`
records, next to the slot's value, the memory it gives the shader access to (hazard tracking, §3).

## 6. Shader path

RSX ucode -> shared decompilers -> **Vulkan-flavoured GLSL 450** (same generator as VK, bindings set 0 = vertex,
set 1 = fragment) -> `spirv::compile_glsl_to_spv` (glslang, already in tree) -> SPIRV-Cross MSL (MSL 3.2, macOS,
framebuffer fetch for subpass inputs, native texture buffers, argument-table friendly discrete bindings)
-> `MTL4Compiler::newLibrary` -> `MTL4::RenderPipelineDescriptor`/`ComputePipelineDescriptor` via `MTL4Compiler` (render
pipelines: §7). Static GLSL used by compute kernels / overlays goes through the same path (`glsl::create_compute_program`,
`glsl::create_graphics_program`).

Compile on RPCS3's pipe-compiler worker threads (thread QoS is inherited by Metal). Background compiles use at most the
VK backend's default worker count (4 on 10 host threads); one more worker only takes jobs the RSX thread waits for
(`prioritize_jobs`, run at emulation QoS). Queue order: jobs the RSX thread waits for, then `COMPILE_AHEAD` jobs (the
shader interpreter's uber pipelines, §8, built at emulation QoS), then the other background jobs (user-initiated QoS),
then the shader cache preload tasks (below, default QoS); the background full-state upgrades of §7 have their own
queue, one at a time, background QoS, and only start when nothing else can. A draw whose pipeline is compiling waits
for it within a small per-frame budget, but only when both shaders already have their MTLLibrary (a new shader's MSL
compile never fits the budget); otherwise it is drawn by the shader interpreter (§8) in the Shader Modes that use it,
or skipped. "Recompiler" builds synchronously and never skips: a pipeline a preload task is building is waited for
(stoppable).

Shader cache preload (MTLGSRender::preload_shader_cache, mtl::program_cache in MTLProgramBuffer.h). Talks 4/6: the uber
shader first, cached pipelines prewarmed in the background at default QoS, never block the frame on compilation.
* Blocking part (RSX thread, before the PPU starts): `rsx::shaders_cache::load()` reads the cached pipelines and
  decompiles their programs on its worker threads (CPU only: ~1.5 ms per shader; every program must be complete before a
  draw can look it up). No dialog and no flips; stoppable between two entries. `add_pipeline_entry` only records each
  pipeline (programs of the cache + normalized state).
* Background: `queue_preloaded_pipelines()` queues one pipe compiler task per pipeline and on_init_thread returns. A
  task builds its pipeline on the worker (GLSL -> MSL ~9 ms per shader, MTLLibrary, unspecialized pipeline from the
  archive, specialization) unless the key exists: it inserts the placeholder itself, so a draw that asks for the
  pipeline first gets its own job ahead of all preload tasks, and a draw that asks while the task runs finds it pending.
  Tasks start once the pipeline archive is ready for lookups (§9). When the last one ran, the outcome is logged with the
  time split, and the archive's first serializer (which holds every cached pipeline) is retired as a complete preload
  (§9). A renderer that stops first drops the queued tasks and reports the preload as interrupted (incomplete).
* Measured before this design (boot logs, 10 threads): 1.2-25.5 s of blocking preload before the PPU started; with
  pipeline archives most of it was the first lookup (§9: 1.6 / 6.8 / 13.3 s), then 6-11 ms of thread time per
  pipeline (0.4-1 s for 624-1009 pipelines); without archives 80-300 ms per pipeline (Wolverine: 1741 in 25.5 s),
  almost all of it MTL4Compiler work. Measured off-device (tools/gereplay, GTA IV cache): GLSL -> SPIR-V -> MSL 9.2 ms
  per shader (glslang 4.8, SPIRV-Cross 4.4), decompiling ~1.5 ms per shader.

Function constants: a GLSL specialization constant (`layout(constant_id = N) const ...`, bool/int/uint/float) becomes
an MSL `[[function_constant(N)]]` with the GLSL value as default (`msl_translation_result::function_constants`).
`glsl::shader::create_specialization(base, constants)` makes a shader that shares the base's MSL, MTLLibrary and
reflection (buffer sizes, storage writes, MSL hash) and only adds constant values; it has a `uid()` of its own and the
pipeline keys mix the values in. A function that declares function constants is always built through
`MTL4::SpecializedFunctionDescriptor` (constants without a value keep their GLSL default; values for constants the MSL
does not declare are ignored).

Generated-code rules (MTLVertexProgram / MTLFragmentProgram emit Metal-specific GLSL; shared Program/ code stays
backend-neutral; `/home/claude/tools` has glslang + SPIRV-Cross + `msldrv` (same options) and `rsxgen` (the real
decompilers on hand-encoded microcode) to inspect the MSL):
* Address spaces: data every thread of a draw reads with a draw-uniform index is a uniform block (MSL `constant`):
  vertex/fragment contexts, constants, texture parameters and the draw parameters (bound at offset 0: a constant
  buffer offset must be 256-byte aligned on macOS, which is also why the conditional rendering predicate, bound at
  offset 0 or 4, stays a storage buffer). Data indexed per thread (stipple pattern, instancing tables) is `device`.
  Neither the MSL nor the GLSL copies uniform arrays into thread memory (vc[] is read in place).
* Push constants (one Vulkan-style space shared by the stages; each stage uploads [0, its block end), 256-aligned in
  the scratch heap): 0..4 vertex `draw_parameters_offset`; fragment 4..32 programmable blending, 32..96 texture LOD
  bias (pre-Apple10), 96..112 `fs_draw_offsets` = the fragment ring offsets (constants, context, texture parameter
  base, stipple), pushed by `update_vertex_env` for every draw, 112..120 depth bounds (only programs performing the
  depth bounds test, §4). They are not a varying: loads they index are uniform.
* Only the RSX vertex position is `invariant` (multipass depth-equal tests across programs); compile options set
  `preserveInvariance`. Invariance only covers identical computations, so the vertex generator removes the one
  difference games rely on the RSX to hide: DP4 of a position whose fetched w is 1 and DPH of it must give the same bits
  (GoldenEye 007: Reloaded draws a DP4 depth pre-pass and tests DPH colour passes against it). Both are emitted as the
  same fused chain `fma(x, m.x, fma(y, m.y, fma(z, m.z, w * m.w)))` (DPH: `m.w` for `w * m.w`), never `dot()` (the
  compiler folds the constant lane of `dot(vec4(v.xyz, 1.0), m)` and may sum in another order), and the vertex fetch
  stores w = 1.0 itself instead of computing scale / scale.
* Math: `MTL::MathModeRelaxed` + fast floating-point functions (reassociation/contraction allowed, IEEE Inf/NaN
  preserved: RSX programs divide by zero and the z-clip transform relies on Inf); `g_cfg.video.disable_msl_fast_math`
  selects `MathModeSafe` + precise functions. No half-precision rewriting (PS3 accuracy first).
* Compute pipelines declare their GLSL local_size as `requiredThreadsPerThreadgroup` (occupancy hint; every dispatch
  uses exactly that size).
* Any change that alters generated MSL, compile options or pipeline descriptors for every shader bumps the pipeline
  archive format version (MTLPipelineArchive.cpp, §9), which discards archives built by the old code.

SPIRV-Cross throws `spirv_cross::CompilerError`; the translation TU (`MTLShaderCompiler.cpp`) is compiled with
`-fexceptions` so it can catch and log (the rest of RPCS3 is `-fno-exceptions`).

## 7. Render pipelines: flexible render pipeline states (MTLPipelineCompiler.{h,cpp})

Every render pipeline goes through `build_render_program()` (`build_graphics_program` for synchronous builds,
`pipe_compiler::compile` for RSX programs and the shader interpreter; overlays and the in-pass clear quads included):
* Full-compile key = shader pair (`shader::uid()`, never reused; a function-constant specialization has its own) +
  `graphics_pipeline_state` without the colour attachment configuration (sample count, alpha to coverage/one, topology
  class, and any field added to the POD). One UNSPECIALIZED pipeline per key: pixel format, write mask, blending state,
  blend factors and operations of every colour attachment are `...Unspecialized`. The first thread that needs it builds
  it; others wait for that build.
* Each variant (colour formats, write masks, blending) = `MTL4Compiler::newRenderPipelineStateBySpecialization` of it
  with the concrete descriptor (Metal only takes the unspecialized properties from it; attachments it leaves out get
  the defaults: no format). This only generates the fragment output part. A deferred request whose unspecialized
  pipeline exists is specialized inline on the requesting (RSX) thread and `pipe_compiler::compile` returns the program
  instead of calling the callback: no compile, no skipped draw, and the compiler inherits the thread's QoS (the program
  buffer and the shader interpreter both use such a program right away). Otherwise a worker builds the unspecialized
  pipeline and specializes; the RSX wait budget applies as before.
* Full state instead: when the pipeline archive lists the full-state pipeline and is ready for lookups (§9: a lookup,
  and no specialization overhead), without a fragment stage, for framebuffer-fetch fragment shaders (programmable blending: the fragment body
  reads the attachments, so it is not compiled without knowing their formats; §4), and for a key whose unspecialized
  build or specialization failed (runtime failure: logged, the first 8 in detail, counted in the telemetry).
* Colour attachment mapping: `get_color_attachment_mapping_state()` is the one place that sets it, for the
  unspecialized, specialization and full-state descriptors alike: `Identity` (fragment output N is colour attachment
  N; render passes do not enable mapping, §4). The property has no Unspecialized value, so it is compiled into the
  unspecialized pipeline and its specializations keep it; it is the same for every pipeline, so no key needs it.
* Background full state (talk 6: specialized pipelines cost a little GPU time): `glsl::program::bind()` counts the
  draws of a specialized program; at `full_state_draw_threshold` (1000) its full-state pipeline is queued as a worker
  job that runs at QOS_CLASS_BACKGROUND, one at a time and only when no other job is queued. The first `bind()` after
  it is ready swaps it in on the binding thread: the specialized pipeline goes to `mtl::get_gc()` (recorded draws may
  use it) and the program takes a new uid, so encoders set the new state. The job shares a `full_state_upgrade` with
  the program (descriptor, result), never the program itself, so either may be destroyed first. This applies to the
  shader interpreter's programs too: the interpreter chooses which program draws (§8), the upgrade replaces the
  pipeline inside a program object, at most once per object.
* Telemetry (30 s stats line "render pipelines"): render pipelines created by specialization (average/longest time),
  with full state (listed in the archive, after a failed specialization), unspecialized pipelines built (average time:
  archive hit or compile), background full-state compiles requested/built/swapped in. Line "shader builds": GLSL -> MSL
  translations, MTLLibraries, pipelines built with archive lookups / compiled without (counts, average times), from
  `record_compile_time` (MTLProgramPipeline.h), which also keeps a per-thread record (`this_thread_compile_timings`,
  reset by the pipe compiler for every job) for the preload and interpreter summaries. Startup logs the policy.

## 8. Shader interpreter (MTLShaderInterpreter.{h,cpp}, MTLShaderInterpreterSource.cpp)

Port of VK's `vk::shader_interpreter`: the shared GLSL interpreters (Program/GLSLInterpreter) execute the RSX
microcode read from an instruction block. Shader Mode (upstream setting, as on Vulkan): "Async Recompiler with Shader
Interpreter" draws with it while a draw's pipeline compiles; "Interpreter only" draws with it everything it can render
exactly, compiles nothing up front (no shader cache preload) and compiles the rest asynchronously (also the render
states whose interpreter pipeline failed to build); "Recompiler" / "Async Recompiler" do not create it.
* Variants (talks 1/4: uber shader while the specialized variant compiles). Each stage is translated and compiled once
  per library: vertex with/without instanced constants; fragment by colour output count (0-4, must match the
  attachments), depth output (depth export or early Z disabled), point sprite coordinates (`[[point_coord]]`, point
  pipelines only). Everything else is one of 17 bool function constants (`interpreter::fragment_feature`: texture
  types, depth compare, depth-as-RGBA8, texel conversion, BX2, flow control, precision modifiers, texcoord control,
  alpha test, A2C, sRGB, 8-bit rounding, output remap, stipple). A feature's code still checks the draw's state from
  the instruction block header, so a pipeline with more features than a program needs draws it identically. The uber
  pipeline of a render state (all features) serves every program; the pipeline specialized for a program's features
  is requested in the background (at most 128 per session) and replaces it on the next draw that finds it ready.
* Pipelines are flexible render pipelines built by the pipe compiler like recompiled ones (§7; `COMPILE_DEFERRED |
  COMPILE_AHEAD` for uber pipelines: queued ahead of other background jobs and the shader cache preload, behind the jobs
  the RSX thread waits for, built at emulation QoS) and are never waited for. They do not use the pipeline archive (§9:
  the fragment functions are specialized with function constants). A render state whose unspecialized interpreter pipeline exists (same libraries, features
  and fixed state, e.g. another colour format) gets its pipeline by inline specialization and draws with it at once.
  `preload()` (renderer start) queues the libraries and the uber pipeline of the most common state (one BGRA8
  attachment). A draw whose interpreter pipeline is not built yet is skipped (the only remaining skip besides
  unsupported programs). Key: the recompiled pipeline properties with depth/stencil format cleared and colour writes
  past the program's outputs off (as `validate_pipeline_properties`), libraries, features.
* Bindings (fit the stage limits: fragment 5 buffers, 64 textures, 16 samplers): vertex = the recompiled layout
  (streams 0-2, context 3, predicate 4, constants 5 or instancing tables 5/6) + instruction block 7; fragment = context
  0, texture parameters 1, stipple 2, instruction block 3, separate image arrays 4-9 (2D/1D x16, 3D x16, cube x16,
  depth compare 2D x8, depth compare cube x4, stencil mirror x4) and the 16 units' samplers 10. Colour arrays are
  indexed by unit; the depth compare and stencil arrays are compacted (slot = rank of the unit among the draw's units
  of that kind). Unused elements hold placeholders of the right type. Every element is declared to the hazard tracker
  (§3). Instruction blocks (uniform buffers): vertex = 16-byte header (base address, entry, output mask, two-sided
  lighting) + ucode; fragment = 32-byte header (shader control with the register count raised to what the program
  uses, texture dimensions, fp ctrl, MRT count, texcoord control, shadow / redirected units, compacted slot masks) +
  ucode, re-uploaded when the program, its state or its embedded constants change. Transform constants are uploaded
  whole (8 KiB) for the interpreter; switching between the interpreter and recompiled programs re-uploads them (and the
  fragment constants when leaving the interpreter).
* Exactness: programs are analysed (cached by ucode hash + export configuration) and left to the recompiler when the
  interpreter cannot run them as the recompiler would. Never interpreted: vertex texture fetch, multisampled texture
  sampling, programmable blending (§4), emulated depth compare, the depth bounds test in the program (§4), depth
  compare on 3D units or depth-as-RGBA8 on cube/3D units, more than 8/4/4 depth compare 2D/cube/stencil mirror units,
  register indices outside the register file,
  branch targets outside the program or backwards, literal constants the interpreter would execute, undefined input
  registers, nested IF/ELSE or loops, loops with a start/step, input registers indexed by the loop counter, registers
  read with the other precision than written (h/r aliasing), TEXBEM/TXPBEM/BEMLUM/TIMESWTEX/POW and undefined
  opcodes, vertex CLI/PSH/POP/undefined opcodes and BRA (jump by address register: its target is unknown to the RSX
  program analysis, so it may lie outside the uploaded microcode). Each reason is logged once. Parity fixes in the shared GLSL:
  two-sided colour selection, fog input, COL0/COL1 clamping, precision modifiers, PKG/UPG sRGB, NRM of a zero vector,
  BEM, TXD (Metal texture hooks), RSQ/LG2 clamps; the Metal wrapper adds the recompiler's texel processing, shadow and
  Z24X8 reads, shader-side LOD bias (pre-Apple10), the ROP epilogue, depth export clamp, z-clip transform, invariant
  position and the conditional rendering predicate.
* Telemetry: renderer start logs that the interpreter is enabled and, once its first pipeline exists, "shader
  interpreter ready N ms after renderer start" with its job's time split (GLSL -> MSL, MTLLibrary, pipelines); the
  30-second statistics add a line ("shader interpreter") with
  interpreted draws per frame, skipped draws by reason and the interpreter pipelines built (uber / specialized /
  failed / total).

## 9. Pipeline binary archive (MTLPipelineArchive.cpp)

* Every pipeline but the render pipeline specializations and the shader interpreter's is built by a capturing
  `MTL4Compiler` with an `MTL4PipelineDataSetSerializer`, and every compile passes the archives of earlier sessions as
  `lookupArchives` (once they are ready, below), so known binaries are reused instead of compiled.
* First lookups: at boot, the first builds with lookups (the interpreter's uber pipeline on a worker, the RSX thread's
  first overlay pipeline) returned together only after 1.6 / 6.8 / 13.3 s with 26 / 44 / 64 MiB of archives (1.5-2 s
  and no RSX stall without archives). Now the archive thread makes the first lookup itself (a trivial compute pipeline
  on a compiler of its own, at user-initiated QoS) and logs "ready for lookups N ms after renderer start (the first
  lookup took N ms)". Until then builds compile without lookups (still recorded), `is_full_state_archived` is false,
  and the shader cache preload waits (`pipeline_archive_ready`). Pipelines of functions specialized with function
  constants (`uses_pipeline_archive`: only the interpreter's, whose uber shader is by far the largest function) are
  never looked up nor recorded. Any pipeline build of 2 s or more is logged (thread, with/without lookups).
* A background thread (utility QoS; user-initiated for the final write at shutdown) retires the capturing compiler
  periodically and writes the retired serializer once (temp file, then rename). Files:
  `<ppu cache>/shaders_cache/metal_pipeline_archive/<fast|precise>/`; `identity.txt` pins the archive format version
  (bumped when the generated MSL, the compile options or the pipeline descriptors change for every shader; 6 today),
  the OS build and the GPU. `RPCS3_METAL_PIPELINE_ARCHIVE=0` disables it.
* Pipeline keys (bookkeeping: a lookup matches the descriptor) hash each shader's MSL once (`shader::msl_hash()`, a
  specialization's is its base's) plus the function constant values of specialized shaders and the state.
* Flexible render pipelines (§7): unspecialized and full-state pipelines are built here (captured, looked up). The
  specializations are not: `newRenderPipelineStateBySpecialization` takes no task options (no lookupArchives) and no
  descriptor can look one up, so they use the device's non-capturing compiler and the archives don't grow with them.
* Full state is preferred when the archive has it, but a blind lookup would compile in full on a miss: each archive
  file gets a `<file>.keys` sidecar listing the keys of the full-state render pipelines its serializer recorded
  (written after the file is in place, deleted with it; a missing/broken one only means those pipelines are specialized
  again). The union over the files opened at boot answers `is_full_state_archived(key)`. So heavily drawn pipelines
  compiled with full state in the background come back with full state from the next boot on.
* Runtime fallback: if a serializer holding unspecialized pipelines cannot be written, they are no longer recorded
  (`unspecialized-not-recorded.txt`, removed when the identity changes) and the archive keeps everything else.
* Shutdown waits up to 3 s for a write in progress. A write that takes longer keeps running; it is joined before
  the next archive opens the directory and at process exit.

## 10. CPU encoding cost: command memory, argument tables, views, residency

* Command memory (mtlutils/command_allocators.h). MTL4CommandAllocators are a pool shared by every command list, not
  owned by the lists: `command_list::begin()` takes one (the most recently used first), `submit()` hands it back
  tagged with the submission's timeline value, and it is reset and reused once that value is reached. A recording that
  is dropped without a commit returns its allocator at the next `begin()`. An allocator keeps the memory of its
  largest recording across `reset()`, so only the allocators of the recordings in flight exist: idle ones are released
  after 2 s, and once a second the largest idle one is released if it is over 32 MiB and more than 4x the median (an
  exceptional recording ballooned it). The ring size (`MTL_MAX_ASYNC_CB_COUNT`) no longer multiplies command memory.
* Argument tables and encoder state. `glsl::program::bind()` is the only writer of the command lists' argument tables
  and of pipeline states / tables on encoders (and declares what the slots give access to, §3). Each table has a
  shadow (`argument_table_shadow`, invalidated when a list begins) and a slot is only written when its value changes;
  `render_bindings()` / `compute_bindings()` (reset when an encoder begins) skip setting the same pipeline state or
  table again on an encoder. Anything else that writes a table must go through the shadow. The renderer's main pass
  caches the rest of its encoder state in `m_encoder_state` (depth-stencil state, rasterizer state, viewport/scissor,
  bias, bounds, stencil reference, blend colour): every path that sets such state directly on the main pass (the
  in-pass clear quad) invalidates what it overwrote.
* Views. Views that are cached for the lifetime of their image (`viewable_image::get_view`, `get_subresource_view`,
  `image_view::as`) stay MTLTexture view objects. A view needed by one recording only (the source of a scaled blit
  from an image without a view cache: typeless helpers, the blit scratch) is `image_view::make_transient()`: an entry
  of an MTL4 texture view pool (`setTextureView`, no allocation). It has a resource ID but no `MTL::Texture`, so it is
  only bound through argument tables. Its owner disposes of it through the GC like any view, and the pool entry is
  reused once it is destroyed, i.e. after the GPU executed the commands that use it. Argument table shadows stay
  correct across entry reuse because they are invalidated whenever a list begins recording.
* Residency. One global residency set, attached to both queues (plus the CAMetalLayer's set). `command_list::submit()`
  calls `commit_residency()`, which commits only when allocations were added since the last commit (the submission
  may use them) or when removals are overdue (64 pending, or the oldest is 50 ms old): removals only give memory back,
  so they ride along with the next commit instead of costing one per submission. `requestResidency()` is not used: it
  applies to the whole set, and the RSX creates surfaces right before their first use, so it would only move the
  residency work of the next commit onto the same thread a moment earlier.
* Telemetry (30 s stats line "encoding per frame"): submissions, argument table writes (skipped as redundant), pipeline
  state / table sets (skipped), transient views, residency commits.

## 11. Presentation, pacing and upscaling (MTLPresent.cpp)

* Metal 4 makes the queue that renders into a drawable wait for it (`waitForDrawable`), which would stall all later
  RSX work for up to a refresh. So `flip()` renders the final image (letterbox, upscaling, overlays) into the frame
  context's `present_image` on the main queue, and a separate present list on the present queue (the device's async
  queue) waits for that frame's timeline value, waits for the drawable, copies, commits, signals the drawable and
  presents. A frame context is recycled only after its present list has completed.
* Pacing (VSync Adaptive/Full): R = `NSScreen.minimumRefreshInterval` (1/120 s on ProMotion), G = measured guest frame
  interval minus display back-pressure. Each frame is shown for k = max(1, floor(G/R + 0.25)) refreshes with
  `presentAfterMinimumDuration(k*R - R/2)`, so 60 fps on 120 Hz is every other refresh and 30 fps every 4th, with no
  1-2-3 refresh jitter. Fullscreen on an Adaptive-Sync screen uses `G - 0.5 ms`. VSync Off: `present()` without
  display sync. Every wait on the display (frame context, `nextDrawable`, hard-sync drain) is added to
  `m_present_pacing.blocked_time`, so waiting for the display is never mistaken for a slow guest frame.
* Pacing must never lower the frame rate the game can reach (it is a feedback loop: held frames delay flips, which the
  guest may wait on in ways the renderer cannot see). G is the lower quartile of the recent intervals, only snapped
  *down* to whole vblanks, floored at the frame limit. While waiting for a frame context the RSX thread keeps serving
  the guest (flush requests, labels held back for zcull reports). A flip that had to wait for the display shows its
  frame one refresh sooner (drains queued frames). While G paces the game slower than its frame limit, frames are paced
  at the frame limit for ~0.3 s every 2 s (doubling up to 32 s while the game really is slower); if the game keeps up,
  the history is dropped. The history is also dropped on presentation mode changes (fullscreen, refresh rate, VSync).
* The layer is opaque (guest alpha is meaningless; lets the compositor skip blending and allows direct-to-display in
  fullscreen), `framebufferOnly = false` (the present list writes the drawable with a copy), and `drawableSize` is
  only written when it changes (every write replaces the drawables and can stall `nextDrawable`).
* Screen properties are sampled on the main thread by `mtl::request_surface_update` (dispatch_async, never blocks);
  the RSX thread reads the cached copy.
* Output scaling "FSR" = MetalFX spatial upscaling (`MTL4FXSpatialScaler`) followed by RCAS sharpening
  (upscalers/rcas_pass, the FSR1 RCAS shader as a compute pass; strength 0 = off). Scalers are built on a worker
  thread and cached (4 sizes); bilinear is used until the scaler is ready, when the image is not upscaled, and for
  stereo 3D. The scaler's passes are external work for the hazard tracker (§3): only RCAS and the present passes that
  read its output wait for them. MetalFX temporal upscaling and frame interpolation are not usable: they need per-pixel
  motion vectors and a jittered projection, which PS3 games do not provide.
* The 30-second statistics (all "Metal: ..." notices, one line per subject): presentation, GPU busy (GPU time, uploads,
  pipeline waits and skipped draws), render passes (§4), encoding per frame (§10), barriers per frame (§3), waits for
  GPU work, RSX thread per guest frame, render pipelines and shader builds (§7), shader interpreter (§8, interpreter
  shader modes only); resources on their own interval.
* "RSX thread per guest frame": wall time per guest flip split into busy (the flipped frames' `rsx::frame_statistics_t`
  — `m_profiler` is always on —, `flip()`, the rest as "other") and idle (`rsx::g_sync_wait_stats`, Common/
  sync_wait_stats.hpp: FIFO out of commands, flip/other NV406E semaphore acquires, frame limiter), plus the PPU's HLE
  waits for the RSX (cellGcmCallback segment waits, cellGcmGetFlipStatus polls while a flip is pending). Little idle
  time while the PPU waits: the RSX thread limits the frame rate; idle mostly "waiting for commands": the guest does.

## 12. Metal gaps and the chosen workaround (owner in brackets)

| Gap | Workaround |
|---|---|
| No fragment/render-target barriers inside a pass | Programmable blending via `[[color(n)]]` (SPIRV-Cross maps `subpassInput`); set `backend_config.supports_programmable_blending = true` [S,R]. Feedback loops: same-pixel reads through framebuffer fetch when detectable, otherwise end the pass before sampling (count splits per frame, log in debug overlay) [R,T] |
| No ordering after the first draw of a pass (TBDR) | Every pass begins with the pass barriers; a draw needing more (vertex stage after earlier fragment work) gets it before the pass's first draw or splits the draw pass (§3) [R] |
| No depth framebuffer fetch | Depth read while bound: end the pass and sample the depth texture (VK "emulate_depth_compare" path) [R] |
| No D24S8 on Apple GPUs | Always `Depth32Float_Stencil8`; existing d24 gather/scatter compute kernels for memory transfers [T,C] |
| Depth bounds only on Apple10 | `setDepthTestBounds` when `caps().depth_bounds`, otherwise the fragment program discards fragments whose own depth is outside the pushed bounds (pixel-rate; exported r1.z after fs_main() for depth exporters), binding and copying nothing (§4) [R,S] |
| No hardware conditional rendering | Shader predicate path (`emulate_conditional_rendering()` = true) fed by `cs_aggregator` from visibility results [R,C] |
| Last provoking vertex | `supports_last_provoking_vertex = false` (smooth fallback) v1 [R] |
| Primitive restart always on | Common code rewrites restart index; widen u16->u32 if restart disabled and 0xFFFF appears [R] |
| Wide lines | 1px lines v1 (logged once) [R] |
| Only 3 border colors | `border_color_t` nearest enum [T,R] |
| No scaled blit | `mtl::copy_scaled_image` = sampled draw through the overlay blit pass [C,T] |
| MSAA writes from compute | Unresolve as per-sample fragment pass [C] |
| Stencil masks not dynamic | `MTLDepthStencilState` cache keyed on all depth/stencil state [R] |
| Logic ops | Not emulated yet: blending off, color written unchanged (logged once). Planned: framebuffer-fetch emulation in the fragment shader [S] |
| Y-up NDC | VK already flips via viewport; Metal: negate `gl_Position.y` in the vertex epilogue (SPIRV-Cross `flip_vert_y`) and keep top-left viewport origin [S,R] |
| 1D textures without mips | Decompiler emits 2D (height 1) for 1D samplers [S,T] |
| Texel buffer size | Vertex streams read via `texture_buffer` views; if > max width, bind a window (VK `window()` logic) [R] |
| Clears inside a pass | Full-frame clears end the draw pass and are deferred into the next pass's `loadAction=Clear`; scissored/masked clears are drawn as a quad in the draw pass (§4) [R,C] |

## 13. Component owners

* **Core** (done): mtlutils/*, MTLHelpers.{h,cpp} (runtime state), MTLResourceManager, MTLImpl.cpp, MTLDeviceQuery.h,
  MTLProgramPipeline.h (contract).
* **S — shaders & pipelines**: MTLProgramPipeline.cpp, MTLShaderCompiler.{h,cpp}, MTLPipelineCompiler.{h,cpp},
  MTLPipelineArchive.{h,cpp}, MTLFragmentProgram.{h,cpp}, MTLVertexProgram.{h,cpp}, MTLCommonDecompiler.{h,cpp},
  MTLProgramBuffer.h, MTLShaderInterpreter.{h,cpp}, MTLShaderInterpreterSource.cpp.
* **C — compute, overlays, resolve, upscaling**: MTLCompute.{h,cpp}, MTLOverlays.{h,cpp}, MTLResolveHelper.{h,cpp},
  upscalers/*.
* **T — textures & surfaces**: MTLFormats.{h,cpp}, MTLTexture.cpp, MTLTextureCache.{h,cpp},
  MTLRenderTargets.{h,cpp}, MTLDMA.{h,cpp}.
* **R — renderer core**: MTLGSRender.{h,cpp}, MTLGSRenderTypes.hpp, MTLDraw.cpp, MTLVertexBuffers.cpp,
  MTLPresent.cpp, MTLQueryPool.{h,cpp}, MTLRenderPass.{h,cpp}, MTLCommandStream.{h,cpp}, scratch heap.

## 14. Compile checking

No macOS toolchain in the dev container. `/home/claude/mtl-check.sh <files>` type-checks with clang against the real
metal-cpp headers and stub Apple SDK headers (RPCS3 warning-as-error set included). Every `.cpp` must pass it.
`.mm` files cannot be checked here — keep them tiny.

Include-order pitfalls found on the first real macOS builds:

- `<objc/runtime.h>` (pulled in by metal-cpp) declares a global `Method` typedef, and `gcm_enums.h` ends with a global
  `using namespace gcm;` (which contains `gcm::Method`). CMake force-includes `objc/runtime.h` into every file under
  `RSX/Metal/` so the order of includes doesn't matter. Code outside this folder must not include metal-cpp; UI code
  goes through `MTLDeviceQuery.h`, which declares `mtl::create_render_thread()`.
- Homebrew's `/opt/homebrew/include` is searched last (`-idirafter`, see `buildfiles/cmake/ForkMacOSHomebrew.cmake`)
  so Homebrew copies of bundled libraries (protobuf, libpng, ...) can't replace the bundled headers.

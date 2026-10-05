#pragma once

// Native Metal 4 renderer (port of VK/VKGSRender.h). Plain C++ (metal-cpp); includable from the Qt UI.

#include "upscalers/upscaling.h"

#include "mtlutils/buffer_object.h"
#include "mtlutils/data_heap.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"
#include "mtlutils/sampler.h"
#include "mtlutils/sync.h"

#include "MTLFrameInspector.h"
#include "MTLGSRenderTypes.hpp"
#include "MTLTextureCache.h"
#include "MTLRenderTargets.h"
#include "MTLOverlays.h"
#include "MTLProgramBuffer.h"
#include "MTLRenderPass.h"
#include "MTLQueryPool.h"
#include "MTLShaderInterpreter.h"

#include "Emu/RSX/GSRender.h"
#include "Emu/RSX/Common/sync_wait_stats.hpp"

#include <deque>
#include <functional>
#include <initializer_list>
#include <unordered_map>

using namespace mtl::upscaling_flags_; // clang workaround.

namespace CA
{
	class MetalLayer;
}

class MTLGSRender : public GSRender, public ::rsx::reports::ZCULL_control
{
private:
	enum frame_context_state : u32
	{
		dirty = 1
	};

	enum flush_queue_state : u32
	{
		ok = 0,
		flushing = 1,
		deadlock = 2
	};

	using vs_binding_table_t = decltype(MTLVertexProgram::binding_table);
	using fs_binding_table_t = decltype(MTLFragmentProgram::binding_table);

private:
	const MTLFragmentProgram* m_fragment_prog = nullptr;

	// Per fragment texture unit mip LOD bias, pushed to programs with requires_lod_bias (pre-Apple10 GPUs)
	std::array<f32, 16> m_fs_lod_bias{};
	const MTLVertexProgram* m_vertex_prog = nullptr;
	mtl::glsl::program* m_program = nullptr;
	mtl::glsl::program* m_prev_program = nullptr;
	mtl::pipeline_props m_pipeline_properties{};
	u64 m_pipeline_renderpass_key = umax;          // Attachment configuration m_pipeline_properties was built for
	mtl::rasterizer_state m_rasterizer_state{};
	mtl::encoder_state m_encoder_state{};

	const vs_binding_table_t* m_vs_binding_table = nullptr;
	const fs_binding_table_t* m_fs_binding_table = nullptr;

	mtl::texture_cache m_texture_cache;
	mtl::surface_cache m_rtts;

	std::unique_ptr<mtl::buffer> null_buffer;
	std::unique_ptr<mtl::buffer_view> null_buffer_view;

	// Placeholder depth textures for shadow samplers without a bound image (depth2d<>/depthcube<> cannot take colour views)
	// [0] = 2D (also 1D, declared as 2D), [1] = Cube
	std::array<std::unique_ptr<mtl::viewable_image>, 2> m_null_depth_textures;

	std::unique_ptr<mtl::upscaler> m_upscaler;

	// Graphics self-check: inspects presented frames and float render targets on the GPU, logs what looks broken
	std::unique_ptr<mtl::frame_inspector> m_frame_inspector;
	u64 m_frame_inspector_tag = 0; // Surfaces written after this shared tag are checked next
	output_scaling_mode m_output_scaling{output_scaling_mode::bilinear};

	std::unique_ptr<mtl::buffer> m_cond_render_buffer;
	u64 m_cond_render_sync_tag = 0;

	std::unique_ptr<mtl::buffer> m_host_object_data; // Host GPU label context (64 KiB, host_visible)

	shared_mutex m_sampler_mutex;
	atomic_t<bool> m_samplers_dirty = { true };
	std::unique_ptr<mtl::sampler> m_stencil_mirror_sampler;
	std::array<mtl::sampler*, rsx::limits::fragment_textures_count> fs_sampler_handles{};
	std::array<mtl::sampler*, rsx::limits::vertex_textures_count> vs_sampler_handles{};

	std::unique_ptr<mtl::buffer_view> m_persistent_attribute_storage;
	std::unique_ptr<mtl::buffer_view> m_volatile_attribute_storage;

	// Shader interpreter (DESIGN.md §8): draws whose pipeline is compiling ("Async Recompiler with Shader Interpreter"),
	// every draw it can render exactly ("Interpreter only")
	mtl::shader_interpreter m_shader_interpreter;
	mtl::data_heap m_vertex_instructions_buffer;     // Instruction blocks: header + microcode
	mtl::data_heap m_fragment_instructions_buffer;
	mtl::glsl::buffer_binding_info m_vertex_instructions_buffer_info{};
	mtl::glsl::buffer_binding_info m_fragment_instructions_buffer_info{};
	u32 m_interpreter_state = 0;       // Program state changes (pipeline_state bits) the instruction blocks lack
	bool m_interpreter_bound = false;  // The last program loaded was the interpreter

	std::pair<const vs_binding_table_t*, const fs_binding_table_t*> get_binding_table() const;

public:
	std::unique_ptr<mtl::vertex_cache> m_vertex_cache;
	std::unique_ptr<mtl::shader_cache> m_shaders_cache;

private:
	std::unique_ptr<mtl::program_cache> m_prog_buffer;

	// Device and presentation surface
	std::unique_ptr<mtl::render_device> m_device;
	mtl::timeline m_timeline;
	void* m_view = nullptr;                      // NSView of the game window
	CA::MetalLayer* m_metal_layer = nullptr;     // Backing layer of m_view (retained, released on teardown)
	bool m_layer_framebuffer_only = true;

	// Presentation (MTLPresent.cpp). The frame's work, including the passes that composite the output into the frame
	// context's present_image, runs on the main queue. A small present list on m_present_queue then waits for that
	// work and for the drawable, copies the image into the drawable and presents it (paced). Drawable waits therefore
	// never hold back the main queue (next frame, DMA readbacks the guest waits on).
	MTL4::CommandQueue* m_present_queue = nullptr; // Device async queue, or the main queue if there is none. Not owned
	mtl::timeline m_present_timeline;              // Signaled by the present lists

	// Written by drawable presented handlers (any thread), read by the RSX thread (telemetry)
	struct present_feedback_t
	{
		atomic_t<f64> refresh_interval{ 1. / 60. };  // Unit of the histogram
		atomic_t<f64> last_presented_time{ 0. };     // presentedTime of the latest displayed drawable
		std::array<atomic_t<u32>, 6> intervals{};    // On-screen time of displayed frames in refreshes: 1, 2, 3, 4, 5, 6+
		atomic_t<u32> dropped{ 0 };                  // Drawables that were never displayed

		void on_presented(f64 presented_time);
	};
	std::shared_ptr<present_feedback_t> m_present_feedback;

	struct present_pacing_t
	{
		f64 refresh_interval = 1. / 60.;   // R: fastest refresh interval of the window's screen
		bool variable_refresh = false;      // Adaptive-Sync/ProMotion screen (minimum != maximum refresh interval)
		bool fullscreen = false;
		u64 surface_serial = 0;             // Last mtl::surface_properties::serial applied
		u64 surface_request_time = 0;       // get_system_time() of the last mtl::request_surface_update

		u64 last_emu_flip_time = 0;         // get_system_time() of the previous guest flip
		u64 blocked_time = 0;               // Presentation back-pressure (us) since the previous guest flip
		u64 flip_blocked_start = 0;         // blocked_time when the current flip started
		u32 catch_up_frames = 0;            // Telemetry: frames shown one refresh shorter to drain queued frames
		std::array<f64, 16> guest_intervals{}; // Recent guest frame intervals without back-pressure (s), ring buffer
		u32 guest_interval_count = 0;       // Valid entries in guest_intervals
		u32 guest_interval_next = 0;        // Next write position in guest_intervals

		// Downward probe (update_pacing_probe): while the measured interval paces the game slower than its frame limit,
		// frames are paced at the frame limit for a moment now and then, to see whether the game keeps up
		u64 probe_start_time = 0;           // get_system_time() when the running probe started (0: no probe running)
		u32 probe_samples = 0;              // Guest frame intervals measured since the probe started
		f64 probe_from_slot = 0.;           // Presentation slot (s) before the probe
		u64 next_probe_time = 0;            // get_system_time() of the next probe (0: none scheduled)
		u64 probe_backoff = 0;              // Delay between probes (us), doubled by every probe that finds no faster pace

		f64 last_present_time = 0.;         // Media time of the previous present request
		f64 min_duration = 0.;              // Last presentAfterMinimumDuration argument (0: unpaced present)
		u32 slot_refreshes = 0;             // Last pacing slot in refreshes (0: unpaced or variable refresh)
		u64 stats_time = 0;                 // Telemetry rate limit (get_system_time())

		// Forgets the measured guest frame intervals and the probe state: pacing starts over at the frame limit
		void reset_guest_intervals()
		{
			guest_interval_count = 0;
			guest_interval_next = 0;
			probe_start_time = 0;
			probe_samples = 0;
			next_probe_time = 0;
			probe_backoff = 0;
		}
	} m_present_pacing;

	// Occlusion queries (visibility result buffers)
	std::unique_ptr<mtl::query_pool_manager> m_occlusion_query_manager;
	bool m_occlusion_query_active = false;
	rsx::reports::occlusion_query_info* m_active_query_info = nullptr;
	std::vector<mtl::occlusion_data> m_occlusion_map;

	shared_mutex m_secondary_cb_guard;
	mtl::command_buffer_chain<MTL_MAX_ASYNC_CB_COUNT> m_secondary_cb_list;

	mtl::command_buffer_chain<MTL_MAX_ASYNC_CB_COUNT> m_primary_cb_list;
	mtl::command_buffer_chunk* m_current_command_buffer = nullptr;

	// Main render pass (VK: renderpass + framebuffer)
	mtl::framebuffer_info m_draw_fbo{};       // Attachments of the draw pass (may be a superset of what the RSX binds)
	mtl::framebuffer_info m_rsx_fbo{};        // Exactly the surfaces the RSX layout binds (prepare_rtts)
	bool m_attachments_retained = false;      // m_draw_fbo keeps attachments the RSX layout dropped (see prepare_rtts)
	bool retain_pass_attachments(const mtl::framebuffer_info& fbo);
	void release_retained_attachments(const mtl::framebuffer_info& bound);
	std::vector<mtl::image*> pass_images() const;
	bool is_retained_attachment(const mtl::image* image) const;
	void drop_retained_attachments();
	void drop_retained_attachments_if_sampled();
	std::array<const MTL::Texture*, 5> m_draw_fbo_textures{};  // Textures m_draw_pass_desc was built with
	mtl::ref<MTL4::RenderPassDescriptor> m_draw_pass_desc;
	const MTL::Buffer* m_draw_pass_visibility_buffer = nullptr;
	mtl::render_pass_tracker m_render_pass;
	u64 m_current_renderpass_key = 0;
	std::vector<mtl::image*> m_fbo_images;
	u32 m_render_pass_splits = 0;               // Passes split this frame (feedback loops, depth reads)

	// Depth-stencil state objects (Metal has no dynamic stencil masks)
	std::unordered_map<u64, mtl::ref<MTL::DepthStencilState>> m_depth_stencil_states;
	u64 m_last_depth_stencil_key = 0;                                 // Last get_depth_stencil_state() lookup (owned by the map)
	MTL::DepthStencilState* m_last_depth_stencil_state = nullptr;

	sizeu m_swapchain_dims{};
	bool swapchain_unavailable = false;
	bool should_reinitialize_swapchain = false;

	u64 m_last_heap_sync_time = 0;
	u32 m_texbuffer_view_size = 0;

	mtl::data_heap m_attrib_ring_info;                         // Vertex data
	mtl::data_heap m_fragment_constants_ring_info;             // Fragment program constants
	mtl::data_heap m_transform_constants_ring_info;            // Transform program constants
	mtl::data_heap m_fragment_env_ring_info;                   // Fragment environment params
	mtl::data_heap m_vertex_env_ring_info;                     // Vertex environment params
	mtl::data_heap m_fragment_texture_params_ring_info;        // Fragment texture params
	mtl::data_heap m_vertex_layout_ring_info;                  // Vertex layout structure
	mtl::data_heap m_index_buffer_ring_info;                   // Index data
	mtl::data_heap m_texture_upload_buffer_ring_info;          // Texture upload heap
	mtl::data_heap m_raster_env_ring_info;                     // Raster control such as polygon and line stipple
	mtl::data_heap m_instancing_buffer_ring_info;              // Instanced rendering data (constants indirection table + instanced constants)

	// Ring buffers are bound whole (by GPU address); shaders index them with the dynamic offsets below.
	mtl::glsl::buffer_binding_info m_instancing_indirection_buffer_info{};
	mtl::glsl::buffer_binding_info m_instancing_constants_array_buffer_info{};

	u64 m_xform_constants_dynamic_offset = 0;          // We manage transform_constants dynamic offset manually to alleviate performance penalty of doing a hot-patch of constants.
	u64 m_vertex_env_dynamic_offset = 0;
	u64 m_vertex_layout_dynamic_offset = 0;
	u64 m_fragment_constants_dynamic_offset = 0;
	u64 m_fragment_env_dynamic_offset = 0;
	u64 m_texture_parameters_dynamic_offset = 0;
	u64 m_stipple_array_dynamic_offset = 0;

	std::unique_ptr<rsx::data_heap::bulk_allocator<256, 96>> m_vertex_env_allocator;
	std::unique_ptr<rsx::data_heap::bulk_allocator<256, 16>> m_transform_constants_allocator;
	std::unique_ptr<rsx::data_heap::bulk_allocator<256, 16>> m_fragment_constants_allocator;

	std::vector<mtl::frame_context_t> m_frame_context_storage;
	u32 m_max_async_frames = 0u;
	// Temp frame context to use if the real frame queue is overburdened. Only used for storage
	mtl::frame_context_t m_aux_frame_context;

	u32 m_current_queue_index = 0;
	mtl::frame_context_t* m_current_frame = nullptr;
	std::deque<mtl::frame_context_t*> m_queued_frames;

	MTL::Viewport m_viewport{};
	MTL::ScissorRect m_scissor{};

	// Scissor clamped to the render target (Metal rejects rectangles outside the attachments). Empty -> draws skipped.
	MTL::ScissorRect get_clamped_scissor() const;

	std::vector<u8> m_draw_buffers;

	shared_mutex m_flush_queue_mutex;
	mtl::flush_request_task m_flush_requests;

	ullong m_last_cond_render_eval_hint = 0;

	// Early submission at render pass boundaries (prepare_rtts): recorded work goes to the GPU every ~1.5 ms instead of
	// waiting for the next flush, so the GPU starts sooner and a later readback/report waits for less
	u32 m_draws_since_submit = 0;
	u64 m_last_submit_us = 0;
	u64 m_early_submits = 0;

	// Offloader thread deadlock recovery
	rsx::atomic_bitmask_t<flush_queue_state> m_queue_status;
	utils::address_range32 m_offloader_fault_range;
	rsx::invalidation_cause m_offloader_fault_cause;

	mtl::draw_call_t m_current_draw {};

	std::unique_ptr<mtl::viewable_image> m_overlay_recording_img;

	//Vertex layout
	rsx::vertex_input_layout m_vertex_layout;

	// A draw whose pipeline is still compiling may wait for it, within a per-frame budget, when its shaders are
	// already compiled (see load_program)
	static constexpr u64 async_compile_wait_budget_us = 8'000;
	u64 m_async_compile_wait_spent_us = 0;
	// Draws the shader interpreter cannot run wait for their recompiled pipeline (load_program), within these budgets
	// One 60 Hz frame per draw, 1.5 per frame: a longer wait is a visible hitch (was 250 ms / 600 ms, then 40 / 60 ms:
	// shader bursts at the start of a game still stalled for several frames in a row). The draw is skipped for this
	// frame when the budget runs out, as the Vulkan backend does for every compiling pipeline.
	static constexpr u64 unsupported_wait_draw_budget_us = 16'000;
	static constexpr u64 unsupported_wait_frame_budget_us = 24'000;
	u64 m_unsupported_wait_spent_us = 0;

	// Pipeline telemetry, reported and reset with the presentation statistics (MTLPresent.cpp)
	u32 m_skipped_draws = 0;      // Draws skipped because their pipeline was still compiling or could not be built
	u64 m_pipeline_wait_us = 0;   // Time load_program() waited for pipelines that were being compiled
	u32 m_interpreter_draws = 0;  // Draws drawn by the shader interpreter
	std::array<u32, static_cast<u32>(mtl::shader_interpreter::skip_reason::count)> m_interpreter_skips{}; // Skipped draws by reason
	u32 m_depth_bounds_draws = 0; // Draws whose fragment program performed the depth bounds test
	u32 m_depth_copies = 0;       // Copies of the depth buffer made for shader reads during its pass (update_depth_copy)
	u64 m_preload_notification_time = 0; // update_shader_preload_notification

	// Shader reads of the draw's depth buffer read a copy of it, never the depth attachment of the pass they run in:
	// texture units sampling the depth buffer the draw pass attaches (soft particles, fog, lights reconstructing
	// positions) and depth compare emulation. (The depth bounds test without hardware support used to read it too;
	// it now tests the fragment's own depth, so it makes no copy.) Metal does not define shader reads of a texture
	// that is an attachment of the running render pass, and Apple GPUs do return wrong depth for some pixels of every
	// tile then, even when the pass never changed depth. The copy is made outside the pass (update_depth_copy) and used for as long as the depth plane of its source
	// is unchanged: `depth_tag` is the source's render_target::content_tag when the copy was made, carried along across
	// the writes that leave depth alone (mark_attachment_writes); `stencil_tag` likewise for reads of the stencil plane.
	// Anything else that changes the tag (depth writes, depth clears, transfers, memory initialization, recycling) makes
	// the next such read copy again.
	struct depth_copy_t
	{
		std::unique_ptr<mtl::viewable_image> image;
		std::vector<std::unique_ptr<mtl::image_view>> views; // Views of `image` like the source's views texture units sample
		const mtl::render_target* source = nullptr;          // Compared, never dereferenced (content tags are never reused)
		u64 depth_tag = 0;   // source->content_tag for which the copy's depth plane is current
		u64 stencil_tag = 0; // source->content_tag for which its stencil plane is current (stencil views, stencil mirrors)

		// Feedback streak (MTLRenderTargets.h) of draws that read the copy and write depth: the next draws of the streak
		// (same material key) keep reading it while only they wrote depth since, as their reads of the attachment
		// memory did before: they see the depth from before the streak, the others see the streak's writes.
		u64 streak_key = 0;
		u64 streak_tag = 0;
	};
	depth_copy_t m_depth_copy;
	bool m_draw_reads_depth_copy = false; // The current draw binds a view of m_depth_copy (bind_texture_env)
	bool m_depth_bounds_notice_logged = false;
	bool m_depth_bounds_relayout_done = false; // The layout was evaluated again for an active depth bounds test (begin)

	// RSX thread time telemetry, reported and reset with the presentation statistics (MTLPresent.cpp): the flipped guest
	// frames' rsx::frame_statistics_t, the time spent in flip() and the RSX/guest sync waits (rsx::g_sync_wait_stats)
	struct rsx_time_stats_t
	{
		u32 frames = 0;                // Guest flips
		u64 draw_calls = 0;
		s64 setup_us = 0;
		s64 vertex_upload_us = 0;
		s64 texture_upload_us = 0;
		s64 draw_exec_us = 0;
		u64 flip_us = 0;               // flip()
		u64 display_wait_us = 0;       // Waits for drawables and frame contexts (present_pacing_t::blocked_time)
		rsx::sync_wait_snapshot waits; // rsx::g_sync_wait_stats when the window started
	} m_rsx_time_stats;

	// Resource usage telemetry (report_resource_usage)
	u64 m_resource_report_time = 0;

	bool m_wide_lines_warning_logged = false;
	u64 m_wide_lines_warning_time = 0;
	bool m_logic_op_warning_logged = false;
	bool m_flat_shading_warning_logged = false;
	u64 m_feedback_draw_key = 0; // Material key of the current draw (feedback streaks, see MTLRenderTargets.h)
	u64 m_fp_ucode_hash = 0;     // Instruction hashes (embedded constants excluded) of the current programs,
	u64 m_vp_ucode_hash = 0;     // updated when the RSX reloads them

public:
	u64 get_cycles() final;
	~MTLGSRender() override;

	MTLGSRender(utils::serial* ar) noexcept;
	MTLGSRender() noexcept : MTLGSRender(nullptr) {}

private:
	void prepare_rtts(rsx::framebuffer_creation_context context);

	void close_and_submit_command_buffer(const mtl::submit_info_t& submit_info = {});

	void flush_command_queue(bool hard_sync = false, bool do_not_switch = false);
	void queue_swap_request();
	void frame_context_cleanup(mtl::frame_context_t *ctx);
	void advance_queued_frames();
	void present(mtl::frame_context_t *ctx);
	bool reinitialize_swapchain();
	void configure_metal_layer();
	void assert_metal_layer_state();

	// Presentation pacing (MTLPresent.cpp)
	void update_present_pacing(bool emu_flip);
	void update_pacing_probe(u64 now);
	f64 get_frame_limit_interval() const;
	f64 get_measured_frame_interval() const;
	f64 get_guest_frame_interval() const;
	f64 get_pacing_slot(f64 guest_interval, u32* refreshes = nullptr) const;
	void present_drawable(mtl::frame_context_t* ctx);

	// Waits for a frame context to leave the present queue (display back-pressure) while serving the guest
	void wait_for_frame_context(mtl::frame_context_t* ctx);
	void serve_guest_during_display_wait();

	mtl::viewable_image* get_present_source(mtl::present_surface_info* info, const rsx::avconf& avconfig);

	// Render pass management
	void update_render_pass_descriptor();
	void begin_render_pass();
	bool draw_reads_deferred_clear() const;
	void close_render_pass(mtl::pass_end_reason reason);
	void invalidate_render_pass();
	void split_render_pass(mtl::pass_split_reason reason);
	bool is_render_pass_open() const;

	// Feedback loops (a draw samples a bound attachment). Tile-based GPUs write attachments to memory when the pass
	// ends, so a read only needs a pass split when the sampled surface was written by the pass that is still open,
	// and not only by earlier draws of the feedback streak the current draw belongs to.
	// `depth_stencil`: the depth buffer counts as written (feedback bookkeeping). `ds_planes`: mtl::aspect_depth and/or
	// aspect_stencil, the planes whose contents may have changed (m_depth_copy; exact for depth, a superset for stencil)
	void mark_attachment_writes(const std::array<bool, 4>& color, bool depth_stencil, u32 ds_planes, bool from_draw = false);
	void update_feedback_streaks(const std::array<bool, 4>& color, bool depth_stencil);
	bool draw_samples_attachment(const mtl::render_target* surface) const;
	std::array<bool, 4> get_live_color_writes() const;
	bool colour_write_after_read(const std::array<bool, 4>& color, bool writer_samples_as_streak) const;
	mtl::render_target* find_bound_attachment(const mtl::image* image) const;
	mtl::pass_split_reason feedback_read_needs_split() const; // count: no split needed
	u64 get_feedback_draw_key() const;
	MTL4::RenderCommandEncoder* get_render_encoder() const;
	void on_render_pass_begin(MTL4::RenderCommandEncoder* encoder);

	MTL::DepthStencilState* get_depth_stencil_state(u64 key);

	void update_draw_state();
	void check_present_status();
	void check_heap_status();

	// Rate-limited log line with the sizes of everything that can grow during a session (caches, pools, heaps, GC)
	void report_resource_usage();

	mtl::vertex_upload_info upload_vertex_data();
	rsx::simple_array<u8> m_scratch_mem;

	bool load_program();
	void load_program_env();

	// Depth bounds test (DESIGN.md §4): the bounds as the hardware test takes them and whether the bound program performs
	// the test in its fragment shader
	std::pair<f32, f32> get_clamped_depth_bounds() const;
	bool draw_reads_depth_bounds() const;

	// Shader reads of the draw's depth buffer (DESIGN.md §4, m_depth_copy): the copy, made first when the planes read
	// (depth, and stencil if `stencil`) are stale (`exact`: the depth plane as it is now, even for a feedback streak),
	// and the view of the copy to bind instead of `view` when `view` reads the depth attachment of the draw pass
	// (`stencil`: the draw also reads its stencil through the view's image)
	mtl::viewable_image* update_depth_copy(mtl::render_target* ds, bool stencil, bool exact = false);
	mtl::image_view* redirect_depth_attachment_read(mtl::image_view* view, bool stencil = false);
	void update_vertex_env(u32 id, const mtl::vertex_upload_info& vertex_info);
	void upload_transform_constants(const rsx::io_buffer& buffer);

	void load_texture_env();
	bool bind_texture_env();
	bool bind_interpreter_texture_env();

	mtl::image_view* get_null_texture_view(rsx::texture_dimension_extended type, bool is_depth);

public:
	void init_buffers(rsx::framebuffer_creation_context context, bool skip_reading = false);
	void set_viewport();
	void set_scissor(bool clip_viewport);
	void bind_viewport();

	// Sync
	void write_barrier(u32 address, u32 range) override;
	void sync_hint(rsx::FIFO::interrupt_hint hint, rsx::reports::sync_hint_payload_t payload) override;
	bool release_GCM_label(u32 type, u32 address, u32 data) override;

	void begin_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	void end_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	bool check_occlusion_query_status(rsx::reports::occlusion_query_info* query) override;
	void get_occlusion_query_result(rsx::reports::occlusion_query_info* query) override;
	void discard_occlusion_query(rsx::reports::occlusion_query_info* query) override;

	// External callback in case we need to suddenly submit a commandlist unexpectedly, e.g in a violation handler
	void emergency_query_cleanup(mtl::command_list* commands);

	// External callback to handle out of video memory problems
	bool on_vram_exhausted(rsx::problem_severity severity);

	// Conditional rendering
	void begin_conditional_rendering(const std::vector<rsx::reports::occlusion_query_info*>& sources) override;
	void end_conditional_rendering() override;

	// Host sync object (host GPU labels are not supported on Metal; kept for API parity)
	void on_guest_texture_read(mtl::command_list& cmd);

	// GRAPH backend
	void patch_transform_constants(rsx::context* ctx, u32 index, u32 count) override;

	// Misc
	bool is_current_program_interpreted() const override;
	std::pair<std::string, std::string> get_programs() const override;

	// The timeline signaled by every submission of this renderer
	mtl::timeline& get_timeline() { return m_timeline; }

protected:
	void clear_surface(u32 mask) override;
	void begin() override;
	void end() override;
	void emit_geometry(u32 sub_index) override;

	// RSX_SHADER_CONTROL_DEPTH_BOUNDS_TEST (+ ROP_MULTISAMPLED for a multisampled depth buffer) for draws with an active
	// depth bounds test, a bound depth buffer and no hardware depth bounds test
	rsx::flags32_t get_backend_fragment_program_export_config() const override;

	void on_init_thread() override;
	void on_exit() override;

	// Shader cache preload (DESIGN.md §6): reads and decompiles, then queues the pipelines for background builds
	void preload_shader_cache();
	void update_shader_preload_notification(); // Flip: shader compilation hint while the preload runs

	void flip(const rsx::display_flip_info_t& info) override;

	void renderctl(u32 request_code, void* args) override;

	void do_local_task(rsx::FIFO::state state) override;
	bool scaled_image_from_memory(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate) override;
	void notify_tile_unbound(u32 tile) override;

	bool on_access_violation(u32 address, bool is_writing) override;
	void on_invalidate_memory_range(const utils::address_range32 &range, rsx::invalidation_cause cause) override;
	void on_semaphore_acquire_wait() override;
	f32 get_gpu_utilization_pct() override;

	// Rolling GPU utilization state (deltas of mtl::peek_gpu_busy_ns over wall time)
	u64 m_gpu_util_last_busy_ns = 0;
	u64 m_gpu_util_last_time_us = 0;
	f32 m_gpu_util_cached_pct = -1.f;
	f32 m_gpu_util_ema = -1.f;       // Driver-reported GPU load, smoothed (get_gpu_utilization_pct)
	u64 m_gpu_util_ema_time_us = 0;
};

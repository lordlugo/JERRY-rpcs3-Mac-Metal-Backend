#pragma once

#include "mtl_api.h"
#include "device.h"
#include "sync.h"
#include "command_allocators.h"
#include "Utilities/mutex.h"

#include <array>
#include <initializer_list>
#include <memory>
#include <span>
#include <vector>

namespace mtl
{
	class buffer;
	class image;
	class image_view;
	struct buffer_view;

	enum argument_table_slot : u32
	{
		table_vertex = 0,
		table_fragment = 1,
		table_compute = 2,
		table_count = 3
	};

	// Why a render pass had to be ended early (telemetry)
	enum class pass_split_reason : u32
	{
		read_after_write = 0, // a draw samples an attachment written by the open pass
		write_after_read,     // strict mode only: a write follows reads of the same attachment
		read_through_copy,    // a draw samples a converted/copied view of a bound surface (made outside the pass)
		vertex_read,          // a draw needs a queue barrier the open pass did not begin with (its vertex stage reads what
		                      // fragment work of an earlier pass wrote), after the pass recorded draws (command_list::
		                      // pass_split_required: on a tile-based GPU a pass's ordering is fixed before its first draw)
		depth_copy,           // a draw reads the depth buffer the pass attaches: its copy is made outside the pass
		count
	};

	// Why the renderer's draw pass (the pass rendering into the RSX surfaces) ended (telemetry)
	enum class pass_end_reason : u32
	{
		framebuffer_change = 0, // the RSX switched to other surfaces
		clear,                  // a full-frame RSX clear (a load action of the next pass)
		feedback,               // a draw reads what the pass wrote, or orders its vertex work after it (pass_split_reason)
		transfer,               // a copy, fill, upload or compute dispatch had to be recorded
		other_pass,             // another render pass (scaled copy, clear, MSAA resolve, overlay)
		query_pool,             // the occlusion query pool was replaced (its visibility buffer is pass state)
		submit,                 // the command list was submitted
		count
	};

	// What the RSX thread was doing when a draw pass ended or another pass began (telemetry). Set with
	// pass_context_scope around the renderer's work; the innermost scope wins.
	enum class pass_context : u32
	{
		other = 0,
		surface_setup,  // framebuffer setup: surface creation, initialization, inheritance, Write Color Buffers copies
		texture_setup,  // the draw's textures: texture cache uploads, copies, gathers, MSAA resolves
		clear,          // RSX clears
		blit,           // RSX image transfers (NV3089)
		query,          // occlusion queries, conditional rendering
		readback,       // guest memory reads of GPU data (flush requests, DMA)
		present,        // flip: output composition, upscaling, overlays
		count
	};

	class pass_context_scope
	{
		pass_context m_previous;

	public:
		explicit pass_context_scope(pass_context context);
		~pass_context_scope();

		pass_context_scope(const pass_context_scope&) = delete;
		pass_context_scope& operator=(const pass_context_scope&) = delete;
	};

	// Render pass structure events (telemetry)
	enum class pass_event : u32
	{
		clear_in_pass = 0,         // scissored or channel-masked RSX clear drawn as a quad in the draw pass
		clear_folded,              // deferred clear folded into the load action of a pass attaching the texture
		clear_pass,                // clear-only pass recorded for deferred clears that no pass folded
		readback_not_speculated,   // Write Color/Depth Buffers: no speculative readback of a surface that stays bound
		clear_kept_pass_open,      // full-frame clear of some planes only, drawn as a quad instead of ending the draw pass
		attachments_retained,      // RSX layout dropped attachments of the open draw pass; the pass kept them (no split)
		retained_attachment_sampled, // a draw sampled a retained attachment: the pass ended and dropped it
		count
	};

	void count_pass_event(pass_event event);

	// Telemetry for the renderer's periodic log line: GPU time of committed work (union of the start-end intervals
	// reported by Metal 4 commit feedback, so overlapping work is not counted twice) and render passes begun.
	struct gpu_stats_t
	{
		static constexpr u32 reason_count = static_cast<u32>(pass_end_reason::count);
		static constexpr u32 context_count = static_cast<u32>(pass_context::count);

		u64 busy_ns = 0;
		u64 draw_render_passes = 0;     // passes of the renderer's framebuffer (the rest: copies, clears, overlays)
		std::array<std::array<u64, context_count>, reason_count> draw_pass_ends{}; // [reason][context]
		std::array<u64, context_count> other_passes{};                            // passes that are not draw passes
		std::array<u64, static_cast<u32>(pass_event::count)> pass_events{};
		u64 attachment_load_bytes = 0;  // attachments loaded from memory at pass begin (loadAction Load)
		u64 attachment_store_bytes = 0; // attachments stored to memory at pass end (storeAction Store)
		u64 feedback_splits = 0;
		std::array<u64, static_cast<u32>(pass_split_reason::count)> splits_by_reason{};
		u64 feedback_reads_in_pass = 0; // feedback reads served without a split (see render_target feedback streaks)
		u64 uploads_ahead = 0;          // image uploads from memory recorded into a list's prologue (no pass end)
		u64 uploads_inline = 0;         // image uploads from memory recorded inline
		u64 uploads_inline_split = 0;   // ... of which ended an open render pass
		u64 submissions = 0;            // command list commits (any queue)
		u64 table_writes = 0;           // argument table slots written (buffer addresses, textures, samplers)
		u64 table_writes_skipped = 0;   // ... not written because the table already held the value
		u64 state_sets = 0;             // pipeline states and argument tables set on an encoder
		u64 state_sets_skipped = 0;     // ... not set because the open encoder already had them

		// Hazard tracking (command_list)
		u64 ordering_points = 0;        // compute commands recorded (each began with the barriers it needed)
		u64 ordering_points_free = 0;   // ... that needed no barrier: they may run concurrently with earlier work
		u64 queue_barriers = 0;         // consumer queue barriers encoded
		u64 pass_barriers = 0;          // ... of which the two every render pass begins with
		u64 first_draw_barriers = 0;    // ... of which in a render pass before its first draw, for what the draw reads
		u64 late_barriers = 0;          // ... of which after the first draw of a pass (a contract violation; stays 0)
		u64 encoder_barriers = 0;       // intra-encoder barriers between conflicting commands of a compute encoder
		u64 full_barriers = 0;          // commands whose accesses were not declared (full barrier fallback)
		u64 submissions_waited = 0;     // submissions that waited for a conflicting earlier one still in flight
		u64 submissions_overlapped = 0; // submissions with earlier ones in flight and no conflict with them
	};

	gpu_stats_t get_gpu_stats_and_reset();
	u64 peek_gpu_busy_ns(); // Non-destructive read of accumulated GPU busy time (does not disturb the telemetry reset)
	void count_feedback_split(pass_split_reason reason = pass_split_reason::read_after_write);
	void count_feedback_read_in_pass();
	void count_image_upload(bool ahead_of_pass, bool ended_pass);

	// Log text: where the render passes of `frames` frames came from and what they cost (see gpu_stats_t)
	std::string describe_render_passes(const gpu_stats_t& stats, u32 frames);

	// Values of a deferred attachment clear (command_list::defer_clear)
	struct attachment_clear_value
	{
		MTL::ClearColor color{};
		f64 depth = 1.;
		u32 stencil = 0;
	};

	// A range of one resource that a GPU command reads or writes (hazard tracking, see command_list).
	// Buffers are tracked by MTL::Buffer with a byte range, textures by their root MTL::Texture (views by the texture they
	// were created from, textures made from a buffer by that buffer) with a mask of subresources.
	struct gpu_access
	{
		enum kind_t : u8
		{
			none = 0,    // nothing (e.g. an empty binding)
			buffer_range,
			texture_subresources,
			unknown      // an access that cannot be described: the command gets a full barrier
		};

		const void* resource = nullptr;
		u64 lo = 0;    // buffer_range: first byte; texture_subresources: subresource mask (see subresource_mask())
		u64 hi = 0;    // buffer_range: end byte (exclusive)
		kind_t kind = none;
		bool write = false;

		bool operator==(const gpu_access&) const = default;

		gpu_access as_write() const
		{
			gpu_access result = *this;
			result.write = true;
			return result;
		}

		static gpu_access undeclared()
		{
			return { .kind = unknown, .write = true };
		}
	};

	// Subresource mask of mip levels [base_level, base_level + level_count) x array layers [base_layer, ...): bit
	// (level % 8) * 8 + (layer % 8). Levels and layers beyond 8 alias lower ones (only ever adds overlaps). 3D textures
	// have one layer (their slices belong to the level).
	u64 subresource_mask(u32 base_level, u32 level_count, u32 base_layer, u32 layer_count);

	gpu_access read_buffer(const mtl::buffer* buf, u64 offset = 0, u64 length = umax);
	gpu_access write_buffer(const mtl::buffer* buf, u64 offset = 0, u64 length = umax);
	gpu_access read_buffer(const MTL::Buffer* buf, u64 offset = 0, u64 length = umax);
	gpu_access write_buffer(const MTL::Buffer* buf, u64 offset = 0, u64 length = umax);
	gpu_access read_buffer(const mtl::buffer_view* view); // The viewed range of the parent buffer
	// Levels / layers of an image (umax counts: to the end)
	gpu_access read_image(const mtl::image* img, u32 base_level = 0, u32 level_count = umax, u32 base_layer = 0, u32 layer_count = umax);
	gpu_access write_image(const mtl::image* img, u32 base_level = 0, u32 level_count = umax, u32 base_layer = 0, u32 layer_count = umax);
	gpu_access read_image(const mtl::image_view* view); // The subresources of the view in the texture it was made from
	// Any MTL::Texture (a view, a drawable, a texture made from a buffer): resolved with Metal queries, whole texture
	gpu_access read_texture(const MTL::Texture* tex);
	gpu_access write_texture(const MTL::Texture* tex);

	// Opaque state that is not a buffer or texture of ours (e.g. the internal resources of a MetalFX scaler): every
	// access is a write of the whole object
	gpu_access write_object(const void* object);

	// State glsl::program::bind() (the only code setting pipeline states and argument tables) set on the open render
	// encoder. Encoder state lasts until the encoder ends; reset whenever a render pass begins.
	struct render_encoder_bindings
	{
		// glsl::program whose pipeline state is set (0: none). Metal draws nothing without a pipeline state, and bind()
		// declares what the draw reads before it sets one: while this is 0, no draw was recorded in the open pass
		// (command_list::draw_access() may still encode barriers then).
		u64 program_uid = 0;
		u32 tables_set = 0;   // Bit (1 << argument_table_slot): that table is set for its stage
	};

	// Same for the open compute encoder (reset whenever a compute encoder begins). Copies and fills recorded into the
	// encoder do not change the dispatch state.
	struct compute_encoder_bindings
	{
		u64 program_uid = 0;
		bool table_set = false;
	};

	// Encoder state sets done / avoided by the caches above since the list last submitted (telemetry)
	struct encoder_state_counters
	{
		u64 sets = 0;
		u64 skipped = 0;

		// Returns `needed`, counting it
		bool count(bool needed)
		{
			(needed ? sets : skipped)++;
			return needed;
		}
	};

	// Contents of one argument table as last written through update_*(). Metal takes a snapshot of a table's bindings
	// when a draw/dispatch is encoded; the table itself is a plain object whose contents persist across encoders and
	// command buffers. So when every write goes through here (glsl::program::bind() is the only writer), a slot whose
	// value already matches needs no write. A table holds values (GPU addresses, resource IDs), not objects: an equal
	// value means identical GPU-visible state, provided the value was live when written. Sampler IDs are re-validated
	// against the live set at every bind (see sampler_liveness), so a destroyed sampler's ID is never written here.
	struct argument_table_shadow
	{
		static constexpr u32 max_textures = 64; // Table size, see command_list::create()

		std::array<MTL::GPUAddress, gpu_capabilities::max_buffers_per_stage> buffers{};
		std::array<u64, max_textures> textures{};                                 // MTL::ResourceID::_impl
		std::array<u64, gpu_capabilities::max_samplers_per_stage> samplers{};     // MTL::ResourceID::_impl
		u32 buffers_known = 0;   // Bit i: buffers[i] is what the table holds
		u64 textures_known = 0;
		u32 samplers_known = 0;

		// Slots written / skipped since the owning list last submitted (telemetry)
		u64 writes = 0;
		mutable u64 writes_skipped = 0;

		// Each returns true, and records the value, if the table does not hold it yet (the caller writes it then)
		bool update_buffer(u32 index, MTL::GPUAddress address)
		{
			return update(buffers[index], buffers_known, u32{1} << index, address);
		}

		bool update_texture(u32 index, MTL::ResourceID id)
		{
			return update(textures[index], textures_known, u64{1} << index, id._impl);
		}

		bool update_sampler(u32 index, MTL::ResourceID id)
		{
			return update(samplers[index], samplers_known, u32{1} << index, id._impl);
		}

		// Every slot unknown: the next bind writes all the slots it uses
		void invalidate()
		{
			buffers_known = 0;
			textures_known = 0;
			samplers_known = 0;
		}

		// Non-destructive read: what address the table holds for a buffer slot (false if never written)
		bool peek_buffer(u32 index, MTL::GPUAddress& address) const
		{
			const u32 bit = u32{1} << index;
			if (!(buffers_known & bit))
			{
				return false;
			}

			address = buffers[index];
			return true;
		}

		// The table is known to hold this sampler ID in this slot (no write needed)
		bool holds_sampler(u32 index, MTL::ResourceID id) const
		{
			const u32 bit = u32{1} << index;
			if (!(samplers_known & bit))
			{
				return false;
			}

			if (samplers[index] != id._impl)
			{
				return false;
			}

			writes_skipped++;
			return true;
		}

	private:
		template <typename M>
		bool update(u64& slot, M& known, M bit, u64 value)
		{
			if ((known & bit) && slot == value)
			{
				writes_skipped++;
				return false;
			}

			slot = value;
			known |= bit;
			writes++;
			return true;
		}
	};

	struct submit_info_t
	{
		// Optional GPU-side waits before this batch executes
		const MTL::Event* wait_event = nullptr;
		u64 wait_value = 0;
		const MTL::Drawable* wait_drawable = nullptr;     // Wait for the drawable to be available before rendering into it
		const MTL::Drawable* signal_drawable = nullptr;   // Signal drawable completion after this batch (then call present())
		bool flush = false;                               // Hint only; Metal commits are always flushed
	};

	// Metal 4 command recording unit (equivalent of vk::command_buffer + command pool).
	//
	// Hazard tracking (Metal 4 resources are untracked; DESIGN.md §3). Every command declares what it reads and writes
	// (gpu_access), and only the barriers those accesses need are encoded, from the stages that produced the conflicting
	// accesses to the stages of the command. Commands that do not conflict run concurrently.
	//  - Compute encoder commands: blit() / dispatch() (the latter through glsl::program::bind()). A conflict with a
	//    command of an earlier encoder gets a consumer queue barrier, with an earlier command of the same encoder an
	//    intra-encoder barrier.
	//  - Render passes (tile-based GPU: all draws of a pass are binned, then shaded tile by tile, so nothing recorded after
	//    its first draw can order the pass's work). Every pass begins with two queue barriers: its vertex, fragment and
	//    tile work after every earlier non-fragment stage, its fragment and tile work (attachment loads, shading,
	//    stores) after every earlier stage. That orders the attachments and everything the fragment stage reads. What
	//    the draws read is declared when it is bound (table_access() from glsl::program::bind(), draw_access() for index
	//    and visibility buffers); what the begin barriers leave out (the vertex stage reading what fragment work of an
	//    earlier pass wrote) gets a queue barrier before the pass's first draw, or, once the pass recorded draws, makes
	//    pass_split_required() true: the renderer ends the pass and draws in a new one. Conflicts inside the open pass
	//    cannot be ordered (no fragment -> fragment barrier on Apple GPUs): feedback loops remain the renderer's job.
	//  - A consumer barrier (A -> B) orders B-work of this pass and every later one after A-work of every earlier pass of
	//    the queue, so an access is ordered for later B-work once one barrier covered it.
	//  - Undeclared accesses (gpu_access::undeclared) get a full barrier and count as writes of everything.
	//  - Across submissions: submit() checks what the list (and its prologue) accesses against the accesses of earlier
	//    submissions to the same queue that may still run, except what the list's pass barriers already order after all
	//    earlier queue work (attachments, fragment reads): compute commands and vertex-stage reads. A conflict makes the
	//    queue wait for the conflicting submission; without one, the list starts while earlier work still runs.
	//
	// Prologue (opt-in, enable_prologue()): a second command buffer, committed in the same commit call right before
	// this list's buffer. It shares the submission's waits, timeline signal, fence and GC event id. Every encoder of the
	// prologue ends with a producer barrier: ALL later work of the queue (all of this list, including work recorded before
	// the prologue work in program order) waits for it. Only record work into it whose inputs were not produced by GPU
	// work of this list and whose outputs no work already recorded in this list reads or writes: texture uploads from
	// memory into images that are new to this list.
	class command_list
	{
	public:
		enum access_type_hint
		{
			flush_only, // Only to be submitted/opened/closed via command flush
			all         // Auxiliary, can be submitted/opened/closed at any time
		}
		access_hint = flush_only;

		enum command_buffer_data_flag : u32
		{
			cb_has_occlusion_task     = 0x01,
			cb_has_blit_transfer      = 0x02,
			cb_has_dma_transfer       = 0x04,
			cb_has_open_query         = 0x08,
			cb_load_occluson_task     = 0x10,
			cb_has_conditional_render = 0x20,
			cb_reload_dynamic_state   = 0x40
		};
		u32 flags = 0;

		enum class encoder_type
		{
			none,
			render,
			compute
		};

		struct hazard_state; // commands.cpp

	protected:
		const render_device* m_device = nullptr;
		MTL4::CommandQueue* m_queue = nullptr;
		timeline* m_timeline = nullptr;

		MTL4::CommandAllocator* m_allocator = nullptr; // From the pool (command_allocators.h) while recording, until submit()
		MTL4::CommandBuffer* m_commands = nullptr;

		MTL4::RenderCommandEncoder* m_render_encoder = nullptr;   // Not owned (lifetime of the encoding)
		u64 m_pass_serial = 0;                                     // Unique id of the open render pass (all lists)
		bool m_draw_pass = false;                                  // The open pass renders into the RSX surfaces
		MTL4::ComputeCommandEncoder* m_compute_encoder = nullptr; // Not owned

		std::unique_ptr<hazard_state> m_hazards;

		std::array<MTL4::ArgumentTable*, table_count> m_argument_tables{};
		std::array<argument_table_shadow, table_count> m_argument_table_shadows{};
		render_encoder_bindings m_render_bindings{};
		compute_encoder_bindings m_compute_bindings{};
		encoder_state_counters m_state_counters{};

		bool m_is_open = false;
		bool m_is_pending = false;
		fence m_submit_fence{};

		// Deferred clears (defer_clear), in the order they were requested. Invariant: no encoder is open while any is
		// pending; beginning an encoder or ending the list records them first (folded into a pass or as clear passes).
		struct deferred_clear
		{
			ref<MTL::Texture> texture;      // Retained: its address cannot be reused while the clear is pending
			u32 planes = 0;                 // mtl::aspect_color, or aspect_depth and/or aspect_stencil
			u32 width = 0;                  // Cleared area [0, width) x [0, height) of level 0, slice 0
			u32 height = 0;
			attachment_clear_value value{};
		};
		std::vector<deferred_clear> m_deferred_clears;
		u32 m_folded_clears = 0;           // Attachments of the pass begun last whose load action is a deferred clear

		// Begins every render encoder of the list (begin_render_pass() after folding deferred clears, the clear-only passes
		// of flush_deferred_clears()) and declares its attachments (hazard tracking)
		MTL4::RenderCommandEncoder* open_render_encoder(MTL4::RenderPassDescriptor* desc, bool draw_pass);
		u32 fold_deferred_clears(MTL4::RenderPassDescriptor* desc, std::array<std::pair<MTL::RenderPassAttachmentDescriptor*, MTL::LoadAction>, 10>& restore, u32& restore_count);

		// Prologue list (created on first use). It is never submitted or waited on by itself: this list commits it and
		// this list's completion implies its completion.
		std::unique_ptr<command_list> m_prologue;
		bool m_prologue_enabled = false;
		bool m_prologue_recorded = false; // m_prologue was begun during the current recording and goes with its commit
		bool m_is_prologue = false;       // This list is the prologue of another one

		std::string m_label;

		// Begins every compute encoder command of the list (blit(), dispatch(), blit_concurrent()): records the pending
		// deferred clears before a new compute encoder, then the barriers the accesses need
		MTL4::ComputeCommandEncoder* compute_command(std::span<const gpu_access> accesses, MTL::Stages stage, bool new_command);
		void end_encoding(MTL4::CommandEncoder* encoder, MTL::Stages stages);
		void flush_encoding_counters();
		void wait_for_conflicting_submissions();
		void record_submission(u64 value, const command_list* prologue);

	public:
		command_list();
		virtual ~command_list();

		command_list(const command_list&) = delete;
		command_list& operator=(const command_list&) = delete;

		void create(const render_device& dev, MTL4::CommandQueue* queue, timeline& tl, std::string_view label);
		void destroy();

		// Begin recording with an allocator from the pool. Waits for this list's previous submission (see wait()/poke()).
		void begin();
		// Close any open encoder and end the command buffer.
		void end();
		// Commit to the queue and signal the timeline. Returns the timeline value that marks completion. The allocator(s)
		// go back to the pool, which reuses them once that value is reached.
		u64 submit(const submit_info_t& info = {});

		bool is_recording() const { return m_is_open; }
		bool is_pending() const { return m_is_pending; }
		const fence& get_fence() const { return m_submit_fence; }

		// --- Encoders -------------------------------------------------------------------------------------------
		// Begin a render pass. Ends any active encoder first. `draw_pass`: the renderer's pass of the RSX surfaces
		// (telemetry: why it ends, see pass_end_reason; draw_access()). Deferred clears of textures this pass attaches
		// (level 0, slice 0, render area equal to the cleared area) become the load action (Clear) of those attachments;
		// the other pending clears are recorded before the pass. `desc` is only modified for the duration of the call.
		// The pass begins with the two pass barriers (class comment); the attachments of the pass as begun (folded clears
		// included) are declared as written by the pass.
		MTL4::RenderCommandEncoder* begin_render_pass(MTL4::RenderPassDescriptor* desc, bool draw_pass = false);
		bool is_render_pass_open() const { return m_render_encoder != nullptr; }

		// Unique id (across all command lists) of the open render pass, 0 when none is open. Attachments written by a
		// pass reach memory when it ends: feedback reads compare a surface's last writing pass with this id.
		u64 open_pass_serial() const { return m_render_encoder ? m_pass_serial : 0; }
		MTL4::RenderCommandEncoder* render_encoder() const { return m_render_encoder; }
		// `reason` is what the caller ends the pass for (telemetry of draw passes). Ends that happen inside this class
		// pass their own: a new compute encoder (transfer), another render pass, the end of the list (submit).
		void end_render_pass(pass_end_reason reason = pass_end_reason::transfer);

		// What a draw of the open render pass accesses outside the argument tables (index buffers: read by the vertex
		// stage; visibility results: written by the fragment stage's tests, stages_attachment). `stages`: the stages of
		// the pass that access it. A conflict the pass barriers do not cover gets a queue barrier if no draw was recorded
		// in the pass yet (render_encoder_bindings::program_uid is 0); otherwise the renderer's draw pass must be split
		// (pass_split_required()). Declare everything a draw reads before its pipeline state is set.
		void draw_access(const gpu_access& access, MTL::Stages stages);
		// What argument table slot `index` of `table` (vertex or fragment) gives the draws of the open render pass.
		// Skipped when the slot was declared with the same access in this pass (glsl::program::bind() calls it for
		// every slot of every draw).
		void table_access(argument_table_slot table, u32 index, bool texture, const gpu_access& access);
		// A draw declared a read of the open draw pass that needs a queue barrier after the pass recorded draws: the
		// barrier was not encoded (it could not order the pass on a tile-based GPU). The draw must not be recorded in
		// this pass: end it, begin a new one and bind again (the barrier then precedes the new pass's first draw).
		// Cleared when a pass begins.
		bool pass_split_required() const;

		// Compute encoder commands. Each call starts a command (ending an open render pass, opening a compute encoder):
		// the barriers its accesses need against earlier commands are encoded, then the encoder is returned to record the
		// command(s) the declaration covers (several copies of one transfer, a bind + dispatch).
		MTL4::ComputeCommandEncoder* blit(std::initializer_list<gpu_access> accesses);
		MTL4::ComputeCommandEncoder* blit(std::span<const gpu_access> accesses);
		MTL4::ComputeCommandEncoder* dispatch(std::span<const gpu_access> accesses);
		// Adds a blit to the command begun by the last blit()/dispatch() of the open compute encoder: ordered after the
		// earlier commands it conflicts with, but not after the other parts of that command (the caller guarantees they
		// are independent, e.g. copies into disjoint rectangles of one texture level)
		MTL4::ComputeCommandEncoder* blit_concurrent(std::initializer_list<gpu_access> accesses);
		// The open compute encoder, for another operation of the command the last blit()/dispatch() declared
		MTL4::ComputeCommandEncoder* compute_encoder() const;

		// Work that a framework (MetalFX) encodes into this list's command buffer with encoders of its own. Ends the open
		// encoder. The caller orders the work after what it conflicts with (MetalFX waits on a fence that a command
		// declared with the same accesses updates); later commands are ordered after it, in every stage.
		void external_work(std::span<const gpu_access> accesses);

		void end_encoder(pass_end_reason reason = pass_end_reason::other_pass);
		encoder_type active_encoder() const;

		// --- Deferred clears -----------------------------------------------------------------------------------
		// Clears [0, width) x [0, height) of level 0, slice 0 of a 2D (or 2D multisample) render target texture with
		// the load action of the next render pass that attaches the texture over exactly that area, instead of a pass
		// of its own followed by a load. Ordered like any command recorded now: ends the open encoder, and if anything
		// else is recorded first (compute, a pass that does not attach the texture that way, the end of the list, direct
		// use of handle()), the pending clears are recorded right before it as clear-only passes. Whoever samples a
		// texture in a pass must flush_deferred_clears() first if it has one pending (the pass could fold it: it would
		// sample the memory before the clear). `planes`: mtl::aspect_color, or aspect_depth and/or aspect_stencil.
		void defer_clear(MTL::Texture* texture, u32 planes, u32 width, u32 height, const attachment_clear_value& value);
		bool has_deferred_clears() const { return !m_deferred_clears.empty(); }
		// A clear of `texture` is pending (of one of `planes`: mtl::aspect_color / aspect_depth / aspect_stencil)
		bool has_deferred_clear(const MTL::Texture* texture, u32 planes = ~0u) const;
		// Records the pending clears now, as clear-only passes (one per size and sample count, up to 8 colour textures
		// and one depth-stencil texture each)
		void flush_deferred_clears();
		// Attachments of the pass begun last whose load action came from a deferred clear: bit i = colour attachment
		// i, bit 8 = depth, bit 9 = stencil
		u32 folded_clears() const { return m_folded_clears; }

		// --- Prologue (see the class comment) ----------------------------------------------------------------------
		// Allows prologue() on this list. Only for lists recorded and submitted by one thread at a time (the renderer's
		// primary lists).
		void enable_prologue() { m_prologue_enabled = true; }
		bool can_record_prologue() const { return m_prologue_enabled && m_is_open; }
		// The prologue list of the current recording (begun on first use), or nullptr if prologues are not enabled or
		// this list is not recording. Record into it with the usual helpers; it has its own encoders and argument tables.
		command_list* prologue();

		// --- Argument tables (one per stage slot, reused across encoders; contents captured at draw/dispatch) ---
		MTL4::ArgumentTable* argument_table(argument_table_slot slot) const { return m_argument_tables[slot]; }
		// What the table of `slot` holds. Whoever writes a table directly must update (or invalidate) this.
		argument_table_shadow& argument_table_contents(argument_table_slot slot) { return m_argument_table_shadows[slot]; }
		// Pipeline state and tables set on the open render / compute encoder
		render_encoder_bindings& render_bindings() { return m_render_bindings; }
		compute_encoder_bindings& compute_bindings() { return m_compute_bindings; }
		encoder_state_counters& state_counters() { return m_state_counters; }

		// --- Misc -----------------------------------------------------------------------------------------------
		// For encoding outside of this class (MetalFX): records the pending deferred clears first
		MTL4::CommandBuffer* handle();
		MTL4::CommandQueue* queue() const { return m_queue; }
		const render_device& device() const { return *m_device; }

		void push_debug_group(std::string_view name);
		void pop_debug_group();

		void clear_flags() { flags = 0; }
		void set_flag(command_buffer_data_flag flag) { flags |= flag; }

		// Completion tracking (called by the renderer's command-buffer ring)
		bool poke();                     // Non-blocking; returns true if no work is pending
		bool wait(u64 timeout_us = 0);   // Blocking; returns false on timeout
	};

	// Forgets what earlier submissions to `queue` accessed (the queue is being destroyed: every submission completed)
	void forget_queue_submissions(const MTL4::CommandQueue* queue);
}

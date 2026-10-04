#pragma once

// Port of VK/VKPipelineCompiler: a pool of "RSX.W" worker threads that translate shaders (GLSL -> MSL -> MTLLibrary)
// and build Metal 4 pipeline state objects, either inline (COMPILE_INLINE) or deferred with a completion callback.
// Deferred jobs go to one queue shared by all workers. Queue order (DESIGN.md §6): jobs the RSX thread waits for, the
// shader interpreter's uber pipelines (COMPILE_AHEAD), the other pipelines draws need, the shader cache preload (tasks,
// once the pipeline archive is ready for lookups), background full-state compiles.
//
// Render pipelines are Metal 4 flexible render pipeline states (DESIGN.md §7):
//  - One UNSPECIALIZED pipeline is compiled per shader pair and full-compile state (graphics_pipeline_state minus the
//    colour attachment configuration: sample count, alpha to coverage/one, topology class, ...), with every colour
//    attachment's pixel format, write mask and blend state left unspecialized.
//  - Each concrete pipeline (colour formats, write masks, blending) is created from it with
//    MTL4Compiler::newRenderPipelineStateBySpecialization, which only generates the fragment output part. A deferred
//    request whose unspecialized pipeline already exists is specialized inline, on the requesting thread.
//  - Specialized pipelines cost a little GPU time (the fragment body cannot be optimized for the attachments). A program
//    drawn full_state_draw_threshold times gets its full-state pipeline compiled on a worker at background QoS; the
//    next glsl::program::bind() swaps it in and hands the specialized one to the GC.
//  - Full-state pipelines are built directly when the pipeline archive holds them (is_full_state_archived), when the
//    pipeline cannot be specialized (no fragment stage, or framebuffer fetch: the fragment body reads the attachments),
//    and for a key whose unspecialized build or specialization failed (logged).

#include "MTLProgramPipeline.h"
#include "MTLPipelineArchive.h" // Pipeline persistence hooks used by MTLGSRender (initialize/preloaded/flush)
#include "Utilities/lockless.h"
#include "Emu/RSX/Common/simple_array.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace mtl
{
	class render_device;

	void initialize_pipe_compiler(int num_worker_threads);
	void destroy_pipe_compiler();

	// The only primitive topology class a render pipeline declares is Point (see build_graphics_program): line and
	// triangle pipelines are identical. Pipeline keys (graphics_pipeline_state::topology_class) therefore use
	// MTL::PrimitiveTopologyClassTriangle for every non-point topology, so the same pipeline is not built twice.
	inline u8 get_pipeline_topology_class(u8 topology_class)
	{
		return topology_class == static_cast<u8>(MTL::PrimitiveTopologyClassPoint)
			? topology_class
			: static_cast<u8>(MTL::PrimitiveTopologyClassTriangle);
	}

	// Draws after which a render pipeline created by specialization is rebuilt with full state in the background
	constexpr u32 full_state_draw_threshold = 1000;

	// Called by glsl::program::bind() for every draw of a program whose render pipeline is a specialization (binding
	// thread only). Counts the draw and queues the background full-state compile when the count reaches
	// full_state_draw_threshold. Returns the full-state pipeline (+1, the caller owns it) once it has been built.
	// `finished` is set when nothing more will come (the pipeline was returned, or it could not be built).
	MTL::RenderPipelineState* poll_full_state_upgrade(const std::shared_ptr<full_state_upgrade>& upgrade, bool& finished);

	// Render pipeline creation counters since the last call (telemetry, MTLPresent.cpp)
	struct pipeline_creation_stats
	{
		u32 specialized = 0;           // Render pipelines created by specialization...
		u64 specialization_us = 0;     // ...the time it took in total...
		u64 specialization_max_us = 0; // ...and the longest one
		u32 full_state = 0;            // Render pipelines created with full state when first needed...
		u32 full_state_archived = 0;   // ...of which the pipeline archive listed as held
		u32 fallbacks = 0;             // ...of which because an unspecialized build or a specialization failed
		u32 unspecialized = 0;         // Unspecialized pipelines built...
		u64 unspecialized_us = 0;      // ...and the time it took (an archive hit takes far less than a compile)
		u32 upgrades_requested = 0;    // Heavily drawn specializations queued for a background full-state compile
		u32 upgrades_built = 0;        // Background full-state compiles finished
		u32 upgrades_swapped = 0;      // Full-state pipelines swapped in by bind()
	};

	pipeline_creation_stats get_pipeline_creation_stats_and_reset();

	// Where the time of shader and pipeline builds goes (record_compile_time, MTLProgramPipeline.h)
	struct compile_timings
	{
		u32 translated = 0;          // Shaders translated GLSL -> SPIR-V -> MSL...
		u64 translate_us = 0;        // ...and the time it took
		u32 libraries = 0;           // MTLLibraries built from MSL
		u64 library_us = 0;
		u32 archive_pipelines = 0;   // Pipelines built with the pipeline archive's lookups (a binary found, or compiled)
		u64 archive_pipeline_us = 0;
		u32 compiled_pipelines = 0;  // Pipelines compiled without lookups
		u64 compiled_pipeline_us = 0;
		u32 specialized = 0;         // Render pipelines created by specialization
		u64 specialization_us = 0;

		void add(const compile_timings& other);
	};

	// The builds of this thread since its record was last reset. Pipe compiler workers reset it before every job, so
	// the job (or its callback) reads its own timings.
	compile_timings& this_thread_compile_timings();

	// Builds of the last get_compile_timings_and_reset() window, all threads (the 30-second statistics)
	compile_timings get_compile_timings_and_reset();

	// Vertex shaders that declare input attributes (GLSL `layout(location = N) in ...`, e.g. overlay passes) get an
	// automatic MTLVertexDescriptor: all attributes interleaved in ONE buffer, ascending location order, tightly packed
	// (4-byte aligned), stride = packed size, per-vertex step. Bind that buffer after program::bind() through the table's
	// shadow (mtl::argument_table_shadow): cmd.argument_table(table_vertex)->setAddress(address,
	// stage_in_vertex_buffer_index) only if cmd.argument_table_contents(table_vertex).update_buffer(...) returns true.
	// RSX vertex programs pull vertices from texel buffers and never use this.
	constexpr u32 stage_in_vertex_buffer_index = 30;

	class pipe_compiler
	{
	public:
		enum op_flag_bits
		{
			COMPILE_DEFAULT = 0,
			COMPILE_INLINE = 1,
			COMPILE_DEFERRED = 2,
			SEPARATE_SHADER_OBJECTS = 4,   // Accepted for parity with VK; Metal stages always have separate binding tables
			USE_LAST_PROVOKING_VERTEX = 8, // Unsupported by Metal (first vertex provokes); ignored, logged once
			COMPILE_AHEAD = 16             // Deferred: queued ahead of the other background jobs (still behind the jobs the
			                               // RSX thread waits for), e.g. the shader interpreter pipeline of a render state
		};

		using op_flags = rsx::flags32_t;

		using callback_t = std::function<void(std::unique_ptr<glsl::program>&)>;

		pipe_compiler();
		~pipe_compiler();

		void initialize(const mtl::render_device* pdev);

		// Compute pipeline. The shader must have been create()d; it is translated (once) by whichever thread gets
		// to it first. The shader object must outlive any deferred job referencing it.
		std::unique_ptr<glsl::program> compile(
			glsl::shader* cs,
			op_flags flags, callback_t callback = {},
			const std::vector<glsl::program_input>& cs_inputs = {});

		// Graphics pipeline from two created shaders + baked fixed-function state (RSX pipelines).
		// Deferred jobs return nullptr immediately and deliver the program (or nullptr on failure) through `callback`
		// on the worker thread, except when the pipeline can be created by specializing an unspecialized pipeline that
		// already exists (a new colour attachment configuration of known shaders): that takes no compile, so it is done
		// on the calling thread and the program is returned (`callback` is not called). Inline jobs return the program
		// (nullptr on failure) and ignore `callback`.
		std::unique_ptr<glsl::program> compile(
			const mtl::pipeline_props& create_info,
			glsl::shader* vs,
			glsl::shader* fs,
			op_flags flags, callback_t callback = {},
			const std::vector<glsl::program_input>& vs_inputs = {},
			const std::vector<glsl::program_input>& fs_inputs = {});

		// Worker thread entry
		void operator()();

		// Shader cache preload (MTLProgramBuffer.h): each task builds one cached pipeline on a worker at QOS_CLASS_DEFAULT
		// (talk 6: pipeline prewarming), in order, after every other job but the background full-state compiles, and not
		// before the pipeline archive is ready for lookups. A task counts as a finished deferred job
		// (get_completed_job_count). Tasks still queued when the pipe compiler is destroyed are dropped without running.
		static void queue_preload_tasks(std::vector<std::function<void()>> tasks);

		// The pipeline archive has become ready for lookups (MTLPipelineArchive.cpp): workers may take preload tasks
		static void on_pipeline_archive_ready();

		// Number of deferred jobs finished so far (built or failed). It is bumped after the job's callback returned,
		// so a pipeline that was not in the cache when this was sampled is there once the count has changed.
		static u32 get_completed_job_count();

		// Waits until the completed job count differs from `count`, at most `timeout_us`. May return early.
		static void wait_for_completed_job(u32 count, u64 timeout_us);

		// A thread is about to wait for a pipeline of this shader pair: its queued jobs move to the front of the queue
		// and run at emulation thread priority, on any idle worker (including the one background jobs leave free).
		// No effect on jobs that are already running.
		static void prioritize_jobs(const glsl::shader* vs, const glsl::shader* fs);

	private:
		friend void initialize_pipe_compiler(int num_worker_threads);
		friend void destroy_pipe_compiler();
		friend MTL::RenderPipelineState* poll_full_state_upgrade(const std::shared_ptr<full_state_upgrade>& upgrade, bool& finished);

		enum class job_kind : u8
		{
			graphics,
			compute,
			preload,            // Shader cache preload task (queue_preload_tasks)
			full_state_upgrade, // Background QoS; never waited for
		};

		struct pipe_compiler_job
		{
			job_kind kind;
			bool urgent = false;          // Waited for (prioritize_jobs): runs at emulation thread priority
			bool ahead = false;           // COMPILE_AHEAD: queued in front of the other background jobs
			bool full_state = false;      // Graphics: build with full state (its specialization failed)
			callback_t callback_func;

			mtl::pipeline_props graphics_data{};
			glsl::shader* shaders[2]{};   // [vs, fs] or [cs, nullptr]
			std::vector<glsl::program_input> inputs[2];
			std::shared_ptr<full_state_upgrade> upgrade;
			std::function<void()> task;   // Preload

			op_flags flags = COMPILE_DEFAULT;

			pipe_compiler_job(
				const mtl::pipeline_props& props,
				glsl::shader* vs,
				glsl::shader* fs,
				const std::vector<glsl::program_input>& vs_in,
				const std::vector<glsl::program_input>& fs_in,
				op_flags flags_,
				callback_t func,
				bool full_state_)
				: kind(job_kind::graphics)
				, ahead(!!(flags_ & COMPILE_AHEAD))
				, full_state(full_state_)
				, callback_func(std::move(func))
				, graphics_data(props)
				, shaders{ vs, fs }
				, inputs{ vs_in, fs_in }
				, flags(flags_)
			{
			}

			pipe_compiler_job(
				glsl::shader* cs,
				const std::vector<glsl::program_input>& cs_in,
				op_flags flags_,
				callback_t func)
				: kind(job_kind::compute)
				, ahead(!!(flags_ & COMPILE_AHEAD))
				, callback_func(std::move(func))
				, shaders{ cs, nullptr }
				, inputs{ cs_in, {} }
				, flags(flags_)
			{
			}

			explicit pipe_compiler_job(std::shared_ptr<full_state_upgrade> upgrade_)
				: kind(job_kind::full_state_upgrade)
				, upgrade(std::move(upgrade_))
			{
			}

			explicit pipe_compiler_job(std::function<void()> task_)
				: kind(job_kind::preload)
				, task(std::move(task_))
			{
			}
		};

		// Deferred jobs of all workers (MTLPipelineCompiler.cpp). One queue, so that an idle worker never waits
		// behind a slow compile that happened to be queued on another worker.
		struct job_queue;
		static job_queue s_queue;

		const mtl::render_device* m_device = nullptr;

		std::unique_ptr<glsl::program> int_compile_compute_pipe(
			glsl::shader* cs,
			const std::vector<glsl::program_input>& cs_inputs,
			op_flags flags);

		std::unique_ptr<glsl::program> int_compile_graphics_pipe(
			const mtl::pipeline_props& create_info,
			glsl::shader* vs,
			glsl::shader* fs,
			const std::vector<glsl::program_input>& vs_inputs,
			const std::vector<glsl::program_input>& fs_inputs,
			bool full_state);

		static void int_build_full_state_upgrade(full_state_upgrade& upgrade);
	};

	// 0 (or negative) = automatic count (the VK backend's default for background compiles plus one worker kept for jobs
	// the RSX thread waits for, see the .cpp). Always capped by device caps().max_compile_tasks. With more than one worker,
	// background (non-urgent) jobs never occupy the last one.
	void initialize_pipe_compiler(int num_worker_threads = -1);
	void destroy_pipe_compiler(); // Joins the workers and drops the jobs that were still queued (callbacks never run)
	pipe_compiler* get_pipe_compiler();

	// Pipelines of shaders specialized with function constants (the shader interpreter's) are built without the pipeline
	// archive (MTLPipelineArchive.h); everything else uses it
	bool uses_pipeline_archive(const glsl::shader& shader);

	// ---- Synchronous building blocks (no worker threads needed; used by the workers and glsl::create_*_program) ----
	// Translate the shader(s) if needed and build the program. nullptr (logged) on failure. Thread-safe.
	std::unique_ptr<glsl::program> build_compute_program(
		glsl::shader& cs,
		const std::vector<glsl::program_input>& cs_inputs);

	// Flexible render pipeline (see the top of this file): specialized from the unspecialized pipeline of (vs, fs,
	// state minus the colour attachment configuration), which is built first if needed (a concurrent build of the same
	// one is waited for), or built with full state. Serves every render pipeline: RSX programs, the shader interpreter
	// (through pipe_compiler::compile), overlays, the in-pass clear quads; there is no other render pipeline builder.
	std::unique_ptr<glsl::program> build_graphics_program(
		glsl::shader& vs,
		glsl::shader& fs,
		const glsl::graphics_pipeline_state& state,
		const std::vector<glsl::program_input>& vs_inputs,
		const std::vector<glsl::program_input>& fs_inputs);
}

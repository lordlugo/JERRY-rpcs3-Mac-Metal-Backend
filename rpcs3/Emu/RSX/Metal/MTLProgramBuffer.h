#pragma once
#include "MTLVertexProgram.h"
#include "MTLFragmentProgram.h"
#include "MTLPipelineCompiler.h"
#include "../Program/ProgramStateCache.h"

#include "Emu/system_config.h"
#include "Emu/Cell/timers.hpp"
#include "Utilities/mutex.h"
#include "util/fnv_hash.hpp"

#include <chrono>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace mtl
{
	// RSX pipelines that could not be built. program_state_cache keeps the null placeholder it inserted for them, which
	// looks exactly like a pipeline that is still being compiled; this set tells the two apart, so that nobody waits
	// for (or announces the compilation of) a pipeline that will never exist.
	class pipeline_failure_registry
	{
	public:
		struct key_type
		{
			u32 vertex_program_id;
			u32 fragment_program_id;
			mtl::pipeline_props properties; // Validated (MTLTraits::make_failure_key)

			bool operator==(const key_type&) const = default;
		};

		// Upper bound so the set cannot grow without limit between clears (program_state_cache::clear).
		// On overflow the set is dropped: at most one rebuild attempt + error line per pipeline per epoch.
		static constexpr usz max_failure_entries = 2048;

		void add(const key_type& key)
		{
			std::lock_guard lock(m_mutex);
			if (m_keys.size() >= max_failure_entries)
			{
				m_keys.clear();
			}
			if (m_keys.insert(key).second)
			{
				// Once per pipeline: the program cache never rebuilds it (the reason was logged by the builder)
				rsx_log.error("Metal: the pipeline for vp id %u / fp id %u (%u color attachments, %u samples) could not be built. Draws that use it are skipped.",
					key.vertex_program_id, key.fragment_program_id, key.properties.state.color_count, key.properties.state.sample_count);
			}
		}

		bool contains(const key_type& key) const
		{
			reader_lock lock(m_mutex);
			return m_keys.contains(key);
		}

		void clear()
		{
			std::lock_guard lock(m_mutex);
			m_keys.clear();
		}

	private:
		struct key_hash
		{
			usz operator()(const key_type& key) const
			{
				return rpcs3::hash64(rpcs3::hash64(rpcs3::hash_struct(key.properties), key.vertex_program_id), key.fragment_program_id);
			}
		};

		mutable shared_mutex m_mutex;
		std::unordered_set<key_type, key_hash> m_keys;
	};

	// Shader cache preload in the background (DESIGN.md §6): progress of the pipelines queued by
	// program_cache::queue_preloaded_pipelines(), shared by their tasks
	struct shader_preload_state
	{
		u64 queue_time = 0;              // get_system_time() when the tasks were queued
		u32 total = 0;
		atomic_t<u32> remaining = 0;
		atomic_t<u32> built = 0;         // Built by their preload task
		atomic_t<u32> present = 0;       // Already built, or being built, for a draw when their task ran
		atomic_t<u32> failed = 0;
		atomic_t<bool> finished = false; // The outcome was reported (the last task ran, or the renderer stopped first)

		std::mutex timings_lock;
		compile_timings timings;         // Where the tasks' time went
		u64 task_us = 0;                 // Their time on the workers
	};

	struct MTLTraits
	{
		using vertex_program_type = MTLVertexProgram;
		using fragment_program_type = MTLFragmentProgram;
		using pipeline_type = mtl::glsl::program;
		using pipeline_storage_type = std::unique_ptr<mtl::glsl::program>;
		using pipeline_properties = mtl::pipeline_props;

		static
			void recompile_fragment_program(const RSXFragmentProgram& RSXFP, fragment_program_type& fragmentProgramData, usz ID)
		{
			fragmentProgramData.Decompile(RSXFP);
			fragmentProgramData.id = static_cast<u32>(ID);
			fragmentProgramData.Compile();
		}

		static
			void recompile_vertex_program(const RSXVertexProgram& RSXVP, vertex_program_type& vertexProgramData, usz ID)
		{
			vertexProgramData.Decompile(RSXVP);
			vertexProgramData.id = static_cast<u32>(ID);
			vertexProgramData.Compile();
		}

		static
			void validate_pipeline_properties(const MTLVertexProgram&, const MTLFragmentProgram& fp, mtl::pipeline_props& properties)
		{
			// Explicitly disable writing to undefined registers
			for (u32 i = 0; i < ::size32(properties.state.color); ++i)
			{
				properties.state.color[i].write_mask &= static_cast<u8>(fp.output_color_masks[i]);
			}
		}

		// program_state_cache builds a new program pair with the properties as they are, but validates them first for
		// programs it already had. Validation is idempotent, so validating here gives one key for both cases.
		static
			pipeline_failure_registry::key_type make_failure_key(const vertex_program_type& vp, const fragment_program_type& fp, mtl::pipeline_props properties)
		{
			validate_pipeline_properties(vp, fp, properties);
			return { vp.id, fp.id, properties };
		}

		static
			pipeline_type* build_pipeline(
				const vertex_program_type& vertexProgramData,
				const fragment_program_type& fragmentProgramData,
				const mtl::pipeline_props& pipelineProperties,
				bool compile_async,
				std::function<pipeline_type*(pipeline_storage_type&)> callback,
				pipeline_failure_registry& failures)
		{
			mtl::pipe_compiler::op_flags compiler_flags = compile_async ? mtl::pipe_compiler::COMPILE_DEFERRED : mtl::pipe_compiler::COMPILE_INLINE;
			compiler_flags |= mtl::pipe_compiler::SEPARATE_SHADER_OBJECTS;

			if (vertexProgramData.Flags() & RSX_SHADER_CONTROL_FLAT_SHADING)
			{
				compiler_flags |= mtl::pipe_compiler::USE_LAST_PROVOKING_VERTEX;
			}

			// The cache's callback leaves the null placeholder in place when the build failed: record the failure
			// before it runs, so that a lookup never sees the placeholder without the failure after a failed build
			auto on_built = [&failures, failure_key = make_failure_key(vertexProgramData, fragmentProgramData, pipelineProperties),
				callback = std::move(callback)](pipeline_storage_type& pipeline) -> pipeline_type*
			{
				if (!pipeline)
				{
					failures.add(failure_key);
				}

				return callback(pipeline);
			};

			auto compiler = mtl::get_pipe_compiler();
			auto result = compiler->compile(
				pipelineProperties,
				vertexProgramData.handle,
				fragmentProgramData.handle,
				compiler_flags, on_built,
				vertexProgramData.uniforms,
				fragmentProgramData.uniforms);

			if (compile_async && !result)
			{
				// Queued: a worker hands the result to on_built (the null result here is not a failure)
				return nullptr;
			}

			// Built on this thread: synchronous, or a deferred request that was a specialization (MTLPipelineCompiler.h)
			return on_built(result);
		}
	};

	struct program_cache : public program_state_cache<MTLTraits>
	{
		using base_type = program_state_cache<MTLTraits>;

		program_cache(decompiler_callback_t callback)
		{
			notify_pipeline_compiled = callback;
		}

		// MTL4 render pipelines do not bake depth/stencil formats, so graphics_pipeline_state::depth_stencil_format is
		// removed from the effective cache key: it is zeroed while the pipeline is looked up / built / stored in the
		// shader cache (the field stays in the POD for shader-cache layout compatibility) and restored afterwards so
		// the caller's own state comparisons are unaffected. Hides program_state_cache::get_graphics_pipeline.
		// Also tells a failed pipeline from one that is still compiling (check_pipeline_failed).
		template <typename... Args>
		auto get_graphics_pipeline(
			rsx::program_cache_hint_t* cache_hint,
			const RSXVertexProgram& vertex_shader,
			const RSXFragmentProgram& fragment_shader,
			mtl::pipeline_props& pipeline_properties,
			bool compile_async,
			bool allow_notification,
			Args&& ...args)
		{
			const u32 depth_stencil_format = std::exchange(pipeline_properties.state.depth_stencil_format, 0u);

			auto result = base_type::get_graphics_pipeline(cache_hint, vertex_shader, fragment_shader, pipeline_properties,
				compile_async, allow_notification, m_failed_pipelines, std::forward<Args>(args)...);

			m_last_pipeline_failed = false;
			if (const auto& [pipeline, vp, fp] = result; !pipeline)
			{
				// Same key as the build (depth/stencil format still zeroed)
				m_last_pipeline_failed = m_failed_pipelines.contains(MTLTraits::make_failure_key(*vp, *fp, pipeline_properties));

				if (m_last_pipeline_failed)
				{
					// Nothing is being compiled: no shader compilation notification
					m_cache_miss_flag = false;
				}
			}

			pipeline_properties.state.depth_stencil_format = depth_stencil_format;
			return result;
		}

		u64 get_hash(const mtl::pipeline_props& props)
		{
			return rpcs3::hash_struct<mtl::pipeline_props>(props);
		}

		u64 get_hash(const RSXVertexProgram& prog)
		{
			return program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(prog);
		}

		u64 get_hash(const RSXFragmentProgram& prog)
		{
			return program_hash_util::fragment_program_utils::get_fragment_program_ucode_hash(prog);
		}

		// ---- Shader cache preload (DESIGN.md §6) ------------------------------------------------------------------
		// rsx::shaders_cache::load() reads the cached pipelines on its worker threads while the RSX thread waits: first
		// preload_programs() for each (decompiles its programs: CPU work only), then add_pipeline_entry() for each, which
		// only records the pipeline. queue_preloaded_pipelines() (RSX thread, after load() has returned, so every program
		// is complete before anything else can see it) queues one pipe compiler task per pipeline and returns: the
		// pipelines are translated, compiled and looked up in the pipeline archive in the background, behind every
		// pipeline a draw asks for (pipe_compiler::queue_preload_tasks).

		void add_pipeline_entry(RSXVertexProgram& vp, RSXFragmentProgram& fp, mtl::pipeline_props& props)
		{
			normalize_cached_programs(vp, fp);
			normalize_cached_pipeline(fp, props);

			// The programs preload_programs() decompiled (the A2C flag set by normalize_cached_pipeline may make a new
			// fragment program: decompiled here, before load() returns)
			const auto vp_search = search_vertex_program(nullptr, vp);
			const auto fp_search = search_fragment_program(nullptr, fp);

			preload_entry entry{ &std::get<0>(vp_search), &std::get<0>(fp_search), props };
			entry.props.state.depth_stencil_format = 0; // Not part of the key (get_graphics_pipeline)

			std::lock_guard lock(m_preload_lock);
			m_preload_entries.push_back(entry);
		}

		void preload_programs(rsx::program_cache_hint_t* cache_hint, const RSXVertexProgram& vp, const RSXFragmentProgram& fp)
		{
			const auto start = std::chrono::steady_clock::now();

			// Same programs as add_pipeline_entry() will look up, except for the A2C flag (it depends on the pipeline
			// state, which is not passed here)
			if (needs_program_normalization(vp, fp))
			{
				RSXVertexProgram vp_ = vp;
				RSXFragmentProgram fp_ = fp;
				normalize_cached_programs(vp_, fp_);

				search_vertex_program(cache_hint, vp_);
				search_fragment_program(cache_hint, fp_);
			}
			else
			{
				search_vertex_program(cache_hint, vp);
				search_fragment_program(cache_hint, fp);
			}

			m_preload_decompile_us += static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
		}

		// Decompiler time of preload_programs() so far (all its threads)
		u64 get_preload_decompile_us() const
		{
			return m_preload_decompile_us;
		}

		// Queues the pipelines recorded by add_pipeline_entry() for the pipe compiler workers. RSX thread, after
		// rsx::shaders_cache::load() returned. Returns how many were queued; when the last one has run, the outcome is
		// logged and the pipeline archive is told (on_pipeline_cache_preloaded).
		u32 queue_preloaded_pipelines()
		{
			std::vector<preload_entry> entries;
			{
				std::lock_guard lock(m_preload_lock);
				entries.swap(m_preload_entries);
			}

			if (entries.empty())
			{
				return 0;
			}

			auto state = std::make_shared<shader_preload_state>();
			state->queue_time = get_system_time();
			state->total = ::size32(entries);
			state->remaining = state->total;
			m_preload = state;

			std::vector<std::function<void()>> tasks;
			tasks.reserve(entries.size());

			for (const preload_entry& entry : entries)
			{
				// The program cache outlives the pipe compiler (destroy_pipe_compiler drops the tasks that did not run)
				tasks.emplace_back([this, state, entry]()
				{
					run_preload_task(*state, entry);
				});
			}

			pipe_compiler::queue_preload_tasks(std::move(tasks));
			return state->total;
		}

		// The renderer stops: if the preload did not finish, logs how far it got and tells the pipeline archive that it
		// is incomplete. Call after destroy_pipe_compiler() (no task runs any more).
		void on_preload_interrupted()
		{
			if (const auto state = m_preload; state && !state->finished.exchange(true))
			{
				rsx_log.notice("Metal: the shader cache preload was interrupted: %u of %u pipeline(s) built, %u already built for draws, %u failed",
					state->built.load(), state->total, state->present.load(), state->failed.load());
				mtl::on_pipeline_cache_preloaded(false);
			}
		}

		// Some cached pipelines have not been through their preload task yet
		bool is_preload_running() const
		{
			const auto state = m_preload;
			return state && state->remaining;
		}

		bool check_cache_missed() const
		{
			return m_cache_miss_flag;
		}

		// The last get_graphics_pipeline() returned no pipeline because it could not be built (not because it is
		// still compiling). Such a pipeline is never retried.
		bool check_pipeline_failed() const
		{
			return m_last_pipeline_failed;
		}

		// Number of entries in each in-memory cache (telemetry for the periodic statistics report; sizes are read
		// under their own locks, so the three counts may race each other by a pipeline)
		struct cache_sizes
		{
			usz pipelines = 0;
			usz vertex_programs = 0;
			usz fragment_programs = 0;
		};

		cache_sizes get_cache_sizes()
		{
			cache_sizes sizes;
			{
				reader_lock lock(m_vertex_mutex);
				sizes.vertex_programs = m_vertex_shader_cache.size();
			}
			{
				reader_lock lock(m_fragment_mutex);
				sizes.fragment_programs = m_fragment_shader_cache.size();
			}
			{
				reader_lock lock(m_pipeline_mutex);
				sizes.pipelines = m_storage.size();
			}
			return sizes;
		}

		// Hides program_state_cache::clear
		void clear()
		{
			base_type::clear();
			m_failed_pipelines.clear();
			m_last_pipeline_failed = false;
		}

	private:
		pipeline_failure_registry m_failed_pipelines;
		bool m_last_pipeline_failed = false; // Like m_cache_miss_flag: describes the last lookup

		// A cached pipeline (add_pipeline_entry): programs of this cache, normalized properties as a draw looks them up
		struct preload_entry
		{
			const MTLVertexProgram* vp = nullptr;
			const MTLFragmentProgram* fp = nullptr;
			mtl::pipeline_props props{};
		};

		std::mutex m_preload_lock;
		std::vector<preload_entry> m_preload_entries;  // Recorded during rsx::shaders_cache::load()
		atomic_t<u64> m_preload_decompile_us = 0;
		std::shared_ptr<shader_preload_state> m_preload;

		enum class preload_result
		{
			built,
			present,
			failed,
		};

		// Builds the pipeline of a cache entry on this (worker) thread, unless it exists or is being built for a draw.
		// Same key and stored result as a draw's get_graphics_pipeline() of the same programs and state, without touching
		// what describes the RSX thread's last lookup (m_cache_miss_flag, m_last_pipeline_failed).
		preload_result build_preloaded_pipeline(const preload_entry& entry)
		{
			// The key a draw looks up (the properties before validation, see program_state_cache::get_graphics_pipeline)
			mtl::pipeline_props props = entry.props;
			const typename decltype(m_storage)::key_type key{ entry.vp->id, entry.fp->id, props };

			{
				std::lock_guard lock(m_pipeline_mutex);
				if (m_storage.find(key) != m_storage.end())
				{
					return preload_result::present;
				}

				// Placeholder: a draw that needs this pipeline now finds it pending (skipped, interpreted, or waited for)
				m_storage[key] = std::move(__null_pipeline_handle);
			}

			// Built with validated properties, as get_graphics_pipeline does for programs that exist
			MTLTraits::validate_pipeline_properties(*entry.vp, *entry.fp, props);

			auto store = [this, key](std::unique_ptr<mtl::glsl::program>& pipeline) -> mtl::glsl::program*
			{
				if (!pipeline)
				{
					// The placeholder stays; the failure is registered (MTLTraits::build_pipeline)
					return nullptr;
				}

				std::lock_guard lock(m_pipeline_mutex);
				auto& result = m_storage[key];
				result = std::move(pipeline);
				return result.get();
			};

			return MTLTraits::build_pipeline(*entry.vp, *entry.fp, props, false, store, m_failed_pipelines)
				? preload_result::built
				: preload_result::failed;
		}

		void run_preload_task(shader_preload_state& state, const preload_entry& entry)
		{
			const auto start = std::chrono::steady_clock::now();

			switch (build_preloaded_pipeline(entry))
			{
			case preload_result::built: state.built++; break;
			case preload_result::present: state.present++; break;
			case preload_result::failed: state.failed++; break;
			}

			{
				// The worker reset this thread's timings when the task started
				std::lock_guard lock(state.timings_lock);
				state.timings.add(this_thread_compile_timings());
				state.task_us += static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
			}

			if (--state.remaining == 0 && !state.finished.exchange(true))
			{
				report_preload_finished(state);
			}
		}

		static void report_preload_finished(shader_preload_state& state)
		{
			compile_timings t;
			u64 task_us;
			{
				std::lock_guard lock(state.timings_lock);
				t = state.timings;
				task_us = state.task_us;
			}

			const auto ms = [](u64 us) { return us / 1000.; };
			const auto avg_ms = [](u64 us, u32 count) { return count ? us / 1000. / count : 0.; };

			rsx_log.notice("Metal: shader cache preload finished in the background %.0f ms after it was queued: %u pipeline(s) built, %u already "
				"built for draws, %u failed; %.0f ms of worker time: GLSL->MSL %u shader(s) %.0f ms (%.1f ms each), MTLLibrary %u in %.0f ms "
				"(%.1f ms each), pipelines with archive lookups %u in %.0f ms (%.1f ms each), compiled without %u in %.0f ms (%.1f ms each), "
				"specialized %u in %.0f ms",
				ms(get_system_time() - state.queue_time), state.built.load(), state.present.load(), state.failed.load(), ms(task_us),
				t.translated, ms(t.translate_us), avg_ms(t.translate_us, t.translated),
				t.libraries, ms(t.library_us), avg_ms(t.library_us, t.libraries),
				t.archive_pipelines, ms(t.archive_pipeline_us), avg_ms(t.archive_pipeline_us, t.archive_pipelines),
				t.compiled_pipelines, ms(t.compiled_pipeline_us), avg_ms(t.compiled_pipeline_us, t.compiled_pipelines),
				t.specialized, ms(t.specialization_us));

			// Every cached pipeline has been built (or found built): the archive's first serializer holds them all
			mtl::on_pipeline_cache_preloaded(true);
		}

		// ---- Shader cache entries --------------------------------------------------------------------------------
		// Entries keep the state they were recorded with. The parts that depend on renderer settings instead of the
		// guest are brought to what this session produces for the same draw; otherwise the preload builds programs
		// and pipelines that no draw can use (e.g. 4-sample A2C pipelines recorded with MSAA on, now that it is off).

		static bool msaa_disabled()
		{
			// Same condition as MTLGSRender (backend_config.supports_hw_msaa/a2c/a2one) and
			// surface_cache_traits::create_new_surface (one sample per surface)
			return g_cfg.video.antialiasing_level == msaa_level::none;
		}

		static bool hardware_depth_bounds()
		{
			// Same condition as MTLGSRender::get_backend_fragment_program_export_config (the renderer exists during the
			// preload)
			return g_render_device && g_render_device->caps().depth_bounds;
		}

		static bool needs_program_normalization(const RSXVertexProgram& vp, const RSXFragmentProgram& fp)
		{
			return (msaa_disabled() &&
				(vp.texture_state.multisampled_textures || fp.texture_state.multisampled_textures || (fp.ctrl & RSX_SHADER_CONTROL_ROP_MULTISAMPLED))) ||
				(hardware_depth_bounds() && (fp.ctrl & RSX_SHADER_CONTROL_DEPTH_BOUNDS_TEST));
		}

		static void normalize_cached_programs(RSXVertexProgram& vp, RSXFragmentProgram& fp)
		{
			if ((fp.ctrl & RSX_SHADER_CONTROL_DEPTH_BOUNDS_TEST) && hardware_depth_bounds())
			{
				// Recorded on a GPU without a hardware depth bounds test: this one tests in hardware and never flags programs
				// for it. The multisampled flag only came with the test unless programmable blending or depth compare
				// emulation asked for it too (get_fragment_program_export_config).
				fp.ctrl &= ~static_cast<u32>(RSX_SHADER_CONTROL_DEPTH_BOUNDS_TEST);
				if (!(fp.ctrl & (RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING | RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE)))
				{
					fp.ctrl &= ~static_cast<u32>(RSX_SHADER_CONTROL_ROP_MULTISAMPLED);
				}
			}

			if (!msaa_disabled())
			{
				return;
			}

			// Without hardware MSAA the RSX never flags multisampled textures or multisampled ROP output
			// (backend_config.supports_hw_msaa = false)
			vp.texture_state.multisampled_textures = 0;
			fp.texture_state.multisampled_textures = 0;
			fp.ctrl &= ~RSX_SHADER_CONTROL_ROP_MULTISAMPLED;
		}

		static void normalize_cached_pipeline(RSXFragmentProgram& fp, mtl::pipeline_props& props)
		{
			auto& state = props.state;
			state.topology_class = get_pipeline_topology_class(state.topology_class);

			if (!msaa_disabled())
			{
				return;
			}

			// Without hardware A2C the RSX has the fragment program emulate it (RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE).
			// decode_rsx_state only enables hardware A2C when A2C is on and the target is multisampled, which is
			// exactly when the RSX leaves the emulation flag out.
			if (state.alpha_to_coverage)
			{
				fp.ctrl |= RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE;
			}

			state.sample_count = 1;
			state.alpha_to_coverage = 0;
			state.alpha_to_one = 0;
		}
	};
}

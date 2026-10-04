#include "stdafx.h"
#include "MTLPipelineCompiler.h"
#include "mtlutils/device.h"

#include "Emu/system_config.h"
#include "Utilities/Thread.h"
#include "Utilities/mutex.h"
#include "util/sysinfo.hpp"
#include "util/asm.hpp"
#include "util/fnv_hash.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <optional>
#include <unordered_map>

#ifdef __APPLE__
#include <pthread.h>
#endif

namespace mtl
{
	// Global list of worker threads
	static std::unique_ptr<named_thread_group<pipe_compiler>> g_pipe_compilers;

	// The full-state version of a render pipeline created by specialization, built in the background. Shared by the
	// program (glsl::program::m_full_state_upgrade) and the job that builds it, so either may go first.
	struct full_state_upgrade
	{
		enum : u32
		{
			pending,
			ready,
			failed,
		};

		mtl::ref<MTL4::RenderPipelineDescriptor> descriptor; // The full-state descriptor the program was specialized with
		u64 key = 0;                                         // Its pipeline archive key
		bool use_archive = true;                             // uses_pipeline_archive() of its shaders
		atomic_t<u32> status = pending;                      // Written by the worker, read by the binding thread
		MTL::RenderPipelineState* pipeline = nullptr;        // +1, published by status = ready

		u32 draws = 0;                                       // Binding thread only
		bool requested = false;                              // Binding thread only

		full_state_upgrade(mtl::ref<MTL4::RenderPipelineDescriptor> desc, u64 key_, bool use_archive_)
			: descriptor(std::move(desc))
			, key(key_)
			, use_archive(use_archive_)
		{
		}

		full_state_upgrade(const full_state_upgrade&) = delete;
		full_state_upgrade& operator=(const full_state_upgrade&) = delete;

		~full_state_upgrade()
		{
			// Built but never swapped in: no GPU work used it
			if (pipeline)
			{
				pipeline->release();
			}
		}
	};

	struct pipe_compiler::job_queue
	{
		std::mutex mutex;
		std::deque<pipe_compiler_job> jobs;   // Oldest first, except for jobs moved to the front by prioritize_jobs() and
		                                      // COMPILE_AHEAD jobs (behind those, ahead of the other background jobs)
		std::deque<pipe_compiler_job> preload;  // Shader cache preload tasks, oldest first
		std::deque<pipe_compiler_job> upgrades; // Background full-state compiles, oldest first
		atomic_t<u32> pushed = 0;             // Bumped by every push, prioritize_jobs() and finished background job; idle workers wait on it
		atomic_t<u32> completed = 0;          // Bumped after every finished deferred job (see get_completed_job_count)

		// Jobs nobody waits for ("background") run on at most this many workers at a time (see initialize_pipe_compiler).
		// Urgent jobs (the RSX thread is waiting for them, prioritize_jobs) may use any idle worker. Order of `jobs`:
		// urgent, then COMPILE_AHEAD (the shader interpreter's uber pipelines: they draw for every program of a render
		// state), then the other background jobs (recompiled pipelines draws need, the interpreter's pipelines
		// specialized for a program). Shader cache preload tasks are background jobs that only start when `jobs` is
		// empty and the pipeline archive is ready for lookups: every pipeline a draw asks for goes ahead of them (a
		// draw never finds a preload task's pipeline half-built unless that task is running). Full-state upgrades are
		// background jobs too, one at a time, and only start when `jobs` is empty and no preload task can start: they
		// replace pipelines that already draw, so they never delay one that is missing.
		u32 background_limit = umax;
		u32 running_background = 0;           // Background jobs being compiled, upgrades included (under `mutex`)
		u32 running_upgrades = 0;             // Upgrades being compiled (under `mutex`)
		static constexpr u32 max_running_upgrades = 1;

		void push(pipe_compiler_job&& job)
		{
			{
				std::lock_guard lock(mutex);

				if (job.kind == job_kind::full_state_upgrade)
				{
					upgrades.push_back(std::move(job));
				}
				else if (job.ahead)
				{
					// Behind the urgent jobs and the earlier COMPILE_AHEAD jobs (FIFO among them), before the rest
					const auto position = std::find_if(jobs.begin(), jobs.end(), FN(!x.urgent && !x.ahead));
					jobs.insert(position, std::move(job));
				}
				else
				{
					jobs.push_back(std::move(job));
				}
			}

			wake_workers();
		}

		// Every idle worker retries (pop() decides who may take what): cheap, a handful of threads per event
		void wake_workers()
		{
			pushed++;
			pushed.notify_all();
		}

		std::optional<pipe_compiler_job> pop()
		{
			std::lock_guard lock(mutex);
			if (jobs.empty())
			{
				if (running_background >= background_limit)
				{
					return std::nullopt;
				}

				// Nothing a draw needs is queued: the shader cache preload, once the archive can look its binaries up
				if (!preload.empty() && pipeline_archive_ready())
				{
					std::optional<pipe_compiler_job> result(std::move(preload.front()));
					preload.pop_front();
					running_background++;
					return result;
				}

				// Nothing else waits: a full-state upgrade may take a background slot
				if (upgrades.empty() || running_upgrades >= max_running_upgrades)
				{
					return std::nullopt;
				}

				std::optional<pipe_compiler_job> result(std::move(upgrades.front()));
				upgrades.pop_front();
				running_background++;
				running_upgrades++;
				return result;
			}

			// Urgent jobs form the head of the queue (prioritize_jobs moves them there), so a non-urgent head means that
			// only background jobs are queued
			if (!jobs.front().urgent && running_background >= background_limit)
			{
				return std::nullopt;
			}

			std::optional<pipe_compiler_job> result(std::move(jobs.front()));
			jobs.pop_front();

			if (!result->urgent)
			{
				running_background++;
			}

			return result;
		}

		void on_background_job_done(bool upgrade)
		{
			{
				std::lock_guard lock(mutex);
				ensure(running_background > 0);
				running_background--;

				if (upgrade)
				{
					ensure(running_upgrades > 0);
					running_upgrades--;
				}
			}

			// A worker may be idle because the background slots were taken
			wake_workers();
		}
	};

	pipe_compiler::job_queue pipe_compiler::s_queue;

	namespace
	{
		bool use_fast_math()
		{
			return !g_cfg.video.disable_msl_fast_math;
		}

		// FNV-1a over 64-bit words (then bytes), length first
		u64 hash_bytes(u64 hash, const void* data, usz size)
		{
			const auto* bytes = static_cast<const u8*>(data);
			hash = rpcs3::hash64(hash, u64{size});

			for (; size >= sizeof(u64); bytes += sizeof(u64), size -= sizeof(u64))
			{
				u64 word;
				std::memcpy(&word, bytes, sizeof(word));
				hash = rpcs3::hash64(hash, word);
			}

			for (; size; bytes++, size--)
			{
				hash = rpcs3::hash64(hash, *bytes);
			}

			return hash;
		}

		enum pipeline_key_kind : u64
		{
			key_full_state_render = 1,
			key_compute = 2,
			key_unspecialized_render = 3,
		};

		// Pipeline key for the pipeline archive (mtl::new_render_pipeline_state): the descriptors built below are a
		// deterministic function of the translated shaders (their MSL, which reflects the binding layout: msl_hash()) and
		// `state` (fast math has separate archives). Never 0, which means "unknown".
		u64 make_pipeline_key(pipeline_key_kind kind, std::initializer_list<u64> msl_hashes, const void* state = nullptr, usz state_size = 0)
		{
			u64 hash = rpcs3::hash64(rpcs3::fnv_seed, u64{kind});

			for (const u64 msl_hash : msl_hashes)
			{
				hash = rpcs3::hash64(hash, msl_hash);
			}

			hash = hash_bytes(hash, state, state_size);
			return hash ? hash : 1;
		}

		// Size in bytes of the vertex fetch formats produced by the translator's stage_in reflection
		u32 get_vertex_format_size(MTL::VertexFormat format)
		{
			switch (format)
			{
			case MTL::VertexFormatHalf: return 2;
			case MTL::VertexFormatHalf2: return 4;
			case MTL::VertexFormatHalf3: return 6;
			case MTL::VertexFormatHalf4: return 8;
			case MTL::VertexFormatFloat: case MTL::VertexFormatInt: case MTL::VertexFormatUInt: return 4;
			case MTL::VertexFormatFloat2: case MTL::VertexFormatInt2: case MTL::VertexFormatUInt2: return 8;
			case MTL::VertexFormatFloat3: case MTL::VertexFormatInt3: case MTL::VertexFormatUInt3: return 12;
			case MTL::VertexFormatFloat4: case MTL::VertexFormatInt4: case MTL::VertexFormatUInt4: return 16;
			default: return 0;
			}
		}

		// Builds the implicit vertex descriptor for vertex shaders with [[stage_in]] attributes (see
		// stage_in_vertex_buffer_index) from the attribute list reflected once at translation time (no MTLFunction
		// reflection per pipeline). Returns an empty ref if the shader fetches no attributes (all RSX programs).
		mtl::ref<MTL::VertexDescriptor> make_stage_in_descriptor(const glsl::shader& vs, const glsl::binding_layout& vs_layout, bool& error)
		{
			error = false;

			const auto& attributes = vs.vertex_attributes();
			if (attributes.empty())
			{
				return {};
			}

			if (vs_layout.buffer_count > stage_in_vertex_buffer_index)
			{
				rsx_log.error("[MSL] Vertex shader uses input attributes and %u buffers; buffer %u is reserved for vertex fetch",
					vs_layout.buffer_count, stage_in_vertex_buffer_index);
				error = true;
				return {};
			}

			auto descriptor = mtl::ref(MTL::VertexDescriptor::alloc()->init());
			u32 offset = 0;

			// Sorted by location: interleaved, tightly packed (4-byte aligned) in location order
			for (const auto& [location, format] : attributes)
			{
				const u32 size = get_vertex_format_size(format);
				if (!size)
				{
					rsx_log.error("[MSL] Unsupported vertex attribute format %u at location %u", static_cast<u32>(format), location);
					error = true;
					return {};
				}

				auto attribute_desc = descriptor->attributes()->object(location);
				attribute_desc->setFormat(format);
				attribute_desc->setOffset(offset);
				attribute_desc->setBufferIndex(stage_in_vertex_buffer_index);
				offset = utils::align(offset + size, 4u);
			}

			auto layout_desc = descriptor->layouts()->object(stage_in_vertex_buffer_index);
			layout_desc->setStride(offset);
			layout_desc->setStepFunction(MTL::VertexStepFunctionPerVertex);
			layout_desc->setStepRate(1);

			return descriptor;
		}

		// The shader's function in its library. A function that declares function constants is always specialized, with
		// the shader's values (glsl::shader::create_specialization); constants without a value keep their GLSL default
		// (SPIRV-Cross declares them optional), so the pipeline compile folds all of them.
		mtl::ref<MTL4::FunctionDescriptor> make_function_descriptor(const glsl::shader& shader)
		{
			auto library_function = mtl::ref(MTL4::LibraryFunctionDescriptor::alloc()->init());
			library_function->setLibrary(shader.library());
			library_function->setName(mtl::ns_str(shader.entry_point()));

			const auto& declared = shader.declared_function_constants();
			if (declared.empty())
			{
				return mtl::ref<MTL4::FunctionDescriptor>(library_function.release_ownership());
			}

			const auto& constants = shader.function_constants();
			auto values = mtl::ref(MTL::FunctionConstantValues::alloc()->init());

			for (const auto& constant : constants)
			{
				const auto found = std::find_if(declared.cbegin(), declared.cend(), FN(x.first == constant.id));
				if (found == declared.cend())
				{
					// Not declared by the MSL (unused by the shader): nothing to set
					continue;
				}

				switch (const MTL::DataType type = found->second)
				{
				case MTL::DataTypeBool:
				{
					const bool value = constant.value != 0;
					values->setConstantValue(&value, type, constant.id);
					break;
				}
				default:
				{
					// DataTypeUInt / DataTypeInt / DataTypeFloat: 32 bits, stored as is
					const u32 value = constant.value;
					values->setConstantValue(&value, type, constant.id);
					break;
				}
				}
			}

			auto specialized = mtl::ref(MTL4::SpecializedFunctionDescriptor::alloc()->init());
			specialized->setFunctionDescriptor(library_function.get());
			specialized->setConstantValues(values.get());
			return mtl::ref<MTL4::FunctionDescriptor>(specialized.release_ownership());
		}

		// Pipeline archive key of a (possibly specialized) shader pair: its MSL does not show the constant values
		u64 mix_function_constants(u64 key, const glsl::shader& shader)
		{
			for (const auto& constant : shader.function_constants())
			{
				key = rpcs3::hash64(key, (u64{constant.id} << 32) | constant.value);
			}

			return key ? key : 1;
		}

		// ---- Flexible render pipeline states ------------------------------------------------------------------------

		// Telemetry (get_pipeline_creation_stats_and_reset)
		struct
		{
			std::atomic<u32> specialized{ 0 };
			std::atomic<u64> specialization_us{ 0 };
			std::atomic<u64> specialization_max_us{ 0 };
			std::atomic<u32> full_state{ 0 };
			std::atomic<u32> full_state_archived{ 0 };
			std::atomic<u32> fallbacks{ 0 };
			std::atomic<u32> unspecialized{ 0 };
			std::atomic<u64> unspecialized_us{ 0 };
			std::atomic<u32> upgrades_requested{ 0 };
			std::atomic<u32> upgrades_built{ 0 };
			std::atomic<u32> upgrades_swapped{ 0 };
		} g_stats;

		// Build times of all threads since the last get_compile_timings_and_reset() (record_compile_time)
		struct
		{
			std::atomic<u32> translated{ 0 };
			std::atomic<u64> translate_us{ 0 };
			std::atomic<u32> libraries{ 0 };
			std::atomic<u64> library_us{ 0 };
			std::atomic<u32> archive_pipelines{ 0 };
			std::atomic<u64> archive_pipeline_us{ 0 };
			std::atomic<u32> compiled_pipelines{ 0 };
			std::atomic<u64> compiled_pipeline_us{ 0 };
		} g_window_timings;

		thread_local compile_timings t_compile_timings;

		// Failures of the flexible path (an unspecialized build, a specialization, a background full-state compile) are
		// runtime failures of a Metal API: the first ones are logged with the error, later ones only counted
		constexpr u32 max_flexible_failure_logs = 8;
		std::atomic<u32> g_flexible_failure_logs{ 0 };

		bool should_log_flexible_failure()
		{
			const u32 count = g_flexible_failure_logs++;
			if (count == max_flexible_failure_logs)
			{
				rsx_log.error("Metal: further flexible render pipeline failures are not logged (see the pipeline statistics)");
			}

			return count < max_flexible_failure_logs;
		}

		// Logical-to-physical colour attachment mapping of a render pipeline: the one place that decides it, for the
		// unspecialized, specialization and full-state descriptors alike. Full-compile state: the property has no
		// Unspecialized value (MTL4::LogicalToPhysicalColorAttachmentMappingState is Identity or Inherited), so an
		// unspecialized pipeline is compiled with it and its specializations keep it; every descriptor of a pipeline
		// carries the same value. It is the same for every pipeline, so no key needs it (a state field that ever selected
		// it would key the unspecialized pipeline: get_unspecialized_state keeps every field but the colour configuration;
		// changing the value for every pipeline bumps the pipeline archive format).
		// Identity: fragment output N is colour attachment N (render passes do not support colour attachment mapping).
		MTL4::LogicalToPhysicalColorAttachmentMappingState get_color_attachment_mapping_state(const glsl::graphics_pipeline_state& /*state*/)
		{
			return MTL4::LogicalToPhysicalColorAttachmentMappingStateIdentity;
		}

		// What an unspecialized pipeline is compiled for: `state` without the colour attachment configuration
		// (formats, write masks, blending), which each specialization provides
		glsl::graphics_pipeline_state get_unspecialized_state(const glsl::graphics_pipeline_state& state)
		{
			glsl::graphics_pipeline_state result = state;
			result.color = {};
			result.color_count = 0;
			result.depth_stencil_format = 0; // Not part of an MTL4 render pipeline
			result.topology_class = get_pipeline_topology_class(state.topology_class);
			return result;
		}

		u64 get_full_state_key(const glsl::shader& vs, const glsl::shader& fs, const glsl::graphics_pipeline_state& state)
		{
			glsl::graphics_pipeline_state key_state = state;
			key_state.depth_stencil_format = 0; // Not part of an MTL4 render pipeline
			key_state.topology_class = get_pipeline_topology_class(state.topology_class);
			const u64 key = mix_function_constants(make_pipeline_key(key_full_state_render,
				{ vs.msl_hash(), state.rasterization_enabled ? fs.msl_hash() : 0 }, &key_state, sizeof(key_state)), vs);
			return state.rasterization_enabled ? mix_function_constants(key, fs) : key;
		}

		// A specialization only replaces the fragment output part, so the fragment body must not depend on the colour
		// attachments. Framebuffer fetch ([[color(n)]]: programmable blending) reads them in the body, which an
		// unspecialized pipeline compiles without knowing their formats: such pipelines are built with full state, like
		// pipelines without a fragment stage (nothing to specialize). input_type_attachment inputs exist exactly for RSX
		// programs with RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING, which the shader interpreter does not run.
		bool is_specializable(const glsl::graphics_pipeline_state& state, const std::vector<glsl::program_input>& fs_inputs)
		{
			return state.rasterization_enabled && std::none_of(fs_inputs.begin(), fs_inputs.end(), [](const glsl::program_input& in)
			{
				return in.type == glsl::input_type_attachment;
			});
		}

		enum class color_attachments_mode
		{
			concrete,      // From `state`: a full-state pipeline, or the descriptor a specialization takes them from
			unspecialized, // Every attachment a specialization may configure is left unspecialized
		};

		// nullptr (logged) if the stage_in vertex descriptor cannot be built
		mtl::ref<MTL4::RenderPipelineDescriptor> make_render_pipeline_descriptor(
			const glsl::shader& vs,
			const glsl::shader& fs,
			const glsl::binding_layout& vs_layout,
			const glsl::graphics_pipeline_state& state,
			color_attachments_mode colors)
		{
			const bool rasterization_enabled = !!state.rasterization_enabled;

			auto descriptor = mtl::ref(MTL4::RenderPipelineDescriptor::alloc()->init());

			auto vs_function = make_function_descriptor(vs);
			descriptor->setVertexFunctionDescriptor(vs_function.get());

			// Metal requires a nil fragment function when rasterization is disabled
			if (rasterization_enabled)
			{
				auto fs_function = make_function_descriptor(fs);
				descriptor->setFragmentFunctionDescriptor(fs_function.get());
			}

			bool vertex_desc_error = false;
			auto vertex_descriptor = make_stage_in_descriptor(vs, vs_layout, vertex_desc_error);
			if (vertex_desc_error)
			{
				return {};
			}

			if (vertex_descriptor)
			{
				descriptor->setVertexDescriptor(vertex_descriptor.get());
			}

			descriptor->setRasterizationEnabled(rasterization_enabled);
			// RSX vertex programs always write gl_PointSize ([[point_size]] in MSL). Metal rejects such a vertex function
			// in a pipeline declared as line/triangle ("Vertex shader writes point size but inputPrimitiveTopology is
			// MTLPrimitiveTopologyClassTriangle"), so only point pipelines declare their class. Unspecified is Metal's
			// default and is only required to be explicit for layered rendering, which RSX never uses. Keys store the class
			// as get_pipeline_topology_class() normalizes it, so this never builds a line and a triangle variant.
			descriptor->setInputPrimitiveTopology(state.topology_class == static_cast<u8>(MTL::PrimitiveTopologyClassPoint)
				? MTL::PrimitiveTopologyClassPoint
				: MTL::PrimitiveTopologyClassUnspecified);
			descriptor->setRasterSampleCount(std::max<u32>(1u, state.sample_count));
			descriptor->setAlphaToCoverageState(state.alpha_to_coverage ? MTL4::AlphaToCoverageStateEnabled : MTL4::AlphaToCoverageStateDisabled);
			descriptor->setAlphaToOneState(state.alpha_to_one ? MTL4::AlphaToOneStateEnabled : MTL4::AlphaToOneStateDisabled);
			descriptor->setColorAttachmentMappingState(get_color_attachment_mapping_state(state));

			if (colors == color_attachments_mode::unspecialized)
			{
				// A specialization only sets the properties that are unspecialized here, and every attachment it leaves
				// out gets the defaults (no format), so all of them are unspecialized, blend equation included
				for (u32 i = 0; i < ::size32(state.color); ++i)
				{
					auto attachment = descriptor->colorAttachments()->object(i);
					attachment->setPixelFormat(MTL::PixelFormatUnspecialized);
					attachment->setWriteMask(MTL::ColorWriteMaskUnspecialized);
					attachment->setBlendingState(MTL4::BlendStateUnspecialized);
					attachment->setSourceRGBBlendFactor(MTL::BlendFactorUnspecialized);
					attachment->setDestinationRGBBlendFactor(MTL::BlendFactorUnspecialized);
					attachment->setRgbBlendOperation(MTL::BlendOperationUnspecialized);
					attachment->setSourceAlphaBlendFactor(MTL::BlendFactorUnspecialized);
					attachment->setDestinationAlphaBlendFactor(MTL::BlendFactorUnspecialized);
					attachment->setAlphaBlendOperation(MTL::BlendOperationUnspecialized);
				}

				return descriptor;
			}

			const u32 color_count = std::min<u32>(state.color_count, ::size32(state.color));
			for (u32 i = 0; i < color_count; ++i)
			{
				const auto& color = state.color[i];
				if (!color.pixel_format)
				{
					continue;
				}

				auto attachment = descriptor->colorAttachments()->object(i);
				attachment->setPixelFormat(static_cast<MTL::PixelFormat>(color.pixel_format));
				attachment->setWriteMask(static_cast<MTL::ColorWriteMask>(color.write_mask & MTL::ColorWriteMaskAll));

				if (color.blend_enable)
				{
					attachment->setBlendingState(MTL4::BlendStateEnabled);
					attachment->setSourceRGBBlendFactor(static_cast<MTL::BlendFactor>(color.src_rgb));
					attachment->setDestinationRGBBlendFactor(static_cast<MTL::BlendFactor>(color.dst_rgb));
					attachment->setRgbBlendOperation(static_cast<MTL::BlendOperation>(color.op_rgb));
					attachment->setSourceAlphaBlendFactor(static_cast<MTL::BlendFactor>(color.src_a));
					attachment->setDestinationAlphaBlendFactor(static_cast<MTL::BlendFactor>(color.dst_a));
					attachment->setAlphaBlendOperation(static_cast<MTL::BlendOperation>(color.op_a));
				}
				else
				{
					attachment->setBlendingState(MTL4::BlendStateDisabled);
				}
			}

			return descriptor;
		}

		// An unspecialized pipeline, built once per key by the first thread that needs it
		struct unspecialized_pipeline
		{
			enum : u32
			{
				building,
				ready,
				failed,
			};

			atomic_t<u32> status = building;
			MTL::RenderPipelineState* pipeline = nullptr; // +1, published by status = ready. Never bound: released directly

			unspecialized_pipeline() = default;
			unspecialized_pipeline(const unspecialized_pipeline&) = delete;
			unspecialized_pipeline& operator=(const unspecialized_pipeline&) = delete;

			~unspecialized_pipeline()
			{
				if (pipeline)
				{
					pipeline->release();
				}
			}
		};

		struct unspecialized_key
		{
			u64 vs_uid = 0;
			u64 fs_uid = 0;
			glsl::graphics_pipeline_state state{}; // get_unspecialized_state()

			bool operator==(const unspecialized_key& other) const
			{
				return vs_uid == other.vs_uid && fs_uid == other.fs_uid && std::memcmp(&state, &other.state, sizeof(state)) == 0;
			}
		};

		struct unspecialized_key_hash
		{
			usz operator()(const unspecialized_key& key) const
			{
				return rpcs3::hash64(rpcs3::hash64(rpcs3::hash_struct(key.state), key.vs_uid), key.fs_uid);
			}
		};

		// Unspecialized pipelines by shader pair and compile state. Shaders are identified by uid (never reused), so an
		// entry can't be taken for another pair's once its shaders are gone; such entries stay until
		// destroy_pipe_compiler() (shaders live as long as the renderer).
		class unspecialized_pipeline_cache
		{
			shared_mutex m_lock;
			std::unordered_map<unspecialized_key, std::shared_ptr<unspecialized_pipeline>, unspecialized_key_hash> m_entries;

		public:
			std::shared_ptr<unspecialized_pipeline> find(const unspecialized_key& key)
			{
				reader_lock lock(m_lock);
				const auto found = m_entries.find(key);
				return found != m_entries.end() ? found->second : nullptr;
			}

			// The entry of `key`. `created`: it is new, and the caller must build it and publish the result.
			std::shared_ptr<unspecialized_pipeline> find_or_create(const unspecialized_key& key, bool& created)
			{
				std::lock_guard lock(m_lock);
				auto& entry = m_entries[key];
				created = !entry;

				if (created)
				{
					entry = std::make_shared<unspecialized_pipeline>();
				}

				return entry;
			}

			void clear()
			{
				std::lock_guard lock(m_lock);
				m_entries.clear();
			}
		};

		unspecialized_pipeline_cache g_unspecialized_pipelines;

		// The unspecialized pipeline for `key`, built here unless another thread builds (or has built) it: then its build
		// is waited for. nullptr if it cannot be built (logged once per key; its pipelines get full state).
		std::shared_ptr<unspecialized_pipeline> get_unspecialized_pipeline(
			const unspecialized_key& key,
			const glsl::shader& vs,
			const glsl::shader& fs,
			const glsl::binding_layout& vs_layout,
			const glsl::graphics_pipeline_state& state,
			bool use_archive)
		{
			bool created = false;
			auto entry = g_unspecialized_pipelines.find_or_create(key, created);

			if (!created)
			{
				// Bounded by a deadline: a builder that dies without publishing must not hang waiters.
				// nullptr falls back to full-state compile, an already-handled outcome.
				static constexpr auto build_wait_timeout = std::chrono::seconds(30);
				const auto wait_start = std::chrono::steady_clock::now();

				u32 status;
				while ((status = entry->status.load()) == unspecialized_pipeline::building)
				{
					if (std::chrono::steady_clock::now() - wait_start > build_wait_timeout)
					{
						rsx_log.error("Metal: timed out waiting for an unspecialized pipeline build; falling back to full-state compile");
						return nullptr;
					}

					entry->status.wait(unspecialized_pipeline::building, atomic_wait_timeout{10'000'000});
				}

				return status == unspecialized_pipeline::ready ? entry : nullptr;
			}

			const auto start = std::chrono::steady_clock::now();
			NS::Error* error = nullptr;
			MTL::RenderPipelineState* pipeline = nullptr;

			if (auto descriptor = make_render_pipeline_descriptor(vs, fs, vs_layout, state, color_attachments_mode::unspecialized))
			{
				// Through the pipeline archive: found there in later sessions
				pipeline = mtl::new_render_pipeline_state(descriptor.get(), &error,
					mix_function_constants(mix_function_constants(make_pipeline_key(key_unspecialized_render, { vs.msl_hash(), fs.msl_hash() },
						&key.state, sizeof(key.state)), vs), fs),
					render_pipeline_kind::unspecialized, use_archive);
			}

			if (pipeline)
			{
				entry->pipeline = pipeline;
				g_stats.unspecialized++;
				g_stats.unspecialized_us += static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
			}
			else if (should_log_flexible_failure())
			{
				rsx_log.error("Metal: cannot build an unspecialized render pipeline (%u samples, topology class %u): %s. "
					"Its render pipelines are compiled with full state.", key.state.sample_count, key.state.topology_class, mtl::to_string(error));
			}

			entry->status = pipeline ? unspecialized_pipeline::ready : unspecialized_pipeline::failed;
			entry->status.notify_all();

			return pipeline ? entry : nullptr;
		}

		// Creates the pipeline of `descriptor`'s colour attachment configuration from `unspecialized` (+1 result).
		// nullptr (logged) on failure.
		MTL::RenderPipelineState* specialize_render_pipeline(const MTL4::RenderPipelineDescriptor* descriptor, MTL::RenderPipelineState* unspecialized,
			const glsl::graphics_pipeline_state& state)
		{
			const auto start = std::chrono::steady_clock::now();

			// The device's compiler, not the pipeline archive's: a specialization can't be looked up (MTLPipelineArchive.cpp).
			// Only the properties that are unspecialized in `unspecialized` are taken from the descriptor.
			NS::Error* error = nullptr;
			MTL::RenderPipelineState* pipeline = g_render_device->compiler()->newRenderPipelineStateBySpecialization(descriptor, unspecialized, &error);

			if (!pipeline)
			{
				if (should_log_flexible_failure())
				{
					rsx_log.error("Metal: cannot specialize a render pipeline (%u color attachments, %u samples): %s. It is compiled with full state.",
						state.color_count, state.sample_count, mtl::to_string(error));
				}

				return nullptr;
			}

			const u64 elapsed_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
			record_compile_time(compile_step::specialization, elapsed_us);
			g_stats.specialized++;
			g_stats.specialization_us += elapsed_us;

			u64 longest = g_stats.specialization_max_us.load(std::memory_order_relaxed);
			while (elapsed_us > longest && !g_stats.specialization_max_us.compare_exchange_weak(longest, elapsed_us, std::memory_order_relaxed))
			{
			}

			return pipeline;
		}

		std::unique_ptr<glsl::program> make_render_program(
			MTL::RenderPipelineState* pipeline,
			const glsl::shader& vs,
			const glsl::shader& fs,
			const glsl::graphics_pipeline_state& state,
			const std::vector<glsl::program_input>& vs_inputs,
			const std::vector<glsl::program_input>& fs_inputs)
		{
			auto result = std::make_unique<glsl::program>(pipeline, vs_inputs, fs_inputs);

			if (vs.needs_buffer_size_buffer())
			{
				result->enable_buffer_size_table(glsl::binding_set_index_vertex);
			}

			if (state.rasterization_enabled && fs.needs_buffer_size_buffer())
			{
				result->enable_buffer_size_table(glsl::binding_set_index_fragment);
			}

			result->set_storage_writes(glsl::binding_set_index_vertex, vs.writes_storage());
			result->set_storage_writes(glsl::binding_set_index_fragment, state.rasterization_enabled && fs.writes_storage());

			return result;
		}

		// A program with a specialized pipeline: it keeps the full-state descriptor for its background upgrade
		std::unique_ptr<glsl::program> make_specialized_program(
			MTL::RenderPipelineState* pipeline,
			mtl::ref<MTL4::RenderPipelineDescriptor> descriptor,
			u64 full_state_key,
			bool use_archive,
			const glsl::shader& vs,
			const glsl::shader& fs,
			const glsl::graphics_pipeline_state& state,
			const std::vector<glsl::program_input>& vs_inputs,
			const std::vector<glsl::program_input>& fs_inputs)
		{
			auto result = make_render_program(pipeline, vs, fs, state, vs_inputs, fs_inputs);
			result->set_full_state_upgrade(std::make_shared<full_state_upgrade>(std::move(descriptor), full_state_key, use_archive));
			return result;
		}

		// Whether the render pipelines of this shader pair use the pipeline archive (uses_pipeline_archive)
		bool uses_pipeline_archive(const glsl::shader& vs, const glsl::shader& fs, const glsl::graphics_pipeline_state& state)
		{
			return mtl::uses_pipeline_archive(vs) && (!state.rasterization_enabled || mtl::uses_pipeline_archive(fs));
		}

		// Synchronous render pipeline build: specialization of the (possibly just built) unspecialized pipeline, or full
		// state (archived, not specializable, `full_state_only`, or the flexible path failed). nullptr (logged) on failure.
		std::unique_ptr<glsl::program> build_render_program(
			glsl::shader& vs,
			glsl::shader& fs,
			const glsl::graphics_pipeline_state& state,
			const std::vector<glsl::program_input>& vs_inputs,
			const std::vector<glsl::program_input>& fs_inputs,
			bool full_state_only)
		{
			ensure(g_render_device);
			mtl::autorelease_scope autorelease;

			const bool fast_math = use_fast_math();
			const bool rasterization_enabled = !!state.rasterization_enabled;

			const auto vs_layout = glsl::build_binding_layout(vs_inputs);
			if (!vs.compile(vs_layout, fast_math))
			{
				return {};
			}

			// Metal requires a nil fragment function when rasterization is disabled
			if (rasterization_enabled && !fs.compile(glsl::build_binding_layout(fs_inputs), fast_math))
			{
				return {};
			}

			// The concrete descriptor: a full-state build, or where a specialization takes the colour attachments from.
			// It must be a deterministic function of the shaders and `state`: it is the archive lookup key.
			auto descriptor = make_render_pipeline_descriptor(vs, fs, vs_layout, state, color_attachments_mode::concrete);
			if (!descriptor)
			{
				return {};
			}

			const u64 key = get_full_state_key(vs, fs, state);
			const bool use_archive = uses_pipeline_archive(vs, fs, state);

			// Full state is preferred when the archive holds it: a lookup, and no specialization overhead on the GPU
			const bool archived = use_archive && is_full_state_archived(key);
			bool fallback = full_state_only;

			if (!full_state_only && !archived && is_specializable(state, fs_inputs))
			{
				const unspecialized_key base_key{ vs.uid(), fs.uid(), get_unspecialized_state(state) };

				if (const auto unspecialized = get_unspecialized_pipeline(base_key, vs, fs, vs_layout, state, use_archive))
				{
					if (MTL::RenderPipelineState* pipeline = specialize_render_pipeline(descriptor.get(), unspecialized->pipeline, state))
					{
						return make_specialized_program(pipeline, std::move(descriptor), key, use_archive, vs, fs, state, vs_inputs, fs_inputs);
					}
				}

				// Logged: this key is compiled with full state
				fallback = true;
			}

			NS::Error* error = nullptr;
			MTL::RenderPipelineState* pipeline = mtl::new_render_pipeline_state(descriptor.get(), &error, key, render_pipeline_kind::full_state, use_archive);

			if (!pipeline)
			{
				rsx_log.error("[MSL] Failed to create render pipeline (%u color attachments, %u samples, topology class %u): %s",
					state.color_count, state.sample_count, state.topology_class, mtl::to_string(error));
				rsx_log.error("[MSL] Vertex MSL:\n%s", vs.get_msl());

				if (rasterization_enabled)
				{
					rsx_log.error("[MSL] Fragment MSL:\n%s", fs.get_msl());
				}

				return {};
			}

			g_stats.full_state++;
			g_stats.full_state_archived += archived ? 1 : 0;
			g_stats.fallbacks += fallback ? 1 : 0;

			return make_render_program(pipeline, vs, fs, state, vs_inputs, fs_inputs);
		}

		// For deferred requests: creates the program right away if that is a specialization of an unspecialized pipeline
		// that already exists (no compile, no waiting for another thread). nullptr otherwise; `failed` is set when the
		// specialization was tried and failed (logged), so that the queued job compiles the pipeline with full state.
		std::unique_ptr<glsl::program> try_build_by_specialization(
			const glsl::shader& vs,
			const glsl::shader& fs,
			const glsl::graphics_pipeline_state& state,
			const std::vector<glsl::program_input>& vs_inputs,
			const std::vector<glsl::program_input>& fs_inputs,
			bool& failed)
		{
			failed = false;

			if (!is_specializable(state, fs_inputs))
			{
				return {};
			}

			// A ready entry implies that both shaders are translated: they were before it was built, a translation is never
			// undone while the shader keeps its uid, and the entry's status publishes their MSL and libraries
			const auto unspecialized = g_unspecialized_pipelines.find({ vs.uid(), fs.uid(), get_unspecialized_state(state) });
			if (!unspecialized || unspecialized->status.load() != unspecialized_pipeline::ready)
			{
				return {};
			}

			// The archive holds the full-state pipeline: a lookup on a worker is preferred (no GPU overhead)
			const u64 key = get_full_state_key(vs, fs, state);
			const bool use_archive = uses_pipeline_archive(vs, fs, state);
			if (use_archive && is_full_state_archived(key))
			{
				return {};
			}

			mtl::autorelease_scope autorelease;

			auto descriptor = make_render_pipeline_descriptor(vs, fs, glsl::build_binding_layout(vs_inputs), state, color_attachments_mode::concrete);
			if (!descriptor)
			{
				// The unspecialized pipeline could be built from the same inputs; not expected
				return {};
			}

			MTL::RenderPipelineState* pipeline = specialize_render_pipeline(descriptor.get(), unspecialized->pipeline, state);
			if (!pipeline)
			{
				failed = true;
				return {};
			}

			return make_specialized_program(pipeline, std::move(descriptor), key, use_archive, vs, fs, state, vs_inputs, fs_inputs);
		}
	}

	bool uses_pipeline_archive(const glsl::shader& shader)
	{
		return shader.declared_function_constants().empty();
	}

	void compile_timings::add(const compile_timings& other)
	{
		translated += other.translated;
		translate_us += other.translate_us;
		libraries += other.libraries;
		library_us += other.library_us;
		archive_pipelines += other.archive_pipelines;
		archive_pipeline_us += other.archive_pipeline_us;
		compiled_pipelines += other.compiled_pipelines;
		compiled_pipeline_us += other.compiled_pipeline_us;
		specialized += other.specialized;
		specialization_us += other.specialization_us;
	}

	compile_timings& this_thread_compile_timings()
	{
		return t_compile_timings;
	}

	void record_compile_time(compile_step step, u64 elapsed_us)
	{
		compile_timings& local = t_compile_timings;

		switch (step)
		{
		case compile_step::translate:
			local.translated++;
			local.translate_us += elapsed_us;
			g_window_timings.translated++;
			g_window_timings.translate_us += elapsed_us;
			break;
		case compile_step::library:
			local.libraries++;
			local.library_us += elapsed_us;
			g_window_timings.libraries++;
			g_window_timings.library_us += elapsed_us;
			break;
		case compile_step::archive_pipeline:
			local.archive_pipelines++;
			local.archive_pipeline_us += elapsed_us;
			g_window_timings.archive_pipelines++;
			g_window_timings.archive_pipeline_us += elapsed_us;
			break;
		case compile_step::compiled_pipeline:
			local.compiled_pipelines++;
			local.compiled_pipeline_us += elapsed_us;
			g_window_timings.compiled_pipelines++;
			g_window_timings.compiled_pipeline_us += elapsed_us;
			break;
		case compile_step::specialization:
			// The window counts are in pipeline_creation_stats
			local.specialized++;
			local.specialization_us += elapsed_us;
			break;
		}
	}

	compile_timings get_compile_timings_and_reset()
	{
		compile_timings result;
		result.translated = g_window_timings.translated.exchange(0);
		result.translate_us = g_window_timings.translate_us.exchange(0);
		result.libraries = g_window_timings.libraries.exchange(0);
		result.library_us = g_window_timings.library_us.exchange(0);
		result.archive_pipelines = g_window_timings.archive_pipelines.exchange(0);
		result.archive_pipeline_us = g_window_timings.archive_pipeline_us.exchange(0);
		result.compiled_pipelines = g_window_timings.compiled_pipelines.exchange(0);
		result.compiled_pipeline_us = g_window_timings.compiled_pipeline_us.exchange(0);
		return result;
	}

	std::unique_ptr<glsl::program> build_compute_program(
		glsl::shader& cs,
		const std::vector<glsl::program_input>& cs_inputs)
	{
		ensure(g_render_device);
		mtl::autorelease_scope autorelease;

		const auto layout = glsl::build_binding_layout(cs_inputs);
		if (!cs.compile(layout, use_fast_math()))
		{
			return {};
		}

		auto function = make_function_descriptor(cs);

		auto descriptor = mtl::ref(MTL4::ComputePipelineDescriptor::alloc()->init());
		descriptor->setComputeFunctionDescriptor(function.get());
		descriptor->setLabel(mtl::ns_str(cs.entry_point()));

		// Occupancy hint: every dispatch of a backend kernel uses exactly its GLSL local_size (compute_task and
		// rcas_pass dispatch with it), so the pipeline is compiled for that threadgroup size instead of the device
		// maximum (1024), which lets the compiler give each thread more registers (fewer spills) for the same occupancy.
		const std::array<u32, 3>& workgroup = cs.workgroup_size();
		const u32 threads_per_group = workgroup[0] * workgroup[1] * workgroup[2];
		descriptor->setRequiredThreadsPerThreadgroup(MTL::Size(workgroup[0], workgroup[1], workgroup[2]));
		descriptor->setMaxTotalThreadsPerThreadgroup(threads_per_group);
		// Apple GPUs execute 32-wide SIMD-groups: no partial SIMD-group when the size is a multiple of 32
		descriptor->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(threads_per_group % 32 == 0);

		// Through the pipeline archive (reuses binaries from earlier sessions, records new ones) when it is enabled
		NS::Error* error = nullptr;
		MTL::ComputePipelineState* pipeline = mtl::new_compute_pipeline_state(descriptor.get(), &error,
			mix_function_constants(make_pipeline_key(key_compute, { cs.msl_hash() }, workgroup.data(), sizeof(workgroup)), cs),
			uses_pipeline_archive(cs));

		if (!pipeline)
		{
			rsx_log.error("[MSL] Failed to create compute pipeline: %s", mtl::to_string(error));
			rsx_log.error("[MSL] Compute MSL:\n%s", cs.get_msl());
			return {};
		}

		auto result = std::make_unique<glsl::program>(pipeline, cs_inputs);
		if (cs.needs_buffer_size_buffer())
		{
			result->enable_buffer_size_table(glsl::binding_set_index_compute);
		}

		result->set_storage_writes(glsl::binding_set_index_compute, cs.writes_storage());

		return result;
	}

	std::unique_ptr<glsl::program> build_graphics_program(
		glsl::shader& vs,
		glsl::shader& fs,
		const glsl::graphics_pipeline_state& state,
		const std::vector<glsl::program_input>& vs_inputs,
		const std::vector<glsl::program_input>& fs_inputs)
	{
		return build_render_program(vs, fs, state, vs_inputs, fs_inputs, false);
	}

	MTL::RenderPipelineState* poll_full_state_upgrade(const std::shared_ptr<full_state_upgrade>& upgrade_ptr, bool& finished)
	{
		full_state_upgrade& upgrade = *upgrade_ptr;
		finished = false;

		if (!upgrade.requested)
		{
			if (++upgrade.draws < full_state_draw_threshold)
			{
				return nullptr;
			}

			upgrade.requested = true;

			if (!g_pipe_compilers)
			{
				// The renderer is shutting down: it stays specialized
				finished = true;
				return nullptr;
			}

			g_stats.upgrades_requested++;
			pipe_compiler::s_queue.push(pipe_compiler::pipe_compiler_job(upgrade_ptr));
			return nullptr;
		}

		switch (upgrade.status.load())
		{
		case full_state_upgrade::ready:
			finished = true;
			g_stats.upgrades_swapped++;
			return std::exchange(upgrade.pipeline, nullptr);
		case full_state_upgrade::failed:
			finished = true;
			return nullptr;
		default:
			return nullptr;
		}
	}

	pipeline_creation_stats get_pipeline_creation_stats_and_reset()
	{
		pipeline_creation_stats result;
		result.specialized = g_stats.specialized.exchange(0);
		result.specialization_us = g_stats.specialization_us.exchange(0);
		result.specialization_max_us = g_stats.specialization_max_us.exchange(0);
		result.full_state = g_stats.full_state.exchange(0);
		result.full_state_archived = g_stats.full_state_archived.exchange(0);
		result.fallbacks = g_stats.fallbacks.exchange(0);
		result.unspecialized = g_stats.unspecialized.exchange(0);
		result.unspecialized_us = g_stats.unspecialized_us.exchange(0);
		result.upgrades_requested = g_stats.upgrades_requested.exchange(0);
		result.upgrades_built = g_stats.upgrades_built.exchange(0);
		result.upgrades_swapped = g_stats.upgrades_swapped.exchange(0);
		return result;
	}

	pipe_compiler::pipe_compiler() = default;

	pipe_compiler::~pipe_compiler() = default;

	void pipe_compiler::initialize(const mtl::render_device* pdev)
	{
		m_device = pdev;
	}

	namespace
	{
		enum class worker_priority
		{
			background, // Compiles nobody waits for yet
			urgent,     // The RSX thread waits for it, or the shader interpreter's uber pipeline of a render state
			preload,    // The shader cache preload: pipelines no draw has asked for yet
			upgrade,    // Full-state versions of pipelines that already work (specialized)
		};

		void set_worker_priority(worker_priority priority)
		{
			switch (priority)
			{
			case worker_priority::background:
				thread_ctrl::set_native_priority(-1);
				break;
			case worker_priority::urgent:
				thread_ctrl::set_native_priority(1);
				break;
			case worker_priority::preload:
#ifdef __APPLE__
				// Talk 6: QOS_CLASS_DEFAULT for pipeline prewarming. Below the pipelines draws need (user-initiated) and the
				// emulation threads (user-interactive), so the OS and the Metal compiler service, which inherits the class,
				// serve those first; above utility work.
				pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, 0);
#else
				thread_ctrl::set_native_priority(-1);
#endif
				break;
			case worker_priority::upgrade:
#ifdef __APPLE__
				// Talk guidance for background full-state compiles: below every thread the game needs. QOS_CLASS_BACKGROUND
				// runs on the efficiency cores, and the Metal compiler service inherits it for this build.
				pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
#else
				thread_ctrl::set_native_priority(-1);
#endif
				break;
			}
		}
	}

	void pipe_compiler::operator()()
	{
		// Pipeline builds are background work: run below the emulation threads (named threads start at
		// QOS_CLASS_USER_INTERACTIVE on macOS; -1 is QOS_CLASS_USER_INITIATED, which still gets performance cores). The
		// Metal compiler service inherits the QoS of the calling thread, so this applies to the MTLLibrary and pipeline
		// builds as well. A job the RSX thread is blocked on (prioritize_jobs) runs at emulation priority instead: the
		// RSX thread is idle meanwhile. Not UTILITY for background jobs: their draws are drawn by the shader interpreter
		// or skipped until they finish (and the interpreter's own pipelines are background jobs too), and UTILITY work is
		// preferably run on the efficiency cores. The CPU share of background compiles is bounded by the number of
		// workers that may run them instead (see initialize_pipe_compiler).
		// Full-state upgrades of specialized pipelines are the exception: those pipelines already draw, so they are
		// compiled at background QoS. The shader cache preload runs at default QoS (talk 6, pipeline prewarming), and the
		// shader interpreter's uber pipelines (COMPILE_AHEAD) at emulation priority: until one exists, draws of its render
		// state whose pipelines are compiling are skipped (talk 4: the uber shader first).
		worker_priority priority = worker_priority::background;
		set_worker_priority(priority);

		while (thread_ctrl::state() != thread_state::aborting)
		{
			// Sampled before the queue is checked: a job pushed after the check changes it and ends the wait
			const u32 pushed = s_queue.pushed.load();

			auto job = s_queue.pop();
			if (!job)
			{
				thread_ctrl::wait_on(s_queue.pushed, pushed);
				continue;
			}

			const bool is_upgrade = job->kind == job_kind::full_state_upgrade;
			const worker_priority job_priority =
				is_upgrade ? worker_priority::upgrade :
				job->kind == job_kind::preload ? worker_priority::preload :
				(job->urgent || job->ahead) ? worker_priority::urgent :
				worker_priority::background;

			if (job_priority != priority)
			{
				priority = job_priority;
				set_worker_priority(priority);
			}

			// The job's own build times (this_thread_compile_timings)
			t_compile_timings = {};

			{
				// Every job gets its own pool: Metal descriptors, errors and strings are autoreleased
				mtl::autorelease_scope autorelease;

				if (job->kind == job_kind::preload)
				{
					job->task();
					job->task = {};
				}
				else if (is_upgrade)
				{
					// Not when the program is gone (the job holds the last reference)
					if (job->upgrade.use_count() > 1)
					{
						int_build_full_state_upgrade(*job->upgrade);
					}

					job->upgrade.reset();
				}
				else
				{
					std::unique_ptr<glsl::program> compiled;
					if (job->kind == job_kind::graphics)
					{
						compiled = int_compile_graphics_pipe(job->graphics_data, job->shaders[0], job->shaders[1], job->inputs[0], job->inputs[1], job->full_state);
					}
					else
					{
						compiled = int_compile_compute_pipe(job->shaders[0], job->inputs[0], job->flags);
					}

					if (job->callback_func)
					{
						job->callback_func(compiled);
					}
				}
			}

			if (!is_upgrade)
			{
				// After the callback: the pipeline (or the failure) is visible to whoever waits for this count to change.
				// Upgrades are never waited for (the program already has a pipeline).
				s_queue.completed++;
				s_queue.completed.notify_all();
			}

			if (!job->urgent)
			{
				s_queue.on_background_job_done(is_upgrade);
			}
		}
	}

	void pipe_compiler::int_build_full_state_upgrade(full_state_upgrade& upgrade)
	{
		// Through the pipeline archive: recorded, and listed in the sidecar, so later sessions build it directly (unless its
		// shaders do not use the archive)
		NS::Error* error = nullptr;
		MTL::RenderPipelineState* pipeline = mtl::new_render_pipeline_state(upgrade.descriptor.get(), &error, upgrade.key,
			render_pipeline_kind::full_state, upgrade.use_archive);

		if (pipeline)
		{
			upgrade.pipeline = pipeline;
			g_stats.upgrades_built++;
		}
		else if (should_log_flexible_failure())
		{
			rsx_log.error("Metal: the background full-state compile of a specialized render pipeline failed (%s); it stays specialized",
				mtl::to_string(error));
		}

		// Not needed any more (the binding thread only reads `pipeline`)
		upgrade.descriptor.reset();

		// Publishes `pipeline`
		upgrade.status = pipeline ? full_state_upgrade::ready : full_state_upgrade::failed;
	}

	u32 pipe_compiler::get_completed_job_count()
	{
		return s_queue.completed.load();
	}

	void pipe_compiler::queue_preload_tasks(std::vector<std::function<void()>> tasks)
	{
		{
			std::lock_guard lock(s_queue.mutex);
			for (auto& task : tasks)
			{
				s_queue.preload.emplace_back(std::move(task));
			}
		}

		s_queue.wake_workers();
	}

	void pipe_compiler::on_pipeline_archive_ready()
	{
		// Idle workers may take preload tasks now
		s_queue.wake_workers();
	}

	void pipe_compiler::wait_for_completed_job(u32 count, u64 timeout_us)
	{
		if (!timeout_us)
		{
			return;
		}

		// Plain atomic wait (not thread_ctrl::wait_on): the caller is the RSX thread, whose thread notifications and task
		// queue would end the wait early; it is bounded by the timeout anyway.
		s_queue.completed.wait(count, atomic_wait_timeout{ timeout_us * 1000 });
	}

	void pipe_compiler::prioritize_jobs(const glsl::shader* vs, const glsl::shader* fs)
	{
		bool found = false;
		{
			std::lock_guard lock(s_queue.mutex);

			// Stable, so the queue order is kept among the moved jobs and among the others
			const auto first_other = std::stable_partition(s_queue.jobs.begin(), s_queue.jobs.end(), [&](const pipe_compiler_job& job)
			{
				return job.kind == job_kind::graphics && job.shaders[0] == vs && job.shaders[1] == fs;
			});

			for (auto it = s_queue.jobs.begin(); it != first_other; ++it)
			{
				it->urgent = true;
				found = true;
			}
		}

		if (found)
		{
			// The worker kept free for urgent jobs may be idle while every background slot is taken
			s_queue.wake_workers();
		}
	}

	std::unique_ptr<glsl::program> pipe_compiler::int_compile_compute_pipe(
		glsl::shader* cs,
		const std::vector<glsl::program_input>& cs_inputs,
		op_flags /*flags*/)
	{
		ensure(cs);
		return build_compute_program(*cs, cs_inputs);
	}

	std::unique_ptr<glsl::program> pipe_compiler::int_compile_graphics_pipe(
		const mtl::pipeline_props& create_info,
		glsl::shader* vs,
		glsl::shader* fs,
		const std::vector<glsl::program_input>& vs_inputs,
		const std::vector<glsl::program_input>& fs_inputs,
		bool full_state)
	{
		ensure(vs && fs);
		return build_render_program(*vs, *fs, create_info.state, vs_inputs, fs_inputs, full_state);
	}

	std::unique_ptr<glsl::program> pipe_compiler::compile(
		glsl::shader* cs,
		op_flags flags, callback_t callback,
		const std::vector<glsl::program_input>& cs_inputs)
	{
		if (flags & COMPILE_INLINE)
		{
			return int_compile_compute_pipe(cs, cs_inputs, flags);
		}

		s_queue.push(pipe_compiler_job(cs, cs_inputs, flags, std::move(callback)));
		return {};
	}

	std::unique_ptr<glsl::program> pipe_compiler::compile(
		const mtl::pipeline_props& create_info,
		glsl::shader* vs,
		glsl::shader* fs,
		op_flags flags, callback_t callback,
		const std::vector<glsl::program_input>& vs_inputs,
		const std::vector<glsl::program_input>& fs_inputs)
	{
		ensure(vs && fs);

		if (flags & USE_LAST_PROVOKING_VERTEX)
		{
			// Metal has no provoking-vertex control. The renderer is expected to report
			// supports_last_provoking_vertex = false so this never happens; keep going with the default convention.
			static atomic_t<bool> s_reported = false;
			if (!s_reported.exchange(true))
			{
				rsx_log.warning("[MSL] Last provoking vertex requested but unsupported by Metal; flat shading may differ.");
			}
		}

		if (flags & COMPILE_INLINE)
		{
			return int_compile_graphics_pipe(create_info, vs, fs, vs_inputs, fs_inputs, false);
		}

		// A new colour attachment configuration of shaders whose unspecialized pipeline exists: no compile, so no reason
		// to make the caller wait for a worker (or skip the draw)
		bool specialization_failed = false;
		if (auto program = try_build_by_specialization(*vs, *fs, create_info.state, vs_inputs, fs_inputs, specialization_failed))
		{
			return program;
		}

		// After a failed specialization the worker builds the pipeline with full state
		s_queue.push(pipe_compiler_job(create_info, vs, fs, vs_inputs, fs_inputs, flags, std::move(callback), specialization_failed));
		return {};
	}

	void initialize_pipe_compiler(int num_worker_threads)
	{
		ensure(g_render_device); // "Cannot initialize pipe compiler before creating a logical device"

		// Library/pipeline builds dominate each job and Metal only runs max_compile_tasks of them concurrently
		const u32 compile_tasks = std::max(1u, g_render_device->caps().max_compile_tasks);
		const int max_workers = static_cast<int>(compile_tasks);

		if (num_worker_threads <= 0)
		{
			// Every job is CPU work: GLSL -> SPIR-V -> MSL in the worker, then the MTLLibrary and pipeline builds in the
			// Metal compiler service, which runs them at the worker's QoS and, with setShouldMaximizeConcurrentCompilation
			// (render_device::create), as many at once as the host has cores (maximumConcurrentCompilationTaskCount). QoS is
			// not a strict priority on macOS: the scheduler time-shares cores between classes, and busy PPU/SPU threads
			// decay in priority. One worker per compilation task (9 on a 10-core M1 Max) therefore takes every core the
			// guest needs during a compile burst, which shows as frames held for several refreshes while the burst lasts.
			// Background compiles get the same share of the host as the VK backend's default worker count
			// (VKPipelineCompiler.cpp), plus one worker that only takes jobs the RSX thread is waiting for
			// (prioritize_jobs), so such a job never queues behind a burst of background compiles.
			const u32 hw_threads = utils::get_thread_count();
			const u32 background_workers =
				hw_threads >= 24 ? 12 :
				hw_threads >= 16 ? 8 :
				hw_threads > 12 ? 6 :
				hw_threads > 8 ? 4 :
				hw_threads == 8 ? 2 : 1;

			num_worker_threads = static_cast<int>(background_workers + 1);

			rsx_log.notice("Async pipeline compiler auto-selected %d worker(s): %u for background compiles, 1 kept for pipelines the renderer waits for "
				"(Metal concurrent compilation tasks: %u, host threads: %u).",
				num_worker_threads, background_workers, compile_tasks, hw_threads);
		}

		if (num_worker_threads > max_workers)
		{
			rsx_log.notice("Pipeline compiler worker count capped from %d to %d (Metal concurrent compilation limit).",
				num_worker_threads, max_workers);
			num_worker_threads = max_workers;
		}

		ensure(num_worker_threads >= 1);

		rsx_log.notice("Metal: shaders are MSL 3.2 built with %s; compute pipelines declare their threadgroup size.",
			use_fast_math() ? "relaxed math (fast functions, IEEE Inf/NaN kept)" : "safe math (precise functions; MSL fast math disabled)");

		rsx_log.notice("Metal: render pipelines are flexible: one unspecialized pipeline per shader pair and fixed state, specialized for each "
			"colour attachment configuration (formats, write masks, blending); pipelines drawn %u times are rebuilt with full state in the "
			"background; full state directly when the pipeline archive holds it or for framebuffer-fetch shaders.", full_state_draw_threshold);

		{
			// With more than one worker, one of them is kept for urgent jobs (see pipe_compiler::job_queue::pop)
			std::lock_guard lock(pipe_compiler::s_queue.mutex);
			pipe_compiler::s_queue.background_limit = num_worker_threads > 1 ? static_cast<u32>(num_worker_threads - 1) : 1u;
			pipe_compiler::s_queue.running_background = 0;
			pipe_compiler::s_queue.running_upgrades = 0;
		}

		g_unspecialized_pipelines.clear();

		// Create the thread pool
		g_pipe_compilers = std::make_unique<named_thread_group<pipe_compiler>>("RSX.W", num_worker_threads);

		// Initialize the workers. At least one inline compiler shall exist (doesn't actually run)
		for (pipe_compiler& compiler : *g_pipe_compilers.get())
		{
			compiler.initialize(g_render_device);
		}
	}

	void destroy_pipe_compiler()
	{
		g_pipe_compilers.reset();

		{
			// No worker is left: drop what was never started (the callbacks reference the program cache, which goes next)
			std::lock_guard lock(pipe_compiler::s_queue.mutex);
			pipe_compiler::s_queue.jobs.clear();
			pipe_compiler::s_queue.preload.clear();
			pipe_compiler::s_queue.upgrades.clear();
			pipe_compiler::s_queue.running_background = 0;
			pipe_compiler::s_queue.running_upgrades = 0;
		}

		// Nothing builds pipelines any more. Specialized pipelines do not need their unspecialized pipeline.
		g_unspecialized_pipelines.clear();
	}

	pipe_compiler* get_pipe_compiler()
	{
		ensure(g_pipe_compilers);

		// Deferred jobs go to the shared queue, so any worker object will do
		return g_pipe_compilers.get()->begin();
	}
}

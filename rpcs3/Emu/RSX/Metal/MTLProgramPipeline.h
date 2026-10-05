#pragma once

// CONTRACT HEADER — shared by the shader/pipeline, compute/overlay and renderer-core components.
// Public declarations here must not change without updating every user.
//
// Shader path: RSX ucode -> (shared decompilers) Vulkan-flavoured GLSL 450 -> glslang SPIR-V -> SPIRV-Cross MSL
//              -> MTL4Compiler library -> MTL4 render/compute pipeline.
// Binding model: every stage owns an MTL4ArgumentTable (see mtl::command_list). Each GLSL (set, binding) is mapped to
// Metal buffer/texture/sampler indices by build_binding_layout(), which is used BOTH when translating to MSL
// (SPIRV-Cross resource bindings) and when binding resources at draw time, so the two can never disagree.

#include "mtlutils/mtl_api.h"
#include "mtlutils/commands.h"
#include "mtlutils/buffer_object.h"
#include "mtlutils/image.h"
#include "Emu/RSX/Program/GLSLTypes.h"
#include "Emu/RSX/Common/simple_array.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mtl
{
	class data_heap;
	struct sampler;
	struct full_state_upgrade; // MTLPipelineCompiler.cpp

	namespace glsl
	{
		enum program_input_type : u32
		{
			input_type_uniform_buffer = 0,
			input_type_texel_buffer,
			input_type_texture,
			input_type_storage_buffer,
			input_type_storage_texture,
			input_type_push_constant,
			input_type_attachment,        // subpassInput -> [[color(n)]] framebuffer fetch (no binding)
			input_type_separate_image,    // textureXD / utextureXD (sampled with a separate sampler): texture slot(s) only
			input_type_sampler,           // sampler / samplerShadow (separate sampler): sampler slot(s) only

			input_type_max_enum,
			input_type_undefined = 0xffff'ffff
		};

		struct push_constant_ref
		{
			u32 offset = 0;
			u32 size = 0;
		};

		// Same shape as vk::glsl::program_input so decompiler code ports 1:1.
		struct program_input
		{
			::glsl::program_domain domain = ::glsl::glsl_invalid_program;
			program_input_type type = input_type_undefined;
			push_constant_ref push_constant{};
			u32 set = 0;
			u32 location = umax;       // GLSL binding (or input attachment index for input_type_attachment)
			u32 array_size = 1;        // > 1 for sampler arrays (interpreter)
			std::string name = "undefined";

			static program_input make(
				::glsl::program_domain domain,
				const std::string& name,
				program_input_type type,
				u32 set,
				u32 location,
				const push_constant_ref& push_constant = {},
				u32 array_size = 1)
			{
				return program_input
				{
					.domain = domain,
					.type = type,
					.push_constant = push_constant,
					.set = set,
					.location = location,
					.array_size = array_size,
					.name = name
				};
			}
		};

		enum binding_set_index : u32
		{
			binding_set_index_vertex = 0,
			binding_set_index_fragment = 1,

			binding_set_index_compute = 0,
			binding_set_index_unified = 0,

			binding_set_index_max_enum = 2,
		};

		// Metal slot assigned to a GLSL (set, binding). umax = unused.
		struct resource_slot
		{
			program_input_type type = input_type_undefined;
			u32 buffer_index = umax;   // uniform/storage buffers and push constants
			u32 texture_index = umax;  // textures, texel buffers, storage textures (first element for arrays)
			u32 sampler_index = umax;  // samplers for combined image samplers (first element for arrays)
			u32 array_size = 1;
		};

		// Per-stage binding layout. Deterministic function of the input list.
		struct binding_layout
		{
			std::unordered_map<u32, resource_slot> slots; // key: GLSL binding location within this set
			u32 push_constant_buffer_index = umax;         // Buffer index used for the push constant block
			u32 push_constant_size = 0;
			u32 buffer_count = 0;
			u32 texture_count = 0;
			u32 sampler_count = 0;
		};

		// Assigns Metal indices for ONE stage (all inputs of the list, whatever their set):
		//  - buffers (UBO, SSBO) 0.. in input order, push-constant block last (index 30 max);
		//  - sampled textures (input_type_texture) first 0.., then texel buffers, storage textures and separate images
		//    (63 max);
		//  - sampler index == texture index for the first 16 sampled texture slots, in input order. Later sampled
		//    textures get sampler_index = umax and the MSL gives them a constant nearest / clamp-to-border(black) /
		//    LOD 0 sampler (exactly the stencil-mirror sampler), so list textures that need real samplers first;
		//  - separate samplers (input_type_sampler) take the sampler slots left after that, in input order;
		//  - input_type_attachment gets no slot (framebuffer fetch [[color(n)]]).
		// Keys are GLSL binding locations; a location may only be declared once per stage (fatal: programming error).
		// Running out of buffer/texture slots is not fatal: the slot stays umax and the shader translation fails.
		// If a stage reads SSBO lengths, its buffer-size table lives at index buffer_count (see shader/program).
		binding_layout build_binding_layout(const std::vector<program_input>& inputs);

	// Creates the bind-time sampler fallbacks (default + null samplers) once the render
	// device exists. Call once during renderer init: the first draw then never creates
	// them on the RSX thread, and a live fallback is guaranteed until teardown.
	void precache_fallback_samplers();

	// Always-on record of the most recent sampler writes that reached the driver
	// (program::bind() -> setSamplerState). Capture is unconditional — deliberately
	// NOT gated on RPCS3_METAL_SAMPLER_TRACE — because a driver fault needs the
	// pre-crash writes without a foresight-enabled repro run. Recording costs one
	// relaxed atomic increment plus a few stores, and only on actual table writes
	// (the bind shadow already skips redundant ones), so the steady-state cost on
	// the RSX thread is ~zero. No logging happens at record time: the ring is
	// dumped into the crash report by sampler_write_section() when a fatal or
	// native crash fires. Bounded (128 entries, oldest overwritten); a torn entry
	// is possible if the crash lands mid-record, which forensics must tolerate.
	struct sampler_write_entry
	{
		u64 prog_uid = 0;
		u64 sampler_id = 0;
		u32 stage = 0;
		u32 slot = 0;
		u32 live = 0;
	};

	class sampler_write_ring
	{
	public:
		static constexpr u32 capacity = 128;

		void record(u64 prog_uid, u32 stage, u32 slot, u64 sampler_id, bool live);
		bool empty() const;
		// Oldest first, at most max_lines, one "prog=.. stage=.. slot=.. id=.. live=.." line each.
		void format_last(std::string& out, u32 max_lines) const;

	private:
		sampler_write_entry m_entries[capacity]{};
		std::atomic<u32> m_total{0};
	};

	sampler_write_ring& global_sampler_write_ring();

	// Crash-section provider (matches thread_ctrl::fatal_context_provider): the
	// recent sampler writes for the crash report. Empty when nothing was recorded
	// so Metal-free sessions stay clean.
	std::string sampler_write_section();

		// Resources handed to program::bind_uniform
		struct buffer_binding_info
		{
			const mtl::buffer* buffer = nullptr;   // or raw:
			MTL::Buffer* raw_buffer = nullptr;
			u64 offset = 0;
			u64 range = 0;

			buffer_binding_info() = default;
			buffer_binding_info(const mtl::buffer* buf, u64 off, u64 len) : buffer(buf), offset(off), range(len) {}
			buffer_binding_info(MTL::Buffer* buf, u64 off, u64 len) : raw_buffer(buf), offset(off), range(len) {}

			MTL::GPUAddress gpu_address() const;
			// The bound range, as read by the shader (hazard tracking; bind() makes storage buffers writes if needed)
			gpu_access access() const;
		};

		struct image_binding_info
		{
			MTL::Texture* texture = nullptr;          // A view (mtl::image_view::value) or full texture
			MTL::SamplerState* sampler = nullptr;     // nullptr for storage images / texelFetch-only textures

			// What bind_uniform writes. Taken from the IDs mtl::image_view / mtl::sampler cache at creation, so binding
			// wrapped objects sends no Objective-C message; raw objects are queried here.
			MTL::ResourceID texture_id{};
			MTL::ResourceID sampler_id{};

			// The subresources the shader can access (hazard tracking). Views of mtl::image carry them; a raw texture is
			// resolved with Metal queries (whole texture).
			gpu_access access{};

			image_binding_info() = default;
			image_binding_info(MTL::Texture* tex, MTL::SamplerState* smp)
				: texture(tex), sampler(smp)
				, texture_id(tex ? tex->gpuResourceID() : MTL::ResourceID{})
				, sampler_id(smp ? smp->gpuResourceID() : MTL::ResourceID{})
				, access(read_texture(tex))
			{}
			image_binding_info(const mtl::image_view* view, const mtl::sampler* smp);
		};

		// Value of a function constant for the pipelines built from a shader. `id` is the GLSL/SPIR-V specialization
		// constant id (layout(constant_id = N)), which the MSL translation declares as [[function_constant(N)]].
		struct function_constant
		{
			u32 id = 0;
			u32 value = 0; // Bits of the value: 0/1 for bool, the integer, or the float's bits (see the declared type)
		};

		// A translated shader stage. GLSL source is kept for the shader cache / debugging.
		class shader
		{
			::glsl::program_domain m_type = ::glsl::program_domain::glsl_vertex_program;
			std::string m_source;       // GLSL (Vulkan semantics)
			std::string m_msl;          // Generated MSL
			std::string m_entry_point;  // MSL entry point name
			MTL::Library* m_library = nullptr;
			u64 m_uid = 0;              // Unique for the process lifetime, assigned by create()
			u64 m_msl_hash = 0;         // FNV-1a of m_msl (pipeline keys), set with the library

			std::mutex m_compile_lock;          // compile() may race between pipe-compiler workers sharing a shader
			bool m_compile_failed = false;      // Do not retry (and re-log) a translation that already failed
			bool m_needs_buffer_sizes = false;  // MSL reads spvBufferSizeConstants (GLSL SSBO .length())
			bool m_writes_storage = true;       // A storage buffer or image is not NonWritable (GLSL readonly)
			std::vector<std::pair<u32, MTL::VertexFormat>> m_vertex_attributes; // Vertex stage inputs (location, format)
			std::array<u32, 3> m_workgroup_size{ 1, 1, 1 }; // Compute stage: GLSL local_size
			std::vector<std::pair<u32, MTL::DataType>> m_declared_constants; // Function constants of the MSL (id, type)

			// Specialization (create_specialization): the shader whose library this one uses, and the constant values
			shader* m_base = nullptr;
			std::vector<function_constant> m_function_constants;

		public:
			shader() = default;
			~shader();

			shader(const shader&) = delete;
			shader& operator=(const shader&) = delete;

			void create(::glsl::program_domain domain, const std::string& source);

			// Makes this shader a specialization of `base`, which must outlive it: it has no source or library of its
			// own. Pipelines built from it use the library of `base` (translated and compiled once, by the first pipeline
			// that needs it) with its function constants set to `constants` (MTL4::SpecializedFunctionDescriptor), so the
			// Metal compiler folds them and drops the code they disable. Constants the MSL does not declare are ignored;
			// declared ones without a value keep the default of their GLSL declaration.
			void create_specialization(shader& base, std::vector<function_constant> constants);

			// GLSL -> SPIR-V -> MSL using the given layout, then MTL4Compiler::newLibrary. Thread-safe.
			// Returns false (and logs the MSL/GLSL) on failure. A specialization compiles its base.
			bool compile(const binding_layout& layout, bool fast_math);

			void destroy();

			::glsl::program_domain domain() const { return m_type; }
			const std::string& get_source() const { return m_base ? m_base->get_source() : m_source; }
			const std::string& get_msl() const { return m_base ? m_base->get_msl() : m_msl; }
			const std::string& entry_point() const { return m_base ? m_base->entry_point() : m_entry_point; }
			MTL::Library* library() const { return m_base ? m_base->library() : m_library; }
			bool is_compiled() const { return library() != nullptr; }

			// Values the pipelines built from this shader set (empty unless this is a specialization)
			const std::vector<function_constant>& function_constants() const { return m_function_constants; }

			// Function constants the translated MSL declares, (id, type), reflected at translation time
			const std::vector<std::pair<u32, MTL::DataType>>& declared_function_constants() const { return m_base ? m_base->declared_function_constants() : m_declared_constants; }

			// Identifies this shader (its source) for the process lifetime; never reused, unlike the object address.
			// Keys the unspecialized render pipelines (MTLPipelineCompiler.cpp).
			u64 uid() const { return m_uid; }

			// Hash of the translated MSL (0 until compiled): pipeline archive keys
			// A specialization has no MSL of its own: its base shader's (pipeline keys add the constant values)
			u64 msl_hash() const { return m_base ? m_base->msl_hash() : m_msl_hash; }

			// True if the MSL expects a buffer-size table (uint per Metal buffer index) at binding_layout::buffer_count.
			// Only set for shaders using GLSL SSBO .length(); pipeline builders forward it to program.
			bool needs_buffer_size_buffer() const { return m_base ? m_base->needs_buffer_size_buffer() : m_needs_buffer_sizes; }

			// True unless every storage buffer and storage image of the shader is read-only (GLSL readonly): whether the
			// stage's storage bindings are declared as writes (hazard tracking). Pipeline builders forward it to program.
			bool writes_storage() const { return m_base ? m_base->writes_storage() : m_writes_storage; }

			// Vertex shaders only: `layout(location = N) in` attributes (reflected at translation time, sorted by
			// location). Empty for RSX vertex programs, which pull their inputs from texel buffers.
			const std::vector<std::pair<u32, MTL::VertexFormat>>& vertex_attributes() const { return m_base ? m_base->vertex_attributes() : m_vertex_attributes; }

			// Compute shaders only: the GLSL local_size (reflected at translation time). Every dispatch of the kernel uses
			// exactly this many threads per threadgroup; the pipeline declares it as its required threadgroup size.
			const std::array<u32, 3>& workgroup_size() const { return m_base ? m_base->workgroup_size() : m_workgroup_size; }
		};

		// Fixed-function state baked into a Metal 4 render pipeline. POD; hashed and serialized raw (shader cache).
		// Flexible render pipeline states (MTLPipelineCompiler.cpp): the colour attachment configuration (color[],
		// color_count) is what a specialization sets; every other field is compiled into the unspecialized pipeline
		// and keys it, so a field added here is part of the full-compile key automatically.
		struct color_attachment_state
		{
			u32 pixel_format = 0;       // MTL::PixelFormat (0 = unused)
			u8 write_mask = 0xF;        // MTL::ColorWriteMask bits (R=8,G=4,B=2,A=1)
			u8 blend_enable = 0;
			u8 src_rgb = 1;             // MTL::BlendFactor
			u8 dst_rgb = 0;
			u8 op_rgb = 0;              // MTL::BlendOperation
			u8 src_a = 1;
			u8 dst_a = 0;
			u8 op_a = 0;
		};

		struct graphics_pipeline_state
		{
			std::array<color_attachment_state, 4> color{};
			u32 color_count = 0;
			u32 depth_stencil_format = 0; // informational only (MTL4 pipelines do not bake depth/stencil formats)
			u8 sample_count = 1;
			u8 alpha_to_coverage = 0;
			u8 alpha_to_one = 0;
			u8 topology_class = 3;        // MTL::PrimitiveTopologyClass (1=point, 2=line, 3=triangle)
			u8 rasterization_enabled = 1;
			u8 pad[3]{};
		};

		static_assert(std::is_trivially_copyable_v<graphics_pipeline_state>);

		// A linked pipeline + its binding layouts + current bindings.
		class program
		{
			MTL::RenderPipelineState* m_render_pipeline = nullptr;
			MTL::ComputePipelineState* m_compute_pipeline = nullptr;

			std::array<std::vector<program_input>, binding_set_index_max_enum> m_inputs;
			std::array<binding_layout, binding_set_index_max_enum> m_layouts;
			std::array<std::vector<resource_slot>, binding_set_index_max_enum> m_table_slots; // m_layouts[].slots, flat for bind()

			// Unique for the process lifetime (unlike the pipeline's address): identifies the pipeline state set on a
			// render encoder, see command_list::render_encoder_bindings
			u64 m_uid = 0;

			struct stage_bindings
			{
				std::array<MTL::GPUAddress, 31> buffers{};
				std::array<MTL::ResourceID, 64> textures{};
				std::array<MTL::ResourceID, 16> samplers{};
				std::vector<u8> push_constants;
				bool dirty = true;
				// Scratch addresses of our last uploads (0 = none): lets write_table skip re-uploading
				// when the shared table still holds our bytes (see the shadow peek below)
				MTL::GPUAddress last_push_constants_addr = 0;
				MTL::GPUAddress last_buffer_sizes_addr = 0;

				std::array<u32, 31> buffer_sizes{};     // Bound ranges, uploaded when the stage needs a buffer-size table
				bool needs_buffer_sizes = false;

				// What the bound resources are (hazard tracking): declared by bind() for every slot the stage uses
				std::array<gpu_access, 31> buffer_access{};
				std::array<gpu_access, 64> texture_access{};
			};
			std::array<stage_bindings, binding_set_index_max_enum> m_bindings;

			// The shader of the stage may write storage buffers / images (reflected at translation, see
			// set_storage_writes()). Until told otherwise every storage binding counts as written.
			std::array<bool, binding_set_index_max_enum> m_stage_writes_storage{ true, true };

			u32 m_compute_threads_per_group = 1;

			// Set while the render pipeline is a specialization: bind() counts the draws, has the full-state pipeline
			// built in the background once they are many and swaps it in (MTLPipelineCompiler.h)
			std::shared_ptr<mtl::full_state_upgrade> m_full_state_upgrade;

			// (set, binding) -> bitmask of stages (m_inputs/m_layouts index) that declare it
			static constexpr u32 max_binding_locations = 128;
			std::array<std::array<u8, max_binding_locations>, binding_set_index_max_enum> m_binding_stage_mask{};
			// Per stage: index + 1 into m_table_slots of the slot at a GLSL binding location (0: none). bind_uniform*()
			// resolves ~12-20 locations per draw; this replaces the hash map lookups of m_layouts[].slots.
			std::array<std::array<u16, max_binding_locations>, binding_set_index_max_enum> m_slot_by_location{};
			bool m_missing_binding_reported = false;
			// Sampler slots already reported by report_dead_sampler (one bit per slot, per stage)
			std::array<u32, binding_set_index_max_enum> m_dead_sampler_reported{};

			void init_layouts();
			template <typename F> void for_each_bound_slot(u32 set_id, u32 binding_point, F&& func);
			void update_full_state_upgrade();
			void report_dead_sampler(u32 stage, u32 sampler_slot, u64 bad_id, u32 texture_slot, u32 array_size);

		public:
			program(MTL::RenderPipelineState* pipeline,
				const std::vector<program_input>& vertex_inputs,
				const std::vector<program_input>& fragment_inputs);

			program(MTL::ComputePipelineState* pipeline,
				const std::vector<program_input>& compute_inputs);

			program(const program&) = delete;
			program(program&&) = delete;
			~program();

			bool is_compute() const { return m_compute_pipeline != nullptr; }
			MTL::RenderPipelineState* render_pipeline() const { return m_render_pipeline; }
			MTL::ComputePipelineState* compute_pipeline() const { return m_compute_pipeline; }
			u32 max_total_threads_per_threadgroup() const;

			bool has_uniform(program_input_type type, std::string_view uniform_name) const;
			std::pair<u32, u32> get_uniform_location(::glsl::program_domain domain, program_input_type type, std::string_view uniform_name) const; // {set, location}

			void bind_uniform(const buffer_binding_info& buffer, u32 set_id, u32 binding_point);     // UBO / SSBO
			void bind_uniform(const image_binding_info& image, u32 set_id, u32 binding_point);       // sampled / storage image
			void bind_uniform(const mtl::buffer_view* view, u32 set_id, u32 binding_point);         // texel buffer
			// Arrays of combined image samplers, separate images (texture of each element) or samplers (sampler of each element)
			void bind_uniform_array(std::span<const image_binding_info> images, u32 set_id, u32 binding_point);
			// Vulkan semantics: [offset, offset + size) addresses the push-constant space shared by all stages; every
			// stage whose push block covers it receives the bytes (set_id is accepted for parity, not needed).
			void push_constants(u32 set_id, u32 offset, u32 size, const void* data);

			// Sets the pipeline state on the active encoder of `cmd` (render encoder for graphics programs, the compute
			// encoder for compute programs), uploads push constants to `scratch` and writes + sets the stage argument tables.
			// Declares what the bound resources give the shaders access to (hazard tracking, see mtl::command_list):
			//  - compute programs: bind() begins the dispatch command (cmd.dispatch()), which orders it after the earlier
			//    work it conflicts with; record the dispatch itself on cmd.compute_encoder();
			//  - graphics programs (a render pass must already be open): every slot is declared for the draws of the pass
			//    (cmd.table_access()) before the pipeline state is set. The pass barriers order the draws after earlier
			//    work; what they leave out gets a barrier before the pass's first draw, or cmd.pass_split_required() asks
			//    the renderer to bind again in a new pass.
			// Uniform buffers, sampled textures and texel buffers are reads; storage buffers and storage textures are
			// writes too unless the stage's shader cannot write them (set_storage_writes()). The push constant block and
			// the buffer-size table live in scratch-heap blocks the CPU writes for this bind only: nothing to declare.
			void bind(mtl::command_list& cmd, mtl::data_heap& scratch);

			// For pipeline builders: the translated shader of `stage_index` (0 = vertex/compute, 1 = fragment) reads
			// SSBO lengths, so bind() uploads the bound storage-buffer ranges to binding_layout::buffer_count.
			void enable_buffer_size_table(u32 stage_index);

			// For pipeline builders: whether the translated shader of `stage_index` may write any storage buffer or image
			// (shader::writes_storage()). A stage that cannot has its storage bindings declared as reads.
			void set_storage_writes(u32 stage_index, bool writes);

			// For pipeline builders: the render pipeline was created by specialization. Every bind() is counted as a
			// draw; a heavily drawn program gets its full-state pipeline compiled in the background, and the bind()
			// after it is ready replaces the specialized pipeline with it (the old one goes to the GC).
			void set_full_state_upgrade(std::shared_ptr<mtl::full_state_upgrade> upgrade);
		};

		// ---- Program creation helpers (for static passes: compute kernels, overlays, blits) ----------------------
		// Synchronous build from GLSL (Vulkan semantics). Returns nullptr and logs on failure.
		std::unique_ptr<program> create_compute_program(const std::string& compute_glsl, const std::vector<program_input>& inputs);

		std::unique_ptr<program> create_graphics_program(
			const std::string& vertex_glsl, const std::vector<program_input>& vertex_inputs,
			const std::string& fragment_glsl, const std::vector<program_input>& fragment_inputs,
			const graphics_pipeline_state& state);
	}

	// Key for RSX graphics pipelines (program_state_cache<MTLTraits>::pipeline_properties). POD, hashed with
	// rpcs3::hash_struct and stored raw in the shader cache.
	struct pipeline_props
	{
		glsl::graphics_pipeline_state state{};

		bool operator==(const pipeline_props& other) const
		{
			return std::memcmp(&state, &other.state, sizeof(state)) == 0;
		}
	};

	static_assert(std::is_trivially_copyable_v<pipeline_props>);

	// Where the time of shader and pipeline builds goes (telemetry, MTLPipelineCompiler.h: compile_timings). Called by
	// the step that took `elapsed_us`, on the thread that did it.
	enum class compile_step : u8
	{
		translate,         // GLSL -> SPIR-V -> MSL (glslang, SPIRV-Cross): CPU work of the building thread
		library,           // MTL4Compiler::newLibrary from MSL
		archive_pipeline,  // A pipeline built with the pipeline archive's lookups (a binary found there, or compiled on a miss)
		compiled_pipeline, // A pipeline compiled without lookups (no archive, lookups not ready yet, or not archived)
		specialization,    // A render pipeline created by specializing an unspecialized one
	};

	void record_compile_time(compile_step step, u64 elapsed_us);

	// Global scratch ring for push constants and small per-draw uniforms (created by MTLGSRender, 16 MiB, grows).
	mtl::data_heap& get_scratch_heap();
}

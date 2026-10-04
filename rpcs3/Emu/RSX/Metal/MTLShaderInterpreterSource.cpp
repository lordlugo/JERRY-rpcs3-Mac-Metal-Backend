#include "stdafx.h"
#include "MTLShaderInterpreter.h"
#include "MTLVertexProgram.h"
#include "MTLFragmentProgram.h"
#include "MTLHelpers.h"

#include "Emu/RSX/Program/GLSLCommon.h"
#include "Emu/RSX/Program/ShaderInterpreter.h"

// GLSL of the Metal shader interpreter libraries (see MTLShaderInterpreter.h). The shared interpreters
// (Program/GLSLInterpreter) run inside a Metal wrapper that supplies what the recompiled programs of this backend do:
// the same resource declarations (MTLVertexDecompilerThread / MTLFragmentDecompilerThread), function constants
// instead of the VK backend's #defines, the texture sampling of RSXFragmentTextureOps.glsl for a dynamic texture unit,
// and the ROP stage of RSXROPEpilogue.glsl.

namespace mtl::interpreter
{
	namespace
	{
		// GLSL names of the fragment function constants: constant_id N = bit N of fragment_feature
		constexpr std::array<const char*, fs_feature_count> s_feature_names =
		{
			"FC_TEXTURES",
			"FC_TEXTURE_2D",
			"FC_TEXTURE_3D",
			"FC_TEXTURE_CUBE",
			"FC_SHADOW",
			"FC_REDIRECT",
			"FC_TEXTURE_CONVERT",
			"FC_TEXTURE_EXPAND",
			"FC_FLOW_CONTROL",
			"FC_PRECISION",
			"FC_TEXCOORD_CONTROL",
			"FC_ALPHA_TEST",
			"FC_ALPHA_TO_COVERAGE",
			"FC_SRGB_OUTPUT",
			"FC_OUTPUT_ROUNDING",
			"FC_OUTPUT_REMAP",
			"FC_POLYGON_STIPPLE",
		};

		// Texture sampling for the unit of the current instruction (_EXTERNAL_TEXTURE_OPS). Mirrors what the recompiler
		// emits for each unit (FragmentProgramDecompiler::handle_tex_srb, the TEX*D macros and MTLFragmentProgram's 1D
		// and LOD bias overrides), selected at run time from the instruction block header.
		constexpr std::string_view s_texture_ops =
R"(
#define _TEX_DIM_1D   0u
#define _TEX_DIM_2D   1u
#define _TEX_DIM_CUBE 2u
#define _TEX_DIM_3D   3u

#define _TEX_OP_SAMPLE 0u
#define _TEX_OP_BIAS   1u
#define _TEX_OP_LOD    2u
#define _TEX_OP_GRAD   3u

// Slot of a unit in a compacted array: its rank among the units of that kind
uint _texture_rank(const in uint mask, const in uint unit)
{
	return uint(bitCount(mask & ((1u << unit) - 1u)));
}

// Texel processing as compiled into the recompiled program: format conversion only with TEXTURE_FORMAT_CONVERT,
// alpha kill only with TEXTURE_ALPHA_KILL (RSXFragmentTextureOps.glsl, _process_texel)
vec4 _interpreter_process_texel(const in vec4 texel, const in uint flags)
{
	if (!FC_TEXTURE_CONVERT)
	{
		return texel;
	}

	uint control = ((fp_control & _CTRL_TEXTURE_FORMAT_CONVERT) != 0u) ? flags : (flags & (1u << ALPHAKILL));
	if ((fp_control & _CTRL_TEXTURE_ALPHA_KILL) == 0u)
	{
		control &= ~(1u << ALPHAKILL);
	}

	return _process_texel(texel, control);
}

// Stencil mirror: nearest, clamp to border (opaque black), level 0; read without a sampler
float _stencil_read(const in uint slot, const in uint unit, const in vec2 uv)
{
	if (!(all(greaterThanEqual(uv, vec2(0.))) && all(lessThan(uv, vec2(1.)))))
	{
		return 0.;
	}

	const ivec2 size = textureSize(usampler2D(texstencil_array[slot], sampler_array[unit]), 0);
	return float(texelFetch(usampler2D(texstencil_array[slot], sampler_array[unit]), ivec2(uv * vec2(size)), 0).x);
}

vec4 _texture_fetch(const in uint op, const in vec4 coord, const in float lod, const in vec4 dpdx, const in vec4 dpdy)
{
	// An instruction that executes for no channel samples nothing (the recompiler emits no code for it)
	if (!FC_TEXTURES || GET_INST_BITS(1, 18, 3) == 0u)
	{
		return vr_zero;
	}

	const uint unit = GET_INST_BITS(0, 17, 4);
	const uint unit_mask = 1u << unit;
	const uint dim = GET_BITS(texture_control, int(unit + unit), 2);
	vec4 result = vr_zero;

	if (FC_SHADOW && (shadow_textures & unit_mask) != 0u)
	{
		// Depth compare (TEX2D_SHADOW / TEX3D_SHADOW, emulated coordinates): no texel processing, no LOD bias. 1D
		// units and projected cube lookups are not implemented by the recompiler either and read 0.
		if (dim == _TEX_DIM_2D)
		{
			const uint slot = _texture_rank(shadow_slot_masks & 0xFFFFu, unit);
			result = vec4(texture(sampler2DShadow(texshadow2d_array[slot], sampler_array[unit]), _texcoord_xform_shadow(coord.xyz, TEX_PARAM(unit))));
		}
		else if (dim == _TEX_DIM_CUBE && inst.opcode != RSX_FP_OPCODE_TXP)
		{
			const uint slot = _texture_rank(shadow_slot_masks >> 16, unit);
			result = vec4(texture(samplerCubeShadow(texshadowcube_array[slot], sampler_array[unit]), _texcoord_xform_shadow(coord, TEX_PARAM(unit))));
		}

		return result;
	}

	// BX2: the recompiler brackets the fetch with _enable_texture_expand / _disable_texture_expand
	const bool expand = FC_TEXTURE_EXPAND && TEST_INST_BIT(0, 21);
	if (expand)
	{
		_enable_texture_expand(unit);
	}

	const float lod_bias = _TEX_LOD_BIAS(unit);

	if (FC_REDIRECT && (redirected_textures & unit_mask) != 0u)
	{
		// Depth and stencil read as RGBA8 (TEX1D/TEX2D_Z24X8_RGBA8): no LOD, no bias. Cube and 3D units are not interpreted.
		const vec2 uv = (dim == _TEX_DIM_1D) ? vec2(_texcoord_xform(coord.x, TEX_PARAM(unit)), 0.5) : _texcoord_xform(coord.xy, TEX_PARAM(unit));
		const float depth = texture(sampler2D(tex2d_array[unit], sampler_array[unit]), uv).r;
		const float stencil = _stencil_read(_texture_rank(redirected_textures, unit), unit, uv);
		result = _interpreter_process_texel(convert_z24x8_to_rgba8(vec2(depth, stencil), TEX_PARAM(unit).remap, TEX_FLAGS(unit)), TEX_FLAGS(unit));
	}
	else if (dim == _TEX_DIM_3D || dim == _TEX_DIM_CUBE)
	{
		// TEX3D_* (cube maps and 3D textures)
		const vec3 uvw = _texcoord_xform(coord.xyz, TEX_PARAM(unit));

		if (dim == _TEX_DIM_3D && FC_TEXTURE_3D)
		{
			switch (op)
			{
			case _TEX_OP_SAMPLE: result = texture(sampler3D(tex3d_array[unit], sampler_array[unit]), uvw, lod_bias); break;
			case _TEX_OP_BIAS: result = texture(sampler3D(tex3d_array[unit], sampler_array[unit]), uvw, lod + lod_bias); break;
			case _TEX_OP_LOD: result = textureLod(sampler3D(tex3d_array[unit], sampler_array[unit]), uvw, lod + lod_bias); break;
			default: result = textureGrad(sampler3D(tex3d_array[unit], sampler_array[unit]), uvw, dpdx.xyz * exp2(lod_bias), dpdy.xyz * exp2(lod_bias)); break;
			}
		}
		else if (dim == _TEX_DIM_CUBE && FC_TEXTURE_CUBE)
		{
			switch (op)
			{
			case _TEX_OP_SAMPLE: result = texture(samplerCube(texcube_array[unit], sampler_array[unit]), uvw, lod_bias); break;
			case _TEX_OP_BIAS: result = texture(samplerCube(texcube_array[unit], sampler_array[unit]), uvw, lod + lod_bias); break;
			case _TEX_OP_LOD: result = textureLod(samplerCube(texcube_array[unit], sampler_array[unit]), uvw, lod + lod_bias); break;
			default: result = textureGrad(samplerCube(texcube_array[unit], sampler_array[unit]), uvw, dpdx.xyz * exp2(lod_bias), dpdy.xyz * exp2(lod_bias)); break;
			}
		}

		result = _interpreter_process_texel(result, TEX_FLAGS(unit));
	}
	else if (FC_TEXTURE_2D)
	{
		// TEX1D_* (2D textures of height 1 sampled at t = 0.5) and TEX2D_*
		const bool is_1d = (dim == _TEX_DIM_1D);
		const vec2 uv = is_1d ? vec2(_texcoord_xform(coord.x, TEX_PARAM(unit)), 0.5) : _texcoord_xform(coord.xy, TEX_PARAM(unit));

		switch (op)
		{
		case _TEX_OP_SAMPLE: result = texture(sampler2D(tex2d_array[unit], sampler_array[unit]), uv, lod_bias); break;
		case _TEX_OP_BIAS: result = texture(sampler2D(tex2d_array[unit], sampler_array[unit]), uv, lod + lod_bias); break;
		case _TEX_OP_LOD: result = textureLod(sampler2D(tex2d_array[unit], sampler_array[unit]), uv, lod + lod_bias); break;
		default:
			result = textureGrad(sampler2D(tex2d_array[unit], sampler_array[unit]), uv,
				(is_1d ? vec2(dpdx.x, 0.) : dpdx.xy) * exp2(lod_bias), (is_1d ? vec2(dpdy.x, 0.) : dpdy.xy) * exp2(lod_bias));
			break;
		}

		result = _interpreter_process_texel(result, TEX_FLAGS(unit));
	}

	if (expand)
	{
		_disable_texture_expand();
	}

	return result;
}

// Entry points of the shared interpreter (TEX and TXP (projected by the caller) / TXB, TXL, TXD)
vec4 _texture(in vec4 coord, float bias)
{
	return _texture_fetch((inst.opcode == RSX_FP_OPCODE_TXB) ? _TEX_OP_BIAS : _TEX_OP_SAMPLE, coord, bias, vr_zero, vr_zero);
}

vec4 _textureLod(in vec4 coord, float lod)
{
	return _texture_fetch(_TEX_OP_LOD, coord, lod, vr_zero, vr_zero);
}

vec4 _textureGrad(in vec4 coord, in vec4 dpdx, in vec4 dpdy)
{
	return _texture_fetch(_TEX_OP_GRAD, coord, 0., dpdx, dpdy);
}

)";
	}

	void source_builder::build_vertex(MTLVertexProgram& prog, u32 library)
	{
		const bool instancing = !!(library & vs_library_instancing);

		// Binding table (VK shader_interpreter::init): vertex streams 0-2, context 3, conditional rendering predicate
		// 4, then the constants
		std::memset(&prog.binding_table, 0xff, sizeof(prog.binding_table));
		prog.binding_table.vertex_buffers_location = 0;
		prog.binding_table.context_buffer_location = 3;
		prog.binding_table.cr_pred_buffer_location = 4;

		if (instancing)
		{
			prog.binding_table.instanced_lut_buffer_location = 5;
			prog.binding_table.instanced_cbuf_location = 6;
		}
		else
		{
			prog.binding_table.cbuf_location = 5;
		}

		static_assert(vertex_instructions_location > 6);

		RSXVertexProgram null_prog;
		null_prog.ctrl = instancing ? RSX_SHADER_CONTROL_INSTANCED_CONSTANTS : 0;

		std::string shader_str;
		ParamArray arr;
		MTLVertexDecompilerThread comp(null_prog, shader_str, arr, prog);
		comp.properties.has_indexed_constants = true;
		comp.m_device_props.emulate_conditional_rendering = mtl::emulate_conditional_rendering();

		ParamType uniforms = { PF_PARAM_UNIFORM, "vec4" };
		uniforms.items.emplace_back("vc[468]", -1);

		std::stringstream builder;

		// Same declarations as the recompiled vertex programs: draw parameters, contexts, predicate, push constants,
		// constants (a uniform block, or the instancing tables), vertex streams
		comp.insertHeader(builder);
		comp.insertConstants(builder, { uniforms });
		comp.insertInputs(builder, {});

		builder <<
			"#define xform_constants_offset get_draw_params().xform_constants_offset\n"
			"#define scale_offset_mat get_vertex_context().scale_offset_mat\n"
			"#define transform_branch_bits get_vertex_context().transform_branch_bits\n"
			"#define point_size get_vertex_context().point_size\n"
			"#define z_near get_vertex_context().z_near\n"
			"#define z_far get_vertex_context().z_far\n\n";

		// Read by every vertex of a draw with the same (uniform) index: a uniform block (MSL constant address space)
		builder <<
			"layout(std430, set=0, binding=" << vertex_instructions_location << ") uniform VertexInstructionBlock\n"
			"{\n"
			"	uint base_address;\n"
			"	uint entry;\n"
			"	uint output_mask;\n"
			"	uint control;\n"
			"	uvec4 vp_instructions[];\n"
			"};\n\n";

		::glsl::shader_properties properties{};
		properties.domain = ::glsl::program_domain::glsl_vertex_program;
		properties.require_lit_emulation = true;
		properties.require_clip_plane_functions = true;
		properties.emulate_zclip_transform = true;
		properties.emulate_depth_clip_only = false; // No fp64 in MSL (as MTLVertexDecompilerThread)
		properties.require_instanced_render = instancing;

		::glsl::insert_glsl_legacy_function(builder, properties);
		::glsl::insert_vertex_input_fetch(builder, ::glsl::glsl_rules_vulkan);

		builder <<
			"// The fragment stage reads its per-draw offsets from push constants: no payload varying\n"
			"#define write_fs_payload()\n\n"
			"// Positions are computed the same way in every program (multipass depth-equal tests, see MTLVertexProgram)\n"
			"invariant gl_Position;\n\n"
			"#define main _interpreter_vs_main\n";

		builder << program_common::interpreter::get_vertex_interpreter();

		builder <<
			"#undef main\n\n"
			"void main()\n"
			"{\n";

		if (comp.m_device_props.emulate_conditional_rendering)
		{
			builder <<
				"	if (cr_predicate_value == 0)\n"
				"	{\n"
				"		gl_Position = vec4(0., 0., 0., -1.);\n"
				"		return;\n"
				"	}\n\n";
		}

		builder <<
			"	_interpreter_vs_main();\n"
			"	gl_Position = apply_zclip_xform(gl_Position, z_near, z_far);\n"
			"}\n";

		auto inputs = comp.get_inputs();
		inputs.push_back(mtl::glsl::program_input::make(
			::glsl::glsl_vertex_program,
			"VertexInstructionBlock",
			mtl::glsl::input_type_uniform_buffer,
			mtl::glsl::binding_set_index_vertex,
			vertex_instructions_location));

		prog.SetInputs(inputs);
		prog.has_indexed_constants = true;
		prog.shader.create(::glsl::program_domain::glsl_vertex_program, builder.str());
		prog.handle = &prog.shader;
	}

	void source_builder::build_fragment(MTLFragmentProgram& prog, u32 library, bool emulate_lod_bias)
	{
		const u32 color_outputs = std::min(library & fs_library_color_output_mask, 4u);
		const bool depth_output = !!(library & fs_library_depth_output);
		const bool point_coord = !!(library & fs_library_point_coord);

		// Binding table (VK shader_interpreter::init): context 0, texture parameters 1, stipple pattern 2
		std::memset(&prog.binding_table, 0xff, sizeof(prog.binding_table));
		prog.binding_table.context_buffer_location = 0;
		prog.binding_table.tex_param_location = 1;
		prog.binding_table.polygon_stipple_params_location = 2;

		RSXFragmentProgram frag;
		frag.ctrl |= RSX_SHADER_CONTROL_INTERPRETER_MODEL;

		u32 len = 0;
		ParamArray arr;
		std::string shader_str;
		MTLFragmentDecompilerThread comp(shader_str, arr, frag, len, prog);
		comp.metal_props.emulate_sampler_lod_bias = emulate_lod_bias;
		comp.properties.has_tex_op = true; // Declares the shader-side LOD bias push constants when emulated

		std::stringstream builder;
		builder <<
			"#version 450\n"
			"#extension GL_EXT_scalar_block_layout : require\n"
			"#extension GL_EXT_uniform_buffer_unsized_array : require\n"
			"#extension GL_ARB_separate_shader_objects : enable\n\n";

		::glsl::insert_subheader_block(builder);

		// Same declarations as the recompiled fragment programs: contexts, texture parameters, stipple pattern, push
		// constants (per-draw offsets, LOD bias)
		comp.insertConstants(builder);

		// Function constants: the uber pipeline keeps their defaults (every feature), specialized pipelines set them
		builder << "// Function constants (MSL [[function_constant(N)]]): features compiled in\n";
		for (u32 id = 0; id < fs_feature_count; ++id)
		{
			builder << "layout(constant_id = " << id << ") const bool " << s_feature_names[id] << " = true;\n";
		}

		builder <<
			"\n"
			"#define _INTERPRETER_FLOW_CONTROL FC_FLOW_CONTROL\n"
			"#define _INTERPRETER_PRECISION_MODIFIERS FC_PRECISION\n"
			"#define _INTERPRETER_TEXCOORD_CONTROL FC_TEXCOORD_CONTROL\n\n";

		// Instruction block (uniform: every fragment of a draw reads the same program). The header holds the draw's
		// program state that the recompiler bakes into the program (shader_interpreter::write_fragment_header).
		builder <<
			"layout(std430, set=1, binding=" << fragment_instructions_location << ") uniform FragmentInstructionBlock\n"
			"{\n"
			"	uint shader_control;      // RSX shader control (register count: at least what the program uses)\n"
			"	uint texture_control;     // Texture unit dimensions, 2 bits per unit\n"
			"	uint fp_control;          // RSXFragmentProgram::ctrl\n"
			"	uint mrt_count;           // Colour outputs of the program\n"
			"	uint texcoord_control;    // RSXFragmentProgram::texcoord_control_mask\n"
			"	uint shadow_textures;     // Units sampled with depth compare\n"
			"	uint redirected_textures; // Depth units read as RGBA8\n"
			"	uint shadow_slot_masks;   // Shadow 2D units | shadow cube units << 16\n"
			"	uvec4 fp_instructions[];\n"
			"};\n\n";

		static_assert(fragment_header_size == 8 * sizeof(u32));

		// Texture units: one image array per sampling type (a unit's image is in the array of its type) and the samplers
		// of the 16 units. Depth compare and stencil mirror arrays are compacted (slot = rank of the unit).
		builder <<
			"layout(set=1, binding=" << fragment_textures_location + 0 << ") uniform texture2D tex2d_array[" << texture_units << "];\n"
			"layout(set=1, binding=" << fragment_textures_location + 1 << ") uniform texture3D tex3d_array[" << texture_units << "];\n"
			"layout(set=1, binding=" << fragment_textures_location + 2 << ") uniform textureCube texcube_array[" << texture_units << "];\n"
			"layout(set=1, binding=" << fragment_textures_location + 3 << ") uniform texture2D texshadow2d_array[" << shadow_2d_slots << "];\n"
			"layout(set=1, binding=" << fragment_textures_location + 4 << ") uniform textureCube texshadowcube_array[" << shadow_cube_slots << "];\n"
			"layout(set=1, binding=" << fragment_textures_location + 5 << ") uniform utexture2D texstencil_array[" << stencil_slots << "];\n"
			"layout(set=1, binding=" << fragment_samplers_location << ") uniform sampler sampler_array[" << texture_units << "];\n\n";

		static_assert(fragment_textures_location + 6 <= fragment_samplers_location);

		builder <<
			"#define fog_param0 fs_contexts[_fs_context_offset].fog_param0\n"
			"#define fog_param1 fs_contexts[_fs_context_offset].fog_param1\n"
			"#define fog_mode fs_contexts[_fs_context_offset].fog_mode\n"
			"#define wpos_scale fs_contexts[_fs_context_offset].wpos_scale\n"
			"#define wpos_bias fs_contexts[_fs_context_offset].wpos_bias\n"
			"#define TEX_PARAM(index) texture_parameters[(index) + _fs_texture_base_index]\n\n";

		builder << fmt::format(
			"#define _CTRL_32_BITS_EXPORTS 0x%x\n"
			"#define _CTRL_DEPTH_EXPORT 0x%x\n"
			"#define _CTRL_ALPHA_TEST 0x%x\n"
			"#define _CTRL_ALPHA_TO_COVERAGE 0x%x\n"
			"#define _CTRL_SRGB_FRAMEBUFFER 0x%x\n"
			"#define _CTRL_8BIT_FRAMEBUFFER 0x%x\n"
			"#define _CTRL_ROP_OUTPUT_REMAP 0x%x\n"
			"#define _CTRL_POLYGON_STIPPLE 0x%x\n"
			"#define _CTRL_TEXTURE_FORMAT_CONVERT 0x%x\n"
			"#define _CTRL_TEXTURE_ALPHA_KILL 0x%x\n\n",
			u32{CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS}, u32{CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT},
			u32{RSX_SHADER_CONTROL_ALPHA_TEST}, u32{RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE}, u32{RSX_SHADER_CONTROL_SRGB_FRAMEBUFFER},
			u32{RSX_SHADER_CONTROL_8BIT_FRAMEBUFFER}, u32{RSX_SHADER_CONTROL_ROP_OUTPUT_REMAP}, u32{RSX_SHADER_CONTROL_POLYGON_STIPPLE},
			u32{RSX_SHADER_CONTROL_TEXTURE_FORMAT_CONVERT}, u32{RSX_SHADER_CONTROL_TEXTURE_ALPHA_KILL});

		builder << (emulate_lod_bias
			? "// Samplers cannot apply the LOD bias on this GPU (see MTLFragmentProgram::requires_lod_bias)\n"
			  "#define _TEX_LOD_BIAS(unit) texture_lod_bias[(unit) / 4u][(unit) % 4u]\n\n"
			: "#define _TEX_LOD_BIAS(unit) 0.\n\n");

		// Shared interpreter configuration: every path compiled in (function constants strip the unused ones), texture
		// sampling and the output stage provided below
		builder <<
			"#define WITH_TEXTURES\n"
			"#define WITH_FLOW_CTRL\n"
			"#define WITH_KIL\n"
			"#define WITH_PACKING\n"
			"#define WITH_TEXCOORD_CONTROL\n"
			"#define _EXTERNAL_TEXTURE_OPS\n"
			"#define _EXTERNAL_ROP_EXPORT\n";

		if (point_coord)
		{
			builder << "#define WITH_POINT_COORD\n";
		}

		builder << "\n";

		// The recompiler's helpers (every option): texel processing, depth reconstruction, sRGB, output rounding,
		// comparison, coverage test, output remapping
		::glsl::shader_properties properties{};
		properties.domain = ::glsl::program_domain::glsl_fragment_program;
		properties.require_lit_emulation = true;
		properties.fp32_outputs = true;
		properties.supports_native_fp16 = false;
		properties.require_texture_ops = true;
		properties.require_tex1D_ops = true;
		properties.require_tex2D_ops = true;
		properties.require_tex3D_ops = true;
		properties.require_tex_shadow_ops = true;
		properties.require_shadowProj_ops = true;
		properties.emulate_shadow_compare = true;
		properties.require_texture_expand = true;
		properties.require_alpha_kill = true;
		properties.require_color_format_convert = true;
		properties.require_depth_conversion = true;
		properties.require_srgb_to_linear = true;
		properties.require_linear_to_srgb = true;
		properties.ROP_output_rounding = true;
		properties.ROP_sRGB_packing = true;
		properties.ROP_alpha_test = true;
		properties.ROP_alpha_to_coverage_test = true;
		properties.ROP_channel_remap = true;

		::glsl::insert_glsl_legacy_function(builder, properties);

		builder <<
			"#undef col0\n"
			"#undef col1\n"
			"#undef col2\n"
			"#undef col3\n\n"
			"// Texture sampling of the shared interpreter, defined after it (it reads the current instruction)\n"
			"vec4 _texture(in vec4 coord, float bias);\n"
			"vec4 _textureLod(in vec4 coord, float lod);\n"
			"vec4 _textureGrad(in vec4 coord, in vec4 dpdx, in vec4 dpdy);\n\n"
			"#define main _interpreter_fs_main\n";

		builder << program_common::interpreter::get_fragment_interpreter();

		builder << "#undef main\n";
		builder << s_texture_ops;

		// Output stage (RSXROPPrologue / RSXROPEpilogue.glsl with this library's colour output count)
		for (u32 i = 0; i < color_outputs; ++i)
		{
			builder << "layout(location=" << i << ") out vec4 rop_out" << i << ";\n";
		}

		if (color_outputs)
		{
			builder << "\n";
		}

		builder <<
			"void main()\n"
			"{\n"
			"	if (FC_POLYGON_STIPPLE && (fp_control & _CTRL_POLYGON_STIPPLE) != 0u)\n"
			"	{\n"
			"		const uvec2 stipple_coord = uvec2(gl_FragCoord.xy) % uvec2(32, 32);\n"
			"		const uint address = stipple_coord.y * 32u + stipple_coord.x;\n"
			"		const uint word_index = _get_bits(address, 7, 3) + _fs_stipple_pattern_array_offset;\n"
			"		const uint sub_index = _get_bits(address, 5, 2);\n\n"
			"		if (!_test_bit(stipple_pattern[word_index][sub_index], int(address & 31u)))\n"
			"		{\n"
			"			discard;\n"
			"		}\n"
			"	}\n\n"
			"	_interpreter_fs_main();\n\n"
			"	const uint rop_control = fs_contexts[_fs_context_offset].rop_control;\n"
			"	const bool fp32_exports = (fp_control & _CTRL_32_BITS_EXPORTS) != 0u;\n"
			"	const bool output_rounding = FC_OUTPUT_ROUNDING && (fp_control & _CTRL_8BIT_FRAMEBUFFER) != 0u;\n"
			"	vec4 color[4];\n"
			"	color[0] = fp32_exports ? regs32[0] : regs16[0];\n"
			"	color[1] = fp32_exports ? regs32[2] : regs16[4];\n"
			"	color[2] = fp32_exports ? regs32[3] : regs16[6];\n"
			"	color[3] = fp32_exports ? regs32[4] : regs16[8];\n\n"
			"	for (int i = 0; i < " << color_outputs << "; ++i)\n"
			"	{\n"
			"		// sRGB conversion happens before output quantization\n"
			"		if (FC_SRGB_OUTPUT && (fp_control & _CTRL_SRGB_FRAMEBUFFER) != 0u) color[i].rgb = linear_to_srgb(color[i]).rgb;\n"
			"		if (output_rounding) color[i] = round_to_8bit(color[i]);\n"
			"	}\n\n"
			"	if (FC_ALPHA_TEST && (fp_control & _CTRL_ALPHA_TEST) != 0u)\n"
			"	{\n"
			"		const float alpha = (" << color_outputs << " >= 1 || !output_rounding) ? color[0].a : round_to_8bit(color[0].a);\n"
			"		const uint alpha_func = _get_bits(rop_control, ALPHA_TEST_FUNC_OFFSET, ALPHA_TEST_FUNC_LENGTH);\n"
			"		if (!comparison_passes(alpha, fs_contexts[_fs_context_offset].alpha_ref, alpha_func))\n"
			"		{\n"
			"			discard;\n"
			"		}\n"
			"	}\n\n"
			"	if (FC_ALPHA_TO_COVERAGE && (fp_control & _CTRL_ALPHA_TO_COVERAGE) != 0u && !coverage_test_passes(color[0]))\n"
			"	{\n"
			"		discard;\n"
			"	}\n\n"
			"	if (FC_OUTPUT_REMAP && (fp_control & _CTRL_ROP_OUTPUT_REMAP) != 0u)\n"
			"	{\n"
			"		const uint ROP_remap = get_ROP_channel_remap();\n"
			"		for (int i = 0; i < " << color_outputs << "; ++i)\n"
			"		{\n"
			"			color[i] = remap_ROP_output(color[i], ROP_remap);\n"
			"		}\n"
			"	}\n";

		for (u32 i = 0; i < color_outputs; ++i)
		{
			builder << "\n	rop_out" << i << " = color[" << i << "];";
		}

		if (depth_output)
		{
			builder <<
				"\n\n"
				"	// Depth export (always from r1, clamped like the recompiler), else the fragment's own depth (early Z disabled)\n"
				"	gl_FragDepth = ((fp_control & _CTRL_DEPTH_EXPORT) != 0u) ? _saturate(regs32[1].z) : gl_FragCoord.z;";
		}

		builder << "\n}\n";

		auto inputs = comp.get_inputs();

		auto input = mtl::glsl::program_input::make(
			::glsl::glsl_fragment_program,
			"FragmentInstructionBlock",
			mtl::glsl::input_type_uniform_buffer,
			mtl::glsl::binding_set_index_fragment,
			fragment_instructions_location);
		inputs.push_back(input);

		static constexpr std::array<std::pair<const char*, u32>, 6> texture_arrays =
		{{
			{ "tex2d_array", texture_units },
			{ "tex3d_array", texture_units },
			{ "texcube_array", texture_units },
			{ "texshadow2d_array", shadow_2d_slots },
			{ "texshadowcube_array", shadow_cube_slots },
			{ "texstencil_array", stencil_slots },
		}};

		for (u32 i = 0; i < texture_arrays.size(); ++i)
		{
			input.name = texture_arrays[i].first;
			input.type = mtl::glsl::input_type_separate_image;
			input.location = fragment_textures_location + i;
			input.array_size = texture_arrays[i].second;
			inputs.push_back(input);
		}

		input.name = "sampler_array";
		input.type = mtl::glsl::input_type_sampler;
		input.location = fragment_samplers_location;
		input.array_size = texture_units;
		inputs.push_back(input);

		prog.SetInputs(inputs);
		prog.shader.create(::glsl::program_domain::glsl_fragment_program, builder.str());
		prog.handle = &prog.shader;
	}
}

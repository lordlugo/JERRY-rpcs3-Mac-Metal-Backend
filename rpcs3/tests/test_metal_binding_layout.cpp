#include <gtest/gtest.h>

#include "Emu/RSX/Metal/MTLProgramPipeline.h"
#include "Emu/RSX/Metal/mtlutils/commands.h"
#include "Emu/RSX/Metal/mtlutils/device.h"

// program::bind() writes every slot of the binding layout straight into an
// MTL4ArgumentTable (setAddress/setTexture/setSamplerState). An out-of-range
// slot index corrupts the heap or faults inside the driver with no context
// (the RSX 0x03c4da4 setSamplerState segfault). build_binding_layout() must
// therefore never emit a slot outside the table capacities (31 buffers, 64
// textures, 16 samplers); exhaustion has to degrade to umax instead.
namespace
{
	using namespace mtl::glsl;

	constexpr u32 kBuffers = mtl::gpu_capabilities::max_buffers_per_stage; // 31
	constexpr u32 kTextures = mtl::argument_table_shadow::max_textures;     // 64
	constexpr u32 kSamplers = mtl::gpu_capabilities::max_samplers_per_stage; // 16

	program_input make_input(program_input_type type, u32 location, u32 array_size = 1)
	{
		return program_input::make(::glsl::glsl_fragment_program, "test_input", type,
			binding_set_index_fragment, location, {}, array_size);
	}

	// Overflow-safe range check, mirroring slot_range_exceeds() in
	// MTLProgramPipeline.cpp: [index, index + count) must lie in [0, limit).
	bool slot_in_range(u32 index, u32 count, u32 limit)
	{
		return index < limit && count <= limit - index;
	}

	// Mirrors the validation program::init_layouts() enforces: any violation here
	// would fault the driver at bind() time instead of failing loudly.
	void expect_slots_in_range(const binding_layout& layout)
	{
		for (const auto& [location, slot] : layout.slots)
		{
			EXPECT_GE(slot.array_size, 1u) << "location " << location;

			if (slot.buffer_index != umax)
			{
				EXPECT_TRUE(slot_in_range(slot.buffer_index, slot.array_size, kBuffers)) << "location " << location;
			}

			if (slot.texture_index != umax)
			{
				EXPECT_TRUE(slot_in_range(slot.texture_index, slot.array_size, kTextures)) << "location " << location;
			}

			if (slot.sampler_index != umax)
			{
				EXPECT_TRUE(slot_in_range(slot.sampler_index, slot.array_size, kSamplers)) << "location " << location;
			}
		}
	}
}

TEST(MetalBindingLayout, CombinedSamplersMatchTexturesUpTo16)
{
	// 20 sampled textures: the first 16 share their texture index as sampler
	// index, the rest get no sampler (constant sampler in MSL).
	std::vector<program_input> inputs;
	for (u32 i = 0; i < 20; ++i)
	{
		inputs.push_back(make_input(input_type_texture, i));
	}

	const binding_layout layout = build_binding_layout(inputs);
	expect_slots_in_range(layout);

	EXPECT_EQ(layout.texture_count, 20u);
	EXPECT_EQ(layout.sampler_count, 16u);

	for (u32 i = 0; i < 20; ++i)
	{
		const resource_slot& slot = layout.slots.at(i);
		EXPECT_EQ(slot.texture_index, i);
		EXPECT_EQ(slot.sampler_index, i < 16 ? i : umax) << "texture " << i;
	}
}

TEST(MetalBindingLayout, OversizedSamplerArrayGetsNoSampler)
{
	// A single texture array wider than the sampler table still gets its
	// textures, but no sampler instead of an out-of-range index.
	std::vector<program_input> inputs{ make_input(input_type_texture, 0, 20) };

	const binding_layout layout = build_binding_layout(inputs);
	expect_slots_in_range(layout);

	const resource_slot& slot = layout.slots.at(0);
	EXPECT_EQ(slot.texture_index, 0u);
	EXPECT_EQ(slot.sampler_index, umax);
}

TEST(MetalBindingLayout, SeparateSamplersTakeRemainingSlots)
{
	// 2 combined textures occupy sampler slots 0-1; 20 separate samplers fill
	// 2-15 and the rest degrade to umax.
	std::vector<program_input> inputs;
	inputs.push_back(make_input(input_type_texture, 0));
	inputs.push_back(make_input(input_type_texture, 1));
	for (u32 i = 0; i < 20; ++i)
	{
		inputs.push_back(make_input(input_type_sampler, 2 + i));
	}

	const binding_layout layout = build_binding_layout(inputs);
	expect_slots_in_range(layout);

	EXPECT_EQ(layout.slots.at(0).sampler_index, 0u);
	EXPECT_EQ(layout.slots.at(1).sampler_index, 1u);

	for (u32 i = 0; i < 20; ++i)
	{
		const resource_slot& slot = layout.slots.at(2 + i);
		EXPECT_EQ(slot.texture_index, umax) << "separate sampler " << i;
		EXPECT_EQ(slot.sampler_index, i < 14 ? 2 + i : umax) << "separate sampler " << i;
	}
}

TEST(MetalBindingLayout, BufferOverflowKeepsUmax)
{
	// 40 uniform buffers: 31 get slots, the rest stay umax (translation of a
	// shader that really uses them fails instead of faulting the driver).
	std::vector<program_input> inputs;
	for (u32 i = 0; i < 40; ++i)
	{
		inputs.push_back(make_input(input_type_uniform_buffer, i));
	}

	const binding_layout layout = build_binding_layout(inputs);
	expect_slots_in_range(layout);

	EXPECT_EQ(layout.buffer_count, 31u);

	for (u32 i = 0; i < 40; ++i)
	{
		EXPECT_EQ(layout.slots.at(i).buffer_index, i < 31 ? i : umax) << "buffer " << i;
	}
}

TEST(MetalBindingLayout, AbsurdArraySizeDegradesToUmax)
{
	// A corrupt array_size near 2^32 must not wrap into an in-range index: with
	// plain `next + count > limit` arithmetic the second buffer below wraps to
	// index 20 (and the second texture to 60), which bind() would write out of
	// bounds. Every slot must degrade to umax instead.
	std::vector<program_input> inputs{
		make_input(input_type_uniform_buffer, 0, 20),
		make_input(input_type_uniform_buffer, 1, 0xFFFF'FFF0),
		make_input(input_type_texture, 2, 60),
		make_input(input_type_texture, 3, 0xFFFF'FFF0),
	};

	const binding_layout layout = build_binding_layout(inputs);
	expect_slots_in_range(layout);

	EXPECT_EQ(layout.slots.at(0).buffer_index, 0u);
	EXPECT_EQ(layout.slots.at(1).buffer_index, umax);
	EXPECT_EQ(layout.slots.at(2).texture_index, 0u);
	EXPECT_EQ(layout.slots.at(3).texture_index, umax);
}

TEST(MetalBindingLayout, HeavyMixedProgramStaysInRange)
{
	// Adversarial mix past every capacity: 30 UBOs, 70 sampled textures,
	// 10 texel buffers, 4 separate images, 20 separate samplers and a push
	// constant block. Nothing may escape the table; overflow degrades to umax.
	std::vector<program_input> inputs;
	u32 location = 0;

	for (u32 i = 0; i < 30; ++i)
	{
		inputs.push_back(make_input(input_type_uniform_buffer, location++));
	}

	for (u32 i = 0; i < 70; ++i)
	{
		inputs.push_back(make_input(input_type_texture, location++));
	}

	for (u32 i = 0; i < 10; ++i)
	{
		inputs.push_back(make_input(input_type_texel_buffer, location++));
	}

	for (u32 i = 0; i < 4; ++i)
	{
		inputs.push_back(make_input(input_type_separate_image, location++));
	}

	for (u32 i = 0; i < 20; ++i)
	{
		inputs.push_back(make_input(input_type_sampler, location++));
	}

	program_input push = program_input::make(::glsl::glsl_fragment_program, "push",
		input_type_push_constant, binding_set_index_fragment, location++, { 0, 64 });
	inputs.push_back(push);

	const binding_layout layout = build_binding_layout(inputs);
	expect_slots_in_range(layout);

	EXPECT_LE(layout.buffer_count, kBuffers);
	EXPECT_LE(layout.texture_count, kTextures);
	EXPECT_LE(layout.sampler_count, kSamplers);

	if (layout.push_constant_buffer_index != umax)
	{
		EXPECT_LT(layout.push_constant_buffer_index, kBuffers);
	}
}

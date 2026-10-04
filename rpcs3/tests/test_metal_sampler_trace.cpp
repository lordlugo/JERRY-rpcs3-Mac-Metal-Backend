#include <gtest/gtest.h>

#include "Emu/RSX/Metal/MTLProgramPipeline.h"

// The sampler-write ring is the always-on half of the setSamplerState
// forensics: program::bind() records every driver-visible sampler write without
// any env var, and the crash report dumps the recent entries. These tests pin
// the record/format contract (order, fields, wrap-around, truncation).

namespace
{
	using namespace mtl::glsl;

	TEST(MetalSamplerTrace, FormatIsOldestFirstWithAllFields)
	{
		sampler_write_ring ring;
		EXPECT_TRUE(ring.empty());

		ring.record(/*prog*/ 7, /*stage*/ 1, /*slot*/ 3, /*id*/ 0x32, /*live*/ true);
		ring.record(7, 0, 0, 0x25, false);
		EXPECT_FALSE(ring.empty());

		std::string out;
		ring.format_last(out, 64);
		EXPECT_EQ(out, "prog=7 stage=1 slot=3 id=0x32 live=1\n"
		               "prog=7 stage=0 slot=0 id=0x25 live=0\n");
	}

	TEST(MetalSamplerTrace, MaxLinesKeepsTheNewest)
	{
		sampler_write_ring ring;
		ring.record(1, 0, 0, 0x10, true);
		ring.record(1, 0, 1, 0x11, true);
		ring.record(1, 0, 2, 0x12, true);

		std::string out;
		ring.format_last(out, 2);
		EXPECT_EQ(out, "prog=1 stage=0 slot=1 id=0x11 live=1\n"
		               "prog=1 stage=0 slot=2 id=0x12 live=1\n");
	}

	TEST(MetalSamplerTrace, WrapsAroundAndDropsOldest)
	{
		sampler_write_ring ring;

		for (u32 i = 0; i < sampler_write_ring::capacity + 5; ++i)
		{
			// Distinct prog uid per write so the surviving window is checkable.
			ring.record(1000 + i, 0, 0, 0x40, true);
		}

		std::string out;
		ring.format_last(out, sampler_write_ring::capacity + 100);

		// Exactly one capacity worth of lines survived...
		u32 lines = 0;
		for (char c : out) lines += (c == '\n');
		EXPECT_EQ(lines, sampler_write_ring::capacity);

		// ...oldest first (prog 1005, the first five overwritten) through newest last.
		EXPECT_TRUE(out.starts_with("prog=1005 stage=0 slot=0 id=0x40 live=1\n"));
		const std::string newest = "prog=1132 stage=0 slot=0 id=0x40 live=1\n";
		EXPECT_TRUE(out.ends_with(newest)) << out.substr(out.size() > 200 ? out.size() - 200 : 0);
	}

	TEST(MetalSamplerTrace, CrashSectionIsEmptyUntilFirstWrite)
	{
		// No other test records sampler writes, so the process-global ring starts
		// empty and the crash report stays clean for Metal-free sessions.
		EXPECT_EQ(sampler_write_section(), "");

		global_sampler_write_ring().record(4242, 1, 7, 0x33, true);

		const std::string section = sampler_write_section();
		EXPECT_NE(section.find("Recent Metal sampler writes"), std::string::npos);
		EXPECT_NE(section.find("prog=4242 stage=1 slot=7 id=0x33 live=1"), std::string::npos);
	}
}

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "Emu/RSX/Metal/MTLDepthBounds.h"

// The depth bounds test compares the depth STORED in the depth buffer at the
// fragment's (x, y) against [min, max]; it has no dependency on the
// fragment's own depth (EXT_depth_bounds_test, Vulkan depthBoundsTestEnable).
// Deferred renderers (GTA IV, Wolverine) limit light volumes with it: testing
// the fragment's own depth instead lit the wrong pixels (flickering light
// patches). The stored depth is read from a copy made outside the draw pass.

namespace
{
	std::string emit()
	{
		std::stringstream OS;
		mtl::append_depth_bounds_test(OS);
		return OS.str();
	}
}

TEST(MetalDepthBounds, TestsStoredDepthFromTheCopy)
{
	const std::string src = emit();
	EXPECT_NE(src.find("texelFetch(depth_bounds_texture, ivec2(gl_FragCoord.xy), 0)"), std::string::npos);
	EXPECT_NE(src.find("_db_z < depth_bounds.x || _db_z > depth_bounds.y"), std::string::npos);
	EXPECT_NE(src.find("discard"), std::string::npos);
}

TEST(MetalDepthBounds, IgnoresFragmentAndExportedDepth)
{
	const std::string src = emit();
	EXPECT_EQ(src.find("gl_FragCoord.z"), std::string::npos);
	EXPECT_EQ(src.find("r1"), std::string::npos);
	EXPECT_EQ(src.find("gl_SampleMask"), std::string::npos);
}

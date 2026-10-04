#pragma once

#include <sstream>

// Depth bounds test snippet for the Metal fragment decompiler (MTLFragmentProgram).
// Dependency-free so unit tests can assert on the exact emission without linking the backend.
//
// Semantics (EXT_depth_bounds_test, Vulkan depthBoundsTestEnable, RSX): the test compares the depth value STORED in
// the depth buffer at the fragment's (x, y) against [min, max], and discards the fragment when it is outside. It has
// no dependency on the fragment's own depth ("Unlike the depth test, the depth bounds test has NO dependency on the
// fragment's window-space depth value", EXT_depth_bounds_test). Deferred renderers use it to limit light volumes to
// the pixels whose scene depth is in range (GTA IV, Wolverine): testing the fragment's own depth instead lights the
// wrong pixels, which shows as flickering light patches as the camera moves.
//
// The stored depth is read from depth_bounds_texture: a copy of the draw's depth buffer made outside the draw pass
// (MTLGSRender::update_depth_copy). Reading the depth attachment itself inside its pass is undefined on Apple GPUs
// (wrong values for some pixels of every tile: grids of squares). Multisampled depth buffers are read at sample 0
// (texelFetch(sampler2DMS, p, 0)); single-sampled ones at level 0 (texelFetch(sampler2D, p, 0)): the same call text.
// The test runs first thing in main(), before the program runs: like the hardware, it does not depend on anything
// the program computes (exported depth included).

namespace mtl
{
	inline void append_depth_bounds_test(std::stringstream& OS)
	{
		OS <<
			"	{\n"
			"		const float _db_z = texelFetch(depth_bounds_texture, ivec2(gl_FragCoord.xy), 0).x;\n"
			"		if (_db_z < depth_bounds.x || _db_z > depth_bounds.y) discard;\n"
			"	}\n\n";
	}
}

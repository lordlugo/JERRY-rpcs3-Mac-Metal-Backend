#include <gtest/gtest.h>

#include "Emu/system_config.h"

// Fork defaults that fix observed issues: construction must reflect them so a
// fresh install gets them without any migration marker files.
TEST(ConfigDefaults, TimeStretchingIsOn)
{
	// Smooths audio over game-side production jitter (skips/silence/underrun chops
	// instead of cutting in and out). See cellAudio xrun stats in RPCS3.log.
	const cfg_root def;
	EXPECT_TRUE(def.audio.enable_time_stretching.get());
}

TEST(ConfigDefaults, VSyncMatchesPlatformDefault)
{
	const cfg_root def;

#ifdef __APPLE__
	// The Metal renderer paces presentation to the display only with VSync enabled
	EXPECT_EQ(def.video.vsync.get(), vsync_mode::adaptive);
#else
	EXPECT_EQ(def.video.vsync.get(), vsync_mode::off);
#endif
}

TEST(ConfigDefaults, HostGPULabelsStayOff)
{
	// GoW-class games can stall with the Metal label path enabled, so it remains
	// opt-in: fresh installs must not turn it on.
	const cfg_root def;
	EXPECT_FALSE(def.video.host_label_synchronization.get());
}

#ifdef __APPLE__
TEST(ConfigDefaults, MetalQualityBufferDefaultsAreOn)
{
	// The Metal backend runs all four of these on the GPU: MSAA
	// resolve/unresolve are fragment passes (MTLResolveHelper), surface init
	// from guest memory uses GPU upload/tiling plus GPU scaling, and depth
	// readback uses GPU blit plus compute depth packing. Fresh installs get
	// the image-quality benefit directly; existing installs are migrated by
	// metal-fork-defaults-v8 (Emu/System.cpp).
	const cfg_root def;
	EXPECT_TRUE(def.video.read_color_buffers.get());
	EXPECT_TRUE(def.video.read_depth_buffer.get());
	EXPECT_TRUE(def.video.write_depth_buffer.get());
	EXPECT_TRUE(def.video.force_hw_MSAA_resolve.get());
}
#endif



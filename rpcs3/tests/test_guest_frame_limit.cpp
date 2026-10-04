#include <gtest/gtest.h>

#include "Emu/RSX/Common/guest_frame_limit.hpp"

// Game speed lock (Emu/RSX/Common/guest_frame_limit.hpp): with Frame Limit
// "Display" on a 120 Hz screen, fixed-timestep games (Ben 10 Ultimate Alien:
// Cosmic Destruction and most PS3 titles) must keep flipping at the vblank
// rate, or their 1/60 s steps run twice per 1/60 s of real time.

TEST(GuestFrameLimit, EffectiveVblank)
{
	EXPECT_DOUBLE_EQ(rsx::effective_vblank_hz(60, false), 60.);
	EXPECT_NEAR(rsx::effective_vblank_hz(60, true), 59.94, 0.001);
	EXPECT_DOUBLE_EQ(rsx::effective_vblank_hz(120, false), 120.);
	EXPECT_DOUBLE_EQ(rsx::effective_vblank_hz(0, false), 0.);
}

TEST(GuestFrameLimit, DisplayRateIsCappedToVblank)
{
	// ProMotion 120 Hz, default 60 Hz vblank: guest held at 60
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(120., 60., 1, true), 60.);
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(144., 60., 1, true), 60.);
	// NTSC fixup
	EXPECT_NEAR(rsx::apply_game_speed_lock(120., 59.94, 1, true), 59.94, 1e-9);
}

TEST(GuestFrameLimit, LimitsAtOrBelowVblankAreUntouched)
{
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(60., 60., 1, true), 60.);
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(30., 60., 1, true), 30.);
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(50., 60., 1, true), 50.);
	// A 60.0001 Hz display report is not a reason to cap
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(60.0001, 60., 1, true), 60.0001);
}

TEST(GuestFrameLimit, UncappedAndDisabledAreUntouched)
{
	// Off / Infinite / boost hotkey: the user asked for no cap
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(0., 60., 1, true), 0.);
	// Lock disabled
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(120., 60., 1, false), 120.);
	// Raised vblank: the user re-timed the game on purpose
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(120., 120., 1, true), 120.);
}

TEST(GuestFrameLimit, RetimedGameRunsAtMultiple)
{
	// Timestep patch applied (1/120 s steps): 120 fps allowed, never more
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(120., 60., 2, true), 120.);
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(144., 60., 2, true), 120.);
	// Multiplier 0 behaves as 1
	EXPECT_DOUBLE_EQ(rsx::apply_game_speed_lock(120., 60., 0, true), 60.);
}

TEST(GuestFrameLimit, FrameIntervalHasNoDrift)
{
	double carry = 0.;
	std::uint64_t total = 0;

	for (int i = 0; i < 60 * 600; i++) // 10 minutes at 60 fps
	{
		const std::uint64_t us = rsx::next_frame_interval_us(60., carry);
		EXPECT_TRUE(us == 16666 || us == 16667);
		total += us;
	}

	// Exactly 600 s, give or take the sub-microsecond carry
	EXPECT_NEAR(static_cast<double>(total), 600'000'000., 1.);

	carry = 0.5;
	EXPECT_EQ(rsx::next_frame_interval_us(0., carry), 0u);
	EXPECT_DOUBLE_EQ(carry, 0.);
}

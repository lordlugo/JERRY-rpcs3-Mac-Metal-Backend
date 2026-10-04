#pragma once

// Guest frame-rate cap that keeps game speed correct at high frame limits.
//
// Most PS3 games advance their simulation by a FIXED step every frame (1/60 s,
// or 1/30 s), because on real hardware they are locked to the 60 Hz vblank.
// When the frame limit lets the guest flip faster than the vblank (Frame Limit
// "Display" on a 120 Hz ProMotion screen, or "120"), such a game runs 120 steps
// of 1/60 s per second: everything moves twice as fast, physics integrates in
// steps it was never tuned for, and timers/animations drift apart ("too fast and
// jumpy").
//
// The fix that is always correct is to keep the GUEST at the vblank rate while
// the HOST keeps presenting at the display rate: on a 120 Hz screen every 60 fps
// guest frame is held for exactly two refreshes by the presenter, which gives
// perfectly even pacing with the game running at its designed speed.
//
// A game whose timestep has been retimed (a timestep patch that verifiably
// applied, see GamePatches.h) can run faster: `logic_rate_multiplier` raises
// the cap to a whole multiple of the vblank rate (2 -> 120 fps at 60 Hz).
//
// Dependency-free on purpose: the policy is pure and unit-tested
// (rpcs3/tests/test_guest_frame_limit.cpp). RSXThread (guest flip limiter) and
// the Metal presenter (pacing target) both route their limit through it so the
// two can never disagree.

#include <cmath>
#include <cstdint>

namespace rsx
{
	// Effective emulated vblank frequency in Hz (vblank thread: period of 1'000'000 us,
	// or 1'001'000 us with the NTSC fixup, split into `vblank_rate` signals).
	inline double effective_vblank_hz(std::int64_t vblank_rate, bool ntsc_fixup)
	{
		if (vblank_rate <= 0)
		{
			return 0.;
		}

		return static_cast<double>(vblank_rate) * 1'000'000. / (ntsc_fixup ? 1'001'000. : 1'000'000.);
	}

	// Applies the game-speed lock to a frame limit in Hz (0 = uncapped).
	// - lock off, no vblank, or an uncapped limit (Off/Infinite/boost hotkey: the user asked for no cap): unchanged.
	// - otherwise the limit never exceeds vblank_hz * multiplier (multiplier 0 is treated as 1).
	inline double apply_game_speed_lock(double limit_hz, double vblank_hz, std::uint32_t logic_rate_multiplier, bool lock)
	{
		if (!lock || !(limit_hz > 0.) || !(vblank_hz > 0.))
		{
			return limit_hz;
		}

		const double cap = vblank_hz * static_cast<double>(logic_rate_multiplier ? logic_rate_multiplier : 1u);

		// Tolerance for display rates that are reported a hair above the vblank (60.0001 Hz screens)
		return limit_hz > cap * 1.0005 ? cap : limit_hz;
	}

	// Drift-free frame interval for the guest flip limiter. 1'000'000 / 60 is 16666.67 us: truncating it every
	// frame runs the guest 0.004% faster than the vblank-locked presenter, so the presenter periodically has two
	// frames due in one slot and drops one (a visible hitch every few minutes). `carry` accumulates the
	// fractional microseconds across frames so the long-run rate is exact.
	inline std::uint64_t next_frame_interval_us(double limit_hz, double& carry)
	{
		if (!(limit_hz > 0.))
		{
			carry = 0.;
			return 0;
		}

		const double exact = 1'000'000. / limit_hz + carry;
		const double whole = std::floor(exact);
		carry = exact - whole;
		return static_cast<std::uint64_t>(whole);
	}
}

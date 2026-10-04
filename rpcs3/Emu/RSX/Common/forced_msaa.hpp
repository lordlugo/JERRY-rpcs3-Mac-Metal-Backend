#pragma once

// Forced host MSAA: every RSX render target the game creates as single-sample is rendered with N hardware samples, for
// every game, transparently. The game still sees a single-sample surface of its own size (guest spp / samples_x /
// samples_y stay 1, so all sample-expanded memory and coordinate math is the identity); the host image is a multisample
// texture of the same (resolution-scaled) size. Wherever the surface is consumed (texture sampling, transfers, memory
// readback, presentation) it goes through the surface's resolve image, which for a forced surface is the averaged
// single-sample image (depth/stencil: sample 0); writes from transfers/uploads go to the resolve image and are broadcast
// back to every sample. See surface_sample_layout::forced (surface_utils.h) and MTLResolveHelper.
//
// Games that request PS3 MSAA themselves (2 or 4 samples) keep the native PS3 sample layout and their sample count.
// Active only with MSAA "Auto" (MSAA "Disabled" turns off all MSAA).

#include "Emu/system_config.h"
#include "Emu/RSX/gcm_enums.h"

namespace rsx
{
	// Host sample count forced onto single-sample surfaces: 1 (off), 2 or 4
	inline u8 forced_msaa_samples()
	{
		if (g_cfg.video.antialiasing_level != msaa_level::_auto)
		{
			return 1;
		}

		const s64 n = g_cfg.video.forced_msaa_samples;
		return n >= 4 ? 4 : n >= 2 ? 2 : 1;
	}

	// True when surfaces bound with this antialias mode are multisampled on the host (PS3 MSAA, or forced MSAA)
	inline bool surface_is_host_multisampled(surface_antialiasing aa)
	{
		return aa != surface_antialiasing::center_1_sample || forced_msaa_samples() > 1;
	}
}

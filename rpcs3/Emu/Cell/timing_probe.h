#pragma once

// Relaxed call counters for the guest timing APIs a game can use to derive its
// frame timestep. Reported periodically by the runtime timestep retimer
// (PPUModule.cpp, game_patches_runtime_on_flip) so a log shows how a game keeps
// time: per-flip vblank-count reads, flip-time reads, system-time reads, etc.
// A relaxed increment is a single uncontended atomic add: free on the hot path.

#include <atomic>
#include <cstdint>

namespace timing_probe
{
	inline std::atomic<std::uint64_t> gcm_get_vblank_count{0};
	inline std::atomic<std::uint64_t> gcm_get_last_flip_time{0};
	inline std::atomic<std::uint64_t> gcm_get_timestamp{0};
	inline std::atomic<std::uint64_t> sys_time_get_system_time{0};
	inline std::atomic<std::uint64_t> sys_time_get_current_time{0};
	inline std::atomic<std::uint64_t> vblank_handler_calls{0};
	inline std::atomic<std::uint64_t> vblank_signals{0};

	inline void hit(std::atomic<std::uint64_t>& counter)
	{
		counter.fetch_add(1, std::memory_order_relaxed);
	}
}

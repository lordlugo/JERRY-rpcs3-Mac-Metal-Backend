#pragma once

// Which sampler gpuResourceIDs are backed by a live sampler object, without including
// any Metal header (so unit tests can use it).
//
// program::bind() records sampler slots by numeric resource ID, but the ID is only a
// snapshot: the sampler object behind it can be destroyed afterwards (pool trim, renderer
// teardown, GPU-completion reclamation on another thread) while the ID lingers in a cached
// program's bindings. The driver segfaults with no context when a dead ID reaches
// setSamplerState, so bind() resolves every slot through this registry and substitutes a
// live fallback instead. All members are thread-safe.
//
// The registry alone cannot close the race: an ID that is live at contains() time can be
// destroyed before the driver's setSamplerState runs (reclamation is not on the RSX
// thread). bind() therefore holds the returned guard across the driver call — removal of
// that ID blocks until the guard releases, making check and use atomic.

#include <cstdint>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <unordered_set>

namespace mtl
{
	// Env-gated sampler tracing (RPCS3_METAL_SAMPLER_TRACE=1): logs every sampler creation
	// and every setSamplerState write so a driver fault can be correlated with the exact
	// slot, ID and sampler state. Off by default; zero overhead when off (one cached flag).
	inline bool sampler_trace_enabled()
	{
		static const bool enabled = (std::getenv("RPCS3_METAL_SAMPLER_TRACE") != nullptr);
		return enabled;
	}

	struct sampler_liveness
	{
		static void add(std::uint64_t id)
		{
			if (!id)
			{
				return;
			}

			std::lock_guard lock(mutex());
			set().insert(id);
		}

		static void remove(std::uint64_t id)
		{
			if (!id)
			{
				return;
			}

			std::lock_guard lock(mutex());
			set().erase(id);
		}

		static bool contains(std::uint64_t id)
		{
			if (!id)
			{
				return false;
			}

			std::lock_guard lock(mutex());
			return set().find(id) != set().end();
		}

		// Budget of always-on sampler-creation log lines per process. Creation is
		// infrequent (hundreds per session) but the ID->config map is the only way
		// to identify a faulting sampler after the fact, so it must not depend on
		// a foresight-enabled env var. Returns true at most creation_trace_budget
		// times per process, then false forever.
		static constexpr unsigned creation_trace_budget = 8192;

		static bool sampler_creation_trace_budget()
		{
			static std::atomic<unsigned> remaining{creation_trace_budget};

			// Saturating take: exactly the first creation_trace_budget calls win,
			// no wraparound below zero.
			unsigned prev = remaining.load(std::memory_order_relaxed);

			while (prev > 0 && !remaining.compare_exchange_weak(prev, prev - 1, std::memory_order_relaxed))
			{
			}

			return prev > 0;
		}

		// Locks the registry and returns the held lock iff id is currently live.
		// The caller keeps the guard alive across the driver's setSamplerState call:
		// remove() of that ID blocks until then, so a live-checked ID cannot be
		// destroyed (and its driver-side resource freed) mid-call. An unowned guard
		// means dead or zero — substitute or skip, never call the driver.
		// The guard is a leaf lock (registry only); the driver call must not call
		// back into this registry.
		static std::unique_lock<std::mutex> retain_if_live(std::uint64_t id)
		{
			std::unique_lock<std::mutex> guard(mutex());

			if (!id || set().find(id) == set().end())
			{
				guard.unlock();
			}

			return guard;
		}

	private:
		// Leaked on purpose: samplers are destroyed during renderer teardown, which must
		// stay valid even if it runs after static destructors.
		static std::unordered_set<std::uint64_t>& set()
		{
			static auto* s_set = new std::unordered_set<std::uint64_t>();
			return *s_set;
		}

		static std::mutex& mutex()
		{
			static auto* s_mutex = new std::mutex();
			return *s_mutex;
		}
	};
}

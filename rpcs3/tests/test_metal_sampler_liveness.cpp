#include <gtest/gtest.h>

#include "Emu/RSX/Metal/mtlutils/sampler_liveness.h"

#include <atomic>
#include <chrono>
#include <thread>

// program::bind() resolves every sampler slot through this registry so a destroyed
// sampler's ID can never reach setSamplerState (native segfault with no context).
// Distinctive IDs below avoid colliding with any live sampler in the process.
namespace
{
	constexpr std::uint64_t kFirstId = 0x5a11'0001ull;
	constexpr std::uint64_t kSecondId = 0x5a11'0002ull;
}

TEST(MetalSamplerLiveness, UnknownIdIsNotLive)
{
	EXPECT_FALSE(mtl::sampler_liveness::contains(kFirstId));
	EXPECT_FALSE(mtl::sampler_liveness::contains(0ull));
}

TEST(MetalSamplerLiveness, AddRemoveRoundTrip)
{
	mtl::sampler_liveness::add(kFirstId);
	EXPECT_TRUE(mtl::sampler_liveness::contains(kFirstId));
	mtl::sampler_liveness::remove(kFirstId);
	EXPECT_FALSE(mtl::sampler_liveness::contains(kFirstId));
}

TEST(MetalSamplerLiveness, ZeroIdIsIgnored)
{
	mtl::sampler_liveness::add(0ull);
	EXPECT_FALSE(mtl::sampler_liveness::contains(0ull));
	mtl::sampler_liveness::remove(0ull);
	EXPECT_FALSE(mtl::sampler_liveness::contains(0ull));
}

TEST(MetalSamplerLiveness, IdsAreIndependent)
{
	mtl::sampler_liveness::add(kFirstId);
	mtl::sampler_liveness::add(kSecondId);
	mtl::sampler_liveness::remove(kFirstId);
	EXPECT_FALSE(mtl::sampler_liveness::contains(kFirstId));
	EXPECT_TRUE(mtl::sampler_liveness::contains(kSecondId));
	mtl::sampler_liveness::remove(kSecondId);
	EXPECT_FALSE(mtl::sampler_liveness::contains(kSecondId));
}

// The 0x448/0x449 setSamplerState segfaults: an ID live at contains() time was
// destroyed on another thread before the driver call ran. bind() holds the guard
// across the call so destruction waits; these pin the guard contract.
namespace
{
	constexpr std::uint64_t kGuardId = 0x5a11'0003ull;
}

TEST(MetalSamplerLiveness, GuardOwnsLockForLiveId)
{
	mtl::sampler_liveness::add(kGuardId);

	// Note: no contains() while the guard is held (non-recursive mutex).
	EXPECT_TRUE(mtl::sampler_liveness::contains(kGuardId));

	{
		auto guard = mtl::sampler_liveness::retain_if_live(kGuardId);
		EXPECT_TRUE(guard.owns_lock());
	}

	mtl::sampler_liveness::remove(kGuardId);
}

TEST(MetalSamplerLiveness, GuardIsReleasedForDeadOrZeroId)
{
	EXPECT_FALSE(mtl::sampler_liveness::retain_if_live(kGuardId).owns_lock());
	EXPECT_FALSE(mtl::sampler_liveness::retain_if_live(0ull).owns_lock());
}

// The always-on creation log is bounded per process so a long session cannot
// flood the log: exactly the first N calls win, then false forever. Consumes
// the whole budget (nothing else in this process uses it); atomic ops only.
TEST(MetalSamplerLiveness, CreationTraceBudgetIsExact)
{
	for (unsigned i = 0; i != mtl::sampler_liveness::creation_trace_budget; i++)
	{
		EXPECT_TRUE(mtl::sampler_liveness::sampler_creation_trace_budget()) << "at take " << i;
	}

	EXPECT_FALSE(mtl::sampler_liveness::sampler_creation_trace_budget());
	EXPECT_FALSE(mtl::sampler_liveness::sampler_creation_trace_budget());
}

TEST(MetalSamplerLiveness, RemovalWaitsForGuardRelease)
{
	mtl::sampler_liveness::add(kGuardId);

	// The guard is taken first (as bind() does before the driver call); a remover
	// starting afterwards must block until it releases.
	auto guard = mtl::sampler_liveness::retain_if_live(kGuardId);
	ASSERT_TRUE(guard.owns_lock());

	std::atomic<bool> entered{false};
	std::atomic<bool> removed{false};

	std::thread remover([&]
	{
		entered = true;
		mtl::sampler_liveness::remove(kGuardId);
		removed = true;
	});

	// Bounded wait for the remover to reach remove() (it then blocks on the
	// held mutex); the join below always runs so a failure cannot terminate.
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (!entered && std::chrono::steady_clock::now() < deadline)
	{
		std::this_thread::yield();
	}
	EXPECT_TRUE(entered);

	// The remover is blocked inside remove(): it cannot complete (and the
	// sampler object cannot be destroyed) while the guard is held.
	// (No contains() here: the guard's mutex is non-recursive.)
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	EXPECT_FALSE(removed);

	guard.unlock();
	remover.join();
	EXPECT_TRUE(removed);
	EXPECT_FALSE(mtl::sampler_liveness::contains(kGuardId));
}

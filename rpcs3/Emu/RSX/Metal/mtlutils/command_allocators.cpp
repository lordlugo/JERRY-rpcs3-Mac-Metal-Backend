#include "stdafx.h"
#include "command_allocators.h"
#include "sync.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <vector>

namespace mtl::command_allocators
{
	namespace
	{
		constexpr u64 idle_release_us = 2'000'000;  // An allocator not handed out for this long is released
		constexpr u64 trim_interval_us = 1'000'000; // How often idle and ballooned allocators are looked for
		constexpr u64 balloon_ratio = 4;            // Ballooned: larger than this many times the median allocator ...
		constexpr u64 balloon_min_size = 32 << 20;  // ... and than this

		u64 now_us()
		{
			return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now().time_since_epoch()).count());
		}

		struct idle_allocator
		{
			MTL4::CommandAllocator* allocator = nullptr;
			u64 size = 0;      // allocatedSize() when handed back
			u64 last_used = 0; // now_us() when it became idle
		};

		struct used_allocator
		{
			MTL4::CommandAllocator* allocator = nullptr;
			u64 size = 0;              // allocatedSize() as last read
			const timeline* owner = nullptr; // In flight: reusable once `owner` reaches `value`
			u64 value = 0;
		};

		struct allocator_pool
		{
			std::mutex mutex;
			const MTL::Device* device = nullptr;

			// Sorted by last_used (every allocator that becomes idle is appended): the back is the most recently
			// used one, handed out first; the front ones are the first to be released when idle.
			std::vector<idle_allocator> idle;
			std::vector<used_allocator> recording;
			std::vector<used_allocator> in_flight;

			u64 last_trim = 0;
			u32 peak_in_use = 0;
			u32 created = 0;
			u32 released = 0;

			static void erase_allocator(std::vector<used_allocator>& list, const MTL4::CommandAllocator* allocator)
			{
				const auto found = std::find_if(list.begin(), list.end(), [allocator](const used_allocator& entry)
				{
					return entry.allocator == allocator;
				});

				ensure(found != list.end(), "Metal: command allocator not handed out by the pool");
				*found = list.back();
				list.pop_back();
			}

			// Resets the in-flight allocators whose submission has completed and makes them idle
			void collect_completed(u64 now)
			{
				// Completed values of the timelines seen by this pass (the RSX and present timelines)
				std::array<std::pair<const timeline*, u64>, 4> completed{};
				u32 timelines = 0;

				const auto completed_value = [&](const timeline* owner) -> u64
				{
					for (u32 i = 0; i < timelines; ++i)
					{
						if (completed[i].first == owner)
						{
							return completed[i].second;
						}
					}

					const u64 value = owner->completed_value();
					if (timelines < completed.size())
					{
						completed[timelines++] = { owner, value };
					}

					return value;
				};

				for (usz i = 0; i < in_flight.size();)
				{
					const used_allocator entry = in_flight[i];
					if (entry.value > completed_value(entry.owner))
					{
						++i;
						continue;
					}

					// The GPU has executed everything recorded with this allocator: its memory can be reused
					entry.allocator->reset();
					idle.push_back({ entry.allocator, entry.size, now });

					in_flight[i] = in_flight.back();
					in_flight.pop_back();
				}
			}

			// Moves the allocators to release into `out`
			void trim(u64 now, std::vector<MTL4::CommandAllocator*>& out)
			{
				// Idle for a while: more allocators than the recordings in flight needed recently (a burst of
				// submissions is over). The front of the list holds the least recently used ones.
				usz stale = 0;
				// (last_used may be a little later than `now`: threads read the clock before taking the lock)
				while (stale < idle.size() && idle[stale].last_used + idle_release_us <= now)
				{
					out.push_back(idle[stale++].allocator);
				}

				idle.erase(idle.begin(), idle.begin() + stale);

				// Ballooned: an allocator keeps the memory of its largest recording, and the most recently used one is
				// handed out first, so one exceptional recording would keep its memory in circulation. Release the
				// largest idle allocator if it is far larger than the median; if recordings that large are still common,
				// the allocator that serves the next one grows again (at most one reallocation per second).
				if (idle.empty())
				{
					return;
				}

				std::vector<u64> sizes;
				sizes.reserve(idle.size() + recording.size() + in_flight.size());
				for (const auto& entry : idle) sizes.push_back(entry.size);
				for (const auto& entry : recording) sizes.push_back(entry.size);
				for (const auto& entry : in_flight) sizes.push_back(entry.size);

				const auto median = sizes.begin() + sizes.size() / 2;
				std::nth_element(sizes.begin(), median, sizes.end());

				const auto largest = std::max_element(idle.begin(), idle.end(), [](const idle_allocator& a, const idle_allocator& b)
				{
					return a.size < b.size;
				});

				if (largest->size > balloon_min_size && largest->size > *median * balloon_ratio)
				{
					out.push_back(largest->allocator);
					idle.erase(largest);
				}
			}

			void note_in_use()
			{
				peak_in_use = std::max(peak_in_use, static_cast<u32>(recording.size() + in_flight.size()));
			}
		};

		// Never destroyed: lists of a renderer that failed to start may still hand allocators back at exit
		allocator_pool& get_pool()
		{
			static allocator_pool* s_pool = new allocator_pool();
			return *s_pool;
		}
	}

	MTL4::CommandAllocator* acquire(const render_device& dev)
	{
		auto& pool = get_pool();
		const u64 now = now_us();

		MTL4::CommandAllocator* result = nullptr;
		std::vector<MTL4::CommandAllocator*> to_release;
		{
			std::lock_guard lock(pool.mutex);
			ensure(!pool.device || pool.device == dev.handle(), "Metal: command allocators of another device are still alive");
			pool.device = dev.handle();

			pool.collect_completed(now);

			if (now >= pool.last_trim + trim_interval_us)
			{
				pool.last_trim = now;
				pool.trim(now, to_release);
				pool.released += ::size32(to_release);
			}

			if (!pool.idle.empty())
			{
				const idle_allocator entry = pool.idle.back();
				pool.idle.pop_back();

				result = entry.allocator;
				pool.recording.push_back({ entry.allocator, entry.size });
				pool.note_in_use();
			}
		}

		for (MTL4::CommandAllocator* allocator : to_release)
		{
			allocator->release();
		}

		if (result)
		{
			return result;
		}

		result = dev.handle()->newCommandAllocator();
		ensure(result, "Metal: failed to create command allocator");

		std::lock_guard lock(pool.mutex);
		pool.recording.push_back({ result, 0 });
		pool.created++;
		pool.note_in_use();
		return result;
	}

	void retire(MTL4::CommandAllocator* allocator, const timeline& tl, u64 value)
	{
		// The calling thread owns the allocator until it is in the pool (allocators are not thread safe)
		const u64 size = allocator->allocatedSize();

		auto& pool = get_pool();
		std::lock_guard lock(pool.mutex);
		allocator_pool::erase_allocator(pool.recording, allocator);
		pool.in_flight.push_back({ allocator, size, &tl, value });
	}

	void recycle(MTL4::CommandAllocator* allocator)
	{
		const u64 size = allocator->allocatedSize();
		allocator->reset();

		auto& pool = get_pool();
		std::lock_guard lock(pool.mutex);
		allocator_pool::erase_allocator(pool.recording, allocator);
		pool.idle.push_back({ allocator, size, now_us() });
	}

	void discard(MTL4::CommandAllocator* allocator)
	{
		{
			auto& pool = get_pool();
			std::lock_guard lock(pool.mutex);
			allocator_pool::erase_allocator(pool.recording, allocator);
			pool.released++;
		}

		allocator->release();
	}

	void release_all()
	{
		auto& pool = get_pool();
		std::lock_guard lock(pool.mutex);

		if (!pool.recording.empty())
		{
			// Lists that are still alive release theirs through discard()
			rsx_log.error("Metal: %u command allocator(s) still recording at renderer teardown", ::size32(pool.recording));
		}

		for (const auto& entry : pool.idle)
		{
			entry.allocator->release();
		}

		// The renderer waited for its queues (bounded); a submission that never completed is abandoned with them
		for (const auto& entry : pool.in_flight)
		{
			entry.allocator->release();
		}

		pool.idle.clear();
		pool.in_flight.clear();
		pool.device = nullptr;
	}

	command_allocator_stats_t get_stats_and_reset()
	{
		auto& pool = get_pool();
		std::lock_guard lock(pool.mutex);

		command_allocator_stats_t stats{};
		for (const auto& entry : pool.idle) stats.allocated_bytes += entry.size;
		for (const auto& entry : pool.recording) stats.allocated_bytes += entry.size;
		for (const auto& entry : pool.in_flight) stats.allocated_bytes += entry.size;

		stats.allocators = ::size32(pool.idle) + ::size32(pool.recording) + ::size32(pool.in_flight);
		stats.peak_in_use = std::exchange(pool.peak_in_use, ::size32(pool.recording) + ::size32(pool.in_flight));
		stats.created = std::exchange(pool.created, 0);
		stats.released = std::exchange(pool.released, 0);
		return stats;
	}
}

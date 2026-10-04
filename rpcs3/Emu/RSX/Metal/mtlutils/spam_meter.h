#pragma once

// Rate-limits a hot error line (e.g. the Metal CB-chain "out of free entries"
// flood: 1126 identical lines in 60 s bury every other signal, including the
// errors that would name the real failure). The first burst passes so a brief
// episode stays fully visible, then one summary per window. Thread-safe and
// dependency-free so unit tests can use it without Metal headers.
#include <atomic>

namespace mtl
{
	struct spam_meter
	{
		static constexpr unsigned burst = 32;
		static constexpr unsigned window = 1024;

		// True when the caller should log now. count() reports the running total.
		bool allow()
		{
			const unsigned n = m_count.fetch_add(1, std::memory_order_relaxed);
			return n < burst || (n % window) == 0;
		}

		unsigned count() const
		{
			return m_count.load(std::memory_order_relaxed);
		}

	private:
		std::atomic<unsigned> m_count{0};
	};
}

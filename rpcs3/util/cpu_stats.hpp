#pragma once

#include "util/types.hpp"
#include <string>
#include <vector>

#ifdef _WIN32
#include <pdh.h>
#include <pdhmsg.h>
#endif

namespace utils
{
	class cpu_stats
	{
		u64 m_last_cpu = 0;
		u64 m_sys_cpu = 0;
		u64 m_usr_cpu = 0;

#ifdef _WIN32
		PDH_HQUERY m_cpu_query = nullptr;
		PDH_HCOUNTER m_cpu_cores = nullptr;
#elif __linux__
		size_t m_previous_idle_time_total = 0;
		size_t m_previous_total_time_total = 0;
		std::vector<size_t> m_previous_idle_times_per_cpu;
		std::vector<size_t> m_previous_total_times_per_cpu;
#elif defined(__APPLE__)
		std::vector<u32> m_previous_busy_ticks_per_cpu;
		std::vector<u32> m_previous_total_ticks_per_cpu;

		// get_usage(): whole-system CPU load from host_statistics(HOST_CPU_LOAD_INFO), smoothed like Activity Monitor
		// style monitors (Redline): previous user/system/idle/nice ticks, last value, EMA state
		u32 m_prev_load_ticks[4]{};
		bool m_have_prev_load = false;
		double m_load_ema = -1.;
		u64 m_load_last_ns = 0;
#endif

	public:
		cpu_stats();
		~cpu_stats();

		double get_usage();

		void init_cpu_query();
		void get_per_core_usage(std::vector<double>& per_core_usage, double& total_usage);

		static u32 get_current_thread_count();

		// Number of efficiency cores, which have the lowest CPU numbers (Apple silicon); 0: unknown or not a hybrid CPU
		static u32 get_efficiency_core_count();

		struct thread_cpu_time
		{
			u64 id = 0;       // Unique system thread id
			u64 time_ns = 0;  // User + system CPU time since the thread started
			std::string name; // OS thread name (empty: unnamed)
		};

		// CPU time of every thread of this process (implemented on macOS; false elsewhere)
		static bool get_thread_cpu_times(std::vector<thread_cpu_time>& threads);
	};
}

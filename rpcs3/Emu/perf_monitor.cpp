#include "stdafx.h"
#include "perf_monitor.hpp"

#include "Emu/System.h"
#include "Emu/Cell/timers.hpp"
#include "Loader/ISO.h"
#include "util/cpu_stats.hpp"
#include "util/sysinfo.hpp"
#include "Utilities/Thread.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <unordered_map>

#ifdef __APPLE__
#include <libkern/OSThermalNotification.h>
#include <notify.h>
#endif

LOG_CHANNEL(perf_log, "PERF");

namespace
{
#ifdef __APPLE__
	// macOS thermal pressure level: notify(3) state of kOSThermalNotificationPressureLevelName (spelled out, as documented
	// in <libkern/OSThermalNotification.h>, so that nothing depends on that symbol being exported), OSThermalPressureLevel
	class thermal_pressure_monitor
	{
		int m_token = -1;
		s64 m_level = -1;

	public:
		thermal_pressure_monitor()
		{
			if (const u32 status = notify_register_check("com.apple.system.thermalpressurelevel", &m_token); status != NOTIFY_STATUS_OK)
			{
				perf_log.warning("Thermal pressure level is not available (notify_register_check: %u)", status);
				m_token = -1;
			}
		}

		~thermal_pressure_monitor()
		{
			if (m_token != -1)
			{
				notify_cancel(m_token);
			}
		}

		thermal_pressure_monitor(const thermal_pressure_monitor&) = delete;
		thermal_pressure_monitor& operator=(const thermal_pressure_monitor&) = delete;

		// Cheap unless the level was posted since the last call. Logs every change as it is seen (every half second).
		void update()
		{
			int posted = 0;

			if (m_token == -1 || notify_check(m_token, &posted) != NOTIFY_STATUS_OK || !posted)
			{
				return;
			}

			u64 state = 0;

			if (notify_get_state(m_token, &state) != NOTIFY_STATUS_OK || static_cast<s64>(state) == m_level)
			{
				return;
			}

			if (m_level == -1)
			{
				perf_log.notice("Thermal pressure: %s", level_name(state));
			}
			else
			{
				perf_log.notice("Thermal pressure changed: %s -> %s", level_name(static_cast<u64>(m_level)), level_name(state));
			}

			m_level = static_cast<s64>(state);
		}

		std::string get() const
		{
			return m_level == -1 ? std::string("unknown") : level_name(static_cast<u64>(m_level));
		}

		static std::string level_name(u64 level)
		{
			switch (level)
			{
			case kOSThermalPressureLevelNominal: return "nominal";
			case kOSThermalPressureLevelModerate: return "moderate";
			case kOSThermalPressureLevelHeavy: return "heavy";
			case kOSThermalPressureLevelTrapping: return "trapping";
			case kOSThermalPressureLevelSleeping: return "sleeping";
			default: return fmt::format("level %u", level);
			}
		}
	};
#endif

#if defined(__APPLE__) && defined(ARCH_ARM64)
	// Effective clock (GHz) of the core this thread runs on: dependent integer adds retire one per cycle on every Apple
	// core. Best of a few ~60 us runs after 1 ms of warm-up (the cluster clock ramps up under load). Its clusters run at
	// most ~3.2 GHz (P-cores) and ~2.1 GHz (E-cores) on an M1 Max: a value that sinks over time means thermal or power
	// throttling (or that the probe ran on an E-core).
	f64 probe_cpu_clock_ghz()
	{
		constexpr u64 adds_per_iteration = 100;
		constexpr u64 iterations = 2000;

		const auto run = [](u64 count)
		{
			u64 x = 0;
			const u64 y = 1;
			const auto start = std::chrono::steady_clock::now();

			for (u64 i = 0; i < count; i++)
			{
				asm volatile(".rept 100\n\tadd %0, %0, %1\n\t.endr" : "+r"(x) : "r"(y));
			}

			return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
		};

		for (const auto warm_up_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(1); std::chrono::steady_clock::now() < warm_up_end;)
		{
			run(iterations / 4);
		}

		u64 best_ns = umax;

		for (u32 i = 0; i < 5; i++)
		{
			best_ns = std::min(best_ns, run(iterations));
		}

		return best_ns ? static_cast<f64>(adds_per_iteration * iterations) / static_cast<f64>(best_ns) : 0.;
	}
#endif

	// Per-thread CPU usage between two calls: the busiest threads by name (threads that share a name are summed up)
	class thread_usage_sampler
	{
		std::unordered_map<u64, u64> m_last_times; // Thread id -> CPU time (ns)
		u64 m_last_sample_time = 0;                 // get_system_time()
		std::vector<utils::cpu_stats::thread_cpu_time> m_threads;

	public:
		static constexpr u64 interval_us = 30'000'000;

		bool is_due(u64 now) const
		{
			return !m_last_sample_time || now - m_last_sample_time >= interval_us;
		}

		// Returns an empty string for the first sample (or if the platform does not provide thread times)
		std::string sample(u64 now)
		{
			if (!utils::cpu_stats::get_thread_cpu_times(m_threads))
			{
				m_last_sample_time = now;
				return {};
			}

			const u64 window_us = now - m_last_sample_time;
			const bool first = !m_last_sample_time;
			m_last_sample_time = now;

			std::unordered_map<u64, u64> times;
			std::map<std::string, std::pair<u64, u32>> by_name; // CPU time (ns) in the window, thread count
			u64 total_ns = 0;

			for (const auto& thread : m_threads)
			{
				times.emplace(thread.id, thread.time_ns);

				// A thread that started after the previous sample used all of its CPU time in this window
				const auto found = m_last_times.find(thread.id);
				const u64 delta = found == m_last_times.end() ? thread.time_ns : thread.time_ns - std::min(thread.time_ns, found->second);

				auto& entry = by_name[thread.name.empty() ? std::string("(unnamed)") : thread.name];
				entry.first += delta;
				entry.second++;
				total_ns += delta;
			}

			m_last_times = std::move(times);

			if (first || !window_us)
			{
				return {};
			}

			std::vector<std::pair<std::string, std::pair<u64, u32>>> sorted(by_name.begin(), by_name.end());
			std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });

			const auto percent = [window_us](u64 ns) { return ns / 10. / static_cast<f64>(window_us); };

			std::string msg = fmt::format("Threads over %.0fs (CPU %% of one core, threads that exited in between are missing):", window_us / 1'000'000.);
			usz shown = 0;

			for (const auto& [name, usage] : sorted)
			{
				if (shown == 16 || percent(usage.first) < 1.)
				{
					break;
				}

				fmt::append(msg, "%s %s", shown++ ? "," : "", name);

				if (usage.second > 1)
				{
					fmt::append(msg, " (%u threads)", usage.second);
				}

				fmt::append(msg, " %.1f%%", percent(usage.first));
			}

			fmt::append(msg, "; %u threads, %.1f%% in total", m_threads.size(), percent(total_ns));
			return msg;
		}
	};
}

void perf_monitor::operator()()
{
	constexpr u64 update_interval_us = 500000; // Update every half second
	constexpr u64 log_interval_us_max = 10000000; // Log at minimum every 10 seconds
	constexpr u64 log_interval_us_min = 500000;  // Log at maximum every half a second (catching possible memory leak)
	constexpr u64 log_mem_increase = 50 * (1024 * 1024);  // Log when memory usage increased by this amount

	u64 elapsed_us = 0;

	utils::cpu_stats stats;
	stats.init_cpu_query();

	u32 logged_pause = 0;
	u64 last_pause_time = umax;
	u64 max_memory_usage = 0;

	std::vector<double> per_core_usage;
	std::string msg;

	// The logged usage is the average of the samples since the previous log line
	std::vector<double> per_core_usage_sum;
	double total_usage_sum = 0.0;
	u32 usage_samples = 0;

	// Hybrid CPUs (Apple silicon): the efficiency cores have the lowest CPU numbers
	const u32 efficiency_cores = utils::cpu_stats::get_efficiency_core_count();

	thread_usage_sampler thread_usage;

#ifdef __APPLE__
	thermal_pressure_monitor thermal_pressure;
	thermal_pressure.update();
#endif

	for (u64 sleep_until = get_system_time();;)
	{
		thread_ctrl::wait_until(&sleep_until, update_interval_us);
		elapsed_us += update_interval_us;

#ifdef __APPLE__
		thermal_pressure.update();
#endif

		double total_usage = 0.0;

		stats.get_per_core_usage(per_core_usage, total_usage);

		per_core_usage_sum.resize(per_core_usage.size());

		for (usz i = 0; i < per_core_usage.size(); i++)
		{
			per_core_usage_sum[i] += per_core_usage[i];
		}

		total_usage_sum += total_usage;
		usage_samples++;

		const u64 current_mem_use = utils::get_memory_usage().second;
		const u64 mem_use_increase = current_mem_use >= max_memory_usage ? current_mem_use - max_memory_usage : 0;

		const u64 log_interval = (mem_use_increase >= log_mem_increase ? log_interval_us_min : log_interval_us_max);

		if (elapsed_us >= log_interval || thread_ctrl::state() == thread_state::aborting)
		{
			max_memory_usage = std::max<u64>(current_mem_use, max_memory_usage);
			elapsed_us = 0;

			// Average over the interval
			total_usage = total_usage_sum / usage_samples;

			for (usz i = 0; i < per_core_usage.size(); i++)
			{
				per_core_usage[i] = per_core_usage_sum[i] / usage_samples;
			}

			std::fill(per_core_usage_sum.begin(), per_core_usage_sum.end(), 0.0);
			total_usage_sum = 0.0;
			usage_samples = 0;

			const bool is_paused = Emu.IsPaused();
			const u64 pause_time = Emu.GetPauseTime();

			if (!is_paused || last_pause_time != pause_time)
			{
				// Resumed or not paused since last check
				logged_pause = 0;
				last_pause_time = pause_time;
			}

			if (is_paused)
			{
				if (logged_pause >= 2)
				{
					// Let's not spam the log when emulation is paused
					// But still emit the message two times so even paused state can be debugged and inspected
					continue;
				}

				logged_pause++;
			}

			msg.clear();
			fmt::append(msg, "CPU Usage: Total: %.1f%%", total_usage);

			if (!per_core_usage.empty())
			{
				fmt::append(msg, ", Cores:");
			}

			for (usz i = 0; i < per_core_usage.size(); i++)
			{
				fmt::append(msg, "%s %.1f%%", i > 0 ? "," : "", per_core_usage[i]);
			}

			if (efficiency_cores && efficiency_cores < per_core_usage.size())
			{
				const auto average = [&](usz begin, usz end)
				{
					double sum = 0.0;

					for (usz i = begin; i < end; i++)
					{
						sum += per_core_usage[i];
					}

					return sum / static_cast<double>(end - begin);
				};

				fmt::append(msg, " (E-cores CPU 0-%u: %.1f%%, P-cores CPU %u-%u: %.1f%%)", efficiency_cores - 1, average(0, efficiency_cores),
					efficiency_cores, per_core_usage.size() - 1, average(efficiency_cores, per_core_usage.size()));
			}

#ifdef __APPLE__
			fmt::append(msg, ", Thermal pressure: %s", thermal_pressure.get());
#endif
#if defined(__APPLE__) && defined(ARCH_ARM64)
			fmt::append(msg, ", Clock probe: %.2f GHz", probe_cpu_clock_ghz());
#endif

			if (max_memory_usage)
			{
				fmt::append(msg, ", RAM Usage: %dMB (Peak: %dMB)", current_mem_use / (1024 * 1024), max_memory_usage / (1024 * 1024));
			}

			perf_log.notice("%s", msg);

			// Where the CPU time goes, every 30 seconds
			if (const u64 now = get_system_time(); thread_usage.is_due(now))
			{
				if (const std::string threads = thread_usage.sample(now); !threads.empty())
				{
					perf_log.notice("%s", threads);
				}

				// Disc reads over the same period (the first report also covers the game list and the boot)
				if (const std::string disc = get_iso_read_stats(); !disc.empty())
				{
					perf_log.notice("%s", disc);
				}
			}

			if (thread_ctrl::state() == thread_state::aborting)
			{
				// Log once before terminating
				break;
			}
		}
	}
}

perf_monitor::~perf_monitor()
{
}

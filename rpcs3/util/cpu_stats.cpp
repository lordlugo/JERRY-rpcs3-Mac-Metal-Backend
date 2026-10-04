#include "util/types.hpp"
#include "util/cpu_stats.hpp"
#include "util/sysinfo.hpp"
#include "util/logs.hpp"
#include "Utilities/StrUtil.h"

#include <algorithm>
#include <cmath>

#ifdef _WIN32
#include "util/asm.hpp"
#include "windows.h"
#include "tlhelp32.h"
#ifdef _MSC_VER
#pragma comment(lib, "pdh.lib")
#endif
#else
#include "fstream"
#include "sstream"
#include "stdlib.h"
#include "sys/times.h"
#endif

#ifdef __APPLE__
# include <mach/mach_init.h>
# include <mach/mach_host.h>
# include <mach/mach_port.h>
# include <mach/processor_info.h>
# include <mach/task.h>
# include <mach/thread_act.h>
# include <mach/thread_info.h>
# include <mach/vm_map.h>
# include <sys/sysctl.h>
# include <time.h>
#endif

#ifdef __linux__
# include <dirent.h>
#endif

#if defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
# include <sys/sysctl.h>
# include <unistd.h>
# if defined(__DragonFly__) || defined(__FreeBSD__)
#  include <sys/user.h>
# endif

# if defined(__NetBSD__)
#  undef KERN_PROC
#  define KERN_PROC KERN_PROC2
#  define kinfo_proc kinfo_proc2
# endif

# if defined(__DragonFly__)
#  define KP_NLWP(kp) (kp.kp_nthreads)
# elif defined(__FreeBSD__)
#  define KP_NLWP(kp) (kp.ki_numthreads)
# elif defined(__NetBSD__)
#  define KP_NLWP(kp) (kp.p_nlwps)
# endif
#endif

LOG_CHANNEL(perf_log, "PERF");

namespace utils
{
#ifdef __APPLE__
	// Busy and total scheduler ticks of every CPU since boot (u32 counters that wrap)
	static bool get_cpu_ticks(std::vector<u32>& busy, std::vector<u32>& total)
	{
		static const host_t host = mach_host_self();

		natural_t cpu_count = 0;
		processor_info_array_t info = nullptr;
		mach_msg_type_number_t info_count = 0;

		if (host_processor_info(host, PROCESSOR_CPU_LOAD_INFO, &cpu_count, &info, &info_count) != KERN_SUCCESS)
		{
			return false;
		}

		const auto load = reinterpret_cast<const processor_cpu_load_info*>(info);
		busy.resize(cpu_count);
		total.resize(cpu_count);

		for (natural_t i = 0; i < cpu_count; i++)
		{
			const auto& ticks = load[i].cpu_ticks;
			busy[i] = ticks[CPU_STATE_USER] + ticks[CPU_STATE_SYSTEM] + ticks[CPU_STATE_NICE];
			total[i] = busy[i] + ticks[CPU_STATE_IDLE];
		}

		vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(info), info_count * sizeof(integer_t));
		return true;
	}
#endif

#ifdef _WIN32
	fmt::win_error pdh_error(PDH_STATUS status)
	{
		return fmt::win_error{static_cast<unsigned long>(status), LoadLibrary(L"pdh.dll")};
	}
#endif

	cpu_stats::cpu_stats()
	{
#ifdef _WIN32
		FILETIME ftime, fsys, fuser;

		GetSystemTimeAsFileTime(&ftime);
		memcpy(&m_last_cpu, &ftime, sizeof(FILETIME));

		GetProcessTimes(GetCurrentProcess(), &ftime, &ftime, &fsys, &fuser);
		memcpy(&m_sys_cpu, &fsys, sizeof(FILETIME));
		memcpy(&m_usr_cpu, &fuser, sizeof(FILETIME));
#else
		struct tms timeSample;

		m_last_cpu = times(&timeSample);
		m_sys_cpu  = timeSample.tms_stime;
		m_usr_cpu  = timeSample.tms_utime;
#endif
	}

	cpu_stats::~cpu_stats()
	{
#ifdef _WIN32
		if (m_cpu_query)
		{
			PDH_STATUS status = PdhCloseQuery(m_cpu_query);
			if (ERROR_SUCCESS != status)
			{
				perf_log.error("Failed to close cpu query of per core cpu usage: %s", pdh_error(status));
			}
		}
#endif
	}

	void cpu_stats::init_cpu_query()
	{
#ifdef _WIN32
		PDH_STATUS status = PdhOpenQuery(NULL, 0, &m_cpu_query);
		if (ERROR_SUCCESS != status)
		{
			perf_log.error("Failed to open cpu query for per core cpu usage: %s", pdh_error(status));
			return;
		}
		status = PdhAddEnglishCounter(m_cpu_query, L"\\Processor(*)\\% Processor Time", 0, &m_cpu_cores);
		if (ERROR_SUCCESS != status)
		{
			perf_log.error("Failed to add processor time counter for per core cpu usage: %s", pdh_error(status));
			return;
		}
		status = PdhCollectQueryData(m_cpu_query);
		if (ERROR_SUCCESS != status)
		{
			perf_log.error("Failed to collect per core cpu usage: %s", pdh_error(status));
			return;
		}
#elif defined(__APPLE__)
		get_cpu_ticks(m_previous_busy_ticks_per_cpu, m_previous_total_ticks_per_cpu);
#endif
	}

	void cpu_stats::get_per_core_usage(std::vector<double>& per_core_usage, double& total_usage)
	{
		total_usage = 0.0;

		per_core_usage.resize(utils::get_thread_count());
		std::fill(per_core_usage.begin(), per_core_usage.end(), 0.0);

#ifdef _WIN32
		if (!m_cpu_cores || !m_cpu_query)
		{
			perf_log.warning("Can not collect per core cpu usage: The required API is not initialized.");
			return;
		}

		PDH_STATUS status = PdhCollectQueryData(m_cpu_query);
		if (ERROR_SUCCESS != status)
		{
			perf_log.error("Failed to collect per core cpu usage: %s", pdh_error(status));
			return;
		}

		DWORD dwBufferSize = 0; // Size of the items buffer
		DWORD dwItemCount = 0;  // Number of items in the items buffer

		status = PdhGetFormattedCounterArray(m_cpu_cores, PDH_FMT_DOUBLE, &dwBufferSize, &dwItemCount, nullptr);
		if (static_cast<PDH_STATUS>(PDH_MORE_DATA) == status)
		{
			std::vector<PDH_FMT_COUNTERVALUE_ITEM> items(utils::aligned_div(dwBufferSize, sizeof(PDH_FMT_COUNTERVALUE_ITEM)));
			if (items.size() >= dwItemCount)
			{
				status = PdhGetFormattedCounterArray(m_cpu_cores, PDH_FMT_DOUBLE, &dwBufferSize, &dwItemCount, items.data());
				if (ERROR_SUCCESS == status)
				{
					ensure(dwItemCount == per_core_usage.size() + 1); // Plus one for _Total

					// Loop through the array and get the instance name and percentage.
					for (usz i = 0; i < dwItemCount; i++)
					{
						const PDH_FMT_COUNTERVALUE_ITEM& item = items[i];
						const std::string token = wchar_to_utf8(item.szName);

						if (const std::string lower = fmt::to_lower(token); lower.find("total") != umax)
						{
							total_usage = item.FmtValue.doubleValue;
							continue;
						}

						if (const auto [success, cpu_index] = string_to_number(token); success && cpu_index < dwItemCount)
						{
							per_core_usage[cpu_index] = item.FmtValue.doubleValue;
						}
						else if (!success)
						{
							perf_log.error("Can not convert string to cpu index for per core cpu usage. (token='%s')", token);
						}
						else
						{
							perf_log.error("Invalid cpu index for per core cpu usage. (token='%s', cpu_index=%d, cores=%d)", token, cpu_index, dwItemCount);
						}
					}
				}
				else if (static_cast<PDH_STATUS>(PDH_CALC_NEGATIVE_DENOMINATOR) == status) // Apparently this is a common uncritical error
				{
					perf_log.notice("Failed to get per core cpu usage: %s", pdh_error(status));
				}
				else
				{
					perf_log.error("Failed to get per core cpu usage: %s", pdh_error(status));
				}
			}
			else
			{
				perf_log.error("Failed to allocate buffer for per core cpu usage. (size=%d, dwItemCount=%d)", items.size(), dwItemCount);
			}
		}

#elif __linux__
#ifndef ANDROID
		m_previous_idle_times_per_cpu.resize(utils::get_thread_count(), 0.0);
		m_previous_total_times_per_cpu.resize(utils::get_thread_count(), 0.0);

		if (std::ifstream proc_stat("/proc/stat"); proc_stat.good())
		{
			std::stringstream content;
			content << proc_stat.rdbuf();
			proc_stat.close();

			const std::string content_str = content.str();
			const std::vector<std::string_view> lines = fmt::split_sv(content_str, {"\n"});
			if (lines.empty())
			{
				perf_log.error("/proc/stat is empty");
				return;
			}

			for (const std::string_view& line : lines)
			{
				const std::vector<std::string_view> tokens = fmt::split_sv(line, {" "});
				if (tokens.size() < 5)
				{
					return;
				}

				const std::string_view token = tokens[0];
				if (!token.starts_with("cpu"))
				{
					return;
				}

				// Get CPU index
				int cpu_index = -1; // -1 for total

				constexpr size_t size_of_cpu = 3;
				if (token.size() > size_of_cpu)
				{
					if (const auto [success, val] = string_to_number(token.substr(size_of_cpu)); success && val < per_core_usage.size())
					{
						cpu_index = val;
					}
					else if (!success)
					{
						perf_log.error("Can not convert string to cpu index for per core cpu usage. (token='%s', line='%s')", token, line);
						continue;
					}
					else
					{
						perf_log.error("Invalid cpu index for per core cpu usage. (cpu_index=%d, cores=%d, token='%s', line='%s')", cpu_index, per_core_usage.size(), token, line);
						continue;
					}
				}

				size_t idle_time = 0;
				size_t total_time = 0;

				for (size_t i = 1; i < tokens.size(); i++)
				{
					if (const auto [success, val] = string_to_number(tokens[i]); success)
					{
						if (i == 4)
						{
							idle_time = val;
						}

						total_time += val;
					}
					else
					{
						perf_log.error("Can not convert string to time for per core cpu usage. (i=%d, token='%s', line='%s')", i, tokens[i], line);
					}
				}

				if (cpu_index < 0)
				{
					const double idle_time_delta = idle_time - std::exchange(m_previous_idle_time_total, idle_time);
					const double total_time_delta = total_time - std::exchange(m_previous_total_time_total, total_time);
					total_usage = 100.0 * (1.0 - idle_time_delta / total_time_delta);
				}
				else
				{
					const double idle_time_delta = idle_time - std::exchange(m_previous_idle_times_per_cpu[cpu_index], idle_time);
					const double total_time_delta = total_time - std::exchange(m_previous_total_times_per_cpu[cpu_index], total_time);
					per_core_usage[cpu_index] = 100.0 * (1.0 - idle_time_delta / total_time_delta);
				}
			}
		}
		else
		{
			perf_log.error("Failed to open /proc/stat (%s)", strerror(errno));
		}
#endif
#elif defined(__APPLE__)
		// Load of every CPU (all processes) since the previous call. Total: this process (get_usage), as before
		if (std::vector<u32> busy, total; get_cpu_ticks(busy, total))
		{
			m_previous_busy_ticks_per_cpu.resize(busy.size());
			m_previous_total_ticks_per_cpu.resize(total.size());

			for (usz i = 0; i < busy.size() && i < per_core_usage.size(); i++)
			{
				const u32 busy_delta = busy[i] - std::exchange(m_previous_busy_ticks_per_cpu[i], busy[i]);
				const u32 total_delta = total[i] - std::exchange(m_previous_total_ticks_per_cpu[i], total[i]);
				per_core_usage[i] = total_delta ? std::min(100.0, 100.0 * busy_delta / total_delta) : 0.0;
			}
		}

		total_usage = get_usage();
#else
		total_usage = get_usage();
#endif
	}

	double cpu_stats::get_usage()
	{
#ifdef __APPLE__
		// RPCS3 Metal fork: whole-system CPU load, measured the way Redline (mac-resource-monitor) does it: the
		// aggregate scheduler ticks of host_statistics(HOST_CPU_LOAD_INFO), (user + system + nice) / (those + idle)
		// since the previous call, with wrapping 32-bit deltas. The process-time estimate below (times(), 10 ms
		// resolution, divided by the logical CPU count) read low and jumpy next to Activity Monitor. Smoothed with
		// a time-scaled EMA (gain 0.4 per second) so the overlay neither lags nor jitters at any refresh rate.
		{
			static const host_t host = mach_host_self();

			host_cpu_load_info_data_t info{};
			mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;

			if (host_statistics(host, HOST_CPU_LOAD_INFO, reinterpret_cast<host_info_t>(&info), &count) == KERN_SUCCESS && count >= HOST_CPU_LOAD_INFO_COUNT)
			{
				const u32 now[4] =
				{
					static_cast<u32>(info.cpu_ticks[CPU_STATE_USER]),
					static_cast<u32>(info.cpu_ticks[CPU_STATE_SYSTEM]),
					static_cast<u32>(info.cpu_ticks[CPU_STATE_IDLE]),
					static_cast<u32>(info.cpu_ticks[CPU_STATE_NICE]),
				};

				const u64 now_ns = static_cast<u64>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));

				if (m_have_prev_load)
				{
					const u64 used = u64{static_cast<u32>(now[0] - m_prev_load_ticks[0])} + u64{static_cast<u32>(now[1] - m_prev_load_ticks[1])} +
						u64{static_cast<u32>(now[3] - m_prev_load_ticks[3])};
					const u64 total = used + u64{static_cast<u32>(now[2] - m_prev_load_ticks[2])};

					if (total)
					{
						const double sample = 100. * static_cast<double>(used) / static_cast<double>(total);

						if (m_load_ema < 0.)
						{
							m_load_ema = sample;
						}
						else
						{
							const double elapsed_s = m_load_last_ns && now_ns > m_load_last_ns ? static_cast<double>(now_ns - m_load_last_ns) / 1e9 : 1.;
							const double alpha = -std::expm1(std::log(0.6) * elapsed_s);
							m_load_ema += alpha * (sample - m_load_ema);
						}
					}
				}

				std::copy(std::begin(now), std::end(now), std::begin(m_prev_load_ticks));
				m_have_prev_load = true;
				m_load_last_ns = now_ns;

				// First call only sets the baseline; until a delta exists report 0 rather than a made-up value
				return std::clamp(m_load_ema < 0. ? 0. : m_load_ema, 0., 100.);
			}

			// host_statistics failed: hold the last good value if there is one, else fall back to the process estimate
			if (m_load_ema >= 0.)
			{
				return std::clamp(m_load_ema, 0., 100.);
			}
		}
#endif

#ifdef _WIN32
		FILETIME ftime, fsys, fusr;
		ULARGE_INTEGER now, sys, usr;
		double percent;

		GetSystemTimeAsFileTime(&ftime);
		memcpy(&now, &ftime, sizeof(FILETIME));

		GetProcessTimes(GetCurrentProcess(), &ftime, &ftime, &fsys, &fusr);
		memcpy(&sys, &fsys, sizeof(FILETIME));
		memcpy(&usr, &fusr, sizeof(FILETIME));

		if (now.QuadPart <= m_last_cpu || sys.QuadPart < m_sys_cpu || usr.QuadPart < m_usr_cpu)
		{
			// Overflow detection. Just skip this value.
			percent = 0.0;
		}
		else
		{
			percent = static_cast<double>((sys.QuadPart - m_sys_cpu) + (usr.QuadPart - m_usr_cpu));
			percent /= (now.QuadPart - m_last_cpu);
			percent /= utils::get_thread_count(); // Let's assume this is at least 1
			percent *= 100;
		}

		m_last_cpu = now.QuadPart;
		m_usr_cpu  = usr.QuadPart;
		m_sys_cpu  = sys.QuadPart;

		return std::clamp(percent, 0.0, 100.0);
#else
		struct tms timeSample;
		clock_t now = times(&timeSample);
		double percent;

		if (now <= static_cast<clock_t>(m_last_cpu) || timeSample.tms_stime < static_cast<clock_t>(m_sys_cpu) || timeSample.tms_utime < static_cast<clock_t>(m_usr_cpu))
		{
			// Overflow detection. Just skip this value.
			percent = 0.0;
		}
		else
		{
			percent = (timeSample.tms_stime - m_sys_cpu) + (timeSample.tms_utime - m_usr_cpu);
			percent /= (now - m_last_cpu);
			percent /= utils::get_thread_count();
			percent *= 100;
		}
		m_last_cpu = now;
		m_sys_cpu  = timeSample.tms_stime;
		m_usr_cpu  = timeSample.tms_utime;

		return std::clamp(percent, 0.0, 100.0);
#endif
	}

	u32 cpu_stats::get_current_thread_count() // static
	{
#ifdef _WIN32
		// first determine the id of the current process
		const DWORD id = GetCurrentProcessId();

		// then get a process list snapshot.
		const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

		// initialize the process entry structure.
		PROCESSENTRY32 entry = {0};
		entry.dwSize         = sizeof(entry);

		// get the first process info.
		BOOL ret = Process32First(snapshot, &entry);
		while (ret && entry.th32ProcessID != id)
		{
			ret = Process32Next(snapshot, &entry);
		}
		CloseHandle(snapshot);
		return ret ? entry.cntThreads : 0;
#elif defined(__APPLE__)
		const task_t task = mach_task_self();
		mach_msg_type_number_t thread_count;
		thread_act_array_t thread_list;
		if (task_threads(task, &thread_list, &thread_count) != KERN_SUCCESS)
		{
			return 0;
		}
		for (mach_msg_type_number_t i = 0; i < thread_count; i++)
		{
			// task_threads() hands out a send right per thread
			mach_port_deallocate(task, thread_list[i]);
		}
		vm_deallocate(task, reinterpret_cast<vm_address_t>(thread_list),
			      sizeof(thread_t) * thread_count);
		return static_cast<u32>(thread_count);
#elif defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__)
		int mib[] = {
			CTL_KERN,
			KERN_PROC,
			KERN_PROC_PID,
			getpid(),
#if defined(__NetBSD__)
			sizeof(struct kinfo_proc),
			1,
#endif
		};
		u_int miblen = std::size(mib);
		struct kinfo_proc info;
		usz size = sizeof(info);
		if (sysctl(mib, miblen, &info, &size, NULL, 0))
		{
			return 0;
		}
		return KP_NLWP(info);
#elif defined(__OpenBSD__)
		int mib[] = {
			CTL_KERN,
			KERN_PROC,
			KERN_PROC_PID | KERN_PROC_SHOW_THREADS,
			getpid(),
			sizeof(struct kinfo_proc),
			0,
		};
		u_int miblen = std::size(mib);

		// get number of structs
		usz size;
		if (sysctl(mib, miblen, NULL, &size, NULL, 0))
		{
			return 0;
		}
		mib[5] = size / mib[4];

		// populate array of structs
		struct kinfo_proc info[mib[5]];
		if (sysctl(mib, miblen, &info, &size, NULL, 0))
		{
			return 0;
		}

		// exclude empty members
		u32 thread_count{0};
		for (int i = 0; i < size / mib[4]; i++)
		{
			if (info[i].p_tid != -1)
				++thread_count;
		}
		return thread_count;
#elif defined(__linux__)
		u32 thread_count{0};

		DIR* proc_dir = opendir("/proc/self/task");
		if (proc_dir)
		{
			// proc available, iterate through tasks and count them
			const struct dirent* entry;
			while ((entry = readdir(proc_dir)) != NULL)
			{
				if (entry->d_name[0] == '.')
					continue;

				++thread_count;
			}

			closedir(proc_dir);
		}
		return thread_count;
#else
		// unimplemented
		return 0;
#endif
	}

	u32 cpu_stats::get_efficiency_core_count() // static
	{
#ifdef __APPLE__
		// Apple silicon: performance level 0 = P-cores, 1 = E-cores. The E-cores have the lowest CPU numbers (the kernel
		// numbers the CPUs in device tree order, efficiency cluster first; M1 Pro/Max: CPU 0-1)
		int levels = 0;
		usz size = sizeof(levels);

		if (sysctlbyname("hw.nperflevels", &levels, &size, nullptr, 0) != 0 || levels < 2)
		{
			return 0;
		}

		int count = 0;
		size = sizeof(count);

		if (sysctlbyname("hw.perflevel1.logicalcpu", &count, &size, nullptr, 0) != 0 || count <= 0)
		{
			return 0;
		}

		return static_cast<u32>(count);
#else
		return 0;
#endif
	}

	bool cpu_stats::get_thread_cpu_times(std::vector<thread_cpu_time>& threads) // static
	{
		threads.clear();

#ifdef __APPLE__
		const task_t task = mach_task_self();
		thread_act_array_t thread_list = nullptr;
		mach_msg_type_number_t thread_count = 0;

		if (task_threads(task, &thread_list, &thread_count) != KERN_SUCCESS)
		{
			return false;
		}

		threads.reserve(thread_count);

		for (mach_msg_type_number_t i = 0; i < thread_count; i++)
		{
			// Only the kernel is asked: a thread that exited in the meantime just fails these calls (dead port name)
			thread_identifier_info_data_t id_info{};
			mach_msg_type_number_t id_count = THREAD_IDENTIFIER_INFO_COUNT;
			thread_extended_info_data_t ext_info{};
			mach_msg_type_number_t ext_count = THREAD_EXTENDED_INFO_COUNT;

			if (thread_info(thread_list[i], THREAD_IDENTIFIER_INFO, reinterpret_cast<thread_info_t>(&id_info), &id_count) == KERN_SUCCESS &&
				thread_info(thread_list[i], THREAD_EXTENDED_INFO, reinterpret_cast<thread_info_t>(&ext_info), &ext_count) == KERN_SUCCESS)
			{
				ext_info.pth_name[MAXTHREADNAMESIZE - 1] = '\0';
				threads.push_back({id_info.thread_id, ext_info.pth_user_time + ext_info.pth_system_time, ext_info.pth_name});
			}

			mach_port_deallocate(task, thread_list[i]);
		}

		vm_deallocate(task, reinterpret_cast<vm_address_t>(thread_list), sizeof(thread_t) * thread_count);
		return true;
#else
		return false;
#endif
	}
}

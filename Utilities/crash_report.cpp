// Detailed crash reporting, see crash_report.h.
//
// The native crash handlers themselves live in Utilities/Thread.cpp (POSIX
// signal_handler/sigill_handler, Windows vectored handler): they own the
// register dumps and the JIT-recovery logic. This module gives them the shared
// pieces - the pure report/summary formatters, the orderly-fatal flag, and a
// signal-safe summary writer for the next-launch notice.
//
// Signal safety notes for record_native_crash(): inside a signal handler only
// async-signal-safe calls are allowed - no locks, no heap, no stdio. It uses
// raw open()/write()/fsync()/close(), manual integer formatting, and pre-copied
// static buffers. The full register/backtrace detail is logged by the installed
// handlers through the normal channels before this runs.

#include "crash_report.h"

#include <atomic>
#include <cstring>

#include "Utilities/StrFmt.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace utils::crash_report
{
	namespace
	{
		constexpr const char* kReportBanner = "=================== RPCS3 CRASH REPORT ===================";
		constexpr const char* kReportFooter = "================= END RPCS3 CRASH REPORT =================";
	}

	std::string format_report(const context& ctx)
	{
		std::string out;
		fmt::append(out, "%s\n", kReportBanner);
		fmt::append(out, "Kind: %s\n", ctx.kind ? ctx.kind : "fatal");

		if (ctx.signal_name)
		{
			fmt::append(out, "Signal: %s (%d)\n", ctx.signal_name, ctx.signal_number);
		}

		if (ctx.has_fault_address)
		{
			fmt::append(out, "Fault address: 0x%llx\n", static_cast<unsigned long long>(ctx.fault_address));
		}

		if (!ctx.reason.empty())
		{
			fmt::append(out, "Reason: %s\n", ctx.reason);
		}

		fmt::append(out, "Thread id: %llu\n", static_cast<unsigned long long>(ctx.thread_id));

		if (!ctx.thread_name.empty())
		{
			fmt::append(out, "Thread name: \"%s\"\n", ctx.thread_name);
		}

		if (!ctx.build_version.empty())
		{
			fmt::append(out, "Build: \"%s\"\n", ctx.build_version);
		}

		if (!ctx.extra.empty())
		{
			fmt::append(out, "---- Emulator context ----\n%s", ctx.extra);

			if (ctx.extra.back() != '\n')
			{
				out += '\n';
			}
		}

		if (!ctx.backtrace.empty())
		{
			fmt::append(out, "---- Backtrace (most recent call first) ----\n");

			for (usz i = 0; i < ctx.backtrace.size(); i++)
			{
				fmt::append(out, "#%u %s\n", static_cast<u32>(i), ctx.backtrace[i]);
			}
		}

		fmt::append(out, "%s\n", kReportFooter);
		return out;
	}

	std::string format_summary(const context& ctx, std::string_view log_path)
	{
		std::string out;

		if (ctx.signal_name)
		{
			fmt::append(out, "The emulator crashed with %s", ctx.signal_name);

			if (ctx.has_fault_address)
			{
				fmt::append(out, " at address 0x%llx", static_cast<unsigned long long>(ctx.fault_address));
			}

			out += ".";
		}
		else if (!ctx.reason.empty())
		{
			fmt::append(out, "The emulator stopped because of an error:\n%s", ctx.reason);
		}
		else
		{
			out += "The emulator stopped because of an error.";
		}

		if (!ctx.thread_name.empty())
		{
			fmt::append(out, "\nThread: \"%s\" (id %llu).", ctx.thread_name.c_str(), static_cast<unsigned long long>(ctx.thread_id));
		}

		if (!ctx.extra.empty())
		{
			fmt::append(out, "\n%s", ctx.extra);
		}

		if (!log_path.empty())
		{
			fmt::append(out, "\n\nA detailed crash report was written to the log file:\n%s", log_path);
		}

		return out;
	}

	bool write_last_summary(const std::string& summary_path, std::string_view text)
	{
		if (summary_path.empty())
		{
			return false;
		}

		if (FILE* f = std::fopen(summary_path.c_str(), "w"))
		{
			const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
			std::fclose(f);
			return ok;
		}

		return false;
	}

	bool read_last_summary(const std::string& summary_path, std::string& out)
	{
		out.clear();

		if (summary_path.empty())
		{
			return false;
		}

		if (FILE* f = std::fopen(summary_path.c_str(), "r"))
		{
			char buf[4096];

			while (const usz n = std::fread(buf, 1, sizeof(buf), f))
			{
				out.append(buf, n);
			}

			std::fclose(f);
		}

		// An empty file is not a crash report (torn write); ignore it.
		return !out.empty();
	}

	bool clear_last_summary(const std::string& summary_path)
	{
		if (summary_path.empty())
		{
			return true;
		}

		return std::remove(summary_path.c_str()) == 0;
	}

#if defined(_WIN32)
	namespace
	{
		std::string s_log_path;
		std::string s_summary_path;
		std::string s_build;
		std::atomic<bool> s_orderly_fatal{false};

		void write_all(HANDLE h, const char* data, usz size)
		{
			while (size)
			{
				DWORD written = 0;

				if (!WriteFile(h, data, static_cast<DWORD>(std::min<usz>(size, 1u << 30)), &written, nullptr) || !written)
				{
					return;
				}

				data += written;
				size -= written;
			}
		}

		// The one-page summary, rendered without any allocation.
		void write_summary(HANDLE h, const char* signal_name, int signal_number, u64 fault_address, bool has_fault_address)
		{
			write_all(h, "The emulator crashed with ", 26);
			write_all(h, signal_name, std::strlen(signal_name));

			if (has_fault_address)
			{
				char addr[19] = " at address 0x";
				for (int i = 0; i < 16; i++)
				{
					addr[14 + i] = "0123456789abcdef"[(fault_address >> (60 - 4 * i)) & 0xf];
				}
				write_all(h, addr, 14 + 16);
			}

			write_all(h, ".\nBuild: \"", 10);
			write_all(h, s_build.c_str(), s_build.size());
			write_all(h, "\"\n", 2);
		}
	}

	void set_crash_files(const std::string& log_path, const std::string& summary_path, const std::string& build_version)
	{
		s_log_path = log_path;
		s_summary_path = summary_path;
		s_build = build_version;
	}

	void notify_orderly_fatal()
	{
		s_orderly_fatal.store(true);
	}

	bool is_orderly_fatal()
	{
		return s_orderly_fatal.load();
	}

	void record_native_crash(const char* signal_name, int signal_number, u64 fault_address, bool has_fault_address)
	{
		if (!signal_name || s_summary_path.empty())
		{
			return;
		}

		(void)signal_number;

		// Best-effort copy into the log as well: whatever the channel logger had
		// buffered but not yet written is otherwise lost on a native crash.
		if (!s_log_path.empty())
		{
			if (HANDLE log = CreateFileA(s_log_path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
				log != INVALID_HANDLE_VALUE)
			{
				write_summary(log, signal_name, signal_number, fault_address, has_fault_address);
				FlushFileBuffers(log);
				CloseHandle(log);
			}
		}

		if (HANDLE sum = CreateFileA(s_summary_path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
			sum != INVALID_HANDLE_VALUE)
		{
			write_summary(sum, signal_name, signal_number, fault_address, has_fault_address);
			FlushFileBuffers(sum);
			CloseHandle(sum);
		}
	}
#else
	namespace
	{
		// Static storage only: the recorder must never touch the heap.
		constexpr usz kPathSize = 4096;
		char s_log_path[kPathSize]{};
		char s_summary_path[kPathSize]{};
		char s_build[256]{};
		std::atomic<bool> s_orderly_fatal{false};

		void copy_capped(char* dst, usz dst_size, const char* src, usz src_size)
		{
			if (!dst_size)
			{
				return;
			}

			const usz n = src_size < dst_size - 1 ? src_size : dst_size - 1;
			std::memcpy(dst, src, n);
			dst[n] = '\0';
		}

		void write_all(int fd, const char* data, usz size)
		{
			while (size)
			{
				const ssize_t n = ::write(fd, data, size);

				if (n <= 0)
				{
					return;
				}

				data += n;
				size -= static_cast<usz>(n);
			}
		}

		void write_str(int fd, const char* s)
		{
			usz n = 0;
			while (s[n]) n++;
			write_all(fd, s, n);
		}

		// Manual hex, no snprintf in a signal handler.
		void write_hex(int fd, u64 value)
		{
			write_str(fd, "0x");

			char digits[16]{};
			for (int i = 15; i >= 0; i--)
			{
				digits[i] = "0123456789abcdef"[value & 0xf];
				value >>= 4;
			}

			int first = 0;
			while (first < 15 && digits[first] == '0') first++;
			write_all(fd, digits + first, 16 - first);
		}

		// The one-page summary, rendered without any allocation.
		void write_summary(int fd, const char* signal_name, u64 fault_address, bool has_fault_address)
		{
			write_str(fd, "The emulator crashed with ");
			write_str(fd, signal_name);

			if (has_fault_address)
			{
				write_str(fd, " at address ");
				write_hex(fd, fault_address);
			}

			write_str(fd, ".\nBuild: \"");
			write_str(fd, s_build);
			write_str(fd, "\"\n");
		}
	}

	void set_crash_files(const std::string& log_path, const std::string& summary_path, const std::string& build_version)
	{
		copy_capped(s_log_path, sizeof(s_log_path), log_path.data(), log_path.size());
		copy_capped(s_summary_path, sizeof(s_summary_path), summary_path.data(), summary_path.size());
		copy_capped(s_build, sizeof(s_build), build_version.data(), build_version.size());
	}

	void notify_orderly_fatal()
	{
		s_orderly_fatal.store(true);
	}

	bool is_orderly_fatal()
	{
		return s_orderly_fatal.load();
	}

	void record_native_crash(const char* signal_name, int signal_number, u64 fault_address, bool has_fault_address)
	{
		(void)signal_number;

		if (!signal_name || !s_summary_path[0])
		{
			return;
		}

		// Best-effort copy into the log as well: whatever the channel logger had
		// buffered but not yet written is otherwise lost on a native crash.
		if (s_log_path[0])
		{
			const int log_fd = ::open(s_log_path, O_WRONLY | O_APPEND | O_CREAT, 0644);

			if (log_fd >= 0)
			{
				write_summary(log_fd, signal_name, fault_address, has_fault_address);
				::fsync(log_fd);
				::close(log_fd);
			}
		}

		const int sum_fd = ::open(s_summary_path, O_WRONLY | O_TRUNC | O_CREAT, 0644);

		if (sum_fd >= 0)
		{
			write_summary(sum_fd, signal_name, fault_address, has_fault_address);
			::fsync(sum_fd);
			::close(sum_fd);
		}
	}
#endif
}

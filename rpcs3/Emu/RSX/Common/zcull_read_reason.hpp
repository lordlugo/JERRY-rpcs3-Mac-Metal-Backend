#pragma once

// RPCS3 Metal fork (telemetry): why the RSX thread is reading zcull reports right now, so a backend that has to wait
// for the GPU to get them can name the cause ("RSX thread waits for GPU work by site" in the Metal log).
namespace rsx::reports
{
	inline thread_local const char* g_read_reason = nullptr;

	struct read_reason_scope
	{
		const char* previous;

		explicit read_reason_scope(const char* reason)
			: previous(g_read_reason)
		{
			g_read_reason = reason;
		}

		~read_reason_scope()
		{
			g_read_reason = previous;
		}

		read_reason_scope(const read_reason_scope&) = delete;
		read_reason_scope& operator=(const read_reason_scope&) = delete;
	};
}

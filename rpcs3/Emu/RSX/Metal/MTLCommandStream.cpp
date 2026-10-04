#include "stdafx.h"
#include "MTLCommandStream.h"
#include "MTLGSRenderTypes.hpp"

#include "Emu/IdManager.h"
#include "Emu/Cell/timers.hpp"
#include "Emu/RSX/RSXOffload.h"
#include "Emu/RSX/RSXThread.h"
#include "Emu/system_config.h"

#include "util/asm.hpp"

#include <thread>

namespace mtl
{
	// Global submit guard to prevent races between commit order and timeline signal order
	static shared_mutex g_submit_mutex;

	// Submissions handed to the offloader thread (queue_submit, MTRSX) that it has not committed yet
	static atomic_t<u32> g_queued_submits = 0;

	void acquire_global_submit_lock()
	{
		g_submit_mutex.lock();
	}

	void release_global_submit_lock()
	{
		g_submit_mutex.unlock();
	}

	u64 queue_submit_now(mtl::command_list& commands, const submit_info_t& info)
	{
		autorelease_scope pool;

		acquire_global_submit_lock();
		const u64 value = commands.submit(info);
		release_global_submit_lock();

		return value;
	}

	FORCE_INLINE
	static void queue_submit_impl(command_buffer_chunk* commands, const submit_info_t& info)
	{
		ensure(commands);

		{
			std::lock_guard lock(commands->guard_mutex);
			queue_submit_now(*commands, info);
		}

		// Signal "flushed"
		commands->submit_queued.release(false);
	}

	void queue_submit(command_buffer_chunk* commands, const submit_info_t& info, bool flush)
	{
		if (auto renderer = rsx::get_current_renderer())
		{
			renderer->get_stats().submit_count++;
		}

		// Access to this method must be externally synchronized.
		// Offloader is guaranteed to never call this for async flushes.
		commands->submit_queued = true;

		if (!flush && g_cfg.video.multithreaded_rsx)
		{
			g_queued_submits++;

			auto packet = new queue_submit_t{ commands, info };
			g_fxo->get<rsx::dma_manager>().backend_ctrl(rctrl_queue_submit, packet);
		}
		else
		{
			queue_submit_impl(commands, info);
		}
	}

	void queue_submit(const queue_submit_t* packet)
	{
		// Flush-only version used by asynchronous submit processing (MTRSX)
		queue_submit_impl(packet->commands, packet->info);
		g_queued_submits--;
	}

	void wait_for_queued_submits()
	{
		if (!g_queued_submits.load() || g_fxo->get<rsx::dma_manager>().is_current_thread())
		{
			return;
		}

		// Never waits for anything the waiting thread holds: the renderer drains the offloader before it queues a submission
		// (MTLGSRender::close_and_submit_command_buffer), so a queued submission is only preceded by jobs that have already
		// run, and committing it takes no lock the callers hold (texture cache, flush requests). Bounded anyway: a fatal
		// error on the offloader thread leaves the count behind.
		const u32 queued = g_queued_submits.load();
		const u64 start = get_system_time();

		for (u32 spin = 1; g_queued_submits.load(); spin++)
		{
			if (spin < 256)
			{
				utils::pause();
				continue;
			}

			std::this_thread::yield();

			if ((spin % 1024) == 0 && get_system_time() - start > 2'000'000)
			{
				rsx_log.error("Metal: %u queued submission(s) were not committed by the offloader thread within 2 s; committing out of order", g_queued_submits.load());
				return;
			}
		}

		static atomic_t<bool> s_logged = false;
		if (!s_logged.exchange(true))
		{
			rsx_log.notice("Metal: a texture cache readback waited %llu us for %u submission(s) queued on the offloader thread, so the GPU runs it after the rendering it reads (logged once)",
				get_system_time() - start, queued);
		}
	}
}

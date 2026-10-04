#include "stdafx.h"
#include "nv406e.h"
#include "nv47_sync.hpp"

#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/Common/sync_wait_stats.hpp"
#include "Emu/system_config.h"

#include "context_accessors.define.h"

namespace rsx
{
	namespace nv406e
	{
		void set_reference(context* ctx, u32 /*reg*/, u32 arg)
		{
			auto& dma = *vm::_ptr<RsxDmaControl>(RSX(ctx)->dma_address);

			// RPCS3 Metal fork: REF tells the game how far the RSX got, so it must not get ahead of the zcull reports
			// queued before it. sync() made sure of that by waiting for the GPU to write them, stalling the RSX thread
			// for the whole queued GPU work (God of War: Ascension: ~6.6 ms per frame, the GPU idle in the meantime).
			// Instead, when the CPU reads reports, REF is written like a deferred semaphore: once the reports before it
			// have landed, in order with the other deferred labels, while the RSX keeps processing commands. The game
			// sees REF advance when the work before it is done on the GPU, as it did after the stall.
			const u32 ref_address = RSX(ctx)->dma_address + 0x48; // RsxDmaControl::ref (resv[0x40], put, get, ref)
			if (RSX(ctx)->defer_label(ref_address, arg))
			{
				// Fragment constants may have been updated (as sync() marks)
				RSX(ctx)->m_graphics_state |= rsx::pipeline_state::fragment_constants_dirty;
				dma.get.release(RSX(ctx)->fifo_ctrl->get_pos());
				return;
			}

			RSX(ctx)->sync();

			// Write ref+get (get will be written again with the same value at command end)
			dma.get.release(RSX(ctx)->fifo_ctrl->get_pos());
			dma.ref.store(arg);
		}

		void semaphore_acquire(context* ctx, u32 /*reg*/, u32 arg)
		{
			RSX(ctx)->sync_point_request.release(true);
			const u32 addr = get_address(REGS(ctx)->semaphore_offset_406e(), REGS(ctx)->semaphore_context_dma_406e());

			// Syncronization point, may be associated with memory changes without actually changing addresses
			RSX(ctx)->m_graphics_state |= rsx::pipeline_state::fragment_program_needs_rehash;

			// Unprotected mapping: see rsx::util::write_gcm_label
			const auto& sema = *vm::get_super_ptr<RsxSemaphore>(addr);
			const auto& atomic_sema = *vm::get_super_ptr<atomic_t<RsxSemaphore>>(addr);

			if (sema == arg)
			{
				// Flip semaphore doesnt need wake-up delay
				if (addr != RSX(ctx)->label_addr + 0x10)
				{
					RSX(ctx)->flush_fifo();
					RSX(ctx)->fifo_wake_delay(2);
				}

				return;
			}
			else
			{
				RSX(ctx)->flush_fifo();

				// The awaited value may depend on a texture read label that is still waiting for zcull reports (directly,
				// or through a CPU thread that waits for it). Never wait with labels held back.
				RSX(ctx)->flush_deferred_labels();
			}

			u64 start = get_system_time();
			u64 last_check_val = start;
			const u64 async_flip_start_us = g_sync_wait_stats.rsx_async_flip_us;
			const bool is_flip_sema = (addr == RSX(ctx)->label_addr + 0x10);

			while (sema != arg)
			{
				if (RSX(ctx)->test_stopped())
				{
					RSX(ctx)->state += cpu_flag::again;
					return;
				}

				if (const auto tdr = static_cast<u64>(g_cfg.video.driver_recovery_timeout))
				{
					const u64 current = get_system_time();

					if (current - last_check_val > 20'000)
					{
						// Suspicious amnount of time has passed
						// External pause such as debuggers' pause or operating system sleep may have taken place
						// Ignore it
						start += current - last_check_val;
					}

					last_check_val = current;

					if ((current - start) > tdr)
					{
						// If longer than driver timeout force exit
						rsx_log.error("nv406e::semaphore_acquire has timed out. semaphore_address=0x%X", addr);
						break;
					}
				}

				if (RSX(ctx)->external_interrupt_lock ||
					(RSX(ctx)->state & (cpu_flag::dbg_global_pause + cpu_flag::exit)) == cpu_flag::dbg_global_pause)
				{
					RSX(ctx)->cpu_wait({});
					continue;
				}

				RSX(ctx)->on_semaphore_acquire_wait();

				if (is_flip_sema)
				{
					// The flip-label producer is frequently another thread (PPU flip/vblank
					// handling) or deferred present work: yield the RSX thread instead of
					// hot-spinning the cacheline so the producer is scheduled sooner.
					std::this_thread::yield();
				}
				else if (get_system_time() - start < 200)
				{
					// Wait until the value changes or until 100us pass.
					utils::spin_on_cacheline_once(atomic_sema, sema, 100);
				}
				else
				{
					// RPCS3 Metal fork: a wait that lasts (the game's SPUs/PPU still producing: God of War: Ascension
					// ~4.6 ms per frame) sleeps in short steps instead of keeping a P-core spinning, which the SPU
					// threads it waits for need. 50 us steps keep the wake-up latency far below a frame.
					thread_ctrl::wait_for(50);
				}
			}

			RSX(ctx)->fifo_wake_delay();

			const u64 wait_us = get_system_time() - start;
			RSX(ctx)->performance_counters.idle_time += wait_us;

			// Flips run while waiting (on_semaphore_acquire_wait) are work, not waiting
			const u64 async_flip_us = g_sync_wait_stats.rsx_async_flip_us - async_flip_start_us;
			g_sync_wait_stats.add(is_flip_sema ? sync_wait::flip_semaphore : sync_wait::semaphore, wait_us - std::min(wait_us, async_flip_us));
		}

		void semaphore_release(context* ctx, u32 reg, u32 arg)
		{
			const u32 offset = REGS(ctx)->semaphore_offset_406e();

			if (offset % 4)
			{
				rsx_log.warning("NV406E semaphore release is using unaligned semaphore, ignoring. (offset=0x%x)", offset);
				return;
			}

			const u32 ctxt = REGS(ctx)->semaphore_context_dma_406e();

			// By avoiding doing this on flip's semaphore release
			// We allow last gcm's registers reset to occur in case of a crash
			if (const bool is_flip_sema = (offset == 0x10 && ctxt == CELL_GCM_CONTEXT_DMA_SEMAPHORE_R);
				!is_flip_sema)
			{
				RSX(ctx)->sync_point_request.release(true);
			}

			const u32 addr = get_address(offset, ctxt);

			// TODO: Check if possible to write on reservations
			if (RSX(ctx)->label_addr >> 28 != addr >> 28)
			{
				rsx_log.error("NV406E semaphore unexpected address. Please report to the developers. (offset=0x%x, addr=0x%x)", offset, addr);
				RSX(ctx)->recover_fifo();
				return;
			}

			if (addr == RSX(ctx)->device_addr + 0x30 && !arg)
			{
				// HW flip synchronization related, 1 is not written without display queue command (TODO: make it behave as real hw)
				arg = 1;
			}

			util::write_gcm_label<false, true>(ctx, reg, addr, arg);
		}
	}
}

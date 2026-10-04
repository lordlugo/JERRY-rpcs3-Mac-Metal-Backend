#include "stdafx.h"
#include "MTLQueryPool.h"
#include "MTLGSRenderTypes.hpp"
#include "MTLHelpers.h"
#include "MTLResourceManager.h"
#include "mtlutils/buffer_object.h"
#include "util/asm.hpp"

#include <cstring>

namespace mtl
{
	// ---------------------------------------------------------------------------------------------------------------
	// query_pool (vkutils/query_pool.hpp)

	query_pool::query_pool(const mtl::render_device& dev, u32 size)
		: m_size(size)
	{
		ensure(size > 0);
		m_buffer = std::make_unique<mtl::buffer>(dev, u64{ size } * slot_size, memory_location::host_visible, "occlusion query pool");
		m_results = static_cast<u64*>(m_buffer->map());
		reset();

		// Take 'size' references on this object
		ref_count.release(static_cast<s32>(size));
	}

	query_pool::~query_pool()
	{
		m_results = nullptr;
		m_buffer.reset();
	}

	MTL::Buffer* query_pool::value() const
	{
		return m_buffer->value();
	}

	void query_pool::reset()
	{
		std::memset(m_results, 0, u64{ m_size } * slot_size);
	}

	void query_pool::rearm()
	{
		reset();
		ref_count.release(static_cast<s32>(m_size));
	}

	void query_pool::release_unused(u32 count)
	{
		ensure(count <= m_size);
		ref_count -= static_cast<s32>(count);
	}

	// ---------------------------------------------------------------------------------------------------------------
	// query_pool_manager (VKQueryPool.cpp)

	inline bool query_pool_manager::poke_query(query_slot_info& query, u32 index)
	{
		// Query is ready if the command list that recorded it has completed.
		// Unlike Vulkan there is no partial availability: the visibility result buffer is written at the end of each
		// render pass, but only the completed value is meaningful when the query spans several passes.
		if (query.ready)
		{
			return true;
		}

		if (!query.owner || !query.owner->is_complete(query.owner_sync_id))
		{
			return false;
		}

		const u64 result = query.pool->result(index);
		query.any_passed = (result != 0);
		query.ready = true;
		query.data = static_cast<u32>(std::min<u64>(result, u32{ umax }));
		return true;
	}

	query_pool_manager::query_pool_manager(const mtl::render_device& dev, u32 num_entries)
	{
		ensure(num_entries > 0);

		owner = &dev;
		query_slot_status.resize(num_entries, {});
		m_slot_generation.resize(num_entries, 0);

		for (unsigned i = 0; i < num_entries; ++i)
		{
			m_available_slots.push_back(i);
		}
	}

	query_pool_manager::~query_pool_manager()
	{
		// The renderer waits for the GPU to go idle before destroying us
		m_current_query_pool.reset();
		m_consumed_pools.clear();
		m_query_pool_cache.clear();
		owner = nullptr;
	}

	void query_pool_manager::allocate_new_pool()
	{
		ensure(!m_current_query_pool);
		{
			std::lock_guard lock(m_query_pool_cache_lock); // If we're creating a new pool, we probably have items in the cache.

			if (!m_query_pool_cache.empty())
			{
				m_current_query_pool = std::move(m_query_pool_cache.front());
				m_query_pool_cache.pop_front();

				// Cached pools are only returned once the GPU is done with them (GC eid scope).
				// Reinitialize the refcount and clear stale results.
				m_current_query_pool->rearm();
			}
			else
			{
				const u32 count = ::size32(query_slot_status);
				m_current_query_pool = std::make_unique<query_pool>(*owner, count);
			}
		}

		m_pool_lifetime_counter = m_current_query_pool->size();
		m_pool_generation++;
	}

	void query_pool_manager::reallocate_pool()
	{
		if (m_current_query_pool)
		{
			// A pool holds one reference per slot it may hand out, and free_query() drops one per slot handed out. A
			// pool retired before handing out all of them (allocate_query: the next free slot was already used in it,
			// i.e. a query of this pool was freed before one of the previous pool, e.g. ZCULL discarding its newest
			// query while older reports are pending) would keep the references of the slots it never handed out: it
			// would stay in the discard pile forever and every retirement would allocate a new pool. Drop them now.
			if (m_pool_lifetime_counter)
			{
				m_current_query_pool->release_unused(m_pool_lifetime_counter);
				m_pool_lifetime_counter = 0;
			}

			if (!m_current_query_pool->has_refs())
			{
				auto ref = std::make_unique<query_pool_ref>(this, m_current_query_pool);
				mtl::get_resource_manager()->dispose(ref);
			}
			else
			{
				m_consumed_pools.emplace_back(std::move(m_current_query_pool));

				// Sanity check
				if (m_consumed_pools.size() > 3)
				{
					rsx_log.error("[Robustness warning] Query pool discard pile size is now %llu. Are we leaking??", m_consumed_pools.size());
				}
			}
		}

		allocate_new_pool();
	}

	void query_pool_manager::run_pool_cleanup()
	{
		for (auto It = m_consumed_pools.begin(); It != m_consumed_pools.end();)
		{
			if (!(*It)->has_refs())
			{
				auto ref = std::make_unique<query_pool_ref>(this, *It);
				mtl::get_resource_manager()->dispose(ref);
				It = m_consumed_pools.erase(It);
			}
			else
			{
				It++;
			}
		}
	}

	void query_pool_manager::set_control_flags(bool precise)
	{
		m_result_mode = precise ? MTL::VisibilityResultModeCounting : MTL::VisibilityResultModeBoolean;
	}

	const MTL::Buffer* query_pool_manager::get_visibility_result_buffer() const
	{
		return m_current_query_pool ? m_current_query_pool->value() : nullptr;
	}

	void query_pool_manager::begin_query(command_buffer_chunk& cmd, MTL4::RenderCommandEncoder* active_pass, u32 index)
	{
		ensure(query_slot_status[index].active == false);
		ensure(m_current_query_pool);

		auto& query_info = query_slot_status[index];
		query_info.pool = m_current_query_pool.get();
		query_info.active = true;
		query_info.ready = false;
		query_info.owner = &cmd;
		query_info.owner_sync_id = cmd.reset_id;

		resume_query(cmd, active_pass, index);
	}

	void query_pool_manager::resume_query(mtl::command_list& cmd, MTL4::RenderCommandEncoder* active_pass, u32 index)
	{
		if (active_pass)
		{
			// The pass accumulates into the slot (read-modify-write) from the fragment stage's tests: ordered after
			// earlier passes that wrote it and copies that read it (the pass barriers), and later copies of the slot wait
			// for this pass's fragment and tile work. The pool is the pass's visibility result buffer (allocate_query
			// ends the pass when the pool changes).
			const u64 offset = u64{ index } * query_pool::slot_size;
			cmd.draw_access(write_buffer(ensure(query_slot_status[index].pool)->get(), offset, query_pool::slot_size), stages_attachment);
			active_pass->setVisibilityResultMode(m_result_mode, offset);
		}
	}

	void query_pool_manager::end_query(command_buffer_chunk& cmd, MTL4::RenderCommandEncoder* active_pass, u32 index)
	{
		auto& query_info = query_slot_status[index];
		ensure(query_info.active);

		// The query is final once the recording list completes
		query_info.owner = &cmd;
		query_info.owner_sync_id = cmd.reset_id;

		if (active_pass)
		{
			active_pass->setVisibilityResultMode(MTL::VisibilityResultModeDisabled, 0);
		}
	}

	bool query_pool_manager::check_query_status(u32 index)
	{
		return poke_query(query_slot_status[index], index);
	}

	u32 query_pool_manager::get_query_result(u32 index)
	{
		// Check for cached result
		auto& query_info = query_slot_status[index];

		if (!query_info.ready && !poke_query(query_info, index))
		{
			// Block on the owning command list; this is a hard sync
			if (query_info.owner && query_info.owner->reset_id == query_info.owner_sync_id)
			{
				if (query_info.owner->is_recording())
				{
					// The caller must submit the list first (the renderer flushes when the query is current). The list
					// may still have been committed mid-recording (DMA flush), so the stored value is the best guess.
					rsx_log.error("Occlusion query %u read while its command list is still recording", index);
				}
				else
				{
					mtl::wait_site_scope wait_site("occlusion query owner list");
					query_info.owner->wait();
				}
			}

			if (!poke_query(query_info, index))
			{
				// Finalize from whatever the GPU has written
				const u64 result = query_info.pool->result(index);
				query_info.any_passed = (result != 0);
				query_info.ready = true;
				query_info.data = static_cast<u32>(std::min<u64>(result, u32{ umax }));
			}
		}

		return query_info.data;
	}

	void query_pool_manager::get_query_result_indirect(mtl::command_list& cmd, u32 index, u32 count, const mtl::buffer* dst, u64 dst_offset, u64 bytes_per_slot)
	{
		// Results are written at the end of each render pass. blit() ends the active pass and orders the copy after the
		// passes that wrote these slots (declared when the queries were armed, see resume_query), so the copy observes
		// all results recorded so far.
		ensure(bytes_per_slot == 4 || bytes_per_slot == query_pool::slot_size);

		const auto pool = ensure(query_slot_status[index].pool);
		const u64 src_offset = u64{ index } * query_pool::slot_size;
		const u64 src_length = u64{ count } * query_pool::slot_size;
		auto encoder = cmd.blit({ read_buffer(pool->get(), src_offset, src_length), write_buffer(dst, dst_offset, u64{ count } * bytes_per_slot) });

		if (bytes_per_slot == query_pool::slot_size)
		{
			encoder->copyFromBuffer(pool->value(), src_offset, dst->value(), dst_offset, src_length);
			return;
		}

		// Low dword only (little endian). Counts larger than 32 bits are not meaningful for RSX. One command.
		for (u32 i = 0; i < count; ++i)
		{
			encoder->copyFromBuffer(pool->value(), src_offset + (u64{ i } * query_pool::slot_size), dst->value(), dst_offset + (u64{ i } * 4), 4);
		}
	}

	void query_pool_manager::free_query(mtl::command_list& /*cmd*/, u32 index)
	{
		// Release reference and discard
		auto& query = query_slot_status[index];

		ensure(query.active);
		query.pool->release();

		if (!query.pool->has_refs())
		{
			// No more refs held, remove if in discard pile
			run_pool_cleanup();
		}

		query = {};
		m_available_slots.push_back(index);
	}

	u32 query_pool_manager::allocate_query(mtl::command_list& cmd)
	{
		const bool slot_reuse = !m_available_slots.empty() && m_slot_generation[m_available_slots.front()] == m_pool_generation;
		if (!m_pool_lifetime_counter || slot_reuse)
		{
			// Pool is exhaused (or the next free slot was already used in it), create a new one
			// The visibility result buffer is part of the render pass descriptor, so the active pass has to end.
			if (cmd.is_render_pass_open())
			{
				cmd.end_render_pass(pass_end_reason::query_pool);
			}

			reallocate_pool();
		}

		if (!m_available_slots.empty())
		{
			m_pool_lifetime_counter--;

			const auto result = m_available_slots.front();
			m_available_slots.pop_front();
			m_slot_generation[result] = m_pool_generation;
			return result;
		}

		return ~0u;
	}

	void query_pool_manager::on_query_pool_released(std::unique_ptr<mtl::query_pool>& pool)
	{
		// Metal buffers are cheap to keep around; always recycle.
		std::lock_guard lock(m_query_pool_cache_lock);
		m_query_pool_cache.emplace_back(std::move(pool));
	}

	query_pool_manager::query_pool_ref::~query_pool_ref()
	{
		m_pool_man->on_query_pool_released(m_object);
	}
}

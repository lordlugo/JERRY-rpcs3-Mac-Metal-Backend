#include "stdafx.h"
#include "device.h"
#include "commands.h"

#include "Emu/RSX/Metal/MTLDeviceQuery.h"
#include "Emu/RSX/Metal/MTLPipelineArchive.h"

#include <chrono>

namespace mtl
{
	render_device* g_render_device = nullptr;

	render_device::~render_device()
	{
		destroy();
	}

	bool render_device::create()
	{
		autorelease_scope pool;

		m_device = MTL::CreateSystemDefaultDevice();
		if (!m_device)
		{
			rsx_log.fatal("Metal: no Metal device available");
			return false;
		}

		m_name = to_string(m_device->name());

		m_caps.metal4 = m_device->supportsFamily(MTL::GPUFamilyMetal4);
		if (!m_caps.metal4)
		{
			rsx_log.fatal("Metal: device '%s' does not support Metal 4 (requires Apple M1 or newer and macOS 26+)", m_name);
			m_device->release();
			m_device = nullptr;
			return false;
		}

		m_caps.apple8 = m_device->supportsFamily(MTL::GPUFamilyApple8);
		m_caps.apple9 = m_device->supportsFamily(MTL::GPUFamilyApple9);
		m_caps.apple10 = m_device->supportsFamily(MTL::GPUFamilyApple10);
		m_caps.depth_bounds = m_caps.apple10;
		m_caps.bc_texture_compression = m_device->supportsBCTextureCompression();
		m_caps.unified_memory = m_device->hasUnifiedMemory();
		m_caps.placement_sparse = m_device->supportsPlacementSparse();
		m_caps.max_threads_per_threadgroup = static_cast<u32>(m_device->maxThreadsPerThreadgroup().width);
		m_caps.max_buffer_length = m_device->maxBufferLength();
		m_caps.recommended_working_set = m_device->recommendedMaxWorkingSetSize();
		// Let the OS run as many shader compilations in parallel as the machine allows (macOS 13.3+).
		m_device->setShouldMaximizeConcurrentCompilation(true);
		m_caps.max_compile_tasks = std::max<u32>(2u, static_cast<u32>(m_device->maximumConcurrentCompilationTaskCount()));
		m_caps.max_texture_size_2d = 16384;
		m_caps.max_texture_size_3d = 2048;

		// Queues
		{
			auto desc = ref(MTL4::CommandQueueDescriptor::alloc()->init());
			desc->setLabel(ns_str("RSX main queue"));
			NS::Error* error = nullptr;
			m_queue = m_device->newMTL4CommandQueue(desc.get(), &error);
			if (!m_queue)
			{
				rsx_log.fatal("Metal: failed to create MTL4CommandQueue: %s", to_string(error));
				destroy();
				return false;
			}

			desc->setLabel(ns_str("RSX async queue"));
			m_async_queue = m_device->newMTL4CommandQueue(desc.get(), &error);
		}

		// Shader compiler (Metal 4 compilation API). Inherits the QoS of the calling thread.
		{
			auto desc = ref(MTL4::CompilerDescriptor::alloc()->init());
			desc->setLabel(ns_str("RSX shader compiler"));
			NS::Error* error = nullptr;
			m_compiler = m_device->newCompiler(desc.get(), &error);
			if (!m_compiler)
			{
				rsx_log.fatal("Metal: failed to create MTL4Compiler: %s", to_string(error));
				destroy();
				return false;
			}
		}

		// One global residency set attached to both queues. Metal 4 has no implicit residency.
		{
			auto desc = ref(MTL::ResidencySetDescriptor::alloc()->init());
			desc->setLabel(ns_str("RSX resources"));
			desc->setInitialCapacity(4096);
			NS::Error* error = nullptr;
			m_residency = m_device->newResidencySet(desc.get(), &error);
			if (!m_residency)
			{
				rsx_log.fatal("Metal: failed to create residency set: %s", to_string(error));
				destroy();
				return false;
			}

			m_queue->addResidencySet(m_residency);
			if (m_async_queue)
			{
				m_async_queue->addResidencySet(m_residency);
			}
		}

		rsx_log.notice("Metal: using device '%s' (Apple8=%d Apple9=%d Apple10=%d, unified=%d, BC=%d, compile tasks=%u, working set=%llu MiB)",
			m_name, m_caps.apple8, m_caps.apple9, m_caps.apple10, m_caps.unified_memory, m_caps.bc_texture_compression,
			m_caps.max_compile_tasks, m_caps.recommended_working_set / 0x100000);

		return true;
	}

	void render_device::destroy()
	{
		// The pipeline archive owns capturing compilers, serializers and archives created from this device. Bounded
		// (UI thread): a write still running after the timeout finishes in the background and keeps what it needs alive.
		if (m_device)
		{
			shutdown_pipeline_archive();
		}

		if (m_residency)
		{
			if (m_queue) m_queue->removeResidencySet(m_residency);
			if (m_async_queue) m_async_queue->removeResidencySet(m_residency);
			m_residency->release();
			m_residency = nullptr;
		}

		// The set is gone, nothing references the evicted allocations anymore
		for (MTL::Allocation* allocation : m_pending_evictions)
		{
			allocation->release();
		}
		m_pending_evictions.clear();

		if (m_compiler)
		{
			m_compiler->release();
			m_compiler = nullptr;
		}

		// Every submission completed (or was abandoned): the queues' submission records go with them
		forget_queue_submissions(m_async_queue);
		forget_queue_submissions(m_queue);

		if (m_async_queue)
		{
			m_async_queue->release();
			m_async_queue = nullptr;
		}

		if (m_queue)
		{
			m_queue->release();
			m_queue = nullptr;
		}

		if (m_device)
		{
			m_device->release();
			m_device = nullptr;
		}
	}

	void render_device::make_resident(const MTL::Allocation* allocation)
	{
		if (!allocation || !m_residency)
		{
			return;
		}

		std::lock_guard lock(m_residency_lock);
		m_residency->addAllocation(allocation);
		m_residency_added = true;
		m_residency_count++;
	}

	void render_device::evict(const MTL::Allocation* allocation)
	{
		if (!allocation || !m_residency)
		{
			return;
		}

		const u64 now_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count());

		std::lock_guard lock(m_residency_lock);
		m_residency->removeAllocation(allocation);
		m_residency_count--;

		if (m_pending_evictions.empty())
		{
			m_first_pending_eviction_us = now_us;
		}

		// The caller releases the allocation right after this. The removal is only staged until the next commit(), and
		// committing a set that still references a deallocated resource crashes inside the Objective-C runtime.
		// Keep it alive until the removal is committed.
		m_pending_evictions.push_back(const_cast<MTL::Allocation*>(allocation)->retain());
	}

	void render_device::commit_residency()
	{
		if (!m_residency)
		{
			return;
		}

		// Additions must be committed before the GPU runs work that may use them. Removals only give memory back, and a
		// commit processes the set: they are committed with the next addition, or once 64 are pending or the oldest
		// one is 50 ms old (a few submissions, not every one of them).
		constexpr usz max_pending_evictions = 64;
		constexpr u64 max_eviction_delay_us = 50'000;

		const u64 now_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count());

		std::vector<MTL::Allocation*> evicted;
		{
			std::lock_guard lock(m_residency_lock);

			const bool removals_due = !m_pending_evictions.empty() &&
				(m_pending_evictions.size() >= max_pending_evictions || now_us - m_first_pending_eviction_us >= max_eviction_delay_us);

			if (!m_residency_added && !removals_due)
			{
				return;
			}

			m_residency->commit();
			m_residency_added = false;
			m_residency_commits++;

			evicted.swap(m_pending_evictions);
		}

		// Removals are committed: the set no longer references these
		for (MTL::Allocation* allocation : evicted)
		{
			allocation->release();
		}
	}

	void render_device::attach_residency_set(const MTL::ResidencySet* set)
	{
		if (!set)
		{
			return;
		}

		m_queue->addResidencySet(set);
		if (m_async_queue)
		{
			m_async_queue->addResidencySet(set);
		}
	}

	u64 render_device::allocated_bytes() const
	{
		return m_device ? m_device->currentAllocatedSize() : 0;
	}

	u64 render_device::resident_allocation_count() const
	{
		reader_lock lock(m_residency_lock);
		return m_residency_count;
	}

	u64 render_device::get_residency_commits_and_reset()
	{
		std::lock_guard lock(m_residency_lock);
		return std::exchange(m_residency_commits, 0);
	}

	// MTLDeviceQuery.h
	std::vector<std::string> get_device_names()
	{
		autorelease_scope pool;
		std::vector<std::string> result;

		if (MTL::Device* dev = MTL::CreateSystemDefaultDevice())
		{
			result.push_back(to_string(dev->name()));
			dev->release();
		}

		return result;
	}

	bool is_metal4_supported()
	{
		autorelease_scope pool;
		bool supported = false;

		if (MTL::Device* dev = MTL::CreateSystemDefaultDevice())
		{
			supported = dev->supportsFamily(MTL::GPUFamilyMetal4);
			dev->release();
		}

		return supported;
	}
}

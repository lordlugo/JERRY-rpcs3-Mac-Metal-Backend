#include "stdafx.h"
#include "sync.h"

#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/RSXFIFO.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace mtl
{
	namespace
	{
		struct cpu_wait_counters
		{
			struct bucket
			{
				atomic_t<u64> count = 0;
				atomic_t<u64> total_us = 0;
				atomic_t<u64> max_us = 0;

				void add(u64 us)
				{
					count++;
					total_us += us;
					max_us.fetch_op([us](u64& value)
					{
						value = std::max(value, us);
					});
				}

				cpu_wait_stats_t::bucket take()
				{
					return { count.exchange(0), total_us.exchange(0), max_us.exchange(0) };
				}
			};

			bucket renderer;
			bucket others;
			std::array<bucket, cpu_wait_stats_t::context_count> renderer_by_context;
		};

		cpu_wait_counters g_cpu_waits;

		// RSX thread only (written by its waits, read by its periodic report)
		struct wait_site
		{
			u32 method = 0;
			const char* name = nullptr; // Named site (wait_site_scope); method is then unused
			u64 count = 0;
			u64 total_us = 0;
		};

		std::array<wait_site, 16> g_other_wait_sites{};

		thread_local const char* g_wait_site = nullptr;

		void note_other_wait_site(u32 method, u64 us, const char* name = nullptr)
		{
			wait_site* slot = nullptr;
			for (auto& site : g_other_wait_sites)
			{
				if (site.count && site.name == name && (name || site.method == method))
				{
					slot = &site;
					break;
				}

				if (!site.count && !slot)
				{
					slot = &site;
				}
			}

			if (!slot)
			{
				// Table full: fold into the smallest entry's method slot
				slot = &*std::min_element(g_other_wait_sites.begin(), g_other_wait_sites.end(), [](const wait_site& a, const wait_site& b) { return a.total_us < b.total_us; });
				*slot = {};
			}

			slot->method = method;
			slot->name = name;
			slot->count++;
			slot->total_us += us;
		}
	}

	wait_site_scope::wait_site_scope(const char* site)
		: m_previous(g_wait_site)
	{
		g_wait_site = site;
	}

	wait_site_scope::~wait_site_scope()
	{
		g_wait_site = m_previous;
	}

	std::string take_other_wait_sites(u32 frames)
	{
		auto sites = g_other_wait_sites;
		g_other_wait_sites = {};

		std::sort(sites.begin(), sites.end(), [](const wait_site& a, const wait_site& b) { return a.total_us > b.total_us; });

		std::string result;
		const f64 div = frames ? static_cast<f64>(frames) : 1.;
		for (u32 i = 0; i < 5 && sites[i].count; i++)
		{
			if (sites[i].name)
			{
				fmt::append(result, "%s%s %.2f ms in %.2f", result.empty() ? "" : ", ", sites[i].name, sites[i].total_us / div / 1000., sites[i].count / div);
			}
			else
			{
				fmt::append(result, "%sunnamed, at method 0x%04x %.2f ms in %.2f", result.empty() ? "" : ", ", sites[i].method, sites[i].total_us / div / 1000., sites[i].count / div);
			}
		}

		return result;
	}

	cpu_wait_stats_t get_cpu_wait_stats_and_reset()
	{
		cpu_wait_stats_t stats{ g_cpu_waits.renderer.take(), g_cpu_waits.others.take() };
		for (u32 i = 0; i < cpu_wait_stats_t::context_count; i++)
		{
			stats.renderer_by_context[i] = g_cpu_waits.renderer_by_context[i].take();
		}
		return stats;
	}

	timeline::~timeline()
	{
		destroy();
	}

	void timeline::create(const render_device& dev, std::string_view label, bool count_waits)
	{
		ensure(!m_event);
		autorelease_scope pool;

		m_event = dev.handle()->newSharedEvent();
		ensure(m_event, "Metal: failed to create shared event");
		m_event->setLabel(ns_str(label));
		m_event->setSignaledValue(0);
		m_next_value = 0;
		m_count_waits = count_waits;
	}

	void timeline::destroy()
	{
		if (m_event)
		{
			m_event->release();
			m_event = nullptr;
		}
	}

	u64 timeline::signal(MTL4::CommandQueue* queue)
	{
		const u64 value = ++m_next_value;
		queue->signalEvent(m_event, value);
		return value;
	}

	void timeline::gpu_wait(MTL4::CommandQueue* queue, u64 value) const
	{
		queue->wait(m_event, value);
	}

	bool timeline::wait(u64 value, u64 timeout_us) const
	{
		if (!m_event || m_event->signaledValue() >= value)
		{
			return true;
		}

		const auto start = std::chrono::steady_clock::now();
		bool signaled = true;

		if (timeout_us == 0)
		{
			// Wait "forever" in large chunks so a hung GPU is still diagnosable in the log.
			while (!m_event->waitUntilSignaledValue(value, 10000))
			{
				rsx_log.error("Metal: waited more than 10s for GPU timeline value %llu (current %llu)", value, m_event->signaledValue());
			}
		}
		else
		{
			const u64 timeout_ms = std::max<u64>(1, timeout_us / 1000);
			signaled = m_event->waitUntilSignaledValue(value, timeout_ms);
		}

		if (m_count_waits)
		{
			const u64 waited_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
			const auto renderer = rsx::get_current_renderer();
			if (renderer && renderer->is_current_thread())
			{
				g_cpu_waits.renderer.add(waited_us);
				const u32 context = std::min<u32>(current_pass_context_index(), cpu_wait_stats_t::context_count - 1);
				g_cpu_waits.renderer_by_context[context].add(waited_us);

				if (g_wait_site)
				{
					// Named wait site (any pass context)
					note_other_wait_site(0, waited_us, g_wait_site);
				}
				else if (context == 0 && renderer->fifo_ctrl)
				{
					// Which guest command made the RSX thread wait (the FIFO method header being executed)
					note_other_wait_site(renderer->fifo_ctrl->last_cmd() & 0xfffc, waited_us);
				}
			}
			else
			{
				g_cpu_waits.others.add(waited_us);
			}
		}

		return signaled;
	}
}

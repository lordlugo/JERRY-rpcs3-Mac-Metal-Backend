#include "stdafx.h"
#include "iokit_gpu_stats.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include <algorithm>
#include <vector>
#endif

#include "Emu/Cell/timers.hpp"

namespace mtl
{
#ifdef __APPLE__
	namespace
	{
		struct accelerator_services
		{
			std::vector<io_service_t> services;
			u64 resolved_us = 0;

			~accelerator_services()
			{
				release();
			}

			void release()
			{
				for (const io_service_t service : services)
				{
					IOObjectRelease(service);
				}

				services.clear();
			}

			void resolve()
			{
				release();

				io_iterator_t iterator = 0;
				if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator) != KERN_SUCCESS)
				{
					return;
				}

				while (const io_service_t service = IOIteratorNext(iterator))
				{
					services.push_back(service);
				}

				IOObjectRelease(iterator);
			}
		};

		accelerator_services g_accelerators;
	}

	f32 read_iokit_gpu_utilization()
	{
		// Driver restarts or GPU changes: refresh the service list now and then (same 30 s as Redline)
		const u64 now_us = get_system_time();
		if (g_accelerators.services.empty() || now_us - g_accelerators.resolved_us >= 30'000'000)
		{
			g_accelerators.resolve();
			g_accelerators.resolved_us = now_us;
		}

		static const CFStringRef stats_key = CFSTR("PerformanceStatistics");
		static const CFStringRef util_keys[] =
		{
			CFSTR("Device Utilization %"),
			CFSTR("Renderer Utilization %"),
			CFSTR("Tiler Utilization %"),
		};

		f32 best = -1.f;

		for (const io_service_t service : g_accelerators.services)
		{
			const CFTypeRef property = IORegistryEntryCreateCFProperty(service, stats_key, kCFAllocatorDefault, 0);
			if (!property)
			{
				continue;
			}

			if (CFGetTypeID(property) == CFDictionaryGetTypeID())
			{
				const auto dict = static_cast<CFDictionaryRef>(property);

				for (const CFStringRef key : util_keys)
				{
					const auto value = static_cast<CFTypeRef>(CFDictionaryGetValue(dict, key));
					if (!value || CFGetTypeID(value) != CFNumberGetTypeID())
					{
						continue;
					}

					double pct = 0.;
					if (CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberDoubleType, &pct) && pct == pct)
					{
						best = std::max(best, static_cast<f32>(std::clamp(pct, 0., 100.)));
					}
				}
			}

			CFRelease(property);
		}

		return best;
	}
#else
	f32 read_iokit_gpu_utilization()
	{
		return -1.f;
	}
#endif
}

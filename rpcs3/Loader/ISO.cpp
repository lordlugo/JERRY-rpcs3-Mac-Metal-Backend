#include "stdafx.h"

#include "ISO.h"
#include "Emu/VFS.h"
#include "Emu/system_utils.hpp"
#include "Emu/System.h"
#include "Crypto/utils.h"
#include "util/asm.hpp"

#include <bit>
#include <chrono>
#include <codecvt>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stack>
#include <span>
#include <cstdlib>

LOG_CHANNEL(sys_log, "SYS");
LOG_CHANNEL(iso_log, "ISO");

namespace
{
	// A raw device only accepts whole sectors, read into a buffer aligned on at least the sector size (the page size covers
	// every device and platform)
	constexpr u64 s_raw_alignment = 16 * 1024;

	// A raw device read is split into requests of this size at most, one syscall each
	constexpr u64 s_raw_request_size = 4 * 1024 * 1024;

	// Raw device reads spanning up to "s_raw_window_read_size" bytes (directory records, file headers, small files) are served
	// from the read-ahead window of the file, refilled with "s_raw_window_size" bytes from the first sector of such a read
	constexpr u64 s_raw_window_read_size = 32 * 1024;
	constexpr u64 s_raw_window_size = 128 * 1024;

	u8* alloc_raw_buffer(u64 size)
	{
#ifdef _WIN32
		return ensure(static_cast<u8*>(_aligned_malloc(size, s_raw_alignment)));
#else
		return ensure(static_cast<u8*>(std::aligned_alloc(s_raw_alignment, utils::align(size, s_raw_alignment))));
#endif
	}

	// Thread-local buffer for the raw device requests of "iso_file::read_source()" whose destination is not aligned, grown on
	// demand (up to one request)
	class raw_request_buffer
	{
		std::unique_ptr<u8[], iso_aligned_deleter> m_data;
		u64 m_size = 0;

	public:
		u8* get(u64 size)
		{
			if (size > m_size)
			{
				m_data.reset();
				m_size = std::bit_ceil(size);
				m_data.reset(alloc_raw_buffer(m_size));
			}

			return m_data.get();
		}
	};

	thread_local raw_request_buffer s_request_buffer;

	// Disc reads, see get_iso_read_stats()
	struct iso_read_counters
	{
		atomic_t<u64> file_reads = 0; // Reads from ISO files and disc image files
		atomic_t<u64> file_bytes = 0;
		atomic_t<u64> device_requests = 0; // Requests to raw devices
		atomic_t<u64> device_bytes = 0;
		atomic_t<u64> window_reads = 0; // Raw device reads served by a read-ahead window
	};

	iso_read_counters s_read_counters;
}

void iso_aligned_deleter::operator()(u8* ptr) const
{
#ifdef _WIN32
	_aligned_free(ptr);
#else
	std::free(ptr);
#endif
}

std::string get_iso_read_stats()
{
	const u64 file_reads = s_read_counters.file_reads.exchange(0);
	const u64 file_bytes = s_read_counters.file_bytes.exchange(0);
	const u64 device_requests = s_read_counters.device_requests.exchange(0);
	const u64 device_bytes = s_read_counters.device_bytes.exchange(0);
	const u64 window_reads = s_read_counters.window_reads.exchange(0);

	if (!file_reads && !device_requests && !window_reads)
	{
		return {};
	}

	return fmt::format("Disc reads: ISO/image files: %.1f MiB in %u reads; raw devices: %.1f MiB in %u requests, %u small reads served by read-ahead",
		file_bytes / 1048576., file_reads, device_bytes / 1048576., device_requests, window_reads);
}

static bool is_iso_file(iso_file& file, u64* size = nullptr)
{
	// The standard identifier ("CD001") follows the type of the first volume descriptor
	if (!file || file.size() < ISO_DESCRIPTORS_OFFSET + 6)
	{
		return false;
	}

	char magic[5];

	if (file.read_at(ISO_DESCRIPTORS_OFFSET + 1, magic, 5) != 5)
	{
		return false;
	}

	const bool ret = magic[0] == 'C' && magic[1] == 'D' && magic[2] == '0' && magic[3] == '0' && magic[4] == '1';

	if (size && ret)
	{
		*size = file.size();
	}

	return ret;
}

// Recognize an ISO (file, optical drive or mounted disc image) and provide the file it is read from (see fs::get_optical_disc_source())
static bool is_iso_source(const std::string& path, std::string& source, u64* size, bool* is_raw_device)
{
	if (is_raw_device)
	{
		*is_raw_device = false;
	}

	if (path.empty())
	{
		return false;
	}

	source = path;
	bool raw_device = false;

	// "source" is updated with the file the disc is read from in case "path" points to an optical drive or a mounted disc image
	if (!fs::get_optical_disc_source(path, &source, &raw_device) && !fs::is_file(path))
	{
		return false;
	}

	if (is_raw_device)
	{
		*is_raw_device = raw_device;
	}

	iso_file file(source);

	return is_iso_file(file, size);
}

bool is_iso_file(const std::string& path, u64* size, bool* is_raw_device)
{
	std::string source;
	return is_iso_source(path, source, size, is_raw_device);
}

// Reset the iv to a particular LBA
static void reset_iv(std::array<u8, 16>& iv, u32 lba)
{
	memset(iv.data(), 0, 12);

	iv[12] = (lba & 0xFF000000) >> 24;
	iv[13] = (lba & 0x00FF0000) >> 16;
	iv[14] = (lba & 0x0000FF00) >> 8;
	iv[15] = (lba & 0x000000FF) >> 0;
}

// Main function that will decrypt the sector(s)
static bool decrypt_data(aes_context& aes, u64 offset, const std::span<u8> buffer, const std::span<u8> out_buffer, u64 size)
{
	// The following preliminary checks are good to be provided.
	// Commented out to gain a bit of performance, just because we know the caller is providing values in the expected range

	//if (size == 0)
	//{
	//	return false;
	//}

	//if ((size % 16) != 0)
	//{
	//	iso_log.error("decrypt_data: Requested ciphertext blocks' size must be a multiple of 16 (%llu)", size);
	//	return;
	//}

	u32 cur_sector_lba = static_cast<u32>(offset / ISO_SECTOR_SIZE); // First sector's LBA
	const u32 sector_count = static_cast<u32>((offset + size - 1) / ISO_SECTOR_SIZE) - cur_sector_lba + 1;
	const u64 sector_offset = offset % ISO_SECTOR_SIZE;

	std::array<u8, 16> iv;
	u64 cur_offset;

	// If the offset is not at the beginning of a sector, the first 16 bytes in the buffer
	// represents the IV for decrypting the next data in the buffer.
	// Otherwise, the IV is based on sector's LBA
	if (sector_offset != 0)
	{
		std::memcpy(iv.data(), buffer.data(), iv.size());
		cur_offset = iv.size();
	}
	else
	{
		reset_iv(iv, cur_sector_lba);
		cur_offset = 0;
	}

	u64 cur_size = sector_offset + size <= ISO_SECTOR_SIZE ? size : ISO_SECTOR_SIZE - sector_offset;
	cur_size -= cur_offset;

	// Partial (or even full) first sector
	if (aes_crypt_cbc(&aes, AES_DECRYPT, cur_size, iv.data(), &buffer[cur_offset], &out_buffer[cur_offset]) != 0)
	{
		iso_log.error("decrypt_data: Error decrypting data on first sector read");
		return false;
	}

	if (sector_count < 2) // If no more sector(s)
	{
		return true;
	}

	cur_offset += cur_size;

	const u32 inner_sector_count = sector_count > 2 ? sector_count - 2 : 0; // Remove first and last sector

	// Inner sector(s), if any
	for (u32 i = 0; i < inner_sector_count; i++)
	{
		reset_iv(iv, ++cur_sector_lba); // Next sector's IV

		if (aes_crypt_cbc(&aes, AES_DECRYPT, ISO_SECTOR_SIZE, iv.data(), &buffer[cur_offset], &out_buffer[cur_offset]) != 0)
		{
			iso_log.error("decrypt_data: Error decrypting data on inner sector(s) read");
			return false;
		}

		cur_offset += ISO_SECTOR_SIZE;
	}

	reset_iv(iv, ++cur_sector_lba); // Next sector's IV

	// Partial (or even full) last sector
	if (aes_crypt_cbc(&aes, AES_DECRYPT, size - cur_offset, iv.data(), &buffer[cur_offset], &out_buffer[cur_offset]) != 0)
	{
		iso_log.error("decrypt_data: Error decrypting data on last sector read");
		return false;
	}

	return true;
}

iso_type_status iso_file_decryption::get_key(const std::string& key_path, aes_context* aes_ctx)
{
	fs::file key_file(key_path);

	// If no ".dkey" and ".key" file exists
	if (!key_file)
	{
		return iso_type_status::ERROR_OPENING_KEY;
	}

	std::array<char, 32> key_str {};
	std::array<u8, 16> key {};

	const u64 key_len = key_file.read(key_str.data(), key_str.size());

	if (key_len == key_str.size() || key_len == key.size())
	{
		// If the key read from the key file is 16 bytes long instead of 32, consider the file as
		// binary (".key") and so not needing any further conversion from hex string to bytes
		if (key_len == key.size())
		{
			std::memcpy(key.data(), key_str.data(), key.size());
		}
		else
		{
			std::string error;
			hex_to_bytes(key.data(), std::string_view(key_str.data(), key_str.size()), key_str.size(), &error);

			if (!error.empty())
			{
				iso_log.error("get_key(%s): %s", key_path, error);
				return iso_type_status::ERROR_PROCESSING_KEY;
			}
		}

		aes_context aes_dec {};

		// If "aes_ctx" not requested
		if (!aes_ctx)
		{
			aes_ctx = &aes_dec;
		}

		// Create the decryption context. If the context is successfully created, fill in "aes_ctx"
		// (if requested) and return REDUMP_ISO
		if (aes_setkey_dec(aes_ctx, key.data(), 128) == 0)
		{
			return iso_type_status::REDUMP_ISO;
		}
	}

	return iso_type_status::ERROR_PROCESSING_KEY;
}

iso_type_status iso_file_decryption::retrieve_key(iso_archive& archive, const fs::file& iso, std::string& key_path, aes_context& aes_ctx)
{
	//
	// Find the first existing file in the archive present on the list of well known encrypted files to use for testing a matching key
	//

	const std::map<std::string, std::string> dec_magics {
		{"PS3_GAME/LICDIR/LIC.DAT", "PS3LICDA"},
		{"PS3_GAME/USRDIR/EBOOT.BIN", "SCE"}
	};

	iso_fs_node* node = nullptr;
	std::string magic_value;

	for (const auto& magic : dec_magics)
	{
		if (iso_fs_node* _node = archive.retrieve(magic.first))
		{
			magic_value = magic.second;
			node = _node;
			break;
		}
	}

	if (!node)
	{
		return iso_type_status::ERROR_OPENING_KEY;
	}

	//
	// Read the first encrypted sector to use for testing a matching key
	//

	std::array<u8, ISO_SECTOR_SIZE> enc_sec;
	std::array<u8, ISO_SECTOR_SIZE> dec_sec;

	// The first sector of the file, read through the ISO already open
	const u64 sector_address = ::at32(node->metadata.extents, 0).start * ISO_SECTOR_SIZE;

	if (::at32(node->metadata.extents, 0).size < ISO_SECTOR_SIZE || iso.read_at(sector_address, enc_sec.data(), ISO_SECTOR_SIZE) != ISO_SECTOR_SIZE)
	{
		return iso_type_status::NOT_ISO;
	}

	// A decrypted disc (e.g. a mounted decrypted ISO image) holds the magic value as is: no key can decrypt it into the magic
	// value too, so the (possibly thousands of) key files are not even read
	if (std::memcmp(magic_value.data(), enc_sec.data(), magic_value.size()) == 0)
	{
		return iso_type_status::ERROR_OPENING_KEY;
	}

	//
	// Scan all the key files present in the redump keys folder, decrypt the read sector and test for a match with file's magic value
	//

	std::vector<fs::dir_entry> entries;

	for (auto&& dir_entry : fs::dir(rpcs3::utils::get_redump_key_dir()))
	{
		// Prefetch entries, it is unsafe to keep fs::dir for a long time or for many operations
		entries.emplace_back(std::move(dir_entry));
	}

	for (auto path_it = entries.begin(); path_it != entries.end(); path_it++)
	{
		const fs::dir_entry dir_entry = std::move(*path_it);

		if (dir_entry.name == "." || dir_entry.name == ".." || dir_entry.is_directory)
		{
			continue;
		}

		key_path = rpcs3::utils::get_redump_key_dir() + dir_entry.name;

		// If no valid key is present on the file
		if (get_key(key_path, &aes_ctx) != iso_type_status::REDUMP_ISO)
		{
			continue;
		}

		// If the decryption fails
		if (!decrypt_data(aes_ctx, sector_address, enc_sec, dec_sec, ISO_SECTOR_SIZE))
		{
			continue;
		}

		// If the decrypted data match the magic value
		if (std::memcmp(magic_value.data(), dec_sec.data(), magic_value.size()) == 0)
		{
			return iso_type_status::REDUMP_ISO;
		}
	}

	return iso_type_status::ERROR_OPENING_KEY;
}

iso_type_status iso_file_decryption::check_type(const std::string& path, std::string* key_path, aes_context* aes_ctx)
{
	std::string source;
	bool raw_device = false;

	if (!is_iso_source(path, source, nullptr, &raw_device))
	{
		return iso_type_status::NOT_ISO;
	}

	// The key files are named after the ISO: a mounted disc image read from its image file is an ISO file named after it
	return find_key_file(raw_device ? path : source, key_path, aes_ctx);
}

// Look for a key file named after the ISO, next to it or in the redump keys folder
iso_type_status iso_file_decryption::find_key_file(const std::string& path, std::string* key_path, aes_context* aes_ctx)
{
	// Remove file extension from file path
	const usz ext_pos = path.rfind('.');
	const std::string name_path = ext_pos == umax ? path : path.substr(0, ext_pos);

	// Detect file name (with no parent folder and no file extension)
	const usz name_pos = name_path.rfind('/');
	const std::string name = name_pos == umax ? name_path : name_path.substr(name_pos);

	const std::array<std::string, 4> key_paths {
		name_path + ".dkey",
		name_path + ".key",
		rpcs3::utils::get_redump_key_dir() + name + ".dkey",
		rpcs3::utils::get_redump_key_dir() + name + ".key"
	};

	for (const std::string& path : key_paths)
	{
		if (fs::is_file(path))
		{
			if (key_path)
			{
				*key_path = path;
			}

			return get_key(path, aes_ctx);
		}
	}

	return iso_type_status::ERROR_OPENING_KEY;
}

bool iso_file_decryption::init(const fs::file& iso, const std::string& path, iso_archive* disc_archive)
{
	// Reset attributes first
	m_enc_type = iso_encryption_type::NONE;
	m_region_info.clear();

	//
	// Store the ISO region information (needed by both the "Redump" type (only on "decrypt()" method) and "3k3y" type)
	//

	std::array<u8, ISO_SECTOR_SIZE * 2> sec0_sec1;

	if (iso.size() < sec0_sec1.size())
	{
		iso_log.error("init: Found only %llu sector(s) (minimum required is 2): '%s'", iso.size(), path);
		return false;
	}

	if (iso.read_at(0, sec0_sec1.data(), sec0_sec1.size()) != sec0_sec1.size())
	{
		iso_log.error("init: Failed to read file: '%s'", path);
		return false;
	}

	// NOTE:
	//
	// Following checks and assigned values are based on PS3 ISO specification.
	// E.g. all even regions (0, 2, 4 etc.) are always unencrypted while the odd ones are encrypted

	const u32 region_count = read_from_ptr<be_t<u32>>(sec0_sec1);

	// Ensure the region count is a proper value
	if (region_count < 1 || region_count > 127) // It's non-PS3ISO
	{
		iso_log.error("init: Failed to read region information (region_count=%lu): '%s'", region_count, path);
		return false;
	}

	m_region_info.resize(region_count * 2 - 1);

	for (usz i = 0; i < m_region_info.size(); i++)
	{
		// Store the region information in address format
		const usz modulo_2 = i % 2;
		m_region_info[i].encrypted = (modulo_2 == 1);
		m_region_info[i].region_first_addr = (i == 0 ? 0ULL : m_region_info[i - 1].region_last_addr + 1ULL);
		m_region_info[i].region_last_addr = (static_cast<u64>(read_from_ptr<be_t<u32>>(sec0_sec1, 12 + (i * 4))) - modulo_2) * ISO_SECTOR_SIZE + ISO_SECTOR_SIZE - 1ULL;
	}

	//
	// Check for Redump type
	//

	std::string key_path;

	// Try to detect the Redump type with a key file named after the ISO (a raw device has no such name). If so, the decryption
	// context is set into "m_aes_dec"
	iso_type_status status = fs::is_optical_raw_device(path) ? iso_type_status::ERROR_OPENING_KEY : find_key_file(path, &key_path, &m_aes_dec);

	// For a disc (optical drive or mounted disc image, "disc_archive" provided), scan the redump keys folder and retrieve (if
	// present) the first key that allows decrypting a sector of the disc
	if (status != iso_type_status::REDUMP_ISO && disc_archive)
	{
		status = retrieve_key(*disc_archive, iso, key_path, m_aes_dec);
	}

	switch (status)
	{
	case iso_type_status::NOT_ISO:
		iso_log.warning("init: Failed to recognize ISO file: '%s'", path);
		break;
	case iso_type_status::REDUMP_ISO:
		iso_log.warning("init: Found matching key file: '%s'", key_path);
		m_enc_type = iso_encryption_type::REDUMP; // SET ENCRYPTION TYPE: REDUMP
		break;
	case iso_type_status::ERROR_OPENING_KEY:
		iso_log.warning("init: Failed to open, or missing, key file: '%s'", key_path);
		break;
	case iso_type_status::ERROR_PROCESSING_KEY:
		iso_log.error("init: Failed to process key file: '%s'", key_path);
		break;
	default:
		break;
	}

	//
	// Check for 3k3y type
	//

	// If encryption type is still set to NONE
	if (m_enc_type == iso_encryption_type::NONE)
	{
		// The 3k3y watermarks located at offset 0xF70: (D|E)ncrypted 3K BLD
		static const u8 k3k3y_enc_watermark[16] = {0x45, 0x6E, 0x63, 0x72, 0x79, 0x70, 0x74, 0x65, 0x64, 0x20, 0x33, 0x4B, 0x20, 0x42, 0x4C, 0x44};
		static const u8 k3k3y_dec_watermark[16] = {0x44, 0x6E, 0x63, 0x72, 0x79, 0x70, 0x74, 0x65, 0x64, 0x20, 0x33, 0x4B, 0x20, 0x42, 0x4C, 0x44};

		if (std::memcmp(&k3k3y_enc_watermark[0], &sec0_sec1[0xF70], sizeof(k3k3y_enc_watermark)) == 0)
		{
			// Grab D1 from the 3k3y sector
			u8 key[16];

			std::memcpy(key, &sec0_sec1[0xF80], 0x10);

			// Convert D1 to KEY and generate the "m_aes_dec" context
			u8 key_d1[] = {0x38, 11, 0xcf, 11, 0x53, 0x45, 0x5b, 60, 120, 0x17, 0xab, 0x4f, 0xa3, 0xba, 0x90, 0xed};
			u8 iv_d1[] = {0x69, 0x47, 0x47, 0x72, 0xaf, 0x6f, 0xda, 0xb3, 0x42, 0x74, 0x3a, 0xef, 170, 0x18, 0x62, 0x87};

			aes_context aes_d1;

			if (aes_setkey_enc(&aes_d1, key_d1, 128) == 0)
			{
				if (aes_crypt_cbc(&aes_d1, AES_ENCRYPT, 16, &iv_d1[0], key, key) == 0)
				{
					if (aes_setkey_dec(&m_aes_dec, key, 128) == 0)
					{
						m_enc_type = iso_encryption_type::ENC_3K3Y; // SET ENCRYPTION TYPE: ENC_3K3Y
					}
				}
			}

			if (m_enc_type == iso_encryption_type::NONE) // If encryption type was not set to ENC_3K3Y for any reason
			{
				iso_log.error("init: Failed to set encryption type to ENC_3K3Y: '%s'", path);
			}
		}
		else if (std::memcmp(&k3k3y_dec_watermark[0], &sec0_sec1[0xF70], sizeof(k3k3y_dec_watermark)) == 0)
		{
			m_enc_type = iso_encryption_type::DEC_3K3Y; // SET ENCRYPTION TYPE: DEC_3K3Y
		}
	}

	switch (m_enc_type)
	{
	case iso_encryption_type::REDUMP:
		iso_log.warning("init: Set 'enc type': REDUMP, 'reg count': %u: '%s'", m_region_info.size(), path);
		break;
	case iso_encryption_type::ENC_3K3Y:
		iso_log.warning("init: Set 'enc type': ENC_3K3Y, 'reg count': %u: '%s'", m_region_info.size(), path);
		break;
	case iso_encryption_type::DEC_3K3Y:
		iso_log.warning("init: Set 'enc type': DEC_3K3Y, 'reg count': %u: '%s'", m_region_info.size(), path);
		break;
	case iso_encryption_type::NONE: // If encryption type was not set for any reason
		iso_log.warning("init: Set 'enc type': NONE, 'reg count': %u: '%s'", m_region_info.size(), path);
		break;
	}

	return true;
}

bool iso_file_decryption::decrypt(u64 offset, const std::span<u8> buffer, const std::string& name)
{
	// If it's a non-encrypted type (or nothing is requested), nothing more to do
	if (m_enc_type == iso_encryption_type::NONE || buffer.empty())
	{
		return true;
	}

	// If it's a 3k3y ISO and data at offset 0xF70 is being requested, we should null it out
	if (m_enc_type == iso_encryption_type::DEC_3K3Y || m_enc_type == iso_encryption_type::ENC_3K3Y)
	{
		constexpr u64 range_start = 0xF70ULL;
		constexpr u64 range_end = 0x1070ULL;
		constexpr u64 range = range_end - range_start;

		const u64 buffer_size = buffer.size();

		ensure(offset <= (u64{umax} - buffer_size)); // Check for overflow
		const u64 buffer_end = offset + buffer_size;

		if (buffer_end > range_start && offset < range_end)
		{
			// Zero out the 0xF70 - 0x1070 overlap
			const u64 buf_overlap_start = offset < range_start ? range_start - offset : 0;
			const u64 buf_overlap_end = buffer_end < range_end ? buffer_size : range;

			std::memset(&buffer[buf_overlap_start], 0x00, buf_overlap_end - buf_overlap_start);
		}

		// If it's a decrypted ISO then return, otherwise go on to the decryption logic
		if (m_enc_type == iso_encryption_type::DEC_3K3Y)
		{
			return true;
		}
	}

	// If it's an encrypted type, decrypt the parts of the request lying in encrypted regions (a request can span several regions)
	const u64 end = offset + buffer.size();
	bool found = false;

	for (const iso_region_info& info : m_region_info)
	{
		if (info.region_last_addr < offset || info.region_first_addr >= end)
		{
			continue;
		}

		found = true;

		if (!info.encrypted)
		{
			continue;
		}

		// NOTE: only the first part can start within a sector (then preceded by the 16 bytes block used as IV), any other part
		//       starts with its region, that is with a sector
		const u64 part_first = std::max(offset, info.region_first_addr);
		const u64 part_end = std::min(end, info.region_last_addr + 1);
		const std::span<u8> part = buffer.subspan(part_first - offset, part_end - part_first);

		decrypt_data(m_aes_dec, part_first, part, part, part.size());
	}

	if (found)
	{
		return true;
	}

	iso_log.error("decrypt: %s: LBA request wasn't in the 'm_region_info' for an encrypted ISO? - RP: 0x%lx, RC: 0x%lx, LR: (0x%016lx - 0x%016lx)",
		name,
		offset,
		static_cast<u32>(m_region_info.size()),
		static_cast<u32>(!m_region_info.empty() ? m_region_info.back().region_first_addr : 0),
		static_cast<u32>(!m_region_info.empty() ? m_region_info.back().region_last_addr : 0));

	return true;
}

iso_file_encrypted::iso_file_encrypted(const std::string& path, bs_t<fs::open_mode> mode, const iso_fs_node& node, std::shared_ptr<iso_file_decryption> dec)
	: iso_file(path, mode, node), m_dec(dec)
{
}

u64 iso_file_encrypted::read_at(u64 offset, void* buffer, u64 size)
{
	u64 max_size = std::min(size, local_extent_remaining(offset));

	if (max_size == 0)
	{
		return 0;
	}

	const u64 total_size = this->size();
	const u64 archive_first_offset = file_offset(offset);
	const u64 archive_end_offset = archive_first_offset + max_size;
	u8* const dest = static_cast<u8*>(buffer);

	// Sectors are decrypted whole, each one on its own (its IV derives from its LBA): the whole sectors of the request are read
	// in one go and decrypted in place, a partial sector at either end is read and decrypted in a buffer of its own.
	// NOTE: "iso_file_decryption::decrypt()" only requires blocks of 16 bytes, so a truncated ISO file may end within its last
	//       sector
	//
	//                    archive_first_offset                                  archive_end_offset
	//                              '-------------------------------------------'
	//                 disc: | ... |xxxx#######|###########|###########|#####xxxxxx| ... |
	//                             '           '                       '
	//                        first_sector   inner_first           inner_end

	const auto read_partial_sector = [&](u64 sector_address, u64 copy_first, u64 copy_end)
	{
		std::array<u8, ISO_SECTOR_SIZE> sector;
		const u64 decrypt_size = utils::align(copy_end - sector_address, 16);
		const u64 total_read = read_source(sector_address, sector.data(), sector.size());

		if (total_read < decrypt_size)
		{
			iso_log.error("read_at: %s: Error reading from file - O: %llu (%llu), S: %llu/%llu/%llu (%llu), TR: %llu", m_meta.name,
				offset, sector_address, decrypt_size, max_size, size, total_size, total_read);

			return false;
		}

		m_dec->decrypt(sector_address, {sector.data(), decrypt_size}, m_meta.name);
		std::memcpy(dest + (copy_first - archive_first_offset), sector.data() + (copy_first - sector_address), copy_end - copy_first);
		return true;
	};

	const u64 first_sector = archive_first_offset - archive_first_offset % ISO_SECTOR_SIZE;
	const u64 inner_first = utils::align(archive_first_offset, ISO_SECTOR_SIZE);
	const u64 inner_end = archive_end_offset - archive_end_offset % ISO_SECTOR_SIZE;

	if (inner_first >= inner_end)
	{
		// No whole sector: the request lies in one sector, or in two
		const u64 second_sector = first_sector + ISO_SECTOR_SIZE;

		if (!read_partial_sector(first_sector, archive_first_offset, std::min(second_sector, archive_end_offset)) ||
			(second_sector < archive_end_offset && !read_partial_sector(second_sector, second_sector, archive_end_offset)))
		{
			return 0;
		}
	}
	else
	{
		if (archive_first_offset < inner_first && !read_partial_sector(first_sector, archive_first_offset, inner_first))
		{
			return 0;
		}

		u8* const inner = dest + (inner_first - archive_first_offset);
		const u64 inner_size = inner_end - inner_first;

		if (const u64 total_read = read_source(inner_first, inner, inner_size); total_read != inner_size)
		{
			iso_log.error("read_at: %s: Error reading from file - O: %llu (%llu), S: %llu/%llu/%llu (%llu), TR: %llu", m_meta.name,
				offset, inner_first, inner_size, max_size, size, total_size, total_read);

			return 0;
		}

		m_dec->decrypt(inner_first, {inner, inner_size}, m_meta.name);

		if (inner_end < archive_end_offset && !read_partial_sector(inner_end, inner_end, archive_end_offset))
		{
			return 0;
		}
	}

	// If present, read the remaining chunk of data on next extent
	if (size > max_size && (offset + max_size) < total_size)
	{
		iso_log.warning("read_at: %s: Extent limit reached reading from file (%llu/%llu)", m_meta.name, max_size, size);
		max_size += read_at(offset + max_size, &reinterpret_cast<u8*>(buffer)[max_size], size - max_size);
	}

	return max_size;
}

template<typename T>
inline T retrieve_endian_int(const u8* buf)
{
	T out {};

	if constexpr (std::endian::little == std::endian::native)
	{
		// First half = little-endian copy
		std::memcpy(&out, buf, sizeof(T));
	}
	else
	{
		// Second half = big-endian copy
		std::memcpy(&out, buf + sizeof(T), sizeof(T));
	}

	return out;
}

// Assumed that directory entry is at file head
static std::optional<iso_fs_metadata> iso_read_directory_entry(fs::file& entry, bool& read_error, bool names_in_ucs2 = false)
{
	read_error = true;
	const auto start_pos = entry.pos();
	u8 entry_length = 0;
	if (!entry.read(entry_length))
	{
		return std::nullopt;
	}

	if (entry_length == 0)
	{
		// Zero marks sector padding, not an unreadable directory record.
		read_error = false;
		return std::nullopt;
	}

	// Batch this set of file reads. This reduces overall time spent in iso_read_directory_entry by ~41%
#pragma pack(push, 1)
	struct iso_entry_header
	{
		//u8 entry_length; // Handled separately
		u8 extended_attribute_length;
		u8 start_sector[8];
		u8 file_size[8];
		u8 year;
		u8 month;
		u8 day;
		u8 hour;
		u8 minute;
		u8 second;
		u8 timezone_value;
		u8 flags;
		u8 file_unit_size;
		u8 interleave;
		u8 volume_sequence_number[4];
		u8 file_name_length;
		//u8 file_name[file_name_length]; // Handled separately
	};
#pragma pack(pop)
	static_assert(sizeof(iso_entry_header) == 32);

	iso_entry_header header{};
	if (entry_length < 1 + sizeof(header) || !entry.read(header)
		|| header.file_name_length > entry_length - 1 - sizeof(header))
	{
		return std::nullopt;
	}

	const u32 start_sector = retrieve_endian_int<u32>(header.start_sector);
	const u32 file_size = retrieve_endian_int<u32>(header.file_size);

	// The recorded ECMA-119 date holds the local time of the recorder, paired with its offset from GMT.
	// std::chrono::sys_days is anchored to the UNIX epoch, so the host time zone never enters the result.
	const std::chrono::year_month_day file_date
	{
		std::chrono::year{1900 + header.year},
		std::chrono::month{header.month},
		std::chrono::day{header.day}
	};

	// The offset from GMT is stored as a signed number of 15 minute intervals (ECMA-119 9.1.5),
	// so it has to be subtracted from the recorded local time in order to obtain UTC
	const auto file_time = std::chrono::sys_days{file_date} + std::chrono::hours{header.hour} + std::chrono::minutes{header.minute}
		+ std::chrono::seconds{header.second} - std::chrono::minutes{static_cast<s8>(header.timezone_value) * 15};

	const std::time_t date_time = static_cast<std::time_t>(file_time.time_since_epoch().count());

	// 2nd flag bit indicates whether a given fs node is a directory
	const bool is_directory = header.flags & 0b00000010;
	const bool has_more_extents = header.flags & 0b10000000;

	std::string file_name;

	if (!entry.read(file_name, header.file_name_length))
	{
		iso_log.error("iso_archive: Failed to read file name");
		return std::nullopt;
	}

	if (file_name.size() == 1 && file_name[0] == '\0')
	{
		file_name = ".";
	}
	else if (file_name == "\1")
	{
		file_name = "..";
	}
	else if (names_in_ucs2) // For strings in joliet descriptor
	{
		// Characters are stored in big endian format
		const be_t<u16>* raw = utils::bless<const be_t<u16>>(file_name.data());
		std::u16string utf16;

		utf16.resize(file_name.size() / 2);

		for (usz i = 0; i < utf16.size(); i++)
		{
			utf16[i] = raw[i];
		}

		file_name = utf16_to_utf8(utf16);
	}

	if (file_name.ends_with(";1"))
	{
		file_name.erase(file_name.end() - 2, file_name.end());
	}

	if (file_name.size() > 1 && file_name.ends_with("."))
	{
		file_name.pop_back();
	}

	// Skip the rest of the entry
	entry.seek(entry_length + start_pos);

	read_error = false;
	return iso_fs_metadata
	{
		.name = std::move(file_name),
		.time = date_time,
		.is_directory = is_directory,
		.has_multiple_extents = has_more_extents,
		.extents =
		{
			iso_extent_info
			{
				.start = start_sector,
				.size = file_size
			}
		}
	};
}

// Largest directory accepted, read into memory at once (a directory of a disc takes a few KiB, a few hundred KiB for thousands
// of files)
constexpr u64 s_max_directory_size = 64 * 1024 * 1024;

static bool iso_form_hierarchy(fs::file& file, iso_fs_node& node, bool use_ucs2_decoding = false, const std::string& parent_path = "")
{
	if (!node.metadata.is_directory)
	{
		return !parent_path.empty();
	}

	const std::string node_path = parent_path + "/" + node.metadata.name;

	if (!parent_path.empty() && !Emu.IsPathInsideDir(node_path, parent_path, false))
	{
		iso_log.error("iso_archive::iso_form_hierarchy: node path outside of parent (parent_path='%s', node_path='%s')", parent_path, node_path);
		return false;
	}

	std::vector<usz> multi_extent_node_indices;

	// Assuming the directory spans a single extent
	const auto& directory_extent = ::at32(node.metadata.extents, 0);
	const u64 start_pos = directory_extent.start * ISO_SECTOR_SIZE;
	const u64 size = file.size();
	if (start_pos > size || directory_extent.size > size - start_pos)
	{
		return false;
	}

	// The whole directory is read at once and its records are parsed from memory: a single request per directory instead of
	// three small reads per record (positions are relative to the directory, which starts on a sector boundary)
	if (directory_extent.size > s_max_directory_size)
	{
		iso_log.error("iso_archive::iso_form_hierarchy: directory too large (node_path='%s', size=%u)", node_path, directory_extent.size);
		return false;
	}

	std::vector<u8> directory_data(directory_extent.size);

	if (file.read_at(start_pos, directory_data.data(), directory_data.size()) != directory_data.size())
	{
		return false;
	}

	fs::file directory(directory_data.data(), directory_data.size());
	const u64 end_pos = directory_data.size();

	while (directory.pos() < end_pos)
	{
		bool read_error = false;
		auto entry = iso_read_directory_entry(directory, read_error, use_ucs2_decoding);
		if (read_error || directory.pos() > end_pos)
		{
			return false;
		}

		if (!entry)
		{
			const u64 new_sector = (directory.pos() / ISO_SECTOR_SIZE) + 1;

			directory.seek(new_sector * ISO_SECTOR_SIZE);
			continue;
		}

		bool extent_added = false;

		// Find previous extent and merge into it, otherwise we push this node's index
		for (usz index : multi_extent_node_indices)
		{
			auto& selected_node = ::at32(node.children, index);

			if (selected_node->metadata.name == entry->name)
			{
				// Merge into selected_node
				selected_node->metadata.extents.push_back(::at32(entry->extents, 0));

				extent_added = true;
				break;
			}
		}

		if (extent_added)
		{
			continue;
		}

		if (entry->has_multiple_extents)
		{
			// Haven't pushed entry to node.children yet so node.children::size() == entry_index
			multi_extent_node_indices.push_back(node.children.size());
		}

		node.children.push_back(std::make_unique<iso_fs_node>(iso_fs_node{
			.metadata = std::move(*entry)
		}));
	}

	// Not needed by the subdirectories
	directory.close();
	directory_data = {};

	for (auto& child_node : node.children)
	{
		if (child_node->metadata.name != "." && child_node->metadata.name != "..")
		{
			if (!iso_form_hierarchy(file, *child_node, use_ucs2_decoding, node_path))
			{
				return false;
			}
		}
	}

	return true;
}

u64 iso_fs_metadata::size() const
{
	u64 total_size = 0;

	for (const auto& extent : extents)
	{
		total_size += extent.size;
	}

	return total_size;
}

iso_archive::iso_archive(const std::string& path)
{
	m_path = path;

	// "m_path" is updated with the file the disc is read from in case "path" points to an optical drive or a mounted disc image
	// (the image file itself, or the raw device)
	const bool is_disc = fs::get_optical_disc_source(path, &m_path);

	// NOTE: the file is opened once here and then handed over to the parsing below. Recognizing the ISO through its
	//       path (i.e. "is_iso_file(m_path)") would open it and read its volume descriptor a second time, which is a
	//       physical read when the path points to an optical drive
	auto file = std::make_unique<iso_file>(m_path);

	if (!is_iso_file(*file))
	{
		iso_log.error("iso_archive: Failed to recognize ISO file: '%s'", path);
		invalidate();
		return;
	}

	// NOTE: "is_iso_file()" reads through "read_at()", which does not move the position, so the file is still at its
	//       beginning here
	fs::file iso_file(std::move(file));

	u8 descriptor_type = -2;
	bool use_ucs2_decoding = false;

	// Skip the system area: scanning it sector by sector would read 16 sectors (a physical read each, on an optical
	// drive) only to find boot data, which could even be mistaken for a volume descriptor.
	// NOTE: "is_iso_file()" above already verified the standard identifier is right here
	iso_file.seek(ISO_DESCRIPTORS_OFFSET);

	do
	{
		const auto descriptor_start = iso_file.pos();

		if (!iso_file.read(descriptor_type))
		{
			iso_log.error("iso_archive: Failed to read volume descriptor: '%s'", path);
			invalidate();
			return;
		}

		// 1 = primary vol descriptor, 2 = joliet SVD
		if (descriptor_type == 1 || descriptor_type == 2)
		{
			use_ucs2_decoding = descriptor_type == 2;

			// Skip the rest of descriptor's data
			iso_file.seek(155, fs::seek_cur);

			bool read_error = false;
			const auto node = iso_read_directory_entry(iso_file, read_error, use_ucs2_decoding);
			if (read_error)
			{
				iso_log.error("iso_archive: Failed to read root directory record: '%s'", path);
				invalidate();
				return;
			}

			if (node)
			{
				m_root = iso_fs_node
				{
					.metadata = node.value()
				};
			}
		}

		iso_file.seek(descriptor_start + ISO_SECTOR_SIZE);
	}
	while (descriptor_type != 255 && iso_file.pos() < iso_file.size());

	if (descriptor_type != 255)
	{
		iso_log.error("iso_archive: Corrupt ISO file '%s': Volume Descriptor Set Terminator not found", path);
		invalidate();
		return;
	}

	if (!iso_form_hierarchy(iso_file, m_root, use_ucs2_decoding))
	{
		iso_log.error("iso_archive: Corrupt ISO file '%s': Failed to form hierarchy", path);
		invalidate();
		return;
	}

	// Only when the archive object is fully set, we can finally initialize the decryption object needing the archive object
	m_dec = std::make_shared<iso_file_decryption>();

	if (!m_dec->init(iso_file, m_path, is_disc ? this : nullptr))
	{
		iso_log.error("iso_archive: Corrupt ISO file '%s': Decryption failed", path);
		invalidate();
		return;
	}
}

iso_fs_node* iso_archive::retrieve(const std::string& passed_path)
{
	if (passed_path.empty() || !is_valid())
	{
		return nullptr;
	}

	const std::string path = std::filesystem::path(passed_path).string();
	const std::string_view path_sv = path;

	usz start = 0;
	usz end = path_sv.find_first_of(fs::delim);

	std::stack<iso_fs_node*> search_stack;

	search_stack.push(&m_root);

	do
	{
		if (search_stack.empty())
		{
			return nullptr;
		}

		const auto* top_entry = search_stack.top();

		if (end == umax)
		{
			end = path.size();
		}

		const std::string_view path_component = path_sv.substr(start, end - start);

		bool found = false;

		if (path_component == ".")
		{
			found = true;
		}
		else if (path_component == "..")
		{
			search_stack.pop();

			found = true;
		}
		else
		{
			for (const auto& entry : top_entry->children)
			{
				if (entry->metadata.name == path_component)
				{
					search_stack.push(entry.get());

					found = true;
					break;
				}
			}
		}

		if (!found)
		{
			return nullptr;
		}

		start = end + 1;
		end = path_sv.find_first_of(fs::delim, start);
	}
	while (start < path.size());

	if (search_stack.empty())
	{
		return nullptr;
	}

	return search_stack.top();
}

void iso_archive::invalidate()
{
	m_root = {};
	m_dec.reset();
}

bool iso_archive::is_valid() const
{
	return !m_root.metadata.name.empty();
}

bool iso_archive::exists(const std::string& path)
{
	return retrieve(path) != nullptr;
}

bool iso_archive::is_file(const std::string& path)
{
	const auto file_node = retrieve(path);

	if (!file_node)
	{
		return false;
	}

	return !file_node->metadata.is_directory;
}

std::unique_ptr<fs::file_base> iso_archive::get_iso_file(const std::string& path, bs_t<fs::open_mode> mode, const iso_fs_node& node)
{
	if (!is_valid())
	{
		return nullptr;
	}

	if (m_dec->get_enc_type() == iso_encryption_type::NONE)
	{
		return std::make_unique<iso_file>(path, mode, node);
	}

	return std::make_unique<iso_file_encrypted>(path, mode, node, m_dec);
}

std::unique_ptr<fs::file_base> iso_archive::open(const std::string& path)
{
	const auto node = retrieve(path);

	if (!node)
	{
		fs::g_tls_error = fs::error::noent;
		return nullptr;
	}

	return get_iso_file(m_path, fs::read, *node);
}

psf::registry iso_archive::open_psf(const std::string& path)
{
	const auto node = retrieve(path);

	if (!node)
	{
		return psf::registry();
	}

	const fs::file psf_file(get_iso_file(m_path, fs::read, *node));

	return psf::load_object(psf_file, path);
}

iso_file::iso_file(const std::string& path, bs_t<fs::open_mode> mode)
{
	m_file = fs::file(path, mode);

	if (!m_file)
	{
		// Should never happen... TODO: throw something?
		iso_log.error("iso_file: Failed to open file: '%s'", path);
		return;
	}

	m_meta.name = path;
	m_meta.extents.push_back({0, m_file.size()});

	// NOTE: the position of "m_file" is never used, every read is positioned (see read_source())
	m_raw_device = fs::is_optical_raw_device(path);
}

iso_file::iso_file(const std::string& path, bs_t<fs::open_mode> mode, const iso_fs_node& node)
	: m_meta(node.metadata)
{
	m_file = fs::file(path, mode);

	if (!m_file)
	{
		// Should never happen... TODO: throw something?
		iso_log.error("iso_file: Failed to open file: '%s'", path);
		return;
	}

	m_raw_device = fs::is_optical_raw_device(path);
}

fs::stat_t iso_file::get_stat()
{
	return fs::stat_t
	{
		.is_directory = false,
		.is_symlink = false,
		.is_writable = false,
		.size = size(),
		.atime = m_meta.time,
		.mtime = m_meta.time,
		.ctime = m_meta.time
	};
}

bool iso_file::trunc(u64 /*length*/)
{
	fs::g_tls_error = fs::error::readonly;
	return false;
}

std::pair<u64, iso_extent_info> iso_file::get_extent_pos(u64 pos) const
{
	ensure(!m_meta.extents.empty());

	auto it = m_meta.extents.begin();

	while (pos >= it->size && it != m_meta.extents.end() - 1)
	{
		pos -= it->size;

		it++;
	}

	return {pos, *it};
}

u64 iso_file::local_extent_remaining(u64 pos) const
{
	const auto [local_pos, extent] = get_extent_pos(pos);

	return local_pos < extent.size ? extent.size - local_pos : 0;
}

u64 iso_file::local_extent_size(u64 pos) const
{
	return get_extent_pos(pos).second.size;
}

// Assumed valid and in bounds
u64 iso_file::file_offset(u64 pos) const
{
	const auto [local_pos, extent] = get_extent_pos(pos);

	return (extent.start * ISO_SECTOR_SIZE) + local_pos;
}

u64 iso_file::read(void* buffer, u64 size)
{
	const auto r = read_at(m_pos, buffer, size);

	m_pos += r;
	return r;
}

u64 iso_file::read_source(u64 address, void* buffer, u64 size)
{
	if (!m_raw_device)
	{
		// An ISO file or a disc image file: any offset and size in one request, cached and read ahead by the system
		const u64 total_read = m_file.read_at(address, buffer, size);

		s_read_counters.file_reads++;
		s_read_counters.file_bytes += total_read;
		return total_read;
	}

	// A raw device only accepts whole sectors (see "s_raw_alignment" for the buffer)
	//
	//                         address                                     address + size
	//                              '-------------------------------------------'
	//           raw device: | ... |xxxx#######|###########|###########|#####xxxxxx| ... |
	//                             '                                                '
	//                        first_sector                                     end_sector
	//
	// A small read is served from the read-ahead window (refilled from "first_sector" if it doesn't hold the whole read), a
	// larger one is read by requests of up to "s_raw_request_size" bytes, straight into the destination if it is aligned,
	// otherwise through a thread-local buffer

	u8* const dest = static_cast<u8*>(buffer);
	const u64 first_sector = address - address % ISO_SECTOR_SIZE;
	const u64 end_sector = utils::align(address + size, ISO_SECTOR_SIZE);

	if (end_sector - first_sector <= s_raw_window_read_size)
	{
		std::lock_guard lock(m_window_mutex);

		if (first_sector < m_window_address || end_sector > m_window_address + m_window_size)
		{
			// Only the whole sectors of the device can be read
			const u64 device_end = m_file.size() - m_file.size() % ISO_SECTOR_SIZE;

			m_window_address = first_sector;
			m_window_size = 0;

			if (first_sector >= device_end)
			{
				return 0;
			}

			if (!m_window)
			{
				m_window.reset(alloc_raw_buffer(s_raw_window_size));
			}

			const u64 total_read = m_file.read_at(first_sector, m_window.get(), std::min(s_raw_window_size, device_end - first_sector));

			s_read_counters.device_requests++;
			s_read_counters.device_bytes += total_read;
			m_window_size = total_read - total_read % ISO_SECTOR_SIZE;
		}
		else
		{
			s_read_counters.window_reads++;
		}

		if (address >= m_window_address + m_window_size)
		{
			return 0;
		}

		const u64 total_read = std::min(size, m_window_address + m_window_size - address);
		std::memcpy(dest, m_window.get() + (address - m_window_address), total_read);
		return total_read;
	}

	const bool direct = first_sector == address && end_sector == address + size && reinterpret_cast<uptr>(buffer) % s_raw_alignment == 0;
	u8* const bounce = direct ? nullptr : s_request_buffer.get(std::min(end_sector - first_sector, s_raw_request_size));
	u64 total_read = 0;

	for (u64 request_address = first_sector; request_address < end_sector;)
	{
		const u64 request = std::min(end_sector - request_address, s_raw_request_size);
		const u64 request_read = m_file.read_at(request_address, direct ? dest + (request_address - address) : bounce, request);

		s_read_counters.device_requests++;
		s_read_counters.device_bytes += request_read;

		// The part of the destination this request covers
		const u64 copy_first = std::max(request_address, address);
		const u64 copy_end = std::min(request_address + request_read, address + size);

		if (copy_end > copy_first)
		{
			if (!direct)
			{
				std::memcpy(dest + (copy_first - address), bounce + (copy_first - request_address), copy_end - copy_first);
			}

			total_read = copy_end - address;
		}

		if (request_read != request)
		{
			break;
		}

		request_address += request;
	}

	return total_read;
}

u64 iso_file::read_at(u64 offset, void* buffer, u64 size)
{
	u64 max_size = std::min(size, local_extent_remaining(offset));

	if (max_size == 0)
	{
		return 0;
	}

	const u64 total_size = this->size();
	const u64 archive_first_offset = file_offset(offset);
	const u64 total_read = read_source(archive_first_offset, buffer, max_size);

	if (total_read != max_size)
	{
		iso_log.error("read_at: %s: Error reading from file - O: %llu (%llu), S: %llu/%llu (%llu), TR: %llu", m_meta.name,
			offset, archive_first_offset, max_size, size, total_size, total_read);

		return 0;
	}

	// If present, read the remaining chunk of data on next extent
	if (size > max_size && (offset + max_size) < total_size)
	{
		iso_log.warning("read_at: %s: Extent limit reached reading from file (%llu/%llu)", m_meta.name, max_size, size);
		max_size += read_at(offset + max_size, &reinterpret_cast<u8*>(buffer)[max_size], size - max_size);
	}

	return max_size;
}

u64 iso_file::write(const void* /*buffer*/, u64 /*size*/)
{
	fs::g_tls_error = fs::error::readonly;
	return 0;
}

u64 iso_file::seek(s64 offset, fs::seek_mode whence)
{
	const s64 new_pos =
		whence == fs::seek_set ? offset :
		whence == fs::seek_cur ? offset + m_pos :
		whence == fs::seek_end ? offset + size() : -1;

	if (new_pos < 0)
	{
		fs::g_tls_error = fs::error::inval;
		return -1;
	}

	// Only the position of this file changes: every read is positioned (no syscall for a seek, or for fs::file::pos())
	m_pos = new_pos;
	return m_pos;
}

u64 iso_file::size()
{
	u64 extent_sizes = 0;

	for (const auto& extent : m_meta.extents)
	{
		extent_sizes += extent.size;
	}

	return extent_sizes;
}

void iso_file::release()
{
	m_file.release();
}

bool iso_dir::read(fs::dir_entry& entry)
{
	if (m_pos < m_node.children.size())
	{
		const auto& selected = m_node.children[m_pos].get()->metadata;

		entry.name = selected.name;
		entry.atime = selected.time;
		entry.mtime = selected.time;
		entry.ctime = selected.time;
		entry.is_directory = selected.is_directory;
		entry.is_symlink = false;
		entry.is_writable = false;
		entry.size = selected.size();

		m_pos++;
		return true;
	}

	return false;
}

void iso_dir::rewind()
{
	m_pos = 0;
}

bool iso_device::stat(const std::string& path, fs::stat_t& info)
{
	const auto relative_path = std::filesystem::relative(std::filesystem::path(path), std::filesystem::path(fs_prefix)).string();

	const auto node = m_archive.retrieve(relative_path);

	if (!node)
	{
		fs::g_tls_error = fs::error::noent;
		return false;
	}

	const auto& meta = node->metadata;

	info = fs::stat_t
	{
		.is_directory = meta.is_directory,
		.is_symlink = false,
		.is_writable = false,
		.size = meta.size(),
		.atime = meta.time,
		.mtime = meta.time,
		.ctime = meta.time
	};

	return true;
}

bool iso_device::statfs(const std::string& path, fs::device_stat& info)
{
	const auto relative_path = std::filesystem::relative(std::filesystem::path(path), std::filesystem::path(fs_prefix)).string();

	const auto node = m_archive.retrieve(relative_path);

	if (!node)
	{
		fs::g_tls_error = fs::error::noent;
		return false;
	}

	const u64 size = node->metadata.size();

	info = fs::device_stat
	{
		.block_size = size,
		.total_size = size,
		.total_free = 0,
		.avail_free = 0
	};

	return true;
}

std::unique_ptr<fs::file_base> iso_device::open(const std::string& path, bs_t<fs::open_mode> mode)
{
	const auto relative_path = std::filesystem::relative(std::filesystem::path(path), std::filesystem::path(fs_prefix)).string();

	const auto node = m_archive.retrieve(relative_path);

	if (!node)
	{
		fs::g_tls_error = fs::error::noent;
		return nullptr;
	}

	if (node->metadata.is_directory)
	{
		fs::g_tls_error = fs::error::isdir;
		return nullptr;
	}

	return m_archive.get_iso_file(m_archive.path(), mode, *node);
}

std::unique_ptr<fs::dir_base> iso_device::open_dir(const std::string& path)
{
	const auto relative_path = std::filesystem::relative(std::filesystem::path(path), std::filesystem::path(fs_prefix)).string();

	const auto node = m_archive.retrieve(relative_path);

	if (!node)
	{
		fs::g_tls_error = fs::error::noent;
		return nullptr;
	}

	if (!node->metadata.is_directory)
	{
		// fs::dir::open -> ::readdir should return ENOTDIR when path is pointing to a file instead of a folder.
		fs::g_tls_error = fs::error::notdir;
		return nullptr;
	}

	return std::make_unique<iso_dir>(*node);
}

void load_iso(const std::string& path)
{
	sys_log.notice("Loading ISO '%s'", path);

	fs::set_virtual_device("iso_overlay_fs_dev", stx::make_shared<iso_device>(path));

	vfs::mount("/dev_bdvd/"sv, iso_device::virtual_device_name + "/");
}

void unload_iso()
{
	sys_log.notice("Unloading ISO");

	fs::set_virtual_device("iso_overlay_fs_dev", stx::shared_ptr<iso_device>());
}

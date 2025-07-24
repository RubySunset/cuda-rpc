#include <cstring>
#include <elf.h>
#include <glog/logging.h>

#include "./image.hpp"


struct nv_fatbin_header {
    uint8_t magic[8];
    uint64_t size;
};

static constexpr uint8_t nv_fatbin_magic[8] = {0x50, 0xed, 0x55, 0xba, 0x01, 0x00, 0x10, 0x00};

static constexpr uint8_t elf32_magic[8] = {ELFMAG0, ELFMAG1, ELFMAG2, ELFMAG3, ELFCLASS32};
static constexpr uint8_t elf64_magic[8] = {ELFMAG0, ELFMAG1, ELFMAG2, ELFMAG3, ELFCLASS64};

size_t
get_image_size(const void* ptr)
{
    // try fatbin (simpler)
    {
        auto header = (nv_fatbin_header*)ptr;
        if (memcmp(header->magic, nv_fatbin_magic, 8) == 0) {
            return sizeof(*header) + header->size;
        }
    }

    // try cubin (ELF32)
    {
        auto header = (Elf32_Ehdr*)ptr;
        if (memcmp(header, elf32_magic, 5) == 0) {
            return header->e_shoff + (header->e_shentsize * header->e_shnum);
        }
    }

    // try cubin (ELF64)
    {
        auto header = (Elf64_Ehdr*)ptr;
        if (memcmp(header, elf64_magic, 5) == 0) {
            return header->e_shoff + (header->e_shentsize * header->e_shnum);
        }
    }

    // try PTX (null-terminated)
    return strlen((const char*)ptr);
}

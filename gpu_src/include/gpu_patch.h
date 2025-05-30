#pragma once

#include <cstdint>

constexpr uint32_t GPU_WARP_SIZE = 32;

constexpr uint32_t MAX_NUM_MEMORY_RANGES = 16384;
constexpr uint32_t MAX_NUM_TENSOR_RANGES = 65536;
constexpr uint32_t MEMORY_ACCESS_BUFFER_SIZE = 1048576*2;


enum class MemoryType
{
    Global,
    Shared,
    Local,
};

// Information regarding a memory access
struct MemoryAccess
{
    uint64_t addresses[GPU_WARP_SIZE];
    uint32_t accessSize;
    uint32_t flags;
    uint64_t warpId;
    MemoryType type;

    // copy constructor
    MemoryAccess(const MemoryAccess& other)
    {
        for (int i = 0; i < GPU_WARP_SIZE; i++)
        {
            addresses[i] = other.addresses[i];
        }
        accessSize = other.accessSize;
        flags = other.flags;
        warpId = other.warpId;
        type = other.type;
    }

    MemoryAccess() = default;

    ~MemoryAccess() = default;
};


struct MemoryRange
{
    uint64_t start;
    uint64_t end;

    bool operator<(const MemoryRange& other) const {
        return start < other.start;
    }

    bool operator==(const MemoryRange& other) const {
        return start == other.start && end == other.end;
    }
};


struct MemoryAccessState
{
    uint32_t size;
    MemoryRange start_end[MAX_NUM_MEMORY_RANGES];
    uint64_t touch[MAX_NUM_MEMORY_RANGES];
};

struct TensorAccessState
{
    uint32_t size;
    MemoryRange start_end[MAX_NUM_TENSOR_RANGES];
    uint64_t touch[MAX_NUM_TENSOR_RANGES];
};

struct DoorBell
{
    volatile bool full;
    volatile uint32_t num_threads;
};

// Main tracking structure that patches get as userdata
struct MemoryAccessTracker
{
    uint32_t currentEntry;
    uint32_t numEntries;
    uint64_t accessCount;
    DoorBell* doorBell;
    MemoryAccess* access_buffer;
    MemoryAccessState* access_state;
    TensorAccessState* tensor_access_state;
};

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include <faiss/gpu/utils/PtxUtils.cuh>

namespace faiss { namespace gpu {
/*
#if __CUDA_ARCH__ >= 350
// Use the CC 3.5+ read-only texture cache (nc)
#define LD_NC_V1 "ld.global.cs.nc.u32"
#define LD_NC_V2 "ld.global.cs.nc.v2.u32"
#define LD_NC_V4 "ld.global.cs.nc.v4.u32"
#else
// Read normally
#define LD_NC_V1 "ld.global.cs.u32"
#define LD_NC_V2 "ld.global.cs.v2.u32"
#define LD_NC_V4 "ld.global.cs.v4.u32"
#endif // __CUDA_ARCH__
*/
// TODO: HADI These assembly instructions should be ported to AMD gfx908 arch
// #define LD_NC_V1 "ld.global.cs.nc.u32"
// #define LD_NC_V2 "ld.global.cs.nc.v2.u32"
// #define LD_NC_V4 "ld.global.cs.nc.v4.u32"

///
/// This file contains loader functions for PQ codes of various byte
/// length.
///

// Type-specific wrappers around the PTX bfe.* instruction, for
// quantization code extraction
inline __device__ unsigned int getByte(unsigned char v,
                                       int pos,
                                       int width) {
  return v;
}

inline __device__ unsigned int getByte(unsigned short v,
                                       int pos,
                                       int width) {
  return getBitfield((unsigned int) v, pos, width);
}

inline __device__ unsigned int getByte(unsigned int v,
                                       int pos,
                                       int width) {
  return getBitfield(v, pos, width);
}

inline __device__ unsigned int getByte(unsigned long v,
                                       int pos,
                                       int width) {
  return getBitfield(v, pos, width);
}

template <int NumSubQuantizers>
struct LoadCode32 {};

template<>
struct LoadCode32<1> {
  static inline __device__ void load(unsigned int code32[1],
                                     unsigned char* p,
                                     int offset) {
    /* TODO: HADI
    p += offset * 1;
    asm("ld.global.cs.u8 {%0}, [%1];" :
        "=r"(code32[0]) : "l"(p));
    */
    code32[0] = *reinterpret_cast<uint8_t*>(p);

  }
};

template<>
struct LoadCode32<2> {
  static inline __device__ void load(unsigned int code32[1],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 2;
    /* TODO: HADI
    asm("ld.global.cs.u16 {%0}, [%1];" :
        "=r"(code32[0]) : "l"(p));
    */
    code32[0] = *reinterpret_cast<uint16_t*>(p);

  }
};

template<>
struct LoadCode32<3> {
  static inline __device__ void load(unsigned int code32[1],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 3;

/* TODO: HADI
    // FIXME: this is a non-coalesced, unaligned, non-vectorized load
    // unfortunately need to reorganize memory layout by warp
    asm("ld.global.cs.u8 {%0}, [%1 + 0];" :
        "=r"(a) : "l"(p));
    asm("ld.global.cs.u8 {%0}, [%1 + 1];" :
        "=r"(b) : "l"(p));
    asm("ld.global.cs.u8 {%0}, [%1 + 2];" :
        "=r"(c) : "l"(p));
*/
    unsigned int a = *p;
    unsigned int b = *(p + 1);
    unsigned int c = *(p + 2);

    // FIXME: this is also slow, since we have to recover the
    // individual bytes loaded
    code32[0] = (c << 16) | (b << 8) | a;
  }
};

template<>
struct LoadCode32<4> {
  static inline __device__ void load(unsigned int code32[1],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 4;
    /* TODO: HADI
    asm("ld.global.cs.u32 {%0}, [%1];" :
        "=r"(code32[0]) : "l"(p));
    */
    code32[0] = *reinterpret_cast<unsigned int*>(p);
  }
};

template<>
struct LoadCode32<8> {
  static inline __device__ void load(unsigned int code32[2],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 8;
    /* TODO: HADI
    asm("ld.global.cs.v2.u32 {%0, %1}, [%2];" :
        "=r"(code32[0]), "=r"(code32[1]) : "l"(p));
    */
    code32[0] = *reinterpret_cast<unsigned int*>(p);
    code32[1] = *reinterpret_cast<unsigned int*>(p + 4);
  }
};

template<>
struct LoadCode32<12> {
  static inline __device__ void load(unsigned int code32[3],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 12;
    /* TODO: HADI
    // FIXME: this is a non-coalesced, unaligned, non-vectorized load
    // unfortunately need to reorganize memory layout by warp
    asm(LD_NC_V1 " {%0}, [%1 + 0];" :
        "=r"(code32[0]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 4];" :
        "=r"(code32[1]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 8];" :
        "=r"(code32[2]) : "l"(p));
    */

    code32[0] = *reinterpret_cast<unsigned int*>(p);
    code32[1] = *reinterpret_cast<unsigned int*>(p + 4);
    code32[2] = *reinterpret_cast<unsigned int*>(p + 8);

  }
};

template<>
struct LoadCode32<16> {
  static inline __device__ void load(unsigned int code32[4],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 16;

    /* TODO: HADI
    asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" :
        "=r"(code32[0]), "=r"(code32[1]),
        "=r"(code32[2]), "=r"(code32[3]) : "l"(p));
    */

    code32[0] = *reinterpret_cast<unsigned int*>(p + 0);
    code32[1] = *reinterpret_cast<unsigned int*>(p + 4);
    code32[2] = *reinterpret_cast<unsigned int*>(p + 8);
    code32[3] = *reinterpret_cast<unsigned int*>(p + 12);

  }
};

template<>
struct LoadCode32<20> {
  static inline __device__ void load(unsigned int code32[5],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 20;
    /* TODO: HADI
    // FIXME: this is a non-coalesced, unaligned, non-vectorized load
    // unfortunately need to reorganize memory layout by warp
    asm(LD_NC_V1 " {%0}, [%1 + 0];" :
        "=r"(code32[0]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 4];" :
        "=r"(code32[1]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 8];" :
        "=r"(code32[2]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 12];" :
        "=r"(code32[3]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 16];" :
        "=r"(code32[4]) : "l"(p));
    */
    
    code32[0] = *reinterpret_cast<unsigned int*>(p);
    code32[1] = *reinterpret_cast<unsigned int*>(p + 4);
    code32[2] = *reinterpret_cast<unsigned int*>(p + 8);
    code32[3] = *reinterpret_cast<unsigned int*>(p + 12);
    code32[4] = *reinterpret_cast<unsigned int*>(p + 16);
  }
};

template<>
struct LoadCode32<24> {
  static inline __device__ void load(unsigned int code32[6],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 24;
    /* TODOL HADI
    // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
    // unfortunately need to reorganize memory layout by warp
    asm(LD_NC_V2 " {%0, %1}, [%2 + 0];" :
        "=r"(code32[0]), "=r"(code32[1]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 8];" :
        "=r"(code32[2]), "=r"(code32[3]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 16];" :
        "=r"(code32[4]), "=r"(code32[5]) : "l"(p));
    */
   
    code32[0] = *reinterpret_cast<unsigned int*>(p + 0);
    code32[1] = *reinterpret_cast<unsigned int*>(p + 4);
    code32[2] = *reinterpret_cast<unsigned int*>(p + 8);
    code32[3] = *reinterpret_cast<unsigned int*>(p + 12);
    code32[4] = *reinterpret_cast<unsigned int*>(p + 16);
    code32[5] = *reinterpret_cast<unsigned int*>(p + 20);
  }
};

template<>
struct LoadCode32<28> {
  static inline __device__ void load(unsigned int code32[7],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 28;
    /* TODO: HADI
    // FIXME: this is a non-coalesced, unaligned, non-vectorized load
    // unfortunately need to reorganize memory layout by warp
    asm(LD_NC_V1 " {%0}, [%1 + 0];" :
        "=r"(code32[0]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 4];" :
        "=r"(code32[1]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 8];" :
        "=r"(code32[2]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 12];" :
        "=r"(code32[3]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 16];" :
        "=r"(code32[4]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 20];" :
        "=r"(code32[5]) : "l"(p));
    asm(LD_NC_V1 " {%0}, [%1 + 24];" :
        "=r"(code32[6]) : "l"(p));
    */
   
    code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
    code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
    code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
    code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
    code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
    code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
    code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
  }
};

template<>
struct LoadCode32<32> {
  static inline __device__ void load(unsigned int code32[8],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 32;
    /* TODO: HADI
    // FIXME: this is a non-coalesced load
    // unfortunately need to reorganize memory layout by warp
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4];" :
        "=r"(code32[0]), "=r"(code32[1]),
        "=r"(code32[2]), "=r"(code32[3]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 16];" :
        "=r"(code32[4]), "=r"(code32[5]),
        "=r"(code32[6]), "=r"(code32[7]) : "l"(p));
    */
   
    code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
    code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
    code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
    code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
    code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
    code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
    code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
    code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
  }
};

template<>
struct LoadCode32<40> {
  static inline __device__ void load(unsigned int code32[10],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 40;
    /* TODO: HADI
    // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
    // unfortunately need to reorganize memory layout by warp
    asm(LD_NC_V2 " {%0, %1}, [%2 + 0];" :
        "=r"(code32[0]), "=r"(code32[1]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 8];" :
        "=r"(code32[2]), "=r"(code32[3]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 16];" :
        "=r"(code32[4]), "=r"(code32[5]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 24];" :
        "=r"(code32[6]), "=r"(code32[7]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 32];" :
        "=r"(code32[8]), "=r"(code32[9]) : "l"(p));
    */
    
    code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
    code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
    code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
    code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
    code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
    code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
    code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
    code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
    code32[8] = *(reinterpret_cast<unsigned int*>(p + 32));
    code32[9] = *(reinterpret_cast<unsigned int*>(p + 36));
  }
};

template<>
struct LoadCode32<48> {
  static inline __device__ void load(unsigned int code32[12],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 48;
    /* TODO: HADI
    // FIXME: this is a non-coalesced load
    // unfortunately need to reorganize memory layout by warp
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4];" :
        "=r"(code32[0]), "=r"(code32[1]),
        "=r"(code32[2]), "=r"(code32[3]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 16];" :
        "=r"(code32[4]), "=r"(code32[5]),
        "=r"(code32[6]), "=r"(code32[7]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 32];" :
        "=r"(code32[8]), "=r"(code32[9]),
        "=r"(code32[10]), "=r"(code32[11]) : "l"(p));
    */

    code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
    code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
    code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
    code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
    code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
    code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
    code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
    code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
    code32[8] = *(reinterpret_cast<unsigned int*>(p + 32));
    code32[9] = *(reinterpret_cast<unsigned int*>(p + 36));
    code32[10] = *(reinterpret_cast<unsigned int*>(p + 40));
    code32[11] = *(reinterpret_cast<unsigned int*>(p + 44));

  }
};

template<>
struct LoadCode32<56> {
  static inline __device__ void load(unsigned int code32[14],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 56;
    /* TODO: HADI 
    // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
    // unfortunately need to reorganize memory layout by warp
    asm(LD_NC_V2 " {%0, %1}, [%2 + 0];" :
        "=r"(code32[0]), "=r"(code32[1]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 8];" :
        "=r"(code32[2]), "=r"(code32[3]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 16];" :
        "=r"(code32[4]), "=r"(code32[5]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 24];" :
        "=r"(code32[6]), "=r"(code32[7]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 32];" :
        "=r"(code32[8]), "=r"(code32[9]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 40];" :
        "=r"(code32[10]), "=r"(code32[11]) : "l"(p));
    asm(LD_NC_V2 " {%0, %1}, [%2 + 48];" :
        "=r"(code32[12]), "=r"(code32[13]) : "l"(p));
    */

    code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
    code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
    code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
    code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
    code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
    code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
    code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
    code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
    code32[8] = *(reinterpret_cast<unsigned int*>(p + 32));
    code32[9] = *(reinterpret_cast<unsigned int*>(p + 36));
    code32[10] = *(reinterpret_cast<unsigned int*>(p + 40));
    code32[11] = *(reinterpret_cast<unsigned int*>(p + 44));
    code32[12] = *(reinterpret_cast<unsigned int*>(p + 48));
    code32[13] = *(reinterpret_cast<unsigned int*>(p + 52));
  }
};

template<>
struct LoadCode32<64> {
  static inline __device__ void load(unsigned int code32[16],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 64;
    /* TODO: HADI
    // FIXME: this is a non-coalesced load
    // unfortunately need to reorganize memory layout by warp
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4];" :
        "=r"(code32[0]), "=r"(code32[1]),
        "=r"(code32[2]), "=r"(code32[3]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 16];" :
        "=r"(code32[4]), "=r"(code32[5]),
        "=r"(code32[6]), "=r"(code32[7]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 32];" :
        "=r"(code32[8]), "=r"(code32[9]),
        "=r"(code32[10]), "=r"(code32[11]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 48];" :
        "=r"(code32[12]), "=r"(code32[13]),
        "=r"(code32[14]), "=r"(code32[15]) : "l"(p));
    */

    code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
    code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
    code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
    code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
    code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
    code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
    code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
    code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
    code32[8] = *(reinterpret_cast<unsigned int*>(p + 32));
    code32[9] = *(reinterpret_cast<unsigned int*>(p + 36));
    code32[10] = *(reinterpret_cast<unsigned int*>(p + 40));
    code32[11] = *(reinterpret_cast<unsigned int*>(p + 44));
    code32[12] = *(reinterpret_cast<unsigned int*>(p + 48));
    code32[13] = *(reinterpret_cast<unsigned int*>(p + 52));  
    code32[14] = *(reinterpret_cast<unsigned int*>(p + 56));
    code32[15] = *(reinterpret_cast<unsigned int*>(p + 60));
  }
};

template<>
struct LoadCode32<96> {
  static inline __device__ void load(unsigned int code32[24],
                                     unsigned char* p,
                                     int offset) {
    p += offset * 96;
    /* TODO: HADI
    // FIXME: this is a non-coalesced load
    // unfortunately need to reorganize memory layout by warp
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4];" :
        "=r"(code32[0]), "=r"(code32[1]),
        "=r"(code32[2]), "=r"(code32[3]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 16];" :
        "=r"(code32[4]), "=r"(code32[5]),
        "=r"(code32[6]), "=r"(code32[7]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 32];" :
        "=r"(code32[8]), "=r"(code32[9]),
        "=r"(code32[10]), "=r"(code32[11]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 48];" :
        "=r"(code32[12]), "=r"(code32[13]),
        "=r"(code32[14]), "=r"(code32[15]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 64];" :
        "=r"(code32[16]), "=r"(code32[17]),
        "=r"(code32[18]), "=r"(code32[19]) : "l"(p));
    asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 80];" :
        "=r"(code32[20]), "=r"(code32[21]),
        "=r"(code32[22]), "=r"(code32[23]) : "l"(p));
    */

    code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
    code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
    code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
    code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
    code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
    code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
    code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
    code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
    code32[8] = *(reinterpret_cast<unsigned int*>(p + 32));
    code32[9] = *(reinterpret_cast<unsigned int*>(p + 36));
    code32[10] = *(reinterpret_cast<unsigned int*>(p + 40));
    code32[11] = *(reinterpret_cast<unsigned int*>(p + 44));
    code32[12] = *(reinterpret_cast<unsigned int*>(p + 48));
    code32[13] = *(reinterpret_cast<unsigned int*>(p + 52));  
    code32[14] = *(reinterpret_cast<unsigned int*>(p + 56));
    code32[15] = *(reinterpret_cast<unsigned int*>(p + 60));  
    code32[16] = *(reinterpret_cast<unsigned int*>(p + 64));
    code32[17] = *(reinterpret_cast<unsigned int*>(p + 68));
    code32[18] = *(reinterpret_cast<unsigned int*>(p + 72));
    code32[19] = *(reinterpret_cast<unsigned int*>(p + 76));
    code32[20] = *(reinterpret_cast<unsigned int*>(p + 80));
    code32[21] = *(reinterpret_cast<unsigned int*>(p + 84));  
    code32[22] = *(reinterpret_cast<unsigned int*>(p + 88));
    code32[23] = *(reinterpret_cast<unsigned int*>(p + 92));
  }
};

} } // namespace

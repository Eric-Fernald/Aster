/* DLPack v0.8 ABI. https://github.com/dmlc/dlpack (Apache 2.0). Trimmed to what Aster exports. */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#include <stddef.h>
#include <stdint.h>

#define DLPACK_MAJOR_VERSION 1
#define DLPACK_MINOR_VERSION 0

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { uint32_t major; uint32_t minor; } DLPackVersion;

typedef enum {
  kDLCPU = 1, kDLCUDA = 2, kDLCUDAHost = 3, kDLOpenCL = 4, kDLVulkan = 7, kDLMetal = 8, kDLVPI = 9,
  kDLROCM = 10, kDLROCMHost = 11, kDLExtDev = 12, kDLCUDAManaged = 13, kDLOneAPI = 14, kDLWebGPU = 15, kDLHexagon = 16
} DLDeviceType;

typedef struct { DLDeviceType device_type; int32_t device_id; } DLDevice;

typedef enum { kDLInt = 0U, kDLUInt = 1U, kDLFloat = 2U, kDLOpaqueHandle = 3U, kDLBfloat = 4U, kDLComplex = 5U, kDLBool = 6U } DLDataTypeCode;

typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;

typedef struct {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

#ifdef __cplusplus
}
#endif
#endif

#ifndef ASTER_C_API_H
#define ASTER_C_API_H

#include <stddef.h>
#include <stdint.h>

#include "dlpack/dlpack.h"

#ifdef _WIN32
#define ASTER_EXPORT __declspec(dllexport)
#else
#define ASTER_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AsterEngine AsterEngine;
typedef struct AsterResult AsterResult;

/* mode: 0 = cpu only, 1 = discrete, 2 = coherent. data_dir may be NULL for defaults. */
ASTER_EXPORT AsterEngine* aster_engine_open(int mode, const char* data_dir, int device_id, char** error_out);
ASTER_EXPORT void aster_engine_close(AsterEngine* e);
ASTER_EXPORT void aster_free_string(char* s);

ASTER_EXPORT int aster_create_table_json(AsterEngine* e, const char* name, const char* schema_json, char** error_out);
ASTER_EXPORT int aster_import_parquet(AsterEngine* e, const char* table, const char* path, char** error_out);
ASTER_EXPORT int aster_compact(AsterEngine* e, const char* table, char** error_out);

/* Substrait binary plan in, result handle out. keep_on_device leaves columns in VRAM for DLPack. */
ASTER_EXPORT AsterResult* aster_query_substrait(AsterEngine* e, const uint8_t* plan, size_t len, int keep_on_device, char** error_out);
ASTER_EXPORT AsterResult* aster_query_substrait_json(AsterEngine* e, const char* json, int keep_on_device, char** error_out);
ASTER_EXPORT void aster_result_free(AsterResult* r);
ASTER_EXPORT int64_t aster_result_num_rows(const AsterResult* r);
ASTER_EXPORT int aster_result_num_columns(const AsterResult* r);
ASTER_EXPORT const char* aster_result_column_name(const AsterResult* r, int i);
ASTER_EXPORT const char* aster_result_column_type(const AsterResult* r, int i);
ASTER_EXPORT char* aster_result_metrics_json(const AsterResult* r);
/* Zero copy DLPack export of one column; the caller owns the capsule and must call its deleter. */
ASTER_EXPORT DLManagedTensor* aster_result_column_dlpack(const AsterResult* r, int i, char** error_out);
/* Arrow C data interface export: fills ArrowArray/ArrowSchema structs allocated by the caller. */
ASTER_EXPORT int aster_result_column_arrow(const AsterResult* r, int i, void* arrow_array, void* arrow_schema, char** error_out);
/* Tab separated text dump for tooling and tests. */
ASTER_EXPORT char* aster_result_to_tsv(const AsterResult* r, int64_t max_rows);

ASTER_EXPORT char* aster_capabilities_tsv(AsterEngine* e);
ASTER_EXPORT char* aster_bandwidth_table(AsterEngine* e);
ASTER_EXPORT const char* aster_version(void);

#ifdef __cplusplus
}
#endif
#endif

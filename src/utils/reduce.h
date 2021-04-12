/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef UTILS_REDUCE_H_
#define UTILS_REDUCE_H_
#include <sys/types.h>
#include <stdio.h>
#include <assert.h>
#include <api/xccl.h>
#include <utils/xccl_log.h>
#include <omp.h>

#define  DO_OP_MAX(_v1, _v2)  (_v1 > _v2 ? _v1 : _v2)
#define  DO_OP_MIN(_v1, _v2)  (_v1 < _v2 ? _v1 : _v2)
#define  DO_OP_SUM(_v1, _v2)  (_v1 + _v2)
#define DO_OP_PROD(_v1, _v2)  (_v1 * _v2)
#define DO_OP_LAND(_v1, _v2)  (_v1 && _v2)
#define DO_OP_BAND(_v1, _v2)  (_v1 & _v2)
#define  DO_OP_LOR(_v1, _v2)  (_v1 || _v2)
#define  DO_OP_BOR(_v1, _v2)  (_v1 | _v2)
#define DO_OP_LXOR(_v1, _v2)  ((!_v1) != (!_v2))
#define DO_OP_BXOR(_v1, _v2)  (_v1 ^ _v2)

#define DO_DT_REDUCE_WITH_OP(src1, src2, dst, count, OP) do {   \
        size_t i;                                               \
        for (i=0; i<count; i++) {                               \
            dst[i] = OP(src1[i], src2[i]);                      \
        }                                                       \
    } while(0)

#define DO_DT_REDUCE(type, op, src1_p, src2_p, dest_p, count) do {      \
        type *src1 = (type *)src1_p;                                    \
        type *src2 = (type *)src2_p;                                    \
        type *dest = (type *)dest_p;                                    \
        switch(op) {                                                    \
        case XCCL_OP_MAX:                                                \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_MAX);   \
            break;                                                      \
        case XCCL_OP_MIN:                                                \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_MIN);   \
            break;                                                      \
        case XCCL_OP_SUM:                                                \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_SUM);   \
            break;                                                      \
        case XCCL_OP_PROD:                                               \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_PROD);  \
            break;                                                      \
        case XCCL_OP_LAND:                                               \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_LAND);  \
            break;                                                      \
        case XCCL_OP_BAND:                                               \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_BAND);  \
            break;                                                      \
        case XCCL_OP_LOR:                                                \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_LOR);   \
            break;                                                      \
        case XCCL_OP_BOR:                                                \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_BOR);   \
            break;                                                      \
        case XCCL_OP_LXOR:                                               \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_LXOR);  \
            break;                                                      \
        case XCCL_OP_BXOR:                                               \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_BXOR);  \
            break;                                                      \
        }                                                               \
    } while(0)

#define DO_DT_REDUCE_FLOAT(type, op, src1_p, src2_p, dest_p, count) do { \
        type *src1 = (type *)src1_p;                                     \
        type *src2 = (type *)src2_p;                                     \
        type *dest = (type *)dest_p;                                     \
        switch(op) {                                                     \
        case XCCL_OP_MAX:                                                \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_MAX);    \
            break;                                                       \
        case XCCL_OP_MIN:                                                \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_MIN);    \
            break;                                                       \
        case XCCL_OP_SUM:                                                \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_SUM);    \
            break;                                                       \
        case XCCL_OP_PROD:                                               \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_PROD);   \
            break;                                                       \
        default:                                                         \
            xccl_error("Floating point dtype does not support "          \
                       "the requested reduce op: %d", op);               \
        }                                                                \
    } while(0)

//             return -1;                                                   \

#define DO_DT_REDUCE_COMPLEX(type, op, src1_p, src2_p, dest_p, count) do { \
        type *src1 = (type *)src1_p;                                       \
        type *src2 = (type *)src2_p;                                       \
        type *dest = (type *)dest_p;                                       \
        switch(op) {                                                       \
        case XCCL_OP_SUM:                                                  \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_SUM);      \
            break;                                                         \
        case XCCL_OP_PROD:                                                 \
            DO_DT_REDUCE_WITH_OP(src1, src2, dest, count, DO_OP_PROD);     \
            break;                                                         \
        default:                                                           \
            xccl_error("Complex dtype does not support the"                \
                       "requested reduce op: %d", op);                     \
            return -1;                                                     \
        }                                                                  \
    } while(0)

static inline
int xccl_dt_reduce(void *sbuf1, void *sbuf2, void *target, size_t count,
                  xccl_dt_t dtype, xccl_op_t op)
{
    switch(dtype) {
    case XCCL_DT_INT16:
#pragma omp parallel
{
        int id = omp_get_thread_num();
        int sz = omp_get_num_threads();
        size_t _count = count / sz;
        sbuf1 = (char *)sbuf1 + (id * (_count * sizeof(int16_t)));
        sbuf2 = (char *)sbuf2 + (id * (_count * sizeof(int16_t)));
        target = (char *)target + (id * (_count * sizeof(int16_t)));
        DO_DT_REDUCE(int16_t, op, sbuf1, sbuf2, target, _count);
}
        break;
    case XCCL_DT_INT32:
#pragma omp parallel
{
        int id = omp_get_thread_num();
        int sz = omp_get_num_threads();
        size_t _count = count / sz;
        sbuf1 = (char *)sbuf1 + (id * (_count * sizeof(int32_t)));
        sbuf2 = (char *)sbuf2 + (id * (_count * sizeof(int32_t)));
        target = (char *)target + (id * (_count * sizeof(int32_t)));
        DO_DT_REDUCE(int32_t, op, sbuf1, sbuf2, target, _count);
}
        break;
    case XCCL_DT_INT64:
#pragma omp parallel
{
        int id = omp_get_thread_num();
        int sz = omp_get_num_threads();
        size_t _count = count / sz;
        sbuf1 = (char *)sbuf1 + (id * (_count * sizeof(int64_t)));
        sbuf2 = (char *)sbuf2 + (id * (_count * sizeof(int64_t)));
        target = (char *)target + (id * (_count * sizeof(int64_t)));
        DO_DT_REDUCE(int64_t, op, sbuf1, sbuf2, target, _count);
}
        break;
    case XCCL_DT_FLOAT32:
        assert(4 == sizeof(float));
#pragma omp parallel
{
        int id = omp_get_thread_num();
        int sz = omp_get_num_threads();
        size_t _count = count / sz;
        sbuf1 = (char *)sbuf1 + (id * (_count * sizeof(float)));
        sbuf2 = (char *)sbuf2 + (id * (_count * sizeof(float)));
        target = (char *)target + (id * (_count * sizeof(float)));
        DO_DT_REDUCE_FLOAT(float, op, sbuf1, sbuf2, target, _count);
}
        break;
    case XCCL_DT_FLOAT64:
        assert(8 == sizeof(double));
#pragma omp parallel
{
        int id = omp_get_thread_num();
        int sz = omp_get_num_threads();
        size_t _count = count / sz;
        sbuf1 = (char *)sbuf1 + (id * (_count * sizeof(double)));
        sbuf2 = (char *)sbuf2 + (id * (_count * sizeof(double)));
        target = (char *)target + (id * (_count * sizeof(double)));
        DO_DT_REDUCE_FLOAT(double, op, sbuf1, sbuf2, target, _count);
}
        break;
    default:
        xccl_error("Unsupported type for reduction (%d)", dtype);
        return -1;
    }
    return 0;
}

#endif

#ifndef TEST_MPI_H
#define TEST_MPI_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <api/xccl.h>
#define STR(x) # x
#define XCCL_CHECK(_call) if (XCCL_OK != (_call)) {\
        fprintf(stderr, "*** XCCL TEST FAIL: %s\n", STR(_call)); \
        MPI_Abort(MPI_COMM_WORLD, -1);             \
    }

extern xccl_team_h xccl_world_team;
int xccl_mpi_test_init(int argc, char **argv,
                       xccl_collective_cap_t coll_types);
int xccl_mpi_test_finalize(void);

#endif
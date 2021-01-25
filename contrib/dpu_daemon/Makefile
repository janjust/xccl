#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

export XCCL_DIR=${XCCL}
export UCX_DIR=${UCX}

CFLAGS = -I$(XCCL_DIR)/include -I$(UCX_DIR)/include
LDFLAGS = -L$(XCCL_DIR)/lib $(XCCL_DIR)/lib/libxccl.so $(UCX_DIR)/lib/libucs.so $(UCX_DIR)/lib/libucp.so -Wl,-rpath -Wl,$(XCCL_DIR)/lib -Wl,-rpath -Wl,$(XCCL_DIR)/lib

rel:
	mpicc -O3 -DNDEBUG -std=c11 $(CFLAGS) -o dpu_server dpu_server.c host_channel.c server_xccl.c $(LDFLAGS)

dbg:
	mpicc -O0 -g -std=c11 $(CFLAGS) -o dpu_server dpu_server.c host_channel.c server_xccl.c $(LDFLAGS)

clean:
	rm -f dpu_server

#include "constmem.hpp"
#include "backend_bc.hpp"
#include "envvar.hpp"
#include "rocshmem/rocshmem.hpp"

namespace rocshmem {

extern Backend *backend;

void init_constant_memory(void) {
  std::string envstr;
  constmem_t constmem_values;

  memset(&constmem_values, 0, sizeof(constmem_t));

  envstr = envvar::gda::alltoallv_wg_algo;

  if (envstr.empty() || envstr.find("GET") != std::string::npos) {
    constmem_values.alltoall_wg_algo = gda::ALLTOALLV_WG_ALGO_GET;
  } else {
    constmem_values.alltoall_wg_algo = gda::ALLTOALLV_WG_ALGO_COPY;
  }

  constmem_values.ipc_first_pe = backend->ipcImpl.ipc_first_pe;
  constmem_values.ipc_stride = backend->ipcImpl.ipc_stride;
  // ipc_shm_size == 0 means IPC disabled (fast early return on device).
  // Non-zero when IPC is available, regardless of stride pattern.
  constmem_values.ipc_shm_size = (backend->ipcImpl.pes_with_ipc_avail != nullptr)
                                 ? backend->ipcImpl.shm_size : 0;

  // Read back the default context pointer from the device symbol
  rocshmem_ctx_t default_ctx_handle{};
  CHECK_HIP(hipMemcpyFromSymbol(&default_ctx_handle,
            HIP_SYMBOL(ROCSHMEM_CTX_DEFAULT),
            sizeof(rocshmem_ctx_t)));
  constmem_values.default_ctx = default_ctx_handle.ctx_opaque;

  int nqp_default = envvar::gda::num_qps_per_pe_default_ctx.get_value();
  int nqp_usr = envvar::gda::num_qps_per_pe_usr_ctx.get_value();
  constmem_values.num_qps_per_pe_default_ctx = nqp_default;
  constmem_values.num_qps_per_pe_usr_ctx = nqp_usr;
  constmem_values.num_qps_default_ctx = nqp_default * backend->getNumPEs();
  constmem_values.num_qps_usr_ctx = nqp_usr * backend->getNumPEs();

  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &constmem_values, sizeof(constmem_t)));
}

}

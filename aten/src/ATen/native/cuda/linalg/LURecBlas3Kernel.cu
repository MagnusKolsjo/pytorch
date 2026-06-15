#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/util/complex.h>


namespace at::native {

namespace {


struct LUNbConfig {
  int nb_small; // outer loop blocking factor when n < nb_crossover_n
  int nb_large; // outer loop blocking factor when n >= nb_crossover_n
};

struct LUTuning {
  int panel_threshold; // rows above this use block size (BS) 1024 tall-panel kernel
  int recnb; // recursive panel base-case width (flat column-by-column below this)
  int nb_crossover_n; // matrix size threshold: n >= this selects nb_large
  LUNbConfig nb_real; // blocking factors for float/double
  LUNbConfig nb_complex; // blocking factors for cfloat/cdouble
};

// Pre-tuned constants per compute capability
static constexpr LUTuning tuning_sm80  = {128,  8, 2048, {64, 256}, {64, 256}};  // A100 (swept 2026-06-10)
static constexpr LUTuning tuning_sm89  = {512, 14, 1024, {48, 256}, {48, 256}};  // L40S (swept 2026-06-10)
static constexpr LUTuning tuning_sm90  = {256, 14, 1024, {52, 256}, {28, 256}};  // H100 (swept 2026-06-09)
static constexpr LUTuning tuning_sm100 = {256,  8, 1536, {16, 256}, {32, 256}};  // GB200 (swept 2026-06-11)

inline LUTuning get_tuning() {
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  const auto compcap = prop->major * 10 + prop->minor;
  switch (compcap) {
    case 80: return tuning_sm80;
    case 89: return tuning_sm89;
    case 90: return tuning_sm90;
    case 100: return tuning_sm100;
    default:
      // Fallback to sm_80
      return tuning_sm80;
  };
}

template<typename scalar_t>
void lu_batched_blas3_kernel_impl(
  scalar_t* dA,
  int batch_count,
  int m,
  int n,
  int64_t matrix_stride,
  int lda,
  int* dpiv,
  int* dinfo,
  const LUTuning& tuning
) {
  // Disable TF32 in GEMMs for accuracy
  NoTF32Guard disable_tf32;

  LUNbConfig nbc;
  if constexpr (c10::is_complex<scalar_t>::value) {
    nbc = tuning.nb_complex;
  } else {
    nbc = tuning.nb_real;
  }
}

} // anonymous namespace

void lu_batched_blas3_kernel(const Tensor& input, const Tensor& pivots, const Tensor& infos) {
  const auto tuning = get_tuning();
  int batch_count = batchCount(input);
  int m = input.size(-2);
  int n = input.size(-1);
  int64_t matrix_stride = matrixStride(input);
  int lda = std::max<int>(input.stride(-1), std::max(1, m));

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "linalg_lu_batched_blas3_kernel", [&] {
    lu_batched_blas3_kernel_impl<scalar_t>(
      input.data_ptr<scalar_t>(), batch_count, m, n, matrix_stride, lda,
      pivots.data_ptr<int>(), infos.data_ptr<int>(),
      tuning
    );
  });
}

} // at::native

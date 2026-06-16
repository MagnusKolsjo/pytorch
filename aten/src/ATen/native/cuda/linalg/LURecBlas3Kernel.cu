#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/util/complex.h>
#include <ATen/native/cuda/MiscUtils.h>


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

// Workspace -- pointer arrays needed by cuBLAS batched TRSM.
// cuBLAS batched TRSM requires device arrays of per-batch pointers (T**).
// We pre-allocate these once and recompute the pointers before each TRSM
// call via build_trms_ptrs_device.
template <typename scalar_t>
struct LUWorkspace {
  LUWorkspace(const Tensor& input) {
    dA_base = input.data_ptr<scalar_t>();
    batch_count = cuda_int_cast(batchCount(input), "batchCount");

    // kLong -- assuming 64 bit addresses
    buffer = at::empty({2, batch_count}, input.options().dtype(at::kLong));
    dL11_array = buffer.select(0, 0).data_ptr<scalar_t*>();
    dA12_array = buffer.select(0, 1).data_ptr<scalar_t*>();
  }

  scalar_t* dA_base;
  int batch_count;
  Tensor buffer;
  scalar_t** dL11_array;
  scalar_t** dA12_array;
};

// Device-side pointer array computation for TRSM.
template <typename scalar_t>
__global__ void build_trsm_ptr_kernel(
  scalar_t* __restrict__ dA, int64_t matrix_stride, int lda, int batch_count,
  scalar_t** __restrict__ dL11_array, int row_off_L11, int col_off_L11,
  scalar_t** __restrict__ dA12_array, int row_off_A12, int col_off_A12
) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_count) return;
  auto* base = dA + b * matrix_stride;
  dL11_array[b] = base + row_off_L11 + static_cast<size_t>(col_off_L11) * lda;
  dA12_array[b] = base + row_off_A12 + static_cast<size_t>(col_off_A12) * lda;
}

template <typename scalar_t>
void build_trsm_ptrs_device(
  scalar_t* dA,
  int64_t matrix_stride,
  LUWorkspace<scalar_t>& ws,
  int lda,
  int row_off_L11, int col_off_L11,
  int row_off_A12, int col_off_A12
) {
  int bc = ws.batch_count;
  int constexpr threads = 64;
  int blocks = (bc + threads - 1) / threads;
  build_trsm_ptr_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    dA, matrix_stride, lda, bc,
    ws.dL11_array, row_off_L11, col_off_L11,
    ws.dA12_array, row_off_A12, col_off_A12
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Apply pivots ipiv[col_start:col_start + nb] to columns [col_lo, col_hi).
// Launches one thread per column with 256-thread blocks.
// Pivots applied sequentially.
template <typename scalar_t>
void batched_apply_pivots(
  scalar_t* dA,
  int64_t matrix_stride,
  int lda,
  int m,
  int col_start,
  int nb,
  const int* dipiv,
  int ipiv_stride,
  int col_lo,
  int col_hi,
  int batch_count
) {
}

template <typename scalar_t>
void lu_batched_blas3_kernel_flat(
  scalar_t* dA,
  int64_t matrix_stride,
  int lda,
  int m,
  int n,
  int col_start,
  int nb,
  int* dipiv,
  int ipiv_stride,
  int* dinfo,
  int batch_count,
  LUWorkspace<scalar_t>& ws,
  const LUTuning& tuning
) {
}

template <typename scalar_t>
void lu_batched_panel_recursive(
  scalar_t* dA,
  int64_t matrix_stride,
  int lda,
  int m,
  int n,
  int col_start,
  int nb,
  int* dipiv,
  int ipiv_stride,
  int* dinfo,
  int batch_count,
  LUWorkspace<scalar_t>& ws,
  const LUTuning& tuning
) {
  if (nb <= 0) return;

  // Base case: use flat panel factorization
  if (nb <= tuning.recnb) {
    lu_batched_blas3_kernel_flat<scalar_t>(
      dA, matrix_stride, lda, m, n,
      col_start, nb,
      dipiv, ipiv_stride, dinfo,
      batch_count, ws, tuning
    );
    return;
  }

  auto n1 = nb / 2;
  auto n2 = nb - n1;

  // 1. Factor left half: columns [col_start, col_start + n1)
  lu_batched_panel_recursive<scalar_t>(
    dA, matrix_stride, lda, m, n,
    col_start, n1,
    dipiv, ipiv_stride, dinfo,
    batch_count, ws, tuning
  );

  // 2. Apply left-half pivots to right half columns [col_start + n1, col_start + nb)
  batched_apply_pivots<scalar_t>(
    dA, matrix_stride, lda, m,
    col_start, n1,
    dipiv, ipiv_stride,
    col_start + n1, col_start + nb, batch_count
  );

  // 3. TRSM: L11 \ A12
  auto m_below = m - col_start - n1;
  auto do_trailing_update = m_below && n1 && n2;
}

template <typename scalar_t>
void lu_batched_blas3_kernel_impl(
  cublasHandle_t handle,
  scalar_t* dA,
  int batch_count,
  int m,
  int n,
  int64_t matrix_stride,
  int lda,
  int* dipiv,
  int* dinfo,
  LUWorkspace<scalar_t>& ws,
  const LUTuning& tuning
) {
  // Disable TF32 in GEMMs for accuracy
  NoTF32Guard disable_tf32;

  // Real/Complex panel config
  LUNbConfig nbc;
  if constexpr (c10::is_complex<scalar_t>::value) {
    nbc = tuning.nb_complex;
  } else {
    nbc = tuning.nb_real;
  }

  // Panel size (columns)
  int nb;
  if (n >= tuning.nb_crossover_n) {
    nb = nbc.nb_large;
  } else {
    nb = nbc.nb_small;
  }

  auto min_mn = std::min(m, n);
  auto ipiv_stride = min_mn;

  // Right-looking blocked LU: step through columns in blocks of nb.
  // Each iteration factors one panel of width actual_nb, then updates the
  // trailing matrix to the right.
  // The panel itself is factored recursively (splitting its width in half
  // down to recnb, same algorithm as MAGMA's dgetrf_recpanel_batched).
  for (int j = 0; j < min_mn; j += nb) {
    auto actual_nb = std::min(nb, min_mn - j);

    // 1. Panel factorization
    // Factor columns [j, j + actual_nb) with rows [j, m).
    // Produces L/U within the panel, and pivot indices ipiv[j:j + actual_nb].
    // Pivots are global row indices (1-based) - rows may be swapped from
    // anywhere in [j, m) into the panel.
    lu_batched_panel_recursive<scalar_t>(
      dA, matrix_stride, lda, m, n,
      j, actual_nb,
      dipiv, ipiv_stride, dinfo,
      batch_count, ws, tuning
    );

    // 2. Propagate pivots to columns outside the panel
    // The panel factorization only swapped rows within columns [j, j + actual_nb).
    // We must apply the same row swaps to the left columns [0, j) and
    // right columns [j + actual_nb, n) so the full row permutation is consistent.
    batched_apply_pivots<scalar_t>(
      dA, matrix_stride, lda, m,
      j, actual_nb,
      dipiv, ipiv_stride,
      0, j, batch_count
    );
    batched_apply_pivots<scalar_t>(
      dA, matrix_stride, lda, m,
      j, actual_nb,
      dipiv, ipiv_stride,
      j + actual_nb, n, batch_count
    );

    // 3. Trailing matrix update
    // After pivoting, the block row looks like:
    //
    // columns:    [0, j)  [j, j + nb)  [j + nb, n)
    // row j:      done    L11 \ U11    U12 (need TRSM)
    // row j + nb: done    L21          A22 (need GEMM)
    //
    // U12: solve L11 @ U12 = A[j:j + nb, j + nb:n] (TRSM)
    // A22: A22 -= L21 @ U12, updating the trailing (m - j - nb) x (n - j - nb) block.
    auto n_right = n - j - actual_nb;
    auto m_below = m - j - actual_nb;
    auto do_trailing_update = n_right && m_below && actual_nb;
    if (n_right && actual_nb) {
      // L11 at (j, j): actual_nb x actual_nb, unit lower triangular
      // A12 at (j, j + actual_nb): actual_nb x n_right - overwritten with U12
      build_trsm_ptrs_device<scalar_t>(
        dA, matrix_stride, ws, lda,
        j, j,
        j, j + actual_nb
      );

      auto constexpr one = static_cast<scalar_t>(1);
      auto constexpr neg_one = static_cast<scalar_t>(-1);
      at::cuda::blas::trsmBatched<scalar_t>(
        handle,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
        actual_nb, n_right, &one,
        ws.dL11_array, lda,
        ws.dA12_array, lda,
        batch_count
      );

      // L12 at (j + actual_nb, j): m_below x actual_nb
      // U12 at (j, j + actual_nb): actula_nb x n_right (from TRSM above)
      // A22 at (j + actual_nb, j + actual_nb): m_below x n_right
      if (do_trailing_update) {
        size_t off_L21 = (j + actual_nb) + static_cast<size_t>(j) * lda;
        size_t off_U12 = j + static_cast<size_t>(j + actual_nb) * lda;
        size_t off_A22 = (j + actual_nb) + static_cast<size_t>(j + actual_nb) * lda;

        at::cuda::blas::bgemm_internal<scalar_t>(
          'n', 'n',
          m_below, n_right, actual_nb,
          neg_one,
          dA + off_L21, lda, matrix_stride,
          dA + off_U12, lda, matrix_stride,
          one,
          dA + off_A22, lda, matrix_stride,
          batch_count
        );
      }
    }
  } // for j in range(0, min(m, n), nb)
}

} // anonymous namespace

void lu_batched_blas3_kernel(const Tensor& input, const Tensor& pivots, const Tensor& infos) {
  const auto tuning = get_tuning();
  int batch_count = cuda_int_cast(batchCount(input), "batchCount");
  int m = cuda_int_cast(input.size(-2), "input.size(-2)");
  int n = cuda_int_cast(input.size(-1), "input.size(-1)");
  int64_t matrix_stride = matrixStride(input);
  // Assuming column-major input with lda >= max(1, m)
  int lda = std::max(cuda_int_cast(input.stride(-1), "input.stride(-1)"), std::max(1, m));

  auto handle = at::cuda::getCurrentCUDABlasHandle();

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "linalg_lu_batched_blas3_kernel", [&] {
    // Workspace for T** arrays in TRSM
    auto ws = LUWorkspace<scalar_t>(input);

    lu_batched_blas3_kernel_impl<scalar_t>(
      handle,
      input.data_ptr<scalar_t>(), batch_count, m, n, matrix_stride, lda,
      pivots.data_ptr<int>(), infos.data_ptr<int>(),
      ws, tuning
    );
  });
}

} // at::native

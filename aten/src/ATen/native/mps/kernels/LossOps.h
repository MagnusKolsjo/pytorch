#pragma once
#include <c10/metal/common.h>

struct CTCLossParams {
  int32_t BLANK;
  uint32_t max_input_length;
  uint32_t max_target_length;
  uint32_t batch_size;
  uint32_t tg_target_stride;
  uint32_t log_probs_time_stride;
  uint32_t log_probs_batch_stride;
  uint32_t log_probs_token_stride;
  uint32_t log_alpha_batch_stride;
  uint32_t log_alpha_time_stride;
  uint32_t log_alpha_target_stride;
};

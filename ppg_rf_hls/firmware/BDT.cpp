#include "BDT.h"
#include "parameters.h"
#include <cstring>

namespace BDT {

bool (*split_fn)(const input_t*, const threshold_t*) =
    !strcmp(splitting_convention,"<=")
        ? [](const input_t *a, const threshold_t *b) { return *a <= *b; }
        : [](const input_t *a, const threshold_t *b) { return *a < *b; };

template<>
void BDT<n_trees, max_depth, n_classes,
         input_arr_t, score_t, threshold_t, unroll>
::tree_scores(input_arr_t x,
              score_t scores[n_trees][fn_classes(n_classes)]) const
{
#pragma HLS inline
#pragma HLS ARRAY_PARTITION variable=scores dim=0

Trees:
  for (int i = 0; i < n_trees; i++) {
  Classes:
    for (int j = 0; j < fn_classes(n_classes); j++) {
      scores[i][j] = trees[i][j].decision_function(x, split_fn);
    }
  }
}

} // namespace BDT

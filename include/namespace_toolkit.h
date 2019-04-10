// namespace_toolkit defines auxiliary functions

#ifndef BIFET_TOOLKIT_H
#define BIFET_TOOLKIT_H

#include <cassert>
#include <memory>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace toolkit {
// Kronecker product of two vectors into vectorized matrix
// the template type must have size() and operator[] implemented
// 1st argument: row direction (spectral domain) vector, index converts into row
// index of final matrix 2nd argument: col direction (spatial domain) vector,
// index converts into column index of final matrix 3rd argument: result vector,
// Fortran-style (column-wise) strached final matrix rslt_vec = vec(col_vec
// \times row_vec^T) template functions must be implemented in header file
template <typename vector_type>
void Kronecker_product(const vector_type *row_vec, const vector_type *col_vec,
                       vector_type *rslt_vec) {
  assert(rslt_vec->size() == col_vec->size() * row_vec->size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (decltype(row_vec->size()) j = 0; j < row_vec->size(); ++j) {
    const double buffer{(*row_vec)[j]};
    decltype(row_vec->size()) I{j * col_vec->size()};
    for (decltype(col_vec->size()) i = 0; i < col_vec->size(); ++i, ++I) {
      (*rslt_vec)[I] = ((*col_vec)[i]) * (buffer);
    }
  }
}
} // namespace toolkit

#endif

// END

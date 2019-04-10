// Simbox class is designed to wrap up two frames (spatial and spectral)
// DSP (dynamic sparsity pattern) in frames will be collected and assembled into
// SP (sparsity pattern) of the simbox
//
// Simbox class is (theoretically) independent of Galerkin methods
// constraints of two domains are not assembled into one constraint
// since assembling may not be as trivial as Kronecker product (for sparsities)
// the way we handle constraints is as following:
// 0. build simbox SP with constrained dof removed (carried out in
// "assemble_sparsity" function)
// 1. assemble per-quadrature-point system matrix and RHS
// 2. copy local to temporary global cachers with per-domain constraints
// 3. conduct Kronecker product for temporary cachers
// 4. accumulating post product cachers into final results
//
// notice:
// according to the design of (dynamic) sparsity pattern in deal.II lib
// only dynamic sparsity is available for carrying out Kronecker product across
// domains DSP is accessable internally, while SP is public two frames are built
// (or wrapped) directly in constructing Simbox object
//
// by default, base Simbox class is fully in cG (two Frame object)
// as DGFrame is derived from Frame
// theoretically base Simbox can represents any cG-dG combinations in two
// domains

#ifndef BIFET_SIMBOX_H
#define BIFET_SIMBOX_H

#include <cassert>
#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <frame.h>
#include <param.h>

template <int spa_dim, int spe_dim> class Simbox {
public:
  Simbox() = default;
  Simbox(const Param *);
  Simbox(const Simbox<spa_dim, spe_dim> &) = delete;
  Simbox(Simbox<spa_dim, spe_dim> &&) = delete;
  Simbox &operator=(const Simbox<spa_dim, spe_dim> &) = delete;
  Simbox &operator=(Simbox<spa_dim, spe_dim> &&) = delete;
  virtual ~Simbox() = default;
  // spatial frame
  std::unique_ptr<Frame<spa_dim>> spatial_frame;
  // spectral frame
  std::unique_ptr<Frame<spe_dim>> spectral_frame;
  // sparsity pattern
  // built from Kronecker product of two domain sparsities
  std::unique_ptr<dealii::SparsityPattern> sparsity;
  // initialize frames
  virtual void init();
  // refine according to per-cell error
  // if no error given, do global refine in corresponding refinement
  // 1st argument: spatial per-cell error
  // 2nd argument: spectral per-cell error
  virtual void refine(const dealii::Vector<float> *spatial_err = nullptr,
                      const dealii::Vector<float> *spectral_err = nullptr);
  // refine only spatial frame
  // if no error given, do global refine
  // 1st argument: spatial per-cell error
  virtual void
  refine_spatial(const dealii::Vector<float> *spatial_err = nullptr);
  // refine only spectral frame
  // if no error given, do global refine
  // 1st argument: spectral per-cell error
  virtual void
  refine_spectral(const dealii::Vector<float> *spectral_err = nullptr);
  // assemble sparsity
  // Kronecker product of spectral and spatial sparsities
  virtual void assemble_sparsity();
  // Kronecker product of two dynamic sparsities
  // cannot apply it to dealii::SparsityPattern
  // whose iterator does not support row/column index tracing
  virtual void Kronecker_product();
#ifdef NDEBUG
protected:
#endif
  // dynamic sparsity pattern
  std::unique_ptr<dealii::DynamicSparsityPattern> dsp;
};

#endif

// END

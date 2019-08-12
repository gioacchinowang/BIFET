// (non)physical modelling of (spatial and/or spectral) Diffusion tensor

#ifndef BIFET_DIFF_H
#define BIFET_DIFF_H

#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

template <int spa_dim, int spe_dim> class Diffusion {
public:
  Diffusion() = default;
  Diffusion(const Diffusion<spa_dim, spe_dim> &) {}
  Diffusion(const Diffusion<spa_dim, spe_dim> &&) = delete;
  Diffusion &operator=(const Diffusion<spa_dim, spe_dim> &) = delete;
  Diffusion &operator=(Diffusion<spa_dim, spe_dim> &&) = delete;
  virtual ~Diffusion() = default;
  virtual Diffusion *clone() const { return new Diffusion(*this); }
  // spatial diffusion
  // 1st argument: spatial position
  // 2nd argument: spectral position
  // 3rd argument: evolution step time for time-dependent problems
  // return: null tensor
  virtual dealii::Tensor<2, spa_dim, double>
  Dxx(const dealii::Point<spa_dim, double> &,
      const dealii::Point<spe_dim, double> &,
      const double &step_time = 0) const;
  // spectral diffusion
  // 1st argument: spatial position
  // 2nd argument: spectral position
  // 3rd argument: evolution step time for time-dependent problems
  // return: null tensor
  virtual dealii::Tensor<2, spe_dim, double>
  Dqq(const dealii::Point<spa_dim, double> &,
      const dealii::Point<spe_dim, double> &,
      const double &step_time = 0) const;
};

#endif

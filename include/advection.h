// (non)physical modelling of (spatial and/or spectral) Advection vector

#ifndef BIFET_ADVEC_H
#define BIFET_ADVEC_H

#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

template <int spa_dim, int spe_dim> class Advection {
public:
  Advection() = default;
  Advection(const Advection<spa_dim, spe_dim> &) {}
  Advection(const Advection<spa_dim, spe_dim> &&) = delete;
  Advection &operator=(const Advection<spa_dim, spe_dim> &) = delete;
  Advection &operator=(Advection<spa_dim, spe_dim> &&) = delete;
  virtual ~Advection() = default;
  virtual Advection *clone() const { return new Advection(*this); }
  // spatial advection
  // 1st argument: spatial position
  // 2nd argument: spectral position
  // 3rd argument: evolution step time for time-dependent problems
  // return: null tensor
  virtual dealii::Tensor<1, spa_dim, double>
  Axx(const dealii::Point<spa_dim, double> &,
      const dealii::Point<spe_dim, double> &,
      const double &step_time = 0) const;
  // spectral advection
  // 1st argument: spatial position
  // 2nd argument: spectral position
  // 3rd argument: evolution step time for time-dependent problems
  // return: null tensor
  virtual dealii::Tensor<1, spe_dim, double>
  Aqq(const dealii::Point<spa_dim, double> &,
      const dealii::Point<spe_dim, double> &,
      const double &step_time = 0) const;
};

#endif

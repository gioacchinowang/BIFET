// (non)physical modelling of growth rate
// should be used on the RHS of PDE

#ifndef BIFET_GROW_H
#define BIFET_GROW_H

#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <param.h>

// can be used to describe growth/damping/decay/spallation rate
// ON RIGHT HAND SIED OF PDE
template <int spa_dim, int spe_dim> class Growth {
public:
  Growth() = default;
  Growth(const Param *) {}
  Growth(const Growth<spa_dim, spe_dim> &) {}
  Growth(const Growth<spa_dim, spe_dim> &&) = delete;
  Growth &operator=(const Growth<spa_dim, spe_dim> &) = delete;
  Growth &operator=(Growth<spa_dim, spe_dim> &&) = delete;
  virtual ~Growth() = default;
  virtual Growth *clone() const { return new Growth(*this); }
  // growth rate
  // 1st argument: spatial position
  // 2nd argument: spectral position
  // 3rd argument: evolution step time for time-dependent problems
  // return: null scalar growth rate
  virtual double rate(const dealii::Point<spa_dim, double> &,
                      const dealii::Point<spe_dim, double> &,
                      const double &step_time = 0) const;
};

#endif

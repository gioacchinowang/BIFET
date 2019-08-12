// (non)physical modelling of scalar source fields

#ifndef BIFET_SOURCE_H
#define BIFET_SOURCE_H

#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <param.h>

template <int spa_dim, int spe_dim> class Source {
public:
  Source() = default;
  Source(const Param *) {}
  Source(const Source<spa_dim, spe_dim> &) {}
  Source(const Source<spa_dim, spe_dim> &&) = delete;
  Source &operator=(const Source<spa_dim, spe_dim> &) = delete;
  Source &operator=(Source<spa_dim, spe_dim> &&) = delete;
  virtual ~Source() = default;
  virtual Source *clone() const { return new Source(*this); }
  // return scalar source distribution
  // 1st argument: spatial position
  // 2nd argument: spectral position
  // 3rd argument: evolution step time for time-dependent problems
  // return: null source
  virtual double value(const dealii::Point<spa_dim, double> &,
                       const dealii::Point<spe_dim, double> &,
                       const double &step_time = 0) const;
};

#endif

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <advection.h>

template class Advection<1, 1>;
template class Advection<2, 1>;
template class Advection<3, 1>;
template class Advection<1, 2>;
template class Advection<2, 2>;
template class Advection<3, 2>;
template class Advection<1, 3>;
template class Advection<2, 3>;
template class Advection<3, 3>;

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spa_dim, double>
Advection<spa_dim, spe_dim>::Axx(const dealii::Point<spa_dim, double> &,
                                 const dealii::Point<spe_dim, double> &) const {
  return dealii::Tensor<1, spa_dim, double>();
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double>
Advection<spa_dim, spe_dim>::Aqq(const dealii::Point<spa_dim, double> &,
                                 const dealii::Point<spe_dim, double> &) const {
  return dealii::Tensor<1, spe_dim, double>();
}

// END

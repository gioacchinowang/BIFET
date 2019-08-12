#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <diffusion.h>

template class Diffusion<1, 1>;
template class Diffusion<2, 1>;
template class Diffusion<3, 1>;
template class Diffusion<1, 2>;
template class Diffusion<2, 2>;
template class Diffusion<3, 2>;
template class Diffusion<1, 3>;
template class Diffusion<2, 3>;
template class Diffusion<3, 3>;

template <int spa_dim, int spe_dim>
dealii::Tensor<2, spa_dim, double>
Diffusion<spa_dim, spe_dim>::Dxx(const dealii::Point<spa_dim, double> &,
                                 const dealii::Point<spe_dim, double> &,
                                 const double &) const {
  return dealii::Tensor<2, spa_dim, double>();
}

template <int spa_dim, int spe_dim>
dealii::Tensor<2, spe_dim, double>
Diffusion<spa_dim, spe_dim>::Dqq(const dealii::Point<spa_dim, double> &,
                                 const dealii::Point<spe_dim, double> &,
                                 const double &) const {
  return dealii::Tensor<2, spe_dim, double>();
}

// END

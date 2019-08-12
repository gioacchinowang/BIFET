#include <deal.II/base/point.h>

#include <growth.h>

template class Growth<1, 1>;
template class Growth<2, 1>;
template class Growth<3, 1>;
template class Growth<1, 2>;
template class Growth<2, 2>;
template class Growth<3, 2>;
template class Growth<1, 3>;
template class Growth<2, 3>;
template class Growth<3, 3>;

template <int spa_dim, int spe_dim>
double Growth<spa_dim, spe_dim>::rate(const dealii::Point<spa_dim, double> &,
                                      const dealii::Point<spe_dim, double> &,
                                      const double &) const {
  return 0.;
}

// END

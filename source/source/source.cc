#include <deal.II/base/point.h>

#include <simbox.h>
#include <solution.h>
#include <source.h>

template class Source<1, 1>;
template class Source<2, 1>;
template class Source<3, 1>;
template class Source<1, 2>;
template class Source<2, 2>;
template class Source<3, 2>;
template class Source<1, 3>;
template class Source<2, 3>;
template class Source<3, 3>;

template <int spa_dim, int spe_dim>
double
Source<spa_dim, spe_dim>::value(const dealii::Point<spa_dim, double> &,
                                const dealii::Point<spe_dim, double> &) const {
  return 0.;
}

// END

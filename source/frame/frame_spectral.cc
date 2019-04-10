#include <cassert>
#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/fe/fe_q.h>

#include <frame.h>
#include <param.h>

template class Frame_spectral<1>;
template class Frame_spectral<2>;
template class Frame_spectral<3>;

template <int dim> Frame_spectral<dim>::Frame_spectral(const Param *par) {
  // setup finite element order
  this->pol_order = par->pip_set.spectral_pol_order;
  // cache refinement limit
  this->min_refine_lv = par->grid_set.spectral_min_refine_lv;
  this->max_refine_lv = par->grid_set.spectral_max_refine_lv;
  // cache refine/coarsen ratio
  this->refine_ratio = par->grid_set.refine_ratio;
  this->coarsen_ratio = par->grid_set.coarsen_ratio;
  // setup simulation box
  switch (dim) {
  case 1:
    this->pivot_min = dealii::Point<dim, double>(par->grid_set.q1_min);
    this->pivot_max = dealii::Point<dim, double>(par->grid_set.q1_max);
    this->block_nums = {par->grid_set.nq1 - 1};
    break;
  case 2:
    this->pivot_min =
        dealii::Point<dim, double>(par->grid_set.q1_min, par->grid_set.q2_min);
    this->pivot_max =
        dealii::Point<dim, double>(par->grid_set.q1_max, par->grid_set.q2_max);
    this->block_nums = {par->grid_set.nq1 - 1, par->grid_set.nq2 - 1};
    break;
  case 3:
    this->pivot_min = dealii::Point<dim, double>(
        par->grid_set.q1_min, par->grid_set.q2_min, par->grid_set.q3_min);
    this->pivot_max = dealii::Point<dim, double>(
        par->grid_set.q1_max, par->grid_set.q2_max, par->grid_set.q3_max);
    this->block_nums = {par->grid_set.nq1 - 1, par->grid_set.nq2 - 1,
                        par->grid_set.nq3 - 1};
    break;
  default:
    assert(dim > 0 and dim < 4);
    break;
  }
  this->fe = std::make_unique<dealii::FE_Q<dim>>(this->pol_order);
}

// END

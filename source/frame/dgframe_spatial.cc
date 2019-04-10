#include <cassert>
#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/fe/fe_dgq.h>

#include <frame.h>
#include <param.h>

template class dGFrame_spatial<1>;
template class dGFrame_spatial<2>;
template class dGFrame_spatial<3>;

template <int dim> dGFrame_spatial<dim>::dGFrame_spatial(const Param *par) {
  // setup finite element order
  this->pol_order = par->pip_set.spatial_pol_order;
  // cache refinement limit
  this->min_refine_lv = par->grid_set.spatial_min_refine_lv;
  this->max_refine_lv = par->grid_set.spatial_max_refine_lv;
  // cache refine/coarsen ratio
  this->refine_ratio = par->grid_set.refine_ratio;
  this->coarsen_ratio = par->grid_set.coarsen_ratio;
  // setup simulation box
  switch (dim) {
  case 1:
    this->pivot_min = dealii::Point<dim, double>(par->grid_set.x1_min);
    this->pivot_max = dealii::Point<dim, double>(par->grid_set.x1_max);
    this->block_nums = {par->grid_set.nx1 - 1};
    break;
  case 2:
    this->pivot_min =
        dealii::Point<dim, double>(par->grid_set.x1_min, par->grid_set.x2_min);
    this->pivot_max =
        dealii::Point<dim, double>(par->grid_set.x1_max, par->grid_set.x2_max);
    this->block_nums = {par->grid_set.nx1 - 1, par->grid_set.nx2 - 1};
    break;
  case 3:
    this->pivot_min = dealii::Point<dim, double>(
        par->grid_set.x1_min, par->grid_set.x2_min, par->grid_set.x3_min);
    this->pivot_max = dealii::Point<dim, double>(
        par->grid_set.x1_max, par->grid_set.x2_max, par->grid_set.x3_max);
    this->block_nums = {par->grid_set.nx1 - 1, par->grid_set.nx2 - 1,
                        par->grid_set.nx3 - 1};
    break;
  default:
    assert(dim > 0 and dim < 4);
    break;
  }
  this->fe = std::make_unique<dealii::FE_DGQ<dim>>(this->pol_order);
}

// END

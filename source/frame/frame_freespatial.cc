#include <cassert>
#include <iostream>
#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>

#include <frame.h>
#include <param.h>

template class Frame_freespatial<1>;
template class Frame_freespatial<2>;
template class Frame_freespatial<3>;

template <int dim> Frame_freespatial<dim>::Frame_freespatial(const Param *par) {
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
  this->fe = std::make_unique<dealii::FE_Q<dim>>(this->pol_order);
}

template <int dim> void Frame_freespatial<dim>::init() {
  // triangulate simulation box
  dealii::GridGenerator::subdivided_hyper_rectangle(
      *(this->triangulation), this->block_nums, this->pivot_min,
      this->pivot_max, true);
  // if min_refine_lv is 0, no refinement operation will be taken
  this->triangulation->refine_global(this->min_refine_lv);
  // enumerate dof
  this->dof_handler->distribute_dofs(*(this->fe));
  // apply dof to constraints
  this->constraints->clear();
  dealii::DoFTools::make_hanging_node_constraints(*(this->dof_handler),
                                                  *(this->constraints));
  // no strong boundary applied
  this->constraints->close();
  // initialize dynamic sparsity
  this->dsp->reinit(this->dof_handler->n_dofs(), this->dof_handler->n_dofs());
  dealii::DoFTools::make_sparsity_pattern(*(this->dof_handler), *(this->dsp),
                                          *(this->constraints),
                                          /*keep_constrained_dofs=*/false);
  this->sparsity->copy_from(*(this->dsp));

#ifdef VERBOSE
  std::cout << std::endl
            << "===========================================" << std::endl
            << "CGFrame built" << std::endl
            << "Number of active cells: "
            << this->triangulation->n_active_cells() << std::endl
            << "Number of degrees of freedom: " << this->dof_handler->n_dofs()
            << std::endl
            << "Number of non-zero in sparsity: "
            << this->sparsity->n_nonzero_elements() << std::endl
            << "Cells: " << std::endl;
  for (const auto &cell : this->dof_handler->cell_iterators()) {
    std::cout << "idx: " << cell->index() << " lv: " << cell->level()
              << " active: " << cell->active() << std::endl;
  }
#endif
}

template <int dim>
void Frame_freespatial<dim>::refine(const dealii::Vector<float> *err_per_cell) {
  if (err_per_cell) {
    // setup refine/corsen rule
    dealii::GridRefinement::refine_and_coarsen_fixed_number(
        *(this->triangulation), *err_per_cell, this->refine_ratio,
        this->coarsen_ratio);
    // setup refine flags
    if (this->triangulation->n_levels() > this->max_refine_lv) {
      // (time saving)
      auto itr_begin = this->triangulation->begin_active(this->max_refine_lv);
      auto itr_end = this->triangulation->end();
      for (auto cell = itr_begin; cell != itr_end; ++cell)
        cell->clear_refine_flag();
    }
    // setup corsen flags
    // (time saving)
    auto itr_begin = this->triangulation->begin_active(this->min_refine_lv);
    auto itr_end = this->triangulation->end_active(this->min_refine_lv);
    for (auto cell = itr_begin; cell != itr_end; ++cell) {
      cell->clear_coarsen_flag();
    }
    // conduct refine/corsen
    this->triangulation->prepare_coarsening_and_refinement();
    this->triangulation->execute_coarsening_and_refinement();
  } else if (this->triangulation->n_levels() <= this->max_refine_lv) {
    this->triangulation->refine_global();
  }
  this->dof_handler->distribute_dofs(*(this->fe));
  // apply new dof to constraints
  this->constraints->clear();
  dealii::DoFTools::make_hanging_node_constraints(*(this->dof_handler),
                                                  *(this->constraints));
  // no strong boundary applied
  this->constraints->close();
  // apply new dof to dynamic sparsity
  this->dsp->reinit(this->dof_handler->n_dofs(), this->dof_handler->n_dofs());
  // do not follow build-condence scheme of applying hanging node constraints
  // use distribute_local_to_global in assembling system matrices and RHS
  dealii::DoFTools::make_sparsity_pattern(*(this->dof_handler), *(this->dsp),
                                          *(this->constraints),
                                          /*keep_constrained_dofs=*/false);
  this->sparsity->copy_from(*(this->dsp));

#ifdef VERBOSE
  std::cout << std::endl
            << "===========================================" << std::endl
            << "CGFrame built" << std::endl
            << "Number of active cells: "
            << this->triangulation->n_active_cells() << std::endl
            << "Number of degrees of freedom: " << this->dof_handler->n_dofs()
            << std::endl
            << "Number of non-zero in sparsity: "
            << this->sparsity->n_nonzero_elements() << std::endl
            << "Cells: " << std::endl;
  for (const auto &cell : this->dof_handler->cell_iterators()) {
    std::cout << "idx: " << cell->index() << " lv: " << cell->level()
              << " active: " << cell->active() << std::endl;
  }
#endif
}

// END

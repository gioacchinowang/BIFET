#include <iostream>
#include <memory>
#include <utility>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <frame.h>
#include <param.h>

template class dGFrame<1>;
template class dGFrame<2>;
template class dGFrame<3>;

template <int dim> void dGFrame<dim>::init() {
  // triangulate simulation box
  dealii::GridGenerator::subdivided_hyper_rectangle(
      *(this->triangulation), this->block_nums, this->pivot_min,
      this->pivot_max, true);
  // if min_refine_lv is 0, no refinement operation will be taken
  this->triangulation->refine_global(this->min_refine_lv);
  // enumerate dof
  this->dof_handler->distribute_dofs(*(this->fe));
  // initialize dynamic sparsity
  this->dsp->reinit(this->dof_handler->n_dofs(), this->dof_handler->n_dofs());
  dealii::DoFTools::make_flux_sparsity_pattern(*(this->dof_handler),
                                               *(this->dsp));
  // make benefit of empty constraints in distribute_local_to_global
  this->constraints->clear();
  this->constraints->close();
  this->sparsity->copy_from(*(this->dsp));

#ifdef VERBOSE
  std::cout << std::endl
            << "===========================================" << std::endl
            << "dGFrame built" << std::endl
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
void dGFrame<dim>::refine(const dealii::Vector<float> *err_per_cell) {
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
  // make benefit of empty constraints in distribute_local_to_global
  this->constraints->clear();
  this->constraints->close();
  // apply new dof to dynamic sparsity
  this->dsp->reinit(this->dof_handler->n_dofs(), this->dof_handler->n_dofs());
  // do not follow build-condence scheme of applying hanging node constraints
  // use distribute_local_to_global in assembling system matrices and RHS
  dealii::DoFTools::make_flux_sparsity_pattern(*(this->dof_handler),
                                               *(this->dsp));
  this->sparsity->copy_from(*(this->dsp));

#ifdef VERBOSE
  std::cout << std::endl
            << "===========================================" << std::endl
            << "dGFrame refined" << std::endl
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

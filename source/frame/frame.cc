#include <iostream>
#include <memory>
#include <utility>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <frame.h>
#include <param.h>

template class Frame<1>;
template class Frame<2>;
template class Frame<3>;

template <int dim> Frame<dim>::Frame() {
  this->triangulation = std::make_unique<dealii::Triangulation<dim>>();
  this->dof_handler =
      std::make_unique<dealii::DoFHandler<dim>>(*(this->triangulation));
  this->dsp = std::make_unique<dealii::DynamicSparsityPattern>();
  this->sparsity = std::make_unique<dealii::SparsityPattern>();
  this->boundary = std::make_unique<Frame<dim>::Boundary>();
  this->bfmap =
      std::make_unique<std::map<dealii::types::boundary_id,
                                const dealii::Function<dim, double> *>>();
  this->constraints = std::make_unique<dealii::ConstraintMatrix>();
}

template <int dim>
double Frame<dim>::Boundary::value(const dealii::Point<dim, double> &,
                                   const unsigned int) const {
  return 0.;
}

template <int dim> void Frame<dim>::bfmap_init() {
  this->bfmap->clear();
  auto bid = this->triangulation->get_boundary_ids();
  for (auto &i : bid)
    this->bfmap->insert(std::make_pair(i, this->boundary.get()));
}

template <int dim> void Frame<dim>::init() {
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
  // apply strong boundary
  this->bfmap_init();
  dealii::VectorTools::interpolate_boundary_values(
      *(this->dof_handler), *(this->bfmap), *(this->constraints));
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
void Frame<dim>::refine(const dealii::Vector<float> *err_per_cell) {
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
  // apply strong boundary
  this->bfmap_init();
  dealii::VectorTools::interpolate_boundary_values(
      *(this->dof_handler), *(this->bfmap), *(this->constraints));
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
            << "CGFrame refined" << std::endl
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

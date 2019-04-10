#include <cassert>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/matrix_tools.h>
#ifndef NDEBUG
#include <deal.II/numerics/vector_tools.h>
#endif

#include <namespace_toolkit.h>
#include <simbox.h>
#include <solution.h>
#include <system.h>

template class System<1, 1>;
template class System<2, 1>;
template class System<3, 1>;
template class System<1, 2>;
template class System<2, 2>;
template class System<3, 2>;
template class System<1, 3>;
template class System<2, 3>;
template class System<3, 3>;

template <int spa_dim, int spe_dim> System<spa_dim, spe_dim>::System() {
  this->Mx = std::make_unique<dealii::SparseMatrix<double>>();
  this->Mq = std::make_unique<dealii::SparseMatrix<double>>();
  this->Mxq = std::make_unique<dealii::SparseMatrix<double>>();
  this->mass_Mxq = std::make_unique<dealii::SparseMatrix<double>>();
  this->Rxq = std::make_unique<dealii::Vector<double>>();
  this->Rx = std::make_unique<dealii::Vector<double>>();
  this->Rq = std::make_unique<dealii::Vector<double>>();
  this->op = std::make_unique<typename System<spa_dim, spe_dim>::Operator>();
  this->rhs = std::make_unique<typename System<spa_dim, spe_dim>::RHS>();
  this->source = std::make_unique<Source<spa_dim, spe_dim>>();
  this->diffusion = std::make_unique<Diffusion<spa_dim, spe_dim>>();
  this->advection = std::make_unique<Advection<spa_dim, spe_dim>>();
  this->growth = std::make_unique<Growth<spa_dim, spe_dim>>();
}

template <int spa_dim, int spe_dim>
System<spa_dim, spe_dim>::System(const Param *) {
  this->Mx = std::make_unique<dealii::SparseMatrix<double>>();
  this->Mq = std::make_unique<dealii::SparseMatrix<double>>();
  this->Mxq = std::make_unique<dealii::SparseMatrix<double>>();
  this->mass_Mxq = std::make_unique<dealii::SparseMatrix<double>>();
  this->Rxq = std::make_unique<dealii::Vector<double>>();
  this->Rx = std::make_unique<dealii::Vector<double>>();
  this->Rq = std::make_unique<dealii::Vector<double>>();
  this->op = std::make_unique<typename System<spa_dim, spe_dim>::Operator>();
  this->rhs = std::make_unique<typename System<spa_dim, spe_dim>::RHS>();
  this->diffusion = std::make_unique<Diffusion<spa_dim, spe_dim>>();
  this->advection = std::make_unique<Advection<spa_dim, spe_dim>>();
  this->source = std::make_unique<Source<spa_dim, spe_dim>>();
  this->growth = std::make_unique<Growth<spa_dim, spe_dim>>();
}

template <int spa_dim, int spe_dim>
System<spa_dim, spe_dim>::System(const System<spa_dim, spe_dim> &s) {
  this->Mx = std::make_unique<dealii::SparseMatrix<double>>();
  this->Mq = std::make_unique<dealii::SparseMatrix<double>>();
  this->Mxq = std::make_unique<dealii::SparseMatrix<double>>();
  this->mass_Mxq = std::make_unique<dealii::SparseMatrix<double>>();
  this->Rx = std::make_unique<dealii::Vector<double>>();
  this->Rq = std::make_unique<dealii::Vector<double>>();
  this->Rxq = std::make_unique<dealii::Vector<double>>();
  this->diffusion.reset(s.diffusion->clone());
  this->advection.reset(s.advection->clone());
  this->source.reset(s.source->clone());
  this->growth.reset(s.growth->clone());
  this->op.reset(s.op->clone());
  this->rhs.reset(s.rhs->clone());
}

template <int spa_dim, int spe_dim>
System<spa_dim, spe_dim>::System(System<spa_dim, spe_dim> &&s) {
  this->Mx.reset(s.Mx.release());
  this->Mq.reset(s.Mq.release());
  this->Mxq.reset(s.Mxq.release());
  this->mass_Mxq.reset(s.mass_Mxq.release());
  this->Rx.reset(s.Rx.release());
  this->Rq.reset(s.Rq.release());
  this->Rxq.reset(s.Rxq.release());
  this->diffusion.reset(s.diffusion.release());
  this->advection.reset(s.advection.release());
  this->source.reset(s.source.release());
  this->growth.reset(s.growth.release());
  this->op.reset(s.op.release());
  this->rhs.reset(s.rhs.release());
}

template <int spa_dim, int spe_dim>
void System<spa_dim, spe_dim>::init(const Simbox<spa_dim, spe_dim> *simbox,
                                    const double &step_time) {
  this->op->init(this, simbox, step_time);
  this->rhs->init(this, simbox, step_time);
}

template <int spa_dim, int spe_dim>
void System<spa_dim, spe_dim>::refine(const Simbox<spa_dim, spe_dim> *simbox,
                                      const double &step_time) {
#ifdef _OPENMP
  auto omp_Mxq =
      std::make_unique<dealii::SparseMatrix<double>>(*(simbox->sparsity));
  auto omp_Rxq = std::make_unique<dealii::Vector<double>>(
      simbox->spatial_frame->dof_handler->n_dofs() *
      simbox->spectral_frame->dof_handler->n_dofs());
#pragma omp parallel
  {
    std::unique_ptr<System<spa_dim, spe_dim>> private_system;
    private_system.reset(this->clone());
    private_system->init(simbox, step_time);
#pragma omp critical
    omp_Mxq->add(1., *(private_system->Mxq));
#pragma omp critical
    omp_Rxq->add(1., *(private_system->Rxq));
  }
  this->Mxq.reset(omp_Mxq.release());
  this->Rxq.reset(omp_Rxq.release());
#else
  this->init(simbox, step_time);
#endif
}

// by default in base class
// construct Mxq from two mass matrices
template <int spa_dim, int spe_dim>
void System<spa_dim, spe_dim>::Operator::init(
    System<spa_dim, spe_dim> *system, const Simbox<spa_dim, spe_dim> *simbox,
    const double &) {
  // auxiliary objects needed for conduct discretized integration
  auto spatial_quadrature_formula = std::make_unique<dealii::QGauss<spa_dim>>(
      simbox->spatial_frame->fe->degree + 1);
  auto spectral_quadrature_formula = std::make_unique<dealii::QGauss<spe_dim>>(
      simbox->spectral_frame->fe->degree + 1);
  // fe_values in spatial domain
  // update_gradients is critical
  auto spatial_fev = std::make_unique<dealii::FEValues<spa_dim>>(
      *(simbox->spatial_frame->fe), *spatial_quadrature_formula,
      dealii::update_values | dealii::update_quadrature_points |
          dealii::update_JxW_values);
  // fe_values in spectral domain
  // update_values is critical
  auto spectral_fev = std::make_unique<dealii::FEValues<spe_dim>>(
      *(simbox->spectral_frame->fe), *spectral_quadrature_formula,
      dealii::update_values | dealii::update_quadrature_points |
          dealii::update_JxW_values);
  // spatial/spectral domain dofs per cell
  const unsigned int spatial_dpc = spatial_fev->dofs_per_cell;
  const unsigned int spectral_dpc = spectral_fev->dofs_per_cell;
  // spatial/spectral domain quadrature points per cell
  const unsigned int spatial_q_points = spatial_quadrature_formula->size();
  const unsigned int spectral_q_points = spectral_quadrature_formula->size();
  // local 2 global indeces holder
  auto spatial_l2g =
      std::make_unique<std::vector<dealii::types::global_dof_index>>(
          spatial_dpc);
  auto spectral_l2g =
      std::make_unique<std::vector<dealii::types::global_dof_index>>(
          spectral_dpc);
  // allocate temporary local (per-cell) matrix holder
  auto cell_Mx =
      std::make_unique<dealii::FullMatrix<double>>(spatial_dpc, spatial_dpc);
  auto cell_Mq =
      std::make_unique<dealii::FullMatrix<double>>(spectral_dpc, spectral_dpc);
  // allocate memory for Mx and Mq
  system->Mx->reinit(*(simbox->spatial_frame->sparsity));
  system->Mq->reinit(*(simbox->spectral_frame->sparsity));
  // allocate memory for Mxq
  system->Mxq->reinit(*(simbox->sparsity));
  // fill Mx as mass matrix
#ifdef _OPENMP
  system->omp_cell_distribute(simbox);
  for (auto spatial_cell = system->it_start; spatial_cell != system->it_end;
       ++spatial_cell)
#else
  for (const auto &spatial_cell :
       simbox->spatial_frame->dof_handler->active_cell_iterators())
#endif
  {
    spatial_fev->reinit(spatial_cell);
    // from per-cell indeces to global indeces
    spatial_cell->get_dof_indices(*spatial_l2g);
    for (unsigned int spatial_qid = 0; spatial_qid < spatial_q_points;
         ++spatial_qid) {
      for (dealii::types::global_dof_index i = 0; i < spatial_dpc; ++i) {
        for (dealii::types::global_dof_index j = 0; j < spatial_dpc; ++j) {
          cell_Mx->set(i, j,
                       spatial_fev->shape_value(i, spatial_qid) *
                           spatial_fev->shape_value(j, spatial_qid) *
                           spatial_fev->JxW(spatial_qid));
        } // j
      }   // i
      simbox->spatial_frame->constraints->distribute_local_to_global(
          *cell_Mx, *spatial_l2g, *(system->Mx));
    } // spatial_q
  }   // spatial_cell
  // fill Mq as mass matrix
  for (const auto &spectral_cell :
       simbox->spectral_frame->dof_handler->active_cell_iterators()) {
    spectral_fev->reinit(spectral_cell);
    // from per-cell indeces to global indeces
    spectral_cell->get_dof_indices(*spectral_l2g);
    for (unsigned int spectral_qid = 0; spectral_qid < spectral_q_points;
         ++spectral_qid) {
      for (dealii::types::global_dof_index alpha = 0; alpha < spectral_dpc;
           ++alpha) {
        for (dealii::types::global_dof_index beta = 0; beta < spectral_dpc;
             ++beta) {
          cell_Mq->set(alpha, beta,
                       spectral_fev->shape_value(alpha, spectral_qid) *
                           spectral_fev->shape_value(beta, spectral_qid) *
                           spectral_fev->JxW(spectral_qid));
        } // beta
      }   // alpha
      simbox->spectral_frame->constraints->distribute_local_to_global(
          *cell_Mq, *spectral_l2g, *(system->Mq));
    } // spectral_q
  }   // spectral_cell
  // invoke Kronecker productor
  system->Operator_Kronecker_accumulate(simbox);
}

// skip 0*0 operations works in serial assembling
// while not improving much performance in multi-threading mode
template <int spa_dim, int spe_dim>
void System<spa_dim, spe_dim>::Operator_Kronecker_accumulate(
    const Simbox<spa_dim, spe_dim> *simbox) {
  // spectral on left, spatial on right
  // loop through non-zero entries in spectral dsp
  //------------------- serial -----------------------------------------------
  auto it_left = simbox->spectral_frame->dsp->begin();
  const auto end_left = simbox->spectral_frame->dsp->end();
  for (; it_left != end_left; ++it_left) {
    //------------------- parallel ---------------------------------------------
    /*
     unsigned int left_length {0};
     auto begin_left = simbox->spectral_frame->dsp->begin();
     const auto end_left = simbox->spectral_frame->dsp->end();
     for(auto it = begin_left;it!=end_left;++it){
     left_length++;
     }
     #ifdef _OPENMP
     #pragma omp parallel for //schedule(static)
     #endif
     for(unsigned int l=0;l<left_length;++l){
     auto it_left = begin_left;
     for(unsigned int ll=0;ll<l;++ll){
     it_left++;
     }
     */
    //----------------------------------------------------------------------
    auto alpha = it_left->row();
    auto beta = it_left->column();
    // time saver
    if ((*(this->Mq))(alpha, beta) == 0)
      continue;
    const double buffer{(*(this->Mq))(alpha, beta)};
    // loop through non-zero entries in spatial dsp
    auto it_right = simbox->spatial_frame->dsp->begin();
    auto end_right = simbox->spatial_frame->dsp->end();
    for (; it_right != end_right; ++it_right) {
      auto i = it_right->row();
      auto j = it_right->column();
      // time saver
      if ((*(this->Mx))(i, j) == 0)
        continue;
      auto I = alpha * simbox->spatial_frame->dsp->n_rows() + i;
      auto J = beta * simbox->spatial_frame->dsp->n_cols() + j;
      // if (I,J) points to zero, "set" will throw an error
      this->Mxq->add(I, J, buffer * (*(this->Mx))(i, j));
    }
  }
}

template <int spa_dim, int spe_dim>
void System<spa_dim, spe_dim>::RHS::init(System<spa_dim, spe_dim> *system,
                                         const Simbox<spa_dim, spe_dim> *simbox,
                                         const double &) {
  // auxiliary objects needed for conduct discretized integration
  auto spatial_quadrature_formula = std::make_unique<dealii::QGauss<spa_dim>>(
      simbox->spatial_frame->fe->degree + 1);
  auto spectral_quadrature_formula = std::make_unique<dealii::QGauss<spe_dim>>(
      simbox->spectral_frame->fe->degree + 1);
  // fe_values in spatial domain
  auto spatial_fev = std::make_unique<dealii::FEValues<spa_dim>>(
      *(simbox->spatial_frame->fe), *spatial_quadrature_formula,
      dealii::update_values | dealii::update_quadrature_points |
          dealii::update_JxW_values);
  // fe_values in spectral domain
  auto spectral_fev = std::make_unique<dealii::FEValues<spe_dim>>(
      *(simbox->spectral_frame->fe), *spectral_quadrature_formula,
      dealii::update_values | dealii::update_quadrature_points |
          dealii::update_JxW_values);
  // spatial/spectral domain dofs per cell
  const unsigned int spatial_dpc = spatial_fev->dofs_per_cell;
  const unsigned int spectral_dpc = spectral_fev->dofs_per_cell;
  // spatial/spectral domain quadrature points per cell
  const unsigned int spatial_q_points = spatial_quadrature_formula->size();
  const unsigned int spectral_q_points = spectral_quadrature_formula->size();
  // local 2 global indeces holder
  auto spatial_l2g =
      std::make_unique<std::vector<dealii::types::global_dof_index>>(
          spatial_dpc);
  auto spectral_l2g =
      std::make_unique<std::vector<dealii::types::global_dof_index>>(
          spectral_dpc);
  // allocate temporary local (per-cell) spatial RHS and spectral RHS vector
  // holder
  auto spatial_cell_rhs = std::make_unique<dealii::Vector<double>>(spatial_dpc);
  auto spectral_cell_rhs =
      std::make_unique<dealii::Vector<double>>(spectral_dpc);
  // allocate global system RHS
  system->Rxq->reinit(simbox->spatial_frame->dof_handler->n_dofs() *
                      simbox->spectral_frame->dof_handler->n_dofs());

  // fill RHS "matrix" with values
  // (integration with base functions over two domains)
  // loop over spatial/spectral domain cells
#ifdef _OPENMP
  system->omp_cell_distribute(simbox);
  for (auto spatial_cell = system->it_start; spatial_cell != system->it_end;
       ++spatial_cell)
#else
  for (const auto &spatial_cell :
       simbox->spatial_frame->dof_handler->active_cell_iterators())
#endif
  {
    spatial_fev->reinit(spatial_cell);
    // from per-cell indeces to global indeces
    spatial_cell->get_dof_indices(*spatial_l2g);
    for (const auto &spectral_cell :
         simbox->spectral_frame->dof_handler->active_cell_iterators()) {
      spectral_fev->reinit(spectral_cell);
      // from per-cell indeces to global indeces
      spectral_cell->get_dof_indices(*spectral_l2g);
      // discrete integration over spectral domain per cell
      for (unsigned int spectral_qid = 0; spectral_qid < spectral_q_points;
           ++spectral_qid) {
        // prepare spectral cell rhs
        for (dealii::types::global_dof_index alpha = 0; alpha < spectral_dpc;
             ++alpha) {
          (*spectral_cell_rhs)[alpha] =
              spectral_fev->shape_value(alpha, spectral_qid) *
              spectral_fev->JxW(spectral_qid);
        }
        // (clean cacher)
        system->Rq->reinit(simbox->spectral_frame->dof_handler->n_dofs());
        // (convert local vector to global vector, with per-domain constraints
        // applied)
        simbox->spectral_frame->constraints->distribute_local_to_global(
            *spectral_cell_rhs, *spectral_l2g, *(system->Rq));
        // discrete integration over spatial domain per cell
        for (unsigned int spatial_qid = 0; spatial_qid < spatial_q_points;
             ++spatial_qid) {
          const double coefficient{system->source->value(
              spatial_fev->quadrature_point(spatial_qid),
              spectral_fev->quadrature_point(spectral_qid))};
          // prepare spatial cell rhs
          for (dealii::types::global_dof_index i = 0; i < spatial_dpc; ++i) {
            (*spatial_cell_rhs)[i] = coefficient *
                                     spatial_fev->shape_value(i, spatial_qid) *
                                     spatial_fev->JxW(spatial_qid);
          }
          // (clean cacher)
          system->Rx->reinit(simbox->spatial_frame->dof_handler->n_dofs());
          // (convert local vector to global vector, with per-domain constraints
          // applied)
          simbox->spatial_frame->constraints->distribute_local_to_global(
              *spatial_cell_rhs, *spatial_l2g, *(system->Rx));
          // accumulate system rhs with Kronecker product
          system->RHS_Kronecker_accumulate();
        } // spatial_qid
      }   // spectral_qid
    }     // spectral_cell loop
  }       // spatial_cell loop
}

template <int spa_dim, int spe_dim>
void System<spa_dim, spe_dim>::RHS_Kronecker_accumulate() {
  // check size of this->Rxq
  assert(this->Rxq->size() == this->Rq->size() * this->Rx->size());
  //#ifdef _OPENMP
  //#pragma omp parallel for //schedule(static)
  //#endif
  for (unsigned int alpha = 0; alpha < this->Rq->size(); ++alpha) {
    // time saver
    if ((*(this->Rq))[alpha] == 0)
      continue;
    const double buffer{(*(this->Rq))[alpha]};
    unsigned int I = alpha * this->Rx->size();
    for (unsigned int i = 0; i < this->Rx->size(); ++i, ++I) {
      // time saver
      if ((*(this->Rx))[i] == 0)
        continue;
      (*(this->Rxq))[I] += buffer * (*(this->Rx))[i];
    }
  }
}

// mass_matrix
template <int spa_dim, int spe_dim>
void System<spa_dim, spe_dim>::assemble_mass_Mxq(
    const Simbox<spa_dim, spe_dim> *simbox) {
  this->Mx->reinit(*(simbox->spatial_frame->sparsity));
  this->Mq->reinit(*(simbox->spectral_frame->sparsity));
  dealii::MatrixCreator::create_mass_matrix(
      *(simbox->spatial_frame->dof_handler),
      dealii::QGauss<spa_dim>(simbox->spatial_frame->fe->degree + 1),
      *(this->Mx), (const dealii::Function<spa_dim, double> *const) nullptr,
      *(simbox->spatial_frame->constraints));
  dealii::MatrixCreator::create_mass_matrix(
      *(simbox->spectral_frame->dof_handler),
      dealii::QGauss<spe_dim>(simbox->spectral_frame->fe->degree + 1),
      *(this->Mq), (const dealii::Function<spe_dim, double> *const) nullptr,
      *(simbox->spectral_frame->constraints));
  this->mass_Mxq->reinit(*(simbox->sparsity));
  // spectral on left, spatial on right
  // loop through non-zero entries in spectral dsp
  //------------------- serial -----------------------------------------------
  auto it_left = simbox->spectral_frame->dsp->begin();
  const auto end_left = simbox->spectral_frame->dsp->end();
  for (; it_left != end_left; ++it_left) {
    auto alpha = it_left->row();
    auto beta = it_left->column();
    // time saver
    if ((*(this->Mq))(alpha, beta) == 0)
      continue;
    const double buffer{(*(this->Mq))(alpha, beta)};
    // loop through non-zero entries in spatial dsp
    auto it_right = simbox->spatial_frame->dsp->begin();
    auto end_right = simbox->spatial_frame->dsp->end();
    for (; it_right != end_right; ++it_right) {
      auto i = it_right->row();
      auto j = it_right->column();
      // time saver
      if ((*(this->Mx))(i, j) == 0)
        continue;
      auto I = alpha * simbox->spatial_frame->dsp->n_rows() + i;
      auto J = beta * simbox->spatial_frame->dsp->n_cols() + j;
      // if (I,J) points to zero, "set" will throw an error
      this->mass_Mxq->add(I, J, buffer * (*(this->Mx))(i, j));
    }
  }
}

#ifndef NDEBUG
// testing const_Rxq
template <int spa_dim, int spe_dim>
void System<spa_dim, spe_dim>::assemble_const_Rxq(
    const Simbox<spa_dim, spe_dim> *simbox) {
  this->Rx->reinit(simbox->spatial_frame->dof_handler->n_dofs());
  this->Rq->reinit(simbox->spectral_frame->dof_handler->n_dofs());
  this->const_Rxq = std::make_unique<dealii::Vector<double>>(
      simbox->spatial_frame->dof_handler->n_dofs() *
      simbox->spectral_frame->dof_handler->n_dofs());
  auto spa_rhs =
      std::make_unique<System<spa_dim, spe_dim>::test_rhs<spa_dim>>();
  auto spe_rhs =
      std::make_unique<System<spa_dim, spe_dim>::test_rhs<spe_dim>>();
  dealii::VectorTools::create_right_hand_side(
      *(simbox->spatial_frame->dof_handler),
      dealii::QGauss<spa_dim>(simbox->spatial_frame->fe->degree + 1), *spa_rhs,
      *(this->Rx), *(simbox->spatial_frame->constraints));
  dealii::VectorTools::create_right_hand_side(
      *(simbox->spectral_frame->dof_handler),
      dealii::QGauss<spe_dim>(simbox->spectral_frame->fe->degree + 1), *spe_rhs,
      *(this->Rq), *(simbox->spectral_frame->constraints));
  toolkit::Kronecker_product<dealii::Vector<double>>(
      this->Rq.get(), this->Rx.get(), this->const_Rxq.get());
}
#endif

#ifdef _OPENMP
template <int spa_dim, int spe_dim>
void System<spa_dim, spe_dim>::omp_cell_distribute(
    const Simbox<spa_dim, spe_dim> *simbox) {
  auto thread_id = static_cast<unsigned int>(omp_get_thread_num());
  auto nthreads = static_cast<unsigned int>(omp_get_num_threads());
  const unsigned int cell_num{
      simbox->spatial_frame->triangulation->n_active_cells()};
  unsigned int start_pos = thread_id * (cell_num / nthreads) +
                           std::min(thread_id, cell_num % nthreads);
  unsigned int end_pos = (thread_id + 1) * (cell_num / nthreads) +
                         std::min(thread_id + 1, cell_num % nthreads);
  this->it_start = simbox->spatial_frame->dof_handler->begin_active();
  this->it_end = simbox->spatial_frame->dof_handler->begin_active();
  for (unsigned int i = 0; i < end_pos; ++i) {
    this->it_end++;
    if (i < start_pos)
      this->it_start++;
  }
}
#endif

// END

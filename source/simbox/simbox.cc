#include <memory>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <frame.h>
#include <param.h>
#include <simbox.h>

template class Simbox<1, 1>;
template class Simbox<2, 1>;
template class Simbox<3, 1>;
template class Simbox<1, 2>;
template class Simbox<2, 2>;
template class Simbox<3, 2>;
template class Simbox<1, 3>;
template class Simbox<2, 3>;
template class Simbox<3, 3>;

template <int spa_dim, int spe_dim>
Simbox<spa_dim, spe_dim>::Simbox(const Param *par) {
  this->spatial_frame = std::make_unique<Frame_spatial<spa_dim>>(par);
  this->spectral_frame = std::make_unique<Frame_spectral<spe_dim>>(par);
  this->sparsity = std::make_unique<dealii::SparsityPattern>();
  this->dsp = std::make_unique<dealii::DynamicSparsityPattern>();
}

template <int spa_dim, int spe_dim> void Simbox<spa_dim, spe_dim>::init() {
  this->spatial_frame->init();
  this->spectral_frame->init();
  this->assemble_sparsity();
}

template <int spa_dim, int spe_dim>
void Simbox<spa_dim, spe_dim>::refine(
    const dealii::Vector<float> *spatial_err,
    const dealii::Vector<float> *spectral_err) {
  this->spatial_frame->refine(spatial_err);
  this->spectral_frame->refine(spectral_err);
  this->assemble_sparsity();
}

template <int spa_dim, int spe_dim>
void Simbox<spa_dim, spe_dim>::refine_spatial(
    const dealii::Vector<float> *spatial_err) {
  this->spatial_frame->refine(spatial_err);
  this->assemble_sparsity();
}

template <int spa_dim, int spe_dim>
void Simbox<spa_dim, spe_dim>::refine_spectral(
    const dealii::Vector<float> *spectral_err) {
  this->spectral_frame->refine(spectral_err);
  this->assemble_sparsity();
}

template <int spa_dim, int spe_dim>
void Simbox<spa_dim, spe_dim>::assemble_sparsity() {
  // spectral DSP at left, spatial DSP at right
  this->Kronecker_product();
  this->sparsity->copy_from(*(this->dsp));
}

// as this function requires external deal.II lib
// we don't absorb it into toolkit namespace
// left and right DSP don't have to be in the same size
template <int spa_dim, int spe_dim>
void Simbox<spa_dim, spe_dim>::Kronecker_product() {
  // reallocate result DSP
  this->dsp->reinit(
      this->spectral_frame->dsp->n_rows() * this->spatial_frame->dsp->n_rows(),
      this->spectral_frame->dsp->n_cols() * this->spatial_frame->dsp->n_cols());
  // loop through non-zero entries in left DSP
  auto it_left = this->spectral_frame->dsp->begin();
  auto end_left = this->spectral_frame->dsp->end();
  for (; it_left != end_left; ++it_left) {
    auto alpha = it_left->row();
    auto beta = it_left->column();
    // loop through non-zero entries in right DSP
    auto it_right = this->spatial_frame->dsp->begin();
    auto end_right = this->spatial_frame->dsp->end();
    for (; it_right != end_right; ++it_right) {
      // get global indeces
      auto I = alpha * this->spatial_frame->dsp->n_rows() + it_right->row();
      auto J = beta * this->spatial_frame->dsp->n_cols() + it_right->column();
      this->dsp->add(I, J);
    }
  }
}

// END

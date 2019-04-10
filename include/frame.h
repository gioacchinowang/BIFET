// cG/dGFrame class handles following quantities in a SINGLE DOMAIN:
// - simulation frame size and grid/mesh resolution
// - triangulation in spatial and spectral spaces
// - finite element quadrature methods, i.e., FE_Q (continuous Galerkin) or
// FE_DGQ (discontinuous Galerkin)
// - degree-of-freedom handler according to 1. and 2.
// - strong boundaries (nature boundaries should be handled manually in weak
// formulation assembling)
// - constraints according to (in)homogeneous strong boundary (and in continuous
// Galerkin, hanging nodes)
// - dynamic sparsity pattern according to dofs
// - refinement of above quantities (with various methods)
//
// special notice:
// spectral space has dimension either 1 or 3
// spectral space in dimension 1, means isotropic spherical frame
// spectral space in diemsnion 3, means right-hand-side Cartesian frame
//
// spatial space has dimension 1, 2 or 3, always uses right-hand-side Cartesian
// frame 1D spatial frame has z coordinate 2D spatial frame has x-z coordinates
// 3D spatial frame has z-x-y coordinates
//
// only dynamic sparsity is addressed in Frame class, full sparsity will be
// handled in Simbox class
//
// cG/dGFrame class is designed to be hidden behind Simbox class
// "init" function is designed as protected, and invoked internally in
// constructor of derived classes "refine" function is public, which is invoked
// in Simbox->refine
//
// the base class (DG)Frame should not operate since the parameters are not
// initialized DGFrame is considered as a base class, although technically
// speaking it is derived from Frame it provides basic discontinuous Galerkin
// initialization and refienment methods by default in Frame, homogeneous null
// strong boundary condition applies to all boundary surfaces by default in
// DGFrame, no strong boundary condition should be applied

#ifndef BIFET_FRAME_H
#define BIFET_FRAME_H

#include <cassert>
#include <memory>
#include <vector>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <param.h>

// designed for continuous Galerkin method
// null strong boundary on all surfaces
template <int dim> class Frame {
public:
  Frame();
  virtual ~Frame() { this->dof_handler->clear(); }
  Frame(const Frame<dim> &) = delete;
  Frame(Frame<dim> &&) = delete;
  Frame &operator=(const Frame<dim> &) = delete;
  Frame &operator=(Frame<dim> &&) = delete;
  // virtual copy
  virtual Frame *clone() const { return new Frame(); }
  // empty unique ptr to triangulation
  std::unique_ptr<dealii::Triangulation<dim>> triangulation;
  // empty unique ptr to dof handler
  std::unique_ptr<dealii::DoFHandler<dim>> dof_handler;
  // dynamic sparsity pattern
  std::unique_ptr<dealii::DynamicSparsityPattern> dsp;
  // sparsity pattern
  // required for building operator matrices
  std::unique_ptr<dealii::SparsityPattern> sparsity;
  // empty unique ptr to finite element
  std::unique_ptr<dealii::FiniteElement<dim>> fe;
  // affine constraints for hanging nodes after refinement or corsening
  // cannot use unique_ptr here, since there exists no default constructor for
  // this class
  std::unique_ptr<dealii::ConstraintMatrix> constraints;
  // initialize objects
  virtual void init();
  // adaptively refine mesh/grid of current triangulation
  // 1st argument: vector of error_per_cell
  // by default pass in nullptr, carry out global refinement
  virtual void refine(const dealii::Vector<float> *error_per_cell = nullptr);
  // nested strong boundary
  class Boundary : public dealii::Function<dim, double> {
  public:
    Boundary() = default;
    Boundary(const Boundary &) = delete;
    Boundary(Boundary &&) = delete;
    Boundary &operator=(const Boundary &) = delete;
    Boundary &operator=(Boundary &&) = delete;
    virtual ~Boundary() = default;
    double value(const dealii::Point<dim, double> &,
                 const unsigned int component = 0) const override;
  };
  std::unique_ptr<typename Frame<dim>::Boundary> boundary;
  // build up boundary function map
  // can be customized in derived classes
  virtual void bfmap_init();
  // position of pivot points in hyper_rectangular triangulation
  dealii::Point<dim, double> pivot_min, pivot_max;
#ifdef NDEBUG
protected:
#endif
  // initial number of blocks in each dimension
  std::vector<unsigned int> block_nums;
  // refinement limits
  unsigned int max_refine_lv, min_refine_lv;
  // finite element polynomial order, will be used for calculating quadrature
  // point number in System and Propagator
  unsigned int pol_order;
  // refine/coarsen ratio
  double refine_ratio, coarsen_ratio;
  // boundary function map
  std::unique_ptr<std::map<dealii::types::boundary_id,
                           const dealii::Function<dim, double> *>>
      bfmap;
};

// derived class
// designed for discontinuous Galerkin method
// dG method requires no hanging-node constraints
template <int dim> class dGFrame : public Frame<dim> {
public:
  dGFrame() = default;
  virtual ~dGFrame() = default;
  dGFrame(const dGFrame<dim> &) = delete;
  dGFrame(dGFrame<dim> &&) = delete;
  dGFrame &operator=(const dGFrame<dim> &) = delete;
  dGFrame &operator=(dGFrame<dim> &&) = delete;
  virtual dGFrame *clone() const { return new dGFrame(); }
  // initialize objects
  void init() override;
  // adaptively refine mesh/grid of current triangulation
  // 1st argument: vector of error_per_cell
  // by default pass in nullptr, carry out global refinement
  void refine(const dealii::Vector<float> *error_per_cell = nullptr) override;
};

//--------------------------- DERIVED CLASSES --------------------------------//

// continuous Galerkin spatial frame
// (inhomogeneous) null strong boundary to all surfaces
// read spatial setting paramters from Param class
template <int dim> class Frame_spatial final : public Frame<dim> {
public:
  Frame_spatial() = default;
  Frame_spatial(const Param *);
  Frame_spatial(const Frame_spatial<dim> &) = delete;
  Frame_spatial(Frame_spatial<dim> &&) = delete;
  Frame_spatial &operator=(const Frame_spatial<dim> &) = delete;
  Frame_spatial &operator=(Frame_spatial<dim> &&) = delete;
  virtual ~Frame_spatial() = default;
  virtual Frame_spatial *clone() const { return new Frame_spatial(); }
};

// continuous Galerkin spectral frame
// (inhomogeneous) null strong boundary to all surfaces
// read spectral setting paramters from Param class
template <int dim> class Frame_spectral final : public Frame<dim> {
public:
  Frame_spectral() = default;
  Frame_spectral(const Param *);
  Frame_spectral(const Frame_spectral<dim> &) = delete;
  Frame_spectral(Frame_spectral<dim> &&) = delete;
  Frame_spectral &operator=(const Frame_spectral<dim> &) = delete;
  Frame_spectral &operator=(Frame_spectral<dim> &&) = delete;
  virtual ~Frame_spectral() = default;
  virtual Frame_spectral *clone() const { return new Frame_spectral(); }
};

// continuous Galerkin spatial frame
// no strong boundary
// read spatial setting paramters from Param class
template <int dim> class Frame_freespatial final : public Frame<dim> {
public:
  Frame_freespatial() = default;
  Frame_freespatial(const Param *);
  Frame_freespatial(const Frame_freespatial<dim> &) = delete;
  Frame_freespatial(Frame_freespatial<dim> &&) = delete;
  Frame_freespatial &operator=(const Frame_freespatial<dim> &) = delete;
  Frame_freespatial &operator=(Frame_freespatial<dim> &&) = delete;
  virtual ~Frame_freespatial() = default;
  virtual Frame_freespatial *clone() const { return new Frame_freespatial(); }
  void init() override;
  void refine(const dealii::Vector<float> *error_per_cell = nullptr) override;
};

// continuous Galerkin spectral frame
// no strong boundary
// read spectral setting paramters from Param class
template <int dim> class Frame_freespectral final : public Frame<dim> {
public:
  Frame_freespectral() = default;
  Frame_freespectral(const Param *);
  Frame_freespectral(const Frame_freespectral<dim> &) = delete;
  Frame_freespectral(Frame_freespectral<dim> &&) = delete;
  Frame_freespectral &operator=(const Frame_freespectral<dim> &) = delete;
  Frame_freespectral &operator=(Frame_freespectral<dim> &&) = delete;
  virtual ~Frame_freespectral() = default;
  virtual Frame_freespectral *clone() const { return new Frame_freespectral(); }
  void init() override;
  void refine(const dealii::Vector<float> *error_per_cell = nullptr) override;
};

// discontinuous Galerkin spatial frame
// no strong boundary, due to dG
// read spatial setting paramters from Param class
template <int dim> class dGFrame_spatial final : public dGFrame<dim> {
public:
  dGFrame_spatial() = default;
  dGFrame_spatial(const Param *);
  dGFrame_spatial(const dGFrame_spatial<dim> &) = delete;
  dGFrame_spatial(dGFrame_spatial<dim> &&) = delete;
  dGFrame_spatial &operator=(const dGFrame_spatial<dim> &) = delete;
  dGFrame_spatial &operator=(dGFrame_spatial<dim> &&) = delete;
  virtual ~dGFrame_spatial() = default;
  virtual dGFrame_spatial *clone() const { return new dGFrame_spatial(); }
};

// discontinuous Galerkin spectral frame
// no strong boundary, due to dG
// read spectral setting paramters from Param class
template <int dim> class dGFrame_spectral final : public dGFrame<dim> {
public:
  dGFrame_spectral() = default;
  dGFrame_spectral(const Param *);
  dGFrame_spectral(const dGFrame_spectral<dim> &) = delete;
  dGFrame_spectral(dGFrame_spectral<dim> &&) = delete;
  dGFrame_spectral &operator=(const dGFrame_spectral<dim> &) = delete;
  dGFrame_spectral &operator=(dGFrame_spectral<dim> &&) = delete;
  virtual ~dGFrame_spectral() = default;
  virtual dGFrame_spectral *clone() const { return new dGFrame_spectral(); }
};

#endif

// END

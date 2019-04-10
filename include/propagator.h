// Propagator class handles building blocks of solving a single PDE
// can be treated as time-dependent or time-independent
// the single PDE
// defines diffusion, advection, growth within upto 6D phase-space doamin
// as the domain is decomposed into two sub-domains where triangulation is
// applied the PDE is represented by a vectorized linear system (or a
// Sylvester-like equation)
//
// the base class takes default settings from all building blocks
// specific definitions are defined in derived classes
// direct sparse matrix solver is taken by default
// which requires no customized precision limit

#ifndef BIFET_PROP_H
#define BIFET_PROP_H

#include <memory>
#include <string>
#include <vector>

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <param.h>
#include <simbox.h>
#include <solution.h>
#include <system.h>

template <int spa_dim, int spe_dim> class Propagator {
public:
  Propagator() = default;
  Propagator(const Param *);
  Propagator(const Propagator<spa_dim, spe_dim> &) = delete;
  Propagator(Propagator<spa_dim, spe_dim> &&) = delete;
  Propagator &operator=(const Propagator<spa_dim, spe_dim> &) = delete;
  Propagator &operator=(Propagator<spa_dim, spe_dim> &&) = delete;
  virtual ~Propagator() = default;
  // solve single PDE in either time-dependent or time-independent way
  virtual void run();
  // initialize lhs and rhs of PDE
  // constrain hanging nodes
  // default problem in base class is U_new = F(x,q)
  virtual void init();
  // solve the linear system C*U_new=F in single step
  // then apply constraints to solution
  // C: system lhs matrix
  // U_new: vectorized solution matrix
  // F: vectorized system rhs matrix
  virtual void solve_single_step();
  // solve the linear system C*U_new=F with time step info
  virtual void solve_time_step();
  // refine system resolution
  // it provides freedom in choosing refine domains and/or schemes
  // if scheme is adaptive_Kelly, use Kelly as default error estimation scheme
  // if scheme is global, refine globally
  // note that solution refinement has already absorbed simbox refinement
  // all the other refinements rely on simbox refinement
  virtual void refine(const double &step_time = 0.);
  // pseudo_refine only update the linear system without changing mesh
  // useful when real refinment is not necessary after every time-dependent
  // solving
  virtual void pseudo_refine(const double &step_time = 0.);
  // record relative difference between new and old solution vectors
  // push new record to evo_ref
  virtual void evo_record();
  // evolution time difference estimator
  // cache system rhs and system matrix
  virtual void evo_step();
  // refine Rxq_cache and eRxq_cache
  // invoked in refine
  virtual void evo_refine(Simbox<spa_dim, spe_dim> *,
                          const dealii::Vector<float> *,
                          const dealii::Vector<float> *, const double &);
  // same as evo_refine but only applies to spatial domain
  virtual void evo_refine_spatial(Simbox<spa_dim, spe_dim> *,
                                  const dealii::Vector<float> *,
                                  const double &);
  // same as evo_refine but only applies to spectral domain
  virtual void evo_refine_spectral(Simbox<spa_dim, spe_dim> *,
                                   const dealii::Vector<float> *,
                                   const double &);
  // evolution breaker
  // stop time evolution if one of the following conditions is satisfied
  // - reaching maximum step size
  // - reaching relative reference value threshold
  virtual bool evo_break();
  // output density field to disk in vtk form
  // a solution of the PDE system phsycially represents phase-space distribution
  // the corresponding density field requires integration of solution over
  // spectral space
  virtual void density_snapshot(const std::string);
  // output spectral snapshot of solution at given spatial position
  virtual void spectral_snapshot(const dealii::Point<spa_dim, double> &,
                                 const std::string header = "Solution");
  // output spatial snapshot of solution at given spectral position
  virtual void spatial_snapshot(const dealii::Point<spe_dim, double> &,
                                const std::string header = "Solution");
  // output spectral snapshot of source term at given spatial position
  virtual void Rxq_spectral_snapshot(const dealii::Point<spa_dim, double> &,
                                     const std::string header = "Rxq");
  // output spatial snapshot of source term at given spectral position
  virtual void Rxq_spatial_snapshot(const dealii::Point<spe_dim, double> &,
                                    const std::string header = "Rxq");
  // evaluate phase-space solution distribution at given position
  virtual double solution_dist(const dealii::Point<spa_dim, double> &,
                               const dealii::Point<spe_dim, double> &) const;
  // evaluate spatial diff of phase-space solution distribution at given
  // position
  virtual dealii::Tensor<1, spa_dim, double>
  solution_distdx(const dealii::Point<spa_dim, double> &,
                  const dealii::Point<spe_dim, double> &) const;
  // evaluate spectral diff of phase-space solution distribution at given
  // position
  virtual dealii::Tensor<1, spe_dim, double>
  solution_distdq(const dealii::Point<spa_dim, double> &,
                  const dealii::Point<spe_dim, double> &) const;
#ifdef NDEBUG
protected:
#endif
  // solver theta value
  double solver_scheme;
  // time dependency
  bool time_dependency = false;
  // physical time at given step
  // physical time diff to next step
  double step_time = 0, physical_timediff = 0;
  // logical step id
  // logical step limit
  unsigned int step_idx = 0, step_lim = 0;
  // refinement cooldown
  unsigned int refine_cd = 0;
  // evolution reference holder
  // reference value represents convergence status of time-dependent solver
  std::unique_ptr<std::vector<double>> evo_ref;
  // evolution reference limit
  double evo_lim;
  // evolution reference cacher
  float evo_cache;
  // system rhs cache, system extra_rhs cache
  std::unique_ptr<dealii::Vector<double>> Rxq_cache, eRxq_cache;
  // refine scheme
  std::string spatial_refine_scheme, spectral_refine_scheme;
  // refine cues
  bool do_spatial_refine, do_spectral_refine;
  // single step solver max step
  unsigned int iteration;
  // single step solver tolerance
  double tolerance;
  // density field vector holder
  std::unique_ptr<dealii::Vector<double>> density_field;
  // spatial err holder
  std::unique_ptr<dealii::Vector<float>> spatial_err;
  // spectral err holder
  std::unique_ptr<dealii::Vector<float>> spectral_err;
  // global simbox holder
  std::unique_ptr<Simbox<spa_dim, spe_dim>> simbox;
  // system holder
  std::unique_ptr<System<spa_dim, spe_dim>> system;
  // solution holder
  std::unique_ptr<Solution<spa_dim, spe_dim>> solution;
};

#endif

// END

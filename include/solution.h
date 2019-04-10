// Solution class keeps two solution vectors, old and new, for time-dependent
// problems solution vectors are templated by dealii::Vector mathematically,
// solution vector represesnts vec(U) where U is the matrix representative of
// finite element solution we define spatial dof as "rows" index of solution
// "matrix U" spectral dof as "column" index of solution "matrix U"
// vectorization requires storing of "matrix U" in column-wise style
// (corresponds to vec operation) solution refinement contains simbox refinement
// inside Initial class is nested inside, which deals with initial conditions by
// default, initial values are set as constant 1.0

#ifndef BIFET_SOLUTION_H
#define BIFET_SOLUTION_H

#include <memory>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <param.h>
#include <simbox.h>

template <int spa_dim, int spe_dim> class Solution {
public:
  Solution();
  Solution(const Param *);
  virtual ~Solution() = default;
  Solution(const Solution<spa_dim, spe_dim> &);
  virtual Solution *clone() const { return new Solution(*this); }
  Solution(Solution<spa_dim, spe_dim> &&);
  Solution &operator=(const Solution<spa_dim, spe_dim> &) noexcept;
  Solution &operator=(Solution<spa_dim, spe_dim> &&) noexcept;
  // solution vector from present time step PDE system
  std::unique_ptr<dealii::Vector<double>> Snew;
  // solution vector from previous time step PDE system
  std::unique_ptr<dealii::Vector<double>> Sold;
  // pick new solution element in matrix way
  // 1st argument, row index (spatial dof index)
  // 2nd argument, column index (spectral dof index)
  virtual double new_el(const unsigned int &, const unsigned int &) const;
  // writeable element pick
  // 1st argument, row index (spatial dof index)
  // 2nd argument, column index (spectral dof index)
  virtual double &new_el(const unsigned int &, const unsigned int &);
  // pick new solution element in matrix way
  // 1st argument, row index (spatial dof index)
  // 2nd argument, column index (spectral dof index)
  virtual double old_el(const unsigned int &, const unsigned int &) const;
  // writeable element pick
  // 1st argument, row index (spatial dof index)
  // 2nd argument, column index (spectral dof index)
  virtual double &old_el(const unsigned int &, const unsigned int &);
  // init Snew matrix with initial conditions
  virtual void init(const Simbox<spa_dim, spe_dim> *);
  // evaluate new solution at given position in two domains
  virtual double evaluate(const Simbox<spa_dim, spe_dim> *,
                          const dealii::Point<spa_dim, double> &,
                          const dealii::Point<spe_dim, double> &) const;
  // evaluate spatial diff of new solution at given position
  virtual dealii::Tensor<1, spa_dim, double>
  evaluatedx(const Simbox<spa_dim, spe_dim> *,
             const dealii::Point<spa_dim, double> &,
             const dealii::Point<spe_dim, double> &) const;
  // evaluate spectral diff of new solution at given position
  virtual dealii::Tensor<1, spe_dim, double>
  evaluatedq(const Simbox<spa_dim, spe_dim> *,
             const dealii::Point<spa_dim, double> &,
             const dealii::Point<spe_dim, double> &) const;
  // refine Snew
  // dof_handlers may (not) have been refined
  // this function occurs after frame refinement
  // both new and old solution vectors will be reinited
  // new solution vector will be re-interpolated
  // refine functions in frames will be invoked here
  // since it is hard to separate it outof solution vector refinement
  // 1st argument: simbox ptr, simbox will be refined, thus cannot be const
  // 1st argument: error per spatial cell
  // 2nd argument: error per spectral cell
  virtual void refine(Simbox<spa_dim, spe_dim> *,
                      const dealii::Vector<float> *spatial_err = nullptr,
                      const dealii::Vector<float> *spectral_err = nullptr);
  // refine Snew
  // but only in spatial domain
  virtual void
  refine_spatial(Simbox<spa_dim, spe_dim> *,
                 const dealii::Vector<float> *spatial_err = nullptr);
  // refine Snew
  // but only in spectral domain
  virtual void
  refine_spectral(Simbox<spa_dim, spe_dim> *,
                  const dealii::Vector<float> *spectral_err = nullptr);
  // pre solving operation due to inhomoneneous constraints
  // apply constraints.set_zero in each domain
  // invoked in Solution::init
  // https://www.dealii.org/developer/doxygen/deal.II/group__constraints.html
  virtual void pre_constraints(const Simbox<spa_dim, spe_dim> *);
  // post solving operation due to constraints
  // apply constraints.distribute in each domain
  virtual void post_constraints(const Simbox<spa_dim, spe_dim> *);
  // re-allocating Snew matrix with new spatial/spectral handlers
  // focus on reshaping its size instead of imposing initial condition
  virtual void new_reshape(const Simbox<spa_dim, spe_dim> *);
  // re-allocating Sold matrix with new spatial/spectral handlers
  // focus on reshaping its size instead of imposing initial condition
  virtual void old_reshape(const Simbox<spa_dim, spe_dim> *);
  // interpolate initial condition
  // invoked only in init function
  virtual void apply_initial_condition(const Simbox<spa_dim, spe_dim> *);
  // nested spatial initial condition
  class Spatial_initial : public dealii::Function<spa_dim, double> {
  public:
    using dealii::Function<spa_dim, double>::operator=;
    Spatial_initial() = default;
    virtual ~Spatial_initial() = default;
    Spatial_initial(const Spatial_initial &s)
        : dealii::Function<spa_dim, double>(s) {
      *this = s;
    }
    virtual Spatial_initial *clone() const {
      return new Spatial_initial(*this);
    }
    double value(const dealii::Point<spa_dim, double> &,
                 const unsigned int component = 0) const override;
  };
  std::unique_ptr<typename Solution<spa_dim, spe_dim>::Spatial_initial> spatial;
  // nested spectral condition
  class Spectral_initial : public dealii::Function<spe_dim, double> {
  public:
    using dealii::Function<spe_dim, double>::operator=;
    Spectral_initial() = default;
    virtual ~Spectral_initial() = default;
    Spectral_initial(const Spectral_initial &s)
        : dealii::Function<spe_dim, double>(s) {
      *this = s;
    }
    virtual Spectral_initial *clone() const {
      return new Spectral_initial(*this);
    }
    double value(const dealii::Point<spe_dim, double> &,
                 const unsigned int component = 0) const override;
  };
  std::unique_ptr<typename Solution<spa_dim, spe_dim>::Spectral_initial>
      spectral;
  // get protected values
  inline unsigned int oldrows() const { return this->n_rows_old; }
  inline unsigned int newrows() const { return this->n_rows_new; }
  inline unsigned int oldcols() const { return this->n_cols_old; }
  inline unsigned int newcols() const { return this->n_cols_new; }
#ifdef NDEBUG
protected:
#endif
  // number of "rows" and "cols" of solution "matrices"
  unsigned int n_rows_old, n_rows_new, n_cols_old, n_cols_new;
};

#endif

// END

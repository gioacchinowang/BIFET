// System class handles THE (PDE) system sparse matrix and rhs vector
// it holds two nested classes, Operator and RHS
// it applies inhomogeneous strong boundary condition and hanging node
// constraints
//
// System assembling depends on which problem it is associated to
// base class use trivial mass system matrix and default source in rhs
// specific illustrations of system initialization can be found in examples

#ifndef BIFET_SYSTEM_H
#define BIFET_SYSTEM_H

#include <memory>
#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#ifdef _OPENMP
#include <deal.II/dofs/dof_handler.h>
#endif

#include <advection.h>
#include <diffusion.h>
#include <growth.h>
#include <param.h>
#include <simbox.h>
#include <solution.h>
#include <source.h>

template <int spa_dim, int spe_dim> class System {
public:
  System();
  System(const Param *);
  virtual ~System() = default;
  // copy ctor
  // do not copy sparse matrices nor solution vectors
  System(const System<spa_dim, spe_dim> &);
  // virtual copy
  virtual System *clone() const { return new System(*this); }
  System(System<spa_dim, spe_dim> &&);
  System &operator=(const System<spa_dim, spe_dim> &) = delete;
  System &operator=(System<spa_dim, spe_dim> &&) = delete;
  // matrix representation of system matrix
  std::unique_ptr<dealii::SparseMatrix<double>> Mxq;
  // vector representation of system rhs
  std::unique_ptr<dealii::Vector<double>> Rxq;
  // temporary spatial (global) matrix holder
  std::unique_ptr<dealii::SparseMatrix<double>> Mx;
  // temporary spectral (global) matrix holder
  std::unique_ptr<dealii::SparseMatrix<double>> Mq;
  // temporary spatial (global) rhs holder
  std::unique_ptr<dealii::Vector<double>> Rx;
  // temporary spectral (global) rhs holder
  std::unique_ptr<dealii::Vector<double>> Rq;
  // initialize system matrix and rhs vector
  // 1st argument: Simbox object pointer
  // 2nd argument: evolution step time for time-dependent problems
  virtual void init(const Simbox<spa_dim, spe_dim> *,
                    const double &step_time = 0);
  // refine system matrix and rhs vector
  // 1st argument: Simbox object pointer
  // 2nd argument: evolution step time for time-dependent problems
  virtual void refine(const Simbox<spa_dim, spe_dim> *,
                      const double &step_time = 0);
  // Kronecker product accumulate process for Operator
  // Mxq will accumulate the result from each call
  // 1st argument: Simbox object pointer
  virtual void Operator_Kronecker_accumulate(const Simbox<spa_dim, spe_dim> *);
  // Kronecker product accumulate process for RHS
  // Rxq will accumulate the result from each call
  virtual void RHS_Kronecker_accumulate();
  // mass matrix Mxq, for time-dependent problems
  std::unique_ptr<dealii::SparseMatrix<double>> mass_Mxq;
  // assembler for mass_Mxq, for time-dependent problems
  virtual void assemble_mass_Mxq(const Simbox<spa_dim, spe_dim> *);
  // this additional part is designed for unit test
  // nested operator class
  class Operator {
  public:
    Operator() = default;
    Operator(const Operator &) {}
    Operator(Operator &&) = delete;
    Operator &operator=(const Operator &) = delete;
    Operator &operator=(Operator &&) = delete;
    virtual Operator *clone() const { return new Operator(*this); }
    virtual ~Operator() = default;
    // initialize Mxq
    // assemble trivial mass matrix in base class
    // 1st argument: Simbox object pointer
    // 2nd argument: evolution step time for time-dependent problems
    virtual void init(System<spa_dim, spe_dim> *,
                      const Simbox<spa_dim, spe_dim> *,
                      const double &step_time = 0);
  };
  std::unique_ptr<typename System<spa_dim, spe_dim>::Operator> op;
  // nested rhs class
  class RHS {
  public:
    RHS() = default;
    RHS(const RHS &) {}
    RHS(RHS &&) = delete;
    RHS &operator=(const RHS &) = delete;
    RHS &operator=(RHS &&) = delete;
    virtual ~RHS() = default;
    virtual RHS *clone() const { return new RHS(*this); }
    // initialize/fill system_rhs with value info
    // have to be invoked when any Frame setting is changed
    // 1st argument: System object pointer
    // 2nd argument: Simbox object pointer
    // 3rd argument: evolution step time for time-dependent problems
    virtual void init(System<spa_dim, spe_dim> *,
                      const Simbox<spa_dim, spe_dim> *,
                      const double &step_time = 0);
  };
  std::unique_ptr<typename System<spa_dim, spe_dim>::RHS> rhs;
  // physics objects
  std::unique_ptr<Diffusion<spa_dim, spe_dim>> diffusion;
  std::unique_ptr<Advection<spa_dim, spe_dim>> advection;
  std::unique_ptr<Growth<spa_dim, spe_dim>> growth;
  std::unique_ptr<Source<spa_dim, spe_dim>> source;
  //------------------------------------------------------------------------//
#ifdef _OPENMP
  typename dealii::DoFHandler<spa_dim>::active_cell_iterator it_start, it_end;
  // prepare spatial cell it_start and it_end
  // according to thread id and total thread number
  virtual void omp_cell_distribute(const Simbox<spa_dim, spe_dim> *simbox);
#endif
  //------------------------------------------------------------------------//
#ifndef NDEBUG
  // testing const_Rxq
  std::unique_ptr<dealii::Vector<double>> const_Rxq;
  // testing assembler for const_Rxq
  void assemble_const_Rxq(const Simbox<spa_dim, spe_dim> *);
  // testing nested rhs function
  template <int dim> class test_rhs : public dealii::Function<dim, double> {
  public:
    test_rhs() = default;
    virtual ~test_rhs() = default;
    double value(const dealii::Point<dim, double> &,
                 const unsigned int component = 0) const override {
      (void)component;
      // corresponds to the default value set in base Source class
      // whose object is initialized inside System contructor
      return 0.;
    }
  };
#endif
  //------------------------------------------------------------------------//
};

#endif

// END

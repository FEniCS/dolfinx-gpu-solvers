#pragma once

#include "util.h"

#include <chrono>
#include <iostream>


struct SetupTimings
{
    double matrix_assembly = 0.0;
    double nullspace_setup = 0.0;
    double gamg_setup = 0.0;
    double petsc_vector_setup = 0.0;
    double build_owners = 0.0;
    double assemble_diagonal = 0.0;
    double lambda_max = 0.0;

    void reset()
    {
        *this = {};
    }

    void print(double total_setup_time) const
    {
        auto print_one = [total_setup_time](const char* name, double time)
        {
            std::cout << name
                      << ": " << time << " s"
                      << " (" << 100.0 * time / total_setup_time << "%)\n";
        };

        const double measured
            = matrix_assembly
            + nullspace_setup
            + gamg_setup
            + petsc_vector_setup
            + build_owners
            + assemble_diagonal
            + lambda_max;
        
        const double other = total_setup_time - measured;

        std::cout << "\n=== Setup timing breakdown ===\n";
        print_one("P1 matrix assembly", matrix_assembly);
        print_one("P1 nullspace setup", nullspace_setup);
        print_one("P1 GAMG setup", gamg_setup);
        print_one("P1 PETSc vector setup", petsc_vector_setup);
        print_one("Build fine node owners", build_owners);
        print_one("Assemble diagonal", assemble_diagonal);
        print_one("Compute lambda_max", lambda_max);
    }
};


struct SolveTimings
{
  double outer_matvec = 0.0;
  double cg_dots = 0.0;
  double cg_vector_ops = 0.0;
  double pre_smooth = 0.0;
  double vcycle_residual = 0.0;
  double restriction = 0.0;
  double coarse_solve = 0.0;
  double prolongation_correction = 0.0;
  double post_smooth = 0.0;

  void reset()
  {
    *this = {};
  }

  void print(double total_solve_time) const
  {
    const double measured
        = outer_matvec
        + cg_dots
        + cg_vector_ops
        + pre_smooth
        + vcycle_residual
        + restriction
        + coarse_solve
        + prolongation_correction
        + post_smooth;

    const double other = total_solve_time - measured;

    auto print_one = [total_solve_time](const char* name, double time)
    {
      std::cout << name
                << ": " << time << " s"
                << " (" << 100.0 * time / total_solve_time << "%)\n";
    };

    std::cout << "\n=== Solve timing breakdown ===\n";

    print_one("Outer CG matvec", outer_matvec);
    print_one("CG dot products", cg_dots);
    print_one("CG vector operations", cg_vector_ops);
    print_one("Pre-smoothing", pre_smooth);
    print_one("V-cycle residual", vcycle_residual);
    print_one("Restriction", restriction);
    print_one("Coarse solve", coarse_solve);
    print_one("Prolongation + correction", prolongation_correction);
    print_one("Post-smoothing", post_smooth);
    print_one("Other", other);
  }
};

inline SetupTimings setup_timings;
inline SolveTimings solve_timings;

template <typename Function>
void time_gpu(double& accumulator, Function&& function)
{
  device_synchronize();

  const auto start = std::chrono::steady_clock::now();

  function();

  device_synchronize();

  const auto end = std::chrono::steady_clock::now();

  accumulator += std::chrono::duration<double>(end - start).count();
}
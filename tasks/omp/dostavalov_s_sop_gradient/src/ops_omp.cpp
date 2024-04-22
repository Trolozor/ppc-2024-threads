// Copyright 2024 Dostavalov Semyon

#include "omp/dostavalov_s_sop_gradient/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <random>
#include <vector>

namespace dostavalov_s_omp {
std::vector<double> randVector(int size) {
  std::vector<double> random_vector(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(MIN_VALUE, MAX_VALUE);

#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    random_vector[i] = dis(gen);
  }

  return random_vector;
}

std::vector<double> randMatrix(int size) {
  std::vector<double> random_matrix(size * size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(MIN_VALUE, MAX_VALUE);

#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      random_matrix[i * size + j] = dis(gen);
    }
  }
#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < i; ++j) {
      random_matrix[i * size + j] = random_matrix[j * size + i];
    }
  }

  return random_matrix;
}

bool OmpSLAYGradient::pre_processing() {
  internal_order_test();

  const double* input_data_A = reinterpret_cast<double*>(taskData->inputs[0]);

  matrix.resize(taskData->inputs_count[0]);
  std::copy(input_data_A, input_data_A + taskData->inputs_count[0], matrix.begin());

  const double* input_data_B = reinterpret_cast<double*>(taskData->inputs[1]);

  vector.resize(taskData->inputs_count[1]);
  std::copy(input_data_B, input_data_B + taskData->inputs_count[1], vector.begin());

  answer.resize(vector.size(), 0);

  return true;
}

bool OmpSLAYGradient::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[1]) {
    return false;
  }

  if (taskData->inputs_count[1] != taskData->outputs_count[0]) {
    return false;
  }

  return true;
}

bool OmpSLAYGradient::run() {
  internal_order_test();

  long size = vector.size();
  std::vector<double> result(size, 0.0);
  std::vector<double> residual = vector;
  std::vector<double> direction = residual;
  std::vector<double> prev_residual = vector;

  double* matrix_ptr = matrix.data();

  while (true) {
    std::vector<double> A_Dir(size, 0.0);

#pragma omp parallel for shared(matrix_ptr, direction, A_Dir)
    for (long i = 0; i < size; ++i) {
      for (long j = 0; j < size; ++j) {
        A_Dir[i] += matrix_ptr[i * size + j] * direction[j];
      }
    }

    double residual_dot_residual = 0.0;
    double A_Dir_dot_direction = 0.0;

#pragma omp parallel for reduction(+ : residual_dot_residual, A_Dir_dot_direction)
    for (long i = 0; i < size; ++i) {
      residual_dot_residual += residual[i] * residual[i];
      A_Dir_dot_direction += A_Dir[i] * direction[i];
    }

    double alpha = residual_dot_residual / A_Dir_dot_direction;

#pragma omp parallel for
    for (long i = 0; i < static_cast<long>(result.size()); ++i) {
      result[i] += alpha * direction[i];
    }

#pragma omp parallel for
    for (long i = 0; i < static_cast<long>(residual.size()); ++i) {
      residual[i] = prev_residual[i] - alpha * A_Dir[i];
    }

    double new_residual = 0.0;
#pragma omp parallel for reduction(+ : new_residual)
    for (long i = 0; i < static_cast<long>(residual.size()); ++i) {
      new_residual += residual[i] * residual[i];
    }

    if (sqrt(new_residual) < TOLERANCE) {
      break;
    }

    double beta = new_residual / residual_dot_residual;

#pragma omp parallel for
    for (long i = 0; i < size; ++i) {
      direction[i] = residual[i] + beta * direction[i];
    }

    prev_residual = residual;
  }
  answer = result;
  return true;
}

bool OmpSLAYGradient::post_processing() {
  internal_order_test();
  long size = answer.size();

#pragma omp parallel for
  for (long i = 0; i < size; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = answer[i];
  }
  return true;
}

bool check_solution(const std::vector<double>& matrixA, const std::vector<double>& vectorB,
                    const std::vector<double>& solutionC) {
  bool solution_correct = true;

  long size = vectorB.size();
  std::vector<double> A_Sol(size, 0.0);

#pragma omp parallel for shared(matrixA, vectorB, solutionC, A_Sol)
  for (long i = 0; i < size; ++i) {
    A_Sol[i] = 0.0;
    for (long j = 0; j < size; ++j) {
      A_Sol[i] += matrixA[i * size + j] * solutionC[j];
    }
  }

  bool stop = false;
#pragma omp parallel for
  for (long i = 0; i < size; ++i) {
    if (!stop) {
      if (std::abs(A_Sol[i] - vectorB[i]) > TOLERANCE) {
        solution_correct = false;
        stop = true;
      }
    }
  }

  return solution_correct;
}

}  // namespace dostavalov_s_omp

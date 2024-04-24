// Copyright 2024 Dostavalov Semyon

#include "stl/dostavalov_s_sop_gradient/include/ops_stl.hpp"

#include <atomic>
#include <cmath>
#include <future>
#include <random>
#include <thread>
#include <vector>

namespace dostavalov_s_stl {
std::vector<double> randVector(int size) {
  std::vector<double> random_vector(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(MIN_VALUE, MAX_VALUE);

  auto worker = [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      random_vector[i] = dis(gen);
    }
  };

  int num_threads = std::thread::hardware_concurrency();
  int chunk_size = size / num_threads;
  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    threads.emplace_back(worker, start, end);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  threads.clear();

  return random_vector;
}

std::vector<double> randMatrix(int size) {
  std::vector<double> random_matrix(size * size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(MIN_VALUE, MAX_VALUE);

  auto worker = [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      for (int j = i; j < size; ++j) {
        double value = dis(gen);
        random_matrix[i * size + j] = value;
        random_matrix[j * size + i] = value;
      }
    }
  };

  int num_threads = std::thread::hardware_concurrency();
  int chunk_size = size * size / num_threads;
  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    threads.emplace_back(worker, start, end);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  threads.clear();

  return random_matrix;
}

std::atomic<double>& operator+=(std::atomic<double>& atomic_value, double value) {
  double current_value = atomic_value.load();
  while (!atomic_value.compare_exchange_weak(current_value, current_value + value))
    ;
  return atomic_value;
}

bool StlSLAYGradient::pre_processing() {
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

bool StlSLAYGradient::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[1]) {
    return false;
  }

  if (taskData->inputs_count[1] != taskData->outputs_count[0]) {
    return false;
  }

  return true;
}

bool StlSLAYGradient::run() {
  internal_order_test();

  int size = vector.size();
  std::vector<double> result(size, 0.0);
  std::vector<double> residual = vector;
  std::vector<double> direction = residual;
  std::vector<double> prev_residual = vector;

  double* matrix_ptr = matrix.data();

  int num_threads = std::thread::hardware_concurrency();
  int block_size = size / num_threads;

  std::vector<double> A_Dir(size, 0.0);

  while (true) {
    A_Dir.assign(size, 0.0);
    std::vector<std::future<void>> futures(num_threads);

    for (int i = 0; i < num_threads; ++i) {
      int startRow = i * block_size;
      int endRow = (i == num_threads - 1) ? size : (i + 1) * block_size;
      futures[i] = std::async(std::launch::async, [&, startRow, endRow]() {
        for (int i = startRow; i < endRow; ++i) {
          for (int j = 0; j < size; ++j) {
            A_Dir[i] += matrix_ptr[i * size + j] * direction[j];
          }
        }
      });
    }

    for (auto& future : futures) {
      future.wait();
    }

    futures.clear();

    double residual_dot_residual = computeDotProduct(residual, residual);
    double A_Dir_dot_direction = computeDotProduct(A_Dir, direction);
    double alpha = residual_dot_residual / A_Dir_dot_direction;

    updateResult(result, direction, alpha);

    updateResidual(residual, prev_residual, A_Dir, alpha);

    double new_residual = computeDotProduct(residual, residual);

    if (sqrt(new_residual) < TOLERANCE) {
      break;
    }

    double beta = new_residual / residual_dot_residual;

    updateDirection(direction, residual, beta);

    prev_residual = residual;
  }
  answer = result;
  return true;
}

double StlSLAYGradient::computeDotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2) {
  double dot_product = 0.0;

  for (int i = 0; i < static_cast<int>(vec1.size()); ++i) {
    dot_product += vec1[i] * vec2[i];
  }

  return dot_product;
}

void StlSLAYGradient::updateResult(std::vector<double>& result, const std::vector<double>& direction, double alpha) {
  std::vector<std::atomic<double>> atomic_result(result.begin(), result.end());

  auto updateElements = [&](int begin, int end) {
    for (long i = begin; i < end; ++i) {
      atomic_result[i] += alpha * direction[i];
    }
  };

  int num_threads = std::thread::hardware_concurrency();
  int chunk_size = result.size() / num_threads;
  std::vector<std::future<void>> futures;

  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? result.size() : (i + 1) * chunk_size;
    futures.push_back(std::async(std::launch::async, updateElements, start, end));
  }

  for (auto& future : futures) {
    future.get();
  }

  futures.clear();

  for (int i = 0; i < static_cast<int>(result.size()); ++i) {
    result[i] = atomic_result[i];
  }
}

void StlSLAYGradient::updateResidual(std::vector<double>& residual, const std::vector<double>& prev_residual,
                                     const std::vector<double>& A_Dir, double alpha) {
  std::vector<std::atomic<double>> atomic_residual(residual.begin(), residual.end());

  auto updateElements = [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      atomic_residual[i] = prev_residual[i] - alpha * A_Dir[i];
    }
  };

  int num_threads = std::thread::hardware_concurrency();
  int chunk_size = residual.size() / num_threads;
  std::vector<std::future<void>> futures;

  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? residual.size() : (i + 1) * chunk_size;
    futures.push_back(std::async(std::launch::async, updateElements, start, end));
  }

  for (auto& future : futures) {
    future.get();
  }

  futures.clear();

  for (int i = 0; i < static_cast<int>(residual.size()); ++i) {
    residual[i] = atomic_residual[i];
  }
}

void StlSLAYGradient::updateDirection(std::vector<double>& direction, const std::vector<double>& residual,
                                      double beta) {
  std::vector<std::atomic<double>> atomic_direction(direction.begin(), direction.end());

  auto updateElements = [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      atomic_direction[i] = residual[i] + beta * direction[i];
    }
  };

  int num_threads = std::thread::hardware_concurrency();
  int chunk_size = direction.size() / num_threads;
  std::vector<std::future<void>> futures;

  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? direction.size() : (i + 1) * chunk_size;
    futures.push_back(std::async(std::launch::async, updateElements, start, end));
  }

  for (auto& future : futures) {
    future.get();
  }

  futures.clear();

  for (int i = 0; i < static_cast<int>(direction.size()); ++i) {
    direction[i] = atomic_direction[i];
  }
}

bool StlSLAYGradient::post_processing() {
  internal_order_test();
  int size = answer.size();

  auto copyData = [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = answer[i];
    }
  };

  int num_threads = std::thread::hardware_concurrency();
  int chunk_size = size / num_threads;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    threads.emplace_back(copyData, start, end);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  threads.clear();

  return true;
}

bool check_solution(const std::vector<double>& matrixA, const std::vector<double>& vectorB,
                    const std::vector<double>& solutionC) {
  bool solution_correct = true;

  long size = vectorB.size();
  std::vector<double> A_Sol(size, 0.0);

  auto computeASol = [&](long begin, long end) {
    for (long i = begin; i < end; ++i) {
      A_Sol[i] = 0.0;
      for (long j = 0; j < size; ++j) {
        A_Sol[i] += matrixA[i * size + j] * solutionC[j];
      }
    }
  };

  int num_threads = std::thread::hardware_concurrency();
  int chunk_size = size / num_threads;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    threads.emplace_back(computeASol, start, end);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (long i = 0; i < size; ++i) {
    double A_Sol_i = 0.0;
    for (long j = 0; j < size; ++j) {
      A_Sol_i += matrixA[i * size + j] * solutionC[j];
    }
    if (std::abs(A_Sol_i - vectorB[i]) > TOLERANCE) {
      solution_correct = false;
      break;
    }
  }

  threads.clear();

  return solution_correct;
}

}  // namespace dostavalov_s_stl

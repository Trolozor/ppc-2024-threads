// Copyright 2024 Dostavalov Semyon
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/dostavalov_s_sop_gradient/include/ops_omp.hpp"

using namespace dostavalov_s_omp;
const int SIZE = 300;

std::shared_ptr<ppc::core::TaskData> createTaskData(std::vector<double> &matrix, std::vector<double> &vector,
                                                    std::vector<double> &result) {
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();

  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataOmp->inputs_count.emplace_back(matrix.size());

  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataOmp->inputs_count.emplace_back(vector.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataOmp->outputs_count.emplace_back(result.size());

  return taskDataOmp;
}

std::shared_ptr<ppc::core::PerfAttr> start_performance_timer() {
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();

  perfAttr->current_timer = [=] {
    auto current_time_point = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();

    return static_cast<double>(duration) * 1e-9;
  };

  return perfAttr;
}

TEST(dostavalov_s_sop_gradient_omp, test_pipeline) {
  std::vector<double> matrix = randMatrix(SIZE);
  std::vector<double> vector = randVector(SIZE);
  std::vector<double> result(SIZE);

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = createTaskData(matrix, vector, result);

  auto testTaskOmp = std::make_shared<OmpSLAYGradient>(taskDataOmp);

  auto perfAttr = start_performance_timer();

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOmp);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_TRUE(check_solution(matrix, vector, result));
}

TEST(dostavalov_s_sop_gradient_omp, test_task_run) {
  std::vector<double> matrix = randMatrix(SIZE);
  std::vector<double> vector = randVector(SIZE);
  std::vector<double> result(SIZE);

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = createTaskData(matrix, vector, result);

  auto testTaskOmp = std::make_shared<OmpSLAYGradient>(taskDataOmp);

  auto perfAttr = start_performance_timer();

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOmp);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_TRUE(check_solution(matrix, vector, result));
}  // namespace dostavalov_s_seq
// Copyright 2024 Dostavalov Semyon
#include <gtest/gtest.h>

#include <vector>

#include "stl/dostavalov_s_sop_gradient/include/ops_stl.hpp"

using namespace dostavalov_s_stl;

std::shared_ptr<ppc::core::TaskData> createTaskData(std::vector<double> &matrix, std::vector<double> &vector,
                                                    std::vector<double> &result) {
  std::shared_ptr<ppc::core::TaskData> taskDataStl = std::make_shared<ppc::core::TaskData>();

  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataStl->inputs_count.emplace_back(matrix.size());

  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataStl->inputs_count.emplace_back(vector.size());

  taskDataStl->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataStl->outputs_count.emplace_back(result.size());

  return taskDataStl;
}

TEST(dostavalov_s_sop_gradient_tbb, Test_Empty) {
  std::vector<double> matrix = {};
  std::vector<double> vector = {};

  std::vector<double> result(vector.size(), 0.0);
  std::vector<double> true_result = {};

  std::shared_ptr<ppc::core::TaskData> taskDataStl = createTaskData(matrix, vector, result);

  StlSLAYGradient testTaskStl(taskDataStl);

  ASSERT_EQ(testTaskStl.validation(), true);
  testTaskStl.pre_processing();
  testTaskStl.run();
  testTaskStl.post_processing();

  ASSERT_EQ(result, true_result);
}

TEST(dostavalov_s_sop_gradient_tbb, Test_Drob_Size_4) {
  std::vector<double> matrix = {1.2, 3.4, 5.6, 7.8, 3.4, 9.1, 2.3, 4.5, 5.6, 2.3, 8.9, 1.2, 7.8, 4.5, 1.2, 6.7};

  std::vector<double> vector = {1.1, 2.2, 3.3, 4.4};

  std::vector<double> result(vector.size(), 0.0);
  std::vector<double> true_result = {0.521693, 0.025623, 0.032353, 0.0263679};

  std::shared_ptr<ppc::core::TaskData> taskDataStl = createTaskData(matrix, vector, result);

  StlSLAYGradient testTaskStl(taskDataStl);

  ASSERT_EQ(testTaskStl.validation(), true);
  testTaskStl.pre_processing();
  testTaskStl.run();
  testTaskStl.post_processing();

  for (size_t i = 0; i < result.size(); i++) {
    ASSERT_LE(std::abs(true_result[i] - result[i]), TOLERANCE);
  }
}

TEST(dostavalov_s_sop_gradient_tbb, Test_Drob_Size_2) {
  std::vector<double> matrix = {2.1, 3.2, 3.2, 5.4};

  std::vector<double> vector = {1.2, 3.4};

  std::vector<double> result(vector.size(), 0.0);
  std::vector<double> true_result = {-4.0, 3.0};

  std::shared_ptr<ppc::core::TaskData> taskDataStl = createTaskData(matrix, vector, result);

  StlSLAYGradient testTaskStl(taskDataStl);

  ASSERT_EQ(testTaskStl.validation(), true);
  testTaskStl.pre_processing();
  testTaskStl.run();
  testTaskStl.post_processing();

  for (size_t i = 0; i < result.size(); i++) {
    ASSERT_LE(std::abs(true_result[i] - result[i]), TOLERANCE);
  }
}
TEST(dostavalov_s_sop_gradient_tbb, Test_Size_4) {
  std::vector<double> matrix = {4.0, 2.0, 1.0, 3.0, 2.0, 5.0, 3.0, 1.0, 1.0, 3.0, 6.0, 2.0, 3.0, 1.0, 2.0, 7.0};

  std::vector<double> vector = {1.0, 2.0, 3.0, 4.0};

  std::vector<double> result(vector.size(), 0.0);
  std::vector<double> true_result = {-0.472727, 0.345455, 0.181818, 0.672727};

  std::shared_ptr<ppc::core::TaskData> taskDataStl = createTaskData(matrix, vector, result);

  StlSLAYGradient testTaskStl(taskDataStl);

  ASSERT_EQ(testTaskStl.validation(), true);
  testTaskStl.pre_processing();
  testTaskStl.run();
  testTaskStl.post_processing();

  for (size_t i = 0; i < result.size(); i++) {
    ASSERT_LE(std::abs(true_result[i] - result[i]), TOLERANCE);
  }
}

TEST(dostavalov_s_sop_gradient_tbb, Test_Size_2) {
  std::vector<double> matrix = {4.0, 1.0, 1.0, 3.0};

  std::vector<double> vector = {1.0, 2.0};

  std::vector<double> result(vector.size(), 0.0);
  std::vector<double> true_result = {0.0909091, 0.636364};

  std::shared_ptr<ppc::core::TaskData> taskDataStl = createTaskData(matrix, vector, result);

  StlSLAYGradient testTaskStl(taskDataStl);

  ASSERT_EQ(testTaskStl.validation(), true);
  testTaskStl.pre_processing();
  testTaskStl.run();
  testTaskStl.post_processing();

  for (size_t i = 0; i < result.size(); i++) {
    ASSERT_LE(std::abs(true_result[i] - result[i]), TOLERANCE);
  }
}

TEST(dostavalov_s_sop_gradient_tbb, Test_Rand_200) {
  int size = 200;
  std::vector<double> matrix = randMatrix(size);
  std::vector<double> vector = randVector(size);

  std::vector<double> result(vector.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataStl = createTaskData(matrix, vector, result);

  StlSLAYGradient testTaskStl(taskDataStl);

  ASSERT_EQ(testTaskStl.validation(), true);
  testTaskStl.pre_processing();
  testTaskStl.run();
  testTaskStl.post_processing();

  ASSERT_TRUE(check_solution(matrix, vector, result));
}

TEST(dostavalov_s_sop_gradient_tbb, Test_Rand_100) {
  int size = 100;
  std::vector<double> matrix = randMatrix(size);
  std::vector<double> vector = randVector(size);

  std::vector<double> result(vector.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataStl = createTaskData(matrix, vector, result);

  StlSLAYGradient testTaskStl(taskDataStl);

  ASSERT_EQ(testTaskStl.validation(), true);
  testTaskStl.pre_processing();
  testTaskStl.run();
  testTaskStl.post_processing();

  ASSERT_TRUE(check_solution(matrix, vector, result));
}
TEST(dostavalov_s_sop_gradient_tbb, Test_Rand_1) {
  int size = 1;
  std::vector<double> matrix = randMatrix(size);
  std::vector<double> vector = randVector(size);

  std::vector<double> result(vector.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataStl = createTaskData(matrix, vector, result);

  StlSLAYGradient testTaskStl(taskDataStl);

  ASSERT_EQ(testTaskStl.validation(), true);
  testTaskStl.pre_processing();
  testTaskStl.run();
  testTaskStl.post_processing();

  ASSERT_TRUE(check_solution(matrix, vector, result));
}
TEST(dostavalov_s_sop_gradient_tbb, Test_Rand_0) {
  int size = 0;
  std::vector<double> matrix = randMatrix(size);
  std::vector<double> vector = randVector(size);

  std::vector<double> result(vector.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataStl = createTaskData(matrix, vector, result);

  StlSLAYGradient testTaskStl(taskDataStl);

  ASSERT_EQ(testTaskStl.validation(), true);
  testTaskStl.pre_processing();
  testTaskStl.run();
  testTaskStl.post_processing();

  ASSERT_TRUE(check_solution(matrix, vector, result));
}  // namespace dostavalov_s_stl

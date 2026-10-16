/**
 * Performance benchmark for the Genetic Programming library
 * Measures execution time for symbolic regression and classification
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
// Include ctimer for end-to-end timing
#include "common.h"
#include "ctimer.h"
#include "fitness.h"
#include "genetic.h"
#include "node.h"
#include "program.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Utility functions
namespace utils {

// Load image from PNG file
std::tuple<std::vector<std::vector<float>>, std::vector<float>,
           std::vector<float>, std::vector<float>>
load_png(const std::string &filename) {
  int width = 0, height = 0, channels = 0;
  // Request RGB for a consistent layout: grayscale is replicated, alpha
  // ignored.
  unsigned char *pixels =
      stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb);

  std::vector<std::vector<float>> X;
  std::vector<float> r, g, b;

  // Row-major pixel order; origin at top left, x rightward and y downward.
  // Pixel centers avoid division by zero for a one-pixel-wide/tall image.
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const size_t i = static_cast<size_t>(row) * width + col;
      X.push_back(
          {(1.0f * col + 0.5f) / width, ((1.0f * row) + 0.5f) / height});
      r.push_back(pixels[3 * i] / 255.0f);
      g.push_back(pixels[3 * i + 1] / 255.0f);
      b.push_back(pixels[3 * i + 2] / 255.0f);
    }
  }

  stbi_image_free(pixels);
  return {std::move(X), std::move(r), std::move(g), std::move(b)};
}

// Load dataset from CSV file
std::pair<std::vector<std::vector<float>>, std::vector<float>>
load_dataset(const std::string &filename) {
  std::vector<std::vector<float>> X;
  std::vector<float> y;

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string line;
  // Skip header
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string value;

    while (std::getline(ss, value, ',')) {
      row.push_back(std::stof(value));
    }

    // Last column is target
    y.push_back(row.back());
    row.pop_back();
    X.push_back(row);
  }

  return {X, y};
}

// Convert 2D vector to column-major vector
std::vector<float>
flatten_column_major(const std::vector<std::vector<float>> &data) {
  if (data.empty())
    return {};

  size_t rows = data.size();
  size_t cols = data[0].size();
  std::vector<float> flattened(rows * cols);

  for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
      flattened[j * rows + i] = data[i][j];
    }
  }

  return flattened;
}

// Split dataset into training and testing sets
template <typename T>
std::pair<T, T> train_test_split(const T &data, float test_size = 0.2) {
  T train, test;
  size_t test_count = static_cast<size_t>(data.size() * test_size);
  size_t train_count = data.size() - test_count;

  std::vector<size_t> indices(data.size());
  std::iota(indices.begin(), indices.end(), 0);
  size_t acc = 0;
  size_t train_pos = 0;
  size_t test_pos = train_count;
  for (size_t i = 0; i < data.size(); ++i) {
    acc += test_count;
    if (acc >= data.size()) {
      indices[test_pos++] = i;
      acc -= data.size();
    } else {
      indices[train_pos++] = i;
    }
  }

  for (size_t i = 0; i < train_count; ++i) {
    train.push_back(data[indices[i]]);
  }

  for (size_t i = train_count; i < data.size(); ++i) {
    test.push_back(data[indices[i]]);
  }

  return {train, test};
}

// Calculate mean squared error
float mean_squared_error(const std::vector<float> &y_true,
                         const std::vector<float> &y_pred) {
  if (y_true.size() != y_pred.size()) {
    throw std::runtime_error("Arrays must have the same length");
  }

  float sum = 0.0f;
  for (size_t i = 0; i < y_true.size(); ++i) {
    float diff = y_true[i] - y_pred[i];
    sum += diff * diff;
  }

  return sum / y_true.size();
}

// Calculate accuracy for classification
float accuracy(const std::vector<float> &y_true,
               const std::vector<float> &y_pred) {
  if (y_true.size() != y_pred.size()) {
    throw std::runtime_error("Arrays must have the same length");
  }

  size_t correct = 0;
  for (size_t i = 0; i < y_true.size(); ++i) {
    if ((y_pred[i] >= 0.5f && y_true[i] >= 0.5f) ||
        (y_pred[i] < 0.5f && y_true[i] < 0.5f)) {
      correct++;
    }
  }

  return static_cast<float>(correct) / y_true.size();
}

} // namespace utils

using namespace genetic;

void insertionSortPrograms(genetic::program *programs, int size) {
  // set any nan values to +inf
  for (int i = 0; i < size; ++i) {
    if (std::isnan(programs[i].raw_fitness_)) {
      programs[i].raw_fitness_ = std::numeric_limits<float>::max();
    }
  }

  // stable sort programs instead of insertion sort
  std::vector<int> order(size);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return programs[a].raw_fitness_ < programs[b].raw_fitness_;
  });

  std::vector<genetic::program> sorted;
  sorted.reserve(size);
  for (int i : order) sorted.push_back(programs[i]);
  for (int i = 0; i < size; ++i) programs[i] = sorted[i];
}

void run_symbolic_regression(const std::string &dataset_file) {
  std::cout << "\n;;===== Symbolic Regression Benchmark =====\n" << std::endl;
  std::cout << "(provide best-program-1 best-program-2)\n";
  // Initialize ctimer for end-to-end timing
  ctimer_t end_to_end_timer;
  ctimer_start(&end_to_end_timer);

  // Load dataset
  std::cout << ";;Loading dataset..." << std::endl;

  auto dataset = utils::load_dataset(dataset_file);
  auto X = dataset.first;
  auto y = dataset.second;

  std::cout << ";;Dataset dimensions: " << X.size() << " samples x "
            << X[0].size() << " features" << std::endl;

  // Split dataset
  auto X_split = utils::train_test_split(X);
  auto y_split = utils::train_test_split(y);

  std::vector<std::vector<float>> X_train = X_split.first;
  std::vector<std::vector<float>> X_test = X_split.second;
  std::vector<float> y_train = y_split.first;
  std::vector<float> y_test = y_split.second;

  // Flatten data for genetic library (column-major)
  std::vector<float> X_train_flat = utils::flatten_column_major(X_train);
  std::vector<float> X_test_flat = utils::flatten_column_major(X_test);

  // Create weights (all 1.0)
  std::vector<float> sample_weights(y_train.size(), 1.0f);

  // Set parameters
  genetic::param params;
  params.population_size = 512;
  params.generations = 16;
  params.tournament_size = 4;
  params.init_depth[0] = 2;
  params.init_depth[1] = 6;
  params.init_method = genetic::init_method_t::half_and_half;
  params.num_features = X_train[0].size(); // Number of features
  params.terminalRatio = 0.05;
  // Function set
  {
    using namespace genetic;
    params.function_set = {node::type::add,  node::type::sub, node::type::mul,
                           node::type::abs,  node::type::sin, node::type::cos,
                           node::type::fdim, node::type::log};
    // Arity set
    params.arity_set = {
        {1,
         {node::type::abs, node::type::sin, node::type::cos, node::type::log}},
        {2,
         {node::type::add, node::type::sub, node::type::mul,
          node::type::fdim}}};
  }

  params.metric = genetic::metric_t::mse; // Use MSE as the fitness metric
  params.parsimony_coefficient = 0.00f;   // Penalize complexity
  params.p_crossover = 0.80f;             // High crossover probability
  params.p_subtree_mutation = 0.05f;
  params.p_hoist_mutation = 0.01f;
  params.p_point_mutation = 0.01f;
  params.max_samples = 1.0f;        // Use all samples
  params.random_state = 2025000000; // For reproducibility

  // Running the symbolic regression
  std::cout << ";;Training symbolic regressor with " << params.population_size
            << " population size and " << params.generations << " generations"
            << std::endl;

  // Create history vector to store programs
  genetic::program_t final_programs;
  final_programs = new genetic::program[params.population_size]();
  std::vector<std::vector<genetic::program>> history;

  // Train the model
  genetic::symFit(X_train_flat.data(), y_train.data(), sample_weights.data(),
                  X_train.size(),    // Number of rows
                  X_train[0].size(), // Number of columns
                  params, final_programs, history);

  // Debug printing
  // for (int i = 0; i < params.population_size; ++i) {
  //   std::cout << "Gen 0: " << i << "=" << genetic::stringify(history[0][i])
  //             << std::endl;
  // }

  // Predict on top 2 candidates
  insertionSortPrograms(final_programs, params.population_size);

  std::vector<float> y_pred1(X_test.size());
  genetic::symRegPredict(X_test_flat.data(), X_test.size(), &final_programs[0],
                         y_pred1.data());

  std::vector<float> y_pred2(X_test.size());
  genetic::symRegPredict(X_test_flat.data(), X_test.size(), &final_programs[1],
                         y_pred2.data());

  // Calculate MSE on test set
  float mse = utils::mean_squared_error(y_test, y_pred1);
  float mse2 = utils::mean_squared_error(y_test, y_pred2);

  // Extract the best programs and print some stats
  if (history.back().size() > 0) {
    genetic::program_t best_program1 = &final_programs[0];
    std::cout << ";;Best program 1 details:" << std::endl;
    std::cout << ";;- Length: " << best_program1->len << " nodes" << std::endl;
    std::cout << ";;- Depth: " << best_program1->depth << std::endl;
    std::cout << ";;- Raw fitness: " << best_program1->raw_fitness_
              << std::endl;
    std::cout << ";;- Test MSE: " << mse << std::endl;

    // Convert to string representation
    std::string program_str = genetic::stringify(*best_program1);
    std::cout << "(define best-program-1 " << program_str << ")" << std::endl;

    genetic::program_t best_program2 = &final_programs[1];
    std::cout << ";;Best program 2 details:" << std::endl;
    std::cout << ";;- Length: " << best_program2->len << " nodes" << std::endl;
    std::cout << ";;- Depth: " << best_program2->depth << std::endl;
    std::cout << ";;- Raw fitness: " << best_program2->raw_fitness_
              << std::endl;
    std::cout << ";;- Test MSE: " << mse2 << std::endl;

    // Convert to string representation
    std::string program_str2 = genetic::stringify(*best_program2);
    std::cout << "(define best-program-2 " << program_str2 << ")" << std::endl;
  }

  // Stop end-to-end timer and print results
  ctimer_stop(&end_to_end_timer);
  ctimer_measure(&end_to_end_timer);
  ctimer_print(end_to_end_timer, "Symbolic Regression (End-to-End)");

  delete[] final_programs;
}

void run_symbolic_classification(const std::string &dataset_file) {
  std::cout << "\n;;===== Symbolic Classification Benchmark =====\n"
            << std::endl;
  std::cout << "(provide best-program-1 best-program-2)\n";

  // Initialize ctimer for end-to-end timing
  ctimer_t end_to_end_timer;
  ctimer_start(&end_to_end_timer);

  // Load dataset
  std::cout << ";;Loading dataset..." << std::endl;

  auto dataset = utils::load_dataset(dataset_file);
  auto X = dataset.first;
  auto y = dataset.second;

  std::cout << ";;Dataset dimensions: " << X.size() << " samples x "
            << X[0].size() << " features" << std::endl;

  // Split dataset
  auto X_split = utils::train_test_split(X);
  auto y_split = utils::train_test_split(y);

  std::vector<std::vector<float>> X_train = X_split.first;
  std::vector<std::vector<float>> X_test = X_split.second;
  std::vector<float> y_train = y_split.first;
  std::vector<float> y_test = y_split.second;

  // Flatten data for genetic library (column-major)
  std::vector<float> X_train_flat = utils::flatten_column_major(X_train);
  std::vector<float> X_test_flat = utils::flatten_column_major(X_test);

  // Create weights (all 1.0)
  std::vector<float> sample_weights(y_train.size(), 1.0f);

  // Set parameters
  genetic::param params;
  params.population_size = 16384;
  params.generations = 16;
  params.tournament_size = 16;
  params.init_depth[0] = 2;
  params.init_depth[1] = 6;
  params.init_method = genetic::init_method_t::half_and_half;
  params.num_features = X_train[0].size(); // Number of features
  params.terminalRatio = 0.05;

  // Function set for classification
  params.function_set = {node::type::add,  node::type::sub, node::type::mul,
                         node::type::sin,  node::type::cos, node::type::sq,
                         node::type::sqrt, node::type::abs, node::type::fdim};

  // Don't worry if you see stuff like sqrt(-5) -> we consider only the absolute
  // value in that case
  params.arity_set = {
      {1,
       {node::type::abs, node::type::sin, node::type::cos, node::type::sq,
        node::type::sqrt}},
      {2,
       {node::type::add, node::type::sub, node::type::mul, node::type::fdim}}};

  params.metric = genetic::metric_t::logloss; // Use log loss forclassification
  params.transformer =
      genetic::transformer_t::sigmoid; // Use sigmoid for binary classification
  params.parsimony_coefficient = 0.01f;
  params.p_crossover = 0.80f; // High crossover probability
  params.p_subtree_mutation = 0.05f;
  params.p_hoist_mutation = 0.01f;
  params.p_point_mutation = 0.01f;
  params.max_samples = 1.0f;  // Use all samples
  params.random_state = 2025; // For reproducibility

  // Running the symbolic classification
  std::cout << ";;Training symbolic classifier with " << params.population_size
            << " population size and " << params.generations << " generations "
            << std::endl;

  // Create history vector to store programs
  genetic::program_t final_programs;
  final_programs = new genetic::program[params.population_size]();

  std::vector<std::vector<genetic::program>> history;

  // Train the model
  genetic::symFit(X_train_flat.data(), y_train.data(), sample_weights.data(),
                  X_train.size(),    // Number of rows
                  X_train[0].size(), // Number of columns
                  params, final_programs, history);

  // // print and check programs from hsitory
  // for (int i = 0; i < params.population_size; ++i) {
  //   std::cout << "Gen " << 16 << " : "
  //             << genetic::stringify(history[history.size() - 1][i])
  //             << std::endl;
  // }

  // Predict classes for best 2 programs acc to training
  insertionSortPrograms(final_programs, params.population_size);
  std::vector<float> y_pred1(X_test.size());
  genetic::symClfPredict(X_test_flat.data(), X_test.size(), params,
                         &final_programs[0], y_pred1.data());

  std::vector<float> y_pred2(X_test.size());
  genetic::symClfPredict(X_test_flat.data(), X_test.size(), params,
                         &final_programs[1], y_pred2.data());

  float acc = utils::accuracy(y_test, y_pred1);
  float acc2 = utils::accuracy(y_test, y_pred2);

  // Extract the best programs and print some stats
  if (history.back().size() > 0) {
    genetic::program_t best_program1 = &final_programs[0];
    std::cout << ";;Best program 1 details:" << std::endl;
    std::cout << ";;- Length: " << best_program1->len << " nodes" << std::endl;
    std::cout << ";;- Depth: " << best_program1->depth << std::endl;
    std::cout << ";;- Raw fitness: " << best_program1->raw_fitness_
              << std::endl;
    std::cout << ";;- Test accuracy: " << acc << std::endl;

    // Convert to string representation
    std::string program_str = genetic::stringify(*best_program1);
    std::cout << "(define best-program-1 " << program_str << ")" << std::endl;

    genetic::program_t best_program2 = &final_programs[1];
    std::cout << ";;Best program 2 details:" << std::endl;
    std::cout << ";;- Length: " << best_program2->len << " nodes" << std::endl;
    std::cout << ";;- Depth: " << best_program2->depth << std::endl;
    std::cout << ";;- Raw fitness: " << best_program2->raw_fitness_
              << std::endl;
    std::cout << ";;- Test accuracy: " << acc2 << std::endl;

    // Convert to string representation
    std::string program_str2 = genetic::stringify(*best_program2);
    std::cout << "(define best-program-2 " << program_str2 << ")" << std::endl;
  }

  // Stop end-to-end timer and print results
  ctimer_stop(&end_to_end_timer);
  ctimer_measure(&end_to_end_timer);
  ctimer_print(end_to_end_timer, "Symbolic Classification (End-to-End)");

  delete[] final_programs;
}

void run_symbolic_reconstruction(const std::string &png_file, int K) {
  if (K < 1) {
    throw std::invalid_argument("K must be positive");
  }
  std::cout << "\n;;===== Symbolic Image Reconstruction =====\n" << std::endl;
  std::cout << R"racket(
(require racket/draw)
(provide red green blue render)

(define-syntax-rule (define-channel (name X) expr ...)
  (define (name X)
    (let ([values (list expr ...)])
      (/ (apply + values) (length values)))))

)racket";
  ctimer_t end_to_end_timer;
  ctimer_start(&end_to_end_timer);

  std::cout << ";;Loading " << png_file << std::endl;
  std::vector<std::vector<float>> X = std::get<0>(utils::load_png(png_file));
  std::vector<float> r = std::get<1>(utils::load_png(png_file));
  std::vector<float> g = std::get<2>(utils::load_png(png_file));
  std::vector<float> b = std::get<3>(utils::load_png(png_file));
  const bool monochrome = (r == g && g == b);
  // Use the same [-pi, pi] coordinate domain for every input image.
  // Exported channel functions still accept normalized (x,y) coordinates.
  const float pi = 3.14159265358979323846f;
  const float input_scale = 2.0f * pi;
  const float input_offset = -pi;
  for (auto &point : X) {
    point[0] = point[0] * input_scale + input_offset;
    point[1] = point[1] * input_scale + input_offset;
  }
  std::cout << std::setprecision(9)
            << "(define (model-input X) (list (+ (* " << input_scale
            << " (list-ref X 0)) " << input_offset << ") (+ (* "
            << input_scale << " (list-ref X 1)) " << input_offset << ")))\n";
  std::cout << ";; " << (monochrome ? "Monochrome: one shared intensity model per run"
                                   : "RGB: three independent channel models per run") << "\n";

  int width = 0, height = 0, channels = 0;
  stbi_info(png_file.c_str(), &width, &height, &channels);
  std::cout << ";;Dataset dimensions: " << X.size()
            << " samples x 2 features\n";
  std::cout << "(define width " << width << ")\n"
            << "(define height " << height << ")\n";

  // All four arrays use the same deterministic split to preserve alignment.
  auto X_split = utils::train_test_split(X);
  auto r_split = utils::train_test_split(r);
  auto g_split = utils::train_test_split(g);
  auto b_split = utils::train_test_split(b);
  std::vector<std::vector<float>> X_train = X_split.first;
  std::vector<std::vector<float>> X_test = X_split.second;
  std::vector<float> X_train_flat = utils::flatten_column_major(X_train);
  std::vector<float> X_test_flat = utils::flatten_column_major(X_test);
  std::vector<float> sample_weights(X_train.size(), 1.0f);

  genetic::param params;
  params.population_size = 128;
  params.generations = 16;
  params.tournament_size = 4;
  params.init_depth[0] = 1;
  params.init_depth[1] = 4;
  params.init_method = genetic::init_method_t::half_and_half;
  params.num_features = 2;
  params.terminalRatio = 0.3f;
  params.const_range[0] = -1.0f;
  params.const_range[1] = 1.0f;
  params.function_set = {node::type::add, node::type::sub, node::type::mul,
                         node::type::sin, node::type::abs};
  params.arity_set = {{1, {node::type::sin, node::type::abs}},
                     {2, {node::type::add, node::type::sub, node::type::mul}}};
  params.metric = genetic::metric_t::mse;
  params.parsimony_coefficient = 0.0001f;
  params.p_crossover = 0.60f;
  params.p_subtree_mutation = 0.20f;
  params.p_hoist_mutation = 0.05f;
  params.p_point_mutation = 0.10f;
  params.max_samples = 1.0f;
  // MSE cannot be negative: always run the fixed generation budget.
  params.stopping_criteria = -1.0f;
  const uint32_t base_seed = 2025000000;

  const char *channel_names[] = {"red", "green", "blue"};
  const std::pair<std::vector<float>, std::vector<float>> *channel_splits[] = {
      &r_split, &g_split, &b_split};
  double rgb_mse = 0.0;
  std::cout << ";;Training " << K << " models per channel, "
            << params.population_size << " candidates, " << params.generations
            << " generations per model\n";

  const int channel_count = monochrome ? 1 : 3;
  for (int channel = 0; channel < channel_count; ++channel) {
    const auto &y_train = channel_splits[channel]->first;
    const auto &y_test = channel_splits[channel]->second;
    std::vector<float> average_predictions(X_test.size(), 0.0f);
    std::cout << "(define-channel (" << channel_names[channel] << " X)\n";

    for (int model = 0; model < K; ++model) {
      params.random_state =
          base_seed + static_cast<uint32_t>(channel) * K + model;
      genetic::program_t final_programs =
          new genetic::program[params.population_size]();
      std::vector<std::vector<genetic::program>> history;
      genetic::symFit(X_train_flat.data(), y_train.data(),
                      sample_weights.data(), X_train.size(),
                      params.num_features, params, final_programs, history);
      insertionSortPrograms(final_programs, params.population_size);
      genetic::program_t best_program = &final_programs[0];

      std::vector<float> predictions(X_test.size());
      genetic::symRegPredict(X_test_flat.data(), X_test.size(), best_program,
                             predictions.data());
      for (size_t i = 0; i < predictions.size(); ++i) {
        average_predictions[i] += predictions[i] / K;
      }
      std::cout << ";; " << channel_names[channel] << " model " << model
                << ", seed " << params.random_state << ", training MSE "
                << best_program->raw_fitness_ << ", test MSE "
                << utils::mean_squared_error(y_test, predictions) << ", nodes "
                << best_program->len << "\n";
      // stringify returns a lambda; apply it to X inside define-channel.
      std::cout << "  (" << genetic::stringify(*best_program) << " (model-input X))\n";
      delete[] final_programs;
    }
    std::cout << ")\n";
    float mse = utils::mean_squared_error(y_test, average_predictions);
    rgb_mse += mse / channel_count;
    std::cout << ";; " << channel_names[channel]
              << " averaged test MSE: " << mse << "\n";
  }
  if (monochrome) {
    std::cout << "(define green red)\n(define blue red)\n";
  }
  std::cout << ";; RGB averaged test MSE (before display clamping): " << rgb_mse
            << "\n";

  // Channel values remain floats; clamp only when converting to PNG bytes.
  // module+ main renders on execution but not when another module imports it.
  std::cout << R"racket(
(define (to-byte value)
  (exact-round (* 255.0 (max 0.0 (min 1.0 value)))))

(define (render [filename "reconstructed.png"])
  (define pixels (make-bytes (* width height 4)))
  (for* ([row (in-range height)] [col (in-range width)])
    (define X (list (/ (+ col 0.5) width) (/ (+ row 0.5) height)))
    (define offset (* 4 (+ (* row width) col)))
    (bytes-set! pixels offset 255)
    (bytes-set! pixels (+ offset 1) (to-byte (red X)))
    (bytes-set! pixels (+ offset 2) (to-byte (green X)))
    (bytes-set! pixels (+ offset 3) (to-byte (blue X))))
  (define bitmap (make-bitmap width height))
  (send bitmap set-argb-pixels 0 0 width height pixels)
  (unless (send bitmap save-file filename 'png)
    (error 'render "Could not save ~a" filename)))

(module+ main
  (command-line
    #:args ([output "reconstructed.png"])
    (render output)))
)racket";

  ctimer_stop(&end_to_end_timer);
  ctimer_measure(&end_to_end_timer);
  ctimer_print(end_to_end_timer, "Symbolic Image Reconstruction (End-to-End)");
}

int main(int argc, char *argv[]) {
  try {
    if (argc < 2) {
      throw std::invalid_argument(
          "Usage: genetic_benchmark "
          "{diabetes|cancer|housing|silk-wave|square-wave}");
    }
    std::cout << "#lang racket\n";
    // Default datasets
    std::string regression_dataset = "benchmark/diabetes.csv";
    std::string classification_dataset = "benchmark/cancer.csv";
    std::string housing_dataset = "benchmark/housing.csv";
    std::string silk_wave = "benchmark/silk-waves.png";
    std::string square_wave = "benchmark/squared-waves.png";

    std::string arg_dset(argv[1]);

    if (arg_dset == "diabetes") {
      run_symbolic_regression(regression_dataset);
    } else if (arg_dset == "cancer") {
      run_symbolic_classification(classification_dataset);
    } else if (arg_dset == "housing") {
      run_symbolic_regression(housing_dataset);
    } else if (arg_dset == "silk-wave") {
      if (argc != 2) {
        throw std::invalid_argument("Usage: genetic_benchmark silk-wave");
      }
      run_symbolic_reconstruction(silk_wave, 3);
    } else if (arg_dset == "square-wave") {
      if (argc != 2) {
        throw std::invalid_argument("Usage: genetic_benchmark square-wave");
      }
      run_symbolic_reconstruction(square_wave, 8);

    } else {
      throw std::invalid_argument("Unknown benchmark: " + arg_dset);
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

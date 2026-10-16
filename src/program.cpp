#include "constants.h"
#include "custom_distributions.h"
#include "node_detail.h"
#include "philox_engine.h"
#include "reg_stack.h"
#include <algorithm>
#include <cstdint>
#include <fitness.h>
#include <node.h>
#include <numeric>
#include <program.h>
#include <random>
#include <stack>
#include <omp.h>

namespace genetic {

constexpr int BLOCK_ROWS = 256;

/**
 * Per-thread arity buffer. grows to the longest program this thread has
 * seen, never shrinks, never overflows. The pointer is valid until the
 * next call on the same thread.
 */
inline const unsigned char *compute_arities(const node *nodes, int len) {
  thread_local std::vector<unsigned char> buf;
  if (buf.size() < static_cast<size_t>(len)) buf.resize(len);
  for (int i = 0; i < len; ++i) {
    const auto type = nodes[i].t;
    buf[i] = detail::is_nonterminal(type)
                 ? static_cast<unsigned char>(detail::arity(type))
                 : 0;
  }
  return buf.data();
}

/**
 * Evaluate one program on rows [r0, r0 + nb), node-major: each node is
 * applied to the whole block before moving to the next node. Writes nb
 * predictions to out. Serial; callers decide how to parallelize.
 */
template <int MaxSize = MAX_STACK_SIZE, int B = BLOCK_ROWS>
inline void eval_block(const program_t prog, const float *data,
                       const uint64_t n_rows, const uint64_t r0, const int nb,
                       float *out) {
  using T = node::type;
  using detail::MIN_VAL;
  const node *nodes = prog->nodes;
  const int len = prog->len;

  alignas(64) float stack[MaxSize][B];
  int top = 0;

// a = in[0] (top of stack), b = in[1] (second), same as evaluate_node
#define UNARY(expr)                                                  \
  {                                                                  \
    float *x = stack[top - 1];                                       \
    for (int j = 0; j < nb; ++j) { const float a = x[j]; x[j] = (expr); } \
  } break;
#define BINARY(expr)                                                 \
  {                                                                  \
    const float *x0 = stack[top - 1];                                \
    float *x1 = stack[top - 2];                                      \
    for (int j = 0; j < nb; ++j) {                                   \
      const float a = x0[j], b = x1[j]; x1[j] = (expr);              \
    }                                                                \
    --top;                                                           \
  } break;

  for (int i = len - 1; i >= 0; --i) {
    const node &nd = nodes[i];
    switch (nd.t) {
    case T::constant: {
      float *d = stack[top++];
      const float v = nd.u.val;
      for (int j = 0; j < nb; ++j) d[j] = v;
    } break;
    case T::variable: {
      const float *col = data + n_rows * nd.u.fid + r0;  // contiguous column
      std::copy(col, col + nb, stack[top++]);
    } break;
    // binary
    case T::add:   BINARY(a + b)
    case T::atan2: BINARY(atan2f(a, b))
    case T::div:   BINARY(fabsf(b) < MIN_VAL ? 1.0f : a / b)
    case T::fdim:  BINARY(fdimf(a, b))
    case T::gt:    BINARY(a > b ? 1.0f : 0.0f)
    case T::max:   BINARY(fmaxf(a, b))
    case T::min:   BINARY(fminf(a, b))
    case T::mul:   BINARY(a * b)
    case T::pow:   BINARY(powf(a, b))
    case T::sub:   BINARY(a - b)
    // unary
    case T::abs:   UNARY(fabsf(a))
    case T::acos:  UNARY(acosf(a))
    case T::acosh: UNARY(acoshf(a))
    case T::asin:  UNARY(asinf(a))
    case T::asinh: UNARY(asinhf(a))
    case T::atan:  UNARY(atanf(a))
    case T::atanh: UNARY(atanhf(a))
    case T::cbrt:  UNARY(cbrtf(a))
    case T::cos:   UNARY(cosf(a))
    case T::cosh:  UNARY(coshf(a))
    case T::cube:  UNARY(a * a * a)
    case T::exp:   UNARY(expf(a))
    case T::inv:   UNARY(fabsf(a) < MIN_VAL ? 0.f : 1.f / a)
    case T::log:   UNARY(fabsf(a) < MIN_VAL ? 0.f : logf(fabsf(a)))
    case T::neg:   UNARY(-a)
    case T::rcbrt: UNARY(1.0f / cbrtf(a))
    case T::rsqrt: UNARY(1.0f / sqrtf(fabsf(a)))
    case T::sin:   UNARY(sinf(a))
    case T::sinh:  UNARY(sinhf(a))
    case T::sq:    UNARY(a * a)
    case T::sqrt:  UNARY(sqrtf(fabsf(a)))
    case T::tan:   UNARY(tanf(a))
    case T::tanh:  UNARY(tanhf(a))
    default: break;
    }
  }
#undef UNARY
#undef BINARY

  std::copy(stack[0], stack[0] + nb, out);
}

/**
 * Execution kernel for a single program. We assume that the input data
 * is stored in column major format.
 */
template <int MaxSize = MAX_STACK_SIZE>
void execute_kernel(const program_t d_progs, const float *data, float *y_pred,
                    const uint64_t n_rows, const uint64_t n_progs) {
  const uint64_t n_blocks = (n_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    for (uint64_t blk = 0; blk < n_blocks; ++blk) {
      const uint64_t r0 = blk * BLOCK_ROWS;
      const int nb = static_cast<int>(
          std::min<uint64_t>(BLOCK_ROWS, n_rows - r0));
      eval_block<MaxSize>(d_progs + pid, data, n_rows, r0, nb,
                          y_pred + pid * n_rows + r0);
    }
  }
}

program::program()
    : len(0), depth(0), raw_fitness_(0.0f), metric(metric_t::mse),
      mut_type(mutation_t::none), nodes(nullptr) {}

program::~program() { delete[] nodes; }

program::program(const program &src)
    : len(src.len), depth(src.depth), raw_fitness_(src.raw_fitness_),
      metric(src.metric), mut_type(src.mut_type) {
  nodes = new node[len];
  std::copy(src.nodes, src.nodes + src.len, nodes);
}

program &program::operator=(const program &src) {
  len = src.len;
  depth = src.depth;
  raw_fitness_ = src.raw_fitness_;
  metric = src.metric;
  mut_type = src.mut_type;

  // Copy nodes
  delete[] nodes;
  nodes = new node[len];
  std::copy(src.nodes, src.nodes + src.len, nodes);

  return *this;
}

void compute_metric(int n_rows, int n_progs, const float *y,
                    const float *y_pred, const float *w, float *score,
                    const param &params) {
  // Call appropriate metric function based on metric defined in params
  if (params.metric == metric_t::pearson) {
    weightedPearson(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::spearman) {
    weightedSpearman(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::mae) {
    meanAbsoluteError(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::mse) {
    meanSquareError(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::rmse) {
    rootMeanSquareError(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::logloss) {
    logLoss(n_rows, n_progs, y, y_pred, w, score);
  } else {
    // This should not be reachable
  }
}

void execute(const program_t &d_progs, const int n_rows, const int n_progs,
             const float *data, float *y_pred) {
  execute_kernel(d_progs, data, y_pred, static_cast<uint64_t>(n_rows),
                 static_cast<uint64_t>(n_progs));
}

void find_fitness(program_t d_prog, float *score, const param &params,
                  const int n_rows, const float *data, const float *y,
                  const float *sample_weights) {

  // Compute predicted values
  std::vector<float> y_pred(n_rows);
  execute(d_prog, n_rows, 1, data, y_pred.data());

  // Compute error
  compute_metric(n_rows, 1, y, y_pred.data(), sample_weights, score, params);
}

void find_batched_fitness(int n_progs, program_t d_progs, float *score,
                          const param &params, const int n_rows,
                          const float *data, const float *y,
                          const float *sample_weights) {

  int n_threads = omp_get_max_threads();

  // parallelize over rows on two-pass layout
  if (n_progs < n_threads) {
    std::vector<float> y_pred(static_cast<uint64_t>(n_rows) * n_progs);
    execute(d_progs, n_rows, n_progs, data, y_pred.data());
    compute_metric(n_rows, n_progs, y, y_pred.data(), sample_weights, score,
                   params);
    return;
  }

  // fused path: one program per thread at a time, execute then score
  // Memory is n_rows floats per thread instead of n_rows * n_progs
  const uint64_t nr = static_cast<uint64_t>(n_rows);
  const uint64_t n_blocks = (nr + BLOCK_ROWS - 1) / BLOCK_ROWS;

  #pragma omp parallel
  {
    std::vector<float> pred(nr);

    #pragma omp for schedule(dynamic)
    for (int p = 0; p < n_progs; ++p) {
      for (uint64_t blk = 0; blk < n_blocks; ++blk) {
        const uint64_t r0 = blk * BLOCK_ROWS;
        const int nb = static_cast<int>(
            std::min<uint64_t>(BLOCK_ROWS, nr - r0));
        eval_block(d_progs + p, data, nr, r0, nb, pred.data() + r0);
      }
      compute_metric(n_rows, 1, y, pred.data(), sample_weights, score + p,
                     params);
    }
  }
}

void set_fitness(program &h_prog, const param &params, const int n_rows,
                 const float *data, const float *y,
                 const float *sample_weights) {

  std::vector<float> score(1);

  find_fitness(&h_prog, score.data(), params, n_rows, data, y, sample_weights);

  // Update host and device score for program
  h_prog.raw_fitness_ = score[0];
}

void set_batched_fitness(int n_progs, std::vector<program> &h_progs,
                         const param &params, const int n_rows,
                         const float *data, const float *y,
                         const float *sample_weights) {

  std::vector<float> score(n_progs);

  find_batched_fitness(n_progs, h_progs.data(), score.data(), params, n_rows,
                       data, y, sample_weights);

  // Update scores on host and device
  // TODO: Find a way to reduce the number of implicit memory transfers
  for (int i = 0; i < n_progs; ++i) {
    h_progs[i].raw_fitness_ = score[i];
  }
}

float get_fitness(const program &prog, const param &params) {
  int crit = params.criterion();
  float penalty = params.parsimony_coefficient * prog.len * (2 * crit - 1);
  return (prog.raw_fitness_ - penalty);
}

/**
 * @brief Get a random subtree of the current program nodes (on CPU)
 *
 * @param pnodes  AST represented as a list of nodes
 * @param len     The total number of nodes in the AST
 * @param rng     Random number generator for subtree selection
 * @return A tuple [first,last) which contains the required subtree
 */
std::pair<int, int> get_subtree(node *pnodes, int len, PhiloxEngine &rng) {
  int start, end;
  start = end = 0;

  // Specify RNG
  uniform_real_distribution_custom<float> dist_uniform(0.0f, 1.0f);
  float bound = dist_uniform(rng);

  // Specify subtree start probs acc to Koza's selection approach
  std::vector<float> node_probs(len, 0.1);
  float sum = 0.1 * len;

  for (int i = 0; i < len; ++i) {
    if (pnodes[i].is_nonterminal()) {
      node_probs[i] = 0.9;
      sum += 0.8;
    }
  }

  // Normalize vector
  for (int i = 0; i < len; ++i) {
    node_probs[i] /= sum;
  }

  // Compute cumulative sum
  std::partial_sum(node_probs.begin(), node_probs.end(), node_probs.begin());

  start = std::lower_bound(node_probs.begin(), node_probs.end(), bound) -
          node_probs.begin();
  end = start;

  // Iterate until all function arguments are satisfied in current subtree
  int num_args = 1;
  while (num_args > end - start) {
    node curr;
    curr = pnodes[end];
    if (curr.is_nonterminal())
      num_args += curr.arity();
    ++end;
  }

  return std::make_pair(start, end);
}

int get_depth(const program &p_out) {
  int depth = 0;
  std::stack<int> arity_stack;
  for (auto i = 0; i < p_out.len; ++i) {
    node curr(p_out.nodes[i]);

    // Update depth
    int sz = arity_stack.size();
    depth = std::max(depth, sz);

    // Update stack
    if (curr.is_nonterminal()) {
      arity_stack.push(curr.arity());
    } else {
      // Only triggered for a depth 0 node
      if (arity_stack.empty())
        break;

      int e = arity_stack.top();
      arity_stack.pop();
      arity_stack.push(e - 1);

      while (arity_stack.top() == 0) {
        arity_stack.pop();
        if (arity_stack.empty())
          break;

        e = arity_stack.top();
        arity_stack.pop();
        arity_stack.push(e - 1);
      }
    }
  }

  return depth;
}

void build_program(program &p_out, const param &params, PhiloxEngine &rng) {
  // Define data structures needed for tree
  std::stack<int> arity_stack;
  std::vector<node> nodelist;
  nodelist.reserve(1 << (MAX_STACK_SIZE));

  // Specify Distributions with parameters
  uniform_int_distribution_custom<int> dist_function(
      0, params.function_set.size() - 1);
  uniform_int_distribution_custom<int> dist_initDepth(params.init_depth[0],
                                                      params.init_depth[1]);
  uniform_int_distribution_custom<int> dist_terminalChoice(0,
                                                           params.num_features);
  uniform_real_distribution_custom<float> dist_constVal(params.const_range[0],
                                                        params.const_range[1]);
  bernoulli_distribution_custom dist_nodeChoice(params.terminalRatio);
  bernoulli_distribution_custom dist_coinToss(0.5);

  // Initialize nodes
  int max_depth = dist_initDepth(rng);
  node::type func = params.function_set[dist_function(rng)];
  node curr_node(func);
  nodelist.push_back(curr_node);
  arity_stack.push(curr_node.arity());

  init_method_t method = params.init_method;
  if (method == init_method_t::half_and_half) {
    // Choose either grow or full for this tree
    bool choice = dist_coinToss(rng);
    method = choice ? init_method_t::grow : init_method_t::full;
  }

  // Fill tree
  while (!arity_stack.empty()) {
    int depth = arity_stack.size();
    p_out.depth = std::max(depth, p_out.depth);
    bool node_choice = dist_nodeChoice(rng);

    if ((node_choice == false || method == init_method_t::full) &&
        depth < max_depth) {
      // Add a function to node list
      curr_node = node(params.function_set[dist_function(rng)]);
      nodelist.push_back(curr_node);
      arity_stack.push(curr_node.arity());
    } else {
      // Add terminal
      int terminal_choice = dist_terminalChoice(rng);
      if (terminal_choice == params.num_features) {
        // Add constant
        float val = dist_constVal(rng);
        curr_node = node(val);
      } else {
        // Add variable
        int fid = terminal_choice;
        curr_node = node(fid);
      }

      // Modify nodelist
      nodelist.push_back(curr_node);

      // Modify stack
      int e = arity_stack.top();
      arity_stack.pop();
      arity_stack.push(e - 1);
      while (arity_stack.top() == 0) {
        arity_stack.pop();
        if (arity_stack.empty()) {
          break;
        }

        e = arity_stack.top();
        arity_stack.pop();
        arity_stack.push(e - 1);
      }
    }
  }

  // Set new program parameters - need to do a copy as
  // nodelist will be deleted using RAII semantics
  p_out.nodes = new node[nodelist.size()];
  std::copy(nodelist.begin(), nodelist.end(), p_out.nodes);

  p_out.len = nodelist.size();
  p_out.metric = params.metric;
  p_out.raw_fitness_ = 0.0f;
}

void point_mutation(const program &prog, program &p_out, const param &params,
                    PhiloxEngine &rng) {
  // deep-copy program
  p_out = prog;

  // Specify RNGs
  uniform_real_distribution_custom<float> dist_uniform(0.0f, 1.0f);
  uniform_int_distribution_custom<int> dist_terminalChoice(0,
                                                           params.num_features);
  uniform_real_distribution_custom<float> dist_constantVal(
      params.const_range[0], params.const_range[1]);

  // Fill with uniform numbers
  std::vector<float> node_probs(p_out.len);
  std::generate(node_probs.begin(), node_probs.end(),
                [&dist_uniform, &rng] { return dist_uniform(rng); });

  // Mutate nodes
  int len = p_out.len;
  for (int i = 0; i < len; ++i) {
    node curr(prog.nodes[i]);

    if (node_probs[i] < params.p_point_replace) {
      if (curr.is_terminal()) {
        int choice = dist_terminalChoice(rng);

        if (choice == params.num_features) {
          // Add a randomly generated constant
          curr = node(dist_constantVal(rng));
        } else {
          // Add a variable with fid=choice
          curr = node(choice);
        }
      } else if (curr.is_nonterminal()) {
        // Replace current function with another function of the same arity
        int ar = curr.arity();
        // CUML_LOG_DEBUG("Arity is %d, curr function is
        // %d",ar,static_cast<std::underlying_type<node::type>::type>(curr.t));
        std::vector<node::type> fset = params.arity_set.at(ar);
        uniform_int_distribution_custom<int> dist_fset(0, fset.size() - 1);
        int choice = dist_fset(rng);
        curr = node(fset[choice]);
      }

      // Update p_out with updated value
      p_out.nodes[i] = curr;
    }
  }
}

void crossover(const program &prog, const program &donor, program &p_out,
               const param &params, PhiloxEngine &rng) {
  // Get a random subtree of prog to replace
  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, rng);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  // Set metric of output program
  p_out.metric = prog.metric;

  // MAX_STACK_SIZE can only handle tree of depth MAX_STACK_SIZE -
  // max(func_arity=2) + 1 Thus we continuously hoist the donor subtree. Actual
  // indices in donor
  int donor_start = 0;
  int donor_end = donor.len;
  int output_depth = 0;
  int iter = 0;
  do {
    ++iter;
    // Get donor subtree
    std::pair<int, int> donor_slice =
        get_subtree(donor.nodes + donor_start, donor_end - donor_start, rng);

    // Get indices w.r.t current subspace [donor_start,donor_end)
    int donor_substart = donor_slice.first;
    int donor_subend = donor_slice.second;

    // Update relative indices to global indices
    donor_substart += donor_start;
    donor_subend += donor_start;

    // Update to new subspace
    donor_start = donor_substart;
    donor_end = donor_subend;

    // Evolve on current subspace
    p_out.len =
        (prog_start) + (donor_end - donor_start) + (prog.len - prog_end);
    delete[] p_out.nodes;
    p_out.nodes = new node[p_out.len];

    // Copy slices using std::copy
    std::copy(prog.nodes, prog.nodes + prog_start, p_out.nodes);
    std::copy(donor.nodes + donor_start, donor.nodes + donor_end,
              p_out.nodes + prog_start);
    std::copy(prog.nodes + prog_end, prog.nodes + prog.len,
              p_out.nodes + (prog_start) + (donor_end - donor_start));

    output_depth = get_depth(p_out);
  } while (output_depth >= MAX_STACK_SIZE);

  // Set the depth of the final program
  p_out.depth = output_depth;
}

void subtree_mutation(const program &prog, program &p_out, const param &params,
                      PhiloxEngine &rng) {
  // Generate a random program and perform crossover
  program new_program;
  build_program(new_program, params, rng);
  crossover(prog, new_program, p_out, params, rng);
}

void hoist_mutation(const program &prog, program &p_out, const param &params,
                    PhiloxEngine &rng) {
  // Replace program subtree with a random sub-subtree

  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, rng);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  std::pair<int, int> sub_slice =
      get_subtree(prog.nodes + prog_start, prog_end - prog_start, rng);
  int sub_start = sub_slice.first;
  int sub_end = sub_slice.second;

  // Update subtree indices to global indices
  sub_start += prog_start;
  sub_end += prog_start;

  p_out.len = (prog_start) + (sub_end - sub_start) + (prog.len - prog_end);
  delete[] p_out.nodes; // added to free memory
  p_out.nodes = new node[p_out.len];
  p_out.metric = prog.metric;

  // Copy node slices using std::copy
  std::copy(prog.nodes, prog.nodes + prog_start, p_out.nodes);
  std::copy(prog.nodes + sub_start, prog.nodes + sub_end,
            p_out.nodes + prog_start);
  std::copy(prog.nodes + prog_end, prog.nodes + prog.len,
            p_out.nodes + (prog_start) + (sub_end - sub_start));

  // Update depth
  p_out.depth = get_depth(p_out);
}

} // namespace genetic

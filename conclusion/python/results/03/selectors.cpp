// CPU selectors for Notebook 03. No Python selection loop is used.
#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

using Result = std::tuple<torch::Tensor, double, double>;

namespace {
void check(const torch::Tensor& v, int64_t R, const torch::Tensor& T) {
    TORCH_CHECK(v.device().is_cpu() && v.scalar_type() == torch::kFloat32
                && v.dim() == 1 && v.is_contiguous(), "v must be contiguous CPU float32");
    TORCH_CHECK(0 <= R && R <= v.numel(), "R must be between 0 and N");
    TORCH_CHECK(T.device().is_cpu() && T.scalar_type() == torch::kFloat64
                && T.dim() == 1 && T.is_contiguous() && T.numel() > v.numel(),
                "T must be contiguous CPU float64 with N+1 entries");
}

// Select the largest k scores, breaking ties by the original index.
// Their internal order is irrelevant because the chosen tiles are disjoint.
std::vector<int64_t> largest(const std::vector<double>& scores, int64_t k) {
    std::vector<int64_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    auto better = [&](int64_t a, int64_t b) {
        return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
    };
    if (k > 0 && k < static_cast<int64_t>(order.size()))
        std::nth_element(order.begin(), order.begin() + k, order.end(), better);
    order.resize(k);
    return order;
}

// Report the selected importance and the cost of maximal, merged runs.
Result summarize(const torch::Tensor& v, torch::Tensor mask, const torch::Tensor& T) {
    const auto* values = v.data_ptr<float>();
    const auto* selected = mask.data_ptr<uint8_t>();
    const auto* cost = T.data_ptr<double>();
    double importance = 0, latency = 0;
    int64_t length = 0;
    for (int64_t i = 0; i < v.numel(); ++i) {
        if (selected[i]) {
            importance += values[i];
            ++length;
        } else {
            latency += cost[length];
            length = 0;
        }
    }
    return {mask, importance, latency + cost[length]};
}
}  // namespace

Result top_r(torch::Tensor v, int64_t R, torch::Tensor T) {
    check(v, R, T);
    auto mask = torch::zeros({v.numel()}, torch::kUInt8);
    const float* values = v.data_ptr<float>();
    std::vector<double> scores(values, values + v.numel());
    for (auto i : largest(scores, R)) mask.data_ptr<uint8_t>()[i] = 1;
    return summarize(v, mask, T);
}

Result tiles(torch::Tensor v, int64_t R, torch::Tensor T, int64_t B) {
    check(v, R, T);
    TORCH_CHECK(B >= 1, "tile width B must be positive");
    const int64_t n = v.numel();
    const auto* values = v.data_ptr<float>();
    std::vector<double> prefix(n + 1, 0);
    for (int64_t i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + values[i];

    if (R < B) {
        int64_t start = 0;
        for (int64_t i = 1; i + R <= n; ++i)
            if (prefix[i + R] - prefix[i] > prefix[start + R] - prefix[start]) start = i;
        auto mask = torch::zeros({n}, torch::kUInt8);
        std::fill_n(mask.data_ptr<uint8_t>() + start, R, uint8_t{1});
        return summarize(v, mask, T);
    }

    const int64_t k = R / B, residual = R % B;
    Result best;
    double best_ratio = -1;
    for (int64_t offset : {int64_t{0}, B / 2}) {
        std::vector<int64_t> starts;
        std::vector<double> scores;
        for (int64_t start = offset; start + B <= n; start += B) {
            starts.push_back(start);
            scores.push_back(prefix[start + B] - prefix[start]);
        }
        if (static_cast<int64_t>(starts.size()) < k) continue;
        auto mask = torch::zeros({n}, torch::kUInt8);
        auto* selected = mask.data_ptr<uint8_t>();
        for (auto j : largest(scores, k))
            std::fill_n(selected + starts[j], B, uint8_t{1});

        if (residual) {
            std::vector<int64_t> occupied(n + 1, 0);
            for (int64_t i = 0; i < n; ++i) occupied[i + 1] = occupied[i] + selected[i];
            int64_t start = -1;
            double score = -1;
            for (int64_t i = 0; i + residual <= n; ++i) {
                double candidate = prefix[i + residual] - prefix[i];
                if (occupied[i + residual] == occupied[i] && candidate > score) {
                    start = i;
                    score = candidate;
                }
            }
            if (start < 0) continue;
            std::fill_n(selected + start, residual, uint8_t{1});
        }
        auto result = summarize(v, mask, T);
        double ratio = std::get<1>(result) / std::get<2>(result);
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best = result;
        }
    }
    TORCH_CHECK(best_ratio >= 0, "no exact-R tiling was feasible");
    return best;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiles", &tiles, "Two-offset saturation tiles, with a disjoint residual");
    m.def("top_r", &top_r, "Top-R importance via partial selection");
}

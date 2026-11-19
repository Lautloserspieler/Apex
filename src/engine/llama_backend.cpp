#include "llama_backend.h"

#include <stdexcept>

#if defined(APEX_WITH_LLAMA)

#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if defined(APEX_WITH_CUDA)
#include <atomic>
#include <cstdio>
#include <optional>
#include <thread>
#endif

namespace apex {
namespace {

using Clock = std::chrono::steady_clock;
using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using SamplerPtr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
using BatchPtr = std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)>;

double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

void configure_batch(llama_batch_ext* batch, const llama_token* tokens,
                     int32_t count, llama_pos first_pos, bool output_logits) {
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < count; ++i) {
        const int32_t index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        if (index < 0) {
            throw std::runtime_error("Unable to add token to llama batch");
        }
        const llama_pos position = first_pos + i;
        if (!llama_batch_ext_set_pos(batch, index, &position)) {
            throw std::runtime_error("Unable to assign position to llama token");
        }
        if (output_logits && i == count - 1 &&
            !llama_batch_ext_set_output_logits(batch, index, true)) {
            throw std::runtime_error("Unable to enable final llama logits");
        }
    }
}

std::string token_piece(const llama_vocab* vocab, llama_token token) {
    std::vector<char> bytes(256);
    int32_t size = llama_token_to_piece(vocab, token, bytes.data(),
                                       static_cast<int32_t>(bytes.size()), 0, false);
    if (size < 0) {
        if (size == INT32_MIN) {
            throw std::runtime_error("Invalid token piece length");
        }
        bytes.resize(static_cast<std::size_t>(-size));
        size = llama_token_to_piece(vocab, token, bytes.data(),
                                   static_cast<int32_t>(bytes.size()), 0, false);
    }
    if (size < 0) {
        throw std::runtime_error("Failed to decode generated token");
    }
    return std::string(bytes.data(), static_cast<std::size_t>(size));
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text) {
    if (text.size() > static_cast<std::size_t>(INT32_MAX)) {
        throw std::length_error("Prompt exceeds tokenizer limit");
    }
    const auto text_len = static_cast<int32_t>(text.size());
    int32_t needed = llama_tokenize(vocab, text.c_str(), text_len,
                                     nullptr, 0, true, true);
    if (needed == INT32_MIN) {
        throw std::runtime_error("Tokenizer reported invalid length");
    }
    if (needed < 0) needed = -needed;
    if (needed == 0) {
        throw std::runtime_error("Tokenizer returned no tokens");
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(needed));
    const int32_t written = llama_tokenize(vocab, text.c_str(), text_len,
                                            tokens.data(), needed, true, true);
    if (written <= 0 || written > needed) {
        throw std::runtime_error("Failed to tokenize prompt");
    }
    tokens.resize(static_cast<std::size_t>(written));
    return tokens;
}

#if defined(APEX_WITH_CUDA)
#if defined(_WIN32)
FILE* popen_compat(const char* command, const char* mode) { return _popen(command, mode); }
int pclose_compat(FILE* stream) { return _pclose(stream); }
#else
FILE* popen_compat(const char* command, const char* mode) { return popen(command, mode); }
int pclose_compat(FILE* stream) { return pclose(stream); }
#endif

struct GpuSample {
    double vram_mib;
    double power_watts;
};

// One-shot nvidia-smi query. Never throws: any failure (binary missing,
// non-zero exit, unparsable output) yields std::nullopt, which the caller
// treats as "no sample this tick" rather than an error - these stats are
// diagnostic, not correctness-critical, and must not make generation fail.
std::optional<GpuSample> query_nvidia_smi() {
    FILE* pipe = popen_compat(
        "nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits -i 0",
        "r");
    if (!pipe) {
        return std::nullopt;
    }
    std::array<char, 256> buffer{};
    std::string line;
    if (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        line = buffer.data();
    }
    if (pclose_compat(pipe) != 0 || line.empty()) {
        return std::nullopt;
    }
    const auto comma = line.find(',');
    if (comma == std::string::npos) {
        return std::nullopt;
    }
    try {
        const double vram_mib = std::stod(line.substr(0, comma));
        const double power_watts = std::stod(line.substr(comma + 1));
        return GpuSample{vram_mib, power_watts};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// Background-thread RAII sampler covering one generate() call. Polls
// `nvidia-smi` every ~100ms to approximate peak VRAM and average power
// draw. This is a coarse, process-spawn-based approximation, not a
// continuous device-side measurement (e.g. via NVML callbacks) - adequate
// for Phase 3 diagnostics, not for fine-grained kernel profiling. Call
// stop() before reading the accumulated stats: it joins the background
// thread, which is the synchronization point that makes the plain (not
// atomic) accumulator fields below safe to read from the calling thread
// afterwards.
class GpuStatsSampler {
public:
    explicit GpuStatsSampler(bool active) : active_(active) {
        if (active_) {
            thread_ = std::thread([this] { run(); });
        }
    }

    ~GpuStatsSampler() { stop(); }

    GpuStatsSampler(const GpuStatsSampler&) = delete;
    GpuStatsSampler& operator=(const GpuStatsSampler&) = delete;

    void stop() {
        if (!active_ || stopped_) {
            return;
        }
        stop_requested_.store(true, std::memory_order_relaxed);
        if (thread_.joinable()) {
            thread_.join();
        }
        stopped_ = true;
    }

    double peak_vram_mib() const { return peak_vram_mib_; }
    double avg_power_watts() const {
        return sample_count_ > 0 ? power_sum_watts_ / static_cast<double>(sample_count_) : 0.0;
    }

private:
    void run() {
        using namespace std::chrono_literals;
        while (!stop_requested_.load(std::memory_order_relaxed)) {
            if (const auto sample = query_nvidia_smi()) {
                peak_vram_mib_ = std::max(peak_vram_mib_, sample->vram_mib);
                power_sum_watts_ += sample->power_watts;
                ++sample_count_;
            }
            std::this_thread::sleep_for(100ms);
        }
    }

    bool active_;
    bool stopped_ = false;
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
    double peak_vram_mib_ = 0.0;
    double power_sum_watts_ = 0.0;
    std::size_t sample_count_ = 0;
};
#endif // APEX_WITH_CUDA

} // namespace

struct LlamaBackend::Impl {
    explicit Impl(const std::string& path, const EngineOptions& opts)
        : options(opts), model(nullptr, &llama_model_free) {
        static std::once_flag initialized;
        std::call_once(initialized, [] { llama_backend_init(); });

        if (options.gpu_layers > 0 && !llama_supports_gpu_offload()) {
            throw std::runtime_error("GPU layers requested, but this build has no GPU offload backend");
        }

        const auto start = Clock::now();
        auto params = llama_model_default_params();
        params.n_gpu_layers = options.gpu_layers;
        // A non-null, empty device list means "no accelerator devices" in
        // pinned llama.cpp b12445. Setting n_gpu_layers=0 alone does NOT do
        // this: with a visible CUDA device the context may still select a
        // CUDA backend for non-weight operations (verified on RTX 5070).
        // Keep the list alive until model destruction because the upstream
        // model parameters may retain this pointer.
        if (options.gpu_layers == 0) {
            params.devices = cpu_only_devices.data();
        }
        model.reset(llama_model_load_from_file(path.c_str(), params));
        load_seconds = seconds(start, Clock::now());

        if (!model) {
            throw std::runtime_error("Unable to parse/load GGUF model: " + path);
        }
        if (llama_model_has_encoder(model.get()) || !llama_model_has_decoder(model.get())) {
            throw std::runtime_error("Phase 2 only supports decoder-only text models");
        }
        vocab = llama_model_get_vocab(model.get());
        if (!vocab) {
            throw std::runtime_error("GGUF model has no compatible vocabulary");
        }
    }

    EngineOptions options;
    // Must outlive model, which may retain llama_model_params.devices.
    std::array<ggml_backend_dev_t, 1> cpu_only_devices{{nullptr}};
    ModelPtr model;
    const llama_vocab* vocab = nullptr;
    double load_seconds = 0.0;
};

LlamaBackend::LlamaBackend(const std::string& path, const EngineOptions& options)
    : impl_(std::make_unique<Impl>(path, options)) {}

LlamaBackend::~LlamaBackend() = default;

GenerationResult LlamaBackend::generate(const std::string& prompt, int max_tokens) const {
    // Each request creates its own context and KV cache, so consecutive
    // generations cannot inadvertently reuse prior prompt state.
    const auto request_start = Clock::now();
    const auto tokens = tokenize(impl_->vocab, prompt);
    const auto tokenize_end = Clock::now();
    const std::size_t required = tokens.size() + static_cast<std::size_t>(max_tokens) - 1;
    if (required >= impl_->options.context_size) {
        throw std::length_error("Prompt + max_tokens exceeds --ctx-size; lower --max-tokens or raise the context limit");
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = impl_->options.context_size;
    ctx_params.n_batch = std::min<std::uint32_t>(512, impl_->options.context_size);
    ctx_params.n_ubatch = ctx_params.n_batch;
    ctx_params.n_seq_max = 1;
    if (impl_->options.threads > 0) {
        // Pin both prefill (batch) and decode thread pools so benchmark
        // runs are comparable across hosts instead of using a host-derived
        // upstream default.
        ctx_params.n_threads = impl_->options.threads;
        ctx_params.n_threads_batch = impl_->options.threads;
    }
    ContextPtr ctx(llama_init_from_model(impl_->model.get(), ctx_params), &llama_free);
    if (!ctx) {
        throw std::runtime_error("Cannot create llama context (possibly out of memory)");
    }

    auto params = llama_sampler_chain_default_params();
    SamplerPtr sampler(llama_sampler_chain_init(params), &llama_sampler_free);
    if (!sampler) throw std::runtime_error("Cannot initialize llama sampler");
    if (impl_->options.temperature == 0.0f) {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(impl_->options.top_k));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(impl_->options.top_p, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(impl_->options.temperature));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(impl_->options.seed));
    }

    BatchPtr batch(llama_batch_ext_init(ctx.get()), &llama_batch_ext_free);
    if (!batch) throw std::runtime_error("Unable to create llama batch");

#if defined(APEX_WITH_CUDA)
    // Only meaningful when the GPU is actually in use; a CPU-only run on a
    // CUDA-enabled build would just report host process VRAM noise.
    GpuStatsSampler gpu_sampler(impl_->options.gpu_layers > 0);
#endif

    GenerationResult output;
    output.stats.prompt_tokens = static_cast<int>(tokens.size());
    output.stats.model_load_seconds = impl_->load_seconds;
    output.stats.stop_reason = "length";

    const auto start = Clock::now();
    output.stats.tokenize_seconds = seconds(request_start, tokenize_end);
    output.stats.context_setup_seconds = seconds(tokenize_end, start);
    const auto batch_size = static_cast<std::size_t>(ctx_params.n_batch);
    for (std::size_t offset = 0; offset < tokens.size(); offset += batch_size) {
        const auto count = static_cast<int32_t>(std::min(batch_size, tokens.size() - offset));
        const bool is_final = offset + static_cast<std::size_t>(count) == tokens.size();
        configure_batch(batch.get(), tokens.data() + offset, count,
                        static_cast<llama_pos>(offset), is_final);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("Prompt prefill failed");
        }
    }
    const auto prefill_end = Clock::now();
    output.stats.prefill_seconds = seconds(start, prefill_end);

    Clock::time_point first_token_time{};
    for (int generated = 0; generated < max_tokens; ++generated) {
        const llama_token id = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        const auto sampled_at = Clock::now();
        if (generated == 0) {
            first_token_time = sampled_at;
            output.stats.ttft_seconds = seconds(start, sampled_at);
        }

        if (llama_vocab_is_eog(impl_->vocab, id)) {
            output.stats.stop_reason = "eos";
            break;
        }

        output.generated_token_ids.push_back(id);
        output.text += token_piece(impl_->vocab, id);
        ++output.stats.generated_tokens;
        if (generated + 1 == max_tokens) break;

        const llama_pos pos = static_cast<llama_pos>(tokens.size() + generated);
        configure_batch(batch.get(), &id, 1, pos, true);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("Autoregressive token decoding failed");
        }
    }
    output.stats.decode_seconds =
        first_token_time == Clock::time_point{} ? 0.0 : seconds(first_token_time, Clock::now());

#if defined(APEX_WITH_CUDA)
    gpu_sampler.stop();
    output.stats.peak_vram_mib = gpu_sampler.peak_vram_mib();
    output.stats.avg_power_watts = gpu_sampler.avg_power_watts();
#endif
    output.stats.request_seconds = seconds(request_start, Clock::now());

    return output;
}

} // namespace apex

#else

namespace apex {

// Linked in mock-only builds so the Engine's type stays the same.
struct LlamaBackend::Impl {};
LlamaBackend::LlamaBackend(const std::string&, const EngineOptions&) {
    throw std::logic_error("Real inference was disabled at build time");
}
LlamaBackend::~LlamaBackend() = default;
GenerationResult LlamaBackend::generate(const std::string&, int) const {
    throw std::logic_error("Real inference was disabled at build time");
}

} // namespace apex
#endif

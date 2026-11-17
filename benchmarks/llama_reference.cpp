// Standalone reference application calling the pinned upstream llama.cpp C API.
// This executable deliberately does not use any Apex Engine or tokenizer code.
// It verifies token ID parity of the Apex adapter, not numerical independence
// from llama.cpp (both use the same pinned compute implementation).
#include "llama.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
using Batch = std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)>;

void check(bool success, const char* message) {
    if (!success) throw std::runtime_error(message);
}

std::vector<llama_token> encode(const llama_vocab* vocab, const std::string& prompt) {
    check(prompt.size() <= static_cast<std::size_t>(INT32_MAX), "Prompt too long");
    const auto size = static_cast<int32_t>(prompt.size());
    int32_t count = llama_tokenize(vocab, prompt.c_str(), size, nullptr, 0, true, true);
    check(count != INT32_MIN, "Invalid tokenizer token count");
    if (count < 0) count = -count;
    check(count > 0, "Prompt tokenization failed");
    std::vector<llama_token> ids(static_cast<std::size_t>(count));
    const int32_t actual = llama_tokenize(
        vocab, prompt.c_str(), size, ids.data(), count, true, true);
    check(actual > 0 && actual <= count, "Prompt tokenization failed");
    ids.resize(static_cast<std::size_t>(actual));
    return ids;
}

void submit(llama_context* context, llama_batch_ext* batch,
            const llama_token* ids, int count, llama_pos start_pos) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        const auto index = llama_batch_ext_add_token(batch, 0, ids[i]);
        check(index >= 0, "Cannot append reference token");
        const auto position = start_pos + i;
        check(llama_batch_ext_set_pos(batch, index, &position), "Cannot set position");
        if (i == count - 1) {
            check(llama_batch_ext_set_output_logits(batch, index, true),
                  "Cannot request reference logits");
        }
    }
    check(llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch) == 0,
          "Reference llama.cpp forward pass failed");
}

void output_ids(const char* prefix, const std::vector<llama_token>& ids) {
    std::cout << prefix;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << ids[i];
    }
    std::cout << '\n';
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: apex_llama_reference MODEL PROMPT CTX_SIZE MAX_TOKENS GPU_LAYERS\n";
        return 2;
    }
    try {
        const std::string model_path(argv[1]);
        const std::string prompt(argv[2]);
        const int context_size = std::stoi(argv[3]);
        const int max_tokens = std::stoi(argv[4]);
        const int gpu_layers = std::stoi(argv[5]);
        check(context_size >= 16 && max_tokens > 0 && gpu_layers >= 0,
              "Invalid reference context, token limit or GPU layers");

        llama_backend_init();
        auto model_options = llama_model_default_params();
        model_options.n_gpu_layers = gpu_layers;
        Model model(llama_model_load_from_file(model_path.c_str(), model_options),
                    &llama_model_free);
        check(model != nullptr, "Reference cannot load GGUF");
        check(!llama_model_has_encoder(model.get()) &&
                  llama_model_has_decoder(model.get()), "Requires decoder-only model");
        const auto* vocab = llama_model_get_vocab(model.get());
        check(vocab != nullptr, "Reference model lacks vocabulary");

        const auto prompt_ids = encode(vocab, prompt);
        check(prompt_ids.size() + static_cast<std::size_t>(max_tokens) - 1 <=
                  static_cast<std::size_t>(context_size), "Prompt exceeds context budget");

        auto context_options = llama_context_default_params();
        context_options.n_ctx = static_cast<std::uint32_t>(context_size);
        context_options.n_batch = std::min<std::uint32_t>(512, context_options.n_ctx);
        context_options.n_ubatch = context_options.n_batch;
        context_options.n_seq_max = 1;
        Context context(llama_init_from_model(model.get(), context_options), &llama_free);
        check(context != nullptr, "Reference context allocation failed");

        Batch batch(llama_batch_ext_init(context.get()), &llama_batch_ext_free);
        check(batch != nullptr, "Reference batch allocation failed");
        for (std::size_t offset = 0; offset < prompt_ids.size();
             offset += context_options.n_batch) {
            const auto n = static_cast<int>(std::min<std::size_t>(
                context_options.n_batch, prompt_ids.size() - offset));
            submit(context.get(), batch.get(), prompt_ids.data() + offset, n,
                   static_cast<llama_pos>(offset));
        }

        Sampler greedy(llama_sampler_init_greedy(), &llama_sampler_free);
        check(greedy != nullptr, "Reference sampler allocation failed");
        std::vector<llama_token> generated;
        for (int i = 0; i < max_tokens; ++i) {
            const llama_token id = llama_sampler_sample(greedy.get(), context.get(), -1);
            if (llama_vocab_is_eog(vocab, id)) break;
            generated.push_back(id);
            if (i + 1 == max_tokens) break;
            const auto position = static_cast<llama_pos>(prompt_ids.size() +
                                                          static_cast<std::size_t>(i));
            submit(context.get(), batch.get(), &id, 1, position);
        }

        output_ids("REF_PROMPT_IDS=", prompt_ids);
        output_ids("REF_TOKEN_IDS=", generated);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[llama.cpp reference error] " << error.what() << '\n';
        return 1;
    }
}

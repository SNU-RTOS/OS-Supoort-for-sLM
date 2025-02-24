#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <random>

// included by nclee
#include <iostream>
#include <sys/mman.h>
#include <linux/perf_event.h>  // For perf_event_attr and PERF_* constants
#include <sys/syscall.h>       // For syscall and __NR_perf_event_open
#include <unistd.h>           // For syscall wrapper and pid_t
#include <sys/ioctl.h>
#include <unordered_map>
#include <stdexcept>
#include <sys/time.h>
#include <sys/resource.h>
#ifndef __NR_perf_event_open
#define __NR_perf_event_open 241  // Syscall number for aarch64
#endif


// AI EDGE TORCH
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/match.h"
#include "ai_edge_torch/generative/examples/cpp/utils.h"
#include "src/sentencepiece_processor.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/signature_runner.h"

// ----------------------
// absl::FLAGS definition
// ----------------------
ABSL_FLAG(std::string, tflite_model, "",
          "Two-signature tflite model for text generation using ODML tools.");
ABSL_FLAG(std::string, sentencepiece_model, "", "Path to the SentencePiece model file.");
ABSL_FLAG(std::string, prompt, "Write an email:", "Input prompt for the model.");
ABSL_FLAG(int, max_decode_steps, -1,
          "Number of tokens to generate. Defaults to the KV cache limit.");
ABSL_FLAG(std::string, start_token, "",
          "Optional start token appended to the beginning of the input prompt.");
ABSL_FLAG(std::string, stop_token, "",
          "Optional stop token that stops the decoding loop if encountered.");
ABSL_FLAG(int, num_threads, 4, "Number of threads to use. Defaults to 4.");
ABSL_FLAG(std::string, weight_cache_path, "",
          "Path for XNNPACK weight caching, e.g., /tmp/model.xnnpack_cache.");
ABSL_FLAG(std::string, lora_path, "", "Optional path to a LoRA artifact.");

namespace
{

    using ai_edge_torch::examples::AlignedAllocator;
    using ai_edge_torch::examples::LoRA;

    // Class for pagefault measurement
    struct PageFaultStats {
        long minor_faults;
        long major_faults;
        double duration_ms;
    
        PageFaultStats(long minor = 0, long major = 0, double dur = 0.0)
            : minor_faults(minor), major_faults(major), duration_ms(dur) {}
    };
    
    class PerfMonitor {
        private:
            struct EventFd {
                int fd_minor;
                int fd_major;
                std::chrono::steady_clock::time_point start_time;
            };
            std::unordered_map<std::string, EventFd> phase_fds;
            
            static long perf_event_open(struct perf_event_attr* hw_event, pid_t pid,
                                      int cpu, int group_fd, unsigned long flags) {
                return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
            }
        
            int setup_page_fault_counter(bool major_fault) {
                struct perf_event_attr pe;
                memset(&pe, 0, sizeof(struct perf_event_attr));
                pe.type = PERF_TYPE_SOFTWARE;
                pe.size = sizeof(struct perf_event_attr);
                pe.config = major_fault ? PERF_COUNT_SW_PAGE_FAULTS_MAJ : PERF_COUNT_SW_PAGE_FAULTS_MIN;
                pe.disabled = 1;
                pe.exclude_kernel = 1;
                pe.exclude_hv = 1;
        
                int fd = perf_event_open(&pe, 0, -1, -1, 0);
                if (fd == -1) {
                    throw std::runtime_error("Error opening perf event");
                }
                return fd;
            }
        
        public:
            void start_phase(const std::string& phase_name) {
                EventFd event_fd;
                event_fd.fd_minor = setup_page_fault_counter(false);
                event_fd.fd_major = setup_page_fault_counter(true);
                event_fd.start_time = std::chrono::steady_clock::now();
        
                // Start counting
                ioctl(event_fd.fd_minor, PERF_EVENT_IOC_RESET, 0);
                ioctl(event_fd.fd_major, PERF_EVENT_IOC_RESET, 0);
                ioctl(event_fd.fd_minor, PERF_EVENT_IOC_ENABLE, 0);
                ioctl(event_fd.fd_major, PERF_EVENT_IOC_ENABLE, 0);
        
                phase_fds[phase_name] = event_fd;
            }
        
            PageFaultStats end_phase(const std::string& phase_name) {
                auto it = phase_fds.find(phase_name);
                if (it == phase_fds.end()) {
                    throw std::runtime_error("Phase not found: " + phase_name);
                }
        
                auto& event_fd = it->second;
                
                // Stop counting
                ioctl(event_fd.fd_minor, PERF_EVENT_IOC_DISABLE, 0);
                ioctl(event_fd.fd_major, PERF_EVENT_IOC_DISABLE, 0);
        
                // Read counts
                long long count_minor, count_major;
                read(event_fd.fd_minor, &count_minor, sizeof(long long));
                read(event_fd.fd_major, &count_major, sizeof(long long));
        
                // Calculate duration
                auto end_time = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - event_fd.start_time).count();
        
                // Clean up
                close(event_fd.fd_minor);
                close(event_fd.fd_major);
                phase_fds.erase(it);
        
                return PageFaultStats{
                    static_cast<long>(count_minor),
                    static_cast<long>(count_major),
                    static_cast<double>(duration)
                };
            }
        };

    class PageFaultMetrics {
        public:
            void RecordStats(const std::string& phase, const PageFaultStats& stats) {
                if (phase_stats.find(phase) == phase_stats.end()) {
                    phase_stats[phase] = std::vector<PageFaultStats>();
                }
                phase_stats[phase].push_back(stats);
            }
        
            void PrintStats() const {
                for (const auto& [phase, stats_vec] : phase_stats) {
                    if (stats_vec.empty()) continue;
        
                    std::cout << "\nPhase: " << phase << "\n";
                    if (stats_vec.size() == 1) {
                        const auto& stats = stats_vec[0];
                        std::cout << "Duration: " << stats.duration_ms << " ms\n"
                                    << "Minor page faults: " << stats.minor_faults << "\n"
                                    << "Major page faults: " << stats.major_faults << "\n";
                    } else {
                        double avg_duration = 0;
                        long total_minor = 0, total_major = 0;
			std::string per_decode = "=== Decoding Result Per Step ===\n";
			int i = 0;
                        for (const auto& stats : stats_vec) {
                            avg_duration += stats.duration_ms;
                            total_minor += stats.minor_faults;
                            total_major += stats.major_faults;
			    per_decode += ("Decode step " + std::to_string(i++) + "\n");
			    per_decode += (" - Major page faults: " + std::to_string(stats.major_faults) + "\n");
                        }
                        avg_duration /= stats_vec.size();
        
                        std::cout << "Number of measurements: " << stats_vec.size() << "\n"
                                    << "Average duration: " << avg_duration << " ms\n"
                                    << "Total minor page faults: " << total_minor << "\n"
                                    << "Total major page faults: " << total_major << "\n"
                                    << "Average minor page faults: " << static_cast<double>(total_minor)/stats_vec.size() << "\n"
                                    << "Average major page faults: " << static_cast<double>(total_major)/stats_vec.size() << "\n"
				    << per_decode << std::endl;
                    }
                }
            }
        
        private:
            std::unordered_map<std::string, std::vector<PageFaultStats>> phase_stats;
        };

    // --------------------------------------------------------------------------
    // A scoped timer that prints the elapsed time when going out of scope
    // --------------------------------------------------------------------------
    class ScopeTimer
    {
    public:
        explicit ScopeTimer(const std::string &name)
            : name_(name),
              start_(std::chrono::high_resolution_clock::now()) {}

        ~ScopeTimer()
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
            std::cout << "\n[INFO] " << name_ << " took " << duration_ms << " ms\n";
        }

    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
    };

    // --------------------------------------------------------------------------
    // Class for measuring decoding metrics (time to first token, average times, etc.)
    // --------------------------------------------------------------------------
    class DecodingMetrics
    {
    public:
        // Called before decoding loop starts
        void StartDecoding()
        {
            decode_start_ = std::chrono::high_resolution_clock::now();
        }

        // Record times for each token
        //   - token_start: time point before inference/sampling starts for a token
        //   - inference_time_ms: how many ms were spent in model inference
        //   - sampling_time_ms : how many ms were spent in sampling the next token
        void RecordTimes(const std::chrono::high_resolution_clock::time_point &token_start,
                         double inference_time_ms, double sampling_time_ms)
        {
            auto token_end = std::chrono::high_resolution_clock::now();
            double decoding_time_ms =
                std::chrono::duration<double, std::milli>(token_end - token_start).count();

            // If this is the first token, record time to first token
            if (!first_token_recorded_)
            {
                first_token_recorded_ = true;
                time_to_first_token_ms_ =
                    std::chrono::duration<double, std::milli>(token_end - decode_start_).count();
            }

            // Track inference time
            total_inference_time_ms_ += inference_time_ms;
            // Track sampling time
            total_sampling_time_ms_ += sampling_time_ms;
            // Track total decoding time
            total_decoding_time_ms_ += decoding_time_ms;

            // Track total tokens
            ++token_count_;
        }

        // Print out final decoding metrics
        void PrintMetrics() const
        {
            double avg_inference_time_ms = 0.0;
            double avg_sampling_time_ms = 0.0;
            double avg_decoding_time_ms = 0.0;
            double avg_inference_speed = 0.0;
            double avg_sampling_speed = 0.0;
            double avg_decoding_speed = 0.0;

            if (token_count_ > 0)
            {
                avg_inference_time_ms = total_inference_time_ms_ / token_count_;
                avg_sampling_time_ms = total_sampling_time_ms_ / token_count_;
                avg_decoding_time_ms = (total_sampling_time_ms_ + total_inference_time_ms_) / token_count_;

                avg_inference_speed = token_count_ / (total_inference_time_ms_ / 1000);
                avg_sampling_speed = token_count_ / (total_sampling_time_ms_ / 1000);
                avg_decoding_speed = token_count_ / (total_decoding_time_ms_ / 1000);
            }

            std::cout << "\n\n================================\n";
            std::cout << "[INFO] Decoding stage completed\n";
            std::cout << "[METRICS] Total Number of Generated Tokens : " << token_count_ << " tokens\n\n";

            std::cout << "[METRICS] Total Inference Latency          : " << total_inference_time_ms_ << " ms\n";
            std::cout << "[METRICS] Total Sampling Latency           : " << total_sampling_time_ms_ << " ms\n";
            std::cout << "[METRICS] Total Decoding Latency           : " << total_decoding_time_ms_ << " ms\n\n";

            std::cout << "[METRICS] Time To First Token              : " << time_to_first_token_ms_ << " ms\n";
            std::cout << "[METRICS] Average Inference Latency        : " << avg_inference_time_ms << " ms/tokens"
                      << "(" << avg_inference_speed << " token/s )\n";
            std::cout << "[METRICS] Average Sampling Latency         : " << avg_sampling_time_ms << " ms/tokens"
                      << "(" << avg_sampling_speed << " token/s )\n";
            std::cout << "[METRICS] Average Decoding Latency         : " << avg_decoding_time_ms << " ms/tokens"
                      << "(" << avg_decoding_speed << " token/s )\n";
        }

    private:
        // Decode start time
        std::chrono::high_resolution_clock::time_point decode_start_;

        // Time to first token
        double time_to_first_token_ms_ = 0.0;
        bool first_token_recorded_ = false;

        // Accumulators
        double total_inference_time_ms_ = 0.0;
        double total_sampling_time_ms_ = 0.0;
        double total_decoding_time_ms_ = 0.0;
        int token_count_ = 0;
    };

    // --------------------------------------------------------------------------
    // A class that provides various sampling methods (Greedy, Top-K, Top-P, etc.)
    // --------------------------------------------------------------------------
    class Sampler
    {
    public:
        // ------------------------
        // Greedy Sampler
        // ------------------------
        static int GreedySampler(const TfLiteTensor *logits)
        {
            float max_value = -std::numeric_limits<float>::infinity();
            int max_index = 0;
            int vocab_size = logits->dims->data[2];

            for (int i = 0; i < vocab_size; ++i)
            {
                if (logits->data.f[i] > max_value)
                {
                    max_value = logits->data.f[i];
                    max_index = i;
                }
            }
            return max_index;
        }

        // ------------------------
        // Top-K Sampler
        // ------------------------
        static int TopKSampler(const TfLiteTensor *logits, int k)
        {
            int vocab_size = logits->dims->data[2];
            std::vector<std::pair<float, int>> sorted_logits;
            sorted_logits.reserve(vocab_size);

            for (int i = 0; i < vocab_size; ++i)
            {
                sorted_logits.emplace_back(logits->data.f[i], i);
            }

            // Partial sort to get the top k elements
            if (k < vocab_size)
            {
                std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + k, sorted_logits.end(),
                                  std::greater<std::pair<float, int>>());
                sorted_logits.resize(k);
            }
            else
            {
                // If k >= vocab_size, no need to cut
                std::sort(sorted_logits.begin(), sorted_logits.end(), std::greater<std::pair<float, int>>());
            }

            // Compute normalized probabilities
            float sum_probs = 0.0f;
            for (auto &pair : sorted_logits)
            {
                sum_probs += std::exp(pair.first);
            }
            std::vector<float> probabilities;
            probabilities.reserve(sorted_logits.size());
            for (auto &pair : sorted_logits)
            {
                probabilities.push_back(std::exp(pair.first) / sum_probs);
            }

            // Multinomial sampling
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

            return sorted_logits[dist(gen)].second;
        }

        // ------------------------
        // Top-P (Nucleus) Sampler
        // ------------------------
        static int TopPSampler(const TfLiteTensor *logits, float p)
        {
            int vocab_size = logits->dims->data[2];
            std::vector<std::pair<float, int>> sorted_logits;
            sorted_logits.reserve(vocab_size);

            for (int i = 0; i < vocab_size; ++i)
            {
                sorted_logits.emplace_back(logits->data.f[i], i);
            }

            // Sort descending by logit value
            std::sort(sorted_logits.begin(), sorted_logits.end(),
                      std::greater<std::pair<float, int>>());

            // Apply softmax to get probabilities
            std::vector<float> probabilities(vocab_size);
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; ++i)
            {
                float val = std::exp(sorted_logits[i].first);
                probabilities[i] = val;
                sum_exp += val;
            }
            for (int i = 0; i < vocab_size; ++i)
            {
                probabilities[i] /= sum_exp;
            }

            // Find the cutoff index where cumulative probability exceeds p
            float cumulative_prob = 0.0f;
            int cutoff_index = vocab_size - 1;
            for (int i = 0; i < vocab_size; ++i)
            {
                cumulative_prob += probabilities[i];
                if (cumulative_prob > p)
                {
                    cutoff_index = i;
                    break;
                }
            }

            // Resize vectors to [0..cutoff_index]
            float new_sum = 0.0f;
            for (int i = 0; i <= cutoff_index; ++i)
            {
                new_sum += probabilities[i];
            }
            for (int i = 0; i <= cutoff_index; ++i)
            {
                probabilities[i] /= new_sum;
            }

            probabilities.resize(cutoff_index + 1);
            sorted_logits.resize(cutoff_index + 1);

            // Multinomial sampling
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
            return sorted_logits[dist(gen)].second;
        }

        // ------------------------
        // Temperature + Top-K + Top-P Sampler
        // ------------------------
        static int TemperatureTopKTopPSampler(const TfLiteTensor *logits,
                                              float temperature, int k, float p)
        {
            int vocab_size = logits->dims->data[2];
            std::vector<std::pair<float, int>> sorted_logits;
            sorted_logits.reserve(vocab_size);

            // 1) Apply Temperature
            std::vector<float> scaled_logits(vocab_size);
            for (int i = 0; i < vocab_size; ++i)
            {
                scaled_logits[i] = logits->data.f[i] / temperature;
            }

            // 2) Softmax over scaled logits
            float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; ++i)
            {
                scaled_logits[i] = std::exp(scaled_logits[i] - max_logit);
                sum_exp += scaled_logits[i];
            }
            for (int i = 0; i < vocab_size; ++i)
            {
                scaled_logits[i] /= sum_exp;
                // Keep index-value pairs for sorting
                sorted_logits.emplace_back(scaled_logits[i], i);
            }

            // 3) Sort descending by probability
            std::sort(sorted_logits.begin(), sorted_logits.end(),
                      std::greater<std::pair<float, int>>());

            // 4) Top-K filter
            int top_k = std::min(k, vocab_size);
            sorted_logits.resize(top_k);

            // 5) Top-P filter within top-k
            float cumulative_prob = 0.0f;
            int cutoff_index = top_k - 1;
            for (int i = 0; i < top_k; ++i)
            {
                cumulative_prob += sorted_logits[i].first;
                if (cumulative_prob > p)
                {
                    cutoff_index = i;
                    break;
                }
            }
            sorted_logits.resize(cutoff_index + 1);

            // 6) Renormalize final probabilities
            float new_sum = 0.0f;
            for (auto &pair : sorted_logits)
            {
                new_sum += pair.first;
            }

            std::vector<float> final_probs;
            final_probs.reserve(sorted_logits.size());
            for (auto &pair : sorted_logits)
            {
                final_probs.push_back(pair.first / new_sum);
            }

            // 7) Multinomial sampling
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(final_probs.begin(), final_probs.end());
            return sorted_logits[dist(gen)].second;
        }
    };

    // --------------------------------------------------------------------------
    // Utility for applying XNNPACK weight caching
    // --------------------------------------------------------------------------
    void ApplyXNNPACKWeightCaching(tflite::Interpreter *interpreter)
    {
        auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
        std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
        delegate_options.weight_cache_file_path = weight_cache_path.c_str();
        delegate_options.num_threads = absl::GetFlag(FLAGS_num_threads);
        delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING;
        delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;

        MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(
                          tflite::Interpreter::TfLiteDelegatePtr(
                              TfLiteXNNPackDelegateCreate(&delegate_options),
                              [](TfLiteDelegate *delegate)
                              { TfLiteXNNPackDelegateDelete(delegate); })) == kTfLiteOk);
    }

    // --------------------------------------------------------------------------
    // Loads the TFLite model
    // --------------------------------------------------------------------------
    std::unique_ptr<tflite::FlatBufferModel> LoadModel()
    {
        std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(absl::GetFlag(FLAGS_tflite_model).c_str());
        MINIMAL_CHECK(model != nullptr);
        return model;
    }

    // --------------------------------------------------------------------------
    // Builds a TFLite interpreter from the model and applies XNNPACK if requested
    // --------------------------------------------------------------------------
    std::unique_ptr<tflite::Interpreter>
    BuildInterpreter(tflite::FlatBufferModel *model, int num_threads)
    {
        tflite::ops::builtin::BuiltinOpResolver resolver;
        // Register GenAI custom ops
        tflite::ops::custom::GenAIOpsRegisterer(&resolver);

        tflite::InterpreterBuilder builder(*model, resolver);
        MINIMAL_CHECK(builder.SetNumThreads(num_threads) == kTfLiteOk);

        std::unique_ptr<tflite::Interpreter> interpreter;
        builder(&interpreter);
        MINIMAL_CHECK(interpreter != nullptr);

        if (!absl::GetFlag(FLAGS_weight_cache_path).empty())
        {
            ApplyXNNPACKWeightCaching(interpreter.get());
        }
        return interpreter;
    }

    // --------------------------------------------------------------------------
    // Constructs KV cache input structures for decode, based on the decode signature
    // --------------------------------------------------------------------------
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>
    BuildKVCache(tflite::Interpreter *interpreter)
    {
        tflite::SignatureRunner *runner = interpreter->GetSignatureRunner("decode");
        if (runner == nullptr)
        {
            return {};
        }

        // Expect runner->input_size() = tokens, input_pos, plus 2*(num_layers)
        size_t num_layers = (runner->input_size() - 2) / 2;
        if (num_layers == 0)
        {
            return {};
        }

        std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
        for (int i = 0; i < num_layers; ++i)
        {
            std::string k_cache_name = "kv_cache_k_" + std::to_string(i);
            std::string v_cache_name = "kv_cache_v_" + std::to_string(i);

            TfLiteTensor *tensor = runner->input_tensor(k_cache_name.c_str());
            size_t count = tensor->bytes / sizeof(float);

            kv_cache.emplace(k_cache_name,
                             std::vector<float, AlignedAllocator<float>>(count, 0.0f));
            kv_cache.emplace(v_cache_name,
                             std::vector<float, AlignedAllocator<float>>(count, 0.0f));
        }
        return kv_cache;
    }

    // --------------------------------------------------------------------------
    // Sets custom memory allocations for the KV cache on the given runner
    // --------------------------------------------------------------------------
    void PrepareRunner(tflite::SignatureRunner *runner,
                       std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache)
    {
        for (auto &[name, cache] : kv_cache)
        {
            TfLiteCustomAllocation allocation{
                .data = static_cast<void *>(cache.data()),
                .bytes = cache.size() * sizeof(float)};

            MINIMAL_CHECK(runner->SetCustomAllocationForInputTensor(name.c_str(), allocation) == kTfLiteOk);
            MINIMAL_CHECK(runner->SetCustomAllocationForOutputTensor(name.c_str(), allocation) == kTfLiteOk);
        }
        MINIMAL_CHECK(runner->AllocateTensors() == kTfLiteOk);
    }

    // --------------------------------------------------------------------------
    // Finds the appropriate "prefill" runner for the given number of tokens.
    // If LoRA is used, it defers to LoRA's specialized runner selection.
    // --------------------------------------------------------------------------
    tflite::SignatureRunner *GetPrefillRunner(
        tflite::Interpreter *interpreter,
        std::size_t num_input_tokens,
        std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache,
        const ai_edge_torch::examples::LoRA *lora)
    {
        tflite::SignatureRunner *runner = nullptr;
        int best_seq_size = -1;
        int delta = std::numeric_limits<int>::max();

        for (const std::string *key : interpreter->signature_keys())
        {
            if (!absl::StrContains(*key, "prefill") || absl::StrContains(*key, "lora"))
            {
                continue;
            }
            TfLiteTensor *input_pos =
                interpreter->GetSignatureRunner(key->c_str())->input_tensor("input_pos");
            int seq_size = input_pos->dims->data[0];

            // Choose the runner where seq_size >= num_input_tokens and
            // (seq_size - num_input_tokens) is minimized
            if (num_input_tokens <= static_cast<size_t>(seq_size) &&
                seq_size - static_cast<int>(num_input_tokens) < delta)
            {
                if (lora == nullptr)
                {
                    runner = interpreter->GetSignatureRunner(key->c_str());
                }
                best_seq_size = seq_size;
                delta = seq_size - static_cast<int>(num_input_tokens);
            }
        }

        // If LoRA is enabled, use the LoRA-specific prefill runner
        if (lora != nullptr)
        {
            runner = lora->GetPrefillRunner(interpreter, best_seq_size);
        }
        MINIMAL_CHECK(runner != nullptr);

        // Prepare KV memory allocations
        PrepareRunner(runner, kv_cache);
        return runner;
    }

    // --------------------------------------------------------------------------
    // Retrieves the decode runner (LoRA-based if needed) and prepares it
    // --------------------------------------------------------------------------
    tflite::SignatureRunner *GetDecodeRunner(
        tflite::Interpreter *interpreter,
        std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache,
        ai_edge_torch::examples::LoRA *lora)
    {
        tflite::SignatureRunner *runner =
            (lora == nullptr)
                ? interpreter->GetSignatureRunner("decode")
                : lora->GetDecodeRunner(interpreter);
        MINIMAL_CHECK(runner != nullptr);

        PrepareRunner(runner, kv_cache);
        return runner;
    }

    // --------------------------------------------------------------------------
    // Loads the SentencePiece model from file
    // --------------------------------------------------------------------------
    std::unique_ptr<sentencepiece::SentencePieceProcessor> LoadSentencePieceProcessor()
    {
        std::ifstream input(absl::GetFlag(FLAGS_sentencepiece_model), std::ios::binary);
        std::string serialized_proto((std::istreambuf_iterator<char>(input)),
                                     std::istreambuf_iterator<char>());

        auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
        MINIMAL_CHECK(processor->LoadFromSerializedProto(serialized_proto).ok());
        return processor;
    }

    void PinFrequentTensors(tflite::Interpreter* interpreter) {
        const tflite::Subgraph& primary_subgraph = interpreter->primary_subgraph();
        const std::vector<int>& execution_plan = primary_subgraph.execution_plan();
        
        // Track tensor access frequency
        std::unordered_map<int, int> tensor_access_count;
        std::unordered_map<int, TfLiteTensor*> tensor_map;
    
        // Count tensor accesses through the execution plan
        for(int node_idx: execution_plan) {
            const auto* node_and_reg = primary_subgraph.node_and_registration(node_idx);
            const TfLiteNode* node = &node_and_reg->first;
    
            // Track input tensors
            for (int i = 0; i < node->inputs->size; ++i) {
                int tensor_idx = node->inputs->data[i];
                tensor_access_count[tensor_idx]++;
                tensor_map[tensor_idx] = interpreter->tensor(tensor_idx);
            }
    
            // Track output tensors
            for (int i = 0; i < node->outputs->size; ++i) {
                int tensor_idx = node->outputs->data[i];
                tensor_access_count[tensor_idx]++;
                tensor_map[tensor_idx] = interpreter->tensor(tensor_idx);
	    }

	    // Track temporary tensors
	    for (int i = 0; i < node->temporaries->size; ++i) {
	    	int tensor_idx = node->temporaries->data[i];
		tensor_access_count[tensor_idx]++;
		tensor_map[tensor_idx] = interpreter->tensor(tensor_idx);
	    }
        }
    
        // Pin tensors based on access count threshold
        const int ACCESS_COUNT_THRESHOLD = 2;  // Pin tensors accessed more than twice
        const size_t MAX_TOTAL_SIZE = 1024*1024*1024;  // 1GB maximum total pinned size
        
        size_t total_pinned_size = 0;
        int pinned_count = 0;
    
        for (const auto& [tensor_idx, access_count] : tensor_access_count) {
            // Skip if access count is below threshold
            if (access_count < ACCESS_COUNT_THRESHOLD) {
                continue;
            }
    
            TfLiteTensor* tensor = tensor_map[tensor_idx];
            
            // Skip if we'd exceed maximum total size
            if (tensor && tensor->data.raw && tensor->bytes > 0 && 
                total_pinned_size + tensor->bytes <= MAX_TOTAL_SIZE) {
    
                if (mlock(tensor->data.raw, tensor->bytes) == 0) {
                    total_pinned_size += tensor->bytes;
                } else {
                    std::cerr << strerror(errno) << "\n";
                }
            }
        }
    }

    ////// Memory Reordering
    std::unique_ptr<uint8_t[]> g_combined_buffer = nullptr;
    size_t g_buffer_size = 0;

    // Function to free the global buffer if needed
    void freeReorganizedTensorBuffer() {
        if (g_combined_buffer) {
            g_combined_buffer.reset();
            g_buffer_size = 0;
            // std::cout << "Freed reorganized tensor buffer\n";
        }
    }

    void reorderTensorSimple(std::unique_ptr<tflite::Interpreter>& interpreter) {
        if (!interpreter) {
            return;
        }
        
        // Free any existing buffer before reorganizing
        freeReorganizedTensorBuffer();
        
        const std::vector<int>& execution_plan = interpreter->execution_plan();
        if (execution_plan.empty()) {
            return;
        }
        
        // First pass: calculate total size needed
        size_t total_size = 0;
        for (int node_index : execution_plan) {
            const auto* node_and_reg = interpreter->node_and_registration(node_index);
            if (!node_and_reg) continue;
    
            const TfLiteNode& node = node_and_reg->first;
            if (!node.inputs || !node.inputs->data) continue;
            
            for (int j = 0; j < node.inputs->size; ++j) {
                int tensor_index = node.inputs->data[j];
                if (tensor_index < 0) continue;
    
                TfLiteTensor* tensor = interpreter->tensor(tensor_index);
                if (!tensor || !tensor->data.raw) continue;
                
                if (tensor->allocation_type != kTfLiteMmapRo) continue;
                
                total_size += tensor->bytes;
            }
        }
        
        if (total_size == 0) return;
        
        try {
            g_combined_buffer = std::make_unique<uint8_t[]>(total_size);
            g_buffer_size = total_size;
        } catch (const std::bad_alloc&) {
            return;
        }
        
        // Second pass: copy tensors and update addresses
        size_t current_offset = 0;
        for (int node_index : execution_plan) {
            const auto* node_and_reg = interpreter->node_and_registration(node_index);
            if (!node_and_reg) continue;
    
            const TfLiteNode& node = node_and_reg->first;
            if (!node.inputs || !node.inputs->data) continue;
            
            for (int j = 0; j < node.inputs->size; ++j) {
                int tensor_index = node.inputs->data[j];
                if (tensor_index < 0) continue;
    
                TfLiteTensor* tensor = interpreter->tensor(tensor_index);
                if (!tensor || !tensor->data.raw) continue;
                
                if (tensor->allocation_type != kTfLiteMmapRo) continue;
                
                // Copy data and update pointer
                std::memcpy(g_combined_buffer.get() + current_offset,
                           tensor->data.raw,
                           tensor->bytes);
                tensor->data.raw = reinterpret_cast<char*>(g_combined_buffer.get() + current_offset);
                current_offset += tensor->bytes;
            }
        }
    }

    // Optional: Function to get the current buffer size
    size_t getReorganizedBufferSize() {
        return g_buffer_size;
    }

    // Optional: Function to check if buffer is allocated
    bool isBufferAllocated() {
        return g_combined_buffer != nullptr;
    }

    // RUSAGE
    struct RUsageRecord {
        rusage start;
        rusage end;
    };

    double toSeconds(const struct timeval& tv) {
        return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
    }

    void PrintRUsage(rusage usage_start, rusage usage_end, const std::string phase_name) {
        double user_time_start = toSeconds(usage_start.ru_utime);
        double user_time_end   = toSeconds(usage_end.ru_utime);
        double sys_time_start  = toSeconds(usage_start.ru_stime);
        double sys_time_end    = toSeconds(usage_end.ru_stime);
        double cpu_time_sec = (user_time_end - user_time_start)
                                + (sys_time_end - sys_time_start);
	    double user_time_sec = (user_time_end - user_time_start);
	    double sys_time_sec = (sys_time_end - sys_time_start);
        std::cout << phase_name << " took \n- "
            << cpu_time_sec << " [sec] CPU time\n- "
	       << user_time_sec << " [sec] User time\n- "
	       << sys_time_sec << " [sec] System time" << std::endl;
    }

    void PrintRUsageRecords(const std::vector<RUsageRecord>& records) {
        for (size_t i = 0; i < records.size(); i++) {
            PrintRUsage(records[i].start, records[i].end, "Decode " + std::to_string(i));
        }
    }

    void AnalyzeDelegateExecution(tflite::Interpreter* interpreter) {
        // Lambda function to print tensor details
        auto print_tensor_details = [](int tensor_idx, TfLiteTensor* tensor) {
            if (!tensor) {
                std::cout << "Tensor " << tensor_idx << " is NULL\n";
                return;
            }
            
            // std::cout << "Tensor: " << tensor_idx << " ";
            
            void* tensor_data_address = tensor->data.raw;
            std::cout << "Data Address: " << tensor_data_address << " ";

            // Tensor Type
            const char* type_name = TfLiteTypeGetName(tensor->type);
            std::cout << "Type: " << (type_name ? type_name : "Unknown") << " ";

            // Tensor Allocation Type
            std::cout << "Allocation Type: ";
            switch (tensor->allocation_type) {
                case kTfLiteArenaRw:
                    std::cout << "Arena RW " << "Bytes: " << tensor->bytes << " ";
                    break;
                case kTfLiteArenaRwPersistent:
                    std::cout << "Arena Persistent " << "Bytes: " << tensor->bytes << " ";
                    break;
                case kTfLiteMmapRo:
                    std::cout << "Mmap " << "Bytes: " << tensor->bytes << " ";
                    break;
                case kTfLiteDynamic:
                    std::cout << "Dynamic " << "Bytes: " << tensor->bytes << " ";
                    break;
                case kTfLiteCustom:
                    std::cout << "Custom " << "Bytes: " << tensor->bytes << " ";
                    break;
                case kTfLitePersistentRo:
                    std::cout << "PersistentRo " << "Bytes: " << tensor->bytes << " ";
                    break;
                case kTfLiteVariantObject:
                    std::cout << "Variant " << "Bytes: " << tensor->bytes << " ";
                    break;
                case kTfLiteMemNone:
                    std::cout << "MemNone " << "Bytes: 0 ";
                    break;
                default:
                    std::cout << "Unknown " << "Bytes: 0 ";
                    break;
            }

            // Tensor Shape
            std::cout << "Shape: [";
            if(tensor->dims && tensor->dims->size > 0){
                for (int dim_idx = 0; dim_idx < tensor->dims->size; ++dim_idx) {
                    std::cout << tensor->dims->data[dim_idx];
                    if (dim_idx < tensor->dims->size - 1) std::cout << ", ";
                }
            }
            std::cout << "]\n";
        };
        
        std::cout << "\n=== Delegate Execution Analysis ===\n";
        std::cout << "===================================\n";
    
        // Analyze the primary subgraph's execution plan
        const tflite::Subgraph& primary_subgraph = interpreter->primary_subgraph();
        // std::cout <<  "Subgraph Name: " << primary_subgraph.name_ << std::endl;
        const std::vector<int>& execution_plan = primary_subgraph.execution_plan();
    
        std::vector<int> delegated_nodes;
        std::vector<int> non_delegated_nodes;
    
        for (int node_idx : execution_plan) {
            const auto* node_and_reg = primary_subgraph.node_and_registration(node_idx);
            const TfLiteNode* node = &node_and_reg->first;
            
            if (node->delegate) {
                delegated_nodes.push_back(node_idx);
            } else {
                non_delegated_nodes.push_back(node_idx);
            }
        }
    
        //std::cout << "Delegate Analysis:\n";
        std::cout << "  Total Nodes: " << execution_plan.size() << "\n";
        std::cout << "  Delegated Nodes: " << delegated_nodes.size() << "\n";
        std::cout << "  Non-Delegated Nodes: " << non_delegated_nodes.size() << "\n";
        int num_input_tensors = 0;
        int num_output_tensors = 0;
        int num_intermediate_tensors = 0;
        int num_temporary_tensors = 0;
        int num_total_tensors = 0;

        // Detailed node analysis based on execution order
        std::cout << "=== Node Details ===" << std::endl;
        for(int node_idx: execution_plan) {
            const auto* node_and_reg = primary_subgraph.node_and_registration(node_idx);
            const TfLiteNode* node = &node_and_reg->first;
            const TfLiteRegistration& registration = node_and_reg->second;

            std::cout << "  Node " << node_idx << ":\n";
            std::cout << "    Operator: " 
                      << (registration.builtin_code ? 
                          tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(registration.builtin_code)) 
                          : "Custom Operator") 
                      << "\n";
            // Print Input Tensor Indices
            std::cout << "    Input Tensors:\n";
            for (int i = 0; i < node->inputs->size; ++i) {
                // 여기서 allocation type하고 address 찍을 수 있을 것 같은데 allocate 한 뒤에 찍어보면
                
                std::cout << "      Input " << i << ": " << node->inputs->data[i] << " ";
                
                uint32_t tensor_idx = node->inputs->data[i];
                auto* tensor = interpreter->tensor(tensor_idx);
                print_tensor_details(tensor_idx, tensor);
                
                ++num_input_tensors;
            }
            
            // Print Output Tensor Indices
            std::cout << "    Output Tensors:\n";
            for (int i = 0; i < node->outputs->size; ++i) {
                std::cout << "      Output " << i << ": " << node->outputs->data[i] << " ";

                uint32_t tensor_idx = node->outputs->data[i];
                auto* tensor = interpreter->tensor(tensor_idx);
                print_tensor_details(tensor_idx, tensor);

                ++num_output_tensors;
            }
            
            // Print Intermediate Tensor Indices
            std::cout << "    Intermediate Tensors:\n";
            for (int i = 0; i < node->intermediates->size; ++i) {
                std::cout << "      Intermediate " << i << ": " << node->intermediates->data[i] << " ";

                uint32_t tensor_idx = node->intermediates->data[i];
                auto* tensor = interpreter->tensor(tensor_idx);
                print_tensor_details(tensor_idx, tensor);

                ++num_intermediate_tensors;
            }
            
            // Print Temporary Tensor Indices
            std::cout << "    Temporary Tensors:\n";
            for (int i = 0; i < node->temporaries->size; ++i) {
                std::cout << "      Temporary " << i << ": " << node->temporaries->data[i] << " ";

                uint32_t tensor_idx = node->temporaries->data[i];
                auto* tensor = interpreter->tensor(tensor_idx);
                print_tensor_details(tensor_idx, tensor);

                ++num_temporary_tensors;
            }
        }
    }
    void PrintWorkingSetSize() {
        FILE* file = fopen("/proc/self/statm", "r");
        if (file) {
            long total_pages = 0, resident_pages = 0;
            // /proc/self/statm의 두 번째 필드는 resident 페이지 수입니다.
            if (fscanf(file, "%ld %ld", &total_pages, &resident_pages) == 2) {
                long page_size = sysconf(_SC_PAGESIZE);  // 페이지 크기 (바이트)
                double working_set_size_mb = (resident_pages * page_size) / (1024.0 * 1024.0);
                std::cout << "[INFO] Current Working Set Size: " 
                          << working_set_size_mb << " MB" << std::endl;
            }
            fclose(file);
        } else {
            std::cerr << "[ERROR] Unable to open /proc/self/statm to read working set size." 
                      << std::endl;
        }
    }

} // end anonymous namespace

// =======================================================================
// main() entry
// =======================================================================
int main(int argc, char *argv[])
{
    // 0. Parse flags
    absl::ParseCommandLine(argc, argv);
    std::cout << "[INFO] Preparing Required Components\n";

    // Global variables
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;
    std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
    std::unique_ptr<ai_edge_torch::examples::LoRA> lora = nullptr;
    std::vector<int> prompt_tokens;
    std::string prompt, start_token, stop_token;
    int stop_token_id = -1;

    // 0-1. Perf monitor initialziation
    PerfMonitor perf_monitor;
    PageFaultMetrics metrics;

    // 0-2. Variable for CPU time only
    rusage usage_start, usage_end;


    PageFaultStats stats;
    // 1. Load Model
    {
        ScopeTimer timer("Model Loading");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Model_Loading");
        model = LoadModel();
        stats = perf_monitor.end_phase("Model_Loading");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Model Loading");
    metrics.RecordStats("Model_Loading", stats);

    // 2. Build Interpreter
    {
        ScopeTimer timer("Interpreter Building");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Build_Interperter");
        interpreter = BuildInterpreter(model.get(), absl::GetFlag(FLAGS_num_threads));
        stats = perf_monitor.end_phase("Build_Interperter");
        getrusage(RUSAGE_SELF, &usage_end);
        // reorderTensorSimple(interpreter);
    }
    PrintRUsage(usage_start, usage_end, "Interpreter Building");
    metrics.RecordStats("Build_Interpreter", stats);

    // Tensor Reordering
    {
        ScopeTimer timer("Tensor Reordering");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Reorder_Tensor");
        // reorderTensorSimple(interpreter);
        stats = perf_monitor.end_phase("Reorder_Tensor");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Tensor Reordering");
    metrics.RecordStats("Reorder_Tensor", stats);

    // 3. Load SentencePiece
    {
        ScopeTimer timer("SentencePiece Loading");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Load_SentencePiece");
        sp_processor = LoadSentencePieceProcessor();
        stats = perf_monitor.end_phase("Load_SentencePiece");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Sentence Piece Loading");
    metrics.RecordStats("Load_SentencePiece", stats);

    // 4. Build KV Cache
    {
        ScopeTimer timer("KV Cache Building");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Build_KVCache");
        kv_cache = BuildKVCache(interpreter.get());
        MINIMAL_CHECK(!kv_cache.empty());
        stats = perf_monitor.end_phase("Build_KVCache");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "KV Cache Building");
    metrics.RecordStats("Build_KVCache", stats);

    // 5. Optionally load LoRA
    // {
    //     ScopeTimer timer("LoRA Loading");
    //     if (!absl::GetFlag(FLAGS_lora_path).empty())
    //     {
    //         lora = ai_edge_torch::examples::LoRA::FromFile(absl::GetFlag(FLAGS_lora_path));
    //         MINIMAL_CHECK(lora != nullptr);
    //     }
    // }

    // 6. Prepare Input Prompt
    {
        ScopeTimer timer("Input Prompt Preparation");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Prepare_Prompt");
        prompt = absl::GetFlag(FLAGS_prompt);
        MINIMAL_CHECK(sp_processor->Encode(prompt, &prompt_tokens).ok());

        start_token = absl::GetFlag(FLAGS_start_token);
        if (!start_token.empty())
        {
            prompt_tokens.insert(prompt_tokens.begin(), sp_processor->PieceToId(start_token));
        }

        stop_token = absl::GetFlag(FLAGS_stop_token);
        if (!stop_token.empty())
        {
            stop_token_id = sp_processor->PieceToId(stop_token);
        }
        stats = perf_monitor.end_phase("Prepare_Prompt");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Input Prompt Preparation");
    metrics.RecordStats("Prepare_Prompt", stats);

    // 7. Prepare Signature Runners
    tflite::SignatureRunner *prefill_runner = nullptr;
    tflite::SignatureRunner *decode_runner = nullptr;
    {
        ScopeTimer timer("Signature Runners Preparation");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Prepare_Runners");
        std::size_t effective_prefill_token_size =
            (prompt_tokens.size() > 0) ? (prompt_tokens.size() - 1) : 0;
            // std::cout << "HELLO";
        prefill_runner = GetPrefillRunner(
            interpreter.get(), effective_prefill_token_size, kv_cache, nullptr);
        // std::cout << "HELLO2";
        MINIMAL_CHECK(prefill_runner != nullptr);
        // std::cout << "HELLO1";

        decode_runner = GetDecodeRunner(interpreter.get(), kv_cache, nullptr);
        // std::cout << "HELLO4";
        MINIMAL_CHECK(decode_runner != nullptr);
        // std::cout << "HELLO3";
        
        stats = perf_monitor.end_phase("Prepare_Runners");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Signature Runner Preparation");
    metrics.RecordStats("Prepare_Runners", stats);

    {
        ScopeTimer timer("Model Analysis");
        AnalyzeDelegateExecution(interpreter.get());
    }

    // Pinning frequently used Tensors
    {
        ScopeTimer timer("Tensor Pinning");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Pin_Tensors");

        // PinFrequentTensors(interpreter.get());

        stats = perf_monitor.end_phase("Pin_Tensors");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Tensor Pinning");
    metrics.RecordStats("Pin_Tensors", stats);
    

    // 8. Access Tensors
    TfLiteTensor *prefill_input = prefill_runner->input_tensor("tokens");
    TfLiteTensor *prefill_input_pos = prefill_runner->input_tensor("input_pos");
    TfLiteTensor *decode_input = decode_runner->input_tensor("tokens");
    TfLiteTensor *decode_input_pos = decode_runner->input_tensor("input_pos");
    TfLiteTensor *kv_cache_k_0 = decode_runner->input_tensor("kv_cache_k_0");

    int max_seq_size = prefill_input->dims->data[1];
    int kv_cache_max_size = kv_cache_k_0->dims->data[1];
    
    // 9. Prefill Stage
    {
        ScopeTimer timer("Prefill Stage");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Prefill");
        int prefill_seq_size = std::min<int>(prompt_tokens.size(), max_seq_size);
        std::cout << prefill_seq_size;
        // Zero out the input tensors
        std::memset(prefill_input->data.i32, 0, prefill_input->bytes);
        std::memset(prefill_input_pos->data.i32, 0, prefill_input_pos->bytes);
        
        // Prefill uses all but the last token from the prompt
        for (int i = 0; i < prefill_seq_size - 1; ++i)
        {
            prefill_input->data.i32[i] = prompt_tokens[i];
            prefill_input_pos->data.i32[i] = i;
        }

        
        // Execute the prefill runner
        MINIMAL_CHECK(prefill_runner->Invoke() == kTfLiteOk);
        stats = perf_monitor.end_phase("Prefill");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Prefill Stage");
    metrics.RecordStats("Prefill", stats);

    // 10. Decoding Stage with separate metrics for inference and sampling
    std::cout << "\nPrompt:\n"
              << prompt << "\n\nOutput Text:\n";

    // Metrics object
    DecodingMetrics decoding_metrics;
    decoding_metrics.StartDecoding();
    PageFaultStats decode_stats;
    std::vector<RUsageRecord> rusageRecords;
    struct RUsageRecord decode_record;
    //rusage decode_start, decode_end;
    {
        // ScopeTimer timer("Decoding Stage");

        // Determine how many tokens to generate
        int max_decode_steps = (absl::GetFlag(FLAGS_max_decode_steps) == -1)
                                   ? kv_cache_max_size
                                   : absl::GetFlag(FLAGS_max_decode_steps);

        int prefill_seq_size = std::min<int>(prompt_tokens.size(), max_seq_size);
        int decode_steps = std::min<int>(max_decode_steps, kv_cache_max_size - prefill_seq_size);
        MINIMAL_CHECK(decode_steps > 0);

        int next_token = prompt_tokens[prefill_seq_size - 1];
        int next_position = prefill_seq_size - 1;

        // Decoding loop
        // Decoding loop
        for (int i = 0; i < decode_steps; ++i)
        {
            // Start time for this token
            auto token_start = std::chrono::high_resolution_clock::now();
            getrusage(RUSAGE_SELF, &decode_record.start);
            perf_monitor.start_phase("Decode_Token_" + std::to_string(i));

            // -----------------------
            // 1) Model Inference
            // -----------------------
            auto inference_start = std::chrono::high_resolution_clock::now();
            decode_input->data.i32[0] = next_token;
            decode_input_pos->data.i32[0] = next_position;
            MINIMAL_CHECK(decode_runner->Invoke() == kTfLiteOk);
            auto inference_end = std::chrono::high_resolution_clock::now();
            double inference_time_ms =
                std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

            // -----------------------
            // 2) Token Sampling
            // -----------------------
            auto sampling_start = std::chrono::high_resolution_clock::now();
            next_token = Sampler::TemperatureTopKTopPSampler(
                decode_runner->output_tensor("logits"), 0.9f, 85, 0.9f);
            auto sampling_end = std::chrono::high_resolution_clock::now();
            double sampling_time_ms =
                std::chrono::duration<double, std::milli>(sampling_end - sampling_start).count();

            next_position++;

            // Check stop token
            if (next_token == stop_token_id)
            {
                break;
            }

            // Decode the single token to text
            std::vector<int> single_token_vec = {next_token};
            std::string single_decoded_text;
            MINIMAL_CHECK(sp_processor->Decode(single_token_vec, &single_decoded_text).ok());
            std::cout << single_decoded_text << std::flush;

            // <-- 추가: 현재 Working Set Size 출력 -->
            PrintWorkingSetSize();

            // End perf recording
            decode_stats = perf_monitor.end_phase("Decode_Token_" + std::to_string(i));
            // Record metrics for this token
            decoding_metrics.RecordTimes(token_start, inference_time_ms, sampling_time_ms);
            getrusage(RUSAGE_SELF, &decode_record.end);
            metrics.RecordStats("Decode", decode_stats);
            rusageRecords.push_back(decode_record);
        }

    }

    // 11. Print decoding metrics (inference vs. sampling)
    decoding_metrics.PrintMetrics();
    // 12. Print Perf results
    metrics.PrintStats();
    // 13. Print RUsage results
    PrintRUsageRecords(rusageRecords);

    return 0;
}

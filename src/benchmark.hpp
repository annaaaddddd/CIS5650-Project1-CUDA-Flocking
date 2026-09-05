#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>

// Benchmarking / profiling harness.
//
// Everything here is configured at *runtime* so that the whole experiment
// matrix (mode x N x blockSize x visualization) can be swept by a script
// without recompiling, and so that instrumentation can be kept out of the
// runs that produce official FPS numbers.

namespace bench {

enum class Mode {
    None,    // no measurement at all
    Fps,     // wall-clock frame time only; no CUDA events, no NVTX
    Total,   // one CUDA-event timer around the whole simulation step
    Stages   // per-stage CUDA-event timers + NVTX ranges (profiling build)
};

enum class SimMode {
    Naive,
    Scattered,
    Coherent
};

struct Config {
    Mode mode = Mode::None;
    int warmup = 300;
    int frames = 5000;

    // Results are appended here, so repeated runs of the same config simply
    // add rows and the whole sweep ends up in one file.
    std::string csvPath = "benchmark_results.csv";

    // Run metadata; echoed into every CSV row so a row can never be
    // mislabelled -- the binary reports the config it actually ran.
    SimMode simMode = SimMode::Naive;
    int numBoids = 5000;
    int blockSize = 128;
    bool visualize = false;
};

extern Config config;

const char *simModeName(SimMode mode);

// Per-stage CUDA-event timing + NVTX ranges. Off for Fps/Total/None so that
// FPS runs carry no instrumentation.
bool stageTimingEnabled();

// The whole-frame GPU timer. On for both Total and Stages.
bool totalTimingEnabled();

// True once `config.frames` samples have been collected; the main loop should
// then report and exit so the sweep script can move on to the next config.
bool complete();

// Print the summary table and, if configured, append rows to the CSV.
void report();

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer();
    ~GpuTimer();

    void begin();

    // Only record the stop event. Does NOT wait for the GPU.
    void end();

    // Wait until this timer's stop event has completed.
    void synchronize();

    // Call after the stop event has completed.
    float elapsed() const;
};

// One measured quantity. Constructing a Stage registers it, so declaring one
// as a function-local static is enough to add it to the report.
struct Stage {
    explicit Stage(const char *label, bool gpuTimed = true);

    const char *label;
    bool gpuTimed;
    bool pending = false;    // timed during the frame currently in flight
    GpuTimer timer;
    std::vector<float> samples;
};

// RAII wrapper: records the start/stop events and pushes an NVTX range around
// the scope. Compiles to a pair of predictable branches when disabled.
class StageScope {
public:
    StageScope(Stage &stage, bool enabled);
    ~StageScope();

    StageScope(const StageScope &) = delete;
    StageScope &operator=(const StageScope &) = delete;

private:
    Stage *stage_;
};

#define PROFILE_STAGE(stageVar) \
    bench::StageScope _bench_scope((stageVar), bench::stageTimingEnabled())

// Called by the host around one simulation step. frameEnd performs the single
// CPU-GPU synchronization for the frame and then harvests every pending stage.
void frameBegin();
void frameEnd();

// Feed a host-measured sample (used for wall-clock frame time in Fps mode).
void submitHostSample(Stage &stage, float ms);

}  // namespace bench

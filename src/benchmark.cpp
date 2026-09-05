#include "benchmark.hpp"

// nvToolsExt.h pulls in windows.h, whose min/max macros break std::min/std::max.
#define NOMINMAX
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace bench {

Config config;

namespace {

std::vector<Stage *> &registry() {
    static std::vector<Stage *> stages;
    return stages;
}

int g_frame = 0;       // frames seen, including warm-up
int g_collected = 0;   // measured frames actually recorded
bool g_reported = false;

struct Summary {
    int samples;
    float mean;
    float median;
    float p95;
    float stddev;
    float minMs;
    float maxMs;
};

Summary summarize(const std::vector<float> &values) {
    Summary s{};
    s.samples = static_cast<int>(values.size());
    if (s.samples == 0) {
        return s;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    s.mean = static_cast<float>(sum / s.samples);

    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    const int n = s.samples;
    s.median = (n % 2 == 0)
        ? 0.5f * (sorted[n / 2 - 1] + sorted[n / 2])
        : sorted[n / 2];

    // Nearest-rank percentile: smallest value at or above the 95th percentile.
    int p95Index = static_cast<int>(std::ceil(0.95 * n)) - 1;
    p95Index = std::min(std::max(p95Index, 0), n - 1);
    s.p95 = sorted[p95Index];

    s.minMs = sorted.front();
    s.maxMs = sorted.back();

    // Sample standard deviation (Bessel-corrected).
    if (n > 1) {
        double variance = 0.0;
        for (float x : values) {
            const double diff = x - s.mean;
            variance += diff * diff;
        }
        s.stddev = static_cast<float>(std::sqrt(variance / (n - 1)));
    }

    return s;
}

void writeCsv() {
    if (config.csvPath.empty()) {
        return;
    }

    // Only write the header when creating the file, so a sweep can append.
    bool needHeader = true;
    {
        std::ifstream probe(config.csvPath);
        needHeader = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(config.csvPath, std::ios::app);
    if (!out) {
        std::cerr << "benchmark: could not open CSV file " << config.csvPath << std::endl;
        return;
    }

    if (needHeader) {
        out << "mode,n,block_size,visualize,stage,samples,"
               "mean_ms,median_ms,p95_ms,stddev_ms,min_ms,max_ms\n";
    }

    out << std::setprecision(6) << std::fixed;
    for (const Stage *stage : registry()) {
        if (stage->samples.empty()) {
            continue;
        }
        const Summary s = summarize(stage->samples);
        out << simModeName(config.simMode) << ','
            << config.numBoids << ','
            << config.blockSize << ','
            << (config.visualize ? 1 : 0) << ','
            << stage->label << ','
            << s.samples << ','
            << s.mean << ','
            << s.median << ','
            << s.p95 << ','
            << s.stddev << ','
            << s.minMs << ','
            << s.maxMs << '\n';
    }
}

}  // namespace

const char *simModeName(SimMode mode) {
    switch (mode) {
        case SimMode::Naive:     return "naive";
        case SimMode::Scattered: return "scattered";
        case SimMode::Coherent:  return "coherent";
    }
    return "unknown";
}

bool stageTimingEnabled() {
    return config.mode == Mode::Stages;
}

bool totalTimingEnabled() {
    return config.mode == Mode::Stages || config.mode == Mode::Total;
}

bool complete() {
    return config.mode != Mode::None && g_collected >= config.frames;
}

GpuTimer::GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

GpuTimer::~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void GpuTimer::begin() {
    cudaEventRecord(start);
}

void GpuTimer::end() {
    cudaEventRecord(stop);
}

void GpuTimer::synchronize() {
    cudaEventSynchronize(stop);
}

float GpuTimer::elapsed() const {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

Stage::Stage(const char *label, bool gpuTimed)
    : label(label), gpuTimed(gpuTimed) {
    registry().push_back(this);
}

StageScope::StageScope(Stage &stage, bool enabled)
    : stage_(enabled ? &stage : nullptr) {
    if (stage_) {
        nvtxRangePushA(stage_->label);
        stage_->timer.begin();
    }
}

StageScope::~StageScope() {
    if (stage_) {
        stage_->timer.end();
        stage_->pending = true;
        nvtxRangePop();
    }
}

void submitHostSample(Stage &stage, float ms) {
    if (config.mode != Mode::Fps) {
        return;
    }

    g_frame++;
    if (g_frame <= config.warmup) {
        return;
    }
    if (g_collected >= config.frames) {
        return;
    }

    stage.samples.push_back(ms);
    g_collected++;
}

// The frame's total timer. Declared here so that its stop event is the last
// one recorded, letting frameEnd get away with a single synchronization.
namespace {
Stage &totalStage() {
    static Stage stage("total");
    return stage;
}
}  // namespace

void frameBegin() {
    if (!totalTimingEnabled()) {
        return;
    }
    totalStage().timer.begin();
}

void frameEnd() {
    if (!totalTimingEnabled()) {
        return;
    }

    Stage &total = totalStage();
    total.timer.end();
    total.pending = true;

    // Every stage's stop event precedes the total's stop event in stream
    // order, so waiting on this one event makes all of them readable.
    total.timer.synchronize();

    g_frame++;
    const bool measuring = g_frame > config.warmup && g_collected < config.frames;

    for (Stage *stage : registry()) {
        if (!stage->pending) {
            continue;
        }
        if (measuring) {
            stage->samples.push_back(stage->timer.elapsed());
        }
        stage->pending = false;
    }

    if (measuring) {
        g_collected++;
    }
}

void report() {
    if (config.mode == Mode::None || g_reported) {
        return;
    }
    g_reported = true;

    std::cout << "\n=== Benchmark: " << simModeName(config.simMode)
              << "  N=" << config.numBoids
              << "  blockSize=" << config.blockSize
              << "  visualize=" << (config.visualize ? "on" : "off")
              << "  warmup=" << config.warmup
              << "  frames=" << g_collected << " ===\n";

    std::cout << std::left << std::setw(28) << "stage"
              << std::right
              << std::setw(10) << "mean"
              << std::setw(10) << "median"
              << std::setw(10) << "p95"
              << std::setw(10) << "stddev"
              << std::setw(10) << "min"
              << std::setw(10) << "max" << "   (ms)\n";

    std::cout << std::fixed << std::setprecision(4);
    for (const Stage *stage : registry()) {
        if (stage->samples.empty()) {
            continue;
        }
        const Summary s = summarize(stage->samples);
        std::cout << std::left << std::setw(28) << stage->label
                  << std::right
                  << std::setw(10) << s.mean
                  << std::setw(10) << s.median
                  << std::setw(10) << s.p95
                  << std::setw(10) << s.stddev
                  << std::setw(10) << s.minMs
                  << std::setw(10) << s.maxMs << '\n';
    }

    if (config.mode == Mode::Fps) {
        for (const Stage *stage : registry()) {
            if (stage->samples.empty()) {
                continue;
            }
            const Summary s = summarize(stage->samples);
            std::cout << "FPS (from mean frame time): " << 1000.0f / s.mean
                      << "\nFPS (from median frame time): " << 1000.0f / s.median
                      << '\n';
            break;
        }
    }

    std::cout << std::flush;
    writeCsv();
}

}  // namespace bench

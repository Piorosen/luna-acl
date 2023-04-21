/*
 * Copyright (c) 2016-2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/runtime/IScheduler.h"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/core/Window.h"
#include "src/common/cpuinfo/CpuInfo.h"
#include "src/runtime/SchedulerUtils.h"

namespace arm_compute
{
IScheduler::IScheduler()
{
    // Work out the best possible number of execution threads
    _num_threads_hint = cpuinfo::num_threads_hint();
}


std::vector<int> IScheduler::get_max_frequency() {
    int cpu_num = (int)cpu_info().get_cpu_num();
    char buffer[256];
    std::vector<int> result;
    for (int num = 0; num < cpu_num; num++) {
        sprintf(buffer, "%s%d/%s", "/sys/devices/system/cpu/cpu", num, "/cpufreq/scaling_max_freq");
        std::ifstream file(buffer, std::ios::in);
        int mem;
        if (file.is_open()) { 
            file >> mem;
        }
        result.push_back(mem);
    }
    return result;
}

std::vector<int> IScheduler::get_min_frequency() {
    int cpu_num = (int)cpu_info().get_cpu_num();
    char buffer[256];
    std::vector<int> result;
    for (int num = 0; num < cpu_num; num++) {
        sprintf(buffer, "%s%d/%s", "/sys/devices/system/cpu/cpu", num, "/cpufreq/scaling_min_freq");
        std::ifstream file(buffer, std::ios::in);
        int mem;
        if (file.is_open()) { 
            file >> mem;
        }
        result.push_back(mem);
    }
    return result;
}


std::vector<int> IScheduler::get_current_frequency() {
    int cpu_num = (int)cpu_info().get_cpu_num();
    char buffer[256];
    std::vector<int> result;
    for (int num = 0; num < cpu_num; num++) {
        sprintf(buffer, "%s%d/%s", "/sys/devices/system/cpu/cpu", num, "/cpufreq/scaling_cur_freq");
        std::ifstream file(buffer, std::ios::in);
        int mem;
        if (file.is_open()) { 
            file >> mem;
        }
        result.push_back(mem);
    }
    return result;
}

void IScheduler::get_set_frequency(int core_pin, int frequency) { 
    char buffer[256];
    sprintf(buffer, "%s%d/%s", "/sys/devices/system/cpu/cpu", core_pin, "/cpufreq/scaling_cur_freq");
    std::ofstream file(buffer, std::ios::out);
    file << frequency;
    file.close();
}

std::vector<int> IScheduler::get_available_frequency(int core_pin) { 
    std::vector<int> result;
    char buffer[256];
    sprintf(buffer, "%s%d/%s", "/sys/devices/system/cpu/cpu", core_pin, "/cpufreq/scaling_available_frequencies");
    std::ifstream file(buffer, std::ios::in);
    int mem;
    while (file.eof()) { 
        file >> mem;
        result.push_back(mem);
    }
    file.close();
    return result;
}

std::pair<std::string, int> IScheduler::get_current_kernel() const { 
    return std::make_pair(cur_kernel_name, cur_kernel_uuid);
}
void IScheduler::set_current_kernel(std::string name, int uuid) { 
    cur_kernel_name = name;
    cur_kernel_uuid = uuid;
}

std::vector<arm_compute::CPUModel> IScheduler::generate_core_thread() { 
    ARM_COMPUTE_ERROR("Feature for generate_core_thread setting is not implemented");
    return std::vector<arm_compute::CPUModel>();
}

void IScheduler::set_tuner_info(std::function<bool(const char*, int)> is_next_kernel, // next layer, max_window
                        std::function<std::vector<int>()> use_core, // core_start, core_end()
                        std::function<std::tuple<int, int, int>(int, int, int, int, int, int)> core_window, // start, end, step(idx, max_idx, max_window, start, end, step)
                        std::function<void(std::string, unsigned int)> measure) { //  void(kernel_name, measure_speed)
    this->on_tuner = true;
    this->is_next_kernel = is_next_kernel;
    this->use_core = use_core;
    this->core_window = core_window;
    this->measure = measure;
}

feature_info IScheduler::get_extract_feature() const { 
    return feature_data;
}

void IScheduler::add_extract_feature(std::string kernel_ops) {
    this->feature_data.kernel_ops.push_back(kernel_ops);
    return;
}
void IScheduler::reset_extract_feature() {
    this->feature_data = feature_info{};
    return;
}


void IScheduler::set_conv_method(int method) {
    conv_method_select = method;
}
int IScheduler::get_conv_method() const { 
    return conv_method_select;
}

void IScheduler::set_gemm_kernelOps(std::string kernelOps) { 
    this->kernel_ops = kernelOps;
}

std::string IScheduler::get_gemm_kerenlOps() const { 
    return kernel_ops;
}

std::vector<int> IScheduler::get_window_result() {
    return _window_size;
}

std::vector<std::string> IScheduler::get_convolution_kernel() { 
    auto copy = vec_get_convolution_kernel;
    vec_get_convolution_kernel.clear();
    return copy;
} 

void IScheduler::add_convolution_kernel(std::string name) { 
    vec_get_convolution_kernel.push_back(name);
}

void IScheduler::reset_window_result() {
    _window_size.clear();
}

void IScheduler::set_armnn_convolution_selection(std::function<std::pair<int, int>(int, int, int)> callback) { 
    conv_method_callback = callback;
}

void IScheduler::set_get_core_current_processing_time(std::function<void(std::vector<std::pair<int, long long>>)> callback) { 
    get_core_current_processing_time = callback;
}


CPUInfo &IScheduler::cpu_info()
{
    return CPUInfo::get();
}


void IScheduler::set_num_threads_with_affinity(unsigned int num_threads, BindFunc func)
{
    ARM_COMPUTE_UNUSED(num_threads, func);
    ARM_COMPUTE_ERROR("Feature for affinity setting is not implemented");
}

unsigned int IScheduler::num_threads_hint() const
{
    return _num_threads_hint;
}

void IScheduler::schedule_common(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
#ifndef BARE_METAL
    const Window &max_window = window;
    
    if(hints.split_dimension() == IScheduler::split_dimensions_all)
    {
        printf("if(hints.split_dimension() == IScheduler::split_dimensions_all)");
        exit(0);
    }
    else
    {
        const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
        const unsigned int get_thread = (unsigned int)(this->core_select.size());
        const unsigned int num_threads    = std::min(num_iterations, get_thread);

        if(num_iterations == 0)
        {
            return;
        }

        if(!kernel->is_parallelisable() || num_threads == 1)
        {
            std::vector<IScheduler::Workload> w(1);
            w[0] = [&max_window, &kernel, &tensors](const ThreadInfo & info) {
                    if(tensors.empty())
                    {
                        kernel->run(max_window, info);
                    }
                    else
                    {
                        kernel->run_op(tensors, max_window, info);
                    }
                };
            run_workloads(w);
        }
        else
        {
            unsigned int num_windows = 0;
            switch(hints.strategy())
            {
                case StrategyHint::STATIC:
                    num_windows = num_threads;
                    break;
                case StrategyHint::DYNAMIC:
                {
                    const unsigned int granule_threshold = (hints.threshold() <= 0) ? num_threads : static_cast<unsigned int>(hints.threshold());
                    // Make sure we don't use some windows which are too small as this might create some contention on the ThreadFeeder
                    num_windows = num_iterations > granule_threshold ? granule_threshold : num_iterations;
                    ARM_COMPUTE_ERROR("DYNAMIC");
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Unknown strategy");
            }
            // Make sure the smallest window is larger than minimim workload size
            num_windows = adjust_num_of_windows(max_window, hints.split_dimension(), num_windows, *kernel, cpu_info());
            if (this->on_tuner){ 

                _window_size.push_back(num_iterations * 100 + num_windows);
                feature_data.layer.push_back({kernel->name(), (int)(num_iterations * 100 + num_windows)});
            }
            
            const int start = max_window[hints.split_dimension()].start();
            const int end = max_window[hints.split_dimension()].end();
            const int step = max_window[hints.split_dimension()].step();
            auto core_win = this->core_window;
            auto on_tuner = this->on_tuner;

            std::vector<IScheduler::Workload> workloads(num_windows);
            for(unsigned int t = 0; t < num_windows; ++t)
            {
                //Capture 't' by copy, all the other variables by reference:
                workloads[t] = [on_tuner, core_win, t, 
                                start, end, step, num_iterations,
                                &hints, &max_window, &num_windows, &kernel, &tensors](const ThreadInfo & info)
                {
                    int w_start, w_end, w_step;
                    w_start = w_end = w_step = -1;
                    if (on_tuner) { 
                        std::tie(w_start, w_end, w_step) = core_win(t, num_windows, num_iterations, start, end, step);
                    }
                    
                    Window win = max_window.split_window(hints.split_dimension(), t, num_windows, w_start, w_end, w_step);
                    win.validate();

                    if(tensors.empty())
                    {
                        kernel->run(win, info);
                    }
                    else
                    {
                        kernel->run_op(tensors, win, info);
                    }
                };
            }
            run_workloads(workloads);
        }
    }
#else  /* !BARE_METAL */
    ARM_COMPUTE_UNUSED(kernel, hints, window, tensors);
#endif /* !BARE_METAL */
}

void IScheduler::run_tagged_workloads(std::vector<Workload> &workloads, const char *tag)
{
    ARM_COMPUTE_UNUSED(tag);
    run_workloads(workloads);
}

std::size_t IScheduler::adjust_num_of_windows(const Window &window, std::size_t split_dimension, std::size_t init_num_windows, const ICPPKernel &kernel, const CPUInfo &cpu_info)
{
    // Mitigation of the narrow split issue, which occurs when the split dimension is too small to split (hence "narrow").
    if(window.num_iterations(split_dimension) < init_num_windows )
    {
        auto recommended_split_dim = Window::DimX;
        for(std::size_t dims = Window::DimY; dims <= Window::DimW; ++dims)
        {
            if(window.num_iterations(recommended_split_dim) < window.num_iterations(dims))
            {
                recommended_split_dim = dims;
            }
        }
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("%lu dimension is not a suitable dimension to split the workload. Recommended: %lu recommended_split_dim", split_dimension,
                                                  recommended_split_dim);
    }

    for(auto t = init_num_windows; t > 0; --t) // Trying the highest number of windows ,init_num_windows, first
    {
        // Try splitting the workload into t, subject to each subworkload size <= mws.
        if((window.num_iterations(split_dimension) / kernel.get_mws(cpu_info, t)) >= t)
        {
            if(t != init_num_windows)
            {
                ARM_COMPUTE_LOG_INFO_MSG_CORE("The scheduler is using a different thread count than the one assigned by the user.");
            }
            return t;
        }
    }
    ARM_COMPUTE_LOG_INFO_MSG_CORE("The scheduler is using single thread instead of the thread count assigned by the user.");
    return 1; //  If the workload is so small that it can't be split, we should run a single thread
}

} // namespace arm_compute

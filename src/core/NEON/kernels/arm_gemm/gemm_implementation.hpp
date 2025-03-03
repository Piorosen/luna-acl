/*
 * Copyright (c) 2018-2020, 2022 Arm Limited.
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

#include "arm_gemm.hpp"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <cstdint>
#include <functional>

namespace arm_gemm {

/* Structure describing an implementation.  For each supported combination
 * of types, a static list of these structures is built up to describe the
 * implementations available.
 */
template<typename Top, typename Tret, class OutputStage = Nothing>
struct GemmImplementation {
    const GemmMethod                                                               method;
    const char *                                                                   name;
    std::function<bool(const GemmArgs &, const OutputStage &)>                     is_supported = {};
    std::function<uint64_t(const GemmArgs &, const OutputStage &)>                 cycle_estimate = {};
    std::function<GemmCommon<Top, Tret> *(const GemmArgs &, const OutputStage &)>  instantiate = {};

    bool do_is_supported(const GemmArgs &args, const OutputStage &os) const {
        if (is_supported != nullptr) {
            return is_supported(args, os);
        } else {
            return true;
        }
    }

    uint64_t do_cycle_estimate(const GemmArgs &args, const OutputStage &os) const {
        if (cycle_estimate != nullptr) {
            return cycle_estimate(args, os);
        } else {
            return 0;
        }
    }

    GemmCommon<Top, Tret> *do_instantiate(const GemmArgs &args, const OutputStage &os) const {
        return instantiate(args, os);
    }

    static GemmImplementation with_estimate(GemmMethod m, const char *n,
                       std::function<bool(const GemmArgs &, const OutputStage &)> is_supported, std::function<uint64_t(const GemmArgs &, const OutputStage &)> cycle_estimate,
                       std::function<GemmCommon<Top, Tret> *(const GemmArgs &, const OutputStage &)> instantiate) {
        GemmImplementation impl(m,n);

        impl.is_supported=is_supported;
        impl.cycle_estimate=cycle_estimate;
        impl.instantiate=instantiate;

        return impl;
    }

    GemmImplementation(const GemmImplementation &) = default;
    GemmImplementation & operator= (const GemmImplementation &) = default;

    GemmImplementation(GemmMethod m, const char * n) : method(m), name(n) {}

    GemmImplementation(GemmMethod m, const char *n,
                       std::function<bool(const GemmArgs &, const OutputStage &)> is_supported, std::function<bool(const GemmArgs &, const OutputStage &)> is_recommended,
                       std::function<GemmCommon<Top, Tret> *(const GemmArgs &, const OutputStage &)> instantiate) :
                       method(m), name(n), is_supported(is_supported),
                       cycle_estimate( [is_recommended](const GemmArgs &args, const OutputStage &os) { return (is_recommended == nullptr) ? 0 : (is_recommended(args, os) ? 0 : UINT64_MAX); } ),
                       instantiate(instantiate) {   }
};

/* Slightly different version of above for straightforward GEMMs with no
 * output stage, so the std::functions there don't have to deal with the
 * unnecessary second argument.  */
template<typename Top, typename Tret>
struct GemmImplementation<Top, Tret, Nothing> {
    const GemmMethod                                          method;
    const char *                                              name;
    std::function<bool(const GemmArgs &)>                     is_supported = {};
    std::function<uint64_t(const GemmArgs &)>                 cycle_estimate = {};
    std::function<GemmCommon<Top, Tret> *(const GemmArgs &)>  instantiate = {};

    bool do_is_supported(const GemmArgs &args, const Nothing &) const {
        if (is_supported != nullptr) {
            return is_supported(args);
        } else {
            return true;
        }
    }

    uint64_t do_cycle_estimate(const GemmArgs &args, const Nothing &) const {
        if (cycle_estimate != nullptr) {
            return cycle_estimate(args);
        } else {
            return 0;
        }
    }

    GemmCommon<Top, Tret> *do_instantiate(const GemmArgs &args, const Nothing &) const {
        return instantiate(args);
    }

    static GemmImplementation with_estimate(GemmMethod m, const char *n,
                       std::function<bool(const GemmArgs &)> is_supported, std::function<uint64_t(const GemmArgs &)> cycle_estimate,
                       std::function<GemmCommon<Top, Tret> *(const GemmArgs &)> instantiate) {
        GemmImplementation impl(m,n);

        impl.is_supported=is_supported;
        impl.cycle_estimate=cycle_estimate;
        impl.instantiate=instantiate;

        return impl;
    }

    GemmImplementation(const GemmImplementation &) = default;
    GemmImplementation & operator= (const GemmImplementation &) = default;

    GemmImplementation(GemmMethod m, const char * n) : method(m), name(n) {}

    GemmImplementation(GemmMethod m, const char *n,
                       std::function<bool(const GemmArgs &)> is_supported, std::function<bool(const GemmArgs &)> is_recommended,
                       std::function<GemmCommon<Top, Tret> *(const GemmArgs &)> instantiate) :
                       method(m), name(n), is_supported(is_supported),
                       cycle_estimate( [is_recommended](const GemmArgs &args) -> uint64_t { return (is_recommended == nullptr) ? 0 : (is_recommended(args) ? 0 : UINT64_MAX); } ),
                       instantiate(instantiate) {   }
};

/* "Main" function implemented for each valid combination of types.
 * Returns a list of GEMM implementation descriptors for processing by the
 * other functions, ended by an implementation with
 * method==GemmMethod::DEFAULT.  */
template<typename Top, typename Tret, class OutputStage = Nothing>
const GemmImplementation<Top, Tret, OutputStage> *gemm_implementation_list();

/*
 * Select a GEMM implementation for the given arguments.
 *
 * The logic here returns the method on the list which supports the
 * requested problem parameters, matches the provided filters (method and/or
 * name string match) and offers the lowest cycle estimate.  A cycle
 * estimate of '0' is treated as a special value, causing the corresponding
 * method to be selected immediately.
 *
 * If no method supports the requested parameters and passes the filters,
 * this function returns false and doesn't touch the provided pointer
 * reference.
 */
template<typename Top, typename Tret, class OutputStage>
bool find_implementation(const GemmArgs &args, const OutputStage &os, const GemmImplementation<Top, Tret, OutputStage> * &impl) {
    auto gemms = gemm_implementation_list<Top, Tret, OutputStage>();
    const GemmConfig *cfg = args._cfg;

    const GemmImplementation<Top, Tret, OutputStage> *saved_impl = nullptr;
    uint64_t best_estimate = 0;
    
    // printf("%d %d %d %d\n", args._Msize, args._Nsize, args._Ksize, args._Ksections);
    arm_compute::NEScheduler::get().get_convolution_kernel();
    
    for (const GemmImplementation<Top, Tret, OutputStage> *i = gemms; i->method != GemmMethod::DEFAULT; i++) {
        /* Skip if this implementation doesn't support these args. */
        if (!i->do_is_supported(args, os)) {
            continue;
        }

        // /* Skip if a specific method is requested and this is a different one. */
        // if (cfg && cfg->method != GemmMethod::DEFAULT && i->method != cfg->method) {
        //     continue;
        // }

        // /* Skip if a filter is to be applied and it doesn't match. */
        // if (cfg && cfg->filter != "" && !strstr(i->name, cfg->filter.c_str())) {
        //     continue;
        // }

        arm_compute::NEScheduler::get().add_extract_feature(i->name);
        arm_compute::NEScheduler::get().add_convolution_kernel(i->name);
    }


    for (const GemmImplementation<Top, Tret, OutputStage> *i = gemms; i->method != GemmMethod::DEFAULT; i++) {
        /* Skip if this implementation doesn't support these args. */
        if (!i->do_is_supported(args, os)) {
            continue;
        }

        // /* Skip if a specific method is requested and this is a different one. */
        // if (cfg && cfg->method != GemmMethod::DEFAULT && i->method != cfg->method) {
        //     continue;
        // }

        // /* Skip if a filter is to be applied and it doesn't match. */
        // if (cfg && cfg->filter != "" && !strstr(i->name, cfg->filter.c_str())) {
        //     continue;
        // }

        /* Test the cycle estimate */
        uint64_t estimate = i->do_cycle_estimate(args, os);

        // std::cout << "default kernel : " << i->name << "\n";
        if (arm_compute::NEScheduler::get().get_gemm_kerenlOps() == i->name) { 
            impl = i;
            return true;
        }

        // /* Short circuit - if the estimate is zero, return this one immediately. */
        // if (estimate==0) {
        //     impl=i;
        //     return true;
        // }
        

        /* Otherwise, remember this is our best so far if we don't yet have
         * a valid candidate, or we beat the estimate.  */
        if ((saved_impl == nullptr) || (estimate < best_estimate)) {
            saved_impl = i;
            best_estimate = estimate;
        }
    }

    /* Return whichever method gave the best estimate. */
    if (saved_impl != nullptr) {
        impl = saved_impl;
        return true;
    }

    return false;
}

template<typename Top, typename Tret, class OutputStage>
std::vector<KernelDescription> get_compatible_kernels(const GemmArgs &args, const OutputStage &os) {
    std::vector<KernelDescription> res;

    /* Find out what the default implementation in so we can set the flag accordingly later. */
    const GemmImplementation<Top, Tret, OutputStage> *default_impl;
    find_implementation(args, os, default_impl);

    auto gemms = gemm_implementation_list<Top, Tret, OutputStage>();

    for (const GemmImplementation<Top, Tret, OutputStage> *i = gemms; i->method != GemmMethod::DEFAULT; i++) {
        /* Check that this implementation supports the presented problem. */

        if (!i->do_is_supported(args, os)) {
            continue;
        }

        res.push_back(KernelDescription(i->method, i->name, i==default_impl, i->do_cycle_estimate(args, os)));
    }

    return res;
}

template<typename Top, typename Tret, class OutputStage>
bool has_opt_gemm(const GemmArgs &args, const OutputStage &os) {
    const GemmImplementation<Top, Tret, OutputStage> *impl;
    return find_implementation<Top, Tret, OutputStage>(args, os, impl);
}

template<typename Top, typename Tret, class OutputStage>
UniqueGemmCommon<Top, Tret> gemm(const GemmArgs &args, const OutputStage &os) {
    const GemmImplementation<Top, Tret, OutputStage> *impl;

    if (find_implementation<Top, Tret, OutputStage>(args, os, impl)) {
        return UniqueGemmCommon<Top, Tret>(impl->do_instantiate(args, os));
    }

    return UniqueGemmCommon<Top, Tret>(nullptr);
}

} // namespace arm_gemm

// This header is not specific to our application and you'll probably want something like this
// for any extension you're building. This includes the infrastructure needed to serialize
// descriptors that are used with the "opaque" parameter of the GPU custom call. In our example
// we'll use this parameter to pass the size of our problem.


#include "kenerl_helpers_matmul.h"


namespace brainpy_lib {

    // Descriptors
    pybind11::bytes build_matmul_descriptor(
            std::uint32_t m,
            std::uint32_t k,
            std::uint32_t n,
            std::uint32_t seed,
            float p
    ) {
        return PackDescriptor(MatMulDescriptor{m, k, n, seed, p});
    }


}  // namespace brainpy_lib


#include "ginkgo/ginkgo.hpp"
#include <iostream>

int main() {
    using mtx = gko::matrix::Dense<double>;

    auto exec = gko::version_info::get();

    std::cout<<exec.cuda_version;
    return 0;
}
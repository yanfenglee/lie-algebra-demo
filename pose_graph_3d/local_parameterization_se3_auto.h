#ifndef SOPHUS_TEST_LOCAL_PARAMETERIZATION_SE3_AUTO_HPP
#define SOPHUS_TEST_LOCAL_PARAMETERIZATION_SE3_AUTO_HPP

#include <ceres/local_parameterization.h>
#include <sophus/se3.hpp>

namespace mytest {

    using namespace Sophus;

    struct LocalParamSE3Autodiff {
        template<typename T>
        bool operator()(const T *T_raw, const T *delta_raw, T *x_plus_delta) const {
            Eigen::Map<SE3<T> const> const raw(T_raw);
            Eigen::Map<Eigen::Matrix<T,6,1> const> const delta(delta_raw);
            Eigen::Map<SE3<T>> T_plus_delta(x_plus_delta);
            T_plus_delta = raw * SE3<T>::exp(delta);
            return true;
        }
    };
}  // namespace test

#endif

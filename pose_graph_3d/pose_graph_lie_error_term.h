

#ifndef POSE_LIE_ERR_H_
#define POSE_LIE_ERR_H_

#include <Eigen/Core>
#include <ceres/sized_cost_function.h>

#include "types.h"

#include <sophus/se3.hpp>

namespace mytest
{

using namespace Sophus;

class PoseLieCostFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseLieCostFunction(const SE3d &t_ab_measured, const Eigen::Matrix<double, 6, 6> &sqrt_information)
        : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information) {}

    template <typename T>
    bool operator()(const T *const a_ptr, const T *const b_ptr,
                    T *residuals_ptr) const
    {

        Eigen::Map<Sophus::SE3<T> const> const t_a(a_ptr);
        Eigen::Map<Sophus::SE3<T> const> const t_b(b_ptr);

        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);

        residuals = (t_ab_measured_.cast<T>().inverse() * t_a.inverse() * t_b).log();

        return true;
    }

private:
    // The measurement for the position of B relative to A in the A frame.
    const SE3d t_ab_measured_;
    // The square root of the measurement information matrix.
    const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

using Mat6x6d = Eigen::Matrix<double, 6, 6>;

// 给定误差求Jr^{-1}的近似
Mat6x6d JRInv(SE3d e)
{
    Mat6x6d J;
    J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log());
    J.block(0, 3, 3, 3) = SO3d::hat(e.translation());
    J.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log());
    J = J * 0.5 + Mat6x6d::Identity();
    return J;
}

class PoseLieAnalyticCostFunction : public ceres::SizedCostFunction<6, 7, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseLieAnalyticCostFunction(const SE3d &t_ab_measured)
        : t_ab_measured_(t_ab_measured) {}

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {

        Eigen::Map<const SE3d> const t_a(parameters[0]);
        Eigen::Map<const SE3d> const t_b(parameters[1]);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> err(residuals);

        auto r = t_ab_measured_.inverse() * t_a.inverse() * t_b;
        err = r.log();

        if (jacobians != NULL)
        {
            //Mat6x6d jinv = JRInv(r);

            if (jacobians[0] != NULL)
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J1(jacobians[0]);
                J1.setZero();
                J1.block<6, 6>(0, 0) = -t_b.inverse().Adj();
            }
            if (jacobians[1] != NULL)
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J1(jacobians[1]);
                J1.setZero();
                J1.block<6, 6>(0, 0) = t_b.inverse().Adj();
            }
        }

        return true;
    }

private:
    // The measurement for the position of B relative to A in the A frame.
    const SE3d t_ab_measured_;
    // The square root of the measurement information matrix.
    //const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

} // namespace mytest

#endif // POSE_LIE_ERR_H_

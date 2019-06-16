

#ifndef POSE_LIE_ERR_H_
#define POSE_LIE_ERR_H_

#include <Eigen/Core>
#include <ceres/sized_cost_function.h>

#include "types.h"

namespace mytest {


// 给定误差求Jr^{-1}的近似
Mat6x6d JRInv( SE3d e )
{
    Mat6x6d J;
    J.block(0,0,3,3) = SO3d::hat(e.so3().log());
    J.block(0,3,3,3) = SO3d::hat(e.translation());
    J.block(3,0,3,3) = Eigen::Matrix3d::Zero(3,3);
    J.block(3,3,3,3) = SO3d::hat(e.so3().log());
    J = J*0.5 + Mat6x6d::Identity();
    return J;
}

class PoseLieCostFunction : public ceres::SizedCostFunction<6, 6, 6> {
 public:
  PoseLieCostFunction(const SE3d& t_ab_measured, const Eigen::Matrix<double, 6, 6>& sqrt_information)
      : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information) {}

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {


    Eigen::Map<SE3d const> const t_a(parameters[0]);
    Eigen::Map<SE3d const> const t_b(parameters[1]);

    Vector6d err = (t_ab_measured_.inverse() * t_a.inverse() * t_b).log();
    for (int i = 0; i < 6; ++i) {
        residuals[i] = err(i);
    }

    if (jacobians != NULL && jacobians[0] != NULL) {

        Mat6x6d J = JRInv(SE3d::exp(err));

        Mat6x6d j1 = -J * t_b.inverse().Adj();
        Mat6x6d j2 = -j1;
        
        memcpy(jacobians[0], j1.data(), 36*sizeof(double));
        memcpy(jacobians[1], j2.data(), 36*sizeof(double));
    }

    return true;
  }

 private:
    // The measurement for the position of B relative to A in the A frame.
    const SE3d t_ab_measured_;
    // The square root of the measurement information matrix.
    const Eigen::Matrix<double, 6, 6> sqrt_information_;
};
} // namespace mytest

#endif  // POSE_LIE_ERR_H_

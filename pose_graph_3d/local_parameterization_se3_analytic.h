#ifndef SOPHUS_TEST_LOCAL_PARAMETERIZATION_SE3_ANALYTIC_HPP
#define SOPHUS_TEST_LOCAL_PARAMETERIZATION_SE3_ANALYTIC_HPP

#include <ceres/local_parameterization.h>
#include <sophus/se3.hpp>

namespace mytest {

class LocalParamSE3Analytic : public ceres::LocalParameterization {
 public:
  virtual ~LocalParamSE3Analytic() {}

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {
    Eigen::Map<SE3d const> const T(T_raw);
    Eigen::Map<Vector6d const> const delta(delta_raw);
    Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = SE3d::exp(delta) * T;
    return true;
  }

  // Jacobian of SE3 plus operation for Ceres
  //
  // dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {

    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > jacobian(jacobian_raw);
    jacobian.setZero();
    jacobian.block<6,6>(0, 0).setIdentity();
    return true;
  }

  virtual int GlobalSize() const { return SE3d::num_parameters; }

  virtual int LocalSize() const { return SE3d::DoF; }
};
}  // namespace test

#endif

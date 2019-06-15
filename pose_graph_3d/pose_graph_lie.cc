

#include <iostream>
#include <fstream>
#include <string>

#include <ceres/ceres.h>

#include "common/read_g2o.h"
#include "types.h"
#include "pose_graph_lie_error_term.h"

namespace mytest {


void BuildOptimizationProblem(const VectorOfConstraints& constraints,
                              MapOfPoses* poses, ceres::Problem* problem) {
  assert(poses != NULL);
  assert(problem != NULL);
  if (constraints.empty()) {
    std::cout << "No constraints, no problem to optimize.";
    return;
  }

  ceres::LossFunction* loss_function = NULL;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  for (VectorOfConstraints::const_iterator constraints_iter =
           constraints.begin();
       constraints_iter != constraints.end(); ++constraints_iter) {
    const Constraint3d& constraint = *constraints_iter;

    MapOfPoses::iterator pose_begin_iter = poses->find(constraint.id_begin);
    CHECK (pose_begin_iter != poses->end())
        << "Pose with ID: " << constraint.id_begin << " not found.";
    MapOfPoses::iterator pose_end_iter = poses->find(constraint.id_end);
    CHECK (pose_end_iter != poses->end())
       << "Pose with ID: " << constraint.id_end << " not found.";

    const Eigen::Matrix<double, 6, 6> sqrt_information =
        constraint.information.llt().matrixL();
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function =
        new PoseLieCostFunction(constraint.t_be.se3, sqrt_information);

    problem->AddResidualBlock(cost_function, loss_function,
                              pose_begin_iter->second.se3.data(),
                              pose_end_iter->second.se3.data());

  }

}


// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
bool OutputPoses(const std::string& filename,  MapOfPoses& poses) {
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile) {
    LOG(ERROR) << "Error opening the file: " << filename;
    return false;
  }
  for (std::map<int, Pose3d, std::less<int>,
                Eigen::aligned_allocator<std::pair<const int, Pose3d> > >::iterator 
                poses_iter = poses.begin(); poses_iter != poses.end(); ++poses_iter) {
    
    std::map<int, Pose3d, std::less<int>,
                Eigen::aligned_allocator<std::pair<const int, Pose3d> > >::value_type& pair = *poses_iter;
    pair.second.FromSE3();
    outfile << pair.first << " " << pair.second.p.transpose() << " "
            << pair.second.q.x() << " " << pair.second.q.y() << " "
            << pair.second.q.z() << " " << pair.second.q.w() << '\n';
  }
  return true;
}


// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem) {
  CHECK(problem != NULL);

  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  std::cout << summary.FullReport() << '\n';

  return summary.IsSolutionUsable();
}

} // namespace mytest

int main(int argc, char** argv) {
  
  google::InitGoogleLogging(argv[0]);
  //CERES_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  std::string FLAGS_input = argv[1];
  if(FLAGS_input == "") std::cerr << "Need to specify the filename to read.";

  mytest::MapOfPoses poses;
  mytest::VectorOfConstraints constraints;

  if (!mytest::ReadG2oFile(FLAGS_input, &poses, &constraints)){

    std::cerr << "Error reading the file: " << FLAGS_input;
    return 1;
  }

  std::cout << "Number of poses: " << poses.size() << '\n';
  std::cout << "Number of constraints: " << constraints.size() << '\n';

  if (!mytest::OutputPoses("poses_original.txt", poses)) { 
      std::cerr << "Error outputting to poses_original.txt";
      return 1;
  }


  ceres::Problem problem;
  mytest::BuildOptimizationProblem(constraints, &poses, &problem);

  if (!mytest::SolveOptimizationProblem(&problem)) {
      std::cerr << "The solve was not successful, exiting." << std::endl;
  }


  if (!mytest::OutputPoses("poses_optimized.txt", poses))
     std::cerr << "Error outputting to poses_original.txt" << std::endl;

  return 0;
  }

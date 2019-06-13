

#include <iostream>
#include <fstream>
#include <string>

#include <ceres/ceres.h>
#include <sophus/so3.h>
#include <sophus/se3.h>

#include "common/read_g2o.h"
#include "types.h"

using Sophus::SE3;
using Sophus::SO3;

namespace mytest {


typedef Eigen::Matrix<double,6,6> Mat6x6d;

// 给定误差求J_R^{-1}的近似
Mat6x6d JRInv( SE3 e )
{
    Mat6x6d J;
    J.block(0,0,3,3) = SO3::hat(e.so3().log());
    J.block(0,3,3,3) = SO3::hat(e.translation());
    J.block(3,0,3,3) = Eigen::Matrix3d::Zero(3,3);
    J.block(3,3,3,3) = SO3::hat(e.so3().log());
    J = J*0.5 + Mat6x6d::Identity();
    return J;
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


  if (!mytest::OutputPoses("poses_optimized.txt", poses))
     std::cerr << "Error outputting to poses_original.txt" << std::endl;

  return 0;
}

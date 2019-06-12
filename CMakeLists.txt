cmake_minimum_required( VERSION 3.5 )
project ( posegraph )

#set( CMAKE_CXX_COMPILER "g++" )
#set( CMAKE_BUILD_TYPE "Release" )
#set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

set(CMAKE_CXX_STANDARD 14)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

############### dependencies ######################
# Eigen
#include_directories( "/usr/include/eigen3" )
find_package( Eigen3 REQUIRED )
include_directories( ${Eigen3} )
# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

# Ceres 
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

# gflag 
find_package( gflags REQUIRED )
include_directories( ${gflags_INCLUDE_DIRS} )

# set( THIRD_PARTY_LIBS 
#     ${OpenCV_LIBS}
#     ${Sophus_LIBRARIES}
#     ${Ceres_LIBS}
# )
############### dependencies ######################
include_directories(./)
add_subdirectory( pose_graph_3d )

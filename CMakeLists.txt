cmake_minimum_required( VERSION 3.5 )
project ( posegraph )

#set( CMAKE_CXX_COMPILER "g++" )
#set( CMAKE_BUILD_TYPE "Release" )
#set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

set(CMAKE_CXX_STANDARD 14)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

## set default release
if (NOT CMAKE_BUILD_TYPE)
  message("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build -DCMAKE_BUILD_TYPE=xxx, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()


############### dependencies ######################

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Eigen
#include_directories( "/usr/include/eigen3" )
find_package( Eigen3 REQUIRED )
include_directories( ${Eigen3} )

# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

if (SOPHUS_FOUND)
  message("sophus directory" === ${Sophus_INCLUDE_DIRS})
else()
  message(FATAL "sophus not found!!!")
endif()

# Ceres 
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

message("ceres directory" ${CERES_INCLUDE_DIRS})


set( THIRD_PARTY_LIBS
        ${CERES_LIBRARIES}
        ${SOPHUS_LIBRARIES}
)
############### dependencies ######################
include_directories(./)
add_subdirectory( pose_graph_3d )

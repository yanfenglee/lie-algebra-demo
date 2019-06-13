
## pose3d
add_executable(pose_graph_3d pose_graph_3d.cc)
target_link_libraries(pose_graph_3d ${CERES_LIBRARIES}  ${Sophus_LIBRARIES})


## use lie algebra
add_executable(pose_graph_lie pose_graph_lie.cc)
target_link_libraries(pose_graph_lie ${CERES_LIBRARIES} ${Sophus_LIBRARIES})

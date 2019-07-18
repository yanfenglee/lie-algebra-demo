
## pose3d
add_executable(pose_graph_3d pose_graph_3d.cc)
target_link_libraries(pose_graph_3d ${THIRD_PARTY_LIBS})


## use lie algebra
add_executable(pose_graph_lie pose_graph_lie.cc)
target_link_libraries(pose_graph_lie ${THIRD_PARTY_LIBS})


## use lie algebra
add_executable(pose_graph_lie_auto pose_graph_lie_autodiff.cc)
target_link_libraries(pose_graph_lie_auto ${THIRD_PARTY_LIBS})


## use lie algebra
add_executable(pose_graph_lie_analytic pose_graph_lie_analytic.cc)
target_link_libraries(pose_graph_lie_analytic ${THIRD_PARTY_LIBS})

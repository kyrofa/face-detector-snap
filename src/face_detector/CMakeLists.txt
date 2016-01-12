cmake_minimum_required(VERSION 2.8.3)
project(face_detector)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  sensor_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_definitions(-std=c++11)

add_executable(face_detector_node src/main.cpp)
target_link_libraries(face_detector_node ${catkin_LIBRARIES})

install(TARGETS face_detector_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES launch/face_detector.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY classifiers
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

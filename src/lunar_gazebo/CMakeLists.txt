cmake_minimum_required(VERSION 3.8)
project(lunar_gazebo)

find_package(ament_cmake REQUIRED)

install(DIRECTORY worlds models launch scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

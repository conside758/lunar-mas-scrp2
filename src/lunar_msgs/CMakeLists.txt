cmake_minimum_required(VERSION 3.8)
project(lunar_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(action_msgs REQUIRED)

set(msg_files
  "msg/ResourceZone.msg"
  "msg/ResourceArray.msg"
  "msg/ResourceDetection.msg"
  "msg/RobotStatus.msg"
  "msg/Task.msg"
)

set(action_files
  "action/Survey.action"
  "action/Excavate.action"
  "action/Transport.action"
)

set(srv_files
  "srv/DetectResources.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${action_files}
  ${srv_files}
  DEPENDENCIES geometry_msgs builtin_interfaces action_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

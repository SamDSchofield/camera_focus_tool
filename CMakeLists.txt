cmake_minimum_required(VERSION 2.8.3)
project(camera_focus_tool)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  rospy
)

#catkin_python_setup()


## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
   cfg/FocusTool.cfg
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES camera_focus_tool
#  CATKIN_DEPENDS dynamic_reconfigure rospy
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

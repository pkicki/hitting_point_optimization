cmake_minimum_required(VERSION 3.0.2)
project(hitting_point_optimization)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -fPIC" )

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(NLopt REQUIRED)
find_package(pinocchio REQUIRED)

find_package(PythonLibs 3.8 REQUIRED)
find_package(Python REQUIRED)
find_package(Boost COMPONENTS python38 system REQUIRED)
#find_package(Boost COMPONENTS system REQUIRED)

set(CMAKE_SHARED_MODULE_PREFIX "")

include_directories(
        include
        /home/piotr/eigen/Eigen
)

add_library(utils STATIC src/utils.cpp)
target_link_libraries(utils
        Eigen3::Eigen
        pinocchio::pinocchio
        )

add_library(hpo_opt_new MODULE src/hpo_opt_new.cpp)
target_link_libraries(hpo_opt_new ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}
        NLopt::nlopt
        Eigen3::Eigen
        Clp
        CoinUtils
        pinocchio::pinocchio
        utils
        )
target_include_directories(hpo_opt_new PRIVATE ${PYTHON_INCLUDE_DIRS})
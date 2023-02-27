cmake_minimum_required(VERSION 3.0)

project(retinaface)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

find_package(tvm REQUIRED)

set(SRC
  retinaface.cpp
  main.cpp
)

set(DLPACK_HEADERS
  "/home/vagrant/opt/tvm/3rdparty/dlpack/include"
)

set(DMLC_HEADERS
  "/home/vagrant/opt/tvm/3rdparty/dmlc-core/include"
)

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} PRIVATE ${DLPACK_HEADERS} ${DMLC_HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} -ltvm_runtime)

set(EXECUTABLE_OUTPUT_PATH
  "/home/vagrant/work/RetinaFace_TVM/cpp/build"
)


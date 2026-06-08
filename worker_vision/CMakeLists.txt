cmake_minimum_required(VERSION 3.16)

project(VisionAssist LANGUAGES CXX)

set(OpenCV_DIR "$ENV{HOME}/opencv-4.10.0/build") 

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)


include_directories(include)

add_executable(main)

target_sources(main
    PRIVATE
        src/imu.cpp
        src/OpticalFlow.cpp
        src/Yolo.cpp
        src/main.cpp
)

target_compile_features(main
    PRIVATE cxx_std_17
)

target_compile_options(main
    PRIVATE
        -g
        -Wall
        -O3
)

target_include_directories(main
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(main
    PRIVATE
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        gpiod
)

cmake_minimum_required(VERSION 3.16)

project(VisionAssist LANGUAGES CXX)


find_package(OpenCV REQUIRED)

add_executable(main)

target_sources(main
    PRIVATE
        #src/OpticalFlow.cpp
        src/yolo.cpp
)

target_compile_features(main
    PRIVATE cxx_std_17
)

target_compile_options(main
    PRIVATE
        -g
        -Wall
)

target_include_directories(main
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(main
    PRIVATE
        ${OpenCV_LIBS}
)

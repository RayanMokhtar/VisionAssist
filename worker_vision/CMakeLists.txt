cmake_minimum_required(VERSION 3.16)

project(VisionAssist LANGUAGES CXX)

set(OpenCV_DIR "/home/opencv/build") 

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# find_package(PkgConfig REQUIRED) #pour client mqtt
# pkg_check_modules(PAHO REQUIRED paho-mqttpp3)


include_directories(include)

add_executable(main)

target_sources(main
    PRIVATE
        src/OpticalFlow.cpp
        src/Yolo.cpp
        src/main.cpp
        src/mqtt.cpp #modif mqtt
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
        /usr/include  #mqtt
)

target_link_libraries(main
    PRIVATE
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        -lpaho-mqttpp3 
        -lpaho-mqtt3as  
        -lpaho-mqtt3c   
)
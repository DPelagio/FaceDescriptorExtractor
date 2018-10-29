cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(face_descr)


include(/Users/pelagio/Desktop/dlib-19.16/dlib/cmake)

set(SOURCE  FaceDescriptorExtractor.cpp)
set(HEADERS FaceDescriptorExtractor.h)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})
else()
   message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
endif()


include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(face_descr main.cpp ${SOURCE} ${HEADERS})
target_link_libraries( face_descr ${OpenCV_LIBS} dlib::dlib)


# To compile this program all you need to do is ask cmake.  You would type
# these commands from within the directory containing this CMakeLists.txt
# file:
#   mkdir build
#   cd build
#   cmake ..
#   cmake --build . --config Release
#
# The cmake .. command looks in the parent folder for a file named
# CMakeLists.txt, reads it, and sets up everything needed to build program.
# Also, note that CMake can generate Visual Studio or XCode project files.  So
# if instead you had written:
#   cd build
#   cmake .. -G Xcode
#
# You would be able to open the resulting Xcode project and compile and edit
# the example programs within the Xcode IDE.  CMake can generate a lot of
# different types of IDE projects.  Run the cmake -h command to see a list of
# arguments to -G to see what kinds of projects cmake can generate for you.  It
# probably includes your favorite IDE in the list.









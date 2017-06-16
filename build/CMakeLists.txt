CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT(ncorr_library)

# Only tested for g++ on Ubuntu 12.04. This assumes all required libraries have been 
# installed, so directories to dependent libraries and their headers are not explicitly 
# included, since the install directories are searched automatically by g++.

# Set files
SET(ncorr_src ../src/ncorr.cpp ../src/Strain2D.cpp ../src/Disp2D.cpp ../src/Data2D.cpp ../src/ROI2D.cpp ../src/Image2D.cpp ../src/Array2D.cpp)
SET(ncorr_h ../include/ncorr.h ../include/Strain2D.h ../include/Disp2D.h ../include/Data2D.h ../include/ROI2D.h ../include/Image2D.h ../include/Array2D.h)

# Set include directory
INCLUDE_DIRECTORIES(../include)

# Set output for library
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)

# Add library
ADD_LIBRARY(ncorr STATIC ${ncorr_src})

# Set C++11 support
set_property(TARGET ncorr PROPERTY CXX_STANDARD 11)
set_property(TARGET ncorr PROPERTY CXX_STANDARD_REQUIRED ON)

# Set -03 optimization
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-O3" COMPILER_SUPPORTS_O3)
if (COMPILER_SUPPORTS_O3)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Disable debugging
ADD_DEFINITIONS(-DNDEBUG)

# Install library
INSTALL(TARGETS ncorr DESTINATION lib)
INSTALL(FILES ${ncorr_h} DESTINATION include)

CMAKE_MINIMUM_REQUIRED(VERSION 3.2.3)

#-------------- customized zone --------------#
SET(CMAKE_CXX_COMPILER "g++")
OPTION(ON_DOCKER "Build on docker image" OFF)
OPTION(ENABLE_DEBUGGING "Enable debugging" ON)
OPTION(ENABLE_EXAMPLE "Enable examples" ON)
OPTION(ENABLE_OPENMP "Enable OPENMP threading" ON)
OPTION(ENABLE_TIMING "Enable timing" OFF)
OPTION(ENABLE_REPORT "Enable verbose report" OFF)
#-------------- instruction ------------------#

#--------------------------------------------#
if(ON_DOCKER)
	set(GTEST_HINTS /usr/local)
else()
	message("apply customized paths")
	set(GTEST_HINTS /home/jiwang/package/googletest)
endif()
#---------------------------------------------#
PROJECT(bifet CXX)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED YES)

# compile flags under defferent options
IF(ENABLE_DEBUGGING AND ENABLE_TIMING)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pedantic -fPIC -Wno-deprecated")
ELSEIF(ENABLE_TIMING)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pedantic -fPIC -Wno-deprecated -DNDEBUG")
ELSEIF(ENABLE_DEBUGGING)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pedantic -fPIC -Wno-deprecated -DNTIMING")
ELSE()
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pedantic -fPIC -Wno-deprecated -DNTIMING -DNDEBUG")
ENDIF()

IF (ENABLE_REPORT)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVERBOSE")
ENDIF()

SET(BIFET_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include)

FIND_PACKAGE(deal.II 8.5.0 REQUIRED
	HINTS 
	${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# openmp support
# if FindOpenMP fails, try add -fopenmp to CMAKE_CXX_FLAGS above
IF(ENABLE_OPENMP)
include(FindOpenMP)
IF(OPENMP_FOUND)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ELSE()
	MESSAGE(FATAL_ERROR "openmp unsupported")
ENDIF()
ENDIF()

SET(SOURCES
	${CMAKE_CURRENT_LIST_DIR}/source/frame/frame.cc
	${CMAKE_CURRENT_LIST_DIR}/source/frame/dgframe.cc
	${CMAKE_CURRENT_LIST_DIR}/source/frame/frame_spatial.cc
	${CMAKE_CURRENT_LIST_DIR}/source/frame/frame_spectral.cc
	${CMAKE_CURRENT_LIST_DIR}/source/frame/dgframe_spatial.cc
	${CMAKE_CURRENT_LIST_DIR}/source/frame/dgframe_spectral.cc
	${CMAKE_CURRENT_LIST_DIR}/source/frame/frame_freespectral.cc
	${CMAKE_CURRENT_LIST_DIR}/source/frame/frame_freespatial.cc
	${CMAKE_CURRENT_LIST_DIR}/source/simbox/simbox.cc
	${CMAKE_CURRENT_LIST_DIR}/source/solution/solution.cc
	${CMAKE_CURRENT_LIST_DIR}/source/system/system.cc
	${CMAKE_CURRENT_LIST_DIR}/source/propagator/propagator.cc
	${CMAKE_CURRENT_LIST_DIR}/source/diffusion/diffusion.cc
	${CMAKE_CURRENT_LIST_DIR}/source/advection/advection.cc
	${CMAKE_CURRENT_LIST_DIR}/source/source/source.cc
	${CMAKE_CURRENT_LIST_DIR}/source/growth/growth.cc
)

IF(ENABLE_TIMING)
	ADD_EXECUTABLE(bifet ${CMAKE_CURRENT_LIST_DIR}/source/main/bifet_profile.cc ${SOURCES})
ELSE()
	ADD_EXECUTABLE(bifet ${CMAKE_CURRENT_LIST_DIR}/source/main/bifet.cc ${SOURCES})
ENDIF()
INCLUDE_DIRECTORIES(${BIFET_INCLUDE})

# find GoogleTest
# GoogleTest package must be assembled INTO testing modules manually
# you can either install GoogleTest and cp src dir into install path
# or just download GoogleTest
IF(ENABLE_DEBUGGING)
	FIND_PATH(GTEST_INC_DIR 
		NAMES gtest/gtest.h
  		HINTS
		${GTEST_HINTS}/include
		${GTEST_HINTS}/googletest/include
	)
	IF(NOT GTEST_INC_DIR)
  		MESSAGE(FATAL_ERROR ${GTEST_INC_DIR})
	ENDIF()
	FIND_PATH(GTEST_SOURCE_DIR gtest-all.cc
  		${GTEST_HINTS}/src
  		${GTEST_HINTS}/googletest/src
	)
	IF(NOT GTEST_SOURCE_DIR)
  		MESSAGE(FATAL_ERROR ${GTEST_SOURCE_DIR})
	ENDIF()
	SET(GTEST_LIB_SOURCES ${GTEST_SOURCE_DIR}/gtest-all.cc)
	SET(GTEST_MAIN_SOURCES ${GTEST_SOURCE_DIR}/gtest_main.cc)
	SET(GTEST_INCLUDE_DIRS ${GTEST_INC_DIR} ${GTEST_SOURCE_DIR} ${GTEST_SOURCE_DIR}/..)
	
	ENABLE_TESTING()
	ADD_SUBDIRECTORY(tests)
ENDIF()

IF (ENABLE_EXAMPLE)
	FIND_PATH(GTEST_INC_DIR 
		NAMES gtest/gtest.h
  		HINTS
		${GTEST_HINTS}/include
		${GTEST_HINTS}/googletest/include
	)
	IF(NOT GTEST_INC_DIR)
  		MESSAGE(FATAL_ERROR ${GTEST_INC_DIR})
	ENDIF()
	FIND_PATH(GTEST_SOURCE_DIR gtest-all.cc
  		${GTEST_HINTS}/src
  		${GTEST_HINTS}/googletest/src
	)
	IF(NOT GTEST_SOURCE_DIR)
  		MESSAGE(FATAL_ERROR ${GTEST_SOURCE_DIR})
	ENDIF()
	SET(GTEST_LIB_SOURCES ${GTEST_SOURCE_DIR}/gtest-all.cc)
	SET(GTEST_MAIN_SOURCES ${GTEST_SOURCE_DIR}/gtest_main.cc)
	SET(GTEST_INCLUDE_DIRS ${GTEST_INC_DIR} ${GTEST_SOURCE_DIR} ${GTEST_SOURCE_DIR}/..)
	
	ENABLE_TESTING()
	ADD_SUBDIRECTORY(examples)
ENDIF()

DEAL_II_SETUP_TARGET(bifet)

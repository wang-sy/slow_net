# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/wangsy/文档/Tools/clion-2020.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/wangsy/文档/Tools/clion-2020.3/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wangsy/Code/slow_net

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wangsy/Code/slow_net/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/slow_net.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/slow_net.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/slow_net.dir/flags.make

CMakeFiles/slow_net.dir/main.cpp.o: CMakeFiles/slow_net.dir/flags.make
CMakeFiles/slow_net.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangsy/Code/slow_net/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/slow_net.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slow_net.dir/main.cpp.o -c /home/wangsy/Code/slow_net/main.cpp

CMakeFiles/slow_net.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slow_net.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangsy/Code/slow_net/main.cpp > CMakeFiles/slow_net.dir/main.cpp.i

CMakeFiles/slow_net.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slow_net.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangsy/Code/slow_net/main.cpp -o CMakeFiles/slow_net.dir/main.cpp.s

CMakeFiles/slow_net.dir/src/base/neuron_based.cc.o: CMakeFiles/slow_net.dir/flags.make
CMakeFiles/slow_net.dir/src/base/neuron_based.cc.o: ../src/base/neuron_based.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangsy/Code/slow_net/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/slow_net.dir/src/base/neuron_based.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slow_net.dir/src/base/neuron_based.cc.o -c /home/wangsy/Code/slow_net/src/base/neuron_based.cc

CMakeFiles/slow_net.dir/src/base/neuron_based.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slow_net.dir/src/base/neuron_based.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangsy/Code/slow_net/src/base/neuron_based.cc > CMakeFiles/slow_net.dir/src/base/neuron_based.cc.i

CMakeFiles/slow_net.dir/src/base/neuron_based.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slow_net.dir/src/base/neuron_based.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangsy/Code/slow_net/src/base/neuron_based.cc -o CMakeFiles/slow_net.dir/src/base/neuron_based.cc.s

CMakeFiles/slow_net.dir/src/default_neuron.cc.o: CMakeFiles/slow_net.dir/flags.make
CMakeFiles/slow_net.dir/src/default_neuron.cc.o: ../src/default_neuron.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangsy/Code/slow_net/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/slow_net.dir/src/default_neuron.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slow_net.dir/src/default_neuron.cc.o -c /home/wangsy/Code/slow_net/src/default_neuron.cc

CMakeFiles/slow_net.dir/src/default_neuron.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slow_net.dir/src/default_neuron.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangsy/Code/slow_net/src/default_neuron.cc > CMakeFiles/slow_net.dir/src/default_neuron.cc.i

CMakeFiles/slow_net.dir/src/default_neuron.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slow_net.dir/src/default_neuron.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangsy/Code/slow_net/src/default_neuron.cc -o CMakeFiles/slow_net.dir/src/default_neuron.cc.s

CMakeFiles/slow_net.dir/test/default_neuron_test.cc.o: CMakeFiles/slow_net.dir/flags.make
CMakeFiles/slow_net.dir/test/default_neuron_test.cc.o: ../test/default_neuron_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangsy/Code/slow_net/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/slow_net.dir/test/default_neuron_test.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slow_net.dir/test/default_neuron_test.cc.o -c /home/wangsy/Code/slow_net/test/default_neuron_test.cc

CMakeFiles/slow_net.dir/test/default_neuron_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slow_net.dir/test/default_neuron_test.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangsy/Code/slow_net/test/default_neuron_test.cc > CMakeFiles/slow_net.dir/test/default_neuron_test.cc.i

CMakeFiles/slow_net.dir/test/default_neuron_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slow_net.dir/test/default_neuron_test.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangsy/Code/slow_net/test/default_neuron_test.cc -o CMakeFiles/slow_net.dir/test/default_neuron_test.cc.s

CMakeFiles/slow_net.dir/src/base/active_functions.cc.o: CMakeFiles/slow_net.dir/flags.make
CMakeFiles/slow_net.dir/src/base/active_functions.cc.o: ../src/base/active_functions.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangsy/Code/slow_net/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/slow_net.dir/src/base/active_functions.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slow_net.dir/src/base/active_functions.cc.o -c /home/wangsy/Code/slow_net/src/base/active_functions.cc

CMakeFiles/slow_net.dir/src/base/active_functions.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slow_net.dir/src/base/active_functions.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangsy/Code/slow_net/src/base/active_functions.cc > CMakeFiles/slow_net.dir/src/base/active_functions.cc.i

CMakeFiles/slow_net.dir/src/base/active_functions.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slow_net.dir/src/base/active_functions.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangsy/Code/slow_net/src/base/active_functions.cc -o CMakeFiles/slow_net.dir/src/base/active_functions.cc.s

# Object files for target slow_net
slow_net_OBJECTS = \
"CMakeFiles/slow_net.dir/main.cpp.o" \
"CMakeFiles/slow_net.dir/src/base/neuron_based.cc.o" \
"CMakeFiles/slow_net.dir/src/default_neuron.cc.o" \
"CMakeFiles/slow_net.dir/test/default_neuron_test.cc.o" \
"CMakeFiles/slow_net.dir/src/base/active_functions.cc.o"

# External object files for target slow_net
slow_net_EXTERNAL_OBJECTS =

slow_net: CMakeFiles/slow_net.dir/main.cpp.o
slow_net: CMakeFiles/slow_net.dir/src/base/neuron_based.cc.o
slow_net: CMakeFiles/slow_net.dir/src/default_neuron.cc.o
slow_net: CMakeFiles/slow_net.dir/test/default_neuron_test.cc.o
slow_net: CMakeFiles/slow_net.dir/src/base/active_functions.cc.o
slow_net: CMakeFiles/slow_net.dir/build.make
slow_net: CMakeFiles/slow_net.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wangsy/Code/slow_net/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable slow_net"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/slow_net.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/slow_net.dir/build: slow_net

.PHONY : CMakeFiles/slow_net.dir/build

CMakeFiles/slow_net.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/slow_net.dir/cmake_clean.cmake
.PHONY : CMakeFiles/slow_net.dir/clean

CMakeFiles/slow_net.dir/depend:
	cd /home/wangsy/Code/slow_net/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wangsy/Code/slow_net /home/wangsy/Code/slow_net /home/wangsy/Code/slow_net/cmake-build-debug /home/wangsy/Code/slow_net/cmake-build-debug /home/wangsy/Code/slow_net/cmake-build-debug/CMakeFiles/slow_net.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/slow_net.dir/depend

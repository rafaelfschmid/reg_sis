# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg

# Include any dependencies generated for this target.
include CMakeFiles/reg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/reg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reg.dir/flags.make

CMakeFiles/reg.dir/reg.c.o: CMakeFiles/reg.dir/flags.make
CMakeFiles/reg.dir/reg.c.o: reg.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/reg.dir/reg.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/reg.dir/reg.c.o   -c /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/reg.c

CMakeFiles/reg.dir/reg.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/reg.dir/reg.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/reg.c > CMakeFiles/reg.dir/reg.c.i

CMakeFiles/reg.dir/reg.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/reg.dir/reg.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/reg.c -o CMakeFiles/reg.dir/reg.c.s

CMakeFiles/reg.dir/reg.c.o.requires:

.PHONY : CMakeFiles/reg.dir/reg.c.o.requires

CMakeFiles/reg.dir/reg.c.o.provides: CMakeFiles/reg.dir/reg.c.o.requires
	$(MAKE) -f CMakeFiles/reg.dir/build.make CMakeFiles/reg.dir/reg.c.o.provides.build
.PHONY : CMakeFiles/reg.dir/reg.c.o.provides

CMakeFiles/reg.dir/reg.c.o.provides.build: CMakeFiles/reg.dir/reg.c.o


CMakeFiles/reg.dir/semblance.c.o: CMakeFiles/reg.dir/flags.make
CMakeFiles/reg.dir/semblance.c.o: semblance.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/reg.dir/semblance.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/reg.dir/semblance.c.o   -c /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/semblance.c

CMakeFiles/reg.dir/semblance.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/reg.dir/semblance.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/semblance.c > CMakeFiles/reg.dir/semblance.c.i

CMakeFiles/reg.dir/semblance.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/reg.dir/semblance.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/semblance.c -o CMakeFiles/reg.dir/semblance.c.s

CMakeFiles/reg.dir/semblance.c.o.requires:

.PHONY : CMakeFiles/reg.dir/semblance.c.o.requires

CMakeFiles/reg.dir/semblance.c.o.provides: CMakeFiles/reg.dir/semblance.c.o.requires
	$(MAKE) -f CMakeFiles/reg.dir/build.make CMakeFiles/reg.dir/semblance.c.o.provides.build
.PHONY : CMakeFiles/reg.dir/semblance.c.o.provides

CMakeFiles/reg.dir/semblance.c.o.provides.build: CMakeFiles/reg.dir/semblance.c.o


CMakeFiles/reg.dir/su.c.o: CMakeFiles/reg.dir/flags.make
CMakeFiles/reg.dir/su.c.o: su.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/reg.dir/su.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/reg.dir/su.c.o   -c /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/su.c

CMakeFiles/reg.dir/su.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/reg.dir/su.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/su.c > CMakeFiles/reg.dir/su.c.i

CMakeFiles/reg.dir/su.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/reg.dir/su.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/su.c -o CMakeFiles/reg.dir/su.c.s

CMakeFiles/reg.dir/su.c.o.requires:

.PHONY : CMakeFiles/reg.dir/su.c.o.requires

CMakeFiles/reg.dir/su.c.o.provides: CMakeFiles/reg.dir/su.c.o.requires
	$(MAKE) -f CMakeFiles/reg.dir/build.make CMakeFiles/reg.dir/su.c.o.provides.build
.PHONY : CMakeFiles/reg.dir/su.c.o.provides

CMakeFiles/reg.dir/su.c.o.provides.build: CMakeFiles/reg.dir/su.c.o


# Object files for target reg
reg_OBJECTS = \
"CMakeFiles/reg.dir/reg.c.o" \
"CMakeFiles/reg.dir/semblance.c.o" \
"CMakeFiles/reg.dir/su.c.o"

# External object files for target reg
reg_EXTERNAL_OBJECTS =

reg: CMakeFiles/reg.dir/reg.c.o
reg: CMakeFiles/reg.dir/semblance.c.o
reg: CMakeFiles/reg.dir/su.c.o
reg: CMakeFiles/reg.dir/build.make
reg: CMakeFiles/reg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable reg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reg.dir/build: reg

.PHONY : CMakeFiles/reg.dir/build

CMakeFiles/reg.dir/requires: CMakeFiles/reg.dir/reg.c.o.requires
CMakeFiles/reg.dir/requires: CMakeFiles/reg.dir/semblance.c.o.requires
CMakeFiles/reg.dir/requires: CMakeFiles/reg.dir/su.c.o.requires

.PHONY : CMakeFiles/reg.dir/requires

CMakeFiles/reg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reg.dir/clean

CMakeFiles/reg.dir/depend:
	cd /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg /home/mau/Documents/Topicos_em_Ling_Prog/topicos-em-linguagem-de-programacao/Trabalho/reg/CMakeFiles/reg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reg.dir/depend


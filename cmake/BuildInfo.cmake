# Build stamp: the one place the version and the commit of a build are decided and checked.
#
#   version  project(singlet VERSION x.y.z) in the root CMakeLists.txt is the only source. It reaches the compiler as
#            SINGLET_VERSION_STRING (include/singlet/version.h has no default). pyproject.toml must agree.
#   commit   -DSINGLET_GIT_COMMIT=<sha> (builds from a tree without .git, e.g. `git archive`), else `git rev-parse HEAD`.
#            It reaches the compiler as SINGLET_GIT_COMMIT. The SINGLET_GIT_SHA environment variable only overrides it at run time.
#   tag      -DSINGLET_EXPECT_VERSION=<x.y.z or vx.y.z> names the release tag being built; configuring fails when it differs
#            from the project version.
#
# Configuring fails when the version is empty or malformed, when it disagrees with the tag or with pyproject.toml, or when the
# singlet binary (SINGLET_BUILD_PIPELINE=ON) would carry no commit. Test-only builds without a commit are stamped "unknown".

set(SINGLET_GIT_COMMIT "" CACHE STRING "git commit compiled into the build (default: git rev-parse HEAD of the source tree)")
set(SINGLET_EXPECT_VERSION "" CACHE STRING "release tag being built (x.y.z or vx.y.z); configuring fails when the project version differs")

# version
if(NOT PROJECT_VERSION)
    message(FATAL_ERROR "singlet: project(singlet VERSION ...) gives no version")
endif()
if(NOT PROJECT_VERSION MATCHES "^[0-9]+\.[0-9]+\.[0-9]+$")
    message(FATAL_ERROR "singlet: project version '${PROJECT_VERSION}' is not x.y.z")
endif()
if(SINGLET_EXPECT_VERSION)
    string(REGEX REPLACE "^v" "" _singlet_expect "${SINGLET_EXPECT_VERSION}")
    if(NOT _singlet_expect STREQUAL PROJECT_VERSION)
        message(FATAL_ERROR "singlet: the project version is ${PROJECT_VERSION} but the tag being built is ${SINGLET_EXPECT_VERSION}: "
                            "bump project(singlet VERSION ...) and pyproject.toml before tagging")
    endif()
endif()
set(_singlet_pyproject "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")
if(EXISTS "${_singlet_pyproject}")
    file(STRINGS "${_singlet_pyproject}" _singlet_pyver REGEX "^version[ \t]*=" LIMIT_COUNT 1)
    string(REGEX REPLACE "^version[ \t]*=[ \t]*\"([^\"]*)\".*$" "\\1" _singlet_pyver "${_singlet_pyver}")
    if(NOT _singlet_pyver STREQUAL PROJECT_VERSION)
        message(FATAL_ERROR "singlet: pyproject.toml version '${_singlet_pyver}' differs from the project version ${PROJECT_VERSION}")
    endif()
endif()

# commit
set(_singlet_commit "${SINGLET_GIT_COMMIT}")
if(NOT _singlet_commit)
    execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse HEAD
                    OUTPUT_VARIABLE _singlet_commit OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(_singlet_commit STREQUAL "unknown")
    set(_singlet_commit "")
endif()
if(_singlet_commit AND NOT _singlet_commit MATCHES "^[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]+$")
    message(FATAL_ERROR "singlet: SINGLET_GIT_COMMIT '${_singlet_commit}' is not a git commit id")
endif()
if(NOT _singlet_commit)
    if(SINGLET_BUILD_PIPELINE)
        message(FATAL_ERROR "singlet: no git commit for the build: pass -DSINGLET_GIT_COMMIT=<sha> (the source tree has no .git)")
    endif()
    set(_singlet_commit "unknown")
endif()

add_compile_definitions(SINGLET_VERSION_STRING="${PROJECT_VERSION}" SINGLET_GIT_COMMIT="${_singlet_commit}")
message(STATUS "singlet ${PROJECT_VERSION} commit ${_singlet_commit}")

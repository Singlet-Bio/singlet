@echo off
rem Builds a static zstd for the Windows wheel build (called by wheels.yml through
rem CIBW_BEFORE_ALL_WINDOWS). Output: C:\zstd\include\*.h and C:\zstd\lib\zstd.lib,
rem which setup.py links as "zstd" (the CL and LINK variables in wheels.yml point there).
setlocal
set ZSTD_VER=1.5.6
curl -fsSL -o zstd.tar.gz https://github.com/facebook/zstd/releases/download/v%ZSTD_VER%/zstd-%ZSTD_VER%.tar.gz || exit /b 1
tar -xzf zstd.tar.gz || exit /b 1
cmake -S zstd-%ZSTD_VER%\build\cmake -B zstd-build -A x64 -DZSTD_BUILD_SHARED=OFF -DZSTD_BUILD_STATIC=ON -DZSTD_BUILD_PROGRAMS=OFF -DZSTD_BUILD_TESTS=OFF -DZSTD_LEGACY_SUPPORT=OFF || exit /b 1
cmake --build zstd-build --config Release || exit /b 1
mkdir C:\zstd\include C:\zstd\lib 2>nul
copy /Y zstd-%ZSTD_VER%\lib\zstd.h C:\zstd\include\ || exit /b 1
copy /Y zstd-%ZSTD_VER%\lib\zdict.h C:\zstd\include\ || exit /b 1
copy /Y zstd-%ZSTD_VER%\lib\zstd_errors.h C:\zstd\include\ || exit /b 1
copy /Y zstd-build\lib\Release\zstd_static.lib C:\zstd\lib\zstd.lib || exit /b 1

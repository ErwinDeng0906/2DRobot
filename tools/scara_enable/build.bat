@echo off
setlocal EnableExtensions
REM 编译 scara_enable.exe → 部署到 SNRobotLab
set SRC=%~dp0scara_enable.c
set OUT=%~dp0scara_enable.exe
set SDKDIR=D:\SNRobotLab
if defined SNROBOTLAB_DIR set SDKDIR=%SNROBOTLAB_DIR%

set "GCC="
if exist "D:\MSYS2\ucrt64\bin\gcc.exe" set "GCC=D:\MSYS2\ucrt64\bin\gcc.exe"
if not defined GCC if exist "D:\MSYS2\mingw64\bin\gcc.exe" set "GCC=D:\MSYS2\mingw64\bin\gcc.exe"
if not defined GCC (
  where gcc >nul 2>&1
  if not errorlevel 1 set "GCC=gcc"
)
if not defined GCC (
  echo ERR: 找不到 gcc。请确认 D:\MSYS2\ucrt64\bin\gcc.exe 存在
  exit /b 1
)

for %%I in ("%GCC%") do set "GCCBIN=%%~dpI"
set "PATH=%GCCBIN%;%PATH%"

echo Using: %GCC%
"%GCC%" -O2 -Wall -Wextra -o "%OUT%" "%SRC%" -lkernel32
if errorlevel 1 exit /b 1
echo Built: %OUT%

if not exist "%SDKDIR%\RobotSDK.dll" (
  echo WARN: %SDKDIR%\RobotSDK.dll 不存在，跳过部署
  exit /b 0
)
copy /Y "%OUT%" "%SDKDIR%\scara_enable.exe" >nul
echo Deployed: %SDKDIR%\scara_enable.exe
endlocal

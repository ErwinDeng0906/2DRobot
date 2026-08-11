@echo off
setlocal EnableExtensions
REM 编译 scara_enable.exe → 部署到 SNRobotLab
REM 部署目录：SNROBOTLAB_DIR → local_config.toml 的 snrobotlab_dir
REM HARDCODED_PATH(dev-only): 下面 MSYS2 / 默认 SDKDIR 仅本机编译用

set SRC=%~dp0scara_enable.c
set OUT=%~dp0scara_enable.exe
set ROOT=%~dp0..\..
set SDKDIR=

if defined SNROBOTLAB_DIR set "SDKDIR=%SNROBOTLAB_DIR%"
if not defined SDKDIR (
  for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$p=Join-Path '%ROOT%' 'local_config.toml'; if(Test-Path $p){ $t=Get-Content -Raw $p; if($t -match 'snrobotlab_dir\s*=\s*\"([^\"]+)\"'){ $matches[1] -replace '\\\\','\' } }"`) do set "SDKDIR=%%I"
)
if not defined SDKDIR set "SDKDIR=D:\SNRobotLab"

set "GCC="
if exist "D:\MSYS2\ucrt64\bin\gcc.exe" set "GCC=D:\MSYS2\ucrt64\bin\gcc.exe"
if not defined GCC if exist "D:\MSYS2\mingw64\bin\gcc.exe" set "GCC=D:\MSYS2\mingw64\bin\gcc.exe"
if not defined GCC (
  where gcc >nul 2>&1
  if not errorlevel 1 set "GCC=gcc"
)
if not defined GCC (
  echo ERR: 找不到 gcc。跑 main.py 不需要 gcc；编译请安装 MSYS2 或把 gcc 加入 PATH
  exit /b 1
)

for %%I in ("%GCC%") do set "GCCBIN=%%~dpI"
set "PATH=%GCCBIN%;%PATH%"

echo Using: %GCC%
echo Deploy dir: %SDKDIR%
"%GCC%" -O2 -Wall -Wextra -o "%OUT%" "%SRC%" -lkernel32
if errorlevel 1 exit /b 1
echo Built: %OUT%

if not exist "%SDKDIR%\RobotSDK.dll" (
  echo WARN: %SDKDIR%\RobotSDK.dll 不存在，跳过部署（请把 %OUT% 手动复制到 SNRobotLab）
  exit /b 0
)
copy /Y "%OUT%" "%SDKDIR%\scara_enable.exe" >nul
echo Deployed: %SDKDIR%\scara_enable.exe
endlocal

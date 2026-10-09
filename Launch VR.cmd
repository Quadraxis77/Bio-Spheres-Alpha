@echo off
setlocal
rem Use a packaged executable when present; otherwise build the source checkout.
set "BIOSPHERES_VR_EXE=%~dp0bio-spheres.exe"
if exist "%BIOSPHERES_VR_EXE%" goto launch
if exist "%~dp0Cargo.toml" goto build_source
set "BIOSPHERES_VR_EXE=%~dp0target\release\bio-spheres.exe"
if exist "%BIOSPHERES_VR_EXE%" goto launch
echo Biospheres could not be found. Place bio-spheres.exe beside this launcher.
pause
exit /b 1

:build_source
echo Building the latest Biospheres release...
pushd "%~dp0"
if errorlevel 1 goto build_failed
cargo build --release --bin bio-spheres
if errorlevel 1 goto build_failed_in_source
popd
set "BIOSPHERES_VR_EXE=%~dp0target\release\bio-spheres.exe"
if not exist "%BIOSPHERES_VR_EXE%" goto build_failed
goto launch

:build_failed_in_source
popd
:build_failed
echo.
echo Biospheres could not be built. Fix the build errors above and try again.
pause
exit /b 1

:launch
echo Checking the headset and native VR graphics...
"%BIOSPHERES_VR_EXE%" --vr-check
if errorlevel 1 goto vr_failed
start "" "%BIOSPHERES_VR_EXE%" --vr %*
exit /b %errorlevel%

:vr_failed
echo.
echo Native VR could not start. The game has not been launched.
echo Keep the headset connected through Steam Link (SteamVR) or Virtual Desktop and check the error above.
pause
exit /b 1

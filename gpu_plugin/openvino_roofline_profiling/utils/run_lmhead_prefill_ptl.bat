@echo off
call D:\river\moe\openvino\release_install\setupvars.bat
set PATH=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release;D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin;%PATH%
set CLI_DevicePerformanceTiming=1
set CLI_DevicePerformanceTimingSkipUnmap=1
set CL=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set B=D:\river\moe\dev_roofline_profiling\utils\build\Release
set R=D:\river\moe\roofline_results
for %%S in (1024 2048 4096 8192) do (
  echo --- LM_Head prefill S=%%S ---
  %CL% %B%\fc_bench.exe %%S 4096 151936 128 50 10 8 u8 > %R%\fc_prefill_LMHead_S%%S.log 2>&1
)
echo DONE

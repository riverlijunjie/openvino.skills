Get-CimInstance Win32_Process -Filter "Name='conv_bench.exe' or Name='fc_bench.exe' or Name='cliloader.exe'" | Select-Object ProcessId, CommandLine | Format-List

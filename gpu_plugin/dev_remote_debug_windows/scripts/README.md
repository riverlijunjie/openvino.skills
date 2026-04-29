# A/B Test Scripts for Windows Remote Debugging

## Overview

Generic A/B test framework for comparing two OpenVINO source variants on a Windows machine.
Automates: source file swap → MSBuild recompile → install → DLL deploy → benchmark runs.

## Files

- **ab_test.bat** — Main runner script (does not need editing)
- **ab_test_config.bat.template** — Configuration template; copy to `ab_test_config.bat` and customize

## Quick Start

```bat
REM 1. Copy the template
copy ab_test_config.bat.template ab_test_config.bat

REM 2. Edit ab_test_config.bat:
REM    - Set OV_DIR, BUILD_DIR, INSTALL_DIR, PY, model paths
REM    - Define FILE_<n>_SRC_a / FILE_<n>_SRC_b / FILE_<n>_DST for each source file pair
REM    - Define SCENARIO_<n>_NAME / SCENARIO_<n>_SETUP for each test scenario

REM 3. Run
ab_test.bat all        REM Run both variants
ab_test.bat a          REM Run variant A only
ab_test.bat b          REM Run variant B only
```

## Configuration Reference

### Paths
| Variable | Description |
|---|---|
| `OV_DIR` | OpenVINO source root |
| `BUILD_DIR` | CMake build directory |
| `INSTALL_DIR` | `cmake --install` prefix |
| `AB_DIR` | Directory containing variant source files |
| `LOG_DIR` | Output directory for per-run stderr logs |
| `PY` | Python executable path |
| `BENCH_WORK_DIR` | Working directory for benchmark command |
| `BENCH_CMD` | Full benchmark command line |
| `DLL_DEPLOY_DIR` | Copy installed DLLs here (e.g. Python site-packages); leave empty to skip |

### Variant Labels
| Variable | Default | Description |
|---|---|---|
| `LABEL_A` | `A` | Display label for variant A (e.g. `baseline`) |
| `LABEL_B` | `B` | Display label for variant B (e.g. `optimized`) |

### Run Settings
| Variable | Default | Description |
|---|---|---|
| `RUNS_PER_SCENARIO` | `3` | Number of benchmark runs per scenario |
| `BUILD_PARALLEL` | `16` | MSBuild parallelism (`--parallel N`) |
| `LOG_FILTER` | `Pipeline initialization` | Grep pattern for stdout summary after each run |

### File Mappings
```bat
set FILE_COUNT=2

set FILE_1_SRC_a=C:\path\to\variant_a\file1.cpp
set FILE_1_SRC_b=C:\path\to\variant_b\file1.cpp
set FILE_1_DST=C:\openvino\src\...\file1.cpp

set FILE_2_SRC_a=...
set FILE_2_SRC_b=...
set FILE_2_DST=...
```

### Scenarios
```bat
set SCENARIO_COUNT=2

set SCENARIO_1_NAME=nommap
set SCENARIO_1_SETUP=echo {"ENABLE_MMAP": false} > config.json

set SCENARIO_2_NAME=mmap
set SCENARIO_2_SETUP=echo {"ENABLE_MMAP": true} > config.json
```

Set `SCENARIO_COUNT=0` to run `BENCH_CMD` directly without any scenario setup.

## MSBuild Incremental Build Note

The script uses `powershell -Command "(Get-Item ...).LastWriteTime = Get-Date"` to touch
each replaced source file after `copy /Y`. This forces MSBuild to recompile changed files,
which otherwise may reuse stale `.obj` files when the content changes but the timestamp
from `copy` preserves the original.

## Output

Logs are written to `LOG_DIR`:
```
<LABEL>_<SCENARIO>_<RUN>.log     (stderr from each benchmark run)
```

Example: `baseline_nommap_1.log`, `optimized_mmap_3.log`

# Intel GPU Device IDs and Driver Requirements

This reference documents Intel GPU device IDs found in RIL DUTs and their driver requirements.

## DG2 / Intel Arc

| Device ID | GPU Name | Generation | Driver requirement |
|-----------|----------|------------|--------------------|
| 8086:5690 | DG2 / Arc A-series | Xe HPG | Neo 22.x+ |
| 8086:56A1 | Arc A380 | Xe HPG | Neo 22.x+ |
| 8086:56A5 | Arc A310 | Xe HPG | Neo 22.x+ |
| 8086:56B0 | DG2 / Arc A-series | Xe HPG | Neo 22.x+ |

Standard SLES 15-SP5 `intel-opencl` 21.x is incompatible with these Arc/DG2 GPUs. Use Intel oneAPI packages or an Ubuntu DUT.

## Alchemist / DG2

| Device ID | GPU Name | Notes |
|-----------|----------|-------|
| 8086:E20B | Arc A770/A750 | Also used as B580 predecessor in some lab notes |
| 8086:E216 | Arc A-series | Different SKU |
| 8086:46A6 | DG2 | Seen in FM lab |

## Battlemage

| Device ID | GPU Name | Generation |
|-----------|----------|------------|
| 8086:E20B | Arc B580 | Xe2 HPG |
| 8086:E216 | Arc B-series | Xe2 HPG |

IGK7-7002 and IGK2-2109 labs have B580 machines.

## Panther Lake Integrated GPU

| Device ID | GPU Name | Generation | Notes |
|-----------|----------|------------|-------|
| 8086:B082 | PTL-H iGPU, 12Xe | Xe3 | Integrated GPU |
| 8086:B080 | PTL-H iGPU | Xe3 | Integrated GPU |

PTL iGPUs have `discrete_gpu_count=0`. Search GTA-X via `integrated_gpu_gfx_device_id`, not `discrete_gpu_gfx_device_id`.

## Meteor Lake Integrated GPU

| Device ID | GPU Name | Generation |
|-----------|----------|------------|
| 8086:7D67 | MTL iGPU | Xe LPG |

## Arrow Lake Integrated GPU

| Device ID | GPU Name | Generation |
|-----------|----------|------------|
| 8086:7D55 | ARL-H iGPU | Xe LPG |

## Intel Compute Runtime Versions

| Version | Release window | Minimum GPU support |
|---------|----------------|---------------------|
| 21.x | 2023-2024 | Integrated GPUs, MTL and older |
| 22.x | 2024-2025 | Arc/DG2, Battlemage |
| 23.x+ | 2025+ | Panther Lake and latest Arc |

## Driver Package Names by Distribution

### Ubuntu 22.04 / 24.04

- `intel-level-zero-gpu` or `libze-intel-gpu1` - Level Zero GPU driver package naming differs by repo/release
- `intel-opencl-icd` - OpenCL ICD
- `intel-igc` / `libigc1` - Intel Graphics Compiler

### SLES 15-SP5

- `intel-opencl` from SLES: version 21.x, too old for Arc/DG2
- `level-zero`: loader only, no GPU driver
- Intel oneAPI repo is required for Arc/Battlemage support

## References

- Intel GPU Device IDs: `https://pci-ids.ucw.cz/read/PC/8086`
- Intel Compute Runtime: `https://github.com/intel/compute-runtime`
- Intel client GPU driver docs: `https://dgpu-docs.intel.com/driver/client/overview.html`

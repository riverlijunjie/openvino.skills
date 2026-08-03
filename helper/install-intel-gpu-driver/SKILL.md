---
name: install-intel-gpu-driver
description: 'Install or validate Intel GPU kernel driver, OpenCL, Level Zero, and media/runtime packages on Windows, Ubuntu, and SLES/RIL DUTs. Use when asked to install Intel GPU drivers, set up Intel Arc/client GPU packages, fix missing intel-opencl-icd or intel-level-zero-gpu, or prepare a DUT before OpenVINO GPU, CM, or OpenCL work.'
argument-hint: "[windows|ubuntu|sles] [remote_os?] [--reboot]"
metadata:
  tags: [intel_gpu, driver, opencl, level_zero, setup, ril]
---

# Install Intel GPU Driver

Use this skill whenever the task includes Intel GPU driver or GPU runtime installation. Other skills should call this skill instead of duplicating OS-specific driver package instructions.

Do not use this skill for CM compiler or `clangFEWrapper` setup. After the GPU driver/runtime is installed, use `install_intel_gpu_cm_env` for CM headers, CM compiler, and CM validation.

## Scope

This skill covers:

- Windows Intel Arc graphics driver installer flow
- Ubuntu Intel client GPU package flow for Ubuntu Desktop 24.04 / 25.10
- SLES 15-SP5 constraints for Intel Arc/DG2 and oneAPI package requirements
- OpenCL and Level Zero runtime package checks
- post-install validation and reboot guidance

## Caller Guidance

- If the user is provisioning a RIL machine for Copilot tools and also asks for GPU setup, call `setup-ril4copilot` for proxy/GitHub/Copilot tooling, then call this skill for the driver decision and validation.
- If the user is reserving or preparing a GTA-X DUT, call `gtax-ril-operator` for reservation and image restore, then call this skill for driver installation on the booted OS.
- If the user is installing the CM environment, call this skill first when the GPU driver/runtime is missing, then call `install_intel_gpu_cm_env` with `--skip-driver`.
- If the user is debugging `clinfo` zero platforms or missing OpenCL runtime, call `diagnose-intel-gpu-opencl` first for root-cause analysis, then call this skill for the repair step.

## Preflight

Run these before installing packages:

```bash
cat /etc/os-release 2>/dev/null || ver
uname -a 2>/dev/null || true
lspci -nn | grep -Ei 'VGA|Display|3D|Intel' || true
ls -l /dev/dri 2>/dev/null || true
```

On remote RIL hosts, prefer `sshsh` for short commands and `putsh`/`getsh` for scripts and logs.

## Ubuntu Desktop 24.04 / 25.10

The Ubuntu path follows Intel client GPU guidance for Ubuntu Desktop 24.04 and 25.10.

Important:

- Ubuntu 25.10 has native support for Lunar Lake, Battlemage, and Panther Lake.
- Ubuntu 24.04 should use the HWE kernel for Lunar Lake, Battlemage, and Panther Lake when required by Intel's hardware table.
- Use the Intel graphics PPA for the documented client GPU package set.

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common pciutils
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get update

sudo apt-get install -y \
  libze-intel-gpu1 \
  libze1 \
  intel-metrics-discovery \
  intel-opencl-icd \
  clinfo \
  intel-gsc \
  intel-media-va-driver-non-free \
  libmfx-gen1 \
  libvpl2 \
  libvpl-tools \
  libva-glx2 \
  va-driver-all \
  vainfo

for group_name in render video; do
  if getent group "${group_name}" >/dev/null 2>&1; then
    sudo usermod -a -G "${group_name}" "${USER}"
  fi
done
```

Validate:

```bash
uname -r
lspci -nnk | grep -A3 -Ei 'VGA|Display|3D'
lsmod | grep -E '^(i915|xe) ' || true
ls -l /dev/dri/render* /dev/dri/by-path 2>/dev/null || true
clinfo -l
if [ -e /dev/dri/renderD128 ]; then
  vainfo --display drm --device /dev/dri/renderD128 2>/dev/null | head -n 20 || true
fi
```

Reboot after kernel driver, firmware, or group membership changes:

```bash
sudo reboot
```

## Ubuntu 22.04 and ComputeSDK/CM Machines

For Ubuntu 22.04 CM machines, prefer installing the exact Intel OpenCL/Level Zero packages required by the ComputeSDK flow in `install_intel_gpu_cm_env`. If the kernel driver is already loaded, call that skill with:

```bash
./install_intel_gpu_stack.sh --skip-driver
```

Use this skill for the base driver/runtime validation, then use `install_intel_gpu_cm_env` for CM-specific `clangFEWrapper`, headers, IGC-CM packages, and `ocloc`/`cmc` validation.

## SLES 15-SP5 RIL DUTs

Do not install the stock SLES `intel-opencl` package on Intel Arc/DG2 DUTs.

The standard SLES 15-SP5 repositories provide outdated Intel GPU packages:

- `intel-opencl` version 21.x - too old for DG2/Arc GPUs
- `level-zero` - loader only, not the GPU driver (`libze_intel_gpu.so`)

The stock `intel-opencl` package can make `clinfo` hang, crash OpenCL initialization, or make the DUT unresponsive on Arc/DG2 systems.

Intel Arc/DG2 requires Intel Compute Runtime (Neo) 22.x or later:

- `intel-level-zero-gpu`
- `intel-opencl` 22.x+ from Intel oneAPI, not SLES stock repositories
- `intel-igc`
- `intel-gmmlib`

If the Intel oneAPI repository is reachable:

```bash
sudo zypper addrepo --refresh https://yum.repos.intel.com/oneapi/sles/15SP5 intel-oneapi
sudo zypper --non-interactive install intel-level-zero-gpu intel-opencl
```

If the oneAPI repo is not reachable, use an Ubuntu DUT where possible or fetch the required RPMs from a networked machine and copy them to the DUT. Do not fall back to SLES stock `intel-opencl` for Arc/DG2.

Validate on SLES:

```bash
ls -la /usr/lib64/libze_intel_gpu.so*
ls -la /etc/OpenCL/vendors/intel.icd
cat /etc/OpenCL/vendors/intel.icd
timeout 15 clinfo -l
ls -la /dev/dri/render*
cat /sys/class/drm/renderD*/device/uevent
```

## Windows Intel Arc Driver

Use this flow for remote Windows DUTs when the user asks for Intel Arc Windows driver installation:

1. Open the Intel Arc Graphics Windows driver page.
2. Download the `gfx_win_*.exe` installer from `downloadmirror.intel.com`.
3. Validate the Authenticode signature and ensure the publisher is Intel Corporation.
4. Run the installer silently.
5. Treat exit codes `0`, `14`, `1000`, and `3010` as non-fatal; `14` and `3010` require reboot, `1000` means no supported hardware was detected but binaries were installed.

`setup-ril4copilot` already automates this flow when `-InstallGpuDriver` is passed on a Windows host. Use this skill as the source of truth for deciding when to install, how to validate, and whether to reboot.

Validation:

```powershell
Get-CimInstance Win32_PnPSignedDriver |
  Where-Object { $_.DeviceClass -eq 'DISPLAY' } |
  Select-Object DeviceName, DriverVersion
```

## Common Validation Signals

Linux:

```bash
lspci -nn | grep -Ei 'VGA|Display|3D'
lspci -nnk | grep -A3 -Ei 'VGA|Display|3D'
lsmod | grep -E '^(i915|xe) ' || true
ls -l /dev/dri /dev/dri/by-path 2>/dev/null || true
dpkg -l 2>/dev/null | grep -E 'intel-fw-gpu|intel-opencl-icd|intel-level-zero-gpu|libigdgmm12|libze-intel-gpu1' || true
rpm -qa 2>/dev/null | grep -Ei 'intel.*opencl|level-zero|igc|gmmlib' || true
timeout 15 clinfo -l
```

Windows:

```powershell
Get-CimInstance Win32_PnPSignedDriver |
  Where-Object { $_.DeviceClass -eq 'DISPLAY' } |
  Select-Object DeviceName, DriverProviderName, DriverVersion
```

## Troubleshooting

- `clinfo` shows zero platforms: run `diagnose-intel-gpu-opencl` to separate PCI enumeration, kernel binding, DRM nodes, ICD file, and userspace runtime issues before reinstalling packages.
- `clinfo` hangs or crashes on SLES 15-SP5: check whether stock SLES `intel-opencl` was installed on Arc/DG2. Remove it and use Intel oneAPI packages or an Ubuntu DUT.
- `/dev/dri/render*` is missing: check kernel driver binding and BIOS/firmware GPU visibility before reinstalling userspace packages.
- Runtime packages installed but no GPU is exposed: compare `i915` vs `xe` binding, kernel version, and `/dev/dri/by-path` with a known-good machine.

## References

- Intel client GPU driver documentation: `https://dgpu-docs.intel.com/driver/client/overview.html#installing-client-gpus-on-ubuntu-desktop`
- Device ID and package reference: `references/intel-gpu-driver-requirements.md`
- Intel GPU driver and runtime packages: `https://github.com/intel/compute-runtime/releases`

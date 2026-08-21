#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

# Install the ROCm HIP SDK inside a manylinux build container. Invoked by
# cibuildwheel via CIBW_BEFORE_ALL_LINUX before the ROCm wheel is built.

set -euo pipefail

version="${1:?Usage: install_rocm.sh <major.minor>}"

# ROCm installs into /opt/rocm-<major.minor.patch>, and the repository for a
# given major.minor tracks the latest patch, so the exact directory is only
# known after the install. Resolve it by glob and take the highest version.
find_rocm_root() {
    local candidates=()
    shopt -s nullglob
    candidates=(/opt/rocm-"${version}"*)
    shopt -u nullglob
    ((${#candidates[@]})) || return 1
    printf '%s\n' "${candidates[@]}" | sort -V | tail -1
}

if rocm_root="$(find_rocm_root)"; then
    ln -sfn "${rocm_root}" /opt/rocm
    "/opt/rocm/bin/hipconfig" --version
    exit 0
fi

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
else
    echo "Cannot determine Linux distribution." >&2
    exit 1
fi

case "${VERSION_ID%%.*}" in
    8)
        repo_platform="rhel8"
        ;;
    9)
        repo_platform="rhel9"
        ;;
    *)
        echo "Unsupported ROCm CI base image: ${ID:-unknown} ${VERSION_ID:-unknown}" >&2
        exit 1
        ;;
esac

if command -v dnf >/dev/null 2>&1; then
    package_manager="dnf"
elif command -v yum >/dev/null 2>&1; then
    package_manager="yum"
else
    echo "Expected dnf or yum in the manylinux container." >&2
    exit 1
fi

"${package_manager}" install -y ca-certificates curl

cat >/etc/yum.repos.d/rocm.repo <<EOF
[ROCm-${version}]
name=ROCm ${version}
baseurl=https://repo.radeon.com/rocm/${repo_platform}/${version}/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

"${package_manager}" clean all
# rocm-hip-sdk provides the HIP runtime and compiler plus the math libraries
# Ginkgo's cmake/hip.cmake requires: hipblas, hipsparse, hiprand, rocrand and
# rocthrust. The kernel driver (amdgpu-dkms) is deliberately not installed --
# building needs only the userspace stack.
"${package_manager}" install -y rocm-hip-sdk

if ! rocm_root="$(find_rocm_root)"; then
    echo "ROCm ${version} was installed but /opt/rocm-${version}* was not found." >&2
    exit 1
fi

ln -sfn "${rocm_root}" /opt/rocm
"/opt/rocm/bin/hipconfig" --version

# ROCm's bundled clang only probes the standard prefixes when looking for a GCC
# installation, but the manylinux image keeps its GCC in a gcc-toolset prefix
# (see DEVTOOLSET_ROOTPATH in pypa/manylinux's Dockerfile) and merely puts it on
# PATH -- so clang finds no libstdc++ and the HIP compiler check fails with
# "'cmath' file not found". Expose the toolchain at a fixed path so the build can
# pass --gcc-toolchain=/opt/manylinux-gcc without hard-coding a GCC version.
gcc_root=""
shopt -s nullglob
for candidate in /opt/rh/gcc-toolset-*/root/usr; do
    [[ -d "${candidate}/include/c++" ]] && gcc_root="${candidate}"
done
shopt -u nullglob

if [[ -z "${gcc_root}" ]]; then
    echo "Could not find a GCC toolset providing C++ headers." >&2
    echo "c++ resolves to: $(command -v c++ 2>/dev/null || echo none)" >&2
    echo "  -> $(readlink -f "$(command -v c++)" 2>/dev/null || echo none)" >&2
    echo "Candidates under /opt/rh:" >&2
    ls -d /opt/rh/* 2>/dev/null >&2 || echo "  (none)" >&2
    exit 1
fi

ln -sfn "${gcc_root}" /opt/manylinux-gcc
echo "Using GCC toolchain at ${gcc_root}"

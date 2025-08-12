#!/bin/bash

# ============================================================================
# Chameleon stack installation (OpenBLAS + OPENMPI + StarPU + CUDA + hwloc)
# NOTE: This version is a complete installation script to set up the Chameleon
#       stack with all required dependencies.
# ============================================================================

set -e

# ----------------------------------------------------------------------------- 
 Colors
# -----------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE} ==================== COMPLETE INSTALLATION ======================"

# ============================================================================
# 1) REQUIREMENTS
# ============================================================================
echo -e "${YELLOW}------------------ 1 Requirement ---------------"
echo -e "${RED} 1. Using a Linux system with NVIDIA GPU ${NC} "
echo -e "${RED} 2. At least 8GB RAM and 16GB disk ${NC}"


echo -e "${YELLOW}  This script will remove locally installed librairies"
read -p "Continue ? [y/N]: " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo -e "${RED} Abandon.${NC}"
    exit 1
fi


# ============================================================================
# 2) SYSTEM DEPENDENCIES
# ============================================================================
echo -e "${PURPLE}------------------ 2 INSTALLING SYSTEM DEPENDENCIES-----------------------------"

sudo apt update 

echo -e "${YELLOW} Optional removal of leftover or corrupted version "
sudo apt remove --purge -y cmake libtool autoconf automake m4 || true
sudo apt autoremove -y
sudo apt autoclean

echo -e "${YELLOW} Installing core libraries and tools... ${NC}"
sudo apt install -y \
    build-essential cmake git pkg-config \
    libhwloc-dev libnuma-dev libomp-dev \
    libtool libtool-bin autoconf automake m4 \
    gfortran curl wget unzip \
    zlib1g-dev libffi-dev libssl-dev \
    libevent-dev libudev-dev libcap-dev \
    libx11-dev libxext-dev libxrender-dev libxtst-dev \
    python3 python3-pip python3-dev \
    libxml2-dev libxslt1-dev \
    ninja-build

echo -e "${GREEN} Base system packages installed.${NC}"


# ============================================================================
# 3) OPENBLAS INSTALLATION
# ============================================================================
echo -e "${PURPLE}------------------ 3 OpenBLAS INSTALLATION -----"

cd ~

echo -e "${YELLOW} Cleaning up previous installation ${NC}"
if [ -d "OpenBLAS" ]; then
    echo  "  OpenBLAS directory already present. Removing it for cleaner installation"
    rm -rf OpenBLAS
fi

echo -e "${YELLOW} OpenBLAS download ${NC}"
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
echo " Compilation with LAPACK + LAPACKE supports..."
make USE_LAPACK=1 LAPACKE=1 -j$(nproc)
echo  " Checking compiled files libopenblas.so and lapacke.h..."
if [[ ! -f libopenblas.so ]] || ! find . -name "lapacke.h" | grep -q lapacke.h; then
    echo -e "${RED} OpenBLAS compilation with LAPACKE failed.${NC}"
    exit 1
echo -e "${GREEN} OpenBLAS compilation with LAPACKE successful.${NC}"
fi
echo " Installation... "
sudo make install

read -p " Is the installation as expected? ? [y/N]: " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo " Abandon."
    exit 1
fi


# ============================================================================
# 4) OPENMPI INSTALLATION
# ============================================================================
echo -e "${PURPLE}------------------ 4 OpenMPI INSTALLATION --------------------"

echo -e "${YELLOW} Cleaning up previous installation ${NC}"
if dpkg -s libopenmpi-dev &> /dev/null || dpkg -s openmpi-bin &> /dev/null; then
    echo -e "${YELLOW} OpenBLAS directory already present. Removing it for a clean installation...${NC}"
    sudo apt remove --purge -y libopenmpi-dev openmpi-bin
    sudo apt autoremove -y
    sudo apt autoclean
fi

echo -e "${YELLOW} Installing OpenMPI via apt...${NC}"
sudo apt install -y libopenmpi-dev openmpi-bin

echo -e "${YELLOW} Verifying installation of OpenMPI...${NC}"
if ! command -v mpicc &> /dev/null; then
    echo -e "${RED} mpicc not found in PATH. OpenMPI installation failed"
    exit 1
fi

echo -e "${YELLOW}  mpicc found: $(which mpicc)${NC}"
mpicc --version | head -n 1
mpirun --version | head -n 1

read -p " Is the OpenMPI installation working as expected? [y/N]: " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo " Abandon."
    exit 1
fi



# ============================================================================
# 5) CUDA INSTALLATION
# ============================================================================
echo -e "${PURPLE}------------------ CUDA INSTALLATION --------------------${NC}"

# NVIDIA repository and package configuration
DISTRO_TAG="ubuntu2404"              # Expected tag for NVIDIA repo (Ubuntu 24.04)
ARCH_TAG="x86_64"                    # Architecture
CUDA_MAJOR="12"                      # Target major version
CUDA_MINOR="9"                       # Target minor version (for fixed meta-package)
CUDA_META_PKG="cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}"   # Preferred fixed meta-package
CUDA_META_PKG_FALLBACK="cuda-toolkit"                      # Rolling fallback if fixed not available
CUDA_HOME_DEFAULT="/usr/local/cuda"   # Default symlink path after CUDA install
HWLOC_PREFIX="/opt/hwloc_cuda"        # hwloc installation prefix
HWLOC_GIT_URL="https://github.com/open-mpi/hwloc.git"
HWLOC_GIT_BRANCH=""                   # Empty = master; e.g. "v2.11.2" for stable version

# Driver requirements for CUDA 12.9 Update 1
MIN_DRIVER_FOR_CUDA12="525.60.13"     # Minimum compatible version
REC_DRIVER_SERIES="575"               # Recommended series (~575.57.08)
REC_DRIVER_PKG="nvidia-driver-575"    # Recommended package (may vary by repo)

# Helper functions
verlte() { printf '%s\n%s' "$1" "$2" | sort -V -C; }
vergte() { printf '%s\n%s' "$2" "$1" | sort -V -C; }

ask_yesno() {
  local prompt="$1"; shift || true
  local default=${1:-N}
  local reply
  read -r -p "${prompt} [y/N]: " reply || reply="${default}"
  case "$reply" in
    [yY]|[yY][eE][sS]) return 0;;
    *) return 1;;
  esac
}

pause_msg() { echo -e "${CYAN}$*${NC}" >&2; }
err()       { echo -e "${RED}ERROR:${NC} $*" >&2; }
info()      { echo -e "${GREEN}$*${NC}"; }
warn()      { echo -e "${YELLOW}WARNING:${NC} $*" >&2; }

# Update package list
echo -e "${YELLOW}Updating package list...${NC}"
sudo apt update -y
sudo apt install -y build-essential dkms pkg-config curl wget gnupg lsb-release ca-certificates software-properties-common

# Remove legacy CUDA installations
echo -e "${YELLOW}Checking for legacy CUDA installations (nvidia-cuda-toolkit)...${NC}"
if dpkg -s nvidia-cuda-toolkit >/dev/null 2>&1; then
  warn "Legacy 'nvidia-cuda-toolkit' detected. Removing..."
  sudo apt remove --purge -y nvidia-cuda-toolkit || true
  sudo apt autoremove -y || true
  sudo apt autoclean -y || true
else
  echo -e "${GREEN}No legacy 'nvidia-cuda-toolkit' package found.${NC}"
fi

# Optional cleanup of previous CUDA driver/toolkit packages
if ask_yesno "Do you want to purge old cuda-* packages (if any) before installation?" N; then
  sudo apt remove --purge -y 'cuda*' 'libnvidia-*' || true
  sudo apt autoremove -y || true
fi

# Configure NVIDIA CUDA repository
echo -e "${YELLOW}Configuring NVIDIA CUDA repository (${DISTRO_TAG})...${NC}"
CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO_TAG}/${ARCH_TAG}/${CUDA_KEYRING_DEB}"

TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

pushd "${TMPDIR}" >/dev/null
if wget -q "${CUDA_KEYRING_URL}"; then
  sudo dpkg -i "${CUDA_KEYRING_DEB}"
else
  err "Failed to download CUDA keyring (${CUDA_KEYRING_URL}). Check network connectivity."
  exit 1
fi
popd >/dev/null

PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO_TAG}/${ARCH_TAG}/cuda-${DISTRO_TAG}.pin"
if wget -q -O cuda-${DISTRO_TAG}.pin "${PIN_URL}"; then
  sudo mv cuda-${DISTRO_TAG}.pin /etc/apt/preferences.d/cuda-repository-pin-600
else
  warn "Unable to retrieve CUDA pin file (non-blocking)."
fi

sudo apt update -y

# NVIDIA driver check/install
echo -e "${PURPLE}------------------ NVIDIA DRIVER CHECK/INSTALL --------------------${NC}"
HAS_NVIDIA_SMI=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA_SMI=true
fi

CURRENT_DRIVER=""
if ${HAS_NVIDIA_SMI}; then
  CURRENT_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 || true)
  echo -e "${GREEN}Detected NVIDIA driver version: ${CURRENT_DRIVER}${NC}"
else
  warn "No functional NVIDIA driver detected."
fi

NEED_DRIVER_INSTALL=false
if [[ -z "${CURRENT_DRIVER}" ]] || ! vergte "${CURRENT_DRIVER}" "${MIN_DRIVER_FOR_CUDA12}"; then
  NEED_DRIVER_INSTALL=true
  warn "NVIDIA driver is too old or missing (min required: ${MIN_DRIVER_FOR_CUDA12})."
fi

if ${NEED_DRIVER_INSTALL}; then
  echo -e "${YELLOW}Installing/upgrading NVIDIA driver...${NC}"
  if apt-cache show "${REC_DRIVER_PKG}" >/dev/null 2>&1; then
    sudo apt install -y "${REC_DRIVER_PKG}"
  else
    warn "${REC_DRIVER_PKG} not available. Using 'ubuntu-drivers autoinstall'."
    sudo ubuntu-drivers autoinstall -y || { err "Driver installation failed."; exit 1; }
  fi
  info "NVIDIA driver installed/updated. A reboot is required."
  if ask_yesno "Reboot now?" N; then
    sudo reboot
  else
    warn "Please reboot before using CUDA."
    exit 0
  fi
else
  echo -e "${GREEN}Current driver is sufficient for CUDA 12.x.${NC}"
fi

# CUDA toolkit installation
echo -e "${PURPLE}------------------ CUDA TOOLKIT INSTALLATION --------------------${NC}"
if apt-cache policy "${CUDA_META_PKG}" 2>/dev/null | grep -q Candidate; then
  CUDA_INSTALL_PKG="${CUDA_META_PKG}"
else
  warn "${CUDA_META_PKG} not available; falling back to ${CUDA_META_PKG_FALLBACK}."
  CUDA_INSTALL_PKG="${CUDA_META_PKG_FALLBACK}"
fi

echo -e "${YELLOW}Installing package ${CUDA_INSTALL_PKG}...${NC}"
sudo apt install -y "${CUDA_INSTALL_PKG}"

if ! command -v nvcc >/dev/null 2>&1 && [ -d "${CUDA_HOME_DEFAULT}/bin" ]; then
  warn "nvcc not in PATH; adding manually..."
  export PATH="${CUDA_HOME_DEFAULT}/bin:${PATH}"
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  err "nvcc not found. Check CUDA installation."
  exit 1
fi

# Environment setup for CUDA
echo -e "${YELLOW}Configuring CUDA environment in ~/.bashrc ...${NC}"
CUDA_HOME_RESOLVED="${CUDA_HOME_DEFAULT}"
if [ -L "${CUDA_HOME_DEFAULT}" ]; then
  CUDA_HOME_RESOLVED="$(readlink -f "${CUDA_HOME_DEFAULT}")"
elif [ ! -d "${CUDA_HOME_DEFAULT}" ]; then
  candidate=$(ls -d /usr/local/cuda-${CUDA_MAJOR}* 2>/dev/null | head -n1 || true)
  [[ -n "${candidate}" ]] && CUDA_HOME_RESOLVED="${candidate}"
fi

{
  echo "# --- CUDA (${CUDA_INSTALL_PKG}) ---"
  echo "export CUDA_HOME=${CUDA_HOME_RESOLVED}"
  echo 'export PATH=${CUDA_HOME}/bin:$PATH'
  echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH'
} >> ~/.bashrc
source ~/.bashrc

# CUDA verification
echo -e "${PURPLE}------------------ CUDA VERIFICATION --------------------${NC}"
if ! nvcc --version >/dev/null 2>&1; then
  err "nvcc not responding after PATH update."
  exit 1
fi
nvcc --version || true

if ! nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi failed; driver may not be loaded."
fi

if ! ask_yesno "CUDA base installation seems OK. Continue with hwloc installation?" Y; then
  echo -e "${YELLOW}User aborted.${NC}"
  exit 0
fi

# hwloc with CUDA support
echo -e "${PURPLE}------------------ INSTALLING HWLOC WITH CUDA SUPPORT --------------------${NC}"
CUDA_FOR_HWLOC="${CUDA_HOME_RESOLVED}"
if [ ! -d "${CUDA_FOR_HWLOC}/include" ] && command -v nvcc >/dev/null 2>&1; then
  CUDA_BINDIR=$(dirname "$(command -v nvcc)")
  CUDA_FOR_HWLOC=$(dirname "${CUDA_BINDIR}")
fi
echo -e "${YELLOW}Detected CUDA path for hwloc: ${CUDA_FOR_HWLOC}${NC}"

cd ~
[ -d hwloc ] && { echo -e "${YELLOW}Removing existing hwloc directory...${NC}"; rm -rf hwloc; }

echo -e "${YELLOW}Cloning hwloc sources...${NC}"
git clone "${HWLOC_GIT_URL}"
cd hwloc
[[ -n "${HWLOC_GIT_BRANCH}" ]] && git checkout "${HWLOC_GIT_BRANCH}"

echo -e "${YELLOW}Generating build scripts (autogen.sh)...${NC}"
./autogen.sh
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo -e "${YELLOW}Configuring hwloc (--enable-cuda --with-cuda=${CUDA_FOR_HWLOC})...${NC}"
./configure \
  --enable-cuda \
  --with-cuda="${CUDA_FOR_HWLOC}" \
  --prefix="${HWLOC_PREFIX}"

echo -e "${YELLOW}Building hwloc...${NC}"
make -j"$(nproc)"

echo -e "${YELLOW}Installing hwloc to ${HWLOC_PREFIX}...${NC}"
sudo make install

# Environment setup for hwloc
echo -e "${YELLOW}Updating ~/.bashrc for hwloc...${NC}"
grep -q "${HWLOC_PREFIX}/bin" ~/.bashrc || echo "export PATH=${HWLOC_PREFIX}/bin:\$PATH" >> ~/.bashrc
grep -q "${HWLOC_PREFIX}/lib" ~/.bashrc || echo "export LD_LIBRARY_PATH=${HWLOC_PREFIX}/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

# Validation
echo -e "${PURPLE}------------------ HWLOC VALIDATION --------------------${NC}"
if command -v lstopo >/dev/null 2>&1; then
  echo -e "${GREEN}lstopo found. Compact output example:${NC}"
  lstopo --no-io --of console | head -n 40 || true
else
  warn "lstopo not found in PATH (check if ~/.bashrc was reloaded)."
fi

if command -v hwloc-info >/dev/null 2>&1; then
  echo -e "${YELLOW}Checking for CUDA devices via hwloc-info...${NC}"
  hwloc-info --objects osdev: | grep -i cuda || echo "(No CUDA devices detected by hwloc; check driver and GPU access)"
fi

info "CUDA + hwloc installation complete. Open a NEW shell or run 'source ~/.bashrc' to activate the environment."







# ============================================================================
# 6) StarPU INSTALLATION
# ============================================================================
echo -e "${PURPLE}------------------ 6 StarPU INSTALLATION ------------------${NC}"

cd ~
echo -e "${YELLOW} Preparing build directory: build-chameleon-stack...${NC}"

if [ -d "$HOME/build-chameleon-stack" ]; then
    echo "  Directory build-chameleon-stack already exists. Removing for a clean setup.${NC}"
    rm -rf "$HOME/build-chameleon-stack"
fi

mkdir -p "$HOME/build-chameleon-stack"
cd "$HOME/build-chameleon-stack"


echo -e "${YELLOW} Cleaning up previous installation ${NC}"
if [ -d "starpu" ]; then
    echo -e "${YELLOW}  StarPU directory already present. Removing it for a clean install.${NC}"
    rm -rf starpu
fi

echo -e "${YELLOW} StarPU download ${NC}"
git clone https://gitlab.inria.fr/starpu/starpu.git
cd starpu
git checkout starpu-1.4


echo -e "${YELLOW} Configuration, Compilation & Installation ${NC}"
./autogen.sh
./configure --prefix=$HOME/.local \
    --with-hwloc=/opt/hwloc_cuda \
    --enable-cuda \
    --enable-opencl \
    --enable-mpi
    --with-cuda-lib=/usr/lib/x86_64-linux-gnu \
    --with-cuda-inc=/usr/include \
    --disable-build-doc \
    --with-blas-lib="-lopenblas" \
    --with-blas-inc="/usr/local/include"
make -j$(nproc)
make install



echo -e "${YELLOW}------------------ Checking StarPU installation ------------------${NC}"
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH
if pkg-config --exists starpu-1.4; then
    echo -e "${GREEN} StarPU version: $(pkg-config --modversion starpu-1.4)${NC}"
else
    echo -e "${RED} StarPU installation failed or not found by pkg-config.${NC}"
    exit 1
fi


read -p " Is the installation as expected? [y/N]: " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo " Abandon."
    exit 1
fi

export LD_LIBRARY_PATH="/opt/hwloc_cuda/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:$CHAMELEON_DIR/lib:$STARPU_PREFIX/lib:$CUDA_HOME/lib64"
export CHAMELEON_DIR=/chemin/vers/chameleon/install
export STARPU_PREFIX=$HOME/.local          
export OPENBLAS_PREFIX=$HOME/OpenBLAS 



# ============================================================================
# 7) ENVIRONMENT CONFIGURATION
# ============================================================================
echo -e "${PURPLE}------------------ 7 Environment Configuration -----------------"

echo -e "${YELLOW} Environment variables to add${NC}"
ENV_VARS=(
'export PATH=$HOME/.local/bin:$PATH'
'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH'
'export LD_LIBRARY_PATH=/opt/hwloc_cuda/lib:$LD_LIBRARY_PATH'
'export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH'
)

echo -e "${YELLOW} Safely add to ~/.bashrc if absent ${NC}"
for VAR in "${ENV_VARS[@]}"; do
    if ! grep -qxF "$VAR" ~/.bashrc; then
        echo "$VAR" >> ~/.bashrc
        echo -e "${GREEN} Added to ~/.bashrc:${NC} $VAR"
    else
        echo -e "${YELLOW} Already in ~/.bashrc:${NC} $VAR"
    fi
done
source ~/.bashrc

echo -e "${GREEN} Environment variables configured and loaded.${NC}"





# ============================================================================
# 8) CHAMELEON INSTALLATION
# ============================================================================
echo -e "${PURPLE}------------------ 8 CHAMELEON INSTALLATION ------------------"

cd ~/build-chameleon-stack

echo -e "${YELLOW} Cleaning up previous installation ${NC}"
if [ -d "chameleon" ]; then
    echo -e "${YELLOW} 'Chameleon' directory already present. Remove for a clean install..${NC}"
    rm -rf chameleon
fi

echo -e "${YELLOW} Chameleon download ${NC}"
git clone --recurse-submodules https://gitlab.inria.fr/solverstack/chameleon.git
cd chameleon || exit 1

mkdir -p $HOME/.local/lib/pkgconfig
cd $HOME/.local/lib/pkgconfig
ln -sf starpu-1.4.pc libstarpu.pc
ln -sf starpu-1.4.pc libstarpumpi.pc

# Export pour pkg-config
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH


cd ~/build-chameleon-stack/chameleon || exit 1
mkdir build && cd build

echo -e "${YELLOW} Configuration, Compilation & Installation ${NC}"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local \
  -DBUILD_SHARED_LIBS=ON \
  -DCHAMELEON_ENABLE_CUDA=ON \
  -DCHAMELEON_USE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCHAMELEON_ENABLE_MPI=ON \
  -DCHAMELEON_ENABLE_OPENCL=ON \
  -DCHAMELEON_SCHED_STARPU=ON \
  -DCHAMELEON_SCHED_PARSEC=OFF \
  -DCHAMELEON_SCHED_QUARK=OFF \
  -DCHAMELEON_ENABLE_TESTING=ON \
  -DCHAMELEON_PREC_D=ON \
  -DBLAS_LIBRARIES="/opt/OpenBLAS/lib/libopenblas.so" \
  -DBLAS_INCLUDE_DIRS="/opt/OpenBLAS/include" \
  -DLAPACKE_INCLUDE_DIR="/opt/OpenBLAS/include" \
  -DLAPACKE_LIBRARY="/opt/OpenBLAS/lib/libopenblas.so" \
  -DCBLAS_LIBRARY="/opt/OpenBLAS/lib/libopenblas.so" \
  -DSTARPU_DIR=$HOME/.local

make -j$(nproc)
make install


read -p "L'installation semble-t-elle correcte ? [y/N]: " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo -e "${RED} Abandon utilisateur.${NC}"
    exit 1
fi

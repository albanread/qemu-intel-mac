#!/bin/bash
# Build QEMU for an Intel Mac, to run on macOS 14 (Sonoma) or later.
#
#   intel-mac/build.sh            the libraries, then QEMU, in build-intel-mac/
#
# Homebrew no longer pours bottles for Intel Macs, and refuses to build
# QEMU from source while /Applications holds an older Xcode.  This builds
# with Apple's Command Line Tools instead:
#
#   1. glib, pcre2, libslirp and capstone, static, into DEPS (default
#      ../deps-macos14 beside this tree), by riscos-pi4/tools/build-deps-macos.sh;
#   2. QEMU's x86_64 system emulator against them, with HVF (Apple's
#      hypervisor), the Cocoa and Metal displays, and user-mode networking.
#
# Homebrew still supplies the tools and the few libraries it can pour:
#   brew install meson ninja pkg-config libpng pixman gettext libffi
# intel-mac/package.sh then makes a folder that needs none of them.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
deps=${DEPS:-$(cd "$here/.." && pwd)/deps-macos14}
build=${BUILD:-$here/build-intel-mac}
targets=${TARGETS:-x86_64-softmmu}
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-14.0}
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}

[ "$(uname -m)" = x86_64 ] || { echo "build.sh: this is the Intel Mac build" >&2; exit 1; }

echo "== the libraries, for macOS $MACOSX_DEPLOYMENT_TARGET, into $deps"
RISCOS_DEPS_PREFIX=$deps bash "$here/riscos-pi4/tools/build-deps-macos.sh"

echo "== QEMU ($targets) in $build"
export PKG_CONFIG_PATH="$deps/lib/pkgconfig:/usr/local/opt/libffi/lib/pkgconfig"
mkdir -p "$build"
cd "$build"
[ -f build.ninja ] || "$here/configure" \
    --target-list="$targets" \
    --enable-hvf --enable-cocoa --enable-slirp --enable-capstone --enable-virtfs \
    --disable-werror --disable-docs --disable-guest-agent \
    --disable-gtk --disable-sdl --disable-vnc --disable-spice \
    --disable-zstd --disable-gcrypt --disable-nettle --disable-gnutls \
    --disable-libssh --disable-curl \
    --cc=/usr/bin/clang --cxx=/usr/bin/clang++ --objcc=/usr/bin/clang \
    --extra-cflags="-mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET" \
    --extra-ldflags="-mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
make -j"$(sysctl -n hw.ncpu)"
echo "== built: $build/qemu-system-x86_64"

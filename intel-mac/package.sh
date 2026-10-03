#!/bin/bash
# Make a self-contained QEMU folder from intel-mac/build.sh's build:
#
#   intel-mac/package.sh OUT      OUT/bin, OUT/lib, OUT/share/qemu
#
#   bin/qemu-system-x86_64   the emulator, its libraries found in ../lib
#   lib/*.dylib              the Homebrew libraries it was linked with
#                            (libpng, pixman, libffi, libintl), copied in
#   share/qemu/              the firmware and option ROMs a PC needs: the
#                            UEFI firmware (OVMF) and its variable store,
#                            and the network cards' boot ROMs
#
# Everything is ad-hoc signed, the emulator with the hypervisor
# entitlement; a release is signed again with a Developer ID.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
build=${BUILD:-$here/build-intel-mac}
out=${1:?usage: package.sh OUT}
exe=$build/qemu-system-x86_64

rm -rf "$out"
mkdir -p "$out/bin" "$out/lib" "$out/share/qemu"
cp "$exe" "$out/bin/"
bin=$out/bin/qemu-system-x86_64

# the non-system libraries, and theirs, copied beside it and relinked
deps_of() { otool -L "$1" | sed 1d | awk '{print $1}' | grep -E '^/(usr/local|opt)/' || true; }
todo=$(deps_of "$bin")
while [ -n "$todo" ]; do
    next=
    for d in $todo; do
        name=$(basename "$d")
        install_name_tool -change "$d" "@executable_path/../lib/$name" "$bin" 2>/dev/null || true
        for l in "$out"/lib/*.dylib; do
            [ -f "$l" ] && install_name_tool -change "$d" "@loader_path/$name" "$l" 2>/dev/null || true
        done
        [ -f "$out/lib/$name" ] && continue
        cp "$(realpath "$d")" "$out/lib/$name"
        chmod u+w "$out/lib/$name"
        install_name_tool -id "@loader_path/$name" "$out/lib/$name"
        next="$next $(deps_of "$out/lib/$name")"
    done
    todo=$next
done
# a library copied after another that needs it: relink once more
for l in "$out"/lib/*.dylib; do
    for d in $(deps_of "$l"); do
        install_name_tool -change "$d" "@loader_path/$(basename "$d")" "$l"
    done
done

# the firmware: UEFI (QEMU's own edk2 build) and the cards' option ROMs
pcbios=$here/pc-bios
bunzip2 -c "$pcbios/edk2-x86_64-code.fd.bz2" > "$out/share/qemu/edk2-x86_64-code.fd"
bunzip2 -c "$pcbios/edk2-i386-vars.fd.bz2" > "$out/share/qemu/edk2-i386-vars.fd"
for f in efi-e1000e.rom efi-e1000.rom efi-virtio.rom kvmvapic.bin linuxboot_dma.bin \
         multiboot_dma.bin bios-256k.bin vgabios-stdvga.bin vgabios-virtio.bin; do
    [ -f "$pcbios/$f" ] && cp "$pcbios/$f" "$out/share/qemu/"
done
cp "$pcbios/edk2-licenses.txt" "$out/share/qemu/" 2>/dev/null || true
cp "$here/COPYING" "$out/COPYING"

# ad-hoc signatures: the libraries, then the emulator with its entitlement
ent=$(mktemp)
cat > "$ent" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>com.apple.security.hypervisor</key><true/></dict></plist>
EOF
for l in "$out"/lib/*.dylib; do codesign -f -s - "$l"; done
codesign -f -s - --entitlements "$ent" "$bin"
rm -f "$ent"

echo "== $out"
otool -L "$bin" | sed 1d | awk '{print "   " $1}' | grep -v '^   /System\|^   /usr/lib' || true
"$bin" --version | head -1

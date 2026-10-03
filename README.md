# QEMU for Intel Macs

**A modern QEMU (11.1) built for Intel Macs, which Homebrew no longer
supports** — with Apple's hypervisor, a Metal display, and a self-contained
build that runs on macOS 14 (Sonoma) and later.

In September 2026 Homebrew stopped supporting macOS on Intel: it no longer
builds bottles for it, and `brew install qemu` there now tries to compile
QEMU from source and refuses unless the very latest Xcode is installed.
Apple's own support for Intel Macs ends with macOS 26. Plenty of Intel Macs
— Mac Pros above all — have years of work left in them, and this is a QEMU
for them: built with the Command Line Tools alone, its libraries static or
carried beside it, so the result runs on a Mac that has no Homebrew at all.

It is also the **source of the QEMU shipped with
[RISC OS BOX x64 for Intel Macs](https://github.com/albanread/Aldershot/blob/main/translation/README.md)**,
published here as the GPL asks.

## What is different from stock QEMU

This is QEMU **v11.1.0** with these additions, all of them on top of
upstream and none of them needed to use QEMU in the ordinary way:

- **Built for Intel Macs, to run on macOS 14 or later.**
  [`intel-mac/build.sh`](intel-mac/build.sh) builds glib, pcre2, libslirp
  and capstone static for macOS 14, then QEMU's x86_64 system emulator with
  HVF (Apple's hypervisor: guests run at the processor's speed), user-mode
  networking and 9p folder sharing. [`intel-mac/package.sh`](intel-mac/package.sh)
  makes a folder with the four remaining libraries (libpng, pixman, libffi,
  libintl) beside the executable, the UEFI firmware (OVMF) and the option
  ROMs a PC guest needs, and the hypervisor entitlement signed in.

- **A Metal display, `-display metal`.** A window drawn by the Mac's GPU
  instead of Cocoa's, for a guest using **virtio-gpu** (or the Raspberry Pi
  framebuffer below):
  - `zoom=1..4` — the window's size in screen pixels per guest pixel, so a
    1280×800 guest is a sharp 1280×800-point window on a Retina screen
    (`zoom=2`) rather than a quarter of one;
  - `scaling=sharp|linear|nearest` — how a resized window is magnified;
    the default, sharp, keeps edges crisp without shimmering;
  - `scanlines=on` — every other row darkened when magnified at least twice;
  - `backdrop=...` — a host-drawn layer beneath the guest's desktop;
  - `vsync=N` — vertical-sync interrupts delivered to the guest;
  - the guest's hardware cursor (the virtio-gpu cursor, with its hot spot)
    drawn by the host at the Mac's mouse.

- **A Raspberry Pi 4 that boots RISC OS 5.** `-M raspi4b` with the
  Cortex-A72 in AArch32 runs RISC OS 5.30 from the RISC OS Open SD image to
  a networked desktop: USB keyboard and mouse through the DWC2 controller's
  FIQ path, Ethernet over USB, sound through the VCHIQ audio service, the
  SD card on EMMC2, and **HostFS** — a RISC OS filing system whose files
  live in a folder on the host. This part comes from
  [RISCOSQEMUA72](https://github.com/albanread/RISCOSQEMUA72) (now
  unmaintained); its documents are in [`riscos-pi4/`](riscos-pi4/README.md).

Upstream QEMU's own README is [README.rst](README.rst).

## Building

On an Intel Mac with the Command Line Tools (`xcode-select --install`) and
these from Homebrew, which still installs them:

```
brew install meson ninja pkg-config cmake libpng pixman gettext libffi
intel-mac/build.sh
intel-mac/package.sh ~/QEMU
~/QEMU/bin/qemu-system-x86_64 --version
```

`build.sh` puts the static libraries in `../deps-macos14` (`DEPS=` to
move them) and QEMU in `build-intel-mac/`.

## Running a PC guest with UEFI

The shape of the command RISC OS BOX x64 is started with (there the disc
is a USB stick, `-device usb-storage`; `disk.img` here is any UEFI-bootable
disc image):

```
cp ~/QEMU/share/qemu/edk2-i386-vars.fd vars.fd
~/QEMU/bin/qemu-system-x86_64 -L ~/QEMU/share/qemu \
  -machine q35 -accel hvf -cpu host -smp 4 -m 4G \
  -drive if=pflash,format=raw,readonly=on,file=$HOME/QEMU/share/qemu/edk2-x86_64-code.fd \
  -drive if=pflash,format=raw,file=vars.fd \
  -drive if=virtio,format=raw,file=disk.img \
  -vga none -device virtio-gpu-pci -display metal,zoom=2 \
  -nic user,model=e1000e \
  -audiodev coreaudio,id=snd -device intel-hda -device hda-output,audiodev=snd
```

`-display cocoa` works too, for guests without virtio-gpu.

## Licence

QEMU is released under the **GNU General Public License, version 2**
([COPYING](COPYING)); some parts are under other compatible licences, as
their files say ([LICENSE](LICENSE)). The additions here are under the same
terms. The UEFI firmware is TianoCore EDK II's, under its own licences
(`pc-bios/edk2-licenses.txt`).

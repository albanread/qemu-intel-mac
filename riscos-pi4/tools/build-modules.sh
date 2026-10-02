#!/bin/zsh
# build-modules.sh — build every module the release ships, on a Mac.
#
#   GVFill,ffa        host:   blitter/build.sh — clang + Homebrew's
#                              llvm-objcopy/readobj
#   HostNet,ffa       host:   hostnet/build-hostnet.sh — clang + the ROSCC
#                              repo beside this one
#   HostFS,ffa        guest:  the Acorn DDE inside RISC OS — hostfs/dde's
#                              Build,feb, run headless on a build share
#   HostFSFiler,ffa   guest:  as HostFS (hostfs/filer's Build,feb)
#
# The two host builds are plain cross-compiles.  The two guest builds boot
# the stock ROM on a share that carries the DDE (a copy of the farm's dev
# machine), with the PREVIOUS HostFS spliced into the ROM to serve the
# share — a module cannot build itself.  A boot task (ZZModBuild,feb) runs
# SetDDE and both Build.febs and writes BUILD_DONE back through HostFS;
# this script polls for it, stops the machine by its pidfile, and copies
# the fresh modules into the tree.  A failed Obey halts the boot, no
# marker arrives, and the screendump under build-modules/ shows the error.
#
#   --only LIST       comma-separated subset: gvfill,hostnet,hostfs
#   --fresh-share     rebuild build-modules/share from DDE (first run or
#                     after the farm machine changes)
#   --timeout SECS    the guest build's limit (default 900; the DDE boots
#                     and links slowly under TCG)
#   --keep            leave the machine running after a failure, for its
#                     screen (DISPLAY_OPT=none here, so look with QMP)
#
#   DDE               the dev machine share to copy !Boot, AcornC.C++ and
#                     CMOS,ff2 from (default: the farm's, bigmacfarm/)
#   BOOT_HOSTFS       the previous HostFS,ffa that boots the build machine
#                     (default: the tree's own hostfs/dde/HostFS,ffa from
#                     the last run, else the HostFS inside the newest
#                     release app's ROM under build-macos/release/; the
#                     filer is not needed to build headless).  Never the
#                     farm share's hostfs,ffa: that is HostFS 1.01, a cmhg
#                     build that relocates its own image at initialisation
#                     and so aborts from ROM -- the 17 Sep "hang", a
#                     DataAbort at the supervisor prompt behind a headless
#                     screen.  mkrom refuses such a module now.
#   RISCOS_IMAGES     the stock RISCOS.IMG and cmos.bin (default:
#                     riscos-images/)
#   QEMU_BIN          the emulator (default: build-macos/)
#
# Everything lands where make-release.sh looks: blitter/GVFill,ffa,
# hostnet/build/HostNet,ffa, hostfs/dde/HostFS,ffa,
# hostfs/filer/HostFSFiler,ffa.  The build share and logs stay under
# build-modules/ (gitignored with /build-*/).
set -e
HERE="${0:A:h}"
ROOT="${HERE:h:h}"
WORK="${WORK:-$ROOT/build-modules}"
DDE="${DDE:-${ROOT:h:h}/bigmacfarm/master/farm-dev-hostfs-2026-09-12}"
BOOT_HOSTFS="${BOOT_HOSTFS:-}"          # resolved under "guest builds"
IMAGES="${RISCOS_IMAGES:-$ROOT/riscos-images}"
Q="${QEMU_BIN:-$ROOT/build-macos/qemu-system-aarch64}"
TIMEOUT=900
ONLY=""
FRESH=0
KEEP=0

die()  { print -u2 "build-modules.sh: $*"; exit 1 }
step() { print -- "\n== $*" }

# The module version, as make-release.sh reads it for its summary
modver() { strings -n 6 "$1" | grep -m1 -oE '[0-9]+\.[0-9]{2} \([0-9]{2} [A-Za-z]+ [0-9]{4}\)' }

while [[ $# -gt 0 ]]; do
    case "$1" in
    --only)       ONLY=",$2,"; shift 2 ;;
    --fresh-share) FRESH=1; shift ;;
    --timeout)    TIMEOUT="$2"; shift 2 ;;
    --keep)       KEEP=1; shift ;;
    *)            die "unknown option: $1" ;;
    esac
done
want() { [[ -z "$ONLY" || "$ONLY" == *",$1,"* ]] }

# ---------------------------------------------------------------- the share
share="$WORK/share"
seed_share() {
    step "the build share (from $DDE)"
    [[ -d "$DDE/AcornC.C++" ]] || die "no DDE at $DDE/AcornC.C++ (DDE=...)"
    mkdir -p "$share"
    # Documents and Apps are not decoration.  The dev machine's stock !Boot
    # reaches outside itself -- Choices and every RO5x0Hook copy of
    # PinSetup load $.Documents.Images.Backdrops.Octagons, and Pinboard
    # pins $.Apps.  Without them the desktop stops on a Wimp error box
    # ("File ... not found") that waits for a click a headless machine
    # never makes, and the tasks after it -- ZZModBuild among them -- never
    # run.  Seeding them is a one-off copy; dropping the two Choices tasks
    # is not enough, because the version hooks carry their own.
    rsync -a "$DDE/!Boot" "$DDE/AcornC.C++" "$DDE/CMOS,ff2" "$share/"
    for extra in Documents Apps; do
        [[ -d "$DDE/$extra" ]] && rsync -a "$DDE/$extra" "$share/"
    done
    print "   !Boot + AcornC.C++ + CMOS,ff2 + Documents + Apps copied (a full copy, once)"
}
(( FRESH )) && rm -rf "$share"
[[ -d "$share/!Boot" && -d "$share/AcornC.C++" ]] || seed_share

# ------------------------------------------------------------- host builds
if want gvfill; then
    step "GVFill (blitter/build.sh)"
    sh "$ROOT/riscos-pi4/blitter/build.sh"
fi

if want hostnet; then
    step "HostNet (hostnet/build-hostnet.sh)"
    ROSCC="${ROSCC:-$ROOT/../ROSCC/target/debug/roscc}"
    [[ -x "$ROSCC" ]] || die "no roscc at $ROSCC — cargo build it in the ROSCC repo, or set ROSCC="
    sh "$ROOT/riscos-pi4/hostnet/build-hostnet.sh"
fi

# ------------------------------------------------------------ guest builds
if want hostfs; then
    step "HostFS + HostFSFiler (the DDE, inside RISC OS)"
    # The previous HostFS: the tree's own build when there has been one,
    # else the one inside the newest release ROM -- the module every
    # release boot runs from, so it is known to run from ROM.  Never the
    # farm share's hostfs,ffa, which is 1.01 and cannot (see the header).
    if [[ -z "$BOOT_HOSTFS" ]]; then
        if [[ -e "$ROOT/riscos-pi4/hostfs/dde/HostFS,ffa" ]]; then
            BOOT_HOSTFS="$ROOT/riscos-pi4/hostfs/dde/HostFS,ffa"
        else
            BOOT_HOSTFS="$WORK/boot-HostFS,ffa"
            mkdir -p "$WORK"
            python3 - "$ROOT" "$BOOT_HOSTFS" <<'PY' || die "no previous HostFS,ffa to boot on: set BOOT_HOSTFS="
import glob, os, struct, sys
root, out = sys.argv[1:]
sys.path.insert(0, os.path.join(root, "riscos-pi4", "tools"))
import mkrom
roms = glob.glob(os.path.join(root, "build-macos", "release", "*.app",
                              "Contents", "Resources", "RISCOS.IMG"))
if not roms:
    raise SystemExit("no release app under build-macos/release/")
rom = max(roms, key=os.path.getmtime)
data = open(rom, "rb").read()
# find_chain answers with titles and the terminator; the walk that
# produced them starts at the first offset that reproduces both
titles, term = mkrom.find_chain(data)
m = next(s for s in range(mkrom.OS_HDR + 0x40, 0x80000, 4)
         if (r := mkrom.walk_chain(data, s)) and r[1] == term
         and len(r[0]) == len(titles))
while True:
    length = struct.unpack_from("<I", data, m - 4)[0]   # module + next size word
    if length == 0:
        raise SystemExit(f"no HostFS in the chain of {rom}")
    body = data[m:m + length - 4]
    if mkrom.module_title(body) == "HostFS":
        open(out, "wb").write(body)
        print(f"   the previous HostFS is the one in {os.path.relpath(rom, root)}")
        break
    m += length
PY
        fi
    fi
    print "   boots on HostFS $(modver "$BOOT_HOSTFS")  $BOOT_HOSTFS"
    for f in "$Q" "$IMAGES/RISCOS.IMG" "$IMAGES/cmos.bin" "$BOOT_HOSTFS" \
             "$ROOT/riscos-pi4/hostfs/dde/Build,feb" \
             "$ROOT/riscos-pi4/hostfs/filer/Build,feb"; do
        [[ -e "$f" ]] || die "missing: $f"
    done

    src="$share/Sources/hostfs"
    rm -rf "$src"
    mkdir -p "$src/dde/o" "$src/filer/o"
    rsync -a "$ROOT/riscos-pi4/hostfs/dde/c" "$ROOT/riscos-pi4/hostfs/dde/s" \
              "$ROOT/riscos-pi4/hostfs/dde/SetDDE,feb" \
              "$ROOT/riscos-pi4/hostfs/dde/Build,feb" "$src/dde/"
    rsync -a "$ROOT/riscos-pi4/hostfs/filer/c" "$ROOT/riscos-pi4/hostfs/filer/s" \
              "$ROOT/riscos-pi4/hostfs/filer/SetDDE,feb" \
              "$ROOT/riscos-pi4/hostfs/filer/Build,feb" "$src/filer/"
    # A fresh marker is the only done the poll trusts; the logs (log,ffd
    # beside the sources) and the old modules go too, so what appears is
    # this run's and only this run's.
    rm -f "$share/BUILD_DONE,ffd" "$src/dde/HostFS,ffa" "$src/filer/HostFSFiler,ffa" \
          "$src/dde/log,ffd" "$src/filer/log,ffd"

    tasks="$share/!Boot/Choices/Boot/Tasks"
    [[ -d "$tasks" ]] || die "the share has no $tasks:!Boot/Choices/Boot/Tasks"
    # The dev machine's pinboard tasks want Documents/ (the Octagons
    # backdrop) and Apps/, neither of which this share carries; the error
    # box they raise sits on the desktop waiting for a click that never
    # comes, and the tasks after them never run.  A build machine has no
    # use for a pinboard.
    rm -f "$tasks/PinSetup,feb" "$tasks/Pinboard,feb"
    # ZZ: after every stock task, as vmmap's harness orders its own.
    cat > "$tasks/ZZModBuild,feb" <<'OBEY'
| Module build task (a build share; never shipped)
Set ModBuild$Root HostFS:$.Sources.hostfs
WimpSlot -min 16M -max 64M
Obey <ModBuild$Root>.dde.SetDDE
Obey <ModBuild$Root>.dde.Build { > HostFS:$.Sources.hostfs.dde.log }
Obey <ModBuild$Root>.filer.Build { > HostFS:$.Sources.hostfs.filer.log }
Echo done { > HostFS:$.BUILD_DONE }
OBEY

    start=$(date +%s)
    rm -f "$WORK/qmp.sock" "$WORK/pid"
    # run-macos.sh is the one launcher: ROM splice, CMOS, share, QMP.  The
    # ROM carries the PREVIOUS HostFS (BOOT_HOSTFS, and no filer — nothing
    # here needs an icon bar) to serve the share the new one is built on.
    RISCOS_IMAGES="$IMAGES" \
    RISCOS_HOSTFS="$share" \
    RISCOS_HOSTFS_MODULES="$BOOT_HOSTFS" \
    RISCOS_BOOT=hostfs \
    RISCOS_QMP="unix:$WORK/qmp.sock,server,nowait" \
    RISCOS_NAME=riscos-modulebuild \
    RISCOS_PIDFILE="$WORK/pid" \
    AUDIODEV=none DISPLAY_OPT=none \
        "$HERE/run-macos.sh" > "$WORK/qemu.log" 2>&1 &
    launcher=$!

    screendump() {
        python3 - "$WORK/qmp.sock" "$WORK/failure.png" <<'PY'
import json, socket, sys
s = socket.socket(socket.AF_UNIX)
s.settimeout(5)
try:
    s.connect(sys.argv[1])
except OSError as e:
    print(f"no QMP for the screendump: {e}", file=sys.stderr)
    raise SystemExit(0)
def rd():
    buf = b""
    while not buf.endswith(b"\r\n"):
        d = s.recv(4096)
        if not d:
            break
        buf += d
    return buf.decode(errors="replace")
rd()
for msg in ({"execute": "qmp_capabilities"},
            {"execute": "screendump",
             "arguments": {"filename": sys.argv[2], "format": "png"}}):
    s.sendall(json.dumps(msg).encode() + b"\r\n")
    rd()
PY
    }

    ok=0 gone=0
    while (( $(date +%s) - start < TIMEOUT )); do
        if [[ -e "$share/BUILD_DONE,ffd" ]]; then ok=1; break; fi
        # run-macos.sh execs the emulator, so $launcher is the machine
        if ! kill -0 "$launcher" 2>/dev/null; then gone=1; break; fi
        sleep 5
    done

    if (( gone )); then
        # It never got as far as a screen: a splice mkrom refused, an
        # image missing, a QEMU that would not start.  The log says which.
        print -u2 "the machine exited before BUILD_DONE; its log, $WORK/qemu.log:"
        sed 's/^/     /' "$WORK/qemu.log" >&2
        exit 1
    fi
    if (( ! ok )); then
        screendump
        print -u2 "the guest build did not finish in ${TIMEOUT}s; the machine's screen is $WORK/failure.png, its log $WORK/qemu.log"
        (( KEEP )) || { [[ -e "$WORK/pid" ]] && kill "$(cat "$WORK/pid")" 2>/dev/null; }
        exit 1
    fi
    [[ -e "$WORK/pid" ]] && kill "$(cat "$WORK/pid")" 2>/dev/null

    for out in "$src/dde/HostFS,ffa" "$src/filer/HostFSFiler,ffa"; do
        [[ -e "$out" ]] || die "BUILD_DONE arrived but $out did not"
        (( $(stat -f %m "$out") >= start )) || die "$out is not from this run"
    done
    print "   relocs (only PC-relative ones may appear):"
    sed 's/^/     /' "$src/dde/o/relocs,ffd" "$src/filer/o/relocs,ffd" 2>/dev/null || true
    print "   dde log:"; sed 's/^/     /' "$src/dde/log,ffd" 2>/dev/null || true
    cp "$src/dde/HostFS,ffa"      "$ROOT/riscos-pi4/hostfs/dde/HostFS,ffa"
    cp "$src/filer/HostFSFiler,ffa" "$ROOT/riscos-pi4/hostfs/filer/HostFSFiler,ffa"
fi

# ----------------------------------------------------------------- summary
step "built"
for m in "GVFill:$ROOT/riscos-pi4/blitter/GVFill,ffa" \
         "HostNet:$ROOT/riscos-pi4/hostnet/build/HostNet,ffa" \
         "HostFS:$ROOT/riscos-pi4/hostfs/dde/HostFS,ffa" \
         "HostFSFiler:$ROOT/riscos-pi4/hostfs/filer/HostFSFiler,ffa"; do
    name="${m%%:*}" mod="${m#*:}"    # not `path': in zsh that sets PATH
    [[ -e "$mod" ]] && print -- "   $name  $(modver "$mod")  $mod" \
                    || print -- "   $name  not built"
done

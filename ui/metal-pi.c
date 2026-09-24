/*
 * metal-pi.c — the Metal front end's board for the emulated Pi 4.
 *
 * The picture is the bcm2835 framebuffer in guest RAM, the palette at
 * the VideoCore RAM base, and the pointer the VCHIQ peer answers for;
 * the blitter says when the guest is drawing, and the vsync generator
 * paces it.  The debugging commands read the ARM core, and the HostNet
 * switch works the vmchannel doorbell (ui/metal-board.h).
 *
 * Linked only into targets with CONFIG_RASPI.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/main-loop.h"
#include "qom/object.h"
#include "system/address-spaces.h"
#include "system/hw_accel.h"
#include "system/reset.h"
#include "system/runstate.h"
#include "hw/core/cpu.h"
#include "hw/display/bcm2835_fb.h"
#include "hw/misc/bcm2835_vchiq.h"
#include "hw/misc/bcm2835_vsyncgen.h"
#include "hw/misc/hostnet.h"
#include "hw/misc/riscos_blitter.h"
#include "hw/misc/vmchannel.h"
#include "target/arm/cpu.h"     /* the Pi's AArch32 core: regs, cpsr */
#include "ui/metal-board.h"

/*
 * The framebuffer view, cached across frames.  Guest RAM is mapped once
 * per config generation and unmapped when the generation moves; the UI
 * thread reads the mapped bytes directly, with no lock, exactly as a
 * monitor would scan a real framebuffer mid-update.
 */
static struct {
    BCM2835FBState *fb;
    uint32_t generation;
    bool mapped;
    void *fb_ptr;
    void *pal_ptr;
    hwaddr fb_len;
} view;

static bool pi_present(void)
{
    return object_resolve_path_type("", TYPE_BCM2835_FB, NULL) != NULL;
}

static int pi_fb_view(MetalFbView *out)
{
    BCM2835FBConfig cfg;
    uint32_t gen, rows;

    if (!view.fb) {
        Object *obj = object_resolve_path_type("", TYPE_BCM2835_FB, NULL);

        if (!obj) {
            return 0;               /* machine not built yet */
        }
        view.fb = BCM2835_FB(obj);
    }

    gen = bcm2835_fb_get_config(view.fb, &cfg);
    if (!cfg.xres || !cfg.yres) {
        return 0;                   /* no mode programmed yet */
    }

    if (view.mapped && gen != view.generation) {
        address_space_unmap(&view.fb->dma_as, view.fb_ptr, view.fb_len,
                            false, 0);
        address_space_unmap(&view.fb->dma_as, view.pal_ptr, 256 * 4,
                            false, 0);
        view.mapped = false;
    }

    if (!view.mapped) {
        /*
         * Map exactly what the guest indexes: the visible rows plus the pan
         * offset, which the fb model applies only when the virtual size
         * exceeds the physical one (the offsets are meaningless otherwise).
         * The palette sits at the VideoCore RAM base, 256 words of
         * 0x00BBGGRR, where SET_PALETTE writes it.
         */
        bool use_offsets = cfg.xres_virtual > cfg.xres ||
                           cfg.yres_virtual > cfg.yres;
        rows = cfg.yres + (use_offsets ? cfg.yoffset : 0);
        hwaddr fb_len = rows * bcm2835_fb_get_pitch(&cfg);
        hwaddr pal_len = 256 * 4;

        view.fb_ptr = address_space_map(&view.fb->dma_as, cfg.base, &fb_len,
                                        false, MEMTXATTRS_UNSPECIFIED);
        view.pal_ptr = address_space_map(&view.fb->dma_as,
                                         view.fb->vcram_base, &pal_len,
                                         false, MEMTXATTRS_UNSPECIFIED);
        if (!view.fb_ptr || !view.pal_ptr) {
            if (view.fb_ptr) {
                address_space_unmap(&view.fb->dma_as, view.fb_ptr, fb_len,
                                    false, 0);
                view.fb_ptr = NULL;
            }
            return 0;
        }

        view.fb_len = fb_len;
        view.generation = gen;
        view.mapped = true;
        /* Once per mode: is the guest actually running two banks? */
        metal_log("fb: %ux%u virtual %ux%u offset %u,%u base 0x%x pitch %u",
                  cfg.xres, cfg.yres, cfg.xres_virtual, cfg.yres_virtual,
                  cfg.xoffset, cfg.yoffset, cfg.base,
                  bcm2835_fb_get_pitch(&cfg));
    }

    rows = view.fb_len / bcm2835_fb_get_pitch(&cfg);

    out->xres = cfg.xres;
    out->yres = cfg.yres;
    /* Offsets only count when the viewport exceeds the physical screen, as
     * the fb model itself decides; mirror that or the image would shift. */
    if (cfg.xres_virtual > cfg.xres || cfg.yres_virtual > cfg.yres) {
        out->xoffset = cfg.xoffset;
        out->yoffset = cfg.yoffset;
    } else {
        out->xoffset = 0;
        out->yoffset = 0;
    }
    out->pitch = bcm2835_fb_get_pitch(&cfg);
    out->bpp = cfg.bpp;
    out->pixo = cfg.pixo;
    out->rows = rows;
    out->generation = view.generation;
    out->gbase = cfg.base;
    out->fb = view.fb_ptr;
    out->palette = view.pal_ptr;
    return 1;
}

static void pi_fb_done(void)
{
    if (view.mapped) {
        address_space_unmap(&view.fb->dma_as, view.fb_ptr, view.fb_len,
                            false, 0);
        address_space_unmap(&view.fb->dma_as, view.pal_ptr, 256 * 4,
                            false, 0);
        view.mapped = false;
    }
    view.fb = NULL;
}

/* The pointer sprite, answered for by the VCHIQ peer. */
static int pi_cursor_view(MetalCursorView *out)
{
    VchiqCursor cur;

    if (!bcm2835_vchiq_get_cursor(&cur)) {
        return 0;                   /* no machine, or the service is closed */
    }
    out->generation = cur.generation;
    out->stale = cur.stale;
    out->visible = cur.visible;
    out->x = cur.x;
    out->y = cur.y;
    out->w = cur.w;
    out->h = cur.h;
    out->img_w = cur.img_w;
    out->img_h = cur.img_h;
    out->disp_w = cur.disp_w;
    out->disp_h = cur.disp_h;
    out->argb = cur.argb;
    return 1;
}

/*
 * The screendump's bytes, read through the fb device's DMA view the same
 * way the decode shader reads them (offsets only when the viewport
 * exceeds the physical screen; palette words 0x00BBGGRR at the VideoCore
 * base for the palettised modes).  BQL held.
 */
static int pi_capture(MetalCapture *c, const char **errmsg)
{
    static BCM2835FBState *fb;
    BCM2835FBConfig cfg;
    uint32_t pitch, bypp, rowlen, xo, yo;

    if (!fb) {
        Object *obj = object_resolve_path_type("", TYPE_BCM2835_FB, NULL);

        if (!obj) {
            *errmsg = "no framebuffer device";
            return -1;
        }
        fb = BCM2835_FB(obj);
    }
    c->generation = bcm2835_fb_get_config(fb, &cfg);
    if (!cfg.xres || !cfg.yres) {
        *errmsg = "no framebuffer mode yet";
        return -1;
    }
    bypp = (cfg.bpp + 7) / 8;
    pitch = bcm2835_fb_get_pitch(&cfg);
    if (cfg.xres_virtual > cfg.xres || cfg.yres_virtual > cfg.yres) {
        xo = cfg.xoffset;
        yo = cfg.yoffset;
    } else {
        xo = yo = 0;
    }

    c->raw = g_malloc_n(cfg.yres, rowlen = cfg.xres * bypp);
    for (uint32_t y = 0; y < cfg.yres; y++) {
        if (address_space_read(&fb->dma_as,
                               cfg.base + ((uint64_t)(yo + y) * pitch
                                           + (uint64_t)xo * bypp),
                               MEMTXATTRS_UNSPECIFIED,
                               c->raw + (uint64_t)y * rowlen, rowlen)
            != MEMTX_OK) {
            /* A mode change between the config snapshot and the read
             * leaves the buffer holding heap that was never the guest's
             * -- encoding it would put host memory into the PNG. */
            g_free(c->raw);
            c->raw = NULL;
            *errmsg = "framebuffer read failed";
            return -1;
        }
    }
    c->pal = NULL;
    if (cfg.bpp <= 8) {
        c->pal = g_malloc(256 * 4);
        address_space_read(&fb->dma_as, fb->vcram_base,
                           MEMTXATTRS_UNSPECIFIED, c->pal, 256 * 4);
    }
    c->xres = cfg.xres;
    c->yres = cfg.yres;
    c->bpp = cfg.bpp;
    c->pixo = cfg.pixo;
    return 0;
}

static bool pi_set_vsync_hz(int hz)
{
    Object *obj = object_resolve_path_type("", TYPE_BCM2835_VSYNCGEN, NULL);

    if (!obj) {
        return false;
    }
    bcm2835_vsyncgen_set_hz(BCM2835_VSYNCGEN(obj), hz);
    return true;
}

/* PC and registers straight from the ARMCPU, BQL held. */
static char *pi_pc(int64_t core)
{
    CPUState *cpu = qemu_get_cpu(core);
    ARMCPU *ac;
    uint64_t v;

    if (!cpu) {
        return NULL;
    }
    cpu_synchronize_state(cpu);
    ac = ARM_CPU(cpu);
    v = is_a64(&ac->env) ? ac->env.pc : ac->env.regs[15];
    return g_strdup_printf("{\"core\":%d,\"pc\":\"0x%" PRIx64 "\"}",
                           (int)core, v);
}

static char *pi_regs(int64_t core)
{
    CPUState *cpu = qemu_get_cpu(core);
    GString *s;
    ARMCPU *ac;
    uint32_t cpsr;

    if (!cpu) {
        return NULL;
    }
    cpu_synchronize_state(cpu);
    ac = ARM_CPU(cpu);
    cpsr = cpsr_read(&ac->env);

    s = g_string_new(NULL);
    g_string_printf(s, "{\"core\":%d", (int)core);
    for (int r = 0; r <= 12; r++) {
        g_string_append_printf(s, ",\"r%d\":\"0x%" PRIx64 "\"",
                               r, (uint64_t)ac->env.regs[r]);
    }
    g_string_append_printf(s, ",\"sp\":\"0x%" PRIx64 "\",\"lr\":\"0x%"
                           PRIx64 "\"",
                           (uint64_t)ac->env.regs[13],
                           (uint64_t)ac->env.regs[14]);
    g_string_append_printf(s, ",\"pc\":\"0x%" PRIx64 "\"",
                           is_a64(&ac->env) ? ac->env.pc
                                            : ac->env.regs[15]);
    g_string_append_printf(s, ",\"cpsr\":\"0x%x\",\"mode\":\"0x%x\"",
                           (unsigned)cpsr, (unsigned)(cpsr & 0x1f));
    g_string_append(s, "}");
    return g_string_free(s, FALSE);
}

static int pi_sockets(void)
{
    return hostnet_sockets_state();
}

/* Machine > HostNet, done in place: move the guest's HostNet module in or
 * out of the load path, open or close the doorbell to match, and reboot
 * RISC OS -- no app relaunch.  HostModules (Repeat RMLoad HostFS:$.Modules
 * -Type &FFA) scans only the top of Modules/, so a module parked in the
 * Disabled/ subfolder is not loaded and the ROM's own Internet stack (with
 * DHCP) runs; brought back into Modules/, it loads and serves the sockets.
 * The reboot re-runs the boot, which dispatches on the module it now finds. */
static void pi_hostnet_apply(bool on)
{
    const char *root = vmchannel_root();
    bool live = false;

    if (root && *root) {
        char *mod = g_build_filename(root, "Modules", "HostNet,ffa", NULL);
        char *offdir = g_build_filename(root, "Modules", "Disabled", NULL);
        char *off = g_build_filename(offdir, "HostNet,ffa", NULL);

        if (on) {
            if (!g_file_test(mod, G_FILE_TEST_EXISTS) &&
                g_file_test(off, G_FILE_TEST_EXISTS) &&
                g_rename(off, mod) != 0) {
                metal_log("hostnet: could not enable the module: %s",
                          strerror(errno));
            }
        } else if (g_file_test(mod, G_FILE_TEST_EXISTS)) {
            g_mkdir_with_parents(offdir, 0755);
            if (g_rename(mod, off) != 0) {
                metal_log("hostnet: could not disable the module: %s",
                          strerror(errno));
            }
        }
        /* The doorbell follows where the module is now, not what was
         * asked: lit with no module, the ROM's stack would boot beneath a
         * tick saying HostNet; dark with one, HostNet would replace the
         * ROM's Internet module, decline, and leave the machine none. */
        live = g_file_test(mod, G_FILE_TEST_IS_REGULAR);
        g_free(mod);
        g_free(offdir);
        g_free(off);
    } else {
        metal_log("hostnet: no disc root; the module was not moved");
    }

    metal_log("hostnet: switched to %s; rebooting RISC OS",
              live ? "HostNet" : "the RISC OS stack");
    /* Called on the UI thread: the core wants the lock, as input does. */
    bql_lock();
    hostnet_set_sockets(live);
    qemu_system_reset_request(SHUTDOWN_CAUSE_HOST_UI);
    bql_unlock();
}

static const MetalBoard pi_board = {
    .name           = "pi4",
    .system         = "RISC OS 5",
    .machine        = "Raspberry Pi 4 (QEMU)",
    .present        = pi_present,
    .fb_view        = pi_fb_view,
    .fb_done        = pi_fb_done,
    .take_damage    = riscos_blitter_take_damage,
    .cursor_view    = pi_cursor_view,
    .capture        = pi_capture,
    .set_vsync_hz   = pi_set_vsync_hz,
    .pc             = pi_pc,
    .regs           = pi_regs,
    .sockets        = pi_sockets,
    .hostnet_apply  = pi_hostnet_apply,
};

static void metal_pi_register(void)
{
    metal_board_register(&pi_board);
}
type_init(metal_pi_register);

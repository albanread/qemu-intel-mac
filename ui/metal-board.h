/*
 * metal-board.h — what the Metal front end needs from the machine.
 *
 * ui/metal.c is the glue between QEMU and the window (ui/metal.m), and
 * it is the same for every machine.  What differs is where the picture
 * comes from.  On the Pi 4 it is the bcm2835 framebuffer, the VCHIQ
 * peer's pointer sprite and the vsync generator (ui/metal-pi.c).  On a
 * ROSGD box it is the resource a virtio-gpu scanout shows, in guest RAM,
 * with the pointer on its cursor plane (ui/metal-virtio-gpu.c).
 *
 * Each source registers a board from a type_init constructor.  A source
 * is linked into a target only where its devices exist, so the x86_64
 * box never references the Pi's symbols.  At display init, with the
 * machine built, the glue picks the first registered board whose
 * present() finds its device.
 *
 * Every entry but present and fb_view may be NULL: that machine lacks
 * the feature, and the glue says so rather than guessing.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef QEMU_UI_METAL_BOARD_H
#define QEMU_UI_METAL_BOARD_H

#include "ui/metal.h"

/*
 * A screendump's raw material, gathered with the BQL held: the visible
 * screen packed row by row, xres * ((bpp + 7) / 8) bytes per row, and
 * for bpp <= 8 the palette, 256 words of 0x00BBGGRR.  Both g_malloc'd;
 * the caller frees them.
 */
typedef struct MetalCapture {
    uint32_t xres, yres, bpp, pixo;
    uint32_t generation;
    uint8_t *raw;
    uint8_t *pal;
} MetalCapture;

typedef struct MetalBoard {
    const char *name;               /* for metal-debug.txt */
    const char *system;             /* the window title: "RISC OS 5" */
    const char *machine;            /* ... "Raspberry Pi 4 (QEMU)" */

    /* BQL held, machine built: is this board's display device here? */
    bool (*present)(void);

    /* The contract of metal_glue_fb_view / _fb_done (metal.h). */
    int (*fb_view)(MetalFbView *out);
    void (*fb_done)(void);

    /* Drawing since last asked, as metal_glue_take_damage (metal.h).
     * NULL: the machine has no such signal. */
    unsigned (*take_damage)(void);

    /* The contract of metal_glue_cursor_view (metal.h). */
    int (*cursor_view)(MetalCursorView *out);

    /* The contract of metal_glue_surfaces_view (metal.h); NULL: none. */
    int (*surfaces_view)(MetalSurfacesView *out);

    /* BQL held.  0 with *c filled, or -1 with *errmsg a literal. */
    int (*capture)(MetalCapture *c, const char **errmsg);

    /* BQL held: the guest vertical sync rate; false if there is none. */
    bool (*set_vsync_hz)(int hz);

    /* BQL held: one core's PC, or all its registers, as a JSON object;
     * NULL for no such core. */
    char *(*pc)(int64_t core);
    char *(*regs)(int64_t core);

    /* The HostNet doorbell (metal.h): 1 open, 0 closed; and the switch. */
    int (*sockets)(void);
    void (*hostnet_apply)(bool on);
} MetalBoard;

/* From a type_init constructor, before the machine exists. */
void metal_board_register(const MetalBoard *b);

#endif /* QEMU_UI_METAL_BOARD_H */

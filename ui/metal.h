/*
 * metal.h — the boundary between the Cocoa/Metal front end (ui/metal.m)
 * and QEMU's C world (ui/metal.c).  The macOS twin of ui/dx11.h: the
 * same calls, so the two front ends stay recognisably one design.
 *
 * Objective-C is C, so this boundary is not forced on us the way the C++
 * one is on Windows.  It is kept anyway: it is the whole contract, and
 * having it written down once is what made the D3D11 front end reviewable.
 */
#ifndef QEMU_UI_METAL_H
#define QEMU_UI_METAL_H

/* Called from QEMU display init, on the main thread inside qemu_init:
 * creates the NSApplication, the window and the Metal device/layer.
 * Nonzero on failure (reported by the caller). */
int metal_backend_init(void);

/* The UI main loop: Cocoa event pump + render/present at vsync.  Returns
 * after the window closes.  Installed as QEMU's qemu_main, so QEMU's main
 * loop runs on its own thread while this owns the main thread. */
int metal_backend_main(void);

/* Cleanly request the machine to power off (window closed). */
void metal_backend_request_shutdown(void);

/* The front end's log: metal-debug.txt beside the process, line
 * buffered, because a windowed build has no console to complain to. */
void metal_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

/* ------------------------------------------------------------------ */
/* Apple Events (riscos-pi4/SCRIPTING.md)                              */

/* Register the command handlers with NSAppleEventManager, one per
 * event id in the C table.  Called from metal_backend_init() on the
 * main thread and again after the pump's finishLaunching; the handlers
 * fire inside the pump's sendEvent:, which is the first thing Sprint
 * E0 verified.  RISCOSQEMU_SCRIPTING_OFF in the environment skips
 * registration (the dev kill switch until the settings file carries
 * one). */
void metal_script_register(void);

/* One command, one reply envelope out.  The event id names the command
 * -- it is what AppleScript actually addressed, so it is authoritative;
 * a direct parameter that parses as a JSON object carries arguments,
 * and a "cmd" key inside it overrides (the raw-text form).  The reply
 * is a heap string the caller frees with g_free; returns false only
 * when no envelope could be produced at all.  Called on the UI
 * thread. */
bool metal_glue_script(uint32_t event_class, uint32_t event_id,
                       const char *json, char **reply);

/* The four-character event codes of every command in the table, for
 * handler registration: writes up to max {class, id} pairs and returns
 * the count.  NSAppleEventManager dispatches exact pairs, so one
 * handler per command id is required -- a handler for one id does not
 * serve another in the same suite. */
size_t metal_glue_script_events(uint32_t *classes, uint32_t *ids,
                                size_t max);

/*
 * A frame's view of the guest framebuffer, gathered by metal_glue_fb_view()
 * on the UI thread without holding the BQL.  The pointers are host
 * addresses into guest RAM, mapped for as long as the view is current.
 */
typedef struct MetalFbView {
    uint32_t xres, yres;            /* visible size, pixels */
    uint32_t xoffset, yoffset;      /* pan within the buffer */
    uint32_t pitch;                 /* bytes per buffer row */
    uint32_t bpp;                   /* guest pixel format, bits */
    uint32_t pixo;                  /* 1 = RGB order, 0 = BGR */
    uint32_t rows;                  /* mapped buffer rows (>= yres+yoffset) */
    uint32_t generation;            /* config generation of this view */
    uint32_t content;               /* moves when the picture may have changed;
                                       0: the board cannot tell (draw every
                                       frame) -- the power saving's signal */
    uint64_t gbase;                 /* guest address of the buffer */
    const void *fb;                 /* raw framebuffer bytes, pitch * rows */
    const void *palette;            /* 256 * 4 bytes, 0x00BBGGRR; always set */
} MetalFbView;

/*
 * Fill `out` with the current framebuffer view, mapping guest RAM for it.
 * Returns 1 when the view is usable, 0 while there is no framebuffer
 * device yet (before machine init), and the mapping is kept across calls
 * until the config generation moves.  Thread: UI only.
 */
int metal_glue_fb_view(MetalFbView *out);

/*
 * Has anything been drawn since this was last asked?  1 or 0, taken and
 * cleared -- on the Pi the blitter raises it on the vCPU thread.  -1 when
 * the machine has no such signal, and every frame is to be copied.
 */
int metal_glue_take_damage(void);

/* The window title's two halves (ui/metal-board.h): the system, such as
 * "RISC OS 5", and the machine, such as "Raspberry Pi 4 (QEMU)". */
void metal_glue_title(const char **system, const char **machine);

/* Drop any cached mapping (called once from the UI loop at exit). */
void metal_glue_fb_done(void);

/*
 * The guest's pointer sprite.  The VCHIQ peer answers the dispmanx
 * requests the ROM sends for its hardware pointer, so the sprite
 * exists only on the host: never written to guest RAM, never torn,
 * committed atomically at UpdateSubmit the way the firmware would at
 * vsync.  The image words are little-endian 0xAARRGGBB.
 */
typedef struct MetalCursorView {
    uint32_t generation;            /* commit generation of this view */
    bool stale;                     /* raced a commit: keep the last one */
    bool visible;
    int32_t x, y;                   /* dest rect, display pixels, top-left */
    int32_t w, h;                   /* dest rect size, display pixels */
    int32_t img_w, img_h;           /* the sprite's own resolution, texels */
    int32_t disp_w, disp_h;         /* the display the rect is measured in */
    int32_t hot_x, hot_y;           /* the hot spot in the rect, display pixels */
    bool has_hot;                   /* hot_x/hot_y are known: the pointer is
                                       at x + hot_x, y + hot_y */
    const void *argb;               /* img_w * img_h words */
} MetalCursorView;

/*
 * Fill `out` with the current pointer sprite, gathered the same
 * lock-free way as the framebuffer view.  Returns 0 only when there is
 * no sprite at all (no machine yet, service closed); a `stale` view is
 * reported so the caller keeps the previous frame's.  Thread: UI only.
 */
int metal_glue_cursor_view(MetalCursorView *out);

/*
 * The surfaces the guest publishes for the host to composite (the ROSGD
 * box's block v2, design 39): each window's backing, its pixels where
 * they lie in the region of guest RAM the board has mapped once, drawn
 * back to front over the backdrop, clipped to its visible rectangles,
 * each after its shadow.  When there is a list the screen itself (the
 * framebuffer view) is not drawn: everything shown is a surface.
 */
#define METAL_SURFACES      128
#define METAL_SURF_RECTS    2048
typedef struct MetalSurface {
    int16_t x, y, w, h;             /* on the screen, guest pixels, y down */
    uint32_t offset, stride;        /* its pixels in the region: R G B A bytes,
                                       premultiplied, alpha &80 a hole */
    uint32_t rects, nrects;         /* visible: a range of the view's rect pool */
    uint32_t shadow_rects, shadow_nrects;   /* where its shadow shows */
    int16_t shadow_dx, shadow_dy, shadow_blur, shadow_strength;
    uint16_t radius;                /* its corners, guest pixels: the shadow's */
} MetalSurface;
typedef struct MetalSurfacesView {
    uint32_t frame;                 /* the list's frame: moves when anything does */
    void *region;                   /* the host's mapping of the region */
    uint64_t region_size;
    uint32_t look;                  /* the depth the desktop asked for (screen.h) */
    uint32_t n;
    MetalSurface s[METAL_SURFACES];
    uint32_t nrects;
    int16_t rect[METAL_SURF_RECTS][4];  /* x0 y0 x1 y1, half-open */
} MetalSurfacesView;

/*
 * Fill `out` with the guest's current list, taken whole under its
 * seqlock (a torn read keeps the last good one).  Returns 1 when there
 * is a list to composite, 0 when the screen is to be drawn as it is.
 * Thread: UI only, after metal_glue_fb_view.
 */
int metal_glue_surfaces_view(MetalSurfacesView *out);

/*
 * Input, called from the UI thread's event handlers.  Each call takes the
 * BQL just long enough to run the QEMU input API, the way cocoa's
 * with_bql does; nothing else of QEMU is touched.
 */
/* An NSEvent keyCode: a macOS virtual key code, which is what the osx
 * keymap is indexed by.  Auto-repeats are dropped by the caller. */
void metal_glue_key(bool down, uint32_t oskeycode);
/* Guest pointer position in guest pixels on a xres x yres screen: sent
 * absolutely, the way the guest's tablet driver maps onto the screen. */
void metal_glue_mouse_abs(int gx, int gy, int xres, int yres);
void metal_glue_mouse_btn(int button, bool down); /* 0 left, 1 middle, 2 right */
void metal_glue_mouse_wheel(int notches);         /* positive = away from user */
/* Keyboard grab.  On Windows this arms a low-level hook that swallows
 * Alt+Tab; macOS has no equivalent that an unsigned, unbundled process
 * may install, so this only records the state for the UI's own use --
 * Command chords keep reaching the host, which is the macOS convention
 * anyway. */
void metal_glue_grab(bool on);

/* Guest vertical sync rate, pulses per second, 0 for none.  The machine's
 * vsync generator delivers them from the timer thread; false if the
 * machine has no generator. */
bool metal_glue_set_vsync_hz(int hz);

/* Scaler and CRT options for the window: scaling is 0 linear, 1 sharp
 * bilinear, 2 nearest.  Set at display init and by the scripting
 * surface's video command; a change rebuilds the pipelines on the
 * next frame. */
void metal_glue_video_opts(int scaling, int scanlines);

/* The layer beneath the guest's desktop, gated: "off" (the default) leaves
 * the feature and its Backdrop menu out; "none", "acorn", "acorn-live",
 * "shader:<file>", "tile:<file>", "picture:<file>" or "wallpaper:<file>"
 * turn it on with that scene (MACOS.md, "The backdrop layer").  Once,
 * after metal_backend_init; a file that cannot be read or compiled gives
 * none. */
void metal_glue_backdrop(const char *spec);
/* The folder of wallpapers (sprite files, images) and, in its Shaders,
 * scenes: the Backdrop menu lists it and a relative file is in it.
 * Before metal_glue_backdrop. */
void metal_glue_backdrop_dir(const char *dir);
/* The window's fit to a mode: screen pixels to a guest pixel, 1 to 4 */
void metal_glue_zoom(int zoom);

/* UI-side reads for the scripting surface, all UI-thread only like the
 * rest of this side of the boundary. */
/* Presented frames since the window opened; monotonic per boot. */
uint32_t metal_ui_frame_count(void);
/* The video options as currently applied (0/1/2, scanlines). */
void metal_ui_video_opts(int *scaling, bool *scanlines);
/* Write the decoded Metal surface (the guest mode after palette/
 * format decode, before scaling) to a PNG at path. */
bool metal_ui_screenshot(const char *path);

/* Reload the named-by-convention snapshot ("desktop") from the Machine
 * menu; the load itself runs as a bottom half on the main loop. */
void metal_glue_load_snapshot(void);

/* The HostNet mode, Machine > HostNet: which network RISC OS runs —
 * hostnet (the Mac serves the guest's sockets through the doorbell) or
 * the RISC OS stack over the emulated card.  metal_glue_sockets() reads
 * the doorbell (1 open, 0 closed, -1 no device) for the menu's tick.
 * metal_ui_hostnet_switch() records the choice (so the next app launch
 * comes up in it) and calls metal_glue_hostnet_apply(), which moves the
 * guest's module in/out of the load path, flips the doorbell to match and
 * reboots RISC OS in place — no app relaunch, one stack or the other. */
int metal_glue_sockets(void);
void metal_glue_hostnet_apply(bool on);
void metal_ui_hostnet_switch(bool on);

#endif /* QEMU_UI_METAL_H */

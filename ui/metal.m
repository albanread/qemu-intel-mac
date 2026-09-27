/*
 * metal.m — the native macOS front end for the emulated Pi 4.
 *
 * The macOS twin of ui/dx11.cpp, and deliberately the same program: each
 * frame the UI thread pulls the fb view over ui/metal.h's boundary
 * (config snapshot plus host pointers into guest RAM, no locks), uploads
 * the raw bytes to a shared buffer, decodes them to linear RGB with a
 * per-format fragment shader, and stretches the decoded surface over the
 * whole content view — the window is the monitor, so every mode fills it,
 * as the Pi's GPU fills a real display.  8 bpp palette and 32 bpp are the
 * two formats RISC OS's BCMVideo ever asks for; the rest of the decoder
 * set is the same source with different constants.
 *
 * Where Windows compiles HLSL at start-up with d3dcompiler, this compiles
 * MSL at start-up with newLibraryWithSource:, and specialises the decode
 * and scale passes with Metal function constants rather than preprocessor
 * macros — the same "one source, several shaders" shape, in the form the
 * platform offers.
 *
 * This file includes no QEMU headers and talks to the emulation only
 * through ui/metal.h.  No ARC: QEMU does not build Objective-C with it.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <ImageIO/ImageIO.h>
#import <CoreServices/CoreServices.h>

#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ui/metal.h"

/* ------------------------------------------------------------------ */
/* State, owned by the UI thread only                                  */

#define METAL_RING      3               /* frames of upload buffer in flight */

@class MetalView;

static struct {
    NSWindow *window;
    MetalView *view;
    CAMetalLayer *layer;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLSamplerState> linear;
    id<MTLSamplerState> repeat;         /* backdrop tiles */
    dispatch_semaphore_t inflight;
    bool ready;
    bool lost;
    /* The content size we last chose ourselves. If the window still has
     * it, the user has not taken over and a mode change may resize it. */
    NSSize auto_size;
    /* input forwarded, for the debug log */
    unsigned keys;
    unsigned mouse_moves;
    unsigned mouse_buttons;
} m;

static uint32_t frame_count;

/* -display metal options, set once at init: how the guest screen is
 * magnified to the content view (sharp-bilinear by default) and whether
 * an optional CRT scanline mask rides on top at 2x+ magnification. */
enum MetalScaling { METAL_SCALING_LINEAR = 0, METAL_SCALING_SHARP,
                    METAL_SCALING_NEAREST };
static struct {
    int scaling;                        /* MetalScaling */
    bool scanlines;
} video_opts = { METAL_SCALING_SHARP, false };

/* -display metal,backdrop=: a layer the host draws beneath the guest's
 * desktop (MACOS.md, "The backdrop layer").  The guest opts pixels in
 * through the transfer byte of a 32bpp pixel -- RISC OS's supremacy
 * byte, which everything the OS draws leaves at 0:
 *
 *   bits 7-6  the layer tag: 10 below (the layer beneath shows through),
 *             01 above (reserved: a layer drawn over the pixel), 00 and
 *             11 nothing -- so the &FF some sprite tools write means
 *             nothing
 *   bits 5-0  reserved, 0
 *
 * A below pixel must also be the backdrop key colour, the Acorn theme's
 * sage, so an EOR drag box drawn over the backdrop changes the colour,
 * drops out of the key and stays visible.  The guest paints the tag by
 * tiling a sprite whose pixels carry it (tools/mkbacktile.py): the
 * kernel's sprite plot and the Wimp's block copies keep all 32 bits. */
enum MetalBackdrop { METAL_BACKDROP_OFF = 0, METAL_BACKDROP_ACORN,
                     METAL_BACKDROP_ACORN_LIVE, METAL_BACKDROP_PICTURE,
                     METAL_BACKDROP_TILE };
#define METAL_BACKDROP_KEY   0x00B4C0B7u  /* 0x00BBGGRR: #B7C0B4 */
#define METAL_ICONBAR_PX     66u          /* guest px, square-pixel modes */
static struct {
    bool enabled;                       /* the gate: backdrop= other than off */
    int mode;                           /* MetalBackdrop: the scene, OFF for none */
    id<MTLTexture> image;               /* PICTURE and TILE */
    double image_scale;                 /* 2 for an @2x tile, else 1 */
    NSString *spec;                     /* as given: the menu ticks it */
    CFAbsoluteTime t0;                  /* acorn-live's clock */
    NSMenuItem *menu_item;              /* Machine > Backdrop, hidden when gated off */
} backdrop = { false, METAL_BACKDROP_OFF, nil, 1.0, nil, 0, nil };

/* Machine > HostNet, the doorbell switch: shown when this machine has
 * the device, ticked when the doorbell is open.  Kept, not released,
 * like the Backdrop item: menuNeedsUpdate reaches for it on every open. */
static NSMenuItem *hostnet_item;

/* The per-mode pipeline: everything that depends on the fb config. */
static struct {
    bool up;
    uint32_t generation;
    uint32_t bpp;
    uint32_t xres, yres, pitch, rows;

    id<MTLBuffer> raw[METAL_RING];      /* guest bytes, pitch * rows */
    unsigned ring;
    bool uploaded;                      /* a ring entry holds a real frame */
    id<MTLBuffer> palette;              /* 256 * RGBA8 */
    uint8_t pal_cache[256 * 4];
    bool pal_valid;

    id<MTLTexture> decoded;             /* linear RGB, xres x yres */
    id<MTLRenderPipelineState> decode;  /* specialised for this bpp */
    id<MTLRenderPipelineState> scale;   /* specialised for the options */
    id<MTLRenderPipelineState> shot;    /* the scale pass into RGBA8, 1:1 */
    uint32_t xoff, yoff, pixo;          /* the view's pan and byte order */
} fb;

static bool fb_failed;
static uint32_t fb_failed_generation;

/* The scale pass is specialised on the options and the backdrop, so a
 * change releases it and the next frame builds it again.  (Until 14 Sep
 * a change only dropped the per-mode pipeline, and the scale pass built
 * at init -- before -display's own options were read -- was never
 * rebuilt, so scaling= and scanlines= did not reach the shader.) */
static void fb_release_scale(void)
{
    [fb.scale release];
    fb.scale = nil;
    [fb.shot release];
    fb.shot = nil;
}

/* The scale pipeline is specialised on the options, so a runtime
 * change drops the pipeline and the next frame rebuilds it -- one
 * frame at worst of the old look, which is also what a mode change
 * costs.  Invalid scaling is ignored, as at init. */
void metal_glue_video_opts(int scaling, int scanlines)
{
    int s = video_opts.scaling;
    bool sl = scanlines != 0;

    if (scaling >= METAL_SCALING_LINEAR && scaling <= METAL_SCALING_NEAREST) {
        s = scaling;
    }
    if (s != video_opts.scaling || sl != video_opts.scanlines) {
        video_opts.scaling = s;
        video_opts.scanlines = sl;
        fb_release_scale();
    }
}

/* UI-side reads for the scripting surface (ui/metal.c's state and
 * video commands); UI thread only, like everything in this file. */
uint32_t metal_ui_frame_count(void)
{
    return frame_count;
}

void metal_ui_video_opts(int *scaling, bool *scanlines)
{
    *scaling = video_opts.scaling;
    *scanlines = video_opts.scanlines;
}

/* The pointer sprite: a pipeline and an image that do not depend on
 * the guest's mode, built once beside the scale pass.  The VCHIQ peer
 * answers the dispmanx requests the ROM sends for its hardware
 * pointer, so the sprite is composited here on top of the scaled
 * frame -- never written to guest RAM, sharp at any window scale, and
 * without the scanline mask: on a real Acorn machine the sprite lay
 * over the CRT, it was not part of its raster. */
#define METAL_CURSOR_TEXELS 64     /* words per side, as the peer defines */
static struct {
    id<MTLRenderPipelineState> pipe;
    id<MTLBuffer> image;            /* METAL_CURSOR_TEXELS^2 ARGB words */
    bool visible;
    uint32_t generation;
    int32_t x, y, w, h;             /* dest rect, display pixels */
    int32_t img_w, img_h;           /* the sprite's own resolution */
    int32_t disp_w, disp_h;         /* the display the rect is measured in */
} ptr;

/* The shader constants, mirroring the Params struct in the MSL below. */
typedef struct {
    uint32_t dim[4];    /* xres, yres, pitch(bytes), bpp */
    uint32_t misc[4];   /* xoffset(px), yoffset(rows), pixo, unused */
    uint32_t post[4];   /* view w, view h, scaling, scanlines */
    uint32_t bd[4];     /* backdrop: key 0x00BBGGRR, time ms, icon bar px, on */
    uint32_t bx[4];     /* backdrop: output px per tile px x1000, unused */
} MetalParams;

/* The front end's log: a file next to the process, because a windowed
 * build has no console to complain to.  METAL_DEBUG=1 also turns on
 * Metal's own validation, which the environment reads at device
 * creation.  Not static: ui/metal_script.m logs here too. */

void metal_log(const char *fmt, ...)
{
    static FILE *f;
    va_list ap;

    if (!f) {
        f = fopen("metal-debug.txt", "w");
        if (!f) {
            return;
        }
        setvbuf(f, NULL, _IOLBF, 0);
    }
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
}

static void metal_screenshot(void);

/* METAL_DEBUG=1 traces the events the pump sees but the guest does not,
 * which is the only way to tell "the app never got the keystroke" from
 * "the view never got it". */
static bool metal_debug(void)
{
    static int on = -1;

    if (on < 0) {
        const char *e = getenv("METAL_DEBUG");
        on = e && *e && *e != '0';
    }
    return on != 0;
}

/* ------------------------------------------------------------------ */
/* Grab: the guest owns the pointer until Ctrl+Alt+G                   */

static void metal_mouse_centre(void);
static void metal_mouse_send_abs(int gx, int gy);

static struct {
    bool on;
} kbd;

/* The host cursor is hidden whenever the guest is drawing the pointer
 * under it -- which is any time it is over the guest's screen in a key
 * window, not just while grabbed.  The tablet keeps the two arrows in
 * the same place, so showing both is showing the same pointer twice.
 * [NSCursor hide] is counted, so it is called only on a change. */
static struct {
    bool hidden;
    bool inside;
} cursor;

static void metal_cursor_sync(void)
{
    bool want = kbd.on || (cursor.inside && [m.window isKeyWindow]);

    if (want == cursor.hidden) {
        return;
    }
    cursor.hidden = want;
    if (want) {
        [NSCursor hide];
    } else {
        [NSCursor unhide];
    }
}

static struct {
    bool swallow_up;                /* the middle-up of a grab click */
    int held_left, held_right;      /* button actually sent, per stream */
    bool down[3];                   /* what the guest believes is pressed */
    /* the guest pointer position we last sent, in guest pixels: the
     * anchor for relative motion while grabbed (absolute while not) */
    int gx, gy;
    bool have_guest;
    int gxres, gyres;               /* guest screen size, from the fb view */
} mouse;

/* A point in the view's coordinates, put on the screen in Quartz's:
 * Cocoa measures up from the bottom of the main screen, CoreGraphics
 * down from its top, and the warp is a CoreGraphics call. */
static CGPoint metal_view_point_to_cg(NSPoint p)
{
    NSPoint win = [m.view convertPoint:p toView:nil];
    NSRect r = [m.window convertRectToScreen:NSMakeRect(win.x, win.y, 0, 0)];
    CGFloat top = NSMaxY([[[NSScreen screens] objectAtIndex:0] frame]);

    return CGPointMake(r.origin.x, top - r.origin.y);
}

static void metal_mouse_centre(void)
{
    NSRect b = [m.view bounds];
    CGPoint c = metal_view_point_to_cg(NSMakePoint(NSMidX(b), NSMidY(b)));

    CGWarpMouseCursorPosition(c);
}

static void metal_set_grab(bool on)
{
    if (on == kbd.on) {
        return;
    }
    kbd.on = on;
    metal_glue_grab(on);
    if (on) {
        if (!mouse.have_guest && mouse.gxres > 1) {
            mouse.gx = mouse.gxres / 2;
            mouse.gy = mouse.gyres / 2;
            mouse.have_guest = true;
        }
        metal_mouse_centre();
        /* Untie the cursor from the mouse: the deltas keep coming, the
         * arrow stays where it was warped, and motion is unbounded. */
        CGAssociateMouseAndMouseCursorPosition(false);
    } else {
        /* Put the host cursor where the guest arrow is, so the two
         * coincide again the moment the pointer is handed back. */
        if (mouse.have_guest && mouse.gxres > 1) {
            NSRect b = [m.view bounds];
            CGFloat x = (CGFloat)mouse.gx * NSWidth(b) / (mouse.gxres - 1);
            CGFloat y = NSHeight(b)
                        - (CGFloat)mouse.gy * NSHeight(b) / (mouse.gyres - 1);
            CGWarpMouseCursorPosition(
                metal_view_point_to_cg(NSMakePoint(x, y)));
        }
        CGAssociateMouseAndMouseCursorPosition(true);
    }
    metal_cursor_sync();
    metal_log("grab %s", on ? "on" : "off");
}

/* ------------------------------------------------------------------ */
/* Mouse.  The guest device is an absolute tablet, so the position is   */
/* always sent as a coordinate: ungrabbed that is the host cursor's     */
/* place in the view scaled to the guest screen (the two arrows never   */
/* diverge); grabbed the cursor is dissociated and the deltas           */
/* accumulate into a virtual position sent absolutely -- it cannot      */
/* drift.                                                              */

static void metal_mouse_send_abs(int gx, int gy)
{
    if (mouse.gxres > 1 && mouse.gyres > 1) {
        if (gx < 0) {
            gx = 0;
        }
        if (gy < 0) {
            gy = 0;
        }
        if (gx > mouse.gxres - 1) {
            gx = mouse.gxres - 1;
        }
        if (gy > mouse.gyres - 1) {
            gy = mouse.gyres - 1;
        }
        if (metal_debug()) {
            metal_log("mouse: sent %d,%d of %dx%d", gx, gy,
                      mouse.gxres, mouse.gyres);
        }
        metal_glue_mouse_abs(gx, gy, mouse.gxres, mouse.gyres);
        mouse.gx = gx;
        mouse.gy = gy;
        mouse.have_guest = true;
        m.mouse_moves++;
    }
}

static void metal_mouse_move(NSEvent *e)
{
    NSRect b;
    NSPoint p;

    if (mouse.gxres < 2 || mouse.gyres < 2) {
        return;                     /* no guest screen to map onto yet */
    }
    b = [m.view bounds];
    if (NSWidth(b) < 1 || NSHeight(b) < 1) {
        return;
    }
    if (!kbd.on) {
        /* Ungrabbed: host cursor and guest arrow coincide.  Cocoa
         * measures up from the bottom of the view; the guest counts
         * rows down from the top. */
        p = [m.view convertPoint:[e locationInWindow] fromView:nil];
        metal_mouse_send_abs(
            (int)(p.x * (mouse.gxres - 1) / (NSWidth(b) - 1)),
            (int)((NSHeight(b) - p.y) * (mouse.gyres - 1) / (NSHeight(b) - 1)));
        return;
    }
    /* Grabbed: the dissociated cursor still reports deltas.  Scale them
     * by guest-per-view pixels, so a window larger than the guest screen
     * does not run the pointer into the edges at double speed. */
    {
        CGFloat dx = [e deltaX], dy = [e deltaY];

        if (dx != 0 || dy != 0) {
            int gdx = (int)(dx * mouse.gxres / NSWidth(b));
            int gdy = (int)(dy * mouse.gyres / NSHeight(b));

            if (gdx == 0 && dx != 0) {
                gdx = dx > 0 ? 1 : -1;      /* never lose a slow move */
            }
            if (gdy == 0 && dy != 0) {
                gdy = dy > 0 ? 1 : -1;
            }
            metal_mouse_send_abs(mouse.gx + gdx, mouse.gy + gdy);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Keyboard.  NSEvent keyCodes are macOS virtual key codes, which the   */
/* osx keymap is indexed by; the modifiers arrive as a flags word and   */
/* are turned back into presses and releases here.                      */

/* The device-dependent modifier bits, which is the only way to tell a
 * left Shift from a right one.  <IOKit/hidsystem/IOLLEvent.h> has them;
 * naming them here keeps the include list to frameworks we already use. */
#define METAL_LCTRL     0x00000001
#define METAL_LSHIFT    0x00000002
#define METAL_RSHIFT    0x00000004
#define METAL_LCMD      0x00000008
#define METAL_RCMD      0x00000010
#define METAL_LALT      0x00000020
#define METAL_RALT      0x00000040
#define METAL_RCTRL     0x00002000

static const struct { uint32_t keycode; uint32_t mask; } metal_mods[] = {
    { 59, METAL_LCTRL  },  { 62, METAL_RCTRL  },
    { 56, METAL_LSHIFT },  { 60, METAL_RSHIFT },
    { 58, METAL_LALT   },  { 61, METAL_RALT   },
    { 55, METAL_LCMD   },  { 54, METAL_RCMD   },
};
#define METAL_SHIFT_ANY (METAL_LSHIFT | METAL_RSHIFT)
#define METAL_CTRL_ANY  (METAL_LCTRL  | METAL_RCTRL)
#define METAL_CMD_ANY   (METAL_LCMD   | METAL_RCMD)
#define METAL_ALT_ANY   (METAL_LALT   | METAL_RALT)

static uint32_t mod_state;              /* the masks we believe are down */
/*
 * Masks spent on a mouse button and therefore hidden from the guest. A
 * modifier used to pick a button must not also arrive as a key: RISC OS
 * reads Shift-Adjust and Adjust as different gestures, so leaking the
 * Shift makes the click land as something else entirely -- which is
 * exactly how Shift-click came to do nothing while Ctrl-click worked.
 */
static uint32_t mod_masked;

static void metal_flags_changed(NSEvent *e)
{
    NSUInteger flags = [e modifierFlags];
    size_t i;

    for (i = 0; i < sizeof(metal_mods) / sizeof(metal_mods[0]); i++) {
        bool now = (flags & metal_mods[i].mask) != 0;
        bool was = (mod_state & metal_mods[i].mask) != 0;

        if (mod_masked & metal_mods[i].mask) {
            continue;               /* it is a mouse button just now */
        }
        if (now != was) {
            metal_glue_key(now, metal_mods[i].keycode);
            if (now) {
                mod_state |= metal_mods[i].mask;
            } else {
                mod_state &= ~metal_mods[i].mask;
            }
        }
    }
    /* Caps Lock latches on the host, so the guest never sees a release
     * unless we make one: press and release on each change of state. */
    if ([e keyCode] == 57) {
        metal_glue_key(true, 57);
        metal_glue_key(false, 57);
    }
}

/* Take a modifier away from the guest for the duration of a click */
static void metal_mods_hold(uint32_t masks)
{
    size_t i;

    for (i = 0; i < sizeof(metal_mods) / sizeof(metal_mods[0]); i++) {
        uint32_t m = metal_mods[i].mask;

        if ((masks & m) && (mod_state & m)) {
            metal_glue_key(false, metal_mods[i].keycode);
            mod_state &= ~m;
            mod_masked |= m;
        }
    }
}

/* And give it back, if it is still physically down when the button is up */
static void metal_mods_restore(NSEvent *e)
{
    NSUInteger flags = [e modifierFlags];
    size_t i;

    if (!mod_masked) {
        return;
    }
    for (i = 0; i < sizeof(metal_mods) / sizeof(metal_mods[0]); i++) {
        uint32_t m = metal_mods[i].mask;

        if ((mod_masked & m) && (flags & m)) {
            metal_glue_key(true, metal_mods[i].keycode);
            mod_state |= m;
        }
    }
    mod_masked = 0;
}

/* Every modifier the guest thinks is down, released.  Called when the
 * window loses focus, so a Command-Tab away does not leave RISC OS
 * holding a phantom Alt down for ever. */
static void metal_release_modifiers(void)
{
    size_t i;

    for (i = 0; i < sizeof(metal_mods) / sizeof(metal_mods[0]); i++) {
        if (mod_state & metal_mods[i].mask) {
            metal_glue_key(false, metal_mods[i].keycode);
        }
    }
    mod_state = 0;
}

/* ------------------------------------------------------------------ */
/* The shaders.  One source, compiled once; the decode pass is          */
/* specialised on the guest's bits per pixel and the scale pass on the  */
/* -display options, with function constants.                           */

static NSString * const SHADER_SRC = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constant uint BPP       [[function_constant(0)]];\n"
"constant uint SCALING   [[function_constant(1)]];\n"
"constant bool SCANLINES [[function_constant(2)]];\n"
"constant uint BACKDROP  [[function_constant(3)]];\n"
"constant bool BACKDROP_IMAGE = (BACKDROP >= 3);\n"
"\n"
"struct Params {\n"
"    uint4 dim;     /* xres, yres, pitch(bytes), bpp */\n"
"    uint4 misc;    /* xoffset(px), yoffset(rows), pixo, unused */\n"
"    uint4 post;    /* view w, view h, scaling, scanlines */\n"
"    uint4 bd;      /* backdrop: key 0x00BBGGRR, time ms, icon bar px, on */\n"
"    uint4 bx;      /* backdrop: output px per tile px x1000, unused */\n"
"};\n"
"\n"
"struct VSOut {\n"
"    float4 pos [[position]];\n"
"    float2 uv;\n"
"};\n"
"\n"
"vertex VSOut vs_main(uint id [[vertex_id]])\n"
"{\n"
"    /* one big triangle covering the target; uv happens to be the same\n"
"     * (0..2) coordinates the position is built from, so the interpolated\n"
"     * uv runs 0..1 across the visible area with y=0 at the top */\n"
"    VSOut o;\n"
"    float2 t = float2((id << 1) & 2, id & 2);\n"
"    o.pos = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);\n"
"    o.uv = t;\n"
"    return o;\n"
"}\n"
"\n"
"fragment float4 ps_decode(VSOut v [[stage_in]],\n"
"                          constant Params &P [[buffer(0)]],\n"
"                          device const uchar *raw [[buffer(1)]],\n"
"                          device const uchar4 *pal [[buffer(2)]])\n"
"{\n"
"    /* visible pixel -> buffer coordinates, pan included */\n"
"    uint bx = uint(v.pos.x) + P.misc.x;\n"
"    uint by = uint(v.pos.y) + P.misc.y;\n"
"    uint row = by * P.dim.z;\n"
"    uint r = 0, g = 0, b = 0;\n"
"\n"
"    if (BPP == 32) {\n"
"        uint tag;\n"
"        r = raw[row + bx * 4 + 0];\n"
"        g = raw[row + bx * 4 + 1];\n"
"        b = raw[row + bx * 4 + 2];\n"
"        tag = raw[row + bx * 4 + 3];\n"
"        if (P.misc.z == 0) { uint t = r; r = b; b = t; }   /* BGR order */\n"
"        /* The transfer byte's layer tag, bits 7-6: 10 is below, and a\n"
"         * below pixel in the backdrop key colour is a hole the layer\n"
"         * beneath shows through -- transparent, premultiplied. */\n"
"        if (P.bd.w != 0 && (tag & 0xC0) == 0x80\n"
"            && (r | (g << 8) | (b << 16)) == P.bd.x) {\n"
"            return float4(0.0);\n"
"        }\n"
"    } else if (BPP == 24) {\n"
"        r = raw[row + bx * 3 + 0];\n"
"        g = raw[row + bx * 3 + 1];\n"
"        b = raw[row + bx * 3 + 2];\n"
"    } else if (BPP == 16) {\n"
"        uint w = uint(raw[row + bx * 2])\n"
"               | (uint(raw[row + bx * 2 + 1]) << 8);       /* RGB565 */\n"
"        r = ((w >> 11) & 31) * 255 / 31;\n"
"        g = ((w >>  5) & 63) * 255 / 63;\n"
"        b = ((w      ) & 31) * 255 / 31;\n"
"    } else {\n"
"        /* palettised: 8 bpp, or sub-byte 1/2/4 with LSB-first packing */\n"
"        uint bpp = P.dim.w;\n"
"        uint byte = raw[row + bx * bpp / 8];\n"
"        uint idx = (byte >> ((bx * bpp) & 7)) & ((1u << bpp) - 1);\n"
"        uchar4 c = pal[idx];\n"
"        /* 0x00BBGGRR words, so as bytes: red, green, blue, unused */\n"
"        return float4(float(c.x), float(c.y), float(c.z), 255.0) / 255.0;\n"
"    }\n"
"    return float4(float(r), float(g), float(b), 255.0) / 255.0;\n"
"}\n"
"\n"
"/* The Acorn mark (design/art/acorn.svg) as a signed distance, in the\n"
" * SVG's own units about its centre, y down, positive inside.  The mark\n"
" * is the union of the cap, a quadratic dome, and the nut, a cubic egg,\n"
" * grown by the half-width of the outline stroke and clipped to the\n"
" * SVG's 34-unit view box -- which is why the watermark the pinboard\n"
" * draws has a flat top and a flat bottom.  Distances to the curved\n"
" * sides are horizontal offsets corrected by the curve's slope. */\n"
"static float acorn_distance(float2 q)\n"
"{\n"
"    const float grow = 0.8;\n"
"    float ax = abs(q.x);\n"
"\n"
"    /* Nothing of the mark reaches past 16 units across or 17 down, so a\n"
"     * pixel a unit clear of that box is plainly outside: say so, and skip\n"
"     * the curve solve below, which is most of the scene's cost.  -outside\n"
"     * is at least a unit, so coverage is 0 for any window taller than 50\n"
"     * pixels, as the exact distance would give. */\n"
"    float outside = max(ax - 16.0, abs(q.y) - 17.0);\n"
"    if (outside > 1.0) {\n"
"        return -outside;\n"
"    }\n"
"\n"
"    /* cap: right side (0,-20) Q (16,-20) (15,-7): x = 32t - 17t^2, y = -20 + 13t^2 */\n"
"    float cy = clamp(q.y, -20.0, -7.0);\n"
"    float ct = sqrt((cy + 20.0) / 13.0);\n"
"    float cxb = 32.0 * ct - 17.0 * ct * ct;\n"
"    float ck = (32.0 - 34.0 * ct) / max(26.0 * ct, 1e-3);\n"
"    float cap = min((cxb - ax) / sqrt(1.0 + ck * ck), -7.0 - q.y);\n"
"\n"
"    /* nut: right side (0,18) C (9,18) (15,7) (13,-7); y falls as t rises */\n"
"    float ny = clamp(q.y, -7.0, 18.0);\n"
"    float lo = 0.0, hi = 1.0;\n"
"    for (int i = 0; i < 18; i++) {\n"
"        float t = 0.5 * (lo + hi), u = 1.0 - t;\n"
"        float by = u * u * u * 18.0 + 3.0 * u * u * t * 18.0\n"
"                 + 3.0 * u * t * t * 7.0 - t * t * t * 7.0;\n"
"        if (by > ny) { lo = t; } else { hi = t; }\n"
"    }\n"
"    float t = 0.5 * (lo + hi), u = 1.0 - t;\n"
"    float nxb = 3.0 * u * u * t * 9.0 + 3.0 * u * t * t * 15.0 + t * t * t * 13.0;\n"
"    float ndx = 3.0 * (u * u * 9.0 + 2.0 * u * t * 6.0 - t * t * 2.0);\n"
"    float ndy = 3.0 * (2.0 * u * t * -11.0 - t * t * 14.0);\n"
"    float nk = ndx / min(ndy, -1e-3);\n"
"    float nut = min(min((nxb - ax) / sqrt(1.0 + nk * nk), 18.0 - q.y), q.y + 7.0);\n"
"\n"
"    return min(max(cap, nut) + grow, 17.0 - abs(q.y));\n"
"}\n"
"\n"
"/* The layer beneath the desktop, in the same gamma-encoded space as the\n"
" * guest's pixels.  px is the output pixel, outPx the view, srcPx the\n"
" * guest screen it stands for. */\n"
"static float3 backdrop_scene(float2 px, float2 outPx, float2 srcPx,\n"
"                             constant Params &P)\n"
"{\n"
"    const float3 ground = float3(183.0, 192.0, 180.0) / 255.0;   /* #B7C0B4 */\n"
"    const float3 ghost  = float3(173.0, 183.0, 168.0) / 255.0;   /* #ADB7A8 */\n"
"    /* view px per guest px, per axis: a window the user has resized\n"
"     * away from the mode's shape stretches the desktop unevenly */\n"
"    float2 k = outPx / srcPx;\n"
"    /* 220 guest px for the SVG's 34 units, centred above the icon bar,\n"
"     * exactly where *Backdrop -Centre put the watermark sprite; the\n"
"     * centre follows the stretch, the mark keeps its shape */\n"
"    const float unitPx = 220.0 / 34.0;\n"
"    float2 centre = float2(srcPx.x * 0.5, (srcPx.y - float(P.bd.z)) * 0.5) * k;\n"
"    float d = acorn_distance((px - centre) / (unitPx * k.y));\n"
"    float cover = clamp(d * unitPx * k.y + 0.5, 0.0, 1.0);\n"
"    float3 c = mix(ground, ghost, cover);\n"
"\n"
"    if (BACKDROP == 2) {\n"
"        /* acorn-live: two soft lights drifting over the ground, one\n"
"         * cool and one warm, on a half-hour loop so the float time\n"
"         * never loses precision; and a faint vignette */\n"
"        float tau = 6.2831853 * float(P.bd.y) / 1800000.0;\n"
"        float2 uv = px / outPx;\n"
"        float aspect = outPx.x / outPx.y;\n"
"        float2 l1 = float2(0.5 + 0.34 * sin(7.0 * tau), 0.42 + 0.22 * sin(5.0 * tau + 1.3));\n"
"        float2 l2 = float2(0.5 + 0.34 * sin(11.0 * tau + 2.1), 0.58 + 0.22 * cos(9.0 * tau));\n"
"        float2 d1 = (uv - l1) * float2(aspect, 1.0);\n"
"        float2 d2 = (uv - l2) * float2(aspect, 1.0);\n"
"        c += 0.075 * exp(-dot(d1, d1) * 3.5) * float3(1.0, 1.0, 0.96);\n"
"        c += 0.05 * exp(-dot(d2, d2) * 4.5) * float3(0.95, 0.72, 0.42);\n"
"        float2 v = (uv - 0.5) * float2(aspect, 1.0);\n"
"        c *= 1.0 - 0.07 * dot(v, v);\n"
"        /* gradients this gentle band in 8 bits: dither them by less\n"
"         * than a level, with interleaved gradient noise */\n"
"        float n = fract(52.9829189 * fract(dot(px, float2(0.06711056, 0.00583715))));\n"
"        c += (n - 0.5) / 255.0;\n"
"    }\n"
"    return c;\n"
"}\n"
"\n"
"fragment float4 ps_scale(VSOut v [[stage_in]],\n"
"                         constant Params &P [[buffer(0)]],\n"
"                         texture2d<float> src [[texture(0)]],\n"
"                         texture2d<float> bgimg [[texture(1), function_constant(BACKDROP_IMAGE)]],\n"
"                         sampler lin [[sampler(0)]],\n"
"                         sampler rep [[sampler(1), function_constant(BACKDROP_IMAGE)]])\n"
"{\n"
"    float2 outPx = float2(P.post.x, P.post.y);\n"
"    float2 srcPx = float2(P.dim.x, P.dim.y);\n"
"    float2 st = v.pos.xy / outPx * srcPx;\n"
"    if (SCALING == 1) {\n"
"        float2 ratio = max(outPx / srcPx, float2(1.0));\n"
"        float2 halfw = 0.5 - 0.5 / ratio;\n"
"        float2 i = floor(st);\n"
"        float2 f = st - i - 0.5;\n"
"        st = i + 0.5 + clamp(f, -halfw, halfw);\n"
"    } else if (SCALING == 2) {\n"
"        st = floor(st) + 0.5;\n"
"    }\n"
"    float4 c = src.sample(lin, st / srcPx);\n"
"    if (SCANLINES) {\n"
"        if (P.post.w != 0 && outPx.y >= srcPx.y * 1.99\n"
"            && (uint(v.pos.y) & 1) != 0) {\n"
"            c.rgb *= 0.8;\n"
"        }\n"
"    }\n"
"    if (BACKDROP == 0) {\n"
"        return float4(c.rgb, 1.0);\n"
"    }\n"
"    float3 bg;\n"
"    if (BACKDROP_IMAGE) {\n"
"        float2 isz = float2(bgimg.get_width(), bgimg.get_height());\n"
"        if (BACKDROP == 4) {\n"
"            /* tile: from the top left, each image pixel covering a\n"
"             * point (a device pixel for an @2x tile) -- repeated by the\n"
"             * sampler, so the seams filter like the rest */\n"
"            float2 uv = v.pos.xy / (isz * float(P.bx.x) / 1000.0);\n"
"            bg = bgimg.sample(rep, uv).rgb;\n"
"        } else {\n"
"            /* picture: fill the view, centred, cropping the overhang */\n"
"            float s = max(outPx.x / isz.x, outPx.y / isz.y);\n"
"            float2 uv = (v.pos.xy - 0.5 * outPx) / (s * isz) + 0.5;\n"
"            bg = bgimg.sample(lin, uv).rgb;\n"
"        }\n"
"    } else {\n"
"        bg = backdrop_scene(v.pos.xy, outPx, srcPx, P);\n"
"    }\n"
"    return float4(c.rgb + bg * (1.0 - c.a), 1.0);\n"
"}\n"
"\n"
"vertex VSOut vs_quad(uint id [[vertex_id]], constant float4 &R [[buffer(0)]])\n"
"{\n"
"    /* one triangle covering the NDC rectangle R = {x0, y0, x1, y1}:\n"
"     * the vertices overshoot to 2x the rectangle's span, the same\n"
"     * trick as vs_main, so the unit square of t lies strictly inside\n"
"     * and no pixel of the rectangle lands on the hypotenuse (a\n"
"     * corner-exact triangle loses its lower right half to the\n"
"     * top-left fill rule).  uv runs 0..1 across the rectangle; the\n"
"     * overhang samples past 1 and the fragment clamps it. */\n"
"    VSOut o;\n"
"    float2 t = float2((id << 1) & 2, id & 2);\n"
"    o.pos = float4(R.xy + t * (R.zw - R.xy), 0, 1);\n"
"    o.uv = t;\n"
"    return o;\n"
"}\n"
"\n"
"fragment float4 ps_pointer(VSOut v [[stage_in]],\n"
"                           constant uint2 &dim [[buffer(0)]],\n"
"                           device const uchar4 *img [[buffer(1)]])\n"
"{\n"
"    /* The triangle overshoots the rectangle on purpose; its overhang\n"
"     * is not clipped by anything, so drop it here.  The image words\n"
"     * are little-endian 0xAARRGGBB, so the bytes are B, G, R, A --\n"
"     * what HWP_Update's REV-plus-alpha leaves in guest memory. */\n"
"    if (v.uv.x > 1.0f || v.uv.y > 1.0f) {\n"
"        discard_fragment();\n"
"    }\n"
"    uint2 t = min(uint2(v.uv * float2(dim)), dim - 1);\n"
"    uchar4 c = img[t.y * dim.x + t.x];\n"
"    return float4(float(c.z), float(c.y), float(c.x), float(c.w)) / 255.0f;\n"
"}\n";

static bool metal_compile_library(void)
{
    NSError *err = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];

    m.library = [m.device newLibraryWithSource:SHADER_SRC
                                       options:opts
                                         error:&err];
    [opts release];
    if (!m.library) {
        metal_log("shader compile failed: %s",
                  [[err localizedDescription] UTF8String]);
        return false;
    }
    return true;
}

/* A render pipeline from vs_main and one fragment function, specialised
 * by the function constants the caller sets. */
static id<MTLRenderPipelineState> metal_pipeline(NSString *fragment,
                                                 MTLFunctionConstantValues *cv,
                                                 MTLPixelFormat format)
{
    MTLRenderPipelineDescriptor *pd;
    id<MTLRenderPipelineState> ps;
    id<MTLFunction> vs, fs;
    NSError *err = nil;

    vs = [m.library newFunctionWithName:@"vs_main"];
    fs = [m.library newFunctionWithName:fragment constantValues:cv error:&err];
    if (!vs || !fs) {
        metal_log("function %s failed: %s", [fragment UTF8String],
                  [[err localizedDescription] UTF8String]);
        [vs release];
        [fs release];
        return nil;
    }
    pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    pd.colorAttachments[0].pixelFormat = format;
    ps = [m.device newRenderPipelineStateWithDescriptor:pd error:&err];
    [pd release];
    [vs release];
    [fs release];
    if (!ps) {
        metal_log("pipeline %s failed: %s", [fragment UTF8String],
                  [[err localizedDescription] UTF8String]);
    }
    return ps;
}

/* ------------------------------------------------------------------ */
/* The per-mode pipeline                                               */

/*
 * Open the window at the guest's own resolution.
 *
 * The shader will stretch any mode to any window, so this is only about
 * what you get before touching anything -- and a 1920x1200 desktop
 * shrunk into 800x600 is not it. The size is in backing pixels, so on a
 * Retina panel the guest lands one pixel to one pixel rather than
 * doubled -- or zoom= of them to each (ROSGD asks for 2, a guest pixel a
 * point) -- and it is clamped to what the screen can actually show.
 *
 * Once the window has been resized by hand it is left alone: the size we
 * chose is remembered, and anything else means the user has an opinion.
 */
static int fit_zoom = 1;                /* screen pixels to a guest pixel */

void metal_glue_zoom(int zoom)
{
    fit_zoom = zoom;
}

static void metal_fit_window_to_mode(uint32_t xres, uint32_t yres)
{
    NSSize now = [[m.window contentView] frame].size;
    CGFloat scale = [m.window backingScaleFactor];
    NSRect visible = [[m.window screen] visibleFrame];
    NSSize want;

    if (!xres || !yres || scale <= 0) {
        return;
    }
    if (m.auto_size.width > 0 &&
        (fabs(now.width - m.auto_size.width) > 1 ||
         fabs(now.height - m.auto_size.height) > 1)) {
        return;                         /* the user has resized it */
    }

    want.width = xres * fit_zoom / scale;
    want.height = yres * fit_zoom / scale;

    /* Leave room for the title bar and the Dock rather than filling the
     * screen exactly; a window you cannot grab is worse than a small one. */
    if (want.width > NSWidth(visible) || want.height > NSHeight(visible) - 40) {
        CGFloat fit = MIN(NSWidth(visible) / want.width,
                          (NSHeight(visible) - 40) / want.height);

        want.width = floor(want.width * fit);
        want.height = floor(want.height * fit);
    }

    [m.window setContentSize:want];
    [m.window center];
    m.auto_size = [[m.window contentView] frame].size;
    metal_log("window fitted to %ux%u at scale %.1f -> %.0fx%.0f points",
              xres, yres, scale, m.auto_size.width, m.auto_size.height);
}

static void fb_release_pipeline(void)
{
    unsigned i;

    for (i = 0; i < METAL_RING; i++) {
        [fb.raw[i] release];
        fb.raw[i] = nil;
    }
    [fb.decoded release];
    fb.decoded = nil;
    [fb.decode release];
    fb.decode = nil;
    fb.up = false;
}

static bool fb_build_pipeline(const MetalFbView *v)
{
    MTLTextureDescriptor *td;
    MTLFunctionConstantValues *cv;
    size_t bytes = (size_t)v->pitch * v->rows;
    uint32_t shader_bpp;
    unsigned i;

    fb_release_pipeline();

    if (!bytes || !v->xres || !v->yres) {
        return false;
    }

    /* The raw guest bytes, a ring deep enough that the frame the GPU is
     * still reading is never the one the CPU is writing.  Shared storage
     * means the "upload" is a memcpy into memory the GPU already sees --
     * on Apple silicon there is nothing further to transfer. */
    for (i = 0; i < METAL_RING; i++) {
        fb.raw[i] = [m.device newBufferWithLength:bytes
                                          options:MTLResourceStorageModeShared];
        if (!fb.raw[i]) {
            metal_log("raw buffer %zu bytes failed", bytes);
            fb_release_pipeline();
            return false;
        }
    }
    fb.ring = 0;
    fb.uploaded = false;                /* new buffers hold nothing yet */

    if (!fb.palette) {
        fb.palette = [m.device newBufferWithLength:256 * 4
                                           options:MTLResourceStorageModeShared];
        if (!fb.palette) {
            metal_log("palette buffer failed");
            fb_release_pipeline();
            return false;
        }
    }
    fb.pal_valid = false;               /* force a refill for the new mode */

    /* The decoded surface: linear RGB at the guest's own resolution, the
     * scale pass's only input and what a screenshot reads back. */
    td = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                       width:v->xres
                                      height:v->yres
                                   mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    fb.decoded = [m.device newTextureWithDescriptor:td];
    if (!fb.decoded) {
        metal_log("decoded texture %ux%u failed", v->xres, v->yres);
        fb_release_pipeline();
        return false;
    }

    /* One decoder per format: 32/24/16 bpp direct, and the palettised
     * decoder, which reads the true depth from the constants and covers
     * 8/4/2/1 -- 8 bpp is the one RISC OS actually asks for.  Anything
     * else is refused as dx11 refuses it: a logged clear screen, not a
     * guess at what the bits mean. */
    switch (v->bpp) {
    case 32: case 24: case 16:
        shader_bpp = v->bpp;
        break;
    case 8: case 4: case 2: case 1:
        shader_bpp = 8;
        break;
    default:
        metal_log("no decoder for bpp=%u", v->bpp);
        fb_release_pipeline();
        return false;
    }
    cv = [[MTLFunctionConstantValues alloc] init];
    [cv setConstantValue:&shader_bpp type:MTLDataTypeUInt atIndex:0];
    fb.decode = metal_pipeline(@"ps_decode", cv, MTLPixelFormatRGBA8Unorm);
    [cv release];
    if (!fb.decode) {
        fb_release_pipeline();
        return false;
    }

    fb.generation = v->generation;
    fb.bpp = v->bpp;
    fb.xres = v->xres;
    fb.yres = v->yres;
    fb.pitch = v->pitch;
    fb.rows = v->rows;
    fb.up = true;
    metal_fit_window_to_mode(v->xres, v->yres);
    metal_log("pipeline up: gen %u, %ux%u, %u bpp, pitch %u, %u rows",
              v->generation, v->xres, v->yres, v->bpp, v->pitch, v->rows);
    return true;
}

/* The scale pass does not depend on the guest's mode, only on the
 * options, so it is built once. */
static bool fb_build_scale(void)
{
    MTLFunctionConstantValues *cv = [[MTLFunctionConstantValues alloc] init];
    uint32_t scaling = (uint32_t)video_opts.scaling;
    bool scanlines = video_opts.scanlines;

    uint32_t bd = (uint32_t)backdrop.mode;

    [cv setConstantValue:&scaling type:MTLDataTypeUInt atIndex:1];
    [cv setConstantValue:&scanlines type:MTLDataTypeBool atIndex:2];
    [cv setConstantValue:&bd type:MTLDataTypeUInt atIndex:3];
    fb.scale = metal_pipeline(@"ps_scale", cv, m.layer.pixelFormat);
    [cv release];
    return fb.scale != nil;
}

/* The backdrop's share of the shader constants, for a frame or a
 * screenshot: the key, a clock on the scene's half-hour loop, the icon
 * bar the acorn is centred above, and whether the layer is on at all. */
static void backdrop_params(MetalParams *p, double px_per_point)
{
    p->bd[0] = METAL_BACKDROP_KEY;
    p->bd[1] = (uint32_t)(fmod(CFAbsoluteTimeGetCurrent() - backdrop.t0,
                               1800.0) * 1000.0);
    p->bd[2] = METAL_ICONBAR_PX;
    p->bd[3] = backdrop.mode != METAL_BACKDROP_OFF;
    p->bx[0] = (uint32_t)(px_per_point / backdrop.image_scale * 1000.0 + 0.5);
    p->bx[1] = p->bx[2] = p->bx[3] = 0;
}

/* An image for backdrop=<file>: anything ImageIO reads, drawn over the
 * key colour (so a transparent PNG sits on sage) into an sRGB RGBA8
 * texture -- the same gamma-encoded space as the guest's pixels. */
static id<MTLTexture> backdrop_load_image(const char *path)
{
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    CGImageSourceRef src = CGImageSourceCreateWithURL((CFURLRef)url, NULL);
    CGImageRef img = src ? CGImageSourceCreateImageAtIndex(src, 0, NULL) : NULL;
    id<MTLTexture> tex = nil;
    size_t w, h;

    if (src) {
        CFRelease(src);
    }
    if (!img) {
        return nil;
    }
    w = CGImageGetWidth(img);
    h = CGImageGetHeight(img);
    if (w > 4096 || h > 4096) {             /* ample for any window; saves VRAM */
        double k = 4096.0 / (double)MAX(w, h);
        w = MAX((size_t)(w * k), 1);
        h = MAX((size_t)(h * k), 1);
    }
    if (w && h) {
        uint8_t *buf = calloc(w * h, 4);
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef ctx = buf ? CGBitmapContextCreate(buf, w, h, 8, w * 4, cs,
                                     kCGImageAlphaPremultipliedLast
                                     | kCGBitmapByteOrder32Big) : NULL;

        if (ctx) {
            MTLTextureDescriptor *td;

            CGContextSetRGBFillColor(ctx, 0xB7 / 255.0, 0xC0 / 255.0,
                                     0xB4 / 255.0, 1.0);
            CGContextFillRect(ctx, CGRectMake(0, 0, w, h));
            CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
            CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
            td = [MTLTextureDescriptor
                  texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                               width:w
                                              height:h
                                           mipmapped:NO];
            td.usage = MTLTextureUsageShaderRead;
            tex = [m.device newTextureWithDescriptor:td];
            [tex replaceRegion:MTLRegionMake2D(0, 0, w, h)
                   mipmapLevel:0
                     withBytes:buf
                   bytesPerRow:w * 4];
            CGContextRelease(ctx);
        }
        CGColorSpaceRelease(cs);
        free(buf);
    }
    CGImageRelease(img);
    return tex;
}

/* Whether a tile is double density: the leaf name carries @2x before
 * its extension, Apple-style (name@2x.png).  Only the leaf, not the
 * whole path: a plain tile in a folder called @2x is an ordinary tile
 * (ROS_PRIVATE#44). */
static bool tile_is_2x(const char *path)
{
    const char *leaf = strrchr(path, '/');
    const char *dot, *p;
    size_t stem;

    leaf = leaf ? leaf + 1 : path;
    dot = strrchr(leaf, '.');
    stem = (dot && dot != leaf) ? (size_t)(dot - leaf) : strlen(leaf);
    for (p = leaf; p + 3 <= leaf + stem; p++) {
        if (p[0] == '@' && p[1] == '2' && p[2] == 'x') {
            return true;
        }
    }
    return false;
}

/* The scene: none, acorn, acorn-live, tile:<image file> or picture:<image
 * file> (a bare path is a picture).  An @2x tile covers a device pixel per
 * image pixel on a Retina screen, any other a point.  Only ever reached
 * with the feature gated on; UI thread only. */
static void backdrop_set_scene(const char *spec)
{
    int mode = METAL_BACKDROP_OFF;
    const char *path = NULL;

    [backdrop.image release];
    backdrop.image = nil;
    backdrop.image_scale = 1.0;
    if (!spec || !*spec || !strcmp(spec, "none") || !strcmp(spec, "off")) {
        mode = METAL_BACKDROP_OFF;
        spec = "none";
    } else if (!strcmp(spec, "acorn")) {
        mode = METAL_BACKDROP_ACORN;
    } else if (!strcmp(spec, "acorn-live")) {
        mode = METAL_BACKDROP_ACORN_LIVE;
    } else if (!strncmp(spec, "tile:", 5)) {
        mode = METAL_BACKDROP_TILE;
        path = spec + 5;
    } else if (!strncmp(spec, "picture:", 8)) {
        mode = METAL_BACKDROP_PICTURE;
        path = spec + 8;
    } else {
        mode = METAL_BACKDROP_PICTURE;
        path = spec;
    }
    if (path) {
        backdrop.image = backdrop_load_image(path);
        if (backdrop.image) {
            if (mode == METAL_BACKDROP_TILE && tile_is_2x(path)) {
                backdrop.image_scale = 2.0;
            }
            metal_log("backdrop: %s %s (%lux%lu)",
                      mode == METAL_BACKDROP_TILE ? "tile" : "picture", path,
                      (unsigned long)backdrop.image.width,
                      (unsigned long)backdrop.image.height);
        } else {
            mode = METAL_BACKDROP_ACORN;
            spec = "acorn";
            metal_log("backdrop: cannot read %s; using acorn", path);
        }
    }
    [backdrop.spec release];
    backdrop.spec = [[NSString alloc] initWithUTF8String:spec];
    if (mode != backdrop.mode) {
        backdrop.mode = mode;
        fb_release_scale();
    }
    backdrop.t0 = CFAbsoluteTimeGetCurrent();
    metal_log("backdrop: scene %s (mode %d)", spec, mode);
}

/* -display metal,backdrop= -- the gate and the first scene, once, at init.
 * off (the default, and what an absent option means) leaves the feature
 * out entirely: the decode ignores the transfer byte, nothing is drawn
 * beneath the desktop, and the Backdrop menu is hidden.  Anything else
 * turns it on and names the scene; none is on with no scene. */
void metal_glue_backdrop(const char *spec)
{
    backdrop.enabled = spec && *spec && strcmp(spec, "off") != 0;
    [backdrop.menu_item setHidden:!backdrop.enabled];
    if (backdrop.enabled) {
        backdrop_set_scene(spec);
    } else {
        backdrop_set_scene("none");
        metal_log("backdrop: gated off");
    }
}

/* Whether RISC OS is painting the below tag in the key colour right now,
 * sampled from the newest frame for the menu's hint: 1 yes, 0 no, -1 not
 * a 32bpp mode, -2 no frame yet. */
static int backdrop_guest_state(void)
{
    const uint8_t *raw;
    uint32_t x, y;

    if (!fb.up || !fb.uploaded) {
        return -2;
    }
    if (fb.bpp != 32) {
        return -1;
    }
    raw = [fb.raw[fb.ring] contents];
    for (y = 0; y < fb.yres && y + fb.yoff < fb.rows; y += 7) {
        const uint8_t *row = raw + (size_t)(y + fb.yoff) * fb.pitch;

        for (x = 0; x < fb.xres; x += 7) {
            const uint8_t *px = row + (size_t)(x + fb.xoff) * 4;
            uint32_t rgb = fb.pixo ? (px[0] | px[1] << 8 | px[2] << 16)
                                   : (px[2] | px[1] << 8 | px[0] << 16);

            if ((px[3] & 0xC0) == 0x80 && rgb == METAL_BACKDROP_KEY) {
                return 1;
            }
        }
    }
    return 0;
}

/* The folder the Backdrop menu lists: Tiles and Pictures inside it. */
static NSString *backdrop_folder(void)
{
    NSString *f = [[NSUserDefaults standardUserDefaults] stringForKey:@"backdropFolder"];

    return f.length ? f : [@"~/Pictures/RISC OS Backdrops" stringByExpandingTildeInPath];
}

/* The images in a folder, by name as the Finder sorts them. */
static NSArray *backdrop_images(NSString *dir, NSSet *extensions)
{
    NSArray *names = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:dir error:NULL];
    NSMutableArray *out = [NSMutableArray array];

    for (NSString *n in [names sortedArrayUsingSelector:@selector(localizedStandardCompare:)]) {
        if (![n hasPrefix:@"."]
            && [extensions containsObject:[[n pathExtension] lowercaseString]]) {
            [out addObject:[dir stringByAppendingPathComponent:n]];
            if (out.count >= 200) {
                break;
            }
        }
    }
    return out;
}

/* The pointer pipeline: straight alpha, so the ROM's anti-fringe fill
 * (transparent pixels carrying the neighbouring colour at alpha 0)
 * behaves exactly as it does against the firmware's compositor. */
static bool ptr_build(void)
{
    MTLRenderPipelineDescriptor *pd;
    id<MTLFunction> vs, fs;
    NSError *err = nil;

    vs = [m.library newFunctionWithName:@"vs_quad"];
    fs = [m.library newFunctionWithName:@"ps_pointer"];
    if (!vs || !fs) {
        metal_log("pointer shader functions missing");
        [vs release];
        [fs release];
        return false;
    }
    pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    pd.colorAttachments[0].pixelFormat = m.layer.pixelFormat;
    pd.colorAttachments[0].blendingEnabled = YES;
    pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pd.colorAttachments[0].destinationRGBBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;
    pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pd.colorAttachments[0].destinationAlphaBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;
    ptr.pipe = [m.device newRenderPipelineStateWithDescriptor:pd error:&err];
    [pd release];
    [vs release];
    [fs release];
    if (!ptr.pipe) {
        metal_log("pointer pipeline failed: %s",
                  err ? [[err localizedDescription] UTF8String] : "?");
        return false;
    }
    ptr.image = [m.device
        newBufferWithLength:METAL_CURSOR_TEXELS * METAL_CURSOR_TEXELS * 4
                    options:MTLResourceStorageModeShared];
    if (!ptr.image) {
        metal_log("pointer image buffer failed");
        return false;
    }
    return true;
}

/*
 * Frames the guest is still drawing are not worth showing, and frames
 * it has finished drawing are worth showing once.
 *
 * The blitter raises a flag whenever a fill or a sprite lands.  Seeing
 * it means the guest is mid-redraw, so the previous, whole frame is
 * left on screen; seeing it stop means the screen has settled, and
 * that is the moment to take a copy.  Waiting for the settle is the
 * point -- what gets copied is a finished picture rather than one
 * caught halfway through a window being painted.
 *
 * METAL_SETTLE_MAX bounds the wait, for two reasons.  Continuous
 * drawing -- dragging, scrolling -- never settles and still has to
 * animate.  And text and lines are plotted straight to memory without
 * passing through the blitter, so nothing raises the flag for them;
 * without a forced copy, typing would never appear.
 */
#define METAL_SETTLE_MAX 4          /* frames to wait for a settle */

static bool fb_damage_on(void)
{
    static int on = -1;

    if (on < 0) {
        const char *e = getenv("METAL_DAMAGE");
        on = e && *e && *e != '0';
    }
    return on != 0;
}

static bool fb_upload(const MetalFbView *v)
{
    static uint64_t frames, copied, on_settle, on_timeout;
    static unsigned held;
    static bool pending;
    int damage = metal_glue_take_damage();
    bool drawing = damage > 0;
    bool settled;

    frames++;
    if (drawing) {
        pending = true;             /* new content, not shown yet */
    }

    /* No damage signal at all (a virtio-gpu scanout): nothing says when
     * the guest has finished, so every frame is copied. */
    settled = pending && !drawing;
    if (damage >= 0 && fb.uploaded && !settled && ++held < METAL_SETTLE_MAX) {
        if (fb_damage_on() && frames % 300 == 0) {
            metal_log("upload: %llu of %llu frames copied (%.1f%%), "
                      "%llu on settle, %llu on timeout",
                      (unsigned long long)copied,
                      (unsigned long long)frames,
                      100.0 * copied / frames,
                      (unsigned long long)on_settle,
                      (unsigned long long)on_timeout);
        }
        return false;
    }
    if (settled) {
        on_settle++;
    } else if (fb.uploaded) {
        on_timeout++;
    }
    held = 0;
    pending = false;

    fb.ring = (fb.ring + 1) % METAL_RING;
    memcpy([fb.raw[fb.ring] contents], v->fb, (size_t)v->pitch * v->rows);
    fb.uploaded = true;
    copied++;

    if (!fb.pal_valid
        || memcmp(fb.pal_cache, v->palette, sizeof(fb.pal_cache)) != 0) {
        memcpy(fb.pal_cache, v->palette, sizeof(fb.pal_cache));
        memcpy([fb.palette contents], fb.pal_cache, sizeof(fb.pal_cache));
        fb.pal_valid = true;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* A frame                                                             */

/*
 * Display space to guest pixels.
 *
 * RISC OS puts its screen on the display the way the GPU would: one
 * scale for both axes, the image centred, and whatever is left over as a
 * margin. So when the mode's aspect does not match the display's, the
 * pointer lives inside that margin -- 1280x1024 on a 16:10 display is
 * scaled by 1.172 and inset by 210 pixels each side.
 *
 * A compositor that stretches the framebuffer across the whole window
 * while placing the pointer as a fraction of the *display* therefore
 * puts it in a box the user cannot reach the edges of. Undoing exactly
 * what the guest did is the whole fix, and it costs nothing when the
 * aspects agree: the margins are zero and this is the identity, which
 * is why 1280x800 on the same display always worked.
 */
static void metal_disp_transform(double disp_w, double disp_h,
                                 double fb_w, double fb_h,
                                 double *scale, double *mx, double *my)
{
    double k = MIN(disp_w / fb_w, disp_h / fb_h);

    if (!(k > 0)) {
        k = 1.0;
    }
    *scale = k;
    *mx = (disp_w - fb_w * k) / 2.0;
    *my = (disp_h - fb_h * k) / 2.0;
}

static void metal_update_title(void)
{
    static uint32_t last_second_frame;
    static double last_tick;
    double now = CFAbsoluteTimeGetCurrent();

    if (last_tick && now > last_tick) {
        unsigned fps = (unsigned)((frame_count - last_second_frame)
                                  / (now - last_tick));
        const char *sys, *mach;

        metal_glue_title(&sys, &mach);
        [m.window setTitle:
            [NSString stringWithFormat:
                @"%s — %ux%u, %u bpp · %u fps — %s",
                sys, fb.xres, fb.yres, fb.bpp, fps, mach]];
    }
    last_tick = now;
    last_second_frame = frame_count;
}

static bool metal_render_frame(void)
{
    MetalFbView v;
    id<CAMetalDrawable> drawable;
    id<MTLCommandBuffer> cb;
    MTLRenderPassDescriptor *rp;
    id<MTLRenderCommandEncoder> enc;
    MetalParams params;
    CGSize target;
    bool have_fb;

    if (++frame_count % 300 == 0) {
        metal_log("frame %u: pipeline %s, fb gen %u, %ux%u bpp %u, "
                  "keys %u, mouse moves %u, button events %u",
                  frame_count, fb.up ? "up" : "down",
                  fb.generation, fb.xres, fb.yres, fb.bpp,
                  m.keys, m.mouse_moves, m.mouse_buttons);
    }
    /* The status line: the window title carries the guest's mode and the
     * presented frame rate, once a second. */
    if (frame_count % 60 == 0) {
        metal_update_title();
    }

    /* METAL_SHOT_EVERY=<frames> writes the decoded surface out on that
     * period.  The decode is what is hard to be sure of -- a wrong
     * shader still presents a frame -- and a window is not always
     * something one can photograph: over a screen share, in CI, or on a
     * machine one is only ever logged in to.  Off unless asked for. */
    {
        static int shot_every = -1;

        if (shot_every < 0) {
            const char *e = getenv("METAL_SHOT_EVERY");
            shot_every = e ? atoi(e) : 0;
        }
        if (shot_every > 0 && frame_count % (uint32_t)shot_every == 0) {
            metal_screenshot();
        }
    }

    have_fb = metal_glue_fb_view(&v) != 0;
    if (have_fb) {
        mouse.gxres = v.xres;           /* the mouse maps onto this screen */
        mouse.gyres = v.yres;
        if (!fb.up || v.generation != fb.generation) {
            if (fb_failed && v.generation == fb_failed_generation) {
                have_fb = false;        /* same mode, same failure */
            } else if (!fb_build_pipeline(&v)) {
                fb_failed = true;
                fb_failed_generation = v.generation;
                have_fb = false;
            } else {
                fb_failed = false;
            }
        }
    } else if (frame_count % 300 == 0) {
        metal_log("frame %u: no fb view yet", frame_count);
    }

    /* Wait for a ring slot before touching one: the frame three back has
     * to have left the GPU. */
    dispatch_semaphore_wait(m.inflight, DISPATCH_TIME_FOREVER);

    drawable = [m.layer nextDrawable];
    if (!drawable) {
        dispatch_semaphore_signal(m.inflight);
        return false;                   /* occluded, or no free surface */
    }
    cb = [m.queue commandBuffer];

    if (have_fb) {
        fb_upload(&v);

        target = m.layer.drawableSize;
        params.dim[0] = v.xres;
        params.dim[1] = v.yres;
        params.dim[2] = v.pitch;
        params.dim[3] = v.bpp;
        params.misc[0] = v.xoffset;
        params.misc[1] = v.yoffset;
        params.misc[2] = v.pixo;
        params.misc[3] = 0;
        params.post[0] = (uint32_t)target.width;
        params.post[1] = (uint32_t)target.height;
        params.post[2] = (uint32_t)video_opts.scaling;
        params.post[3] = video_opts.scanlines ? 1 : 0;
        backdrop_params(&params, m.layer.contentsScale);
        fb.xoff = v.xoffset;
        fb.yoff = v.yoffset;
        fb.pixo = v.pixo;

        /* decode pass: raw bytes -> linear RGB */
        rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = fb.decoded;
        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        enc = [cb renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:fb.decode];
        [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
        [enc setFragmentBuffer:fb.raw[fb.ring] offset:0 atIndex:1];
        [enc setFragmentBuffer:fb.palette offset:0 atIndex:2];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0 vertexCount:3];
        [enc endEncoding];
    }

    /* scale pass: the decoded surface over the whole content view */
    rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = drawable.texture;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.05, 0.08, 1);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    enc = [cb renderCommandEncoderWithDescriptor:rp];
    if (have_fb && !fb.scale && !fb_build_scale()) {
        have_fb = false;                /* logged by metal_pipeline */
    }
    if (have_fb && fb.scale) {
        [enc setRenderPipelineState:fb.scale];
        [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
        [enc setFragmentTexture:fb.decoded atIndex:0];
        if (backdrop.mode >= METAL_BACKDROP_PICTURE) {
            [enc setFragmentTexture:backdrop.image atIndex:1];
            [enc setFragmentSamplerState:m.repeat atIndex:1];
        }
        [enc setFragmentSamplerState:m.linear atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0 vertexCount:3];
    }

    /* The pointer sprite, on top of the scaled frame.  The peer
     * commits whole transactions at UpdateSubmit, so a generation move
     * is always a complete new sprite and position; a read that raced
     * one is reported stale and the previous frame's is kept. */
    {
        MetalCursorView cv;
        bool torn = false;

        if (ptr.pipe && metal_glue_cursor_view(&cv) && !cv.stale
            && cv.generation != ptr.generation) {
            MetalCursorView again;

            memcpy([ptr.image contents], cv.argb,
                   METAL_CURSOR_TEXELS * METAL_CURSOR_TEXELS * 4);
            /* A commit that landed mid-copy tears what was just
             * uploaded: take nothing from this read, draw nothing,
             * and the next frame's read has the whole sprite. */
            torn = !metal_glue_cursor_view(&again) || again.stale
                   || again.generation != cv.generation;
            if (!torn) {
                ptr.generation = cv.generation;
                ptr.visible = cv.visible;
                ptr.x = cv.x;
                ptr.y = cv.y;
                ptr.w = cv.w;
                ptr.h = cv.h;
                ptr.img_w = cv.img_w;
                ptr.img_h = cv.img_h;
                ptr.disp_w = cv.disp_w;
                ptr.disp_h = cv.disp_h;
            }
        }
        if (have_fb && ptr.pipe && !torn && ptr.visible
            && ptr.w > 0 && ptr.h > 0
            && ptr.img_w > 0 && ptr.img_h > 0) {
            float rect[4];
            uint32_t dim[2];
            double dw = ptr.disp_w > 0 ? ptr.disp_w : (int32_t)v.xres;
            double dh = ptr.disp_h > 0 ? ptr.disp_h : (int32_t)v.yres;

            /* The dest rect is in display pixels -- the space the ROM's
             * own scale arithmetic produced it in -- and the view is
             * that display; the image keeps its own resolution and is
             * stretched to the rect, whatever mode is underneath. */
            double k, mx, my, sx, sy, sw, sh;

            metal_disp_transform(dw, dh, v.xres, v.yres, &k, &mx, &my);
            sx = ((double)ptr.x - mx) / k;   /* display -> guest pixels */
            sy = ((double)ptr.y - my) / k;
            sw = (double)ptr.w / k;
            sh = (double)ptr.h / k;

            rect[0] = (float)(2.0 * sx / (double)v.xres - 1.0);
            rect[1] = (float)(1.0 - 2.0 * sy / (double)v.yres);
            rect[2] = (float)(rect[0] + 2.0 * sw / (double)v.xres);
            rect[3] = (float)(rect[1] - 2.0 * sh / (double)v.yres);
            dim[0] = (uint32_t)ptr.img_w;
            dim[1] = (uint32_t)ptr.img_h;
            [enc setRenderPipelineState:ptr.pipe];
            [enc setVertexBytes:rect length:sizeof(rect) atIndex:0];
            [enc setFragmentBytes:dim length:sizeof(dim) atIndex:0];
            [enc setFragmentBuffer:ptr.image offset:0 atIndex:1];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0 vertexCount:3];
        }
    }
    [enc endEncoding];

    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (done.error) {
            metal_log("gpu: command buffer failed: %ld (%s)",
                      (long)done.error.code,
                      done.error.localizedDescription.UTF8String ?: "");
        }
        dispatch_semaphore_signal(m.inflight);
    }];
    [cb presentDrawable:drawable];
    [cb commit];
    return have_fb;
}

/* ------------------------------------------------------------------ */
/* Screenshot: the decoded surface to a PNG file                       */

/* Where the periodic screenshots land.  A properly-launched app has
 * cwd=/, so a bare file name would vanish; the app support directory
 * is the configured home (SCRIPTING.md section 7) and the scripting
 * surface's screenshot command writes there too. */
static NSString *metal_shot_path(unsigned n)
{
    NSArray *dirs = NSSearchPathForDirectoriesInDomains(
        NSApplicationSupportDirectory, NSUserDomainMask, YES);
    NSString *base = dirs.count
        ? [dirs[0] stringByAppendingPathComponent:@"RISCOSQEMU"]
        : NSTemporaryDirectory();

    [[NSFileManager defaultManager] createDirectoryAtPath:base
                             withIntermediateDirectories:YES
                                              attributes:nil error:NULL];
    return [base stringByAppendingPathComponent:
            [NSString stringWithFormat:@"metal-screenshot-%u.png", n]];
}

static bool metal_screenshot_to(NSString *path)
{
    id<MTLBuffer> staging;
    id<MTLCommandBuffer> cb;
    id<MTLBlitCommandEncoder> blit;
    CGColorSpaceRef cs;
    CGContextRef ctx;
    CGImageRef img;
    CGImageDestinationRef dst;
    NSURL *url;
    size_t stride;
    bool ok = false;

    if (!fb.up || !fb.decoded) {
        return false;
    }
    stride = (size_t)fb.xres * 4;
    staging = [m.device newBufferWithLength:stride * fb.yres
                                    options:MTLResourceStorageModeShared];
    if (!staging) {
        return false;
    }
    cb = [m.queue commandBuffer];

    /* With a backdrop the decoded surface has holes in it: run the scale
     * pass once more, at 1:1 into an RGBA8 texture, so the picture is
     * the composite a human sees. */
    id<MTLTexture> source = fb.decoded;
    id<MTLTexture> composite = nil;

    if (backdrop.mode != METAL_BACKDROP_OFF) {
        if (!fb.shot) {
            MTLFunctionConstantValues *scv = [[MTLFunctionConstantValues alloc] init];
            uint32_t nearest = METAL_SCALING_NEAREST;
            bool no = false;
            uint32_t bd = (uint32_t)backdrop.mode;

            [scv setConstantValue:&nearest type:MTLDataTypeUInt atIndex:1];
            [scv setConstantValue:&no type:MTLDataTypeBool atIndex:2];
            [scv setConstantValue:&bd type:MTLDataTypeUInt atIndex:3];
            fb.shot = metal_pipeline(@"ps_scale", scv, MTLPixelFormatRGBA8Unorm);
            [scv release];
        }
        MTLTextureDescriptor *std_ = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                         width:fb.xres
                                        height:fb.yres
                                     mipmapped:NO];
        std_.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        std_.storageMode = MTLStorageModePrivate;
        composite = fb.shot ? [m.device newTextureWithDescriptor:std_] : nil;
        if (composite) {
            MetalParams sp;
            MTLRenderPassDescriptor *srp = [MTLRenderPassDescriptor renderPassDescriptor];
            id<MTLRenderCommandEncoder> senc;

            memset(&sp, 0, sizeof(sp));
            sp.dim[0] = fb.xres;
            sp.dim[1] = fb.yres;
            sp.post[0] = fb.xres;
            sp.post[1] = fb.yres;
            sp.post[2] = METAL_SCALING_NEAREST;
            /* the window's tile size, carried into guest pixels */
            backdrop_params(&sp, m.layer.contentsScale * (double)fb.yres
                                 / MAX(m.layer.drawableSize.height, 1.0));
            srp.colorAttachments[0].texture = composite;
            srp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            srp.colorAttachments[0].storeAction = MTLStoreActionStore;
            senc = [cb renderCommandEncoderWithDescriptor:srp];
            [senc setRenderPipelineState:fb.shot];
            [senc setFragmentBytes:&sp length:sizeof(sp) atIndex:0];
            [senc setFragmentTexture:fb.decoded atIndex:0];
            if (backdrop.mode >= METAL_BACKDROP_PICTURE) {
                [senc setFragmentTexture:backdrop.image atIndex:1];
                [senc setFragmentSamplerState:m.repeat atIndex:1];
            }
            [senc setFragmentSamplerState:m.linear atIndex:0];
            [senc drawPrimitives:MTLPrimitiveTypeTriangle
                     vertexStart:0 vertexCount:3];
            [senc endEncoding];
            source = composite;
        }
    }

    blit = [cb blitCommandEncoder];
    [blit copyFromTexture:source
              sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(fb.xres, fb.yres, 1)
                 toBuffer:staging
        destinationOffset:0
   destinationBytesPerRow:stride
 destinationBytesPerImage:stride * fb.yres];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    [composite release];

    /* The pointer is composited over the frame, not in it, and a
     * screenshot shows what a human sees: blend the sprite in with the
     * same straight alpha the pipeline uses.  The dest rect is in
     * display pixels, so it is mapped into the decoded surface's own
     * guest pixels first; the image is indexed by its own resolution. */
    {
        MetalCursorView cv;

        if (metal_glue_cursor_view(&cv) && !cv.stale && cv.visible
            && cv.w > 0 && cv.h > 0 && cv.img_w > 0 && cv.img_h > 0
            && cv.disp_w > 0 && cv.disp_h > 0) {
            uint8_t *px = [staging contents];
            const uint8_t *sp = cv.argb;
            double k, mx, my;
            double tx, ty;
            int32_t gx0, gy0, gx1, gy1;

            metal_disp_transform(cv.disp_w, cv.disp_h, fb.xres, fb.yres,
                                 &k, &mx, &my);
            gx0 = (int32_t)(((double)cv.x - mx) / k);
            gy0 = (int32_t)(((double)cv.y - my) / k);
            gx1 = (int32_t)(((double)cv.x + cv.w - mx) / k);
            gy1 = (int32_t)(((double)cv.y + cv.h - my) / k);
            /* sprite texels per guest pixel, for the stretch */
            tx = (double)cv.img_w / ((double)cv.w / k);
            ty = (double)cv.img_h / ((double)cv.h / k);
            int32_t x0 = gx0 < 0 ? 0 : gx0;
            int32_t y0 = gy0 < 0 ? 0 : gy0;
            int32_t sx, sy;

            if (gx1 > (int32_t)fb.xres) {
                gx1 = (int32_t)fb.xres;
            }
            if (gy1 > (int32_t)fb.yres) {
                gy1 = (int32_t)fb.yres;
            }
            for (sy = y0; sy < gy1; sy++) {
                for (sx = x0; sx < gx1; sx++) {
                    /* nearest sprite texel for this guest pixel; the
                     * sprite bytes are B,G,R,A, the surface R,G,B,A */
                    int32_t ix = (int32_t)((sx - gx0) * tx);
                    int32_t iy = (int32_t)((sy - gy0) * ty);

                    if (ix >= cv.img_w) {
                        ix = cv.img_w - 1;
                    }
                    if (iy >= cv.img_h) {
                        iy = cv.img_h - 1;
                    }
                    const uint8_t *s = sp + (size_t)(iy * cv.img_w + ix) * 4;
                    uint8_t *d = px + (size_t)(sy * fb.xres + sx) * 4;
                    int a = s[3];

                    d[0] = (uint8_t)(d[0] + (s[2] - d[0]) * a / 255);
                    d[1] = (uint8_t)(d[1] + (s[1] - d[1]) * a / 255);
                    d[2] = (uint8_t)(d[2] + (s[0] - d[2]) * a / 255);
                }
            }
        }
    }

    cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    ctx = CGBitmapContextCreate([staging contents], fb.xres, fb.yres, 8,
                                stride, cs,
                                kCGImageAlphaNoneSkipLast
                                | kCGBitmapByteOrder32Big);
    img = ctx ? CGBitmapContextCreateImage(ctx) : NULL;

    url = [NSURL fileURLWithPath:path];
    dst = img ? CGImageDestinationCreateWithURL((CFURLRef)url,
                                                CFSTR("public.png"), 1, NULL)
              : NULL;
    if (dst) {
        CGImageDestinationAddImage(dst, img, NULL);
        CGImageDestinationFinalize(dst);
        CFRelease(dst);
        metal_log("screenshot: %s (%ux%u)", [path UTF8String],
                  fb.xres, fb.yres);
        ok = true;
    } else {
        metal_log("screenshot failed: %s", [path UTF8String]);
    }
    if (img) {
        CGImageRelease(img);
    }
    if (ctx) {
        CGContextRelease(ctx);
    }
    CGColorSpaceRelease(cs);
    [staging release];
    return ok;
}

static void metal_screenshot(void)
{
    static unsigned n;

    metal_screenshot_to(metal_shot_path(++n));
}

/* The scripting surface's screenshot command lands here; the C side
 * has already chosen and confined the path. */
bool metal_ui_screenshot(const char *path)
{
    return path && metal_screenshot_to([NSString stringWithUTF8String:path]);
}

/* ------------------------------------------------------------------ */
/* The window                                                          */

@interface MetalView : NSView
@end

@implementation MetalView

- (CALayer *)makeBackingLayer
{
    CAMetalLayer *l = [CAMetalLayer layer];

    l.device = m.device;
    l.pixelFormat = MTLPixelFormatBGRA8Unorm;
    l.framebufferOnly = YES;
    l.maximumDrawableCount = METAL_RING;
    l.displaySyncEnabled = YES;      /* the present is the pacing */
    l.allowsNextDrawableTimeout = YES;
    return l;
}

- (BOOL)wantsUpdateLayer          { return YES; }
- (BOOL)acceptsFirstResponder     { return YES; }
- (BOOL)isOpaque                  { return YES; }
- (BOOL)acceptsFirstMouse:(NSEvent *)e { (void)e; return YES; }

/* The drawable is in backing pixels; the view is in points.  Keeping
 * them in step is what makes a Retina display show the guest at its own
 * resolution rather than a doubled blur. */
- (void)updateDrawableSize
{
    NSRect b = [self convertRectToBacking:[self bounds]];

    if (NSWidth(b) > 0 && NSHeight(b) > 0) {
        m.layer.contentsScale = [[self window] backingScaleFactor];
        m.layer.drawableSize = CGSizeMake(NSWidth(b), NSHeight(b));
    }
}

- (void)setFrameSize:(NSSize)size
{
    [super setFrameSize:size];
    [self updateDrawableSize];
}

- (void)viewDidChangeBackingProperties
{
    [super viewDidChangeBackingProperties];
    [self updateDrawableSize];
}

/* One tracking area over the whole view, so entering and leaving the
 * guest's screen is an event rather than something polled per frame --
 * and so plain movement is one too: without NSTrackingMouseMoved (or
 * acceptsMouseMovedEvents) Cocoa sends mouseMoved: nowhere, and the
 * guest's arrow followed the host's only while a button was held. */
- (void)updateTrackingAreas
{
    NSTrackingArea *area;

    for (NSTrackingArea *old in [self trackingAreas]) {
        [self removeTrackingArea:old];
    }
    area = [[NSTrackingArea alloc]
            initWithRect:[self bounds]
                 options:NSTrackingMouseEnteredAndExited
                         | NSTrackingMouseMoved
                         | NSTrackingActiveInKeyWindow
                         | NSTrackingInVisibleRect
                   owner:self
                userInfo:nil];
    [self addTrackingArea:area];
    [area release];
    [super updateTrackingAreas];
}

- (void)mouseEntered:(NSEvent *)e
{
    (void)e;
    cursor.inside = true;
    metal_cursor_sync();
}

- (void)mouseExited:(NSEvent *)e
{
    (void)e;
    cursor.inside = false;
    metal_cursor_sync();
}

/* Keyboard.  Ctrl+Alt+G is the grab toggle and never reaches the guest;
 * so is F13, where a PC keyboard's PrintScreen lands, which takes the
 * screenshot.  Auto-repeat is dropped: the guest does its own. */
- (void)keyDown:(NSEvent *)e
{
    NSUInteger f = [e modifierFlags];

    if (metal_debug()) {
        metal_log("view: keyDown code %u", (unsigned)[e keyCode]);
    }

    if ([e keyCode] == 5 /* G */
        && (f & NSEventModifierFlagControl)
        && (f & NSEventModifierFlagOption)) {
        metal_set_grab(!kbd.on);
        return;
    }
    if ([e keyCode] == 105 /* F13 */) {
        metal_screenshot();
        return;
    }
    if ([e isARepeat]) {
        return;
    }
    m.keys++;
    metal_glue_key(true, [e keyCode]);
}

- (void)keyUp:(NSEvent *)e
{
    if ([e keyCode] == 105) {
        return;
    }
    metal_glue_key(false, [e keyCode]);
}

- (void)flagsChanged:(NSEvent *)e
{
    m.keys++;
    metal_flags_changed(e);
}

/* Mouse.  Middle click captures the pointer (Ctrl+Alt+G or focus loss
 * releases).  Left clicks go straight through to the guest, now that the
 * tablet keeps the two arrows together, so entering the window costs
 * nothing. */
- (void)mouseMoved:(NSEvent *)e       { metal_mouse_move(e); }
- (void)mouseDragged:(NSEvent *)e     { metal_mouse_move(e); }
- (void)rightMouseDragged:(NSEvent *)e { metal_mouse_move(e); }
- (void)otherMouseDragged:(NSEvent *)e { metal_mouse_move(e); }

/*
 * RISC OS wants three buttons -- Select, Menu and Adjust -- and a Mac
 * has one. Control-click is Menu; Command, Option or Shift is Adjust.
 *
 * Three things make it fiddlier than it looks. macOS turns a
 * Control-click into a right click before we ever see it, so Menu has to
 * be recognised on that stream too. The modifier may be let go before
 * the button is, so the button sent is latched at press time and that is
 * what gets released. And the modifier must be taken away from the guest
 * while it is being a button: RISC OS reads Shift-Adjust and Adjust as
 * different gestures, so leaking the Shift makes the click land as
 * something else -- which is how Shift-click came to do nothing at all
 * while Control-click worked.
 *
 * Command is the one to reach for. RISC OS has no Command key, so
 * nothing is lost to it, whereas Shift-Select and Control-Select are
 * gestures the Filer really uses.
 */
static int metal_button_for(NSEvent *e, int plain)
{
    NSUInteger f = [e modifierFlags];

    if (f & NSEventModifierFlagControl) {
        metal_mods_hold(METAL_CTRL_ANY);
        return 1;                       /* Menu */
    }
    if (f & NSEventModifierFlagCommand) {
        metal_mods_hold(METAL_CMD_ANY);
        return 2;                       /* Adjust */
    }
    if (f & NSEventModifierFlagOption) {
        metal_mods_hold(METAL_ALT_ANY);
        return 2;                       /* Adjust */
    }
    if (f & NSEventModifierFlagShift) {
        metal_mods_hold(METAL_SHIFT_ANY);
        return 2;                       /* Adjust */
    }
    return plain;
}

/*
 * Every button press and release goes through here, so the guest's idea
 * of what is held is something we know rather than something we hope.
 * A press that never gets its release leaves RISC OS dragging for ever
 * -- the Wimp ends a drag on the button going up, and if it never does
 * the window simply follows the pointer until something else intervenes.
 */
static void metal_button(int button, bool down)
{
    if ((unsigned)button > 2 || mouse.down[button] == down) {
        return;                     /* never send the same edge twice */
    }
    mouse.down[button] = down;
    if (metal_debug()) {
        metal_log("mouse: button %d %s", button, down ? "down" : "up");
    }
    metal_glue_mouse_btn(button, down);
    m.mouse_buttons++;
}

/* Let go of anything still held. Focus loss, a grab ending and shutdown
 * all need this: the host stops telling us about a button we never saw
 * released, and the guest is left holding it. */
static void metal_buttons_release_all(void)
{
    int b;

    for (b = 0; b < 3; b++) {
        if (mouse.down[b]) {
            metal_button(b, false);
        }
    }
}

- (void)mouseDown:(NSEvent *)e
{
    mouse.held_left = metal_button_for(e, 0);
    if (metal_debug()) {
        metal_log("view: mouseDown -> button %d (flags %#lx)",
                  mouse.held_left, (unsigned long)[e modifierFlags]);
    }
    metal_button(mouse.held_left, true);
}

- (void)mouseUp:(NSEvent *)e
{
    metal_button(mouse.held_left, false);
    metal_mods_restore(e);
}

- (void)rightMouseDown:(NSEvent *)e
{
    mouse.held_right = metal_button_for(e, 2);
    if (metal_debug()) {
        metal_log("view: rightMouseDown -> button %d (flags %#lx)",
                  mouse.held_right, (unsigned long)[e modifierFlags]);
    }
    metal_button(mouse.held_right, true);
}

- (void)rightMouseUp:(NSEvent *)e
{
    metal_button(mouse.held_right, false);
    metal_mods_restore(e);
}

/*
 * A real middle button is Menu, not a grab toggle: the Windows front end
 * spends it on capturing the pointer, but the tablet already keeps the
 * two arrows together here, so there is nothing to capture and Menu is
 * the better use of it. Ctrl+Alt+G still grabs.
 */
- (void)otherMouseDown:(NSEvent *)e
{
    if ([e buttonNumber] != 2) {
        return;
    }
    metal_button(1, true);
}

- (void)otherMouseUp:(NSEvent *)e
{
    if ([e buttonNumber] != 2) {
        return;
    }
    metal_button(1, false);
}

- (void)scrollWheel:(NSEvent *)e
{
    /* A wheel notch is one line; a trackpad reports fractions of one, so
     * accumulate and send whole notches. */
    static CGFloat acc;
    int notches;

    acc += [e hasPreciseScrollingDeltas] ? [e scrollingDeltaY] / 10.0
                                         : [e scrollingDeltaY];
    notches = (int)acc;
    if (notches) {
        acc -= notches;
        metal_glue_mouse_wheel(notches);
    }
}

@end

/* ------------------------------------------------------------------ */
/* Application delegate: the menu's targets, and the window's fate      */

@interface MetalDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate,
                                     NSMenuDelegate>
@end

@implementation MetalDelegate

- (void)windowWillClose:(NSNotification *)n
{
    (void)n;
    /* Logged unconditionally: a session that ended on its own leaves
     * nothing else behind saying so, and the two ways out look
     * identical from the exit status. */
    metal_log("shutting down: window closed");
    m.lost = true;
}

- (void)windowDidResignKey:(NSNotification *)n
{
    (void)n;
    metal_set_grab(false);          /* never hold the host pointer hostage */
    metal_release_modifiers();
    metal_buttons_release_all();    /* nor a button it will never release */
    cursor.inside = false;          /* and never a cursor, either */
    metal_cursor_sync();
}

- (void)windowDidBecomeKey:(NSNotification *)n
{
    (void)n;
    /* The pointer may already be over the guest's screen: no enter event
     * is coming for a window that was activated under a still cursor. */
    cursor.inside = NSMouseInRect(
        [m.view convertPoint:[m.window mouseLocationOutsideOfEventStream]
                    fromView:nil],
        [m.view bounds], [m.view isFlipped]);
    metal_cursor_sync();
}

- (void)windowDidResize:(NSNotification *)n
{
    (void)n;
    [m.view updateDrawableSize];
}

- (void)quitAction:(id)sender
{
    /* Not [NSApp terminate:]: the machine is asked to power off and the
     * UI loop unwinds, which is what gets the disc written back. */
    (void)sender;
    metal_log("shutting down: Quit (Cmd-Q)");
    m.lost = true;
}

- (void)grabAction:(id)sender     { (void)sender; metal_set_grab(!kbd.on); }
- (void)shotAction:(id)sender     { (void)sender; metal_screenshot(); }
- (void)snapshotAction:(id)sender { (void)sender; metal_glue_load_snapshot(); }

- (void)fullScreenAction:(id)sender
{
    (void)sender;
    [m.window toggleFullScreen:nil];
}

/* The Backdrop menu (MACOS.md, "The backdrop layer"): the Acorn scenes,
 * None, and Tiles and Pictures submenus listing the images in the
 * backdrops folder -- rebuilt each time it opens, so a file dropped into
 * the folder is there next time.  A choice applies at once and is kept
 * as the backdrop default, which the app's launcher passes next time.
 * The menu is there only when backdrop= gates the feature on. */
static NSMenuItem *backdrop_item(NSString *title, NSString *spec, id target)
{
    NSMenuItem *it = [[NSMenuItem alloc] initWithTitle:title
                                                action:@selector(backdropAction:)
                                         keyEquivalent:@""];

    [it setTarget:target];
    [it setRepresentedObject:spec];
    [it setState:[spec isEqualToString:backdrop.spec] ? NSControlStateValueOn
                                                      : NSControlStateValueOff];
    return [it autorelease];
}

static NSMenuItem *backdrop_note(NSString *title)
{
    NSMenuItem *it = [[NSMenuItem alloc] initWithTitle:title action:NULL keyEquivalent:@""];

    [it setEnabled:NO];
    return [it autorelease];
}

static NSMenu *backdrop_list(NSString *title, NSString *kind, NSArray *groups,
                             id target)
{
    NSMenu *menu = [[NSMenu alloc] initWithTitle:title];
    bool any = false;

    [menu setAutoenablesItems:NO];
    for (NSArray *group in groups) {
        if (group.count && any) {
            [menu addItem:[NSMenuItem separatorItem]];
        }
        for (NSString *path in group) {
            [menu addItem:backdrop_item([[path lastPathComponent] stringByDeletingPathExtension],
                                        [NSString stringWithFormat:@"%@:%@", kind, path],
                                        target)];
            any = true;
        }
    }
    if (!any) {
        [menu addItem:backdrop_note([NSString stringWithFormat:
            @"No images in the %@ folder", title])];
    }
    return [menu autorelease];
}

- (void)menuNeedsUpdate:(NSMenu *)menu
{
    NSSet *ext = [NSSet setWithObjects:@"png", @"jpg", @"jpeg", @"heic", @"heif",
                                       @"tif", @"tiff", @"gif", @"bmp", @"webp", nil];
    NSString *dir = backdrop_folder();
    NSMenuItem *sub;

    if ([[menu title] isEqualToString:@"Machine"]) {
        /* The doorbell switch: shown only where the device exists, and
         * ticked from what the device says rather than what was last
         * clicked, so a script's change is reflected too. */
        int sockets = metal_glue_sockets();

        [hostnet_item setHidden:sockets < 0];
        [hostnet_item setState:sockets > 0 ? NSControlStateValueOn
                                           : NSControlStateValueOff];
        return;
    }
    if (![[menu title] isEqualToString:@"Backdrop"] || !backdrop.enabled) {
        return;
    }
    [menu removeAllItems];
    switch (backdrop_guest_state()) {
    case 0:
        [menu addItem:backdrop_note(@"RISC OS is not drawing a backdrop to show through")];
        [menu addItem:[NSMenuItem separatorItem]];
        break;
    case -1:
        [menu addItem:backdrop_note(@"Backdrops need a 16 million colour screen mode")];
        [menu addItem:[NSMenuItem separatorItem]];
        break;
    default:
        break;
    }
    [menu addItem:backdrop_item(@"Acorn", @"acorn", self)];
    [menu addItem:backdrop_item(@"Acorn, Moving", @"acorn-live", self)];
    [menu addItem:backdrop_item(@"None", @"none", self)];
    [menu addItem:[NSMenuItem separatorItem]];

    sub = [[[NSMenuItem alloc] initWithTitle:@"Tiles" action:NULL keyEquivalent:@""] autorelease];
    [sub setSubmenu:backdrop_list(@"Tiles", @"tile",
        @[ backdrop_images([dir stringByAppendingPathComponent:@"Tiles"], ext) ], self)];
    [menu addItem:sub];
    sub = [[[NSMenuItem alloc] initWithTitle:@"Pictures" action:NULL keyEquivalent:@""] autorelease];
    [sub setSubmenu:backdrop_list(@"Pictures", @"picture",
        @[ backdrop_images([dir stringByAppendingPathComponent:@"Pictures"], ext),
           backdrop_images(@"/System/Library/Desktop Pictures", [NSSet setWithObject:@"heic"]) ],
        self)];
    [menu addItem:sub];
    [menu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *open = [[[NSMenuItem alloc] initWithTitle:@"Open Backdrops Folder"
                                                   action:@selector(openBackdropsFolderAction:)
                                            keyEquivalent:@""] autorelease];
    [open setTarget:self];
    [menu addItem:open];
    NSMenuItem *choose = [[[NSMenuItem alloc] initWithTitle:@"Choose Backdrops Folder\u2026"
                                                     action:@selector(chooseBackdropsFolderAction:)
                                              keyEquivalent:@""] autorelease];
    [choose setTarget:self];
    [menu addItem:choose];
}

- (void)backdropAction:(NSMenuItem *)sender
{
    NSString *spec = [sender representedObject];

    if (spec && backdrop.enabled) {
        backdrop_set_scene([spec UTF8String]);
        [[NSUserDefaults standardUserDefaults] setObject:backdrop.spec forKey:@"backdrop"];
    }
}

/* Machine > HostNet: the network mode.  One stack or the other, never
 * both, and switching reboots RISC OS into the other -- the module moves,
 * the doorbell follows it, and the boot takes its form from the Internet
 * module it finds.  The confirmation is because a reboot loses the
 * running desktop, as Load Snapshot does. */
- (void)hostnetAction:(NSMenuItem *)sender
{
    bool toHostNet = metal_glue_sockets() <= 0;
    NSAlert *a = [[[NSAlert alloc] init] autorelease];
    NSString *to = toHostNet ? @"HostNet"
                             : @"the RISC OS network stack";

    [a setMessageText:[NSString stringWithFormat:
        @"Switch networking to %@?", to]];
    [a setInformativeText:[NSString stringWithFormat:
        @"HostNet has your Mac serve RISC OS's sockets; the RISC OS stack "
         @"runs its own networking over an emulated card. The machine will "
         @"restart in the new mode."]];
    [a addButtonWithTitle:@"Switch and Restart"];
    [a addButtonWithTitle:@"Cancel"];
    if ([a runModal] != NSAlertFirstButtonReturn) {
        return;
    }
    metal_ui_hostnet_switch(toHostNet);
}

/* Switch the network mode in place: record the choice so the next app
 * launch comes up in it, then hand off to the backend, which moves the
 * guest's HostNet module in or out of the load path, flips the doorbell to
 * match and reboots RISC OS.  No app relaunch -- the earlier fork/open of
 * the whole bundle was unreliable, and there is no need: qemu carries both
 * the doorbell and the emulated card, so a guest reboot lands in whichever
 * mode the module now selects. */
void metal_ui_hostnet_switch(bool on)
{
    NSBundle *b = [NSBundle mainBundle];

    if (b && b.bundleIdentifier) {
        [[NSUserDefaults standardUserDefaults] setObject:(on ? @"on" : @"off")
                                                  forKey:@"hostnet"];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
    metal_glue_hostnet_apply(on);
}

- (void)openBackdropsFolderAction:(id)sender
{
    NSString *dir = backdrop_folder();
    NSFileManager *fm = [NSFileManager defaultManager];

    (void)sender;
    [fm createDirectoryAtPath:[dir stringByAppendingPathComponent:@"Tiles"]
  withIntermediateDirectories:YES attributes:nil error:NULL];
    [fm createDirectoryAtPath:[dir stringByAppendingPathComponent:@"Pictures"]
  withIntermediateDirectories:YES attributes:nil error:NULL];
    [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:dir]];
}

- (void)chooseBackdropsFolderAction:(id)sender
{
    NSOpenPanel *panel = [NSOpenPanel openPanel];

    (void)sender;
    panel.canChooseDirectories = YES;
    panel.canChooseFiles = NO;
    panel.allowsMultipleSelection = NO;
    panel.prompt = @"Use Folder";
    panel.message = @"Choose a folder with Tiles and Pictures folders inside it.";
    panel.directoryURL = [NSURL fileURLWithPath:backdrop_folder()];
    [panel beginSheetModalForWindow:m.window completionHandler:^(NSModalResponse r) {
        if (r == NSModalResponseOK && panel.URL) {
            [[NSUserDefaults standardUserDefaults] setObject:[panel.URL path]
                                                      forKey:@"backdropFolder"];
        }
    }];
}

@end

static MetalDelegate *delegate;

static void metal_add_item(NSMenu *menu, NSString *title, SEL action,
                           NSString *key, NSEventModifierFlags mods)
{
    NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:title
                                                  action:action
                                           keyEquivalent:key];

    [item setKeyEquivalentModifierMask:mods];
    [item setTarget:delegate];
    [menu addItem:item];
    [item release];
}

static void metal_build_menu(void)
{
    NSMenu *bar = [[NSMenu alloc] init];
    NSMenu *app = [[NSMenu alloc] init];
    NSMenu *machine = [[NSMenu alloc] initWithTitle:@"Machine"];
    NSMenuItem *appItem = [[NSMenuItem alloc] init];
    NSMenuItem *machineItem = [[NSMenuItem alloc] init];

    metal_add_item(app, @"Quit RISC OS", @selector(quitAction:), @"q",
                   NSEventModifierFlagCommand);
    [appItem setSubmenu:app];
    [bar addItem:appItem];

    metal_add_item(machine, @"Grab Pointer", @selector(grabAction:), @"g",
                   NSEventModifierFlagControl | NSEventModifierFlagOption);
    metal_add_item(machine, @"Save Screenshot", @selector(shotAction:), @"s",
                   NSEventModifierFlagCommand);
    metal_add_item(machine, @"Load Snapshot", @selector(snapshotAction:), @"",
                   0);
    [machine addItem:[NSMenuItem separatorItem]];
    {
        NSMenu *bd = [[NSMenu alloc] initWithTitle:@"Backdrop"];
        NSMenuItem *bdItem = [[NSMenuItem alloc] initWithTitle:@"Backdrop"
                                                        action:NULL
                                                 keyEquivalent:@""];

        [bd setDelegate:delegate];
        [bd setAutoenablesItems:NO];
        [bdItem setSubmenu:bd];
        /* Hidden until -display metal,backdrop= gates the feature on
         * (metal_glue_backdrop): the options are read after the menu is
         * built, and an absent option means off.  Kept, not released:
         * the gate shows or hides it. */
        [bdItem setHidden:YES];
        backdrop.menu_item = bdItem;
        [machine addItem:bdItem];
        [bd release];
    }
    {
        NSMenuItem *hn = [[NSMenuItem alloc] initWithTitle:@"HostNet"
                                                    action:@selector(hostnetAction:)
                                             keyEquivalent:@""];

        [hn setTarget:delegate];
        /* Hidden until menuNeedsUpdate finds the doorbell in this
         * machine, and ticked from the device on every open.  Kept, not
         * released, like the Backdrop item above. */
        [hn setHidden:YES];
        [machine addItem:hn];
        hostnet_item = hn;
    }
    metal_add_item(machine, @"Toggle Full Screen",
                   @selector(fullScreenAction:), @"f",
                   NSEventModifierFlagCommand | NSEventModifierFlagControl);
    /* menuNeedsUpdate ticks Machine > HostNet from the device on every
     * open; the Backdrop submenu keeps its own delegate above. */
    [machine setDelegate:delegate];
    [machineItem setSubmenu:machine];
    [bar addItem:machineItem];

    [NSApp setMainMenu:bar];
    [bar release];
    [app release];
    [machine release];
    [appItem release];
    [machineItem release];
}

static bool metal_create_window(void)
{
    NSRect frame = NSMakeRect(0, 0, 800, 600);
    NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                     | NSWindowStyleMaskMiniaturizable
                     | NSWindowStyleMaskResizable;

    m.window = [[NSWindow alloc] initWithContentRect:frame
                                           styleMask:style
                                             backing:NSBackingStoreBuffered
                                               defer:NO];
    if (!m.window) {
        return false;
    }
    {
        const char *sys, *mach;

        metal_glue_title(&sys, &mach);
        [m.window setTitle:[NSString stringWithFormat:@"%s — %s", sys, mach]];
    }
    [m.window setDelegate:delegate];
    [m.window setAcceptsMouseMovedEvents:YES];
    [m.window setReleasedWhenClosed:NO];
    [m.window center];

    m.view = [[MetalView alloc] initWithFrame:frame];
    [m.view setWantsLayer:YES];
    m.layer = (CAMetalLayer *)[m.view layer];
    [m.window setContentView:m.view];
    [m.window makeFirstResponder:m.view];
    [m.view updateDrawableSize];
    return m.layer != nil;
}

/* ------------------------------------------------------------------ */
/* The boundary, called from ui/metal.c                                */

int metal_backend_init(void)
{
    MTLSamplerDescriptor *sd;

    m.device = MTLCreateSystemDefaultDevice();
    if (!m.device) {
        metal_log("no Metal device");
        return -1;
    }
    metal_log("device: %s", [[m.device name] UTF8String]);

    m.queue = [m.device newCommandQueue];
    if (!m.queue) {
        metal_log("no command queue");
        return -1;
    }

    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    delegate = [[MetalDelegate alloc] init];
    [NSApp setDelegate:delegate];
    metal_build_menu();
    metal_script_register();

    if (!metal_create_window()) {
        metal_log("no window");
        return -1;
    }
    if (!metal_compile_library() || !fb_build_scale() || !ptr_build()) {
        return -1;
    }

    sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    m.linear = [m.device newSamplerStateWithDescriptor:sd];
    sd.sAddressMode = MTLSamplerAddressModeRepeat;
    sd.tAddressMode = MTLSamplerAddressModeRepeat;
    m.repeat = [m.device newSamplerStateWithDescriptor:sd];
    [sd release];
    if (!m.linear) {
        metal_log("no sampler");
        return -1;
    }

    m.inflight = dispatch_semaphore_create(METAL_RING);
    m.ready = true;
    return 0;
}

int metal_backend_main(void)
{
    if (!m.ready) {
        return 1;
    }

    /* We run our own event pump rather than -[NSApp run], so the frame
     * and the events interleave the way the Windows front end's do.
     * finishLaunching is what -run would have called first. */
    [NSApp finishLaunching];
    metal_script_register();      /* again: after AppKit's scripting init */
    [m.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    metal_log("window: visible %d, miniaturized %d, key %d, occlusion %#lx, "
              "drawable %.0fx%.0f",
              (int)[m.window isVisible], (int)[m.window isMiniaturized],
              (int)[m.window isKeyWindow],
              (unsigned long)[m.window occlusionState],
              m.layer.drawableSize.width, m.layer.drawableSize.height);

    while (!m.lost) {
        @autoreleasepool {
            NSEvent *e;

            while ((e = [NSApp nextEventMatchingMask:NSEventMaskAny
                                           untilDate:[NSDate distantPast]
                                              inMode:NSDefaultRunLoopMode
                                             dequeue:YES]) != nil) {
                if (metal_debug()) {
                    NSEventType t = [e type];

                    if (t == NSEventTypeKeyDown || t == NSEventTypeKeyUp
                        || t == NSEventTypeFlagsChanged) {
                        id fr = [m.window firstResponder];

                        metal_log("pump: key event type %ld code %u, "
                                  "our window %d, app active %d, key win %d, "
                                  "first responder %s",
                                  (long)t, (unsigned)[e keyCode],
                                  (int)([e window] == m.window),
                                  (int)[NSApp isActive],
                                  (int)[m.window isKeyWindow],
                                  fr ? [NSStringFromClass([fr class]) UTF8String]
                                     : "(none)");
                    }
                }
                [NSApp sendEvent:e];
            }
            if (m.lost) {
                break;
            }
            /* Nothing is composited for a minimised window, so present
             * nothing.  Occlusion is deliberately *not* consulted: a
             * window can be fully covered and still want frames (a
             * screenshot, a screen recording, the moment it is
             * uncovered), and allowsNextDrawableTimeout is what keeps an
             * unseen window from wedging the loop -- nextDrawable gives
             * up and returns nil rather than blocking for ever. */
            if ([m.window isMiniaturized] || ![m.window isVisible]) {
                usleep(50 * 1000);
                continue;
            }
            metal_render_frame();
        }
    }

    metal_set_grab(false);
    cursor.inside = false;
    metal_cursor_sync();            /* never exit with the cursor hidden */
    metal_backend_request_shutdown();
    metal_glue_fb_done();

    /* Returning from here means returning from qemu_main's stand-in, which
     * unwinds main() while the main-loop thread is inside qemu_cleanup() --
     * the process dies under it.  The main loop calls exit(); outlive it. */
    for (;;) {
        pause();
    }
    return 0;                           /* not reached */
}

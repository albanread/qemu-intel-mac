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

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
/* Power: a frame is drawn only when there is something new -- the guest's
 * picture (MetalFbView.content), its pointer, the window, or an event --
 * and the loop sleeps between.  A board that cannot tell (content 0) is
 * drawn every frame, as it always was. */
static bool frame_wanted = true, frame_skipped;

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
                     METAL_BACKDROP_TILE, METAL_BACKDROP_SHADER };
#define METAL_BACKDROP_KEY   0x00B4C0B7u  /* 0x00BBGGRR: #B7C0B4 */
#define METAL_ICONBAR_PX     66u          /* guest px, square-pixel modes */
#define METAL_SCENE_MAX_FPS  15.0         /* animated backdrops, at most */
static struct {
    bool enabled;                       /* the gate: backdrop= other than off */
    int mode;                           /* MetalBackdrop: the scene, OFF for none */
    id<MTLTexture> image;               /* PICTURE and TILE */
    double image_scale;                 /* image px per guest px: 2 for @2x or 180 dpi */
    NSString *spec;                     /* as given: the menu ticks it */
    CFAbsoluteTime t0;                  /* the scene's clock */
    NSMenuItem *menu_item;              /* Machine > Backdrop, hidden when gated off */
    NSString *dir;                      /* backdrop-dir=: the disc's Wallpapers */

    /* SHADER: a scene from a .metal file, compiled at run time
     * (backdrop_scene_load), drawn into a texture of its own at no more
     * than its frame rate, and composited from there every frame */
    id<MTLRenderPipelineState> scene_pipe;
    id<MTLTexture> scene_tex[2];        /* [scene_cur] is the newest; two for feedback */
    int scene_cur;
    bool feedback;                      /* //! feedback: the last frame is an input */
    double fps;                         /* //! fps: 0 draws once */
    double resolution;                  /* //! resolution: of the window's pixels; 0 = guest */
    uint32_t state_rows;                /* //! state: rows kept below the visible scene */
    uint32_t scene_frame;               /* frames since the scene (or its size) began */
    uint32_t scene_w, scene_h;          /* the visible scene, texels */
    bool scene_valid;
    CFAbsoluteTime scene_next;          /* when an animated scene is next due */
    volatile bool scene_failed;         /* a GPU fault: drop to none */
    int guest_shows;                    /* the newest frame has tagged pixels */
    CFAbsoluteTime guest_checked;
} backdrop = { .mode = METAL_BACKDROP_OFF, .image_scale = 1.0 };

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
    int32_t hot_x, hot_y;           /* the hot spot, when has_hot */
    bool has_hot;
} ptr;

/* The shader constants, mirroring the Params struct in the MSL below. */
typedef struct {
    uint32_t dim[4];    /* xres, yres, pitch(bytes), bpp */
    uint32_t misc[4];   /* xoffset(px), yoffset(rows), pixo, unused */
    uint32_t post[4];   /* view w, view h, scaling, scanlines */
    uint32_t bd[4];     /* backdrop: key 0x00BBGGRR, time ms, icon bar px, on */
    uint32_t bx[4];     /* backdrop: output px per tile px x1000, unused */
    float bs[4];        /* backdrop shader scene: visible w, h in its texels */
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
    /* the positions sent lately, oldest overwritten: what the guest's
     * pointer, a VSync or two behind, may still be showing */
    struct {
        int gx, gy;
        double t;
    } sent[32];
    unsigned sent_next;
} mouse;

#define METAL_ECHO_SECONDS 0.25     /* how late the guest's pointer may be */

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
        mouse.sent[mouse.sent_next].gx = gx;
        mouse.sent[mouse.sent_next].gy = gy;
        mouse.sent[mouse.sent_next].t = CACurrentMediaTime();
        mouse.sent_next = (mouse.sent_next + 1) % 32;
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
"    uint4 misc;    /* xoffset(px), yoffset(rows), pixo, the desktop's look */\n"
"    uint4 post;    /* view w, view h, scaling, scanlines */\n"
"    uint4 bd;      /* backdrop: key 0x00BBGGRR, time ms, icon bar px, on */\n"
"    uint4 bx;      /* backdrop: output px per tile px x1000, unused */\n"
"    float4 bs;     /* backdrop shader scene: visible w, h in its texels */\n"
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
/* The desktop's look (design 37, Sprint 6; design 39): a desktop asked
 * for fewer colours is drawn at 32 bpp and shown in the colours of the
 * depth asked for -- the PC presenter's quantising pass.  Nearest colour,
 * as ColourTrans picks it (its weights, green most), no dither: 2 colours
 * by brightness, 4 greys, the Wimp's 16 colours, RISC OS's 256 (each gun
 * two bits and a tint shared by all three), 16 or 256 greys, and 4096,
 * 32K or 64K colours by truncating the guns.  look is block v2's: the
 * Log2BPP asked for plus one in its low four bits, &200 grey, &4000 64K,
 * &10000 4096 colours. */
"constant float3 wimp16[16] = {\n"
"    float3(1.0), float3(0.866667), float3(0.733333), float3(0.6), float3(0.466667), float3(0.333333),\n"
"    float3(0.2), float3(0.0), float3(0.0, 0.266667, 0.6), float3(0.933333, 0.933333, 0.0),\n"
"    float3(0.0, 0.8, 0.0), float3(0.866667, 0.0, 0.0), float3(0.933333, 0.933333, 0.733333),\n"
"    float3(0.333333, 0.533333, 0.0), float3(1.0, 0.733333, 0.0), float3(0.0, 0.733333, 1.0) };\n"
"static float lum(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }\n"
"static float3 look_of(float3 c, uint lk)\n"
"{\n"
"    int l2 = int(lk & 15u) - 1;\n"
"    if (l2 < 0) return c;\n"
"    bool grey = (lk & 0x200u) != 0;\n"
"    int fmt = (lk & 0x10000u) ? 2 : (lk & 0x4000u) ? 1 : 0;\n"
"    const float3 W = float3(2.0, 4.0, 1.0);\n"
"    float3 q = c;\n"
"    if (l2 >= 4) {\n"
"        float3 n = fmt == 1 ? float3(31.0, 63.0, 31.0) : fmt == 2 ? float3(15.0) : float3(31.0);\n"
"        q = floor(c * n + 0.5) / n;\n"
"    } else if (l2 == 0) {\n"
"        q = float3(lum(c) >= 0.5 ? 1.0 : 0.0);\n"
"    } else if (l2 == 1 || grey) {\n"
"        float n = l2 == 1 ? 3.0 : l2 == 2 ? 15.0 : 255.0;\n"
"        q = float3(floor(lum(c) * n + 0.5) / n);\n"
"    } else if (l2 == 2) {\n"
"        float best = 1e9;\n"
"        for (int i = 0; i < 16; i++) {\n"
"            float3 d = c - wimp16[i];\n"
"            float e = dot(d * d, W);\n"
"            if (e < best) { best = e; q = wimp16[i]; }\n"
"        }\n"
"    } else {\n"
"        float best = 1e9;\n"
"        for (int t = 0; t < 4; t++) {\n"
"            float3 hi = clamp(floor((c * 15.0 - float(t)) / 4.0 + 0.5), 0.0, 3.0);\n"
"            float3 p = (hi * 4.0 + float(t)) / 15.0, d = c - p;\n"
"            float e = dot(d * d, W);\n"
"            if (e < best) { best = e; q = p; }\n"
"        }\n"
"    }\n"
"    return q;\n"
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
"        return float4(look_of(c.rgb, P.misc.w), 1.0);\n"
"    }\n"
"    float3 bg;\n"
"    if (BACKDROP_IMAGE) {\n"
"        float2 isz = float2(bgimg.get_width(), bgimg.get_height());\n"
"        if (BACKDROP == 5) {\n"
"            /* a shader's scene, rendered at its own size (and no faster\n"
"             * than its frame rate) into a texture: stretched over the\n"
"             * view, bilinear by hand, since a feedback scene's float\n"
"             * texture need not be filterable */\n"
"            float2 sp = v.pos.xy / outPx * P.bs.xy - 0.5;\n"
"            float2 i = floor(sp), f = sp - i;\n"
"            int2 hi = int2(P.bs.xy) - 1;\n"
"            uint2 a = uint2(clamp(int2(i), int2(0), hi));\n"
"            uint2 b = uint2(clamp(int2(i) + 1, int2(0), hi));\n"
"            float3 c00 = bgimg.read(a).rgb, c10 = bgimg.read(uint2(b.x, a.y)).rgb;\n"
"            float3 c01 = bgimg.read(uint2(a.x, b.y)).rgb, c11 = bgimg.read(b).rgb;\n"
"            bg = saturate(mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y));\n"
"        } else if (BACKDROP == 4) {\n"
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
"    return float4(look_of(c.rgb + bg * (1.0 - c.a), P.misc.w), 1.0);\n"
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

    /* Resize about the top-left corner, where the user put the window:
     * a mode change must not move it.  Only keep it on the screen. */
    {
        NSRect before = [m.window frame];
        NSPoint top_left = NSMakePoint(NSMinX(before), NSMaxY(before));
        NSRect after;

        [m.window setContentSize:want];
        [m.window setFrameTopLeftPoint:top_left];
        after = [m.window constrainFrameRect:[m.window frame]
                                    toScreen:[m.window screen]];
        [m.window setFrame:after display:YES];
    }
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
 * bar the acorn is centred above, whether the layer is on at all, the
 * tile's size (output pixels per guest pixel over the image's own
 * density), and a shader scene's visible size. */
static void backdrop_params(MetalParams *p, double px_per_guest_px)
{
    p->bd[0] = METAL_BACKDROP_KEY;
    p->bd[1] = (uint32_t)(fmod(CFAbsoluteTimeGetCurrent() - backdrop.t0,
                               1800.0) * 1000.0);
    p->bd[2] = METAL_ICONBAR_PX;
    p->bd[3] = backdrop.mode != METAL_BACKDROP_OFF;
    p->bx[0] = (uint32_t)(px_per_guest_px / backdrop.image_scale * 1000.0 + 0.5);
    p->bx[1] = p->bx[2] = p->bx[3] = 0;
    p->bs[0] = (float)MAX(backdrop.scene_w, 1u);
    p->bs[1] = (float)MAX(backdrop.scene_h, 1u);
    p->bs[2] = p->bs[3] = 0.0f;
}

/* Whether the window is dense enough for a double-density image: two or
 * more of the screen's pixels to a RISC OS pixel (a Retina window, as
 * the app opens it). */
static bool backdrop_dense(void)
{
    if (fb.up && fb.yres && m.layer.drawableSize.height > 0) {
        return m.layer.drawableSize.height / (double)fb.yres >= 1.75;
    }
    return fit_zoom >= 2 || (m.layer && m.layer.contentsScale >= 2.0);
}

/* An RGBA8 image, gamma encoded as the guest's pixels are, as a texture */
static id<MTLTexture> backdrop_texture(const uint8_t *rgba, size_t w, size_t h)
{
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:w height:h mipmapped:NO];
    id<MTLTexture> tex;

    td.usage = MTLTextureUsageShaderRead;
    tex = [m.device newTextureWithDescriptor:td];
    [tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
             withBytes:rgba bytesPerRow:w * 4];
    return tex;
}

/* ---- RISC OS sprite files --------------------------------------------
 *
 * The disc's wallpapers (Resources/Wallpapers) are sprite files, ,ff9 on
 * the Mac side, which ImageIO cannot read; so they are read here.  A
 * sprite file is a sprite area without its first word: the count, the
 * offset (+4) of the first sprite, the offset of the free space.  Each
 * sprite is a 44-byte header -- next, name[12], width in words - 1,
 * height - 1, first and last bit used, image and mask offsets, the mode
 * -- then the palette, if any (pairs of &BBGGRR00 words), the image and
 * the mask.  Read here:
 *
 *   - old sprites, whose mode is a screen mode number (its depth and its
 *     pixel shape, 90 x 45 dpi for mode 12 or 15), with a mask as deep
 *     as the image and a left-hand wastage;
 *   - new sprites (RISC OS 3.5), whose mode word gives the type -- 1, 2,
 *     4 or 8 bpp, 16 bpp 1:5:5:5 or 32 bpp -- and the dpi in x and y,
 *     with a 1 bpp mask, or an 8 bpp alpha mask (RISC OS 5: bit 31);
 *   - RISC OS 5's extended mode words (type 15): the type in bits 20-26
 *     (adding 16 bpp 5:6:5 and 4:4:4:4), the pixel shape as eigen
 *     factors, and the RGB-order and alpha flags;
 *   - a sprite's own palette, else the desktop's default for its depth
 *     (the Wimp's 16 colours, the 256-colour mode's).
 *
 * Masked pixels are laid on the key colour, as a transparent PNG is.
 * The result has square pixels: a 90 x 45 dpi sprite has its rows
 * doubled.  Its density is its dpi: 90 is a RISC OS pixel, 180 (as a
 * theme's Sprites11 has) two to one. */

static uint32_t spr_word(const uint8_t *b, size_t len, size_t off, bool *ok)
{
    if (off > len || len - off < 4) {
        *ok = false;
        return 0;
    }
    return (uint32_t)b[off] | (uint32_t)b[off + 1] << 8
         | (uint32_t)b[off + 2] << 16 | (uint32_t)b[off + 3] << 24;
}

typedef struct {
    int bpp;            /* 1, 2, 4, 8, 16 or 32 */
    int type;           /* the sprite type: 1..6, 10 (5:6:5), 16 (4:4:4:4) */
    int xdpi, ydpi;
    bool old;           /* a mode number: mask as deep as the image */
    bool alpha_mask;    /* an 8 bpp alpha mask */
    bool rgb;           /* red in the low bits (RISC OS 5's RGB order) */
    bool alpha;         /* the pixel's top bits are alpha, not transfer */
} SprMode;

static bool spr_mode(uint32_t mode, SprMode *sm)
{
    /* the numbered modes a sprite may name: depth, and eigen factors */
    static const struct { uint8_t mode, bpp, xeig, yeig; } modes[] = {
        { 0, 1, 1, 2 }, { 1, 2, 2, 2 }, { 2, 4, 3, 2 }, { 4, 1, 2, 2 },
        { 5, 2, 3, 2 }, { 8, 2, 1, 2 }, { 9, 4, 2, 2 }, { 10, 8, 3, 2 },
        { 12, 4, 1, 2 }, { 13, 8, 2, 2 }, { 15, 8, 1, 2 }, { 16, 4, 1, 2 },
        { 17, 4, 1, 2 }, { 18, 1, 1, 1 }, { 19, 2, 1, 1 }, { 20, 4, 1, 1 },
        { 21, 8, 1, 1 }, { 24, 8, 1, 2 }, { 25, 1, 1, 1 }, { 26, 2, 1, 1 },
        { 27, 4, 1, 1 }, { 28, 8, 1, 1 }, { 31, 4, 1, 1 }, { 32, 8, 1, 1 },
        { 33, 1, 1, 2 }, { 34, 2, 1, 2 }, { 35, 4, 1, 2 }, { 36, 8, 1, 2 },
    };
    int t;

    memset(sm, 0, sizeof(*sm));
    if (mode < 256) {
        size_t i;

        for (i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
            if (modes[i].mode == mode) {
                sm->bpp = modes[i].bpp;
                sm->type = sm->bpp == 1 ? 1 : sm->bpp == 2 ? 2 : sm->bpp == 4 ? 3 : 4;
                sm->xdpi = 180 >> modes[i].xeig;
                sm->ydpi = 180 >> modes[i].yeig;
                sm->old = true;
                return true;
            }
        }
        return false;
    }
    sm->alpha_mask = (mode >> 31) & 1;
    t = (mode >> 27) & 15;
    if (t == 15) {                      /* RISC OS 5's extended mode word */
        t = (mode >> 20) & 127;
        sm->xdpi = 180 >> ((mode >> 4) & 3);
        sm->ydpi = 180 >> ((mode >> 6) & 3);
        sm->rgb = (mode >> 14) & 1;
        sm->alpha = (mode >> 15) & 1;
    } else {
        sm->xdpi = (mode >> 1) & 0x1FFF;
        sm->ydpi = (mode >> 14) & 0x1FFF;
    }
    sm->type = t;
    switch (t) {
    case 1: sm->bpp = 1; break;
    case 2: sm->bpp = 2; break;
    case 3: sm->bpp = 4; break;
    case 4: sm->bpp = 8; break;
    case 5: case 10: case 16: sm->bpp = 16; break;
    case 6: sm->bpp = 32; break;
    default: return false;              /* CMYK, 24 bpp packed, JPEG: not wallpapers */
    }
    if (sm->xdpi <= 0 || sm->ydpi <= 0) {
        sm->xdpi = sm->ydpi = 90;
    }
    return true;
}

static uint32_t spr_default_colour(int bpp, uint32_t i)   /* 0x00RRGGBB */
{
    static const uint32_t wimp16[16] = {
        0xFFFFFF, 0xDDDDDD, 0xBBBBBB, 0x999999, 0x777777, 0x555555, 0x333333,
        0x000000, 0x004499, 0xEEEE00, 0x00CC00, 0xDD0000, 0xEEEEBB, 0x558800,
        0xFFBB00, 0x00BBFF };
    static const uint32_t grey4[4] = { 0xFFFFFF, 0xBBBBBB, 0x777777, 0x000000 };
    uint32_t t, r, g, b;

    switch (bpp) {
    case 1: return i ? 0x000000 : 0xFFFFFF;
    case 2: return grey4[i & 3];
    case 4: return wimp16[i & 15];
    default:
        /* the 256-colour mode's: bits 0-1 a tint, 2 R2, 3 B2, 4 R3,
         * 5 G2, 6 G3, 7 B3 */
        t = i & 3;
        r = (((i >> 2) & 1) | ((i >> 4) & 1) << 1) << 2 | t;
        g = (((i >> 5) & 1) | ((i >> 6) & 1) << 1) << 2 | t;
        b = (((i >> 3) & 1) | ((i >> 7) & 1) << 1) << 2 | t;
        return (r * 17) << 16 | (g * 17) << 8 | (b * 17);
    }
}

/* One sprite at offset at in the file, as RGBA8 with square pixels laid
 * on the key colour; NULL if it cannot be read.  *dpi is its density. */
static uint8_t *spr_decode(const uint8_t *f, size_t len, size_t at,
                           size_t *ow, size_t *oh, int *dpi)
{
    bool ok = true;
    uint32_t wwords = spr_word(f, len, at + 16, &ok);
    uint32_t hm1 = spr_word(f, len, at + 20, &ok);
    uint32_t fbit = spr_word(f, len, at + 24, &ok) & 31;
    uint32_t lbit = spr_word(f, len, at + 28, &ok) & 31;
    uint32_t img = spr_word(f, len, at + 32, &ok);
    uint32_t msk = spr_word(f, len, at + 36, &ok);
    uint32_t modew = spr_word(f, len, at + 40, &ok);
    uint32_t pal[256];
    SprMode sm;
    size_t w, h, stride, mstride, x, y, rx = 1, ry = 1, outw, outh;
    uint8_t *px, *out;
    bool has_mask, has_pal = false;

    if (!ok || !spr_mode(modew, &sm) || wwords >= 8192 || hm1 >= 8192) {
        return NULL;
    }
    if (!sm.old) {
        fbit = 0;
    }
    stride = ((size_t)wwords + 1) * 4;
    w = (((size_t)wwords + 1) * 32 - fbit - (31 - lbit)) / (size_t)sm.bpp;
    h = (size_t)hm1 + 1;
    if (!w || at + img + stride * h > len) {
        return NULL;
    }
    has_mask = msk != img;
    if (sm.bpp <= 8) {
        uint32_t entries = (MIN(img, has_mask ? msk : img) - MIN(img, 44u)) / 8;
        uint32_t need = 1u << sm.bpp, i;

        if (entries >= need) {
            for (i = 0; i < need; i++) {
                uint32_t c = spr_word(f, len, at + 44 + 8 * i, &ok);   /* &BBGGRR00 */

                pal[i] = ((c >> 8) & 0xFF) << 16 | ((c >> 16) & 0xFF) << 8 | (c >> 24);
            }
            has_pal = ok;
        }
        if (!has_pal) {
            for (i = 0; i < need; i++) {
                pal[i] = spr_default_colour(sm.bpp, i);
            }
        }
    }
    mstride = sm.alpha_mask ? (w + 3) / 4 * 4 : sm.old ? stride : (w + 31) / 32 * 4;
    if (has_mask && at + msk + mstride * h > len) {
        has_mask = false;               /* a mask past the end: ignore it */
    }

    /* square the pixels: a 90 x 45 dpi sprite has each row twice */
    if (sm.ydpi < sm.xdpi && sm.xdpi % sm.ydpi == 0) {
        ry = (size_t)(sm.xdpi / sm.ydpi);
    } else if (sm.xdpi < sm.ydpi && sm.ydpi % sm.xdpi == 0) {
        rx = (size_t)(sm.ydpi / sm.xdpi);
    }
    if (rx > 4 || ry > 4) {
        rx = ry = 1;
    }
    *dpi = MAX(sm.xdpi, sm.ydpi);
    outw = w * rx;
    outh = h * ry;
    px = malloc(w * h * 4);
    out = malloc(outw * outh * 4);
    if (!px || !out) {
        free(px);
        free(out);
        return NULL;
    }
    for (y = 0; y < h; y++) {
        const uint8_t *row = f + at + img + y * stride;
        const uint8_t *mrow = f + at + msk + y * mstride;

        for (x = 0; x < w; x++) {
            size_t bit = fbit + x * (size_t)sm.bpp;
            uint32_t word = (uint32_t)row[(bit >> 5) * 4] | (uint32_t)row[(bit >> 5) * 4 + 1] << 8
                          | (uint32_t)row[(bit >> 5) * 4 + 2] << 16
                          | (uint32_t)row[(bit >> 5) * 4 + 3] << 24;
            uint32_t r, g, b, a = 255, v;
            uint8_t *o = px + (y * w + x) * 4;

            if (sm.bpp == 32) {
                r = word & 0xFF;
                g = (word >> 8) & 0xFF;
                b = (word >> 16) & 0xFF;
                if (sm.alpha) {
                    a = word >> 24;
                }
            } else if (sm.bpp == 16) {
                v = (word >> (bit & 31)) & 0xFFFF;
                if (sm.type == 10) {            /* 5:6:5 */
                    r = (v & 31) * 255 / 31;
                    g = ((v >> 5) & 63) * 255 / 63;
                    b = ((v >> 11) & 31) * 255 / 31;
                } else if (sm.type == 16) {     /* 4:4:4:4 */
                    r = (v & 15) * 17;
                    g = ((v >> 4) & 15) * 17;
                    b = ((v >> 8) & 15) * 17;
                    if (sm.alpha) {
                        a = ((v >> 12) & 15) * 17;
                    }
                } else {                        /* 1:5:5:5 */
                    r = (v & 31) * 255 / 31;
                    g = ((v >> 5) & 31) * 255 / 31;
                    b = ((v >> 10) & 31) * 255 / 31;
                    if (sm.alpha) {
                        a = (v >> 15) ? 255 : 0;
                    }
                }
            } else {
                uint32_t c;

                v = (word >> (bit & 31)) & ((1u << sm.bpp) - 1);
                c = pal[v];
                r = c >> 16;
                g = (c >> 8) & 0xFF;
                b = c & 0xFF;
            }
            if (sm.rgb && sm.bpp >= 16) {
                uint32_t t = r;
                r = b;
                b = t;
            }
            if (has_mask) {
                if (sm.alpha_mask) {
                    a = mrow[x];
                } else if (sm.old) {
                    uint32_t mw = (uint32_t)mrow[(bit >> 5) * 4] | (uint32_t)mrow[(bit >> 5) * 4 + 1] << 8
                                | (uint32_t)mrow[(bit >> 5) * 4 + 2] << 16
                                | (uint32_t)mrow[(bit >> 5) * 4 + 3] << 24;
                    uint32_t mm = sm.bpp == 32 ? 0xFFFFFFFFu : (1u << sm.bpp) - 1;

                    a = (mw >> (bit & 31)) & mm ? 255 : 0;
                } else {
                    a = (mrow[x >> 3] >> (x & 7)) & 1 ? 255 : 0;
                }
            }
            /* on the key colour, #B7C0B4 */
            o[0] = (uint8_t)((r * a + 0xB7 * (255 - a)) / 255);
            o[1] = (uint8_t)((g * a + 0xC0 * (255 - a)) / 255);
            o[2] = (uint8_t)((b * a + 0xB4 * (255 - a)) / 255);
            o[3] = 255;
        }
    }
    for (y = 0; y < outh; y++) {
        for (x = 0; x < outw; x++) {
            memcpy(out + (y * outw + x) * 4, px + ((y / ry) * w + x / rx) * 4, 4);
        }
    }
    free(px);
    *ow = outw;
    *oh = outh;
    return out;
}

/* A sprite file's picture: its first sprite -- or, where the file holds
 * the same sprite at several densities (NAME, NAME22, NAME11, as a
 * theme's Tools file does), the one that suits the window: the densest
 * up to 180 dpi on a Retina window, else the one nearest 90 dpi. */
static uint8_t *spr_load(const char *path, size_t *w, size_t *h, int *dpi)
{
    NSData *d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    const uint8_t *f = d.bytes;
    size_t len = d.length, at, best_at = 0;
    bool ok = true, dense = backdrop_dense();
    uint32_t count, first, i;
    char base[13] = "";
    int best = -1;

    if (!d || len < 12 + 44) {
        return NULL;
    }
    count = spr_word(f, len, 0, &ok);
    first = spr_word(f, len, 4, &ok);
    if (!ok || !count || first < 16 || count > 100000) {
        return NULL;
    }
    at = first - 4;
    for (i = 0; i < count && at + 44 <= len; i++) {
        uint32_t next = spr_word(f, len, at, &ok);
        uint32_t modew = spr_word(f, len, at + 40, &ok);
        char name[13];
        SprMode sm;
        size_t n;

        memcpy(name, f + at + 4, 12);
        name[12] = 0;
        for (n = 0; name[n]; n++) {
            name[n] = (char)tolower((unsigned char)name[n]);
        }
        if (i == 0) {
            strcpy(base, name);
            n = strlen(base);
            if (n > 2 && (!strcmp(base + n - 2, "11") || !strcmp(base + n - 2, "22"))) {
                base[n - 2] = 0;
            }
        }
        if (spr_mode(modew, &sm)) {
            size_t bl = strlen(base);
            bool same = !strncmp(name, base, bl)
                        && (!name[bl] || !strcmp(name + bl, "11") || !strcmp(name + bl, "22"));
            int dp = MAX(sm.xdpi, sm.ydpi);
            int score = dense ? (dp <= 180 ? dp : 0) : (dp <= 90 ? dp + 1000 : 1000 - dp);

            if (same && score > best) {
                best = score;
                best_at = at;
            }
        }
        if (!ok || next < 44) {
            break;
        }
        at += next;
    }
    return best >= 0 ? spr_decode(f, len, best_at, w, h, dpi) : NULL;
}

/* An image file by ImageIO, as RGBA8 on the key colour; NULL if it
 * cannot read it */
static uint8_t *img_load(const char *path, size_t *ow, size_t *oh)
{
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    CGImageSourceRef src = CGImageSourceCreateWithURL((CFURLRef)url, NULL);
    CGImageRef img = src ? CGImageSourceCreateImageAtIndex(src, 0, NULL) : NULL;
    uint8_t *buf = NULL;
    size_t w, h;

    if (src) {
        CFRelease(src);
    }
    if (!img) {
        return NULL;
    }
    w = CGImageGetWidth(img);
    h = CGImageGetHeight(img);
    if (w > 4096 || h > 4096) {             /* ample for any window; saves VRAM */
        double k = 4096.0 / (double)MAX(w, h);
        w = MAX((size_t)(w * k), 1);
        h = MAX((size_t)(h * k), 1);
    }
    if (w && h) {
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef ctx;

        buf = calloc(w * h, 4);
        ctx = buf ? CGBitmapContextCreate(buf, w, h, 8, w * 4, cs,
                                          kCGImageAlphaPremultipliedLast
                                          | kCGBitmapByteOrder32Big) : NULL;
        if (ctx) {
            CGContextSetRGBFillColor(ctx, 0xB7 / 255.0, 0xC0 / 255.0,
                                     0xB4 / 255.0, 1.0);
            CGContextFillRect(ctx, CGRectMake(0, 0, w, h));
            CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
            CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
            CGContextRelease(ctx);
        } else {
            free(buf);
            buf = NULL;
        }
        CGColorSpaceRelease(cs);
    }
    CGImageRelease(img);
    *ow = w;
    *oh = h;
    return buf;
}

/* Whether a leaf names a sprite file: ,ff9 (HostFS's name for a file of
 * type Sprite), or .ff9 or .spr */
static bool is_sprite_leaf(NSString *leaf)
{
    NSString *l = [leaf lowercaseString];

    return [l hasSuffix:@",ff9"] || [l hasSuffix:@".ff9"] || [l hasSuffix:@".spr"];
}

/* A leaf without HostFS's ,xxx file type */
static NSString *leaf_untyped(NSString *leaf)
{
    NSUInteger n = leaf.length;

    if (n > 4 && [leaf characterAtIndex:n - 4] == ',') {
        return [leaf substringToIndex:n - 4];
    }
    return leaf;
}

static bool is_shader_leaf(NSString *leaf)
{
    return [[[leaf_untyped(leaf) pathExtension] lowercaseString] isEqualToString:@"metal"];
}

static bool is_image_leaf(NSString *leaf)
{
    static NSSet *ext;

    if (!ext) {
        ext = [[NSSet alloc] initWithObjects:@"png", @"jpg", @"jpeg", @"heic", @"heif",
                                             @"tif", @"tiff", @"gif", @"bmp", @"webp", nil];
    }
    return [ext containsObject:[[leaf_untyped(leaf) pathExtension] lowercaseString]];
}

/* The name a menu shows: the leaf with no file type and no extension */
static NSString *leaf_title(NSString *leaf)
{
    NSString *t = leaf_untyped(leaf);

    if (is_sprite_leaf(leaf) && ![t isEqualToString:leaf]) {
        return t;                       /* Baize,ff9: Baize */
    }
    return [t stringByDeletingPathExtension];
}

/* The double-density sibling of an image or sprite file, if there is
 * one: NAME@2x for either (Apple's name), NAME11 for a sprite file
 * (RISC OS's, as in Sprites11) -- same folder, same type */
static NSString *backdrop_dense_sibling(NSString *path)
{
    NSString *dir = [path stringByDeletingLastPathComponent];
    NSString *leaf = [path lastPathComponent];
    NSString *untyped = leaf_untyped(leaf);
    NSString *type = [leaf substringFromIndex:untyped.length];      /* ,ff9 or "" */
    NSString *ext = [untyped pathExtension];
    NSString *stem = [untyped stringByDeletingPathExtension];
    NSString *dot = ext.length ? [@"." stringByAppendingString:ext] : @"";
    NSMutableArray *names = [NSMutableArray array];
    NSFileManager *fm = [NSFileManager defaultManager];

    [names addObject:[NSString stringWithFormat:@"%@@2x%@%@", stem, dot, type]];
    if (is_sprite_leaf(leaf)) {
        [names addObject:[NSString stringWithFormat:@"%@11%@%@", stem, dot, type]];
    }
    for (NSString *n in names) {
        NSString *p = [dir stringByAppendingPathComponent:n];

        if (![n isEqualToString:leaf] && [fm fileExistsAtPath:p]) {
            return p;
        }
    }
    return nil;
}

/* An image for a tile or a picture: a sprite file read here, anything
 * else by ImageIO, drawn over the key colour (so a transparent PNG sits
 * on sage) into an RGBA8 texture -- the same gamma-encoded space as the
 * guest's pixels.  On a Retina window a double-density sibling is taken
 * if there is one.  *scale is the image's pixels per RISC OS pixel:
 * 2 for NAME@2x, dpi / 90 for a sprite; *w and *h its size. */
static id<MTLTexture> backdrop_load_image(NSString *path, double *scale,
                                          size_t *w, size_t *h)
{
    NSString *dense = backdrop_dense() ? backdrop_dense_sibling(path) : nil;
    NSString *use = dense ? dense : path;
    const char *p = [use fileSystemRepresentation];
    uint8_t *rgba = NULL;
    int dpi = 90;
    id<MTLTexture> tex = nil;

    *scale = 1.0;
    if (is_sprite_leaf([use lastPathComponent])) {
        rgba = spr_load(p, w, h, &dpi);
        *scale = dpi / 90.0;
    } else {
        rgba = img_load(p, w, h);
        if (!rgba) {
            rgba = spr_load(p, w, h, &dpi);   /* a sprite file under another name */
            *scale = dpi / 90.0;
        } else if ([[[use lastPathComponent] stringByDeletingPathExtension] hasSuffix:@"@2x"]) {
            *scale = 2.0;
        }
    }
    if (*scale < 0.25 || *scale > 8.0) {
        *scale = 1.0;
    }
    if (rgba) {
        tex = backdrop_texture(rgba, *w, *h);
        free(rgba);
        if (dense) {
            metal_log("backdrop: the double-density %s", [use fileSystemRepresentation]);
        }
    } else if (dense) {
        return backdrop_load_image(path, scale, w, h);  /* the sibling did not read */
    }
    return tex;
}

/* ---- Shader scenes -------------------------------------------------
 *
 * A scene is a Metal shading language file in the backdrops folder's
 * Shaders (the disc's Resources/Wallpapers/Shaders), compiled when it
 * is chosen.  The file defines one function, which gives the colour of
 * one pixel of the scene:
 *
 *     float3 scene(SceneIn in)
 *
 * SceneIn, and a few helpers (hash, noise, fbm, dither), are defined
 * before the file; see SCENE_PRELUDE below and the ReadMe in Shaders.
 * Lines beginning //! set how it is drawn:
 *
 *     //! fps 12            frames a second, at most 15; 0 draws it once
 *     //! resolution 0.5    the scene's pixels as a fraction of the
 *                           window's; "guest" is one per RISC OS pixel
 *     //! feedback          the scene sees its last frame: then it is
 *                           float4 scene(SceneIn in, texture2d<float> previous),
 *                           kept as 32-bit floats; .rgb is shown, .a is
 *                           the shader's own
 *     //! state 1           with feedback, rows kept below the visible
 *                           scene for the shader's own state
 *
 * A file that does not compile leaves the backdrop at none, with the
 * compiler's message in the log; a scene that faults on the GPU does
 * the same.  The display goes on either way. */
static NSString * const SCENE_PRELUDE = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct SceneIn {\n"
"    float2 pos;      /* this pixel's centre, in scene pixels; (0,0) is the top left */\n"
"    float2 size;     /* the visible scene, in scene pixels */\n"
"    float  scale;    /* scene pixels to a RISC OS pixel */\n"
"    float  time;     /* seconds since the scene was chosen (wraps after a day) */\n"
"    float  daytime;  /* the Mac's local time of day, in hours, 0 to 24 */\n"
"    uint   frame;    /* frames since the scene began, or its size changed: 0 is the first */\n"
"    float3 key;      /* the backdrop key colour, #B7C0B4, the desktop's sage */\n"
"    float  iconbar;  /* the icon bar's height, in scene pixels */\n"
"    float2 guest;    /* the RISC OS screen, in its own pixels */\n"
"    uint   rows;     /* the state rows below the visible scene (//! state) */\n"
"};\n"
"\n"
"struct SceneU { float4 a, b, c; uint4 d; };\n"
"struct SceneV { float4 pos [[position]]; };\n"
"\n"
"vertex SceneV scene_vs(uint id [[vertex_id]])\n"
"{\n"
"    SceneV o;\n"
"    float2 t = float2((id << 1) & 2, id & 2);\n"
"    o.pos = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);\n"
"    return o;\n"
"}\n"
"\n"
"/* a hash of a point: 0 to 1, the same for the same point */\n"
"static float hash12(float2 p)\n"
"{\n"
"    uint2 q = uint2(int2(floor(p))) * uint2(1597334673u, 3812015801u);\n"
"    uint n = (q.x ^ q.y) * 1597334673u;\n"
"    return float(n >> 8) * (1.0 / 16777216.0);\n"
"}\n"
"static float2 hash22(float2 p)\n"
"{\n"
"    return float2(hash12(p), hash12(p + float2(37.0, 17.0)));\n"
"}\n"
"/* value noise, 0 to 1, smooth, a feature a unit across */\n"
"static float noise(float2 p)\n"
"{\n"
"    float2 i = floor(p), f = fract(p);\n"
"    float2 u = f * f * (3.0 - 2.0 * f);\n"
"    return mix(mix(hash12(i), hash12(i + float2(1, 0)), u.x),\n"
"               mix(hash12(i + float2(0, 1)), hash12(i + float2(1, 1)), u.x), u.y);\n"
"}\n"
"/* fractal noise: octaves of noise, each twice as fine and half as strong */\n"
"static float fbm(float2 p, int octaves)\n"
"{\n"
"    float v = 0.0, a = 0.5;\n"
"    for (int i = 0; i < octaves; i++) {\n"
"        v += a * noise(p);\n"
"        p = float2x2(1.6, 1.2, -1.2, 1.6) * p;\n"
"        a *= 0.5;\n"
"    }\n"
"    return v;\n"
"}\n"
"/* less than one level of 8-bit noise, to add to a gentle gradient so it\n"
" * does not band */\n"
"static float3 dither(float2 pos)\n"
"{\n"
"    float n = fract(52.9829189 * fract(dot(pos, float2(0.06711056, 0.00583715))));\n"
"    return float3((n - 0.5) / 255.0);\n"
"}\n"
"/* the previous frame's texel at a scene position (feedback scenes) */\n"
"static float4 fetch(texture2d<float> t, float2 pos)\n"
"{\n"
"    int2 hi = int2(t.get_width(), t.get_height()) - 1;\n"
"    return t.read(uint2(clamp(int2(floor(pos)), int2(0), hi)));\n"
"}\n"
"\n";

static NSString * const SCENE_POSTLUDE = @"\n"
"#if BACKDROP_FEEDBACK\n"
"fragment float4 scene_fs(SceneV v [[stage_in]], constant SceneU &U [[buffer(0)]],\n"
"                         texture2d<float> previous [[texture(0)]])\n"
"#else\n"
"fragment float4 scene_fs(SceneV v [[stage_in]], constant SceneU &U [[buffer(0)]])\n"
"#endif\n"
"{\n"
"    SceneIn in;\n"
"    in.pos = v.pos.xy;\n"
"    in.size = U.a.xy;\n"
"    in.scale = U.a.z;\n"
"    in.time = U.a.w;\n"
"    in.daytime = U.b.x;\n"
"    in.iconbar = U.b.y;\n"
"    in.guest = U.b.zw;\n"
"    in.key = U.c.xyz;\n"
"    in.frame = U.d.x;\n"
"    in.rows = U.d.y;\n"
"#if BACKDROP_FEEDBACK\n"
"    return scene(in, previous);\n"
"#else\n"
"    return float4(saturate(scene(in)), 1.0);\n"
"#endif\n"
"}\n";

typedef struct {
    float a[4];         /* size w, h; scale; time */
    float b[4];         /* daytime; icon bar; guest w, h */
    float c[4];         /* key r, g, b; unused */
    uint32_t d[4];      /* frame; state rows; unused */
} SceneParams;

static void backdrop_scene_release(void)
{
    [backdrop.scene_pipe release];
    backdrop.scene_pipe = nil;
    for (int i = 0; i < 2; i++) {
        [backdrop.scene_tex[i] release];
        backdrop.scene_tex[i] = nil;
    }
    backdrop.scene_cur = 0;
    backdrop.scene_valid = false;
    backdrop.scene_w = backdrop.scene_h = 0;
    backdrop.scene_failed = false;
}

/* Compile a scene file; false (and the reason in the log) if it will not */
static bool backdrop_scene_load(NSString *path)
{
    NSError *err = nil;
    NSString *text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding
                                                  error:&err];
    NSString *leaf = [path lastPathComponent], *src;
    MTLCompileOptions *opts;
    id<MTLLibrary> lib;
    id<MTLFunction> vs, fs;
    MTLRenderPipelineDescriptor *pd;
    double fps = 12.0, resolution = 1.0;
    bool feedback = false;
    uint32_t rows = 0;

    if (!text) {
        text = [NSString stringWithContentsOfFile:path encoding:NSISOLatin1StringEncoding
                                            error:&err];
    }
    if (!text || text.length > 512 * 1024) {
        metal_log("backdrop: cannot read the shader %s", [path fileSystemRepresentation]);
        return false;
    }
    for (NSString *line in [text componentsSeparatedByCharactersInSet:
                            [NSCharacterSet newlineCharacterSet]]) {
        NSString *t = [line stringByTrimmingCharactersInSet:
                       [NSCharacterSet whitespaceCharacterSet]];
        NSArray *w;
        NSString *key, *val;

        if (![t hasPrefix:@"//!"]) {
            continue;
        }
        w = [[[t substringFromIndex:3] stringByTrimmingCharactersInSet:
              [NSCharacterSet whitespaceCharacterSet]]
             componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        key = w.count ? [w[0] lowercaseString] : @"";
        val = w.count > 1 ? w[1] : @"";
        if ([key isEqualToString:@"fps"]) {
            fps = MIN(MAX(val.doubleValue, 0.0), METAL_SCENE_MAX_FPS);
        } else if ([key isEqualToString:@"feedback"]) {
            feedback = true;
        } else if ([key isEqualToString:@"resolution"]) {
            resolution = [[val lowercaseString] isEqualToString:@"guest"]
                         ? 0.0 : MIN(MAX(val.doubleValue, 0.125), 1.0);
        } else if ([key isEqualToString:@"state"]) {
            rows = (uint32_t)MIN(MAX(val.intValue, 0), 64);
        }
    }
    if (!feedback) {
        rows = 0;
    }
    src = [NSString stringWithFormat:@"%@#define BACKDROP_FEEDBACK %d\n#line 1 \"%@\"\n%@\n%@",
           SCENE_PRELUDE, feedback ? 1 : 0, [leaf stringByReplacingOccurrencesOfString:@"\""
                                                                            withString:@"'"],
           text, SCENE_POSTLUDE];
    opts = [[MTLCompileOptions alloc] init];
    lib = [m.device newLibraryWithSource:src options:opts error:&err];
    [opts release];
    if (!lib) {
        metal_log("backdrop: the shader %s does not compile, so none:\n%s",
                  [path fileSystemRepresentation],
                  [[err localizedDescription] UTF8String]);
        return false;
    }
    vs = [lib newFunctionWithName:@"scene_vs"];
    fs = [lib newFunctionWithName:@"scene_fs"];
    [lib release];
    if (!vs || !fs) {
        metal_log("backdrop: the shader %s has no scene function", [path fileSystemRepresentation]);
        [vs release];
        [fs release];
        return false;
    }
    pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    pd.colorAttachments[0].pixelFormat = feedback ? MTLPixelFormatRGBA32Float
                                                  : MTLPixelFormatRGBA8Unorm;
    backdrop_scene_release();
    backdrop.scene_pipe = [m.device newRenderPipelineStateWithDescriptor:pd error:&err];
    [pd release];
    [vs release];
    [fs release];
    if (!backdrop.scene_pipe) {
        metal_log("backdrop: the shader %s: no pipeline, so none: %s",
                  [path fileSystemRepresentation], [[err localizedDescription] UTF8String]);
        return false;
    }
    backdrop.feedback = feedback;
    backdrop.fps = fps;
    backdrop.resolution = resolution;
    backdrop.state_rows = rows;
    metal_log("backdrop: shader %s: %.0f fps, resolution %s, feedback %d, state rows %u",
              [path fileSystemRepresentation], fps,
              resolution > 0 ? [[NSString stringWithFormat:@"%.3g of the window's", resolution]
                                UTF8String] : "a RISC OS pixel's",
              (int)feedback, rows);
    return true;
}

/* The scene's size in texels for this window and this screen mode */
static void backdrop_scene_size(CGSize target, uint32_t gw, uint32_t gh,
                                uint32_t *w, uint32_t *h)
{
    if (backdrop.resolution <= 0) {
        *w = gw;
        *h = gh;
    } else {
        *w = (uint32_t)lround(target.width * backdrop.resolution);
        *h = (uint32_t)lround(target.height * backdrop.resolution);
    }
    *w = MIN(MAX(*w, 1u), 8192u);
    *h = MIN(MAX(*h, 1u), 8192u - backdrop.state_rows);
}

/* Draw the scene's next frame, in a command buffer of its own -- so a
 * shader that faults on the GPU loses its own frame, never the desktop's
 * -- committed ahead of the frame that shows it. */
static void backdrop_scene_render(CGSize target, uint32_t gw, uint32_t gh)
{
    id<MTLCommandBuffer> cb;
    MTLRenderPassDescriptor *rp;
    id<MTLRenderCommandEncoder> enc;
    SceneParams sp;
    uint32_t w, h, n = backdrop.feedback ? 2 : 1;
    double now = CFAbsoluteTimeGetCurrent();
    time_t tt = time(NULL);
    struct tm lt;

    if (!backdrop.scene_pipe || !gw || !gh) {
        return;
    }
    backdrop_scene_size(target, gw, gh, &w, &h);
    cb = [m.queue commandBuffer];
    if (!backdrop.scene_tex[0] || w != backdrop.scene_w || h != backdrop.scene_h) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:backdrop.feedback ? MTLPixelFormatRGBA32Float
                                                                 : MTLPixelFormatRGBA8Unorm
                                         width:w height:h + backdrop.state_rows
                                     mipmapped:NO];

        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        for (uint32_t i = 0; i < 2; i++) {
            [backdrop.scene_tex[i] release];
            backdrop.scene_tex[i] = i < n ? [m.device newTextureWithDescriptor:td] : nil;
            if (i < n && !backdrop.scene_tex[i]) {
                metal_log("backdrop: no %ux%u scene texture", w, h);
                return;
            }
            if (i < n) {
                /* a feedback scene's first "last frame" is zero, not junk */
                rp = [MTLRenderPassDescriptor renderPassDescriptor];
                rp.colorAttachments[0].texture = backdrop.scene_tex[i];
                rp.colorAttachments[0].loadAction = MTLLoadActionClear;
                rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
                rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
            }
        }
        backdrop.scene_w = w;
        backdrop.scene_h = h;
        backdrop.scene_cur = 0;
        backdrop.scene_frame = 0;
    }
    localtime_r(&tt, &lt);
    memset(&sp, 0, sizeof(sp));
    sp.a[0] = (float)w;
    sp.a[1] = (float)h;
    sp.a[2] = (float)h / (float)gh;
    sp.a[3] = (float)fmod(now - backdrop.t0, 86400.0);
    sp.b[0] = (float)(lt.tm_hour + lt.tm_min / 60.0 + lt.tm_sec / 3600.0);
    sp.b[1] = (float)METAL_ICONBAR_PX * sp.a[2];
    sp.b[2] = (float)gw;
    sp.b[3] = (float)gh;
    sp.c[0] = 0xB7 / 255.0f;
    sp.c[1] = 0xC0 / 255.0f;
    sp.c[2] = 0xB4 / 255.0f;
    sp.d[0] = backdrop.scene_frame;
    sp.d[1] = backdrop.state_rows;

    rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = backdrop.scene_tex[backdrop.feedback ? backdrop.scene_cur ^ 1 : 0];
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:backdrop.scene_pipe];
    [enc setFragmentBytes:&sp length:sizeof(sp) atIndex:0];
    if (backdrop.feedback) {
        [enc setFragmentTexture:backdrop.scene_tex[backdrop.scene_cur] atIndex:0];
    }
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (done.error) {
            metal_log("backdrop: the shader faulted on the GPU: %ld (%s)",
                      (long)done.error.code,
                      done.error.localizedDescription.UTF8String ?: "");
            backdrop.scene_failed = true;
        }
    }];
    [cb commit];
    if (backdrop.feedback) {
        backdrop.scene_cur ^= 1;
    }
    backdrop.scene_frame++;
    backdrop.scene_valid = true;
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

/* Whether a moving scene wants a frame now: it is due, the window can be
 * seen (not minimised, not covered) and RISC OS shows the backdrop at
 * all -- looked at once a second -- so a hidden scene costs nothing. */
static bool backdrop_scene_due(void)
{
    double now;

    if (backdrop.mode == METAL_BACKDROP_SHADER) {
        if (backdrop.fps <= 0 || !backdrop.scene_pipe) {
            return false;
        }
    } else if (backdrop.mode != METAL_BACKDROP_ACORN_LIVE) {
        return false;
    }
    if (!([m.window occlusionState] & NSWindowOcclusionStateVisible)) {
        return false;
    }
    now = CFAbsoluteTimeGetCurrent();
    if (now - backdrop.guest_checked >= 1.0) {
        backdrop.guest_shows = backdrop_guest_state() == 1;
        backdrop.guest_checked = now;
    }
    return backdrop.guest_shows && now >= backdrop.scene_next;
}

/* The folder the Backdrop menu lists and a relative scene names: the one
 * chosen with Choose Backdrops Folder, else backdrop-dir= (the launcher
 * gives the disc's Resources/Wallpapers), else ~/Pictures/RISC OS
 * Backdrops. */
static NSString *backdrop_folder(void)
{
    NSString *f = [[NSUserDefaults standardUserDefaults] stringForKey:@"backdropFolder"];
    BOOL isdir = NO;

    if (f.length && [[NSFileManager defaultManager] fileExistsAtPath:f isDirectory:&isdir]
        && isdir) {
        return f;
    }
    if (backdrop.dir.length) {
        return backdrop.dir;
    }
    return [@"~/Pictures/RISC OS Backdrops" stringByExpandingTildeInPath];
}

/* A scene's path as the spec keeps it: relative to the folder when it is
 * in it, so a choice follows the disc */
static NSString *backdrop_relative(NSString *path)
{
    NSString *dir = [backdrop_folder() stringByAppendingString:@"/"];

    return [path hasPrefix:dir] ? [path substringFromIndex:dir.length] : path;
}

static NSString *backdrop_resolve(NSString *path)
{
    path = [path stringByExpandingTildeInPath];
    return [path isAbsolutePath] ? path
                                 : [backdrop_folder() stringByAppendingPathComponent:path];
}

/* The scene, by its spec:
 *
 *   none            nothing drawn beneath (the default)
 *   acorn           the Acorn mark, built in
 *   acorn-live      the same with two drifting lights, built in
 *   shader:FILE     a scene from a .metal file
 *   tile:FILE       an image or sprite file, tiled from the top left
 *   picture:FILE    an image or sprite file, filling the window, centred
 *   wallpaper:FILE  a tile if it is 512 RISC OS pixels or less each way,
 *                   else a picture -- what the menu's Wallpapers give
 *   FILE            a .metal file is a shader, anything else a wallpaper
 *
 * A relative FILE is in the backdrops folder (backdrop_folder).  An
 * image's density is its own (backdrop_load_image), so a tile has one
 * image pixel to a RISC OS pixel, or two for a double-density one.
 * Anything that cannot be read or compiled gives none.  Only ever
 * reached with the feature gated on; UI thread only. */
static void backdrop_set_scene(const char *spec_c)
{
    NSString *spec = spec_c ? [NSString stringWithUTF8String:spec_c] : @"none";
    NSString *kind = nil, *file = nil;
    int mode = METAL_BACKDROP_OFF;

    if (!spec) {
        spec = @"none";
    }
    [backdrop.image release];
    backdrop.image = nil;
    backdrop.image_scale = 1.0;
    backdrop_scene_release();
    if (!spec.length || [spec isEqualToString:@"none"] || [spec isEqualToString:@"off"]) {
        spec = @"none";
    } else if ([spec isEqualToString:@"acorn"]) {
        mode = METAL_BACKDROP_ACORN;
    } else if ([spec isEqualToString:@"acorn-live"]) {
        mode = METAL_BACKDROP_ACORN_LIVE;
    } else {
        for (NSString *k in @[ @"shader", @"tile", @"picture", @"wallpaper" ]) {
            if ([spec hasPrefix:[k stringByAppendingString:@":"]]) {
                kind = k;
                file = [spec substringFromIndex:k.length + 1];
            }
        }
        if (!kind) {
            file = spec;
            kind = is_shader_leaf([spec lastPathComponent]) ? @"shader" : @"wallpaper";
        }
    }
    if (file) {
        NSString *path = backdrop_resolve(file);

        if ([kind isEqualToString:@"shader"]) {
            if (backdrop_scene_load(path)) {
                mode = METAL_BACKDROP_SHADER;
            }
        } else {
            size_t w = 0, h = 0;
            double scale = 1.0;

            backdrop.image = backdrop_load_image(path, &scale, &w, &h);
            if (backdrop.image) {
                backdrop.image_scale = scale;
                if ([kind isEqualToString:@"tile"]) {
                    mode = METAL_BACKDROP_TILE;
                } else if ([kind isEqualToString:@"picture"]) {
                    mode = METAL_BACKDROP_PICTURE;
                } else {
                    mode = (w <= 512 * scale && h <= 512 * scale) ? METAL_BACKDROP_TILE
                                                                  : METAL_BACKDROP_PICTURE;
                }
                metal_log("backdrop: %s %s (%zux%zu, %.3g to a RISC OS pixel)",
                          mode == METAL_BACKDROP_TILE ? "tile" : "picture",
                          [path fileSystemRepresentation], w, h, scale);
            } else {
                metal_log("backdrop: cannot read %s, so none", [path fileSystemRepresentation]);
            }
        }
        if (mode == METAL_BACKDROP_OFF) {
            spec = @"none";
        }
    }
    [backdrop.spec release];
    backdrop.spec = [spec copy];
    if (mode != backdrop.mode) {
        backdrop.mode = mode;
        fb_release_scale();
    }
    backdrop.t0 = CFAbsoluteTimeGetCurrent();
    backdrop.scene_next = 0;
    backdrop.guest_checked = 0;
    frame_wanted = true;
    metal_log("backdrop: scene %s (mode %d)", [spec UTF8String], mode);
}

/* -display metal,backdrop-dir= -- the folder of wallpapers and shaders,
 * before backdrop= (ui/metal.c calls this first) */
void metal_glue_backdrop_dir(const char *dir)
{
    [backdrop.dir release];
    backdrop.dir = dir && *dir
        ? [[[NSString stringWithUTF8String:dir] stringByExpandingTildeInPath] copy] : nil;
    metal_log("backdrop: folder %s", dir ? dir : "(none)");
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

/* The files of a folder the menu lists, by name as the Finder sorts
 * them, leaving out a double-density sibling (NAME@2x, or NAME11 for a
 * sprite file) when NAME is there too: it is drawn in NAME's place on a
 * Retina window. */
static NSArray *backdrop_files(NSString *dir, bool images, bool shaders)
{
    NSArray *names = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:dir error:NULL];
    NSMutableArray *out = [NSMutableArray array];
    NSMutableSet *titles = [NSMutableSet set];

    for (NSString *n in names) {
        if ((images && (is_image_leaf(n) || is_sprite_leaf(n))) || (shaders && is_shader_leaf(n))) {
            [titles addObject:[leaf_title(n) lowercaseString]];
        }
    }
    for (NSString *n in [names sortedArrayUsingSelector:@selector(localizedStandardCompare:)]) {
        NSString *t = [leaf_title(n) lowercaseString];
        BOOL isdir = NO;

        if ([n hasPrefix:@"."]) {
            continue;
        }
        if (!((images && (is_image_leaf(n) || is_sprite_leaf(n))) || (shaders && is_shader_leaf(n)))) {
            continue;
        }
        if ([[NSFileManager defaultManager] fileExistsAtPath:[dir stringByAppendingPathComponent:n]
                                                 isDirectory:&isdir] && isdir) {
            continue;
        }
        if (images && t.length > 3 && [t hasSuffix:@"@2x"]
            && [titles containsObject:[t substringToIndex:t.length - 3]]) {
            continue;
        }
        if (images && is_sprite_leaf(n) && t.length > 2 && [t hasSuffix:@"11"]
            && [titles containsObject:[t substringToIndex:t.length - 2]]) {
            continue;
        }
        [out addObject:[dir stringByAppendingPathComponent:n]];
        if (out.count >= 200) {
            break;
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
        metal_log("fps %u (%ux%u, %u bpp)", fps, fb.xres, fb.yres, fb.bpp);
        [m.window setTitle:
            [NSString stringWithFormat:
                @"%s — %ux%u, %u bpp · %u fps — %s",
                sys, fb.xres, fb.yres, fb.bpp, fps, mach]];
    }
    last_tick = now;
    last_second_frame = frame_count;
}

/* Is the guest's pointer, at gx,gy, one of the positions sent in the
 * last METAL_ECHO_SECONDS (to a pixel: the guest rounds through OS
 * units)?  Then it is only late, and the host's mouse is where it is
 * going; otherwise the guest has put it somewhere of its own. */
static bool metal_pointer_is_echo(int32_t gx, int32_t gy)
{
    double now = CACurrentMediaTime();
    unsigned i;

    for (i = 0; i < 32; i++) {
        if (mouse.sent[i].t > 0 && now - mouse.sent[i].t <= METAL_ECHO_SECONDS
            && abs(mouse.sent[i].gx - gx) <= 1 && abs(mouse.sent[i].gy - gy) <= 1) {
            return true;
        }
    }
    return abs(mouse.gx - gx) <= 1 && abs(mouse.gy - gy) <= 1;
}

/* ------------------------------------------------------------------ */
/* The surfaces pass (design 39): the box's windows, composited here   */

/*
 * The ROSGD box publishes its windows as surfaces (block v2): each a
 * backing whose pixels lie in one region of guest RAM, which the board
 * has mapped once and which is wrapped here, once, as an MTLBuffer with
 * no copy.  Each frame with a list, the guest-sized texture the decode
 * pass would fill is cleared to transparent and the surfaces are drawn
 * into it, back to front: for each, its shadow (darkening what is
 * already there, the PC presenter's rounded-box field), then its pixels,
 * read where the Wimp drew them -- nothing is uploaded.  The scale pass
 * then lays that over the backdrop as it lays the decoded screen.
 */
static NSString *const SURF_SRC = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct SQuad {\n"
"    float4 rect;    /* where it draws: x0 y0 x1 y1, guest pixels */\n"
"    float4 box;     /* the surface (or the shadow's dropped box): x y w h */\n"
"    uint4 u;        /* offset, stride (bytes), kind (0 pixels, 1 shadow), 0 */\n"
"    float4 sh;      /* shadow: radius, blur, strength 0..1, 0 */\n"
"};\n"
"struct SOut {\n"
"    float4 pos [[position]];\n"
"    uint inst [[flat]];\n"
"};\n"
"vertex SOut vs_surf(uint vid [[vertex_id]], uint iid [[instance_id]],\n"
"                    device const SQuad *q [[buffer(0)]],\n"
"                    constant float2 &screen [[buffer(1)]])\n"
"{\n"
"    float4 r = q[iid].rect;\n"
"    float2 c = float2((vid & 1) ? r.z : r.x, (vid & 2) ? r.w : r.y);\n"
"    SOut o;\n"
"    o.pos = float4(c.x / screen.x * 2.0 - 1.0, 1.0 - c.y / screen.y * 2.0, 0.0, 1.0);\n"
"    o.inst = iid;\n"
"    return o;\n"
"}\n"
"fragment float4 ps_surf(SOut in [[stage_in]],\n"
"                        device const SQuad *q [[buffer(0)]],\n"
"                        device const uint *region [[buffer(1)]])\n"
"{\n"
"    SQuad Q = q[in.inst];\n"
"    float2 p = in.pos.xy;\n"
"    if (Q.u.z == 1) {\n"
"        /* the shadow: strength falling off with the distance to the\n"
"         * dropped box, rounded by the window's corners, over the blur */\n"
"        float2 h = Q.box.zw * 0.5;\n"
"        float r = min(Q.sh.x, min(h.x, h.y));\n"
"        float2 d = abs(p - (Q.box.xy + h)) - (h - float2(r));\n"
"        float dist = max(length(max(d, float2(0.0))) - r, 0.0);\n"
"        float t = clamp(1.0 - dist / max(Q.sh.y, 1.0), 0.0, 1.0);\n"
"        return float4(0.0, 0.0, 0.0, Q.sh.z * t * t * (3.0 - 2.0 * t));\n"
"    }\n"
"    int2 l = int2(floor(p - Q.box.xy));\n"
"    if (l.x < 0 || l.y < 0 || l.x >= int(Q.box.z) || l.y >= int(Q.box.w)) {\n"
"        discard_fragment();\n"
"    }\n"
"    uint w = region[(Q.u.x + uint(l.y) * Q.u.y) / 4 + uint(l.x)];\n"
"    if ((w >> 24) == 0x80) {\n"
"        return float4(0.0);     /* the below tag: a hole to the backdrop */\n"
"    }\n"
"    return float4(float(w & 255), float((w >> 8) & 255), float((w >> 16) & 255),\n"
"                  float(w >> 24)) / 255.0;  /* premultiplied */\n"
"}\n"
/* The icons whose pictures this window draws (ROS_SURF_CLOCK and
 * ROS_SURF_MONITOR): the PC presenter's shaders, in MSL, drawn at the
 * window's own resolution.  DeskClock's face: its styles' colours (face,
 * hour, minute, second, centre, tick, rim, inner, &00BBGGRR), dots and
 * digits; 4 x 4 samples a pixel, premultiplied.  DeskMeter's graph:
 * pixel art, a block for each of its 52 x 34 pixels. */
"struct FaceU {\n"
"    float4 box;     /* the icon on the drawable: x, y down, w, h */\n"
"    float4 clip;    /* this draw's rectangle: x0 y0 x1 y1 */\n"
"    float2 target;  /* the drawable's size */\n"
"    uint style, kind;\n"
"    float4 hms;     /* hours, minutes, seconds */\n"
"    float hist[24]; /* the meter's, 0-255, oldest first */\n"
"};\n"
"struct FOut { float4 pos [[position]]; };\n"
"vertex FOut vs_face(uint vid [[vertex_id]], constant FaceU &U [[buffer(0)]])\n"
"{\n"
"    float2 c = float2((vid & 1) ? U.clip.z : U.clip.x, (vid & 2) ? U.clip.w : U.clip.y);\n"
"    FOut o;\n"
"    o.pos = float4(c.x / U.target.x * 2.0 - 1.0, 1.0 - c.y / U.target.y * 2.0, 0.0, 1.0);\n"
"    return o;\n"
"}\n"
"constant uint ST[40] = {\n"
"  0x00E0F2F8u, 0x00202020u, 0x00A08040u, 0x002020C0u, 0x00202020u, 0x00202020u, 0x00201A10u, 0x00C0A070u,\n"
"  0x001F1410u, 0x00F0F0F0u, 0x00E0D0B0u, 0x002A8AFFu, 0x00F0F0F0u, 0x00CFF4C4u, 0x00443C2Eu, 0x00A8C56Fu,\n"
"  0x000E1014u, 0x00E5FF00u, 0x007F3DFFu, 0x004DE9FFu, 0x00FFFFFFu, 0x00E5FF00u, 0x007F3DFFu, 0x00FFE500u,\n"
"  0x00EEF6EAu, 0x002B3124u, 0x007B8A7Bu, 0x004D67E8u, 0x002B3124u, 0x002F6B4Fu, 0x002F6B4Fu, 0x00BFDCCBu,\n"
"  0x00181212u, 0x002020F0u, 0x002020F0u, 0x002020F0u, 0x002020F0u, 0x00504040u, 0x00504040u, 0x00504040u };\n"
"constant int DOTS[5] = { 0, 0, 0, 1, 1 };\n"
"constant int DIGI[5] = { 0, 0, 0, 0, 1 };\n"
"constant int SEGS[10] = { 0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F };\n"
"static float3 fc_rgb(uint c) { return float3(float(c & 255u), float((c >> 8) & 255u), float((c >> 16) & 255u)) / 255.0; }\n"
"static float capsule(float2 p, float2 a, float2 b, float r) {\n"
"    float2 pa = p - a, v = b - a;\n"
"    float t = clamp(dot(pa, v) / dot(v, v), 0.0, 1.0);\n"
"    return length(pa - v * t) - r;\n"
"}\n"
"static float disc(float2 p, float2 c, float r) { return length(p - c) - r; }\n"
"static float hand(float2 p, float c, float a, float t0, float t1, float hw) {\n"
"    float s = sin(a), k = cos(a);\n"
"    return capsule(p, float2(c + t0 * s, c - t0 * k), float2(c + t1 * s, c - t1 * k), hw);\n"
"}\n"
"static bool digit_at(float2 p, int d, float x0, float y0, float wd, float ht) {\n"
"    const float hw = 0.75;\n"
"    float m = y0 + ht / 2.0, x1 = x0 + wd, y1 = y0 + ht;\n"
"    float4 seg[7] = { float4(x0, y0, x1, y0), float4(x1, y0, x1, m), float4(x1, m, x1, y1),\n"
"                      float4(x0, y1, x1, y1), float4(x0, m, x0, y1), float4(x0, y0, x0, m),\n"
"                      float4(x0, m, x1, m) };\n"
"    for (int i = 0; i < 7; i++)\n"
"        if (((SEGS[d] >> i) & 1) != 0 && capsule(p, seg[i].xy, seg[i].zw, hw) < 0.0)\n"
"            return true;\n"
"    return false;\n"
"}\n"
"static int face_at(float2 p, int style, float3 hms) {\n"
"    const float c = 17.0, R = 17.0 - 0.6, PI = 3.14159265358979;\n"
"    float2 cc = float2(c);\n"
"    if (disc(p, cc, R) >= 0.0) return -1;\n"
"    if (disc(p, cc, R - 1.7) >= 0.0) return 6;\n"
"    float h = hms.x, m = hms.y, s = hms.z;\n"
"    if (DIGI[style] != 0) {\n"
"        float wd = 5.0, ht = 9.0, gap = 2.6;\n"
"        float x0 = c - (4.0 * wd + 3.0 * gap + 1.2) / 2.0, y0 = c - ht / 2.0;\n"
"        int ih = int(h), im = int(m);\n"
"        int dg[4] = { ih / 10, ih % 10, im / 10, im % 10 };\n"
"        for (int i = 0; i < 4; i++) {\n"
"            float dx = x0 + float(i) * (wd + gap) + (i >= 2 ? 1.2 : 0.0);\n"
"            if (digit_at(p, dg[i], dx, y0, wd, ht)) return 1;\n"
"        }\n"
"        float kx = c + 0.1;\n"
"        if ((int(s) & 1) == 0 && (disc(p, float2(kx, c - 2.2), 0.8) < 0.0 ||\n"
"                                  disc(p, float2(kx, c + 2.2), 0.8) < 0.0)) return 1;\n"
"        return 0;\n"
"    }\n"
"    float ha = (fmod(h, 12.0) + m / 60.0 + s / 3600.0) / 12.0 * 2.0 * PI;\n"
"    float ma = (m + s / 60.0) / 60.0 * 2.0 * PI;\n"
"    float sa = s / 60.0 * 2.0 * PI;\n"
"    if (disc(p, cc, 1.0) < 0.0) return 3;\n"
"    if (disc(p, cc, 1.9) < 0.0) return 4;\n"
"    if (hand(p, c, sa, -3.8, R - 3.0, 0.42) < 0.0) return 3;\n"
"    if (hand(p, c, ma, -1.5, R - 3.4, 0.95) < 0.0) return 2;\n"
"    if (hand(p, c, ha, -1.5, R * 0.55, 1.35) < 0.0) return 1;\n"
"    for (int i = 0; i < 12; i++) {\n"
"        float a = float(i) / 12.0 * 2.0 * PI;\n"
"        bool major = i % 3 == 0;\n"
"        if (DOTS[style] != 0) {\n"
"            if (major && disc(p, float2(c + (R - 4.6) * sin(a), c - (R - 4.6) * cos(a)), 1.3) < 0.0)\n"
"                return 5;\n"
"        } else if (hand(p, c, a, R - (major ? 5.6 : 4.4), R - 2.9, major ? 0.8 : 0.45) < 0.0) {\n"
"            return major ? 6 : 5;\n"
"        }\n"
"    }\n"
"    if (disc(p, cc, R - 2.3) >= 0.0) return 7;\n"
"    return 0;\n"
"}\n"
"static float3 mt_pal(uint st, int c) {\n"
"    if (st == 0) {\n"
"        float3 t[8] = { float3(0, 10, 0), float3(32, 192, 32), float3(224, 192, 32), float3(224, 64, 32),\n"
"                        float3(32, 192, 32), float3(32, 80, 32), float3(68, 192, 32), float3(32, 192, 32) };\n"
"        return t[c];\n"
"    }\n"
"    if (st == 1) {\n"
"        float3 t[8] = { float3(20, 20, 30), float3(32, 192, 96), float3(224, 192, 32), float3(224, 64, 64),\n"
"                        float3(32, 192, 96), float3(48, 48, 48), float3(96, 96, 96), float3(32, 192, 96) };\n"
"        return t[c];\n"
"    }\n"
"    return float3(255, 0, 255);   /* a style it does not know */\n"
"}\n"
"static int mt_pixel(int x, int y, constant float *hist) {\n"
"    const int W = 52, H = 34;\n"
"    int cx = min(x, W - 1 - x), cy = min(y, H - 1 - y);\n"
"    if (cx + cy < 2 && cx < 2 && cy < 2) return -1;   /* the cut corners */\n"
"    if (y < 2 || y >= H - 2 || x < 2 || x >= W - 2) return 6;\n"
"    int gh = H - 6;\n"
"    int up = H - 1 - y - 3;\n"
"    int i = (x - 4) / 2;\n"
"    if (x >= 4 && x < W - 3 && i < 24) {\n"
"        float frac = floor(hist[i] * 1000.0 / 255.0);\n"
"        int h = int(floor((float(gh) * frac + 500.0) / 1000.0));\n"
"        if (frac > 0.0 && up >= 0 && up < h)\n"
"            return frac < 500.0 ? 1 : frac < 750.0 ? 2 : 3;\n"
"    }\n"
"    bool inside = y >= 3 && y < H - 3 && x >= 3 && x < W - 3;\n"
"    if (inside && y - 3 == gh / 2 && ((x - 3) & 3) < 2) return 5;\n"
"    return 0;\n"
"}\n"
"fragment float4 ps_face(FOut in [[stage_in]], constant FaceU &U [[buffer(0)]])\n"
"{\n"
"    if (U.kind == 2) {\n"
"        float2 px = in.pos.xy - U.box.xy;\n"
"        int2 q = int2(floor(px / U.box.zw * float2(52.0, 34.0)));\n"
"        if (q.x < 0 || q.y < 0 || q.x > 51 || q.y > 33) discard_fragment();\n"
"        int c = mt_pixel(q.x, q.y, U.hist);\n"
"        if (c < 0) discard_fragment();\n"
"        return float4(mt_pal(U.style % 2, c) / 255.0, 1.0);\n"
"    }\n"
"    float2 px = in.pos.xy - 0.5 - U.box.xy;\n"
"    float2 k = 34.0 / U.box.zw;\n"
"    int style = int(U.style % 5);\n"
"    float3 acc = float3(0.0);\n"
"    float n = 0.0;\n"
"    for (int j = 0; j < 4; j++)\n"
"        for (int i = 0; i < 4; i++) {\n"
"            int c = face_at((px + (float2(i, j) + 0.5) / 4.0) * k, style, U.hms.xyz);\n"
"            if (c >= 0) {\n"
"                acc += fc_rgb(ST[style * 8 + c]);\n"
"                n += 1.0;\n"
"            }\n"
"        }\n"
"    if (n == 0.0) discard_fragment();\n"
"    float a = n / 16.0;\n"
"    return float4(acc / n * a, a);  /* premultiplied */\n"
"}\n";

typedef struct SQuad {
    float rect[4];
    float box[4];
    uint32_t u[4];
    float sh[4];
} SQuad;

#define SURF_QUADS 8192

static struct {
    bool tried;
    id<MTLRenderPipelineState> pipe;
    id<MTLBuffer> region;               /* the guest's region, no copy */
    void *region_host;
    uint64_t region_size;
    id<MTLBuffer> quads[METAL_RING];
    unsigned ring;
    MetalSurfacesView view;
    bool on;                            /* this frame composites the list */
    id<MTLRenderPipelineState> face;    /* the clock's and the meter's, into the drawable */
    id<MTLRenderPipelineState> face_shot;   /* the same, into a screenshot's RGBA texture */
    uint32_t drawn_frame;
    double face_due;                    /* when a listed clock is next to be drawn */
    uint64_t frames, quads_drawn;
} surf;

static bool surf_build(void)
{
    NSError *err = nil;
    MTLCompileOptions *opts;
    id<MTLLibrary> lib;
    id<MTLFunction> vs, fs;
    MTLRenderPipelineDescriptor *pd;

    if (surf.tried) {
        return surf.pipe != nil;
    }
    surf.tried = true;
    opts = [[MTLCompileOptions alloc] init];
    lib = [m.device newLibraryWithSource:SURF_SRC options:opts error:&err];
    [opts release];
    if (!lib) {
        metal_log("surfaces: shader compile failed: %s",
                  [[err localizedDescription] UTF8String]);
        return false;
    }
    vs = [lib newFunctionWithName:@"vs_surf"];
    fs = [lib newFunctionWithName:@"ps_surf"];
    pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    pd.colorAttachments[0].blendingEnabled = YES;
    pd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    surf.pipe = [m.device newRenderPipelineStateWithDescriptor:pd error:&err];
    [pd release];
    [vs release];
    [fs release];
    [lib release];
    if (!surf.pipe) {
        metal_log("surfaces: pipeline failed: %s", [[err localizedDescription] UTF8String]);
        return false;
    }
    for (unsigned i = 0; i < METAL_RING; i++) {
        surf.quads[i] = [m.device newBufferWithLength:SURF_QUADS * sizeof(SQuad)
                                              options:MTLResourceStorageModeShared];
    }
    {
        id<MTLLibrary> flib;
        id<MTLFunction> fvs, ffs;
        MTLCompileOptions *fo = [[MTLCompileOptions alloc] init];

        flib = [m.device newLibraryWithSource:SURF_SRC options:fo error:&err];
        [fo release];
        fvs = [flib newFunctionWithName:@"vs_face"];
        ffs = [flib newFunctionWithName:@"ps_face"];
        pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction = fvs;
        pd.fragmentFunction = ffs;
        pd.colorAttachments[0].pixelFormat = m.layer.pixelFormat;
        pd.colorAttachments[0].blendingEnabled = YES;
        pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
        pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        surf.face = flib ? [m.device newRenderPipelineStateWithDescriptor:pd error:&err] : nil;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        surf.face_shot = flib ? [m.device newRenderPipelineStateWithDescriptor:pd error:&err] : nil;
        [pd release];
        [fvs release];
        [ffs release];
        [flib release];
        if (!surf.face) {
            metal_log("surfaces: the faces' pipeline failed: %s",
                      err ? [[err localizedDescription] UTF8String] : "?");
        }
    }
    metal_log("surfaces: pipeline up");
    return true;
}

/* The region as a buffer: made once, again only if the guest names another */
static bool surf_region(void)
{
    MetalSurfacesView *v = &surf.view;

    if (surf.region && surf.region_host == v->region && surf.region_size == v->region_size) {
        return true;
    }
    [surf.region release];
    surf.region = nil;
    surf.region_host = v->region;
    surf.region_size = v->region_size;
    if (!v->region || !v->region_size) {
        return false;
    }
    surf.region = [m.device newBufferWithBytesNoCopy:v->region length:v->region_size
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    metal_log("surfaces: region %p +%llu MB %s", v->region,
              (unsigned long long)(v->region_size >> 20),
              surf.region ? "wrapped, no copy" : "NOT wrapped");
    return surf.region != nil;
}

/* The list's quads into this frame's buffer: n of them */
static uint32_t surf_quads(SQuad *q)
{
    const MetalSurfacesView *v = &surf.view;
    uint32_t n = 0;

    for (uint32_t i = 0; i < v->n; i++) {
        const MetalSurface *e = &v->s[i];

        for (uint32_t k = 0; k < e->shadow_nrects && n < SURF_QUADS; k++) {
            const int16_t *r = v->rect[e->shadow_rects + k];
            SQuad *o = &q[n++];

            o->rect[0] = r[0], o->rect[1] = r[1], o->rect[2] = r[2], o->rect[3] = r[3];
            o->box[0] = e->x + e->shadow_dx, o->box[1] = e->y + e->shadow_dy;
            o->box[2] = e->w, o->box[3] = e->h;
            o->u[0] = 0, o->u[1] = 0, o->u[2] = 1, o->u[3] = 0;
            o->sh[0] = e->radius, o->sh[1] = e->shadow_blur;
            o->sh[2] = e->shadow_strength / 255.0f, o->sh[3] = 0;
        }
        for (uint32_t k = 0; k < e->nrects && n < SURF_QUADS; k++) {
            const int16_t *r = v->rect[e->rects + k];
            SQuad *o = &q[n++];

            o->rect[0] = r[0], o->rect[1] = r[1], o->rect[2] = r[2], o->rect[3] = r[3];
            o->box[0] = e->x, o->box[1] = e->y, o->box[2] = e->w, o->box[3] = e->h;
            o->u[0] = e->offset, o->u[1] = e->stride, o->u[2] = 0, o->u[3] = 0;
            memset(o->sh, 0, sizeof(o->sh));
        }
    }
    return n;
}

typedef struct FaceU {
    float box[4], clip[4], target[2];
    uint32_t style, kind;
    float hms[4];
    float hist[24];
} FaceU;

/* The clock's face and the meter's graph, into the drawable at its own
 * resolution, clipped to what can be seen of each, over the scaled
 * frame (enc is its encoder) */
static void surf_faces(id<MTLRenderCommandEncoder> enc, id<MTLRenderPipelineState> pipe,
                       CGSize target, uint32_t xres, uint32_t yres)
{
    const MetalSurfacesView *v = &surf.view;
    double sx = target.width / MAX(xres, 1u), sy = target.height / MAX(yres, 1u);
    struct timeval tv;
    struct tm lt;
    time_t tt;
    FaceU u;

    if (!surf.on || !v->nicons || !pipe) {
        return;
    }
    gettimeofday(&tv, NULL);
    tt = tv.tv_sec;
    localtime_r(&tt, &lt);
    [enc setRenderPipelineState:pipe];
    for (uint32_t i = 0; i < v->nicons; i++) {
        const MetalIcon *ic = &v->icon[i];

        memset(&u, 0, sizeof(u));
        u.box[0] = ic->x * sx, u.box[1] = ic->y * sy, u.box[2] = ic->w * sx, u.box[3] = ic->h * sy;
        u.target[0] = target.width, u.target[1] = target.height;
        u.style = ic->style, u.kind = ic->kind;
        u.hms[0] = lt.tm_hour, u.hms[1] = lt.tm_min;
        u.hms[2] = lt.tm_sec + (ic->rate > 1 ? tv.tv_usec / 1e6f : 0.0f);
        for (int k = 0; k < 24; k++) {
            u.hist[k] = ic->data[k];
        }
        for (uint32_t r = 0; r < ic->nrects; r++) {
            const int16_t *q = v->rect[ic->rects + r];

            u.clip[0] = floor(q[0] * sx), u.clip[1] = floor(q[1] * sy);
            u.clip[2] = ceil(q[2] * sx), u.clip[3] = ceil(q[3] * sy);
            if (u.clip[2] <= u.clip[0] || u.clip[3] <= u.clip[1]) {
                continue;
            }
            [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
            [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        }
    }
}

/* Whether a listed clock is due to be drawn again: at the rate it asks,
 * else at each second */
static bool surf_face_due(void)
{
    const MetalSurfacesView *v = &surf.view;
    double now = CFAbsoluteTimeGetCurrent(), step = 0;

    if (!surf.on) {
        return false;
    }
    for (uint32_t i = 0; i < v->nicons; i++) {
        if (v->icon[i].kind == 1 && v->icon[i].nrects) {
            double s = v->icon[i].rate > 1 ? 1.0 / MIN(v->icon[i].rate, 60u) : 1.0;

            step = step ? MIN(step, s) : s;
        }
    }
    if (!step || now < surf.face_due) {
        return false;
    }
    surf.face_due = (floor(now / step) + 1) * step;
    return true;
}

/* This frame's surfaces into fb.decoded, in place of the decode pass */
static void surf_draw(id<MTLCommandBuffer> cb, uint32_t xres, uint32_t yres)
{
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    id<MTLRenderCommandEncoder> enc;
    id<MTLBuffer> qb = surf.quads[surf.ring];
    float screen[2] = { (float)xres, (float)yres };
    uint32_t n;

    surf.ring = (surf.ring + 1) % METAL_RING;
    n = surf_quads((SQuad *)[qb contents]);
    rp.colorAttachments[0].texture = fb.decoded;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    enc = [cb renderCommandEncoderWithDescriptor:rp];
    if (n) {
        [enc setRenderPipelineState:surf.pipe];
        [enc setVertexBuffer:qb offset:0 atIndex:0];
        [enc setVertexBytes:screen length:sizeof(screen) atIndex:1];
        [enc setFragmentBuffer:qb offset:0 atIndex:0];
        [enc setFragmentBuffer:surf.region offset:0 atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
              instanceCount:n];
    }
    [enc endEncoding];
    surf.frames++;
    surf.quads_drawn += n;
    if (surf.frames % 300 == 1) {
        metal_log("surfaces: frame %u, %u surfaces, %u quads (%llu frames drawn)",
                  surf.view.frame, surf.view.n, n, (unsigned long long)surf.frames);
    }
}

static bool metal_screenshot_to(NSString *path);

/* Pacing (design 39): frames presented each second, to metal-debug.txt
 * for each second that presented any, and once when it falls idle -- an
 * idle desktop is to present none */
static uint32_t pace_presented;
static char want_window[256];           /* a "window:NAME" shot request: the drawable */
static uint32_t pace_why[6];            /* why: screen, mode, pointer, size, list, asked */

static void metal_pace(void)
{
    static double t0;
    static uint32_t last;
    double now = CFAbsoluteTimeGetCurrent();

    if (!t0) {
        t0 = now;
    }
    if (now - t0 < 1.0) {
        return;
    }
    if (pace_presented || last) {
        metal_log("pace: %u frames presented in %.2f s%s (screen %u mode %u pointer %u "
                  "size %u list %u asked %u)", pace_presented, now - t0,
                  surf.on ? ", from the surfaces" : "", pace_why[0], pace_why[1],
                  pace_why[2], pace_why[3], pace_why[4], pace_why[5]);
    }
    last = pace_presented;
    pace_presented = 0;
    memset(pace_why, 0, sizeof(pace_why));
    t0 = now;
}

static bool metal_render_frame(void)
{
    metal_pace();
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

    /* METAL_SHOT_DIR/shot.request, a file a test writes with a name in
     * it: a frame is drawn now, and on the next pass the composite goes
     * to METAL_SHOT_DIR/<name>.png and the request is removed -- a picture
     * of what the window shows, when asked, idle desktop or not */
    {
        static char want[256];
        static int tick;
        const char *dir = getenv("METAL_SHOT_DIR");

        if (want[0]) {
            NSString *path = [NSString stringWithFormat:@"%s/%s.png", dir, want];
            bool ok = metal_screenshot_to(path);

            metal_log("shot: %s.png %s", want, ok ? "written" : "FAILED");
            want[0] = 0;
        } else if (dir && *dir && ++tick % 6 == 0) {
            char req[512];
            FILE *f;

            snprintf(req, sizeof(req), "%s/shot.request", dir);
            f = fopen(req, "r");
            if (f) {
                if (!fgets(want, sizeof(want), f)) {
                    snprintf(want, sizeof(want), "shot");
                }
                want[strcspn(want, "\r\n /")] = 0;
                if (!want[0]) {
                    snprintf(want, sizeof(want), "shot");
                }
                if (!strncmp(want, "window:", 7)) {
                    /* the drawable as the window shows it, this frame */
                    snprintf(want_window, sizeof(want_window), "%s", want + 7);
                    want[0] = 0;
                }
                fclose(f);
                unlink(req);
                frame_wanted = true;
            }
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
    /* the guest's surfaces, if it publishes a list: then they, and not the
     * screen, are what is shown (design 39) */
    surf.on = have_fb && metal_glue_surfaces_view(&surf.view) && surf_build() && surf_region();

    /* Power: nothing new, no frame (and nothing copied, decoded or
     * presented); the loop sleeps instead */
    {
        static uint32_t drawn_content, drawn_gen;
        static CGSize drawn_size;
        MetalCursorView pc = { 0 };
        uint32_t pgen = ptr.generation;
        CGSize size = m.layer.drawableSize;

        if (ptr.pipe && metal_glue_cursor_view(&pc) && !pc.stale) {
            pgen = pc.generation;
        }
        if (have_fb && v.content && !frame_wanted && v.content == drawn_content
            && v.generation == drawn_gen && pgen == ptr.generation
            && size.width == drawn_size.width && size.height == drawn_size.height
            && (!surf.on || surf.view.frame == surf.drawn_frame) && !surf_face_due()) {
            frame_skipped = true;
            return have_fb;
        }
        if (!have_fb || !v.content || v.content != drawn_content) {
            pace_why[0]++;
        }
        if (have_fb && v.generation != drawn_gen) {
            pace_why[1]++;
        }
        if (pgen != ptr.generation) {
            pace_why[2]++;
        }
        if (size.width != drawn_size.width || size.height != drawn_size.height) {
            pace_why[3]++;
        }
        if (surf.on && surf.view.frame != surf.drawn_frame) {
            pace_why[4]++;
        }
        if (frame_wanted) {
            pace_why[5]++;
        }
        frame_wanted = false;
        surf.drawn_frame = surf.on ? surf.view.frame : 0;
        drawn_content = have_fb ? v.content : 0;
        drawn_gen = have_fb ? v.generation : 0;
        drawn_size = size;
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
        if (!surf.on) {
            fb_upload(&v);              /* the screen; a list's surfaces are read in place */
        }

        target = m.layer.drawableSize;
        params.dim[0] = v.xres;
        params.dim[1] = v.yres;
        params.dim[2] = v.pitch;
        params.dim[3] = v.bpp;
        params.misc[0] = v.xoffset;
        params.misc[1] = v.yoffset;
        params.misc[2] = v.pixo;
        params.misc[3] = surf.on ? surf.view.look : 0;     /* the desktop's look */
        params.post[0] = (uint32_t)target.width;
        params.post[1] = (uint32_t)target.height;
        params.post[2] = (uint32_t)video_opts.scaling;
        params.post[3] = video_opts.scanlines ? 1 : 0;
        backdrop_params(&params, target.height / (double)MAX(v.yres, 1u));
        fb.xoff = v.xoffset;

        /* a shader's scene: drawn when it is new, when the window's size
         * or the mode changes it, and when it is due (main loop); in its
         * own command buffer, ahead of this frame's */
        if (backdrop.mode == METAL_BACKDROP_SHADER && backdrop.scene_pipe) {
            uint32_t sw, sh;
            double now = CFAbsoluteTimeGetCurrent();

            backdrop_scene_size(target, v.xres, v.yres, &sw, &sh);
            if (!backdrop.scene_valid || sw != backdrop.scene_w || sh != backdrop.scene_h
                || (backdrop.fps > 0 && now >= backdrop.scene_next)) {
                backdrop_scene_render(target, v.xres, v.yres);
                if (backdrop.fps > 0) {
                    backdrop.scene_next = now + 1.0 / backdrop.fps;
                }
            }
            backdrop_params(&params, target.height / (double)MAX(v.yres, 1u));
        } else if (backdrop.mode == METAL_BACKDROP_ACORN_LIVE) {
            backdrop.scene_next = CFAbsoluteTimeGetCurrent() + 1.0 / 12.0;
        }
        fb.yoff = v.yoffset;
        fb.pixo = v.pixo;

        /* decode pass: raw bytes -> linear RGB; or, with a list, the
         * surfaces in its place */
        if (surf.on) {
            surf_draw(cb, v.xres, v.yres);
        } else {
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
    if (have_fb && fb.scale && backdrop.mode == METAL_BACKDROP_SHADER
        && !backdrop.scene_tex[backdrop.scene_cur]) {
        have_fb = false;                /* no scene texture: nothing to bind */
    }
    if (have_fb && fb.scale) {
        [enc setRenderPipelineState:fb.scale];
        [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
        [enc setFragmentTexture:fb.decoded atIndex:0];
        if (backdrop.mode >= METAL_BACKDROP_PICTURE) {
            [enc setFragmentTexture:backdrop.mode == METAL_BACKDROP_SHADER
                                    ? backdrop.scene_tex[backdrop.scene_cur] : backdrop.image
                            atIndex:1];
            [enc setFragmentSamplerState:m.repeat atIndex:1];
        }
        [enc setFragmentSamplerState:m.linear atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0 vertexCount:3];
    }

    if (have_fb && surf.on) {
        surf_faces(enc, surf.face, target, v.xres, v.yres);
    }

    /* The pointer sprite, on top of the scaled frame.  The peer
     * commits whole transactions at UpdateSubmit, so a generation move
     * is always a complete new sprite and position; a read that raced
     * one is reported stale and the previous frame's is kept. */
    {
        MetalCursorView cv = { 0 };
        bool torn = false;

        if (ptr.pipe && metal_glue_cursor_view(&cv) && !cv.stale
            && cv.generation != ptr.generation) {
            MetalCursorView again = { 0 };

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
                ptr.hot_x = cv.hot_x;
                ptr.hot_y = cv.hot_y;
                ptr.has_hot = cv.has_hot;
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

            double px = ptr.x, py = ptr.y;

            /* The guest moves its pointer once a VSync, a frame or two
             * after the mouse has moved: drawn where the guest has it,
             * the arrow trails the hand and steps.  So while the host
             * cursor is the guest's (ungrabbed, over the view, in the
             * key window) and the guest's pointer is only late -- at a
             * position sent lately -- its sprite is drawn at the mouse,
             * by its hot spot, every frame.  Where the guest has put
             * the pointer itself (a drag held to its box, a program
             * moving it, the pointer unlinked from the mouse) it is
             * drawn there. */
            if (ptr.has_hot && !kbd.on && cursor.inside && [m.window isKeyWindow]
                && mouse.have_guest && ptr.disp_w == mouse.gxres
                && ptr.disp_h == mouse.gyres
                && metal_pointer_is_echo(ptr.x + ptr.hot_x, ptr.y + ptr.hot_y)) {
                px = mouse.gx - ptr.hot_x;
                py = mouse.gy - ptr.hot_y;
            }
            metal_disp_transform(dw, dh, v.xres, v.yres, &k, &mx, &my);
            sx = (px - mx) / k;             /* display -> guest pixels */
            sy = (py - my) / k;
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
    {
        id<MTLBuffer> wbuf = nil;
        NSUInteger ww = drawable.texture.width, wh = drawable.texture.height;
        const char *wdir = getenv("METAL_SHOT_DIR");

        if (want_window[0] && wdir && !m.layer.framebufferOnly) {
            id<MTLBlitCommandEncoder> wb = [cb blitCommandEncoder];

            wbuf = [m.device newBufferWithLength:ww * wh * 4 options:MTLResourceStorageModeShared];
            [wb copyFromTexture:drawable.texture sourceSlice:0 sourceLevel:0
                   sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(ww, wh, 1)
                       toBuffer:wbuf destinationOffset:0 destinationBytesPerRow:ww * 4
                destinationBytesPerImage:ww * wh * 4];
            [wb endEncoding];
        }
        [cb presentDrawable:drawable];
        [cb commit];
        if (wbuf) {
            NSString *path = [NSString stringWithFormat:@"%s/%s.png", wdir, want_window];
            CGColorSpaceRef wcs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            CGContextRef wctx;
            CGImageRef wimg;
            CGImageDestinationRef wdst;

            [cb waitUntilCompleted];        /* a test's, now and then: worth the wait */
            wctx = CGBitmapContextCreate([wbuf contents], ww, wh, 8, ww * 4, wcs,
                                         kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
            wimg = wctx ? CGBitmapContextCreateImage(wctx) : NULL;
            wdst = wimg ? CGImageDestinationCreateWithURL((CFURLRef)[NSURL fileURLWithPath:path],
                                                          CFSTR("public.png"), 1, NULL) : NULL;
            if (wdst) {
                CGImageDestinationAddImage(wdst, wimg, NULL);
                CGImageDestinationFinalize(wdst);
                CFRelease(wdst);
            }
            metal_log("shot: window %s.png (%lux%lu) %s", want_window, (unsigned long)ww,
                      (unsigned long)wh, wdst ? "written" : "FAILED");
            if (wimg) {
                CGImageRelease(wimg);
            }
            if (wctx) {
                CGContextRelease(wctx);
            }
            CGColorSpaceRelease(wcs);
            [wbuf release];
            want_window[0] = 0;
        }
    }
    pace_presented++;
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
    const char *dir = getenv("METAL_SHOT_DIR");     /* a test's own folder */

    if (dir && *dir) {
        base = [NSString stringWithUTF8String:dir];
    }

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

    if (backdrop.mode != METAL_BACKDROP_OFF
        && (backdrop.mode != METAL_BACKDROP_SHADER || backdrop.scene_tex[backdrop.scene_cur])) {
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
            sp.misc[3] = surf.on ? surf.view.look : 0;
            /* the window's tile size, carried into guest pixels */
            backdrop_params(&sp, 1.0);   /* the tile's size in guest pixels */
            srp.colorAttachments[0].texture = composite;
            srp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            srp.colorAttachments[0].storeAction = MTLStoreActionStore;
            senc = [cb renderCommandEncoderWithDescriptor:srp];
            [senc setRenderPipelineState:fb.shot];
            [senc setFragmentBytes:&sp length:sizeof(sp) atIndex:0];
            [senc setFragmentTexture:fb.decoded atIndex:0];
            if (backdrop.mode >= METAL_BACKDROP_PICTURE) {
                [senc setFragmentTexture:backdrop.mode == METAL_BACKDROP_SHADER
                                         ? backdrop.scene_tex[backdrop.scene_cur] : backdrop.image
                                 atIndex:1];
                [senc setFragmentSamplerState:m.repeat atIndex:1];
            }
            [senc setFragmentSamplerState:m.linear atIndex:0];
            [senc drawPrimitives:MTLPrimitiveTypeTriangle
                     vertexStart:0 vertexCount:3];
            /* the icons this window draws itself, as the window shows them */
            surf_faces(senc, surf.face_shot, CGSizeMake(fb.xres, fb.yres), fb.xres, fb.yres);
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
        MetalCursorView cv = { 0 };

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
    /* a test's pictures of the window itself read the drawable back */
    l.framebufferOnly = getenv("METAL_SHOT_DIR") ? NO : YES;
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

/* The Backdrop menu (MACOS.md, "The backdrop layer"), rebuilt each time
 * it opens, so a file dropped into the folder is there next time:
 *
 *   None
 *   Shaders ▸     the .metal files in the folder's Shaders (the built-in
 *                 Acorn scenes when there are none)
 *   Wallpapers ▸  the sprite files and images in the folder itself,
 *                 each a tile, or a picture if it is large
 *   Tiles ▸, Pictures ▸  the folder's Tiles and Pictures, if it has
 *                 them; Pictures has the Mac's own desktop pictures too
 *   Open Backdrops Folder, Choose Backdrops Folder..., and Use the
 *   Disc's Wallpapers when another folder has been chosen
 *
 * The folder is the disc's Resources/Wallpapers, which the launcher
 * passes as backdrop-dir=, unless another has been chosen.  A choice
 * applies at once and is kept as the backdrop default, which the app's
 * launcher passes next time.  The menu is there only when backdrop=
 * gates the feature on. */
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
                             NSString *empty, id target)
{
    NSMenu *menu = [[NSMenu alloc] initWithTitle:title];
    bool any = false;

    [menu setAutoenablesItems:NO];
    for (NSArray *group in groups) {
        if (group.count && any) {
            [menu addItem:[NSMenuItem separatorItem]];
        }
        for (NSString *path in group) {
            [menu addItem:backdrop_item(leaf_title([path lastPathComponent]),
                                        [NSString stringWithFormat:@"%@:%@", kind,
                                                  backdrop_relative(path)],
                                        target)];
            any = true;
        }
    }
    if (!any) {
        [menu addItem:backdrop_note(empty)];
    }
    return [menu autorelease];
}

static void backdrop_add_submenu(NSMenu *menu, NSString *title, NSMenu *sub)
{
    NSMenuItem *it = [[[NSMenuItem alloc] initWithTitle:title action:NULL
                                          keyEquivalent:@""] autorelease];

    [it setSubmenu:sub];
    [menu addItem:it];
}

- (void)menuNeedsUpdate:(NSMenu *)menu
{
    NSString *dir = backdrop_folder();
    NSFileManager *fm = [NSFileManager defaultManager];
    NSArray *shaders;
    NSString *sub;
    BOOL isdir = NO;

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
        [menu addItem:backdrop_note(@"RISC OS is not drawing a backdrop to show through:")];
        [menu addItem:backdrop_note(@"turn Composite on in Appearance's Wallpaper page")];
        [menu addItem:[NSMenuItem separatorItem]];
        break;
    case -1:
        [menu addItem:backdrop_note(@"Backdrops need a 16 million colour screen mode")];
        [menu addItem:[NSMenuItem separatorItem]];
        break;
    default:
        break;
    }
    [menu addItem:backdrop_item(@"None", @"none", self)];
    [menu addItem:[NSMenuItem separatorItem]];

    shaders = backdrop_files([dir stringByAppendingPathComponent:@"Shaders"], false, true);
    if (shaders.count) {
        backdrop_add_submenu(menu, @"Shaders",
                             backdrop_list(@"Shaders", @"shader", @[ shaders ], @"", self));
    } else {
        NSMenu *b = [[[NSMenu alloc] initWithTitle:@"Shaders"] autorelease];

        [b setAutoenablesItems:NO];
        [b addItem:backdrop_item(@"Acorn", @"acorn", self)];
        [b addItem:backdrop_item(@"Acorn, Moving", @"acorn-live", self)];
        [b addItem:backdrop_note(@"No .metal files in the Shaders folder")];
        backdrop_add_submenu(menu, @"Shaders", b);
    }
    backdrop_add_submenu(menu, @"Wallpapers",
                         backdrop_list(@"Wallpapers", @"wallpaper",
                                       @[ backdrop_files(dir, true, false) ],
                                       @"No sprite files or images in the folder", self));
    sub = [dir stringByAppendingPathComponent:@"Tiles"];
    if ([fm fileExistsAtPath:sub isDirectory:&isdir] && isdir) {
        backdrop_add_submenu(menu, @"Tiles",
                             backdrop_list(@"Tiles", @"tile", @[ backdrop_files(sub, true, false) ],
                                           @"No images in the Tiles folder", self));
    }
    backdrop_add_submenu(menu, @"Pictures",
        backdrop_list(@"Pictures", @"picture",
                      @[ backdrop_files([dir stringByAppendingPathComponent:@"Pictures"], true, false),
                         backdrop_files(@"/System/Library/Desktop Pictures", true, false) ],
                      @"No pictures", self));
    [menu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *open = [[[NSMenuItem alloc] initWithTitle:@"Open Backdrops Folder"
                                                   action:@selector(openBackdropsFolderAction:)
                                            keyEquivalent:@""] autorelease];
    [open setTarget:self];
    [menu addItem:open];
    NSMenuItem *choose = [[[NSMenuItem alloc] initWithTitle:@"Choose Backdrops Folder…"
                                                     action:@selector(chooseBackdropsFolderAction:)
                                              keyEquivalent:@""] autorelease];
    [choose setTarget:self];
    [menu addItem:choose];
    if (backdrop.dir.length && ![dir isEqualToString:backdrop.dir]) {
        NSMenuItem *disc = [[[NSMenuItem alloc] initWithTitle:@"Use the Disc's Wallpapers"
                                                       action:@selector(discBackdropsFolderAction:)
                                                keyEquivalent:@""] autorelease];
        [disc setTarget:self];
        [menu addItem:disc];
    }
}

- (void)backdropAction:(NSMenuItem *)sender
{
    NSString *spec = [sender representedObject];

    if (spec && backdrop.enabled) {
        backdrop_set_scene([spec UTF8String]);
        /* what the launcher passes next time: what took, so a scene that
         * would not compile is not tried again */
        [[NSUserDefaults standardUserDefaults] setObject:backdrop.spec forKey:@"backdrop"];
        [[NSUserDefaults standardUserDefaults] synchronize];
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

    (void)sender;
    [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES
                                               attributes:nil error:NULL];
    [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:dir]];
}

- (void)discBackdropsFolderAction:(id)sender
{
    (void)sender;
    [[NSUserDefaults standardUserDefaults] removeObjectForKey:@"backdropFolder"];
}

- (void)chooseBackdropsFolderAction:(id)sender
{
    NSOpenPanel *panel = [NSOpenPanel openPanel];

    (void)sender;
    panel.canChooseDirectories = YES;
    panel.canChooseFiles = NO;
    panel.allowsMultipleSelection = NO;
    panel.prompt = @"Use Folder";
    panel.message = @"Choose a folder of wallpapers -- sprite files and pictures -- "
                    @"with scenes in a Shaders folder inside it.";
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
    /* Where the user last left the window, else the middle of the screen
     * the first time */
    if (![m.window setFrameUsingName:@"BOXMetalWindow"]) {
        [m.window center];
    }
    [m.window setFrameAutosaveName:@"BOXMetalWindow"];

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
                frame_wanted = true;    /* input, a resize: draw */
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
            /* a moving backdrop's next frame, or its fault: none */
            if (backdrop.scene_failed) {
                metal_log("backdrop: the scene failed, so none");
                backdrop_set_scene("none");
            } else if (backdrop_scene_due()) {
                frame_wanted = true;
            }
            metal_render_frame();
            if (frame_skipped) {
                frame_skipped = false;
                usleep(8 * 1000);       /* nothing new: half a frame's wait */
            }
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

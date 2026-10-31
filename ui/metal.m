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
#import <QuartzCore/CATransaction.h>
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
#include <pthread.h>

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
        {
            /* over a page the Mac's cursor is shown, in WebKit's shape;
             * the guest's pointer is under the page there */
            bool in = !metal_browser_covers(p) && NSPointInRect(p, b);

            if (in != cursor.inside) {
                cursor.inside = in;
                metal_cursor_sync();
            }
        }
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

/* Every other key the guest has been told is down, by macOS key code.
 * The key-up goes to whichever window has the keyboard when the key is
 * let go: when the browser's page takes it (a Return in !Browser's
 * address field opens a window), or a Command-Tab leaves, the guest
 * never hears the key go up and RISC OS repeats it for ever. */
static uint8_t keys_held[128];

static void metal_key_held(uint32_t code, bool down)
{
    if (code < sizeof(keys_held)) {
        keys_held[code] = down;
    }
}

static void metal_release_keys(void)
{
    uint32_t i;

    for (i = 0; i < sizeof(keys_held); i++) {
        if (keys_held[i]) {
            keys_held[i] = 0;
            metal_glue_key(false, i);
        }
    }
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
 * already there, the PC presenter's rounded-box field), then its pixels.
 * The scale pass then lays that over the backdrop as it lays the decoded
 * screen.
 *
 * The pixels are never read where the Wimp draws them.  The Wimp
 * publishes a list when every backing is whole (at Wimp_Poll), and the
 * next task begins drawing at once: a frame read from the backings
 * themselves showed windows cleared and half drawn, flickering.  So each
 * new list's damage is copied, by the GPU, into a texture per backing,
 * and the list is acked only when the copy is done; the Wimp draws into
 * no backing till then (ROSGD_PRESENT_SNAPSHOT).  Every frame shown is a
 * list exactly as it was published.
 */
static NSString *const SURF_SRC = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct SQuad {\n"
"    float4 rect;    /* where it draws: x0 y0 x1 y1, guest pixels */\n"
"    float4 box;     /* the surface (or the shadow's dropped box): x y w h */\n"
"    uint4 u;        /* offset, stride (bytes), kind (0 pixels, 1 shadow), format */\n"
"    float4 sh;      /* shadow: radius, blur, strength 0..1, 0; a page: its pixels w, h */\n"
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
"    if (Q.u.w != 0) {\n"
"        /* a Linux program's page: B G R A bytes, its own size scaled to\n"
"         * the window's, premultiplied (format 1) or opaque (2) */\n"
"        uint2 b = uint2(float2(l) * Q.sh.xy / Q.box.zw);\n"
"        uint w = region[(Q.u.x + b.y * Q.u.y) / 4 + b.x];\n"
"        float4 c = float4(float((w >> 16) & 255), float((w >> 8) & 255), float(w & 255),\n"
"                          float(w >> 24)) / 255.0;\n"
"        return Q.u.w == 2 ? float4(c.rgb, 1.0) : c;\n"
"    }\n"
"    uint w = region[(Q.u.x + uint(l.y) * Q.u.y) / 4 + uint(l.x)];\n"
"    if ((w >> 24) == 0x80) {\n"
"        return float4(0.0);     /* the below tag: a hole to the backdrop */\n"
"    }\n"
"    return float4(float(w & 255), float((w >> 8) & 255), float((w >> 16) & 255),\n"
"                  float(w >> 24)) / 255.0;  /* premultiplied */\n"
"}\n"
/* the same, from a copy of the backing in the GPU's own memory (a GPU
 * without unified memory, which reads guest RAM across its bus) */
"fragment float4 ps_surf_tex(SOut in [[stage_in]],\n"
"                            device const SQuad *q [[buffer(0)]],\n"
"                            texture2d<float> t [[texture(0)]])\n"
"{\n"
"    SQuad Q = q[in.inst];\n"
"    int2 l = int2(floor(in.pos.xy - Q.box.xy));\n"
"    if (l.x < 0 || l.y < 0 || l.x >= int(Q.box.z) || l.y >= int(Q.box.w)) {\n"
"        discard_fragment();\n"
"    }\n"
"    float4 c = t.read(uint2(l));\n"
"    if (uint(round(c.a * 255.0)) == 0x80) {\n"
"        return float4(0.0);     /* the below tag: a hole to the backdrop */\n"
"    }\n"
"    return c;                   /* premultiplied */\n"
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

/* The backings' copies in the GPU's own memory (tex_mode): by their place
 * in the region and their size; made whole, then kept up to date by the
 * damage each new list carries; let go when unused a while */
#define SURF_TEX 160
static struct {
    uint32_t window, offset, w, h, stride;
    id<MTLTexture> tex;
    uint32_t idle;                      /* lists since it was last in one */
} stex[SURF_TEX];

/* per surface of the frame: its quads in the buffer, shadow then pixels */
static uint32_t sq_sh0[METAL_SURFACES], sq_shn[METAL_SURFACES];
static uint32_t sq_px0[METAL_SURFACES], sq_pxn[METAL_SURFACES];

static struct {
    bool tried;
    id<MTLRenderPipelineState> pipe;
    id<MTLBuffer> region;               /* the guest's region, no copy */
    void *region_host;
    uint64_t region_size;
    id<MTLBuffer> quads[METAL_RING];
    unsigned ring;
    MetalSurfacesView view;             /* the frame's: a copy of taken, the render loop's own */
    MetalSurfacesView taken;            /* the list last taken, its damage copied (under mx) */
    pthread_mutex_t mx;                 /* taken, the copies, the board's view: the taker's and
                                           the render loop's in turn */
    bool taker;                         /* the taking thread is running */
    bool on;                            /* this frame composites the list */
    id<MTLRenderPipelineState> face;    /* the clock's and the meter's, into the drawable */
    id<MTLRenderPipelineState> tex_pipe;    /* the surfaces from their GPU copies */
    bool tex_mode;                      /* a GPU without unified memory: copies */
    uint32_t uploaded_frame;            /* the list whose damage is in the copies */
    id<MTLRenderPipelineState> face_shot;   /* the same, into a screenshot's RGBA texture */
    uint32_t drawn_frame, drawn_pages;
    double face_due;                    /* when a listed clock is next to be drawn */
    uint64_t frames, quads_drawn;
    uint64_t copied, copies;            /* bytes copied out of the backings, lists copied */
} surf = { .mx = PTHREAD_MUTEX_INITIALIZER };

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
    {
        id<MTLFunction> tfs = [lib newFunctionWithName:@"ps_surf_tex"];

        pd.fragmentFunction = tfs;
        surf.tex_pipe = tfs ? [m.device newRenderPipelineStateWithDescriptor:pd error:nil] : nil;
        [tfs release];
    }
    /* copies of the backings, on every GPU: a list as it was published,
     * never the Wimp part way through its next (and on a GPU with memory
     * of its own, not guest RAM read across the bus for every pixel of
     * every frame, which drew 8 frames a second on the Mac Pro's Radeon) */
    surf.tex_mode = surf.tex_pipe != nil;
    metal_log("surfaces: %s", surf.tex_mode ? "copies of the backings, kept by their damage"
                                            : "NO copies: the backings read where they lie");
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
    MetalSurfacesView *v = &surf.taken;

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

        sq_sh0[i] = n;
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
        sq_shn[i] = n - sq_sh0[i];
        sq_px0[i] = n;
        for (uint32_t k = 0; k < e->nrects && n < SURF_QUADS; k++) {
            const int16_t *r = v->rect[e->rects + k];
            SQuad *o = &q[n++];

            o->rect[0] = r[0], o->rect[1] = r[1], o->rect[2] = r[2], o->rect[3] = r[3];
            o->box[0] = e->x, o->box[1] = e->y, o->box[2] = e->w, o->box[3] = e->h;
            o->u[0] = e->offset, o->u[1] = e->stride, o->u[2] = 0, o->u[3] = e->format;
            memset(o->sh, 0, sizeof(o->sh));
            o->sh[0] = e->bw, o->sh[1] = e->bh;
        }
        sq_pxn[i] = n - sq_px0[i];
    }
    return n;
}

/* A backing's copy, found by its window and place in the region; made
 * (and *whole set) when there is none.  nil: no room for another. */
static id<MTLTexture> surf_texture(const MetalSurface *e, bool *whole)
{
    int k, spare = -1;

    *whole = false;
    for (k = 0; k < SURF_TEX; k++) {
        if (stex[k].tex && stex[k].window == e->window && stex[k].offset == e->offset
            && stex[k].w == (uint32_t)e->w && stex[k].h == (uint32_t)e->h
            && stex[k].stride == e->stride) {
            return stex[k].tex;
        }
        if (spare < 0 && (!stex[k].tex || stex[k].idle > 120)) {
            spare = k;
        }
    }
    if (spare < 0) {
        return nil;
    }
    k = spare;
    [stex[k].tex release];
    {
        MTLTextureDescriptor *td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:e->w height:e->h
                                                           mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        stex[k].tex = [m.device newTextureWithDescriptor:td];
    }
    stex[k].window = e->window, stex[k].offset = e->offset;
    stex[k].w = e->w, stex[k].h = e->h, stex[k].stride = e->stride;
    stex[k].idle = 0;
    *whole = stex[k].tex != nil;
    return stex[k].tex;
}

/* A backing's copy if it has one */
static id<MTLTexture> surf_texture_of(const MetalSurface *e)
{
    for (int k = 0; k < SURF_TEX; k++) {
        if (stex[k].tex && stex[k].window == e->window && stex[k].offset == e->offset
            && stex[k].w == (uint32_t)e->w && stex[k].h == (uint32_t)e->h
            && stex[k].stride == e->stride) {
            return stex[k].tex;
        }
    }
    return nil;
}

/* The copy of one backing's rectangle into its texture */
static void surf_blit(id<MTLBlitCommandEncoder> b, id<MTLTexture> t, const MetalSurface *e,
                      int x0, int y0, int x1, int y1)
{
    x0 = MAX(x0, 0), y0 = MAX(y0, 0), x1 = MIN(x1, e->w), y1 = MIN(y1, e->h);
    if (x1 <= x0 || y1 <= y0) {
        return;
    }
    [b copyFromBuffer:surf.region
         sourceOffset:e->offset + (NSUInteger)y0 * e->stride + (NSUInteger)x0 * 4
    sourceBytesPerRow:e->stride
  sourceBytesPerImage:(NSUInteger)e->stride * (NSUInteger)(y1 - y0)
           sourceSize:MTLSizeMake(x1 - x0, y1 - y0, 1)
            toTexture:t
     destinationSlice:0
     destinationLevel:0
    destinationOrigin:MTLOriginMake(x0, y0, 0)];
}

/* A new list taken: its damage copied out of the backings into their
 * textures, by the GPU, waited for, and then the list acked -- the Wimp
 * draws into the backings again only after that.  Called at every pass of
 * the loop, a frame presented or not (and with the window minimised), so
 * that the Wimp is never kept waiting for a frame nobody sees. */
static void surf_take(void)
{
    const MetalSurfacesView *v = &surf.taken;
    id<MTLCommandBuffer> cb;
    id<MTLBlitCommandEncoder> b = nil;
    uint64_t bytes = 0;

    if (v->frame == surf.uploaded_frame) {
        return;
    }
    if (surf.on && surf.tex_mode) {
        for (int k = 0; k < SURF_TEX; k++) {
            stex[k].idle++;
        }
        cb = [m.queue commandBuffer];
        for (uint32_t i = 0; i < v->n; i++) {
            const MetalSurface *e = &v->s[i];
            bool whole;
            id<MTLTexture> t;

            if (e->format != 0 || !(t = surf_texture(e, &whole))) {
                continue;
            }
            for (int k = 0; k < SURF_TEX; k++) {
                if (stex[k].tex == t) {
                    stex[k].idle = 0;
                }
            }
            if (!whole && e->ndamage == 0) {
                continue;
            }
            if (!b) {
                b = [cb blitCommandEncoder];
            }
            if (whole || e->ndamage == METAL_DAMAGE_ALL) {
                surf_blit(b, t, e, 0, 0, e->w, e->h);
                bytes += (uint64_t)e->w * e->h * 4;
            } else {
                for (uint32_t r = 0; r < e->ndamage; r++) {
                    const int16_t *q = v->rect[e->damage + r];

                    surf_blit(b, t, e, q[0], q[1], q[2], q[3]);
                    bytes += (uint64_t)MAX(q[2] - q[0], 0) * MAX(q[3] - q[1], 0) * 4;
                }
            }
        }
        if (b) {
            [b endEncoding];
            [cb commit];
            /* the copy, and any frame still reading these textures, done:
             * then the backings are the Wimp's again */
            [cb waitUntilCompleted];
            surf.copied += bytes;
            surf.copies++;
        }
    }
    surf.uploaded_frame = v->frame;
    metal_glue_surfaces_ack(v->frame);
}

/*
 * The taker: a list the box publishes is taken, its damage copied and the
 * list acked within a millisecond, whatever the render loop is doing.  The
 * loop takes lists too, but only at each pass, and a pass may wait most of
 * a frame for a drawable; meanwhile the Wimp, which draws into no backing
 * until its last list is acked, stood still -- a drag spent most of its
 * time so (17 Oct).
 */
static void *surf_taker(void *arg)
{
    unsigned n = 0;

    (void)arg;
    for (;;) {
        usleep(500);
        pthread_mutex_lock(&surf.mx);
        if (++n % 200 == 0) {
            metal_glue_beat();          /* ten a second, a frame drawn or not */
        }
        if (surf.on && metal_glue_surfaces_pending()) {
            metal_glue_surfaces_view(&surf.taken);
            @autoreleasepool {
                surf_take();
            }
        }
        pthread_mutex_unlock(&surf.mx);
    }
    return NULL;
}

typedef struct FaceU {
    float box[4], clip[4], target[2];
    uint32_t style, kind;
    float hms[4];
    float hist[24];
} FaceU;

/* ---- a game pane, drawn here (design 50) -------------------------------
 *
 * On the PC a game pane renders itself: there is a GPU in the box.  On
 * the Mac there is none -- the guest's Mesa is llvmpipe and virtio-gpu
 * is a scanout -- so the guest writes its planes and its instance lists
 * into a block, and this draws them.  The guest does no rendering at
 * all; its layers keep the same interfaces they have on the PC, and
 * only the last call of a frame differs.
 *
 * M0 is the canvas: the index plane resolved through two palettes.
 * Indices 16..255 come from the global palette and 0..15 from the
 * PER-SCANLINE one, which is what makes this a retro canvas rather than
 * a framebuffer -- a gradient sky or a copper bar costs a palette row
 * instead of a screen of pixels.  Over it go the sprites, every one in
 * one instanced draw (M1), and the text, likewise (M2); under it, when
 * the block names one, a cosmos scene (M3) -- the twelve backdrops the
 * PC draws, generated into MSL by the same script that makes its GLSL.
 *
 * The three are drawn into the pane's own 640x360 picture, in a pass
 * before the frame's main one, and the picture is then shown at the
 * window's size a canvas pixel to a block of screen pixels -- as the
 * PC's pane draws into its canvas and scales that to the screen.  A
 * sprite lands on whole canvas pixels, and is never smeared across
 * two, whatever the zoom.
 *
 * NOTHING HERE TRUSTS THE BLOCK.  It is memory a guest program wrote.
 * The magic, the version and the size are checked, the counts are
 * clamped, and the planes are read at fixed offsets so there is no
 * offset arithmetic to get wrong.
 */
#define GP_MAGIC    0x4E415047u         /* "GPAN" */
#define GP_VERSION  1u
#define GP_W        640
#define GP_H        360
#define GP_PAL_GLOBAL 256
#define GP_PAL_LINE 16
#define GP_CANVAS_OVER 1u

/* The block's head, and the planes' places in it.  It mirrors
 * include/rosgd/gamepane.h in the box; the _Static_asserts there and
 * the size check here are what keep the two honest. */
typedef struct GpHead {
    uint32_t magic, version, frame, flags, scene;
    float    seconds;
    uint32_t nsprites, nglyphs, art_frame, reserved[7];
} GpHead;
#define GP_OFF_IDX        (sizeof(GpHead))
#define GP_OFF_PAL_GLOBAL (GP_OFF_IDX + (size_t)GP_W * GP_H)
#define GP_OFF_PAL_LINE   (GP_OFF_PAL_GLOBAL + (size_t)GP_PAL_GLOBAL * 4)
#define GP_SPRITES  8192
#define GP_GLYPHS   4096
#define GP_SLOTS    32
#define GP_COLOURS  16
#define GP_ATLAS    512
#define GP_FONT_W   128
#define GP_FONT_H   64
#define GP_OFF_SPRITE     (GP_OFF_PAL_LINE + (size_t)GP_PAL_LINE * GP_H * 4)
#define GP_OFF_GLYPH      (GP_OFF_SPRITE + (size_t)GP_SPRITES * 32)
#define GP_OFF_ATLAS      (GP_OFF_GLYPH + (size_t)GP_GLYPHS * 32)
#define GP_OFF_SPAL       (GP_OFF_ATLAS + (size_t)GP_ATLAS * GP_ATLAS)
#define GP_OFF_FONT       (GP_OFF_SPAL + (size_t)GP_SLOTS * GP_COLOURS * 4)
#define GP_SIZE           (GP_OFF_FONT + (size_t)GP_FONT_W * GP_FONT_H)
/* design 56's tile layers, after all of that: a block with them is known
   by its size, so a block without them still draws */
#define GP_LAYERS   4
#define GP_MAP_W    128
#define GP_MAP_H    64
#define GP_TILES_W  256
#define GP_TILES_H  128
#define GP_OFF_LAYER      GP_SIZE
#define GP_OFF_MAP        (GP_OFF_LAYER + (size_t)GP_LAYERS * 32)
#define GP_OFF_TILES      (GP_OFF_MAP + (size_t)GP_LAYERS * GP_MAP_H * GP_MAP_W * 2)
#define GP_SIZE_TILES     (GP_OFF_TILES + (size_t)GP_TILES_W * GP_TILES_H)

/* a game pane's 32 bytes in its surface entry */
typedef struct GpData {
    uint32_t offset, size;
    uint16_t canvas_w, canvas_h;
    uint8_t  reserved[16];
} GpData;

#include "metal-cosmos.h"

static NSString * const GP_SRC = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct GpU {\n"
"    float4 box;          /* the pane on the target, in target pixels */\n"
"    float4 clip;         /* the rectangle being drawn */\n"
"    float2 target;\n"
"    uint   over;         /* 1: index 0 is a hole */\n"
"    uint   pad;\n"
"};\n"
"struct GpV { float4 pos [[position]]; float2 uv; };\n"
"\n"
"/* the clipped rectangle as a strip, so each visible piece of the pane\n"
"   is one draw and nothing is painted where the pane cannot be seen */\n"
"vertex GpV gp_vs(uint id [[vertex_id]], constant GpU &U [[buffer(0)]])\n"
"{\n"
"    float2 c = float2(id & 1, (id >> 1) & 1);\n"
"    float2 p = mix(U.clip.xy, U.clip.zw, c);\n"
"    GpV o;\n"
"    o.pos = float4(p / U.target * float2(2, -2) + float2(-1, 1), 0, 1);\n"
"    /* where that lands in the pane's own 0..1 */\n"
"    o.uv = (p - U.box.xy) / U.box.zw;\n"
"    return o;\n"
"}\n"
"\n"
"/* layer 1, into the pane's own 640x360 frame */\n"
"fragment float4 gp_canvas(GpV v [[stage_in]], constant GpU &U [[buffer(0)]],\n"
"                          texture2d<uint> idx [[texture(0)]],\n"
"                          texture2d<float> palGlobal [[texture(1)]],\n"
"                          texture2d<float> palLine [[texture(2)]])\n"
"{\n"
"    int px = clamp(int(v.uv.x * 640.0), 0, 639);\n"
"    int py = clamp(int(v.uv.y * 360.0), 0, 359);\n"
"    uint c = idx.read(uint2(px, py)).r;\n"
"    if (U.over != 0u && c == 0u) {\n"
"        discard_fragment();\n"
"    }\n"
"    float4 p = c < 16u ? palLine.read(uint2(c, py))\n"
"                       : palGlobal.read(uint2(c, 0));\n"
"    /* the palettes keep the Direct3D backend's B G R A, as they do on\n"
"       the PC, so a game's palette data needs no rearranging on either\n"
"       machine; the alpha is the canvas's own and never the palette's */\n"
"    return float4(p.bgr, 1.0);\n"
"}\n"
"\n"
"/* Design 56's tile layers, all four in one pass, the front one first:\n"
"   the first cell with a pixel that is not 0 is the colour.  A map\n"
"   repeats at its own size, so the scroll is two integers and the\n"
"   pixel's place is a remainder, kept positive by hand. */\n"
"struct GpTL { int sx; int sy; ushort mw; ushort mh; uint on; uint res[4]; };\n"
"fragment float4 gp_tiles(GpV v [[stage_in]], constant GpTL *L [[buffer(1)]],\n"
"                         texture2d<uint> map [[texture(0)]],\n"
"                         texture2d<uint> tiles [[texture(1)]],\n"
"                         texture2d<float> palGlobal [[texture(2)]],\n"
"                         texture2d<float> palLine [[texture(3)]])\n"
"{\n"
"    int px = clamp(int(v.uv.x * 640.0), 0, 639);\n"
"    int py = clamp(int(v.uv.y * 360.0), 0, 359);\n"
"    for (int l = 3; l >= 0; l--) {\n"
"        if (L[l].on == 0u || L[l].mw == 0 || L[l].mh == 0) {\n"
"            continue;\n"
"        }\n"
"        int pw = int(L[l].mw) * 8, ph = int(L[l].mh) * 8;\n"
"        int wx = ((px + L[l].sx) % pw + pw) % pw;\n"
"        int wy = ((py + L[l].sy) % ph + ph) % ph;\n"
"        uint cell = map.read(uint2(wx >> 3, l * 64 + (wy >> 3))).r;\n"
"        if (cell == 0u || cell > 512u) {\n"
"            continue;\n"
"        }\n"
"        uint t = cell - 1u;\n"
"        uint c = tiles.read(uint2((t % 32u) * 8u + uint(wx & 7),\n"
"                                  (t / 32u) * 8u + uint(wy & 7))).r;\n"
"        if (c == 0u) {\n"
"            continue;\n"
"        }\n"
"        float4 p = c < 16u ? palLine.read(uint2(c, py))\n"
"                           : palGlobal.read(uint2(c, 0));\n"
"        return float4(p.bgr, 1.0);\n"
"    }\n"
"    discard_fragment();\n"
"    return float4(0);\n"
"}\n"
"\n"
"/* The finished frame onto the screen: a canvas pixel to a block of\n"
"   screen pixels, never blended between two, as a retro screen is */\n"
"fragment float4 gp_show(GpV v [[stage_in]], texture2d<float> frame [[texture(0)]])\n"
"{\n"
"    int px = clamp(int(v.uv.x * 640.0), 0, 639);\n"
"    int py = clamp(int(v.uv.y * 360.0), 0, 359);\n"
"    return float4(frame.read(uint2(px, py)).rgb, 1.0);\n"
"}\n"
"\n"
"/* Layer 2: every sprite in one instanced draw.  A record is the guest's\n"
"   own eight floats (x, y, ax, ay, w, h, slot, alpha), read as two\n"
"   float4s, so a frame's list crosses with one memcpy. */\n"
"struct GpSprite { float4 a; float4 b; };\n"
"struct GpSV {\n"
"    float4 pos [[position]];\n"
"    float2 atl;\n"
"    float2 org [[flat]];\n"
"    float2 wh [[flat]];\n"
"    float  slot [[flat]];\n"
"    float  alpha [[flat]];\n"
"};\n"
"vertex GpSV gp_spr_vs(uint id [[vertex_id]], uint inst [[instance_id]],\n"
"                      const device GpSprite *s [[buffer(0)]])\n"
"{\n"
"    GpSprite r = s[inst];\n"
"    float2 corner = float2(id & 1, (id >> 1) & 1);\n"
"    float2 wh = clamp(r.b.xy, float2(0), float2(512));\n"
"    /* the slot's high bits: 32 flip x, 64 flip y, 128 and 256 the\n"
"       scale less one */\n"
"    int sv = int(r.b.z + 0.5);\n"
"    float scale = float(((sv >> 7) & 3) + 1);\n"
"    float2 p = r.a.xy + corner * wh * scale;\n"
"    float2 from = float2((sv & 32) ? 1.0 - corner.x : corner.x,\n"
"                         (sv & 64) ? 1.0 - corner.y : corner.y);\n"
"    GpSV o;\n"
"    /* the canvas is 640x360, so 320 and 180 are its half-extents */\n"
"    o.pos = float4(p.x / 320.0 - 1.0, 1.0 - p.y / 180.0, 0, 1);\n"
"    o.atl = r.a.zw + from * wh;\n"
"    o.org = r.a.zw;\n"
"    o.wh = wh;\n"
"    o.slot = float(sv & 31);\n"
"    o.alpha = clamp(r.b.w, 0.0, 1.0);\n"
"    return o;\n"
"}\n"
"fragment float4 gp_spr(GpSV v [[stage_in]], texture2d<uint> atlas [[texture(0)]],\n"
"                       texture2d<float> spal [[texture(1)]])\n"
"{\n"
"    int x0 = int(v.org.x), y0 = int(v.org.y);\n"
"    /* to this sprite's own rect first, then to the sheet: the first\n"
"       clamp is what stops a sprite fringing with its neighbour */\n"
"    int ax = clamp(int(v.atl.x), x0, x0 + int(v.wh.x) - 1);\n"
"    int ay = clamp(int(v.atl.y), y0, y0 + int(v.wh.y) - 1);\n"
"    ax = clamp(ax, 0, 511);\n"
"    ay = clamp(ay, 0, 511);\n"
"    uint c = atlas.read(uint2(ax, ay)).r & 15u;\n"
"    if (c == 0u) {\n"
"        discard_fragment();\n"
"    }\n"
"    int slot = clamp(int(v.slot + 0.5), 0, 31);\n"
"    float4 p = spal.read(uint2(c, slot));\n"
"    return float4(p.bgr, v.alpha);\n"
"}\n"
"\n"
"/* Layer 3: every glyph in one instanced draw; the colour travels in\n"
"   the instance, which is what makes a title, a score and an overlay\n"
"   one draw.  The record is (x, y, cx, cy, r, g, b, scale). */\n"
"struct GpTV {\n"
"    float4 pos [[position]];\n"
"    float2 atl;\n"
"    float2 org [[flat]];\n"
"    float3 col [[flat]];\n"
"};\n"
"vertex GpTV gp_txt_vs(uint id [[vertex_id]], uint inst [[instance_id]],\n"
"                      const device GpSprite *g [[buffer(0)]])\n"
"{\n"
"    GpSprite r = g[inst];\n"
"    float2 corner = float2(id & 1, (id >> 1) & 1);\n"
"    float2 p = r.a.xy + corner * 8.0 * clamp(r.b.w, 0.0, 16.0);\n"
"    GpTV o;\n"
"    o.pos = float4(p.x / 320.0 - 1.0, 1.0 - p.y / 180.0, 0, 1);\n"
"    o.atl = r.a.zw + corner * 8.0;\n"
"    o.org = r.a.zw;\n"
"    o.col = clamp(float3(r.b.xyz), float3(0), float3(1));\n"
"    return o;\n"
"}\n"
"fragment float4 gp_txt(GpTV v [[stage_in]], texture2d<uint> font [[texture(0)]])\n"
"{\n"
"    int x0 = int(v.org.x), y0 = int(v.org.y);\n"
"    /* to this glyph's own cell first, then to the atlas, against a\n"
"       garbage instance */\n"
"    int ax = clamp(clamp(int(v.atl.x), x0, x0 + 7), 0, 127);\n"
"    int ay = clamp(clamp(int(v.atl.y), y0, y0 + 7), 0, 63);\n"
"    if (font.read(uint2(ax, ay)).r == 0u) {\n"
"        discard_fragment();\n"
"    }\n"
"    return float4(v.col, 1.0);\n"
"}\n";

typedef struct GpU {
    float box[4], clip[4], target[2];
    uint32_t over, pad;
} GpU;

/* A pane's own textures and its own picture.  Each pane on the screen
 * has one (design 56: !GamePane runs a program in each of several
 * windows at once), found by its block's offset in the region, which is
 * the pane's identity for as long as it is open. */
#define GP_PANES 4

typedef struct GpPane {
    bool used;
    uint32_t offset;                    /* the block's, in the region */
    uint64_t seen;                      /* the frame it was last on the screen */
    id<MTLTexture> idx, pal_global, pal_line, atlas, spal, font;
    id<MTLTexture> map, tiles;          /* design 56's tile layers */
    id<MTLTexture> frame;               /* the pane's own 640x360 picture */
    uint32_t drawn;                     /* the block frame the picture is of */
    uint32_t art;                       /* the art_frame last uploaded */
    bool art_any;
} GpPane;

static struct {
    id<MTLRenderPipelineState> cosmos, canvas, show, spr, txt, tiles;
    GpPane pane[GP_PANES];
    uint64_t tick;
    bool tried;
} gp;

static id<MTLRenderPipelineState> gp_pipe(id<MTLLibrary> lib, NSString *vs, NSString *fs,
                                          bool blend)
{
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    NSError *err = nil;
    id<MTLRenderPipelineState> p;

    pd.vertexFunction = [lib newFunctionWithName:vs];
    pd.fragmentFunction = [lib newFunctionWithName:fs];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (blend) {
        /* G3's own blend state: the sprite's alpha over the canvas */
        pd.colorAttachments[0].blendingEnabled = YES;
        pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    }
    p = [m.device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!p) {
        metal_log("game pane %s: %s", [fs UTF8String], [[err localizedDescription] UTF8String]);
    }
    return p;
}

static id<MTLTexture> gp_texture(MTLPixelFormat f, NSUInteger w, NSUInteger h, MTLTextureUsage u)
{
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f
                                                                                  width:w height:h
                                                                              mipmapped:NO];
    td.usage = u;
    return [m.device newTextureWithDescriptor:td];
}

static void gp_setup(void)
{
    NSError *err = nil;
    id<MTLLibrary> lib;

    if (gp.tried) {
        return;
    }
    gp.tried = true;
    /* the pane's own passes, and the twelve scenes appended: they are
       generated (metal-cosmos.h), and declare nothing the passes do */
    lib = [m.device newLibraryWithSource:[NSString stringWithFormat:@"%@%s", GP_SRC, GP_COSMOS_MSL]
                                 options:nil error:&err];
    if (!lib) {
        metal_log("game pane: %s", [[err localizedDescription] UTF8String]);
        return;
    }
    gp.cosmos = gp_pipe(lib, @"gp_vs", @"gp_cosmos", false);
    gp.canvas = gp_pipe(lib, @"gp_vs", @"gp_canvas", false);
    gp.show = gp_pipe(lib, @"gp_vs", @"gp_show", false);
    gp.spr = gp_pipe(lib, @"gp_spr_vs", @"gp_spr", true);
    gp.txt = gp_pipe(lib, @"gp_txt_vs", @"gp_txt", false);
    gp.tiles = gp_pipe(lib, @"gp_vs", @"gp_tiles", false);
    if (!gp.cosmos || !gp.canvas || !gp.show || !gp.spr || !gp.txt || !gp.tiles) {
        gp.canvas = nil;                /* said in the log; the desktop goes on */
        return;
    }
}

/* The pane whose block is at this offset: the one it had, or a free one,
 * or the one longest off the screen.  Its textures are made the first
 * time, at the canvas's fixed size: a retro canvas with a settable
 * resolution is a framebuffer with extra steps. */
static GpPane *gp_pane_for(uint32_t offset)
{
    GpPane *p = NULL;

    for (int i = 0; i < GP_PANES; i++) {
        if (gp.pane[i].used && gp.pane[i].offset == offset) {
            return &gp.pane[i];
        }
    }
    for (int i = 0; i < GP_PANES && !p; i++) {
        if (!gp.pane[i].used) {
            p = &gp.pane[i];
        }
    }
    if (!p) {
        p = &gp.pane[0];
        for (int i = 1; i < GP_PANES; i++) {
            if (gp.pane[i].seen < p->seen) {
                p = &gp.pane[i];
            }
        }
    }
    if (!p->frame) {
        p->idx = gp_texture(MTLPixelFormatR8Uint, GP_W, GP_H, MTLTextureUsageShaderRead);
        p->pal_global = gp_texture(MTLPixelFormatRGBA8Unorm, GP_PAL_GLOBAL, 1, MTLTextureUsageShaderRead);
        p->pal_line = gp_texture(MTLPixelFormatRGBA8Unorm, GP_PAL_LINE, GP_H, MTLTextureUsageShaderRead);
        p->atlas = gp_texture(MTLPixelFormatR8Uint, GP_ATLAS, GP_ATLAS, MTLTextureUsageShaderRead);
        p->spal = gp_texture(MTLPixelFormatRGBA8Unorm, GP_COLOURS, GP_SLOTS, MTLTextureUsageShaderRead);
        p->font = gp_texture(MTLPixelFormatR8Uint, GP_FONT_W, GP_FONT_H, MTLTextureUsageShaderRead);
        /* the four maps one above the other */
        p->map = gp_texture(MTLPixelFormatR16Uint, GP_MAP_W, GP_MAP_H * GP_LAYERS,
                            MTLTextureUsageShaderRead);
        p->tiles = gp_texture(MTLPixelFormatR8Uint, GP_TILES_W, GP_TILES_H, MTLTextureUsageShaderRead);
        p->frame = gp_texture(MTLPixelFormatBGRA8Unorm, GP_W, GP_H,
                              MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
    }
    p->used = true;
    p->offset = offset;
    p->drawn = 0;                       /* a new pane: nothing of it drawn yet */
    p->art_any = false;
    return p;
}

/* A frame of the block into the pane's own picture, if the block holds
 * one worth taking.  Returns false when it does not -- then the last
 * picture is shown again, because a pane that tears is worse than one
 * that repeats, and at sixty frames a second nobody sees the repeat. */
static bool gp_render(id<MTLCommandBuffer> cb, GpPane *p, const uint8_t *blk, size_t size)
{
    GpHead h;
    uint32_t after, ns, ng;
    uint8_t layers[GP_LAYERS * 32];
    bool tiled = false;
    id<MTLBuffer> sprites = nil, glyphs = nil;
    MTLRenderPassDescriptor *rp;
    id<MTLRenderCommandEncoder> enc;
    GpU u;

    if (size < GP_SIZE) {
        return false;
    }
    memcpy(&h, blk, sizeof(h));
    if (h.magic != GP_MAGIC || h.version != GP_VERSION || (h.frame & 1u)) {
        return false;                   /* not a block, or being written */
    }
    if (h.frame == p->drawn) {
        return true;                    /* unchanged: the picture stands */
    }
    [p->idx replaceRegion:MTLRegionMake2D(0, 0, GP_W, GP_H) mipmapLevel:0
                withBytes:blk + GP_OFF_IDX bytesPerRow:GP_W];
    [p->pal_global replaceRegion:MTLRegionMake2D(0, 0, GP_PAL_GLOBAL, 1) mipmapLevel:0
                       withBytes:blk + GP_OFF_PAL_GLOBAL bytesPerRow:GP_PAL_GLOBAL * 4];
    [p->pal_line replaceRegion:MTLRegionMake2D(0, 0, GP_PAL_LINE, GP_H) mipmapLevel:0
                     withBytes:blk + GP_OFF_PAL_LINE bytesPerRow:GP_PAL_LINE * 4];
    /* the art changes when a game loads it and almost never after: taken
       only when its number moves, which is the difference between 900 KB
       a frame and 25 */
    if (!p->art_any || h.art_frame != p->art) {
        [p->atlas replaceRegion:MTLRegionMake2D(0, 0, GP_ATLAS, GP_ATLAS) mipmapLevel:0
                      withBytes:blk + GP_OFF_ATLAS bytesPerRow:GP_ATLAS];
        [p->spal replaceRegion:MTLRegionMake2D(0, 0, GP_COLOURS, GP_SLOTS) mipmapLevel:0
                     withBytes:blk + GP_OFF_SPAL bytesPerRow:GP_COLOURS * 4];
        [p->font replaceRegion:MTLRegionMake2D(0, 0, GP_FONT_W, GP_FONT_H) mipmapLevel:0
                     withBytes:blk + GP_OFF_FONT bytesPerRow:GP_FONT_W];
        if (size >= GP_SIZE_TILES) {
            [p->tiles replaceRegion:MTLRegionMake2D(0, 0, GP_TILES_W, GP_TILES_H) mipmapLevel:0
                          withBytes:blk + GP_OFF_TILES bytesPerRow:GP_TILES_W];
        }
    }
    if (size >= GP_SIZE_TILES) {
        memcpy(layers, blk + GP_OFF_LAYER, sizeof(layers));
        for (int l = 0; l < GP_LAYERS; l++) {
            uint32_t on;
            memcpy(&on, layers + l * 32 + 12, sizeof(on));
            tiled |= on != 0;
        }
        if (tiled) {
            [p->map replaceRegion:MTLRegionMake2D(0, 0, GP_MAP_W, GP_MAP_H * GP_LAYERS) mipmapLevel:0
                        withBytes:blk + GP_OFF_MAP bytesPerRow:GP_MAP_W * 2];
        }
    }
    ns = MIN(h.nsprites, (uint32_t)GP_SPRITES);
    ng = MIN(h.nglyphs, (uint32_t)GP_GLYPHS);
    /* the lists copied out now, while the frame is known to be whole:
       the GPU reads them later, when the guest may be writing the next */
    if (ns) {
        sprites = [m.device newBufferWithBytes:blk + GP_OFF_SPRITE length:(NSUInteger)ns * 32
                                       options:MTLResourceStorageModeShared];
    }
    if (ng) {
        glyphs = [m.device newBufferWithBytes:blk + GP_OFF_GLYPH length:(NSUInteger)ng * 32
                                      options:MTLResourceStorageModeShared];
    }
    /* The frame is read again AFTER everything is copied: if the guest
       began a new one meanwhile, what was taken is a mix of two and must
       not be shown.  This is the reader's half of the seqlock and the
       only thing that makes the single block safe. */
    memcpy(&after, blk + offsetof(GpHead, frame), sizeof(after));
    if (after != h.frame) {
        return false;
    }
    p->drawn = h.frame;
    p->art = h.art_frame;
    p->art_any = true;

    rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = p->frame;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    enc = [cb renderCommandEncoderWithDescriptor:rp];

    memset(&u, 0, sizeof(u));
    u.box[2] = GP_W, u.box[3] = GP_H;
    u.clip[2] = GP_W, u.clip[3] = GP_H;
    u.target[0] = GP_W, u.target[1] = GP_H;
    u.over = (h.flags & GP_CANVAS_OVER) ? 1u : 0u;
    if (h.scene < GP_COSMOS_SCENES) {
        /* layer 0, opaque and first: the frame's clear as well.  Its time
           is the guest's wall clock, so the sky breathes at the display's
           rate whatever the game's own step */
        float tparm[4] = { isfinite(h.seconds) ? h.seconds : 0.0f, (float)h.scene, 0, 0 };

        [enc setRenderPipelineState:gp.cosmos];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
        [enc setFragmentBytes:tparm length:sizeof(tparm) atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    }
    [enc setRenderPipelineState:gp.canvas];
    [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
    [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
    [enc setFragmentTexture:p->idx atIndex:0];
    [enc setFragmentTexture:p->pal_global atIndex:1];
    [enc setFragmentTexture:p->pal_line atIndex:2];
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    if (tiled) {
        [enc setRenderPipelineState:gp.tiles];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
        [enc setFragmentBytes:layers length:sizeof(layers) atIndex:1];
        [enc setFragmentTexture:p->map atIndex:0];
        [enc setFragmentTexture:p->tiles atIndex:1];
        [enc setFragmentTexture:p->pal_global atIndex:2];
        [enc setFragmentTexture:p->pal_line atIndex:3];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    }
    if (sprites) {
        [enc setRenderPipelineState:gp.spr];
        [enc setVertexBuffer:sprites offset:0 atIndex:0];
        [enc setFragmentTexture:p->atlas atIndex:0];
        [enc setFragmentTexture:p->spal atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
              instanceCount:ns];
    }
    if (glyphs) {
        [enc setRenderPipelineState:gp.txt];
        [enc setVertexBuffer:glyphs offset:0 atIndex:0];
        [enc setFragmentTexture:p->font atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
              instanceCount:ng];
    }
    [enc endEncoding];
    return true;
}

/* The game panes on the screen, each whole inside the region.  calls f
 * for each; a block that does not fit is passed over. */
static void gp_each(void (^f)(const MetalIcon *ic, const GpData *d))
{
    const MetalSurfacesView *v = &surf.view;

    if (!surf.on || !v->nicons || !v->region) {
        return;
    }
    for (uint32_t i = 0; i < v->nicons; i++) {
        const MetalIcon *ic = &v->icon[i];
        GpData d;

        if (ic->kind != ROSGD_SURF_GAMEPANE || !ic->nrects) {
            continue;
        }
        memcpy(&d, ic->data, sizeof(d));
        if (!d.size || (uint64_t)d.offset + d.size > v->region_size) {
            continue;
        }
        f(ic, &d);
    }
}

/* Before the frame's main pass: each visible pane's own picture made, in
 * a pass of its own.  Returns whether there is a picture to show. */
static bool surf_gamepane_render(id<MTLCommandBuffer> cb)
{
    __block bool any = false;

    gp.tick++;
    gp_each(^(const MetalIcon *ic, const GpData *d) {
        (void)ic;
        gp_setup();
        if (!gp.canvas) {
            return;
        }
        GpPane *p = gp_pane_for(d->offset);
        p->seen = gp.tick;
        gp_render(cb, p, (const uint8_t *)surf.view.region + d->offset, d->size);
        any = any || p->drawn != 0;
    });
    return any;
}

/* Each pane's picture, drawn where the Wimp says it goes and only where
 * it can be seen. */
static void surf_gamepanes(id<MTLRenderCommandEncoder> enc, CGSize target,
                           uint32_t xres, uint32_t yres)
{
    const MetalSurfacesView *v = &surf.view;
    double sx = target.width / MAX(xres, 1u), sy = target.height / MAX(yres, 1u);

    if (!gp.canvas) {
        return;
    }
    gp_each(^(const MetalIcon *ic, const GpData *d) {
        GpPane *p = NULL;
        GpU u;

        for (int i = 0; i < GP_PANES; i++) {
            if (gp.pane[i].used && gp.pane[i].offset == d->offset) {
                p = &gp.pane[i];
            }
        }
        if (!p || !p->drawn) {
            return;
        }
        memset(&u, 0, sizeof(u));
        u.box[0] = ic->x * sx, u.box[1] = ic->y * sy;
        u.box[2] = ic->w * sx, u.box[3] = ic->h * sy;
        u.target[0] = target.width, u.target[1] = target.height;
        [enc setRenderPipelineState:gp.show];
        [enc setFragmentTexture:p->frame atIndex:0];
        for (uint32_t r = 0; r < ic->nrects; r++) {
            const int16_t *q = v->rect[ic->rects + r];

            u.clip[0] = floor(q[0] * sx), u.clip[1] = floor(q[1] * sy);
            u.clip[2] = ceil(q[2] * sx), u.clip[3] = ceil(q[3] * sy);
            if (u.clip[2] <= u.clip[0] || u.clip[3] <= u.clip[1]) {
                continue;
            }
            [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        }
    });
}

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

        if (ic->kind != 1 && ic->kind != 2) {
            continue;                   /* the clock and the meter only: a game pane has its own pass */
        }
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

/* ---- a tile map (ROSGD_SURF_TILES) ---------------------------------------
 * !Micropolis's city, drawn by the window from the tiles themselves: the
 * box writes a map of tile numbers, a list of sprites and, once, the two
 * pictures they are cut from into a backing in the region
 * (rosgd/include/rosgd/tiles.h, which this mirrors); here a tile is looked
 * up in the map texture and its pixel read from the atlas, at whatever
 * size the icon is on the screen.  The icon is the whole map and the Wimp
 * scrolls it, so a zoom or a scroll costs the box nothing at all. */

#define TL_MAGIC    0x53454C54u
#define TL_VERSION  4u
#define TL_MAP      128
#define TL_SPRITES  256
#define TL_CLASSES  1024
#define TL_SPR_W    2048
#define TL_SPR_H    1024
#define TL_ATLAS_MAX_W 1024
#define TL_ATLAS_MAX_H 4096
#define TL_Q_RICH   2
typedef struct TlHead {
    uint32_t magic, version, frame, art;
    uint16_t map_w, map_h, tile, atlas_cols;
    uint32_t ntiles, nsprites;
    int16_t outline[4];
    uint32_t outline_rgba, back_rgba;
    uint16_t atlas_w, atlas_h;      /* the tiles' picture, after the fixed part */
    uint8_t quality, daylight, rate, overlay;
    uint32_t reserved[2];
} TlHead;
#define TL_OFF_MAP      sizeof(TlHead)
#define TL_OFF_SPRITE   (TL_OFF_MAP + (size_t)TL_MAP * TL_MAP * 2)
#define TL_OFF_OVL      (TL_OFF_SPRITE + (size_t)TL_SPRITES * 16)
#define TL_OFF_CLS      (TL_OFF_OVL + (size_t)TL_MAP * TL_MAP * 4)
#define TL_OFF_SPR      (TL_OFF_CLS + (size_t)TL_CLASSES * 4)
#define TL_OFF_ATLAS    (TL_OFF_SPR + (size_t)TL_SPR_W * TL_SPR_H * 4)
#define TL_SIZE(w, h)   (TL_OFF_ATLAS + (uint64_t)(w) * (h) * 5)
#define TL_LIVE         TL_OFF_CLS      /* what changes with the map */
_Static_assert(sizeof(TlHead) == 64, "a tile map's head is 64 bytes");

/* its 32 bytes in the surface entry */
typedef struct TlData {
    uint32_t backing, offset, size, frame;
    uint8_t reserved[16];
} TlData;

static NSString * const TL_SRC = @""
"/* tl_map_hd -- the tile map, drawn larger and richer than its art (Metal).\n"
" *\n"
" * A drop-in for the TlU struct, tl_vs and tl_map in metal.m's TL_SRC; the\n"
" * sprites' functions (tl_spr_vs, tl_spr) stay as they are.  TlU gains four\n"
" * fields at its end, which the C side's TlU must gain too:\n"
" *\n"
" *     int32_t quality;     0 the original, 1 xBR-lite, 2 xBR-lite + enrichment,\n"
" *                          3 sharp bilinear + enrichment, 4 EPX/MMPX-style + enrichment\n"
" *     float   time;        seconds, for the water (wrap it every hour or so)\n"
" *     float   daylight;    1 day ... 0 night\n"
" *     int32_t ovl;         1: the overlay over the map (texture 4)\n"
" *\n"
" * tl_map takes the material atlas as texture(3): R8Uint, the atlas's size,\n"
" * each texel's material | lamp << 4, made once when the art changes\n"
" * (tl_matlas.c, or emu.material_atlas); and the class table as texture(2): R32Uint, 1024 x 1, classes.py's\n"
" * c_array(): class | size << 8 | (lx | ly << 4) << 16 | height/64 << 24.\n"
" *\n"
" * The same maths as tl_map_hd.glsl and emu.py, function for function; Metal\n"
" * has no mutable globals, so the hood and the window live in an Hd.\n"
" */\n"
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct TlU {\n"
"    float4 box;          /* the whole map on the target, target pixels */\n"
"    float4 clip;         /* the visible piece being drawn */\n"
"    float2 target;\n"
"    int    ntiles, pad;\n"
"    int4   geo;          /* the map's width and height in tiles, a tile's edge,\n"
"                            tiles across the atlas */\n"
"    int4   outline;      /* a frame round tiles x0 y0 to x1 y1 */\n"
"    float4 ocol;\n"
"    int    quality;\n"
"    float  time;\n"
"    float  daylight;\n"
"    int    ovl;          /* 1: the overlay over the map */\n"
"};\n"
"struct TlV { float4 pos [[position]]; };\n"
"\n"
"vertex TlV tl_vs(uint id [[vertex_id]], constant TlU &U [[buffer(0)]])\n"
"{\n"
"    float2 c = float2(id & 1, (id >> 1) & 1);\n"
"    float2 p = mix(U.clip.xy, U.clip.zw, c);\n"
"    TlV o;\n"
"    o.pos = float4(p / U.target * float2(2, -2) + float2(-1, 1), 0, 1);\n"
"    return o;\n"
"}\n"
"\n"
"constant uint C_DIRT = 0, C_WATER = 1, C_SHORE = 2, C_TREE = 3, C_FLOOD = 5;\n"
"constant uint C_ROAD = 8, C_BRIDGE = 9, C_POWER = 10, C_RAIL = 11, C_LOT = 12, C_HOUSE = 13, C_BUILD = 14;\n"
"constant int M_GROUND = 0, M_SAND = 1, M_WATER = 2, M_TREE = 3, M_TREED = 4, M_GRASS = 5, M_ROAD = 6,\n"
"             M_STRUCT = 7, M_OTHER = 8;\n"
"\n"
"constant float3 EARTH_A = float3(0.70, 0.45, 0.34), EARTH_B = float3(0.84, 0.58, 0.41);\n"
"constant float3 SAND_A = float3(0.80, 0.68, 0.46), SAND_B = float3(0.90, 0.80, 0.58);\n"
"constant float3 GRASS_A = float3(0.26, 0.56, 0.20), GRASS_B = float3(0.46, 0.72, 0.28);\n"
"constant float3 ASPH = float3(0.25, 0.26, 0.28);\n"
"constant float3 TREE_DARK = float3(0.07, 0.24, 0.10), TREE_LIGHT = float3(0.50, 0.76, 0.26);\n"
"constant float3 SHALLOW = float3(0.33, 0.62, 0.74), DEEP = float3(0.10, 0.25, 0.52);\n"
"constant float3 FOAM = float3(0.88, 0.94, 0.96);\n"
"constant float3 NIGHT = float3(0.16, 0.20, 0.36), LAMP = float3(1.00, 0.82, 0.45);\n"
"constant float3 COOL = float3(0.90, 0.95, 1.08), WARM = float3(1.06, 1.00, 0.92);\n"
"constant float3 L3 = float3(-1.0, -1.0, 1.4) / 1.9899749;     /* towards the light */\n"
"constant float EQ = 0.5;\n"
"\n"
"struct Hd {\n"
"    int4 geo;\n"
"    int ntiles;\n"
"    int2 T0;\n"
"    uint TA[9];         /* each tile's origin in the atlas: x | y << 16 */\n"
"    uint TC[9];         /* its class word: class | size << 8 | lx|ly << 16 | height << 24 */\n"
"    int2 BASE, CI, S;\n"
"    uint WRGB[16];      /* the window's texels, packed unorm8 */\n"
"    uint WCL[16];       /* their class words, with the lamp bit at 11 and the material at 12-15 */\n"
"};\n"
"static uint4 cword(uint w) { return uint4(w & 255u, (w >> 8) & 7u, (w >> 16) & 255u, w >> 24); }\n"
"static int wmat(thread const Hd &h, int i) { return int((h.WCL[i] >> 12) & 15u); }\n"
"static float3 wrgb(thread const Hd &h, int i) { return unpack_unorm4x8_to_float(h.WRGB[i]).rgb; }\n"
"\n"
"static void hood(thread Hd &h, float2 m, texture2d<ushort> map, texture2d<uint> cls)\n"
"{\n"
"    h.T0 = int2(floor(m / float(h.geo.z)));\n"
"    for (int k = 0; k < 9; k++) {\n"
"        int2 t = clamp(h.T0 + int2(k % 3 - 1, k / 3 - 1), int2(0), h.geo.xy - 1);\n"
"        int n = int(map.read(uint2(t)).r);\n"
"        if (n >= h.ntiles) n = 0;\n"
"        h.TA[k] = uint(n % h.geo.w * h.geo.z) | uint(n / h.geo.w * h.geo.z) << 16;\n"
"        h.TC[k] = cls.read(uint2(n, 0)).r;\n"
"    }\n"
"}\n"
"\n"
"/* any map pixel near the fragment -> its atlas texel, through the map */\n"
"static float4 px(thread Hd &h, int2 p, thread uint &c, thread uint &mt, bool want,\n"
"                  texture2d<float> atlas, texture2d<uint> mats)\n"
"{\n"
"    p = clamp(p, int2(0), h.geo.xy * h.geo.z - 1);\n"
"    int2 t = p / h.geo.z;\n"
"    int k = (t.y - h.T0.y + 1) * 3 + (t.x - h.T0.x + 1);\n"
"    c = h.TC[k];\n"
"    uint2 a = uint2(int2(h.TA[k] & 0xffffu, h.TA[k] >> 16) + p - t * h.geo.z);\n"
"    mt = want ? mats.read(a).r : 0u;\n"
"    return atlas.read(a);\n"
"}\n"
"\n"
"static uint lowbias32(uint x)\n"
"{\n"
"    x ^= x >> 16; x *= 0x7feb352du;\n"
"    x ^= x >> 15; x *= 0x846ca68bu;\n"
"    x ^= x >> 16;\n"
"    return x;\n"
"}\n"
"\n"
"static float hash2(int ix, int iy, uint seed)\n"
"{\n"
"    uint h = lowbias32(uint(ix) + lowbias32(uint(iy) + seed));\n"
"    return float(h >> 8) * (1.0 / 16777216.0);\n"
"}\n"
"\n"
"static float vnoise(float2 p, uint seed)\n"
"{\n"
"    float2 i = floor(p), f = p - i;\n"
"    float2 u = f * f * (3.0 - 2.0 * f);\n"
"    int2 k = int2(i);\n"
"    float a = hash2(k.x, k.y, seed), b = hash2(k.x + 1, k.y, seed);\n"
"    float c = hash2(k.x, k.y + 1, seed), d = hash2(k.x + 1, k.y + 1, seed);\n"
"    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);\n"
"}\n"
"\n"
"static float yuv_dist(float3 a, float3 b)\n"
"{\n"
"    float3 d = a - b;\n"
"    float y = dot(d, float3(0.299, 0.587, 0.114));\n"
"    float u = dot(d, float3(-0.169, -0.331, 0.5));\n"
"    float v = dot(d, float3(0.5, -0.419, -0.081));\n"
"    return 48.0 * abs(y) + 7.0 * abs(u) + 6.0 * abs(v);\n"
"}\n"
"\n"
"static float3 tent_field(thread const float *v, float2 t)\n"
"{\n"
"    float b[4];\n"
"    for (int q = 0; q < 4; q++) {\n"
"        int i = 1 + (q & 1), j = 1 + (q >> 1);\n"
"        float s = 0.0;\n"
"        for (int di = -1; di <= 1; di++)\n"
"            for (int dj = -1; dj <= 1; dj++)\n"
"                s += v[(i + di) * 4 + (j + dj)] * ((di == 0 ? 0.5 : 0.25) * (dj == 0 ? 0.5 : 0.25));\n"
"        b[q] = s;\n"
"    }\n"
"    float top = mix(b[0], b[1], t.x), bot = mix(b[2], b[3], t.x);\n"
"    return float3(mix(top, bot, t.y), mix(b[1] - b[0], b[3] - b[2], t.y), bot - top);\n"
"}\n"
"\n"
"static int widx(thread const Hd &h, int dx, int dy) { return (h.CI.x + h.S.x * dx) * 4 + (h.CI.y + h.S.y * dy); }\n"
"static float3 P(thread const Hd &h, int dx, int dy) { return wrgb(h, widx(h, dx, dy)); }\n"
"\n"
"static float rdist(float2 pt, float2 r0, float t)\n"
"{\n"
"    float2 d = max(max(r0 - pt, pt - (r0 + t)), float2(0));\n"
"    return length(d);\n"
"}\n"
"\n"
"static float3 shade(thread Hd &h, int idx, float2 m, float n_low, float n_mid, float n_fine, float sF, float2 sg,\n"
"                    float psh, float shd, float ao, bool own_built, thread float3 &emis, float wgt)\n"
"{\n"
"    float3 rgbx = wrgb(h, idx);\n"
"    int mat = wmat(h, idx);\n"
"    uint4 cl = cword(h.WCL[idx]);\n"
"    if (mat == M_GROUND && own_built && sF > 0.45) mat = M_STRUCT;\n"
"    float size = float(max(cl.g, 1u));\n"
"    float2 tf = m / float(h.geo.z) - floor(m / float(h.geo.z));\n"
"    float2 u = (float2(float(cl.b & 15u), float(cl.b >> 4)) + tf) / size;\n"
"    float3 ground = mix(EARTH_A, EARTH_B, n_low) * (0.92 + 0.16 * n_mid) * (0.96 + 0.08 * n_fine);\n"
"    float3 sand = mix(SAND_A, SAND_B, n_mid) * (0.96 + 0.08 * n_fine);\n"
"    float3 grass = mix(GRASS_A, GRASS_B, n_low) * (0.94 + 0.12 * n_fine);\n"
"    float3 road = ASPH * (0.90 + 0.10 * n_mid + 0.08 * n_fine);\n"
"    float3 n = float3(-sg * 2.2, 1.0);\n"
"    float diff = clamp(dot(n, L3) / length(n), 0.0, 1.0);\n"
"    float lit = 0.62 + 0.55 * diff;\n"
"    float grad = 1.10 - 0.20 * (u.x + u.y) * 0.5;\n"
"    float3 tint = mix(COOL, WARM, smoothstep(0.55, 1.0, diff));\n"
"    float3 col;\n"
"    if (mat == M_STRUCT) col = rgbx * (lit * grad) * tint;\n"
"    else if (mat == M_ROAD) col = road;\n"
"    else if (mat == M_GRASS) col = grass;\n"
"    else if (mat == M_GROUND || mat == M_TREE || mat == M_TREED) col = ground;\n"
"    else if (mat == M_SAND || mat == M_WATER) col = cl.r == C_SHORE ? sand : ground;\n"
"    else col = rgbx * (own_built ? grad : 1.0);\n"
"    if (mat != M_STRUCT) {\n"
"        float occl = clamp(0.55 * psh + 0.5 * shd + 0.28 * ao +\n"
"                           0.35 * smoothstep(0.15, 0.7, sF) * (own_built ? 1.0 : 0.0), 0.0, 0.75);\n"
"        col *= 1.0 - occl;\n"
"    }\n"
"    if (mat == M_STRUCT && (h.WCL[idx] & 0x800u) != 0u) {\n"
"        int2 tp = h.BASE + int2(idx / 4, idx % 4);\n"
"        if (hash2(tp.x, tp.y, 7u) > 0.4) emis += LAMP * wgt;\n"
"    }\n"
"    return col;\n"
"}\n"
"\n"
"fragment float4 tl_map(TlV v [[stage_in]], constant TlU &U [[buffer(0)]],\n"
"                       texture2d<ushort> map [[texture(0)]],\n"
"                       texture2d<float> atlas [[texture(1)]],\n"
"                       texture2d<uint> cls [[texture(2)]],\n"
"                       texture2d<uint> mats [[texture(3)]],\n"
"                       texture2d<float> ovl [[texture(4)]])\n"
"{\n"
"    Hd h;\n"
"    h.geo = U.geo;\n"
"    h.ntiles = U.ntiles;\n"
"    float2 size = float2(U.geo.xy * U.geo.z);\n"
"    float2 m = (v.pos.xy - U.box.xy) / U.box.zw * size;      /* the map's pixels */\n"
"    float zoom = U.box.z / size.x;                           /* target pixels a map pixel */\n"
"    int2 t = int2(floor(m / float(U.geo.z)));\n"
"    if (t.x < 0 || t.y < 0 || t.x >= U.geo.x || t.y >= U.geo.y) {\n"
"        discard_fragment();\n"
"    }\n"
"    float4 colour;\n"
"    if (U.quality <= 0) {\n"
"        int n = int(map.read(uint2(t)).r);\n"
"        if (n >= U.ntiles) {\n"
"            n = 0;\n"
"        }\n"
"        int2 a = int2(n % U.geo.w, n / U.geo.w) * U.geo.z\n"
"                 + clamp(int2(floor(m)) - t * U.geo.z, int2(0), int2(U.geo.z - 1));\n"
"        colour = atlas.read(uint2(a));\n"
"    } else {\n"
"        hood(h, m, map, cls);\n"
"        int2 c = int2(floor(m));\n"
"        h.BASE = int2(floor(m - 0.5)) - 1;\n"
"        for (int i = 0; i < 4; i++)\n"
"            for (int j = 0; j < 4; j++) {\n"
"                uint cl, mt;\n"
"                float4 a = px(h, h.BASE + int2(i, j), cl, mt, U.quality != 1, atlas, mats);\n"
"                h.WRGB[i * 4 + j] = pack_float_to_unorm4x8(a);\n"
"                h.WCL[i * 4 + j] = (cl & 0xffff07ffu) | (mt & 15u) << 12 | ((mt >> 4) & 1u) << 11;\n"
"            }\n"
"        float2 f = m - floor(m);\n"
"        float2 tt = (m - 0.5) - floor(m - 0.5);\n"
"        bool2 hi = f >= float2(0.5);\n"
"        h.S = int2(hi.x ? 1 : -1, hi.y ? 1 : -1);\n"
"        h.CI = int2(hi.x ? 1 : 2, hi.y ? 1 : 2);\n"
"        float2 fp = float2(hi.x ? f.x : 1.0 - f.x, hi.y ? f.y : 1.0 - f.y);\n"
"\n"
"        int pi[4] = {0, 0, 0, 0};\n"
"        float pw[4] = {0, 0, 0, 0};\n"
"        int np = 1;\n"
"        float3 E = P(h, 0, 0), Fx = P(h, 1, 0), H = P(h, 0, 1), I = P(h, 1, 1);\n"
"        if (U.quality == 1 || U.quality == 2) {\n"
"            float3 C = P(h, 1, -1), G = P(h, -1, 1);\n"
"            float3 F4 = P(h, 2, 0), H5 = P(h, 0, 2), I4 = P(h, 2, 1), I5 = P(h, 1, 2), C4 = P(h, 2, -1), G5 = P(h, -1, 2);\n"
"            float d1 = yuv_dist(E, C) + yuv_dist(E, G) + yuv_dist(I, F4) + yuv_dist(I, H5) + 4.0 * yuv_dist(H, Fx);\n"
"            float d2 = yuv_dist(H, G5) + yuv_dist(H, I5) + yuv_dist(Fx, I4) + yuv_dist(Fx, C4) + 4.0 * yuv_dist(E, I);\n"
"            bool edge = d1 < d2 && yuv_dist(E, Fx) > EQ && yuv_dist(E, H) > EQ;\n"
"            int newi = yuv_dist(E, Fx) <= yuv_dist(E, H) ? widx(h, 1, 0) : widx(h, 0, 1);\n"
"            float cov = clamp((fp.x + fp.y - 1.5) * zoom * 0.70710678 + 0.5, 0.0, 1.0);\n"
"            if (!edge) cov = 0.0;\n"
"            pi[0] = widx(h, 0, 0); pw[0] = 1.0 - cov;\n"
"            pi[1] = newi; pw[1] = cov;\n"
"            np = 2;\n"
"        } else if (U.quality == 3) {\n"
"            float2 fs = clamp((tt - 0.5) * zoom + 0.5, 0.0, 1.0);\n"
"            pi[0] = 1 * 4 + 1; pw[0] = (1.0 - fs.x) * (1.0 - fs.y);\n"
"            pi[1] = 2 * 4 + 1; pw[1] = fs.x * (1.0 - fs.y);\n"
"            pi[2] = 1 * 4 + 2; pw[2] = (1.0 - fs.x) * fs.y;\n"
"            pi[3] = 2 * 4 + 2; pw[3] = fs.x * fs.y;\n"
"            np = 4;\n"
"        } else {\n"
"            float3 B = P(h, 0, -1), D = P(h, -1, 0);\n"
"            bool take = yuv_dist(Fx, H) <= EQ && yuv_dist(E, Fx) > EQ && yuv_dist(Fx, B) > EQ &&\n"
"                        yuv_dist(H, D) > EQ && !(yuv_dist(E, I) <= EQ);\n"
"            pi[0] = take ? widx(h, 1, 0) : widx(h, 0, 0); pw[0] = 1.0;\n"
"        }\n"
"\n"
"        float3 col = float3(0);\n"
"        if (U.quality == 1) {\n"
"            for (int k = 0; k < np; k++) col += wrgb(h, pi[k]) * pw[k];\n"
"        } else {\n"
"            float wv[16], tv[16], sv[16];\n"
"            for (int k = 0; k < 16; k++) {\n"
"                int wm = wmat(h, k);\n"
"                wv[k] = wm == M_WATER ? 1.0 : 0.0;\n"
"                tv[k] = wm == M_TREE ? 1.0 : wm == M_TREED ? 0.55 : 0.0;\n"
"                sv[k] = wm == M_STRUCT ? 1.0 : 0.0;\n"
"            }\n"
"            float wF = tent_field(wv, tt).x;\n"
"            float3 tfl = tent_field(tv, tt), sfl = tent_field(sv, tt);\n"
"            float tF = tfl.x, sF = sfl.x;\n"
"\n"
"            float2 mn = m * (16.0 / float(h.geo.z));\n"
"            int st = max(h.geo.z / 16, 1);       /* texels a 16-pixel texel */\n"
    "            float n_low = vnoise(mn * (1.0 / 37.0), 11u) * 0.65 + vnoise(mn * (1.0 / 13.0), 12u) * 0.35;\n"
"            float n_mid = vnoise(mn * (1.0 / 4.5), 13u);\n"
"            float n_fine = vnoise(mn * 1.3, 14u);\n"
"\n"
"            float tile = float(U.geo.z);\n"
"            float2 q = m / tile - 0.5;\n"
"            int2 qi = int2(floor(q));\n"
"            float2 qf = q - floor(q);\n"
"            float dv[4];\n"
"            for (int k = 0; k < 4; k++) {\n"
"                int2 o = qi + int2(k & 1, k >> 1) - h.T0 + 1;\n"
"                uint cl = h.TC[o.y * 3 + o.x] & 255u;\n"
"                dv[k] = (cl == C_WATER || cl == C_BRIDGE) ? 1.0 : cl == C_SHORE ? 0.35 : 0.0;\n"
"            }\n"
"            float depth = mix(mix(dv[0], dv[1], qf.x), mix(dv[2], dv[3], qf.x), qf.y);\n"
"            uint own = h.TC[4] & 255u;\n"
"            bool own_built = own == C_HOUSE || own == C_BUILD;\n"
"            /* shadow and occlusion fall only on what is not a structure */\n"
"            bool need = false;\n"
"            for (int k = 0; k < np; k++) need = need || wmat(h, pi[k]) != M_STRUCT;\n"
"            float ao = 0.0, shd = 0.0, psh = 0.0;\n"
"            for (int k = 0; k < 9 && need; k++) {\n"
"                if (k == 4) continue;\n"
"                uint cl = h.TC[k] & 255u;\n"
"                if (!((cl == C_HOUSE || cl == C_BUILD) && !own_built)) continue;\n"
"                float hh = float(h.TC[k] >> 24) / 64.0;\n"
"                float2 r0 = float2(h.T0 + int2(k % 3 - 1, k / 3 - 1)) * tile;\n"
"                float a_k = 1.0 - smoothstep(0.0, 0.30 * tile, rdist(m, r0, tile));\n"
"                float s_k = 1.0 - smoothstep(0.0, 0.22 * tile, rdist(m - float2(hh * tile * 0.55), r0, tile));\n"
"                ao = max(ao, a_k);\n"
"                shd = max(shd, s_k * clamp(hh * 1.6, 0.0, 1.0));\n"
"            }\n"
"\n"
"            uint pc, pt;\n"
"            px(h, c - st, pc, pt, true, atlas, mats);\n"
"            int pm = int(pt & 15u);\n"
"            for (int dk = 1; dk <= 5 && need; dk++) {\n"
"                uint bc, bt;\n"
"                px(h, c - (dk + 1) * st, bc, bt, true, atlas, mats);\n"
"                int bm = int(bt & 15u);\n"
"                float hpx = float(pc >> 24) / 64.0 * 16.0 * 0.9;\n"
"                bool caster = (pm == M_STRUCT || pm == M_TREE) && (bm == M_STRUCT || bm == M_TREE);\n"
"                if (caster && hpx >= float(dk)) psh = max(psh, 1.0 - float(dk) / 8.0);\n"
"                pc = bc; pm = bm;\n"
"            }\n"
"\n"
"            float3 emis = float3(0);\n"
"            for (int k = 0; k < np; k++)\n"
"                col += shade(h, pi[k], m, n_low, n_mid, n_fine, sF, sfl.yz, psh, shd, ao, own_built, emis, pw[k]) * pw[k];\n"
"            int nm = wmat(h, widx(h, 0, 0));\n"
"            bool soft_ok = nm == M_GROUND || nm == M_TREE || nm == M_TREED || nm == M_GRASS || nm == M_SAND ||\n"
"                           nm == M_WATER;\n"
"\n"
"            float ct = smoothstep(0.22, 0.42, tF) * (soft_ok ? 1.0 : 0.0);\n"
"            if (ct > 0.0) {\n"
"            float crown = vnoise(mn * 0.42, 21u);\n"
"            float cgx = (vnoise(float2(mn.x + 0.5, mn.y) * 0.42, 21u) - crown) * 2.0;\n"
"            float cgy = (vnoise(float2(mn.x, mn.y + 0.5) * 0.42, 21u) - crown) * 2.0;\n"
"            float3 tn = float3(-(tfl.y * 1.8 + cgx * 0.9), -(tfl.z * 1.8 + cgy * 0.9), 1.0);\n"
"            float tdiff = clamp(dot(tn, L3) / length(tn), 0.0, 1.0);\n"
"            float tl = clamp(tdiff * (0.45 + 0.55 * clamp(tF * 1.3, 0.0, 1.0)) + (crown - 0.5) * 0.25, 0.0, 1.0);\n"
"            float3 canopy = mix(TREE_DARK, TREE_LIGHT, tl);\n"
"            canopy = mix(canopy, canopy * float3(1.15, 1.0, 0.55), n_low * 0.45);\n"
"            col = mix(col, canopy, ct);\n"
"            }\n"
"\n"
"            float cw = smoothstep(0.40, 0.52, wF) * (soft_ok ? 1.0 : 0.0);\n"
"            if (cw > 0.0) {\n"
"            float wa = vnoise(mn * 0.09 + U.time * float2(0.30, 0.11), 31u);\n"
"            float2 r = float2(mn.x * 0.8 - mn.y * 0.6, mn.x * 0.6 + mn.y * 0.8);\n"
"            float wb = vnoise(r * 0.33 + U.time * float2(-0.21, 0.47), 32u);\n"
"            float wgx = (vnoise((r + float2(0.6, 0.0)) * 0.33 + U.time * float2(-0.21, 0.47), 32u) - wb) * 1.6;\n"
"            float wgy = (vnoise((r + float2(0.0, 0.6)) * 0.33 + U.time * float2(-0.21, 0.47), 32u) - wb) * 1.6;\n"
"            float deep = depth * smoothstep(0.55, 1.0, wF);\n"
"            float3 water = mix(SHALLOW, DEEP, deep) * (0.90 + 0.14 * wa + 0.08 * wb);\n"
"            float3 wn = normalize(float3(-wgx * 1.5, -wgy * 1.5, 1.0));\n"
"            float3 rv = 2.0 * wn.z * wn - float3(0, 0, 1);\n"
"            float spec = pow(clamp(dot(rv, L3), 0.0, 1.0), 16.0) * 0.22;\n"
"            water += spec * U.daylight;\n"
"            float foam = (1.0 - smoothstep(0.50, 0.78, wF)) * (0.55 + 0.45 * wb);\n"
"            water = mix(water, FOAM, foam * 0.55);\n"
"            water *= 1.0 - 0.30 * psh - 0.18 * shd;\n"
"            col = mix(col, water, cw);\n"
"            }\n"
"\n"
"            col = col * mix(NIGHT, float3(1), U.daylight) + emis * (1.0 - U.daylight);\n"
"        }\n"
"        colour = float4(clamp(col, 0.0, 1.0), 1.0);\n"
"    }\n"
"    if (U.ovl != 0) {\n"
"        /* the statistics' colours over the map, smoothly from square to square */\n"
"        constexpr sampler lin(filter::linear, address::clamp_to_edge);\n"
"        float4 o = ovl.sample(lin, m / float(U.geo.z) / 128.0);\n"
"        colour = float4(o.rgb + colour.rgb * (1.0 - o.a), 1.0);\n"
"    }\n"
"    if (U.outline.z > U.outline.x && t.x >= U.outline.x && t.y >= U.outline.y\n"
"        && t.x < U.outline.z && t.y < U.outline.w) {\n"
"        float2 k = U.box.zw / size;\n"
"        float2 lo = (m - float2(U.outline.xy * U.geo.z)) * k;\n"
"        float2 hi2 = (float2(U.outline.zw * U.geo.z) - m) * k;\n"
"        if (min(min(lo.x, lo.y), min(hi2.x, hi2.y)) < 2.0) {\n"
"            colour = U.ocol;\n"
"        }\n"
"    }\n"
"    return colour;\n"
"}\n"
"\n"
"/* the sprites */\n"
"struct TlS { short4 at; ushort4 cell; };\n"
"struct TlSV { float4 pos [[position]]; float2 uv; uint4 cell [[flat]]; };\n"
"/* a sprite: its box in the map, cut to the visible piece, as a strip */\n"
"vertex TlSV tl_spr_vs(uint id [[vertex_id]], uint inst [[instance_id]],\n"
"                      constant TlU &U [[buffer(0)]], constant TlS *S [[buffer(1)]])\n"
"{\n"
"    TlS s = S[inst];\n"
"    float k = U.box.z / float(U.geo.x * U.geo.z);\n"
"    float4 b = float4(U.box.xy + float2(s.at.xy) * k, U.box.xy + float2(s.at.xy + s.at.zw) * k);\n"
"    float4 q = float4(max(b.xy, U.clip.xy), min(b.zw, U.clip.zw));\n"
"    if (q.z <= q.x || q.w <= q.y) {\n"
"        q = float4(0);\n"
"    }\n"
"    float2 c = float2(id & 1, (id >> 1) & 1);\n"
"    float2 p = mix(q.xy, q.zw, c);\n"
"    TlSV o;\n"
"    o.pos = float4(p / U.target * float2(2, -2) + float2(-1, 1), 0, 1);\n"
"    o.uv = (p - b.xy) / max(b.zw - b.xy, float2(1e-3));\n"
"    o.cell = uint4(s.cell);\n"
"    return o;\n"
"}\n"
"fragment float4 tl_spr(TlSV v [[stage_in]], texture2d<float> spr [[texture(0)]])\n"
"{\n"
"    int2 a = int2(v.cell.xy) + clamp(int2(floor(v.uv * float2(v.cell.zw))), int2(0), int2(v.cell.zw) - 1);\n"
"    return spr.read(uint2(a));\n"
"}\n";

typedef struct TlU {
    float box[4], clip[4], target[2];
    int32_t ntiles, pad;
    int32_t geo[4], outline[4];
    float ocol[4];
    int32_t quality;            /* the rich drawing (tl_map_hd.metal) */
    float time, daylight;
    int32_t ovl;
} TlU;

static struct {
    id<MTLRenderPipelineState> map, spr;            /* into the window's BGRA */
    id<MTLRenderPipelineState> map_shot, spr_shot;  /* into a screenshot's RGBA */
    id<MTLTexture> mapt, atlas, sprt, clst, matt, ovlt;
    uint32_t frame, art;
    int atlas_w, atlas_h;           /* the atlas and materials textures' size */
    bool have, tried;
    uint8_t live[TL_LIVE];          /* the last whole frame's head, map and sprites */
} tl;

static id<MTLRenderPipelineState> tl_pipe(id<MTLLibrary> lib, NSString *vs, NSString *fs,
                                          MTLPixelFormat f)
{
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    NSError *err = nil;
    id<MTLRenderPipelineState> p;

    pd.vertexFunction = [lib newFunctionWithName:vs];
    pd.fragmentFunction = [lib newFunctionWithName:fs];
    pd.colorAttachments[0].pixelFormat = f;
    /* premultiplied, as the box's pictures are */
    pd.colorAttachments[0].blendingEnabled = YES;
    pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    p = [m.device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!p) {
        metal_log("tile map %s: %s", [fs UTF8String], [[err localizedDescription] UTF8String]);
    }
    return p;
}

static void tl_setup(void)
{
    NSError *err = nil;
    id<MTLLibrary> lib;

    if (tl.tried) {
        return;
    }
    tl.tried = true;
    lib = [m.device newLibraryWithSource:TL_SRC options:nil error:&err];
    if (!lib) {
        metal_log("tile map: %s", [[err localizedDescription] UTF8String]);
        return;
    }
    tl.map = tl_pipe(lib, @"tl_vs", @"tl_map", MTLPixelFormatBGRA8Unorm);
    tl.spr = tl_pipe(lib, @"tl_spr_vs", @"tl_spr", MTLPixelFormatBGRA8Unorm);
    tl.map_shot = tl_pipe(lib, @"tl_vs", @"tl_map", MTLPixelFormatRGBA8Unorm);
    tl.spr_shot = tl_pipe(lib, @"tl_spr_vs", @"tl_spr", MTLPixelFormatRGBA8Unorm);
    tl.mapt = gp_texture(MTLPixelFormatR16Uint, TL_MAP, TL_MAP, MTLTextureUsageShaderRead);
    tl.sprt = gp_texture(MTLPixelFormatRGBA8Unorm, TL_SPR_W, TL_SPR_H, MTLTextureUsageShaderRead);
    tl.clst = gp_texture(MTLPixelFormatR32Uint, TL_CLASSES, 1, MTLTextureUsageShaderRead);
    tl.ovlt = gp_texture(MTLPixelFormatRGBA8Unorm, TL_MAP, TL_MAP, MTLTextureUsageShaderRead);
}

/* The first visible tile map, and its block, whole inside the region */
static const MetalIcon *tl_first(TlData *d)
{
    const MetalSurfacesView *v = &surf.view;

    if (!surf.on || !v->nicons || !v->region) {
        return NULL;
    }
    for (uint32_t i = 0; i < v->nicons; i++) {
        const MetalIcon *ic = &v->icon[i];

        if (ic->kind != ROSGD_SURF_TILES || !ic->nrects) {
            continue;
        }
        memcpy(d, ic->data, sizeof(*d));
        if (d->size < TL_OFF_ATLAS || d->offset == 0xFFFFFFFFu
            || (uint64_t)d->offset + d->size > v->region_size) {
            continue;
        }
        return ic;
    }
    return NULL;
}

/* A frame of the block into the textures, if it is whole and new; whether
 * there is a frame to draw (the last one, if this one is being written) */
static bool tl_take(const uint8_t *blk, uint32_t size)
{
    TlHead h;
    uint32_t after;

    memcpy(&h, blk, sizeof(h));
    if (h.magic != TL_MAGIC || h.version != TL_VERSION || (h.frame & 1u)) {
        return tl.have;
    }
    if (tl.have && h.frame == tl.frame) {
        return true;
    }
    /* copied out, and read again AFTER: if the box began a new frame
       meanwhile the copy is a mix of two, and the last whole one stands */
    static uint8_t next[TL_LIVE];
    memcpy(next, blk, sizeof(next));
    memcpy(&after, blk + offsetof(TlHead, frame), sizeof(after));
    if (after != h.frame) {
        return tl.have;
    }
    int aw = h.atlas_w, ah = h.atlas_h;
    if (aw < 1 || ah < 1 || aw > TL_ATLAS_MAX_W || ah > TL_ATLAS_MAX_H || TL_SIZE(aw, ah) > size) {
        return tl.have;                 /* a block too short for what it says it holds */
    }
    memcpy(tl.live, next, sizeof(tl.live));
    if (aw != tl.atlas_w || ah != tl.atlas_h) {
        /* the atlas's size is the box's: textures of that size */
        tl.atlas = gp_texture(MTLPixelFormatRGBA8Unorm, aw, ah, MTLTextureUsageShaderRead);
        tl.matt = gp_texture(MTLPixelFormatR8Uint, aw, ah, MTLTextureUsageShaderRead);
        tl.atlas_w = aw, tl.atlas_h = ah;
        tl.have = false;
    }
    if (!tl.atlas || !tl.matt) {
        return false;
    }
    if (!tl.have || h.art != tl.art) {
        /* the pictures, the classes and the materials: once, when the box
           first writes them */
        [tl.atlas replaceRegion:MTLRegionMake2D(0, 0, aw, ah) mipmapLevel:0
                      withBytes:blk + TL_OFF_ATLAS bytesPerRow:(NSUInteger)aw * 4];
        [tl.matt replaceRegion:MTLRegionMake2D(0, 0, aw, ah) mipmapLevel:0
                     withBytes:blk + TL_OFF_ATLAS + (size_t)aw * ah * 4 bytesPerRow:(NSUInteger)aw];
        [tl.sprt replaceRegion:MTLRegionMake2D(0, 0, TL_SPR_W, TL_SPR_H) mipmapLevel:0
                     withBytes:blk + TL_OFF_SPR bytesPerRow:TL_SPR_W * 4];
        [tl.clst replaceRegion:MTLRegionMake2D(0, 0, TL_CLASSES, 1) mipmapLevel:0
                     withBytes:blk + TL_OFF_CLS bytesPerRow:TL_CLASSES * 4];
        tl.art = h.art;
    }
    [tl.mapt replaceRegion:MTLRegionMake2D(0, 0, TL_MAP, TL_MAP) mipmapLevel:0
                 withBytes:tl.live + TL_OFF_MAP bytesPerRow:TL_MAP * 2];
    if (((const TlHead *)tl.live)->overlay) {
        [tl.ovlt replaceRegion:MTLRegionMake2D(0, 0, TL_MAP, TL_MAP) mipmapLevel:0
                     withBytes:tl.live + TL_OFF_OVL bytesPerRow:TL_MAP * 4];
    }
    tl.frame = h.frame;
    tl.have = true;
    return true;
}

/* The map, then its sprites, where the Wimp says the icon is and only
 * where it can be seen */
static void surf_tiles(id<MTLRenderCommandEncoder> enc, bool shot, CGSize target,
                       uint32_t xres, uint32_t yres)
{
    const MetalSurfacesView *v = &surf.view;
    double sx = target.width / MAX(xres, 1u), sy = target.height / MAX(yres, 1u);
    const MetalIcon *ic;
    const TlHead *h;
    TlData d;
    TlU u;
    uint32_t ns, cap;

    ic = tl_first(&d);
    if (!ic) {
        return;
    }
    tl_setup();
    if (!tl.map || !tl.spr || !tl.map_shot || !tl.spr_shot || !tl.mapt) {
        return;
    }
    if (!tl_take((const uint8_t *)surf.view.region + d.offset, d.size)) {
        return;
    }
    h = (const TlHead *)tl.live;
    if (h->map_w < 1 || h->map_h < 1 || h->map_w > TL_MAP || h->map_h > TL_MAP || h->tile < 1
        || h->atlas_cols < 1 || h->atlas_cols * h->tile > tl.atlas_w) {
        return;
    }
    cap = MIN((uint32_t)(h->atlas_cols * (tl.atlas_h / h->tile)), (uint32_t)TL_CLASSES);
    memset(&u, 0, sizeof(u));
    u.box[0] = ic->x * sx, u.box[1] = ic->y * sy, u.box[2] = ic->w * sx, u.box[3] = ic->h * sy;
    u.target[0] = target.width, u.target[1] = target.height;
    u.ntiles = (int32_t)MIN(h->ntiles, cap);
    u.geo[0] = h->map_w, u.geo[1] = h->map_h, u.geo[2] = h->tile, u.geo[3] = h->atlas_cols;
    for (int k = 0; k < 4; k++) {
        u.outline[k] = h->outline[k];
        u.ocol[k] = ((h->outline_rgba >> (8 * k)) & 255) / 255.0f;
    }
    u.quality = MIN((int32_t)h->quality, TL_Q_RICH);
    u.time = (float)fmod(CFAbsoluteTimeGetCurrent(), 3600.0);
    u.daylight = h->daylight / 255.0f;
    u.ovl = h->overlay ? 1 : 0;
    ns = MIN(h->nsprites, (uint32_t)TL_SPRITES);
    for (uint32_t r = 0; r < ic->nrects; r++) {
        const int16_t *q = v->rect[ic->rects + r];

        u.clip[0] = floor(q[0] * sx), u.clip[1] = floor(q[1] * sy);
        u.clip[2] = ceil(q[2] * sx), u.clip[3] = ceil(q[3] * sy);
        if (u.clip[2] <= u.clip[0] || u.clip[3] <= u.clip[1]) {
            continue;
        }
        [enc setRenderPipelineState:shot ? tl.map_shot : tl.map];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
        [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
        [enc setFragmentTexture:tl.mapt atIndex:0];
        [enc setFragmentTexture:tl.atlas atIndex:1];
        [enc setFragmentTexture:tl.clst atIndex:2];
        [enc setFragmentTexture:tl.matt atIndex:3];
        [enc setFragmentTexture:tl.ovlt atIndex:4];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        if (ns) {
            [enc setRenderPipelineState:shot ? tl.spr_shot : tl.spr];
            [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
            [enc setVertexBytes:tl.live + TL_OFF_SPRITE length:(NSUInteger)ns * 16 atIndex:1];
            [enc setFragmentTexture:tl.sprt atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
                  instanceCount:ns];
        }
    }
}

/* ---- a tower (ROSGD_SURF_TOWER) --------------------------------------------
 * !Tower's tower, drawn by the window from two blocks the box writes into
 * two backings in the region (rosgd/include/rosgd/tower.h, which this
 * mirrors; rosgd/ports/tower/docs/graphics-design.md is the design): the
 * live block (the cells, the rooms, the fixtures, the stops, the movers,
 * section by section with a generation each) and the art block (the
 * tables and the pictures, uploaded once for each of its gens).  The
 * field (P1, design 7.5) is one quad over each visible piece of the icon,
 * which is the whole tower: the Wimp scrolls it and a zoom is a new size
 * of icon; the fixtures (P2) and the movers (P3) are instanced over it.
 * All three qualities (classic, smooth, rich: design 7.1), as the live
 * block asks.  The library is built for rich (TW_Q 2); if it will not
 * build, again for classic only (TW_Q 0: the smooth and rich parts left
 * out), and the tower is drawn in classic whatever it asks.
 * ROSGD_TOWER_FAIL_RICH=1 makes the rich library fail, to prove that.
 *
 * The shader text is the design's section 7.5 as it stands; the CPU twin
 * (ports/tower/surface, twin.rs) is its reference, and classic at 4x is to
 * match it exactly.  For that the mip levels are made here as the twin
 * makes them (a rounded 2 x 2 box), not by the GPU, and the library is
 * built without fast maths (design 8.3). */

#define TW_MAGIC        0x52574F54u     /* "TOWR" */
#define TW_ART_MAGIC    0x41525754u     /* "TWRA" */
#define TW_VERSION      1u
#define TW_GRID_W       375
#define TW_GRID_H       120
#define TW_STRIDE       384
#define TW_NROOMS       16384
#define TW_NART         1024
#define TW_NSPR         1024
#define TW_RAMP         48
#define TW_PROFILES     16
#define TW_LIVE_SIZE    712960u
#define TW_NFIX         512
#define TW_NSHAFTS      64
#define TW_NROWS        128
#define TW_NMOVERS      8192
#define TW_OFF_FIX      630976u
#define TW_OFF_STOPS    639168u
#define TW_OFF_MOVERS   647424u
#define TW_OFF_CELLS    192u
#define TW_OFF_ROOMS    368832u
#define TW_F_WEEKEND    1u
#define TW_A_F_AUX      1u
enum { TW_CELLS, TW_ROOMS, TW_FIX, TW_STOPS, TW_PAL, TW_MOVERS, TW_NSECT };

typedef struct TwHead {             /* struct ros_tower's first 192 bytes */
    uint32_t magic, version, frame, art;
    uint16_t grid_w, grid_h, ground_row, scale;
    uint16_t nrooms, nfixtures, nmovers;
    uint8_t quality, rate, view, flags;
    uint16_t hour, day;
    int16_t outline[4];
    uint16_t reserved0;
    uint32_t outline_rgba, back_rgba;
    uint32_t gen[8];
    uint32_t room_gen[16];
    uint32_t reserved[10];
} TwHead;
_Static_assert(sizeof(TwHead) == 192, "a tower's header is 192 bytes");
_Static_assert(offsetof(TwHead, gen) == 56, "gen at 56");

typedef struct TwArtHead {          /* struct ros_tower_art, before its pictures */
    uint32_t magic, version, gen, size;
    uint16_t scale, atlas_w, atlas_h, spr_w, spr_h, nart, nsprites, emit_shift, flags, lang;
    uint32_t reserved[7];
    uint32_t art[TW_NART][4];
    uint32_t spr[TW_NSPR][4];
    uint8_t ramp[4][TW_RAMP][4];
    uint8_t prof[2][TW_PROFILES][24];
} TwArtHead;
_Static_assert(sizeof(TwArtHead) == 34368, "the art block's pictures start at 34,368");

typedef struct TwData {             /* struct ros_surf_tower: the surface's 32 bytes */
    uint32_t backing, offset, size, frame;
    uint32_t art_backing, art_offset, art_size, art_gen;
} TwData;

typedef struct TwU {                /* design 7.2: 480 bytes, the same in C, GLSL and Metal */
    float box[4], clip[4], view[4], clock[4];
    float zen[4], hor[4], amb[4], misc[4];
    int32_t grid[4], counts[4], sizes[4], mode[4], outline[4];
    float ocol[4];
    float pal[16][4];
} TwU;
_Static_assert(sizeof(TwU) == 480, "the tower's uniforms are 480 bytes");

static NSString * const TW_SRC = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct TwU {\n"
"    float4 box, clip, view, clock;\n"
"    float4 zen, hor, amb, misc;\n"
"    int4   grid, counts, sizes, mode, outline;\n"
"    float4 ocol;\n"
"    float4 pal[16];\n"
"};\n"
"struct Res {\n"
"    constant TwU *u;\n"
"    texture2d<ushort> cells;  texture2d<uint> rooms;\n"
"    texture2d<uint>   arttab; texture2d<uint> sprtab;\n"
"    texture2d<ushort> aux;\n"
"    texture2d<float>  albedo, emis, sprites, spremis, prof;\n"
"};\n"
"#define U (*R.u)\n"
"\n"
"constexpr sampler TRI(coord::normalized, filter::linear, mip_filter::linear,\n"
"                      address::clamp_to_edge);\n"
"constexpr sampler PROF(coord::normalized, filter::linear,\n"
"                       s_address::repeat, t_address::clamp_to_edge);\n"
"\n"
"constant float2 CELL = float2(4.0, 16.0);\n"
"constant uint S_VACANT = 1, S_SALE = 2, S_STUCK = 4, S_CLOSED = 8, S_INFESTED = 16,\n"
"              S_SHOWING = 32, S_SELECTED = 128, S_LIT = 256, S_DARK = 512;\n"
"constant uint A_REPEAT = 1, A_LAMP_ALL = 2, A_SOLID = 4, A_GROUND = 8, A_ANIM = 16,\n"
"              A_EMIT = 32, A_AUX = 64, A_FIRE = 128;\n"
"constant uint M_GLASS = 2, M_SCREEN = 7;\n"
"constant uint ART_SKY_FAR = 1, ART_SKY_NEAR = 2;\n"
"constant uint SPR_ROACH = 256;\n"
"constant float3 LAMP = float3(1.00, 0.90, 0.74);\n"
"constant float3 DARK_IN = float3(0.10, 0.12, 0.20);\n"
"constant float3 SLAB = float3(0.46, 0.46, 0.48), PART = float3(0.24, 0.24, 0.27);\n"
"constant float3 LUMA = float3(0.299, 0.587, 0.114);\n"
"\n"
"static float  gmod(float a, float b)   { return a - b * floor(a / b); }\n"
"static float2 gmod(float2 a, float2 b) { return a - b * floor(a / b); }\n"
"static float4 unpack8(uint w) { return float4(uint4(w, w >> 8, w >> 16, w >> 24) & 255u) / 255.0; }\n"
"\n"
"#if TW_Q >= 2 && defined(TW_FAIL)\n"
"this is not a shader;\n"
"#endif\n"
"static uint hash(uint x) {\n"
"    x ^= x >> 16; x *= 0x7feb352du;\n"
"    x ^= x >> 15; x *= 0x846ca68bu;\n"
"    x ^= x >> 16;\n"
"    return x;\n"
"}\n"
"static float h01(uint x) { return float(hash(x) >> 8) * (1.0 / 16777216.0); }\n"
"static float hash2(int ix, int iy, uint seed) { return h01(uint(ix) + hash(uint(iy) + seed)); }\n"
"static float vnoise(float2 p, uint seed) {\n"
"    float2 i = floor(p), f = p - i;\n"
"    float2 u = f * f * (3.0 - 2.0 * f);\n"
"    int2 k = int2(i);\n"
"    float a = hash2(k.x, k.y, seed), b = hash2(k.x + 1, k.y, seed);\n"
"    float c = hash2(k.x, k.y + 1, seed), d = hash2(k.x + 1, k.y + 1, seed);\n"
"    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);\n"
"}\n"
"\n"
"static uint4 artrec(thread const Res &R, uint a) {\n"
"    return R.arttab.read(uint2(min(a, uint(U.counts.y) - 1u), 0));\n"
"}\n"
"static uint4 sprrec(thread const Res &R, uint s) {\n"
"    return R.sprtab.read(uint2(min(s, uint(U.counts.z) - 1u), 0));\n"
"}\n"
"static uint4 room(thread const Res &R, uint id) { return R.rooms.read(uint2(id & 127u, id >> 7)); }\n"
"static uint4 cellat(thread const Res &R, int2 c) {\n"
"    if (c.x < 0 || c.y < 0 || c.x >= U.grid.x || c.y >= U.grid.y) return uint4(0);\n"
"    uint4 k = uint4(R.cells.read(uint2(c)));\n"
"    if (k.x >= uint(U.counts.x)) k.x = 0;\n"
"    return k;\n"
"}\n"
"static uint4 rect(uint4 A, uint fr) {\n"
"    uint n = max(A.w >> 16, 1u), w = (A.y & 0xffffu) / n;\n"
"    return uint4((A.x & 0xffffu) + (fr % n) * w, A.x >> 16, w, A.y >> 16);\n"
"}\n"
"static float4 atl(thread const Res &R, uint4 r, float2 p, float lod) {\n"
"    float e = 0.5 * exp2(ceil(lod));\n"
"    float2 q = clamp(p, float2(e), max(float2(r.zw) - e, float2(e))) + float2(r.xy);\n"
"    return R.albedo.sample(TRI, q / float2(U.sizes.xy), level(lod));\n"
"}\n"
"static float4 atl_emis(thread const Res &R, uint4 r, float2 p, float lod) {\n"
"    float e = 0.5 * exp2(ceil(lod));\n"
"    float2 q = clamp(p, float2(e), max(float2(r.zw) - e, float2(e))) + float2(r.xy);\n"
"    return R.emis.sample(TRI, q / float2(U.sizes.xy), level(max(lod - float(U.mode.y), 0.0)));\n"
"}\n"
"static uint2 auxat(thread const Res &R, uint4 r, float2 p) {\n"
"    int2 q = clamp(int2(floor(p)), int2(0), int2(r.zw) - 1) + int2(r.xy);\n"
"    return uint2(R.aux.read(uint2(q)).rg);\n"
"}\n"
"static float4 spr(thread const Res &R, uint s, uint fr, float2 uv, float lod, thread float4 &glow) {\n"
"    uint4 S = sprrec(R, s);\n"
"    uint n = max(S.w >> 16, 1u);\n"
"    float2 sz = float2(float(S.y & 0xffffu), float(S.y >> 16));\n"
"    float e = 0.5 * exp2(ceil(lod));\n"
"    float2 q = clamp(clamp(uv, 0.0, 1.0) * sz, float2(e), max(sz - e, float2(e)))\n"
"             + float2(float((S.x & 0xffffu) + (fr % n) * (S.y & 0xffffu)), float(S.x >> 16));\n"
"    float2 t = q / float2(U.sizes.zw);\n"
"    glow = (S.w & (A_EMIT | 64u)) != 0 ? R.spremis.sample(TRI, t, level(lod)) : float4(0);\n"
"    return R.sprites.sample(TRI, t, level(lod));\n"
"}\n"
"static float3 over(float3 under, float4 top) { return top.rgb + under * (1.0 - top.a); }\n"
"\n"
"static float night_alpha(float hour) {\n"
"    float h = hour < 7.0 ? hour + 24.0 : hour;\n"
"    if (h < 20.0) return 0.0;\n"
"    if (h < 21.0) return h - 20.0;\n"
"    if (h < 28.0) return 1.0;\n"
"    if (h < 30.0) return 1.0 - (h - 28.0) * 0.5;\n"
"    return 0.0;\n"
"}\n"
"static float3 outdoors(thread const Res &R, float3 c, float3 glow) {\n"
"    if (U.counts.w == 0) return c;\n"
"    return c * U.amb.rgb + glow * (1.0 - U.misc.x);\n"
"}\n"
"static float lamp(thread const Res &R, uint4 Rm, uint group) {\n"
"    uint st = Rm.z & 0xffffu;\n"
"    if ((st & (S_DARK | S_CLOSED)) != 0) return 0.0;\n"
"    if ((st & S_LIT) != 0) return 1.0;\n"
"    float chance = R.prof.sample(PROF, float2((U.clock.x + 0.5) / 24.0,\n"
"                                (float((Rm.y >> 16) & 15u) + 0.5) / 16.0), level(0.0)).r;\n"
"    return h01(Rm.w ^ group * 0x9e3779b9u ^ uint(U.clock.z) * 0x85ebca6bu) < chance ? 1.0 : 0.0;\n"
"}\n"
"\n"
"static float3 grad3(float3 a, float3 b, float3 c, float mid, float t) {\n"
"    return t < mid ? mix(a, b, t / mid) : mix(b, c, (t - mid) / (1.0 - mid));\n"
"}\n"
"static float3 sky(thread const Res &R, float2 m) {\n"
"    float ground = float(U.grid.z) * CELL.y;\n"
"    float t = clamp(m.y / ground, 0.0, 1.0);\n"
"    if (U.counts.w == 0) {\n"
"        float3 d = grad3(float3(0.102, 0.227, 0.431), float3(0.357, 0.659, 0.831),\n"
"                         float3(0.706, 0.867, 0.941), 0.6, t);\n"
"        float3 n = grad3(float3(0.016, 0.035, 0.102), float3(0.039, 0.071, 0.208),\n"
"                         float3(0.055, 0.122, 0.290), 0.5, t);\n"
"        return mix(d, n, night_alpha(U.clock.x));\n"
"    }\n"
"#if TW_Q >= 1\n"
"    float3 c = mix(U.zen.rgb, U.hor.rgb, t * t);\n"
"    int2 sq = int2(floor(m / 10.0));\n"
"    uint h = hash(uint(sq.x) * 73856093u ^ uint(sq.y) * 19349663u);\n"
"    if ((h & 255u) == 0) {\n"
"        float2 sp = (float2(sq) + float2(float((h >> 8) & 255u), float((h >> 16) & 255u)) / 255.0) * 10.0;\n"
"        float d = length(m - sp) * float(U.grid.w) / U.view.z;\n"
"        c += U.misc.y * (1.0 - smoothstep(0.6, 1.6, d)) * float3(0.90, 0.94, 1.00);\n"
"    }\n"
"#if TW_Q >= 2\n"
"    if (U.counts.w >= 2 && U.misc.z > 0.0) {\n"
"        float k = clamp((U.clock.x - 6.0) / 13.0, 0.0, 1.0);\n"
"        float2 sun = float2(1500.0 * (0.1 + 0.8 * k), ground * (0.9 - 0.7 * sin(3.14159265 * k)));\n"
"        float d = length(m - sun);\n"
"        c += U.misc.z * (float3(1.0, 0.95, 0.80) * (1.0 - smoothstep(14.0, 15.0, d))\n"
"                         + float3(1.0, 0.80, 0.50) * 0.25 * exp2(-d / 75.0));\n"
"    }\n"
"#endif\n"
"    for (uint s = ART_SKY_FAR; s <= ART_SKY_NEAR && U.counts.w >= 1; s++) {\n"
"        uint4 A = artrec(R, s), r = rect(A, 0);\n"
"        float S = float(U.grid.w);\n"
"        float hgt = float(r.w) / S;\n"
"        float y = m.y - (ground - hgt);\n"
"        if (y >= 0.0) {\n"
"            float2 p = float2(gmod(m.x * S, float(r.z)), y * S);\n"
"            float4 g = (A.w & A_EMIT) != 0 ? atl_emis(R, r, p, U.view.w) : float4(0);\n"
"            float4 a = atl(R, r, p, U.view.w);\n"
"            c = outdoors(R, a.rgb, g.rgb) + c * (1.0 - a.a);\n"
"        }\n"
"    }\n"
"    return c;\n"
"#else\n"
"    return float3(0);\n"
"#endif\n"
"}\n"
"\n"
"static float4 piece(thread const Res &R, uint a, float2 m, int2 c, thread float3 &glow) {\n"
"    uint4 A = artrec(R, a);\n"
"    if ((A.w & A_SOLID) != 0) { glow = float3(0); return unpack8(A.x); }\n"
"    uint fr = (A.w & A_ANIM) != 0 ? uint(U.clock.y * 8.0) + hash(uint(c.x) * 31u + uint(c.y)) : 0u;\n"
"    uint4 r = rect(A, fr);\n"
"    float S = float(U.grid.w);\n"
"    float2 p = float2(gmod(m.x * S, float(r.z)),\n"
"                      (A.w & A_GROUND) != 0 ? m.y * S - (float(U.grid.z) * CELL.y * S - float(r.w))\n"
"                                            : gmod(m.y * S, float(r.w)));\n"
"    glow = (A.w & A_EMIT) != 0 ? atl_emis(R, r, p, U.view.w).rgb : float3(0);\n"
"    return atl(R, r, p, U.view.w);\n"
"}\n"
"\n"
"static float3 roaches(thread const Res &R, uint4 Rm, float2 l, float2 size, float floor_y, float3 col) {\n"
"    int n = min(6, 2 + int((Rm.x >> 16) & 511u) / 4);\n"
"    for (int k = 0; k < 6; k++) {\n"
"        if (k >= n) break;\n"
"        uint hk = hash(Rm.w + uint(k) * 0x9e3779b9u);\n"
"        float dir = (hk & 1u) != 0 ? 1.0 : -1.0;\n"
"        float speed = 1.5 + float((hk >> 1) & 7u) * 0.25;\n"
"        float bx = gmod(float(hk >> 8) + dir * speed * U.clock.y, size.x - 4.0) + 2.0;\n"
"        float by = floor_y - 0.5 - float(k & 1);\n"
"        float2 d = (l - float2(bx - 1.25, by - 2.0)) / float2(2.5, 2.0);\n"
"        if (d.x >= 0.0 && d.y >= 0.0 && d.x < 1.0 && d.y < 1.0) {\n"
"            if (dir < 0.0) d.x = 1.0 - d.x;\n"
"            float4 g;\n"
"            float4 b = spr(R, SPR_ROACH, uint(U.clock.y * 9.0) + uint(k), d, U.view.w, g);\n"
"            col = over(col, b);\n"
"        }\n"
"    }\n"
"    return col;\n"
"}\n"
"\n"
"static float4 shade_room(thread const Res &R, uint4 Rm, float2 m, int2 c) {\n"
"    float x0 = float(Rm.x & 511u), yt = float((Rm.x >> 9) & 127u);\n"
"    float wc = float((Rm.x >> 16) & 511u), hf = float(max((Rm.x >> 25) & 7u, 1u));\n"
"    uint st = Rm.z & 0xffffu, occ = (Rm.z >> 16) & 255u;\n"
"    float2 size = float2(wc, hf) * CELL;\n"
"    float2 l = m - float2(x0, yt) * CELL;\n"
"    float fy = gmod(l.y, CELL.y);\n"
"    int q = U.counts.w;\n"
"    float lod = U.view.w, S = float(U.grid.w);\n"
"\n"
"    float gx = q == 0 ? 1.0 : 0.75, gy = 1.0;\n"
"    if (l.x >= size.x - gx || fy >= CELL.y - gy) {\n"
"        if (q == 0) return float4(0);\n"
"#if TW_Q >= 1\n"
"        float3 s = fy >= CELL.y - gy ? SLAB : PART;\n"
"        return float4(s * mix(float3(0.45), float3(1.0), U.misc.x), 1.0);\n"
"#else\n"
"        return float4(0);\n"
"#endif\n"
"    }\n"
"\n"
"    uint4 A = artrec(R, Rm.y & 0xffffu);\n"
"    float3 col, glow = float3(0);\n"
"    uint mat = 0, grp = 0, ht = 0;\n"
"    uint4 r = uint4(0);\n"
"    float2 p = float2(0);\n"
"    if ((A.w & A_SOLID) != 0) {\n"
"        col = unpack8(A.x).rgb;\n"
"    } else {\n"
"        uint fr = ((A.w & A_ANIM) != 0 || (st & S_SHOWING) != 0) ? uint(U.clock.y * 6.0) : 0u;\n"
"        r = rect(A, fr);\n"
"        uint period = A.z >> 16;\n"
"        p = (A.w & A_REPEAT) != 0\n"
"            ? float2(gmod(l.x * S, float(period != 0 ? period : r.z)), l.y / size.y * float(r.w))\n"
"            : l / size * float2(r.zw);\n"
"        float4 a = atl(R, r, p, lod);\n"
"        if ((A.w & A_AUX) != 0) {\n"
"            uint2 x = auxat(R, r, p);\n"
"            uint slot = x.r >> 4;\n"
"            mat = x.r & 15u; grp = x.g & 15u; ht = x.g >> 4;\n"
"            uint alt = A.z & 0xffffu;\n"
"            if (slot != 0 && alt != 0 && float(occ) > h01(Rm.w ^ slot * 0x27d4eb2fu) * 255.0)\n"
"                a = atl(R, rect(artrec(R, alt), fr), p, lod);\n"
"        }\n"
"        col = a.rgb;\n"
"        if ((A.w & A_EMIT) != 0) glow = atl_emis(R, r, p, lod).rgb;\n"
"    }\n"
"\n"
"#if TW_Q >= 1\n"
"    if (q >= 1) {\n"
"        float on = lamp(R, Rm, (A.w & A_LAMP_ALL) != 0 ? 0u : grp);\n"
"        float day = U.misc.x;\n"
"        float3 base = col;\n"
"        col = base * mix(mix(DARK_IN, float3(1.0), day), mix(LAMP, float3(1.0), day), on)\n"
"            + glow * on * (1.0 - day);\n"
"        if (mat == M_GLASS) col = mix(col, sky(R, m), 0.7 - 0.35 * on);\n"
"        if (mat == M_SCREEN && (st & S_SHOWING) != 0)\n"
"            col += float3(0.6, 0.7, 0.9) * (0.25 + 0.2 * vnoise(float2(U.clock.y * 4.0, x0), 51u));\n"
"#if TW_Q >= 2\n"
"        if (q >= 2) {\n"
"            float ao = (1.0 - 0.22 * (1.0 - smoothstep(0.0, 2.5, fy)))\n"
"                     * (1.0 - 0.18 * (1.0 - smoothstep(0.0, 2.0, min(l.x, size.x - gx - l.x))));\n"
"            float sh = 0.0;\n"
"            if ((A.w & A_AUX) != 0)\n"
"                for (int k = 1; k <= 3; k++) {\n"
"                    uint2 y = auxat(R, r, p - float2(1.0, 2.0) * float(k) * exp2(lod));\n"
"                    if (float(y.g >> 4) > float(ht) + float(k)) sh = max(sh, 1.0 - 0.25 * float(k));\n"
"                }\n"
"            col *= ao * (1.0 - 0.35 * sh);\n"
"            if (day < 0.95) {\n"
"                uint4 kl = cellat(R, int2(int(x0) - 1, c.y)), kr = cellat(R, int2(int(x0 + wc), c.y));\n"
"                float s = 0.0;\n"
"                if (kl.x != 0) s += lamp(R, room(R, kl.x), 0u) * exp2(-l.x / 6.0);\n"
"                if (kr.x != 0) s += lamp(R, room(R, kr.x), 0u) * exp2(-(size.x - l.x) / 6.0);\n"
"                col += base * LAMP * s * 0.35 * (1.0 - day);\n"
"            }\n"
"            float w = fract((m.x + m.y * 0.35) / 225.0);\n"
"            float streak = smoothstep(0.40, 0.45, w) - smoothstep(0.52, 0.60, w);\n"
"            col = mix(col, U.hor.rgb, U.misc.w * (0.4 + 0.6 * streak));\n"
"        }\n"
"#endif\n"
"    }\n"
"#endif\n"
"\n"
"    if ((st & S_STUCK) != 0) col = float3(dot(col, LUMA) * 0.85);\n"
"    uint sign = Rm.z >> 24;\n"
"    if (sign != 0 && l.y < CELL.y) {\n"
"        float4 g;\n"
"        col = over(col, spr(R, sign, 0u, l / float2(size.x, CELL.y), lod, g));\n"
"    }\n"
"    if ((st & S_INFESTED) != 0) col = roaches(R, Rm, l, size, size.y - gy, col);\n"
"    uint ev = (Rm.y >> 20) & 3u;\n"
"    if (U.mode.x == 1 && ev != 0) col = over(col, U.pal[ev]);\n"
"    if (U.mode.x == 2) col = over(col, U.pal[4 + int((Rm.y >> 24) & 255u) * 12 / 256]);\n"
"    if ((st & S_SELECTED) != 0) {\n"
"        float e = 2.0 * U.view.z / S;\n"
"        if (l.x < e || l.y < e || l.x > size.x - gx - e || l.y > size.y - gy - e) col = U.ocol.rgb;\n"
"    }\n"
"    return float4(col, 1.0);\n"
"}\n"
"\n"
"#if TW_Q >= 2\n"
"static float3 burn(thread const Res &R, float3 col, float2 m, float fy) {\n"
"    float up = 1.0 - fy / CELL.y;\n"
"    float n = vnoise(float2(m.x / 3.0, m.y / 3.0 + U.clock.y * 2.5), 61u) * 0.7\n"
"            + vnoise(float2(m.x / 1.25, m.y / 1.25 + U.clock.y * 5.0), 62u) * 0.3;\n"
"    float flame = smoothstep(0.35, 0.75, n * (1.25 - up));\n"
"    float3 hot = mix(float3(0.85, 0.20, 0.05), float3(1.0, 0.85, 0.35), flame);\n"
"    return mix(col * 0.30, hot, flame) + float3(0.35, 0.10, 0.02) * (1.0 - up);\n"
"}\n"
"#endif\n"
"\n"
"static float4 field(thread const Res &R, float2 f) {\n"
"    float2 world = float2(U.grid.xy) * CELL;\n"
"    float2 m = f / U.box.zw * world;\n"
"    if (m.x < 0.0 || m.y < 0.0 || m.x >= world.x || m.y >= world.y) return float4(0);\n"
"    int2 c = int2(floor(m / CELL));\n"
"    uint4 k = cellat(R, c);\n"
"    float3 col = c.y < U.grid.z ? sky(R, m) : float3(0.24, 0.13, 0.06);\n"
"    float3 g;\n"
"    if (k.y != 0) {\n"
"        float4 b = piece(R, k.y, m, c, g);\n"
"        col = outdoors(R, b.rgb, g) + col * (1.0 - b.a);\n"
"    }\n"
"    if (k.x != 0) {\n"
"        float4 r = shade_room(R, room(R, k.x), m, c);\n"
"        col = mix(col, r.rgb, r.a);\n"
"    }\n"
"    if (k.z != 0) {\n"
"        float4 b = piece(R, k.z, m, c, g);\n"
"        if (U.counts.w == 0 && (artrec(R, k.z).w & A_FIRE) != 0)\n"
"            col = mix(col, float3(0.87, 0.20, 0.20), 0.5 * b.a);\n"
"#if TW_Q >= 2\n"
"        else if (U.counts.w >= 2) col = mix(col, burn(R, col, m, gmod(m.y, CELL.y)), b.a * 0.6);\n"
"#endif\n"
"        col = over(col, float4(outdoors(R, b.rgb, g), b.a));\n"
"    }\n"
"    int4 o = U.outline;\n"
"    if (o.z > o.x && c.x >= o.x && c.y >= o.y && c.x < o.z && c.y < o.w) {\n"
"        float2 kk = U.box.zw / world;\n"
"        float2 lo = (m - float2(o.xy) * CELL) * kk, hi = (float2(o.zw) * CELL - m) * kk;\n"
"        if (min(min(lo.x, lo.y), min(hi.x, hi.y)) < 2.0) col = U.ocol.rgb;\n"
"    }\n"
"    return float4(clamp(col, 0.0, 1.0), 1.0);\n"
"}\n"
"\n"
"struct TwV { float4 pos [[position]]; };\n"
"\n"
"vertex TwV tw_vs(uint id [[vertex_id]], constant TwU &u [[buffer(0)]]) {\n"
"    float2 k = float2(id & 1, (id >> 1) & 1);\n"
"    float2 p = mix(u.clip.xy, u.clip.zw, k);\n"
"    TwV o;\n"
"    o.pos = float4(p / u.view.xy * float2(2, -2) + float2(-1, 1), 0, 1);\n"
"    return o;\n"
"}\n"
"\n"
"fragment float4 tw_field(TwV v [[stage_in]], constant TwU &u [[buffer(0)]],\n"
"        texture2d<ushort> cells   [[texture(0)]], texture2d<uint>  rooms   [[texture(1)]],\n"
"        texture2d<uint>   arttab  [[texture(2)]], texture2d<uint>  sprtab  [[texture(3)]],\n"
"        texture2d<float>  albedo  [[texture(4)]], texture2d<float> emis    [[texture(5)]],\n"
"        texture2d<ushort> aux     [[texture(6)]], texture2d<float> sprites [[texture(7)]],\n"
"        texture2d<float>  spremis [[texture(8)]], texture2d<float> prof    [[texture(9)]])\n"
"{\n"
"    Res R = { &u, cells, rooms, arttab, sprtab, aux, albedo, emis, sprites, spremis, prof };\n"
"    float4 c = field(R, v.pos.xy - u.box.xy);       /* Metal's origin is the top left */\n"
"    if (c.a == 0.0) discard_fragment();\n"
"    return c;\n"
"}\n"
"\n"
"/* ---- P2, the fixtures (design 7.7) --------------------------------------- */\n"
"constant uint FIX_CARLIFT = 5;\n"
"\n"
"struct TwF { float4 pos [[position]]; float2 lp; uint4 F [[flat]]; };\n"
"\n"
"vertex TwF tw_fix_vs(uint id [[vertex_id]], uint inst [[instance_id]],\n"
"                     constant TwU &u [[buffer(0)]], texture2d<uint> fixtab [[texture(0)]])\n"
"{\n"
"    uint4 r = fixtab.read(uint2(min(int(inst), u.mode.z - 1), 0));\n"
"    float over_ = float((r.z >> 8) & 255u);\n"
"    float2 o  = float2(float(r.x & 511u), float((r.x >> 9) & 127u)) * CELL - float2(0.0, over_);\n"
"    float2 sz = float2(float((r.x >> 16) & 255u) * CELL.x, float((r.x >> 24) & 127u) * CELL.y + over_);\n"
"    float2 k  = float2(id & 1, (id >> 1) & 1);\n"
"    TwF v;\n"
"    v.lp = k * sz;\n"
"    float2 world = float2(u.grid.xy) * CELL;\n"
"    float2 t = u.box.xy + (o + v.lp) / world * u.box.zw;\n"
"    v.pos = float4(t / u.view.xy * float2(2, -2) + float2(-1, 1), 0, 1);\n"
"    v.F = r;\n"
"    return v;\n"
"}\n"
"\n"
"fragment float4 tw_fix(TwF v [[stage_in]], constant TwU &u [[buffer(0)]],\n"
"        texture2d<ushort> cells   [[texture(0)]], texture2d<uint>  rooms   [[texture(1)]],\n"
"        texture2d<uint>   arttab  [[texture(2)]], texture2d<uint>  sprtab  [[texture(3)]],\n"
"        texture2d<float>  albedo  [[texture(4)]], texture2d<float> emis    [[texture(5)]],\n"
"        texture2d<ushort> aux     [[texture(6)]], texture2d<float> sprites [[texture(7)]],\n"
"        texture2d<float>  spremis [[texture(8)]], texture2d<float> prof    [[texture(9)]],\n"
"        texture2d<ushort> stops   [[texture(10)]])\n"
"{\n"
"    Res R = { &u, cells, rooms, arttab, sprtab, aux, albedo, emis, sprites, spremis, prof };\n"
"    uint4 F = v.F;\n"
"    uint kind = (F.y >> 16) & 255u, base = F.y & 0xffffu;\n"
"    float S = float(u.grid.w), lod = u.view.w;\n"
"    float over_ = float((F.z >> 8) & 255u);\n"
"    if (kind <= 2 || kind == FIX_CARLIFT) {\n"
"        if (v.lp.y < over_) {                       /* the top, above the highest floor, standing on it */\n"
"            uint4 r = rect(artrec(R, base + 1u), 0);\n"
"            float2 p = float2(v.lp.x * S, (v.lp.y - over_) * S + float(r.w));\n"
"            float4 c = atl(R, r, p, lod);\n"
"            return float4(outdoors(R, c.rgb, float3(0)), c.a);\n"
"        }\n"
"        int top = int((F.x >> 9) & 127u), rows = int((F.x >> 24) & 127u);\n"
"        int row = top + int(floor((v.lp.y - over_) / CELL.y));\n"
"        float fy = gmod(v.lp.y - over_, CELL.y);\n"
"        float2 p = float2(v.lp.x * S, fy * S);\n"
"        if (row >= top + rows - 1) {                /* the pit, under the lowest floor */\n"
"            uint4 r = rect(artrec(R, base + 2u), 0);\n"
"            if (p.y >= float(r.w)) return float4(0);\n"
"            float4 c = atl(R, r, p, lod);\n"
"            return float4(outdoors(R, c.rgb, float3(0)), c.a);\n"
"        }\n"
"        uint s = uint(stops.read(uint2(clamp(row, 0, 127), F.z & 63u)).r);\n"
"        uint a = (s & 1u) != 0 ? base + 4u + ((s >> 4) & 3u) : base + 3u;\n"
"        float4 body = atl(R, rect(artrec(R, base), 0), p, lod);\n"
"        uint4 r = rect(artrec(R, a), 0);\n"
"        float4 front = atl(R, r, p, lod);\n"
"        float4 g = (artrec(R, a).w & A_EMIT) != 0 ? atl_emis(R, r, p, lod) : float4(0);\n"
"        float calling = (s & 6u) != 0 ? 1.0 : 0.25;\n"
"        float3 c = over(outdoors(R, body.rgb, float3(0)),\n"
"                        float4(outdoors(R, front.rgb, g.rgb * calling), front.a));\n"
"        return float4(c, max(body.a, front.a));\n"
"    }\n"
"    uint4 A = artrec(R, base);\n"
"    uint fr = (A.w & A_ANIM) != 0 ? uint(u.clock.y * 8.0) : 0u;\n"
"    uint4 r = rect(A, fr);\n"
"    float2 sz = float2(float((F.x >> 16) & 255u) * CELL.x, float((F.x >> 24) & 127u) * CELL.y + over_);\n"
"    float2 p = v.lp / sz * float2(r.zw);\n"
"    float4 c = atl(R, r, p, lod);\n"
"    float4 g = (A.w & A_EMIT) != 0 ? atl_emis(R, r, p, lod) : float4(0);\n"
"    return float4(outdoors(R, c.rgb, g.rgb), c.a);\n"
"}\n"
"\n"
"/* ---- P3, the movers (design 7.9) ------------------------------------------ */\n"
"constant uint SPR_TINT = 64, P_TINTED = 0x8000;\n"
"constant float3 TOPS[16] = {\n"
"    float3(0x2f, 0x5d, 0x50) / 255.0, float3(0x3b, 0x5b, 0x8a) / 255.0, float3(0x6b, 0x5a, 0x7e) / 255.0,\n"
"    float3(0x7a, 0x6a, 0x4f) / 255.0, float3(0x4a, 0x4f, 0x57) / 255.0, float3(0xb9, 0xa3, 0xd6) / 255.0,\n"
"    float3(0x2f, 0x4a, 0x7a) / 255.0, float3(0xe9, 0xee, 0xf2) / 255.0, float3(0xff, 0x5f, 0xa2) / 255.0,\n"
"    float3(0xd8, 0x19, 0x19) / 255.0, float3(0), float3(0), float3(0), float3(0), float3(0), float3(0) };\n"
"constant float3 LEGS[4] = { float3(0x3b, 0x42, 0x50) / 255.0, float3(0x2e, 0x34, 0x40) / 255.0,\n"
"                            float3(0xa3, 0x3a, 0x6c) / 255.0, float3(0x8a, 0x10, 0x10) / 255.0 };\n"
"constant float3 SKINS[4] = { float3(0xf1, 0xc2, 0x7d) / 255.0, float3(0xe0, 0xac, 0x69) / 255.0,\n"
"                             float3(0xc6, 0x86, 0x42) / 255.0, float3(0x8d, 0x55, 0x24) / 255.0 };\n"
"\n"
"struct TwM { float4 pos [[position]]; float2 uv; float2 at; uint4 M [[flat]]; uint4 SP [[flat]]; };\n"
"\n"
"vertex TwM tw_mov_vs(uint id [[vertex_id]], uint inst [[instance_id]],\n"
"                     constant TwU &u [[buffer(0)]],\n"
"                     texture2d<ushort> movers [[texture(11)]], texture2d<uint> sprtab [[texture(3)]])\n"
"{\n"
"    int i = min(int(inst), u.mode.w - 1);\n"
"    uint4 mv = uint4(movers.read(uint2(i & 255, i >> 8)));\n"
"    uint4 s = sprtab.read(uint2(min(mv.z & 1023u, uint(u.counts.z) - 1u), 0));\n"
"    float2 feet = float2(mv.xy) * 0.25;\n"
"    float2 sz = float2(float(s.z & 0xffffu), float(s.z >> 16)) * 0.25;\n"
"    float2 k = float2(id & 1, (id >> 1) & 1);\n"
"    float2 p = feet + (k - float2(0.5, 1.0)) * sz;\n"
"    float2 world = float2(u.grid.xy) * CELL;\n"
"    float2 t = u.box.xy + p / world * u.box.zw;\n"
"    TwM o;\n"
"    o.pos = float4(t / u.view.xy * float2(2, -2) + float2(-1, 1), 0, 1);\n"
"    o.uv = float2((mv.z & 1024u) != 0 ? 1.0 - k.x : k.x, k.y);\n"
"    o.at = p;\n"
"    o.M = mv; o.SP = s;\n"
"    return o;\n"
"}\n"
"\n"
"fragment float4 tw_mov(TwM v [[stage_in]], constant TwU &u [[buffer(0)]],\n"
"        texture2d<ushort> cells   [[texture(0)]], texture2d<uint>  rooms   [[texture(1)]],\n"
"        texture2d<uint>   arttab  [[texture(2)]], texture2d<uint>  sprtab  [[texture(3)]],\n"
"        texture2d<float>  albedo  [[texture(4)]], texture2d<float> emis    [[texture(5)]],\n"
"        texture2d<ushort> aux     [[texture(6)]], texture2d<float> sprites [[texture(7)]],\n"
"        texture2d<float>  spremis [[texture(8)]], texture2d<float> prof    [[texture(9)]])\n"
"{\n"
"    Res R = { &u, cells, rooms, arttab, sprtab, aux, albedo, emis, sprites, spremis, prof };\n"
"    if ((v.M.z & 2048u) != 0) {                     /* BEHIND: hidden by any room */\n"
"        int2 c = int2(floor(v.at / CELL));\n"
"        if (c.x >= 0 && c.y >= 0 && c.x < u.grid.x && c.y < u.grid.y &&\n"
"            cells.read(uint2(c)).x != 0) discard_fragment();\n"
"    }\n"
"    if ((v.M.z & 8192u) != 0) {                     /* WITHIN: only over room M.w */\n"
"        int2 c = int2(floor(v.at / CELL));\n"
"        if (c.x < 0 || c.y < 0 || c.x >= u.grid.x || c.y >= u.grid.y ||\n"
"            cells.read(uint2(c)).x != v.M.w) discard_fragment();\n"
"    }\n"
"    uint fr = (v.SP.w & 16u) != 0 ? uint(u.clock.y * 8.0) : (v.M.z & 8192u) != 0 ? 0u : (v.M.w & 15u);\n"
"    float4 g;\n"
"    float4 a = spr(R, v.M.z & 1023u, fr, v.uv, u.view.w, g);\n"
"    if ((v.SP.w & SPR_TINT) != 0) {                 /* a person: the tint layer's weights */\n"
"        float4 w = g;\n"
"        g = float4(0);\n"
"        if ((v.M.w & P_TINTED) != 0) {\n"
"            float3 top = TOPS[(v.M.w >> 4) & 15u], legs = LEGS[(v.M.w >> 8) & 3u], skin = SKINS[(v.M.w >> 10) & 3u];\n"
"            a.rgb = a.rgb * (1.0 - w.r - w.g - w.b) + (top * w.r + skin * w.g + legs * w.b) * a.a;\n"
"        }\n"
"    }\n"
"    if (a.a <= 0.0 && g.a <= 0.0) discard_fragment();\n"
"    float3 c = (v.M.z & 4096u) != 0 && u.counts.w >= 1\n"
"             ? a.rgb * mix(LAMP, float3(1.0), u.misc.x) + g.rgb * (1.0 - u.misc.x)\n"
"             : outdoors(R, a.rgb, g.rgb);\n"
"    return float4(c, a.a);\n"
"}\n"
"\n";

#define TW_RING 3                       /* per-frame textures: one the GPU may read, one written */

static struct {
    id<MTLRenderPipelineState> field, field_shot, fix, fix_shot, mov, mov_shot;
    id<MTLTexture> cells, rooms, arttab, sprtab, albedo, emis, aux, sprites, spremis, prof;
    id<MTLTexture> fixtab;
    id<MTLTexture> stops[TW_RING], movers[TW_RING];  /* design 8.4: a ring, as they change each frame */
    int ring;
    uint32_t frame, art_gen, gen[8], room_gen[16];
    int weekend;                    /* the lamp profiles uploaded: 0 weekday, 1 weekend, -1 none */
    int maxq;                       /* the best quality the library was built for */
    bool have, art_have, tried;
    uint8_t *live;                  /* the last whole frame of the live block */
    uint8_t *next;
    TwArtHead *ah;                  /* the art block's head, as uploaded */
} tw;

static void tw_setup(void)
{
    NSError *err = nil;
    id<MTLLibrary> lib;
    MTLCompileOptions *opts;

    if (tw.tried) {
        return;
    }
    tw.tried = true;
    tw.weekend = -1;
    tw.live = calloc(1, TW_LIVE_SIZE);
    tw.next = calloc(1, TW_LIVE_SIZE);
    tw.ah = calloc(1, sizeof(TwArtHead));
    if (!tw.live || !tw.next || !tw.ah) {
        return;
    }
    /* no fast maths: the twin's arithmetic, as the design asks (8.3) */
    opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    /* rich, else the classic program (design 3.3, safe_program) */
    for (int q = 2; q >= 0; q -= 2) {
        const char *fail = getenv("ROSGD_TOWER_FAIL_RICH");
        NSMutableDictionary *mac = [NSMutableDictionary dictionary];
        mac[@"TW_Q"] = @(q);
        if (fail && *fail == '1') {
            mac[@"TW_FAIL"] = @1;
        }
        opts.preprocessorMacros = mac;
        err = nil;
        lib = [m.device newLibraryWithSource:TW_SRC options:opts error:&err];
        if (lib) {
            tw.maxq = q;
            metal_log("tower: the %s program", q ? "rich" : "classic");
            break;
        }
        metal_log("tower: the %s program will not build: %s", q ? "rich" : "classic",
                  [[err localizedDescription] UTF8String]);
    }
    [opts release];
    if (!lib) {
        return;
    }
    tw.field = tl_pipe(lib, @"tw_vs", @"tw_field", MTLPixelFormatBGRA8Unorm);
    tw.field_shot = tl_pipe(lib, @"tw_vs", @"tw_field", MTLPixelFormatRGBA8Unorm);
    tw.fix = tl_pipe(lib, @"tw_fix_vs", @"tw_fix", MTLPixelFormatBGRA8Unorm);
    tw.fix_shot = tl_pipe(lib, @"tw_fix_vs", @"tw_fix", MTLPixelFormatRGBA8Unorm);
    tw.mov = tl_pipe(lib, @"tw_mov_vs", @"tw_mov", MTLPixelFormatBGRA8Unorm);
    tw.mov_shot = tl_pipe(lib, @"tw_mov_vs", @"tw_mov", MTLPixelFormatRGBA8Unorm);
    [lib release];
    tw.cells = gp_texture(MTLPixelFormatRGBA16Uint, TW_STRIDE, TW_GRID_H, MTLTextureUsageShaderRead);
    tw.rooms = gp_texture(MTLPixelFormatRGBA32Uint, 128, TW_NROOMS / 128, MTLTextureUsageShaderRead);
    tw.arttab = gp_texture(MTLPixelFormatRGBA32Uint, TW_NART, 1, MTLTextureUsageShaderRead);
    tw.sprtab = gp_texture(MTLPixelFormatRGBA32Uint, TW_NSPR, 1, MTLTextureUsageShaderRead);
    tw.prof = gp_texture(MTLPixelFormatR8Unorm, 24, TW_PROFILES, MTLTextureUsageShaderRead);
    tw.fixtab = gp_texture(MTLPixelFormatRGBA32Uint, TW_NFIX, 1, MTLTextureUsageShaderRead);
    for (int k = 0; k < TW_RING; k++) {
        tw.stops[k] = gp_texture(MTLPixelFormatR8Uint, TW_NROWS, TW_NSHAFTS, MTLTextureUsageShaderRead);
        tw.movers[k] = gp_texture(MTLPixelFormatRGBA16Uint, 256, TW_NMOVERS / 256, MTLTextureUsageShaderRead);
    }
}

/* The first visible tower, and its two blocks, whole inside the region */
static const MetalIcon *tw_first(TwData *d)
{
    const MetalSurfacesView *v = &surf.view;

    if (!surf.on || !v->nicons || !v->region) {
        return NULL;
    }
    for (uint32_t i = 0; i < v->nicons; i++) {
        const MetalIcon *ic = &v->icon[i];

        if (ic->kind != ROSGD_SURF_TOWER || !ic->nrects) {
            continue;
        }
        memcpy(d, ic->data, sizeof(*d));
        if (d->size < TW_LIVE_SIZE || d->offset == 0xFFFFFFFFu
            || (uint64_t)d->offset + d->size > v->region_size) {
            continue;
        }
        return ic;
    }
    return NULL;
}

/* half size, a rounded 2 x 2 box: the twin's mip levels (pack.rs halve) */
static uint8_t *tw_halve(const uint8_t *src, int w, int h)
{
    int hw = w / 2, hh = h / 2;
    uint8_t *d = malloc((size_t)hw * hh * 4 + 1);

    if (!d) {
        return NULL;
    }
    for (int y = 0; y < hh; y++) {
        const uint8_t *a = src + (size_t)(2 * y) * w * 4, *b = a + (size_t)w * 4;
        uint8_t *o = d + (size_t)y * hw * 4;
        for (int x = 0; x < hw; x++) {
            for (int c = 0; c < 4; c++) {
                unsigned v = a[8 * x + c] + a[8 * x + 4 + c] + b[8 * x + c] + b[8 * x + 4 + c];
                o[4 * x + c] = (uint8_t)((v + 2) / 4);
            }
        }
    }
    return d;
}

/* an RGBA8 picture with its mip levels 0 .. max made as the twin makes them */
static id<MTLTexture> tw_picture(const uint8_t *px, int w, int h, int max)
{
    int levels = 1;
    while (levels <= max && (w >> levels) >= 1 && (h >> levels) >= 1
           && (w >> (levels - 1)) >= 2 && (h >> (levels - 1)) >= 2) {
        levels++;
    }
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:w height:h
                                                                              mipmapped:YES];
    td.mipmapLevelCount = levels;
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [m.device newTextureWithDescriptor:td];
    const uint8_t *cur = px;
    uint8_t *own = NULL;

    for (int l = 0; t && l < levels; l++) {
        int lw = w >> l, lh = h >> l;
        [t replaceRegion:MTLRegionMake2D(0, 0, lw, lh) mipmapLevel:l
               withBytes:cur bytesPerRow:(NSUInteger)lw * 4];
        if (l + 1 < levels) {
            uint8_t *n = tw_halve(cur, lw, lh);
            free(own);
            own = n;
            cur = n;
            if (!n) {
                break;
            }
        }
    }
    free(own);
    return t;
}

/* The art block's tables and pictures into textures, when its gen moved */
static bool tw_take_art(const uint8_t *blk, uint32_t size, uint32_t gen)
{
    const TwArtHead *a = (const TwArtHead *)blk;

    if (tw.art_have && gen == tw.art_gen) {
        return true;
    }
    if (size < sizeof(TwArtHead) || a->magic != TW_ART_MAGIC || a->version != TW_VERSION) {
        return tw.art_have;
    }
    uint64_t aw = a->atlas_w, ah = a->atlas_h, sw = a->spr_w, sh = a->spr_h;
    uint32_t es = a->emit_shift > 3 ? 3 : a->emit_shift;
    uint64_t at_emis = sizeof(TwArtHead) + aw * ah * 4;
    uint64_t at_aux = at_emis + (aw >> es) * (ah >> es) * 4;
    uint64_t at_spr = at_aux + ((a->flags & TW_A_F_AUX) ? aw * ah * 2 : 0);
    uint64_t at_spremis = at_spr + sw * sh * 4;
    uint64_t total = at_spremis + sw * sh * 4;
    if (!aw || !ah || !sw || !sh || aw > 16384 || ah > 16384 || sw > 16384 || sh > 16384 || total > size) {
        return tw.art_have;             /* a block too short for what it says it holds */
    }
    memcpy(tw.ah, a, sizeof(TwArtHead));
    [tw.albedo release];
    [tw.emis release];
    [tw.aux release];
    [tw.sprites release];
    [tw.spremis release];
    tw.albedo = tw_picture(blk + sizeof(TwArtHead), (int)aw, (int)ah, 3);
    tw.emis = tw_picture(blk + at_emis, (int)(aw >> es), (int)(ah >> es), 2);
    if (a->flags & TW_A_F_AUX) {
        tw.aux = gp_texture(MTLPixelFormatRG8Uint, aw, ah, MTLTextureUsageShaderRead);
        [tw.aux replaceRegion:MTLRegionMake2D(0, 0, aw, ah) mipmapLevel:0
                    withBytes:blk + at_aux bytesPerRow:aw * 2];
    } else {
        static const uint8_t none[2];
        tw.aux = gp_texture(MTLPixelFormatRG8Uint, 1, 1, MTLTextureUsageShaderRead);
        [tw.aux replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:none bytesPerRow:2];
    }
    tw.sprites = tw_picture(blk + at_spr, (int)sw, (int)sh, 3);
    tw.spremis = tw_picture(blk + at_spremis, (int)sw, (int)sh, 3);
    [tw.arttab replaceRegion:MTLRegionMake2D(0, 0, TW_NART, 1) mipmapLevel:0
                   withBytes:a->art bytesPerRow:TW_NART * 16];
    [tw.sprtab replaceRegion:MTLRegionMake2D(0, 0, TW_NSPR, 1) mipmapLevel:0
                   withBytes:a->spr bytesPerRow:TW_NSPR * 16];
    tw.weekend = -1;                    /* the lamp profiles again */
    tw.art_gen = gen;
    tw.art_have = tw.albedo && tw.emis && tw.aux && tw.sprites && tw.spremis;
    metal_log("tower: art gen %u: atlas %llu x %llu, sprites %llu x %llu, %u entries",
              gen, aw, ah, sw, sh, a->nart);
    return tw.art_have;
}

/* A frame of the live block into the textures, if it is whole and new:
 * only the sections whose generation moved.  Whether there is a frame to
 * draw (the last one, if this one is being written) */
static bool tw_take(const uint8_t *blk)
{
    TwHead h;
    uint32_t after;

    memcpy(&h, blk, sizeof(h));
    if (h.magic != TW_MAGIC || h.version != TW_VERSION || (h.frame & 1u)) {
        return tw.have;
    }
    if (tw.have && h.frame == tw.frame) {
        return true;
    }
    /* copied out, and read again AFTER: if the box began a new frame
       meanwhile the copy is a mix of two, and the last whole one stands */
    memcpy(tw.next, blk, TW_LIVE_SIZE);
    memcpy(&after, blk + offsetof(TwHead, frame), sizeof(after));
    if (after != h.frame) {
        return tw.have;
    }
    const TwHead *n = (const TwHead *)tw.next;
    if (!tw.have || n->gen[TW_CELLS] != tw.gen[TW_CELLS]) {
        [tw.cells replaceRegion:MTLRegionMake2D(0, 0, TW_STRIDE, TW_GRID_H) mipmapLevel:0
                      withBytes:tw.next + TW_OFF_CELLS bytesPerRow:TW_STRIDE * 8];
    }
    for (int k = 0; k < 16; k++) {
        /* a chunk of 1024 rooms is eight rows of the 128-wide texture */
        if (!tw.have || n->room_gen[k] != tw.room_gen[k]) {
            [tw.rooms replaceRegion:MTLRegionMake2D(0, k * 8, 128, 8) mipmapLevel:0
                          withBytes:tw.next + TW_OFF_ROOMS + (size_t)k * 1024 * 16 bytesPerRow:128 * 16];
        }
    }
    if (!tw.have || n->gen[TW_FIX] != tw.gen[TW_FIX]) {
        [tw.fixtab replaceRegion:MTLRegionMake2D(0, 0, TW_NFIX, 1) mipmapLevel:0
                       withBytes:tw.next + TW_OFF_FIX bytesPerRow:TW_NFIX * 16];
    }
    if (!tw.have || n->gen[TW_STOPS] != tw.gen[TW_STOPS] || n->gen[TW_MOVERS] != tw.gen[TW_MOVERS]) {
        /* what moves: into the next of the ring, the one drawn from now */
        tw.ring = (tw.ring + 1) % TW_RING;
        [tw.stops[tw.ring] replaceRegion:MTLRegionMake2D(0, 0, TW_NROWS, TW_NSHAFTS) mipmapLevel:0
                               withBytes:tw.next + TW_OFF_STOPS bytesPerRow:TW_NROWS];
        uint32_t nm = n->nmovers > TW_NMOVERS ? TW_NMOVERS : n->nmovers;
        uint32_t rows = (nm + 255) / 256;
        if (rows) {
            [tw.movers[tw.ring] replaceRegion:MTLRegionMake2D(0, 0, 256, rows) mipmapLevel:0
                                    withBytes:tw.next + TW_OFF_MOVERS bytesPerRow:256 * 8];
        }
    }
    memcpy(tw.gen, n->gen, sizeof(tw.gen));
    memcpy(tw.room_gen, n->room_gen, sizeof(tw.room_gen));
    uint8_t *t = tw.live;
    tw.live = tw.next;
    tw.next = t;
    tw.frame = h.frame;
    tw.have = true;
    return true;
}

/* tw_light(): the sky ramp at an hour, linear between its two half-hour
 * columns, in float from the bytes (the twin's, as both presenters carry) */
static void tw_light(const TwArtHead *a, float hour, float out[4][4])
{
    float t = hour * 2.0f - TW_RAMP * floorf(hour * 2.0f / TW_RAMP);
    int i0 = (int)floorf(t);
    if (i0 > TW_RAMP - 1) {
        i0 = TW_RAMP - 1;
    }
    int i1 = (i0 + 1) % TW_RAMP;
    float f = t - (float)i0;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            float v0 = a->ramp[r][i0][c] / 255.0f, v1 = a->ramp[r][i1][c] / 255.0f;
            out[r][c] = v0 * (1.0f - f) + v1 * f;
        }
    }
}

static void tw_unpack8(uint32_t w, float o[4])
{
    for (int k = 0; k < 4; k++) {
        o[k] = ((w >> (8 * k)) & 255) / 255.0f;
    }
}

/* The tower, where the Wimp says the icon is and only where it can be seen */
static void surf_tower(id<MTLRenderCommandEncoder> enc, bool shot, CGSize target,
                       uint32_t xres, uint32_t yres)
{
    const MetalSurfacesView *v = &surf.view;
    double sx = target.width / MAX(xres, 1u), sy = target.height / MAX(yres, 1u);
    const MetalIcon *ic;
    const TwHead *h;
    TwData d;
    TwU u;

    ic = tw_first(&d);
    if (!ic) {
        return;
    }
    tw_setup();
    if (!tw.field || !tw.field_shot || !tw.cells || !tw.live) {
        return;
    }
    if (d.art_offset == 0xFFFFFFFFu || (uint64_t)d.art_offset + d.art_size > v->region_size
        || !tw_take_art((const uint8_t *)v->region + d.art_offset, d.art_size, d.art_gen)) {
        return;                         /* no pictures yet: nothing drawn */
    }
    if (!tw_take((const uint8_t *)v->region + d.offset)) {
        return;
    }
    h = (const TwHead *)tw.live;
    const TwArtHead *a = tw.ah;
    int want = (h->flags & TW_F_WEEKEND) ? 1 : 0;
    if (tw.weekend != want) {
        /* the lamp profiles the header's day names (design 6.5) */
        [tw.prof replaceRegion:MTLRegionMake2D(0, 0, 24, TW_PROFILES) mipmapLevel:0
                     withBytes:a->prof[want] bytesPerRow:24];
        tw.weekend = want;
    }
    memset(&u, 0, sizeof(u));
    u.box[0] = ic->x * sx, u.box[1] = ic->y * sy, u.box[2] = ic->w * sx, u.box[3] = ic->h * sy;
    float tpp = (float)a->scale * (TW_GRID_W * 4) / MAX(u.box[2], 1.0f);
    float lod = log2f(tpp);
    lod = lod < 0 ? 0 : lod > 3 ? 3 : lod;
    float hour = (float)(h->hour % 1536) / 64.0f;
    float light[4][4];
    tw_light(a, hour, light);
    u.view[0] = target.width, u.view[1] = target.height, u.view[2] = tpp, u.view[3] = lod;
    u.clock[0] = hour;
    u.clock[1] = (float)fmod(CFAbsoluteTimeGetCurrent(), 3600.0);
    u.clock[2] = h->day;
    u.clock[3] = target.height;
    memcpy(u.zen, light[0], 16);
    memcpy(u.hor, light[1], 16);
    memcpy(u.amb, light[2], 16);
    memcpy(u.misc, light[3], 16);
    u.grid[0] = TW_GRID_W, u.grid[1] = TW_GRID_H;
    u.grid[2] = h->ground_row > TW_GRID_H ? TW_GRID_H : h->ground_row;
    u.grid[3] = a->scale;
    u.counts[0] = h->nrooms > TW_NROOMS ? TW_NROOMS : h->nrooms;
    u.counts[1] = a->nart > TW_NART ? TW_NART : a->nart;
    u.counts[2] = a->nsprites > TW_NSPR ? TW_NSPR : a->nsprites;
    u.counts[3] = MIN(MIN(h->quality, 2), tw.maxq);   /* as the block asks, as far as the library goes */
    u.sizes[0] = a->atlas_w, u.sizes[1] = a->atlas_h, u.sizes[2] = a->spr_w, u.sizes[3] = a->spr_h;
    u.mode[0] = h->view, u.mode[1] = a->emit_shift, u.mode[2] = h->nfixtures, u.mode[3] = h->nmovers;
    for (int k = 0; k < 4; k++) {
        u.outline[k] = h->outline[k];
    }
    tw_unpack8(h->outline_rgba, u.ocol);
    for (int k = 0; k < 16; k++) {
        uint32_t p;
        memcpy(&p, tw.live + TW_OFF_ROOMS + (size_t)TW_NROOMS * 16 + 512 * 16 + 64 * 128 + 4 * k, 4);
        tw_unpack8(p, u.pal[k]);
    }
    if (u.counts[1] < 1 || u.counts[2] < 1 || u.counts[0] < 1) {
        return;
    }
    uint32_t nfix = h->nfixtures > TW_NFIX ? TW_NFIX : h->nfixtures;
    uint32_t nmov = h->nmovers > TW_NMOVERS ? TW_NMOVERS : h->nmovers;
    u.mode[2] = (int32_t)nfix, u.mode[3] = (int32_t)nmov;
    [enc setFragmentTexture:tw.cells atIndex:0];
    [enc setFragmentTexture:tw.rooms atIndex:1];
    [enc setFragmentTexture:tw.arttab atIndex:2];
    [enc setFragmentTexture:tw.sprtab atIndex:3];
    [enc setFragmentTexture:tw.albedo atIndex:4];
    [enc setFragmentTexture:tw.emis atIndex:5];
    [enc setFragmentTexture:tw.aux atIndex:6];
    [enc setFragmentTexture:tw.sprites atIndex:7];
    [enc setFragmentTexture:tw.spremis atIndex:8];
    [enc setFragmentTexture:tw.prof atIndex:9];
    [enc setFragmentTexture:tw.stops[tw.ring] atIndex:10];
    [enc setVertexTexture:tw.fixtab atIndex:0];
    [enc setVertexTexture:tw.sprtab atIndex:3];
    [enc setVertexTexture:tw.movers[tw.ring] atIndex:11];
    for (uint32_t r = 0; r < ic->nrects; r++) {
        const int16_t *q = v->rect[ic->rects + r];

        u.clip[0] = floor(q[0] * sx), u.clip[1] = floor(q[1] * sy);
        u.clip[2] = ceil(q[2] * sx), u.clip[3] = ceil(q[3] * sy);
        if (u.clip[2] <= u.clip[0] || u.clip[3] <= u.clip[1]) {
            continue;
        }
        [enc setRenderPipelineState:shot ? tw.field_shot : tw.field];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
        [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        /* the fixtures and the movers, instanced, under the piece's scissor */
        MTLScissorRect sc;
        double x0 = MAX(u.clip[0], 0.0), y0 = MAX(u.clip[1], 0.0);
        double x1 = MIN(u.clip[2], target.width), y1 = MIN(u.clip[3], target.height);
        if (x1 <= x0 || y1 <= y0 || (!nfix && !nmov)) {
            continue;
        }
        sc.x = (NSUInteger)x0, sc.y = (NSUInteger)y0;
        sc.width = (NSUInteger)(x1 - x0), sc.height = (NSUInteger)(y1 - y0);
        [enc setScissorRect:sc];
        if (nfix && tw.fix && tw.fix_shot) {
            [enc setRenderPipelineState:shot ? tw.fix_shot : tw.fix];
            [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:nfix];
        }
        if (nmov && tw.mov && tw.mov_shot) {
            [enc setRenderPipelineState:shot ? tw.mov_shot : tw.mov];
            [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:nmov];
        }
        sc.x = 0, sc.y = 0, sc.width = (NSUInteger)target.width, sc.height = (NSUInteger)target.height;
        [enc setScissorRect:sc];
    }
}

/* ---- a 3D scene (ROSGD_SURF_SCENE3D) -----------------------------------------
 * Antiquity's scene (0 A.D.'s engine for the BOX), drawn by the window from
 * a live block and up to eight asset pages the box keeps in backings in the
 * region (scene3d.h here is the box's rosgd/include/rosgd/scene3d.h; the
 * design is Antiquity's docs/SCENE3D.md, section 6.2 the picture and
 * section 10 this presenter).  The pages' records are uploaded as they
 * appear: a buffer pair a mesh, a texture each (BC1-3 as the Mac's GPU
 * takes them, else decoded here), skins and terrain textures kept as they
 * are.  Each frame's instances and bone matrices go into the next of a ring
 * of three textures.  The scene is drawn into a colour and depth target of
 * its own the size of the icon's box (surf_scene3d_render, before the
 * frame's main pass), then shown in the icon's visible rectangles
 * (surf_scene3d).  The terrain is the engine's splats (SCENE3D.md 6.3):
 * for the visible patches, the map's sides, then the opaque splats, then
 * the blends through the alpha map's shapes, a run of one texture a draw,
 * lit per vertex and shaded by the LOS grid; the models a draw an
 * instance, skinned in the vertex shader from the bone pool.  The library is built without
 * fast maths, and the level of detail is chosen as the CPU twin chooses it
 * (twin.rs: once a triangle for the terrain, in the vertex shader; from the
 * derivatives for the models), so that the shot can be compared with the
 * twin's picture.  ROSGD_SCENE3D_CPU_BC=1 decodes BC on the CPU as the twin
 * does, for the parity check. */

#include "scene3d.h"

#define S3_RING 3
#define S3_INST_W 2048
#define S3_BONE_W 1536

typedef struct S3U {                /* a draw's uniforms: the same in the shader */
    float vp[16];
    float sun_dir[4], sun_rgb[4], amb[4];
    float player[4], object[4], tint[4];
    float tm[4];                    /* a terrain texture's matrix: u = a x + b z, v = c x + d z */
    float tsize[4];                 /* the texture's width, height, last level */
    float am[4];                    /* the alpha map: shape stride, texture width, shapes */
    float side[4];                  /* the map's sides' colour; w the water's height (or 0) */
    int32_t mode[4];                /* shading bits (256: a blend), has a texture, alpha is 1,
                                       the instance */
    int32_t terr[4];                /* vertices a side, the mesh's bones, the patch's px, pz */
    int32_t edge[4];                /* a side: its first vertex i, j and its step */
    float vsize[4];                 /* the target's width and height */
} S3U;
_Static_assert(sizeof(S3U) == 288, "a scene draw's uniforms are 288 bytes");

typedef struct S3ShowU {
    float box[4], clip[4], target[4];
} S3ShowU;

typedef struct S3HudU {             /* the HUD's draws (scene3d.h's HUD section) */
    float size[4];                  /* the HUD's units across and down */
    int32_t mode[4];                /* has a texture */
} S3HudU;

static NSString * const S3_SRC = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct S3U {\n"
"    float4 vp[4];\n"
"    float4 sun_dir, sun_rgb, amb, player, object, tint, tm, tsize, am, side;\n"
"    int4 mode, terr, edge;\n"
"    float4 vsize;\n"
"};\n"
"struct V { float4 pos [[position]]; float2 uv; float3 n; float2 wxz; float lod [[flat]]; };\n"
"static float4 clip_of(constant S3U &u, float3 p) {\n"
"    return u.vp[0] * p.x + u.vp[1] * p.y + u.vp[2] * p.z + u.vp[3];\n"
"}\n"
"static float4 metal_z(float4 c) { return float4(c.x, c.y, (c.z + c.w) * 0.5, c.w); }\n"
"static int hraw(texture2d<ushort> h, int i, int j) { return int(h.read(uint2(i, j)).r); }\n"
"static float3 tpos(texture2d<ushort> h, int i, int j, int verts) {\n"
"    int hi = clamp(i, 0, verts - 1), hj = clamp(j, 0, verts - 1);\n"
"    return float3(float(i) * 4.0, float(h.read(uint2(hi, hj)).r) * (1.0 / 92.0), float(j) * 4.0);\n"
"}\n"
"static float3 edgev(float3 p, float3 b) {\n"
"    float3 d = p - b; float l = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);\n"
"    return l > 0.0 ? d / l : d;\n"
"}\n"
"static float3 crs(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }\n"
"/* CTerrain::CalcNormal, as SCENE3D.md 6.3 words it */\n"
"static float3 vnormal(texture2d<ushort> h, int i, int j, int verts) {\n"
"    float3 b = tpos(h, i, j, verts), z = float3(0.0);\n"
"    float3 left = i > 0 ? edgev(tpos(h, i - 1, j, verts), b) : z;\n"
"    float3 right = i < verts - 1 ? edgev(tpos(h, i + 1, j, verts), b) : z;\n"
"    float3 up = j > 0 ? edgev(tpos(h, i, j - 1, verts), b) : z;\n"
"    float3 down = j < verts - 1 ? edgev(tpos(h, i, j + 1, verts), b) : z;\n"
"    float3 n = crs(up, left) + crs(left, down) + crs(down, right) + crs(right, up);\n"
"    float l = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);\n"
"    return l > 0.00001 ? n / l : n;\n"
"}\n"
"/* the twin's level for a triangle: texels a pixel, by area, once a triangle */\n"
"static float tri_lod(constant S3U &u, float4 c0, float4 c1, float4 c2, float2 t0, float2 t1, float2 t2) {\n"
"    if (c0.w <= 0.0 || c1.w <= 0.0 || c2.w <= 0.0) return 0.0;\n"
"    float w = u.vsize.x, h = u.vsize.y;\n"
"    float x0 = (c0.x * (1.0 / c0.w) + 1.0) * 0.5 * w, y0 = (1.0 - c0.y * (1.0 / c0.w)) * 0.5 * h;\n"
"    float x1 = (c1.x * (1.0 / c1.w) + 1.0) * 0.5 * w, y1 = (1.0 - c1.y * (1.0 / c1.w)) * 0.5 * h;\n"
"    float x2 = (c2.x * (1.0 / c2.w) + 1.0) * 0.5 * w, y2 = (1.0 - c2.y * (1.0 / c2.w)) * 0.5 * h;\n"
"    float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);\n"
"    float uva = abs((t1.x - t0.x) * (t2.y - t0.y) - (t2.x - t0.x) * (t1.y - t0.y)) * u.tsize.x * u.tsize.y;\n"
"    float r = uva / abs(area);\n"
"    float l = r > 1.0 ? 0.5 * log2(r) : 0.0;\n"
"    return min(floor(l), u.tsize.z);\n"
"}\n"
"struct TV { float4 pos [[position]]; float2 wxz; float3 light; float lod [[flat]]; int2 g [[flat]]; uint sc [[flat]]; };\n"
"/* a splat: a quad over its tile, its splat by vid / 6 */\n"
"vertex TV s3_splat_vs(uint vid [[vertex_id]], constant S3U &u [[buffer(0)]],\n"
"                     const device uint *spl [[buffer(1)]], texture2d<ushort> hg [[texture(0)]]) {\n"
"    uint w = spl[vid / 6u];\n"
"    int gi = u.terr.z * 16 + int(w & 15u), gj = u.terr.w * 16 + int((w >> 4) & 15u), verts = u.terr.x;\n"
"    bool d = hraw(hg, gi, gj) + hraw(hg, gi + 1, gj + 1) < hraw(hg, gi, gj + 1) + hraw(hg, gi + 1, gj);\n"
"    /* split along (gi+1, gj)-(gi, gj+1) when d, else along (gi, gj)-(gi+1, gj+1) */\n"
"    const int2 A[6] = { int2(0,0), int2(1,0), int2(0,1), int2(1,0), int2(1,1), int2(0,1) };\n"
"    const int2 B[6] = { int2(0,0), int2(1,0), int2(1,1), int2(1,1), int2(0,1), int2(0,0) };\n"
"    uint c6 = vid % 6u, tri = c6 / 3u * 3u;\n"
"    float4 c[3]; float2 uv[3]; int2 k[3];\n"
"    for (uint q = 0; q < 3u; q++) {\n"
"        k[q] = d ? A[tri + q] : B[tri + q];\n"
"        float3 p = tpos(hg, gi + k[q].x, gj + k[q].y, verts);\n"
"        c[q] = clip_of(u, p);\n"
"        uv[q] = float2(u.tm.x * p.x + u.tm.y * p.z, u.tm.z * p.x + u.tm.w * p.z);\n"
"    }\n"
"    uint me = c6 - tri;\n"
"    int ci = gi + k[me].x, cj = gj + k[me].y;\n"
"    float3 n = vnormal(hg, ci, cj, verts);\n"
"    float ndl = clamp(-(n.x * u.sun_dir.x + n.y * u.sun_dir.y + n.z * u.sun_dir.z), 0.0, 1.0);\n"
"    TV o;\n"
"    o.pos = metal_z(c[me]);\n"
"    o.wxz = float2(float(ci) * 4.0, float(cj) * 4.0);\n"
"    o.light = u.sun_rgb.rgb * ndl;\n"
"    o.lod = u.mode.y != 0 ? tri_lod(u, c[0], c[1], c[2], uv[0], uv[1], uv[2]) : 0.0;\n"
"    o.g = int2(gi, gj);\n"
"    o.sc = (w >> 8) & 127u;\n"
"    return o;\n"
"}\n"
"fragment float4 s3_splat_fs(TV in [[stage_in]], constant S3U &u [[buffer(0)]],\n"
"                            texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]],\n"
"                            texture2d<float> los [[texture(1)]], texture2d<float> am [[texture(2)]],\n"
"                            sampler asmp [[sampler(1)]]) {\n"
"    constexpr sampler lin(coord::normalized, filter::linear, address::clamp_to_edge);\n"
"    float x = in.wxz.x, z = in.wxz.y;\n"
"    float2 uv = float2(u.tm.x * x + u.tm.y * z, u.tm.z * x + u.tm.w * z);\n"
"    float4 base = u.mode.y != 0 ? tex.sample(smp, uv, level(in.lod)) : float4(0.5, 0.5, 0.5, 1.0);\n"
"    float f = los.sample(lin, float2((x / 4.0 + 0.5) / 520.0, (z / 4.0 + 0.5) / 513.0)).r;\n"
"    f = clamp(f - 0.03, 0.0, 1.0) / (1.0 - 0.03);\n"
"    float3 c = clamp(base.rgb * (in.light + u.amb.rgb) * f, 0.0, 1.0);\n"
"    if ((u.mode.x & 256) == 0) return float4(c, 1.0);\n"
"    uint shape = in.sc & 15u, code = (in.sc >> 4) & 7u;\n"
"    float s = x / 4.0 - float(in.g.x), t = z / 4.0 - float(in.g.y);\n"
"    float a = (code & 1u) != 0u ? t : s, b = (code & 1u) != 0u ? s : t;\n"
"    if ((code & 2u) != 0u) a = 1.0 - a;\n"
"    if ((code & 4u) != 0u) b = 1.0 - b;\n"
"    float u0 = float(shape * uint(u.am.x) + 2u) / u.am.y, u1 = float((shape + 1u) * uint(u.am.x) - 2u) / u.am.y;\n"
"    float al = 1.0 - am.sample(asmp, float2(u0 + a * (u1 - u0), b), level(0.0)).r;\n"
"    return float4(c, al);\n"
"}\n"
"struct SDV { float4 pos [[position]]; };\n"
"/* the map's sides: from each edge vertex (or the water above it) down to 0 */\n"
"vertex SDV s3_side_vs(uint vid [[vertex_id]], constant S3U &u [[buffer(0)]], texture2d<ushort> hg [[texture(0)]]) {\n"
"    int k = int(vid / 6u), c = int(vid % 6u);\n"
"    int2 a = int2(u.edge.x + k * u.edge.z, u.edge.y + k * u.edge.w), b = a + u.edge.zw;\n"
"    const int pick[6] = { 0, 1, 3, 0, 3, 2 };   /* a0 a1 b1, a0 b1 b0 */\n"
"    int2 v = pick[c] >= 2 ? b : a;\n"
"    bool top = (pick[c] & 1) != 0;\n"
"    float y = top ? max(float(hg.read(uint2(v)).r) * (1.0 / 92.0), u.side.w) : 0.0;\n"
"    SDV o;\n"
"    o.pos = metal_z(clip_of(u, float3(float(v.x) * 4.0, y, float(v.y) * 4.0)));\n"
"    return o;\n"
"}\n"
"fragment float4 s3_side_fs(SDV in [[stage_in]], constant S3U &u [[buffer(0)]]) {\n"
"    return float4(u.side.rgb, 1.0);\n"
"}\n"
"struct MV {\n"
"    float3 pos [[attribute(0)]];\n"
"    float3 nrm [[attribute(1)]];\n"
"    float2 uv [[attribute(2)]];\n"
"    uchar4 bone [[attribute(3)]];\n"
"    uchar4 wt [[attribute(4)]];\n"
"};\n"
"static float3 row3(float4 r0, float4 r1, float4 r2, float3 p, float w) {\n"
"    return float3(r0.x * p.x + r0.y * p.y + r0.z * p.z + r0.w * w,\n"
"                  r1.x * p.x + r1.y * p.y + r1.z * p.z + r1.w * w,\n"
"                  r2.x * p.x + r2.y * p.y + r2.z * p.z + r2.w * w);\n"
"}\n"
"static float3 nz(float3 v) { float l = sqrt(v.x * v.x + v.y * v.y + v.z * v.z); return l > 0.0 ? v / l : v; }\n"
"vertex V s3_model_vs(MV v [[stage_in]], constant S3U &u [[buffer(0)]],\n"
"                     texture2d<uint> inst [[texture(1)]], texture2d<float> bones [[texture(2)]]) {\n"
"    uint at = uint(u.mode.w) * 4u;\n"
"    uint2 ip = uint2(at % 2048u, at / 2048u);\n"
"    float4 m0 = as_type<float4>(inst.read(ip)), m1 = as_type<float4>(inst.read(ip + uint2(1, 0)));\n"
"    float4 m2 = as_type<float4>(inst.read(ip + uint2(2, 0)));\n"
"    uint first = inst.read(ip + uint2(3, 0)).y;\n"
"    uint nb = uint(u.terr.y);\n"
"    float3 p = float3(0.0), n = float3(0.0);\n"
"    if (first != 0xFFFFFFFFu && first + nb + 1u <= 49152u) {\n"
"        for (uint q = 0; q < 4u; q++) {\n"
"            uint b = uint(v.bone[q]);\n"
"            if (b == 255u || b > nb) break;\n"
"            uint x = (first + b) * 3u;\n"
"            uint2 bp = uint2(x % 1536u, x / 1536u);\n"
"            float4 r0 = bones.read(bp), r1 = bones.read(bp + uint2(1, 0)), r2 = bones.read(bp + uint2(2, 0));\n"
"            float w = float(v.wt[q]) / 255.0;\n"
"            p += row3(r0, r1, r2, v.pos, 1.0) * w;\n"
"            n += row3(r0, r1, r2, v.nrm, 0.0) * w;\n"
"        }\n"
"        n = nz(n);\n"
"    } else {\n"
"        p = row3(m0, m1, m2, v.pos, 1.0);\n"
"        n = nz(row3(m0, m1, m2, v.nrm, 0.0));\n"
"    }\n"
"    V o;\n"
"    o.pos = metal_z(clip_of(u, p));\n"
"    o.uv = v.uv;\n"
"    o.n = n;\n"
"    o.wxz = float2(p.x, p.z);\n"
"    o.lod = -1.0;\n"
"    return o;\n"
"}\n"
"fragment float4 s3_fs(V in [[stage_in]], constant S3U &u [[buffer(0)]],\n"
"                      texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]],\n"
"                      texture2d<float> los [[texture(1)]]) {\n"
"    float4 base = float4(0.5, 0.5, 0.5, 1.0);\n"
"    if (u.mode.y != 0) {\n"
"        float2 t = in.uv * u.tsize.xy;\n"
"        float2 dx = dfdx(t), dy = dfdy(t);\n"
"        float l = in.lod;\n"
"        if (l < 0.0) {\n"
"            float r = abs(dx.x * dy.y - dx.y * dy.x);\n"
"            l = min(floor(r > 1.0 ? 0.5 * log2(r) : 0.0), u.tsize.z);\n"
"        }\n"
"        base = tex.sample(smp, in.uv, level(l));\n"
"        if (u.mode.z != 0) base.a = 1.0;\n"
"    }\n"
"    if ((u.mode.x & 4) != 0 && base.a < 0.5) discard_fragment();\n"
"    float3 rgb = base.rgb;\n"
"    if ((u.mode.x & 1) != 0) rgb = rgb * (u.player.rgb + (1.0 - u.player.rgb) * base.a);\n"
"    else if ((u.mode.x & 2) != 0) rgb = rgb * (u.object.rgb + (1.0 - u.object.rgb) * base.a);\n"
"    float3 n = nz(in.n);\n"
"    float ndl = max(-(n.x * u.sun_dir.x + n.y * u.sun_dir.y + n.z * u.sun_dir.z), 0.0);\n"
"    float3 light = u.sun_rgb.rgb * ndl + u.amb.rgb;\n"
"    /* the fog of war, as on the terrain, unless the instance ignores it (mode 4096) */\n"
"    float f = 1.0;\n"
"    if ((u.mode.x & 4096) == 0) {\n"
"        constexpr sampler lin(coord::normalized, filter::linear, address::clamp_to_edge);\n"
"        f = los.sample(lin, float2((in.wxz.x / 4.0 + 0.5) / 520.0, (in.wxz.y / 4.0 + 0.5) / 513.0)).r;\n"
"        f = clamp(f - 0.03, 0.0, 1.0) / (1.0 - 0.03);\n"
"    }\n"
"    return float4(clamp(rgb * light * u.tint.rgb * f, 0.0, 1.0), 1.0);\n"
"}\n"
"struct ShowU { float4 box, clip, target; };\n"
"struct SV { float4 pos [[position]]; };\n"
"vertex SV s3_show_vs(uint vid [[vertex_id]], constant ShowU &u [[buffer(0)]]) {\n"
"    float2 c = float2((vid & 1u) ? u.clip.z : u.clip.x, (vid & 2u) ? u.clip.w : u.clip.y);\n"
"    SV o;\n"
"    o.pos = float4(c.x / u.target.x * 2.0 - 1.0, 1.0 - c.y / u.target.y * 2.0, 0.0, 1.0);\n"
"    return o;\n"
"}\n"
"fragment float4 s3_show_fs(SV in [[stage_in]], constant ShowU &u [[buffer(0)]],\n"
"                           texture2d<float> pic [[texture(0)]]) {\n"
"    int2 p = int2(floor(in.pos.xy - floor(u.box.xy)));\n"
"    p = clamp(p, int2(0), int2(int(pic.get_width()) - 1, int(pic.get_height()) - 1));\n"
"    return pic.read(uint2(p));\n"
"}\n"
"/* the HUD: a quad (48 bytes) six vertices, over the scene in list order */\n"
"struct HQ { float4 r; float4 uv; uint tf; uint c1; uint c2; uint pad; };\n"
"struct HudU { float4 size; int4 mode; };\n"
"struct HV { float4 pos [[position]]; float2 uv; float4 col; uint flags [[flat]]; };\n"
"vertex HV s3_hud_vs(uint vid [[vertex_id]], constant HudU &u [[buffer(0)]], const device HQ *q [[buffer(1)]]) {\n"
"    HQ x = q[vid / 6u];\n"
"    uint c = vid % 6u;\n"
"    float2 t = float2((c == 1u || c == 3u || c == 4u) ? 1.0 : 0.0, (c == 2u || c == 4u || c == 5u) ? 1.0 : 0.0);\n"
"    float2 p = mix(x.r.xy, x.r.zw, t);\n"
"    HV o;\n"
"    o.pos = float4(p.x / u.size.x * 2.0 - 1.0, 1.0 - p.y / u.size.y * 2.0, 0.0, 1.0);\n"
"    o.uv = mix(x.uv.xy, x.uv.zw, t);\n"
"    o.flags = x.tf >> 16;\n"
"    float4 a = unpack_unorm4x8_to_float(x.c1), b = unpack_unorm4x8_to_float(x.c2);\n"
"    o.col = (o.flags & 4u) != 0u ? mix(a, b, t.y) : a;\n"
"    return o;\n"
"}\n"
"fragment float4 s3_hud_fs(HV in [[stage_in]], constant HudU &u [[buffer(0)]],\n"
"                          texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
"    float4 c = in.col;\n"
"    if (u.mode.x != 0) {\n"
"        float4 x = t.sample(s, in.uv);\n"
"        if ((in.flags & 1u) != 0u) x = float4(1.0, 1.0, 1.0, x.r);\n"
"        else if ((in.flags & 2u) != 0u) { float l = (x.r * 77.0 + x.g * 150.0 + x.b * 29.0) / 256.0; x = float4(l, l, l, x.a); }\n"
"        c *= x;\n"
"    }\n"
"    return c;\n"
"}\n";

typedef struct S3Mesh {
    id<MTLBuffer> vb, ib;
    uint32_t nidx, nbones;
} S3Mesh;

typedef struct S3Tex {
    id<MTLTexture> t;
    uint16_t w, h;
    uint8_t levels, wrap, opaque;   /* opaque: BC1, whose alpha is 1 whatever the GPU says */
} S3Tex;

static struct {
    bool tried, ok, bc, cpu_bc;
    id<MTLRenderPipelineState> splat, blend, side, model, show, show_shot, hud;
    id<MTLDepthStencilState> depth, depth_blend, depth_none;
    id<MTLTexture> canvas;          /* the HUD's canvas (the minimap) */
    id<MTLBuffer> hudq[S3_RING];    /* the HUD's quads, a ring as the instances */
    id<MTLSamplerState> repeat, clamp;
    id<MTLTexture> heights, los, inst[S3_RING], bones[S3_RING];
    id<MTLTexture> color, zbuf;     /* the scene's picture and its depth */
    id<MTLBuffer> splats;           /* the terrain's splats, as the block has them */
    struct { uint16_t tex; uint32_t stride, shapes; bool have; } am;    /* the alpha map */
    int ring, hud_ring;
    bool have, drawn_any;
    uint32_t frame, epoch, gen[S3_NSECT], patch_gen[ROS_S3_PATCHES];
    uint32_t page_done[ROS_S3_PAGES];       /* records uploaded from each page */
    uint32_t page_epoch[ROS_S3_PAGES];
    struct ros_scene3d *cur;        /* the last whole frame: its header, terrain list, heights,
                                       tiles, instances and bones */
    uint8_t *hdr_next;              /* a frame's first 2,048 bytes, while it is copied */
    S3Mesh mesh[ROS_S3_MESH_IDS];
    S3Tex tex[ROS_S3_TEX_IDS];
    uint32_t skin[ROS_S3_SKIN_IDS][4];      /* the record's p[0..3]; p[3] the object colour */
    bool skin_have[ROS_S3_SKIN_IDS];
    struct { uint16_t diffuse; uint32_t base; float m[4]; bool have; } terr[ROS_S3_TERRAIN_IDS];
    uint32_t drawn_frame, drawn_w, drawn_h, drawn_assets, assets;
    uint64_t shown;
} s3;

static id<MTLRenderPipelineState> s3_pipe(id<MTLLibrary> lib, NSString *vs, NSString *fs,
                                          MTLPixelFormat f, bool depth, MTLVertexDescriptor *vd, bool blend)
{
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    NSError *err = nil;
    id<MTLRenderPipelineState> p;

    pd.vertexFunction = [[lib newFunctionWithName:vs] autorelease];
    pd.fragmentFunction = [[lib newFunctionWithName:fs] autorelease];
    pd.colorAttachments[0].pixelFormat = f;
    if (depth) {
        pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    }
    if (vd) {
        pd.vertexDescriptor = vd;
    }
    if (blend) {
        /* a blend splat over what is there: source alpha, one less it (the
         * target's own alpha kept) */
        pd.colorAttachments[0].blendingEnabled = YES;
        pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorZero;
        pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
    }
    if (blend && [fs isEqualToString:@"s3_hud_fs"]) {
        /* the HUD: straight alpha over the scene, as the twin blends it */
        pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    }
    p = [m.device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!p) {
        metal_log("scene3d %s: %s", [fs UTF8String], [[err localizedDescription] UTF8String]);
    }
    [pd release];
    return p;
}

static void s3_setup(void)
{
    NSError *err = nil;
    MTLCompileOptions *opts;
    id<MTLLibrary> lib;

    if (s3.tried) {
        return;
    }
    s3.tried = true;
    s3.cur = calloc(1, sizeof(struct ros_scene3d));
    s3.hdr_next = calloc(1, 2048);
    if (!s3.cur || !s3.hdr_next) {
        return;
    }
    opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    lib = [m.device newLibraryWithSource:S3_SRC options:opts error:&err];
    [opts release];
    if (!lib) {
        metal_log("scene3d: the program will not build: %s", [[err localizedDescription] UTF8String]);
        return;
    }
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    vd.attributes[0].format = MTLVertexFormatFloat3, vd.attributes[0].offset = 0;
    vd.attributes[1].format = MTLVertexFormatFloat3, vd.attributes[1].offset = 12;
    vd.attributes[2].format = MTLVertexFormatFloat2, vd.attributes[2].offset = 24;
    vd.attributes[3].format = MTLVertexFormatUChar4, vd.attributes[3].offset = 40;
    vd.attributes[4].format = MTLVertexFormatUChar4, vd.attributes[4].offset = 44;
    for (int k = 0; k < 5; k++) {
        vd.attributes[k].bufferIndex = 1;
    }
    vd.layouts[1].stride = sizeof(struct ros_s3_vertex);
    s3.splat = s3_pipe(lib, @"s3_splat_vs", @"s3_splat_fs", MTLPixelFormatRGBA8Unorm, true, nil, false);
    s3.blend = s3_pipe(lib, @"s3_splat_vs", @"s3_splat_fs", MTLPixelFormatRGBA8Unorm, true, nil, true);
    s3.side = s3_pipe(lib, @"s3_side_vs", @"s3_side_fs", MTLPixelFormatRGBA8Unorm, true, nil, false);
    s3.model = s3_pipe(lib, @"s3_model_vs", @"s3_fs", MTLPixelFormatRGBA8Unorm, true, vd, false);
    s3.show = s3_pipe(lib, @"s3_show_vs", @"s3_show_fs", MTLPixelFormatBGRA8Unorm, false, nil, false);
    s3.show_shot = s3_pipe(lib, @"s3_show_vs", @"s3_show_fs", MTLPixelFormatRGBA8Unorm, false, nil, false);
    s3.hud = s3_pipe(lib, @"s3_hud_vs", @"s3_hud_fs", MTLPixelFormatRGBA8Unorm, true, nil, true);
    [lib release];
    MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor alloc] init];
    dd.depthCompareFunction = MTLCompareFunctionLess;
    dd.depthWriteEnabled = YES;
    s3.depth = [m.device newDepthStencilStateWithDescriptor:dd];
    dd.depthCompareFunction = MTLCompareFunctionLessEqual;     /* the blends: on the opaque, unwritten */
    dd.depthWriteEnabled = NO;
    s3.depth_blend = [m.device newDepthStencilStateWithDescriptor:dd];
    dd.depthCompareFunction = MTLCompareFunctionAlways;        /* the HUD: over everything */
    dd.depthWriteEnabled = NO;
    s3.depth_none = [m.device newDepthStencilStateWithDescriptor:dd];
    [dd release];
    for (int k = 0; k < 2; k++) {
        MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
        sd.minFilter = MTLSamplerMinMagFilterLinear;
        sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.mipFilter = MTLSamplerMipFilterNearest;
        sd.sAddressMode = sd.tAddressMode = k ? MTLSamplerAddressModeClampToEdge : MTLSamplerAddressModeRepeat;
        if (k) {
            s3.clamp = [m.device newSamplerStateWithDescriptor:sd];
        } else {
            s3.repeat = [m.device newSamplerStateWithDescriptor:sd];
        }
        [sd release];
    }
    s3.heights = gp_texture(MTLPixelFormatR16Uint, ROS_S3_HSTRIDE, ROS_S3_VERTS, MTLTextureUsageShaderRead);
    s3.los = gp_texture(MTLPixelFormatR8Unorm, ROS_S3_HSTRIDE, ROS_S3_VERTS, MTLTextureUsageShaderRead);
    s3.canvas = gp_texture(MTLPixelFormatRGBA8Unorm, ROS_S3_CANVAS, ROS_S3_CANVAS, MTLTextureUsageShaderRead);
    for (int k = 0; k < S3_RING; k++) {
        s3.inst[k] = gp_texture(MTLPixelFormatRGBA32Uint, S3_INST_W, ROS_S3_INSTANCES * 4 / S3_INST_W,
                                MTLTextureUsageShaderRead);
        s3.bones[k] = gp_texture(MTLPixelFormatRGBA32Float, S3_BONE_W, ROS_S3_BONES * 3 / S3_BONE_W,
                                 MTLTextureUsageShaderRead);
    }
    if (@available(macOS 11.0, *)) {
        s3.bc = [m.device supportsBCTextureCompression];
    } else {
        s3.bc = true;
    }
    const char *e = getenv("ROSGD_SCENE3D_CPU_BC");
    s3.cpu_bc = e && *e == '1';
    s3.ok = s3.splat && s3.blend && s3.side && s3.model && s3.show && s3.show_shot && s3.depth
            && s3.depth_blend && s3.heights && s3.los && s3.hud && s3.depth_none && s3.canvas;
    metal_log("scene3d: the classic program; BC textures %s", !s3.bc ? "decoded here (the GPU has none)"
              : s3.cpu_bc ? "decoded here (ROSGD_SCENE3D_CPU_BC)" : "on the GPU");
}

/* The first visible scene, and its live block, whole inside the region */
static const MetalIcon *s3_first(struct ros_surf_scene3d *d)
{
    const MetalSurfacesView *v = &surf.view;

    if (!surf.on || !v->nicons || !v->region) {
        return NULL;
    }
    for (uint32_t i = 0; i < v->nicons; i++) {
        const MetalIcon *ic = &v->icon[i];

        if (ic->kind != ROSGD_SURF_SCENE3D || !ic->nrects) {
            continue;
        }
        memcpy(d, ic->data, sizeof(*d));
        /* a version 1 broker's block ends at the LOS grid (no HUD) */
        if (d->size < ROS_S3_V1_SIZE || d->offset == 0xFFFFFFFFu
            || (uint64_t)d->offset + d->size > v->region_size) {
            continue;
        }
        return ic;
    }
    return NULL;
}

/* Every GPU object of the last asset set let go (a new epoch: another map) */
static void s3_drop(void)
{
    for (int i = 0; i < ROS_S3_MESH_IDS; i++) {
        [s3.mesh[i].vb release];
        [s3.mesh[i].ib release];
    }
    for (int i = 0; i < ROS_S3_TEX_IDS; i++) {
        [s3.tex[i].t release];
    }
    memset(s3.mesh, 0, sizeof(s3.mesh));
    memset(s3.tex, 0, sizeof(s3.tex));
    memset(s3.skin_have, 0, sizeof(s3.skin_have));
    memset(s3.terr, 0, sizeof(s3.terr));
    memset(s3.page_done, 0, sizeof(s3.page_done));
    memset(&s3.am, 0, sizeof(s3.am));
    s3.assets++;
}

static uint32_t s3_unpack5(uint32_t c, int below, int bits)
{
    uint32_t f = (c >> below) & ((1u << bits) - 1u);
    return (f << (8 - bits)) | (f >> (bits - (8 - bits)));
}

/* A 4 x 4 block of BC1 (fmt 0 or 1), BC2 (2) or BC3 (3) to RGBA, as the
 * twin decodes it (Antiquity's antiq-assets dds.rs, decode_block) */
static void s3_bc_block(int fmt, const uint8_t *b, uint8_t px[16][4])
{
    const uint8_t *cb = fmt >= 2 ? b + 8 : b;
    uint32_t rc0 = cb[0] | cb[1] << 8, rc1 = cb[2] | cb[3] << 8;
    uint32_t sel = cb[4] | cb[5] << 8 | cb[6] << 16 | (uint32_t)cb[7] << 24;
    uint32_t c[4][4];
    uint32_t rc[2] = { rc0, rc1 };
    for (int i = 0; i < 2; i++) {
        c[i][0] = s3_unpack5(rc[i], 11, 5), c[i][1] = s3_unpack5(rc[i], 5, 6);
        c[i][2] = s3_unpack5(rc[i], 0, 5), c[i][3] = 255;
    }
    if (fmt <= 1 && rc0 <= rc1) {
        for (int k = 0; k < 3; k++) {
            c[2][k] = (c[0][k] + c[1][k]) / 2;
        }
        c[2][3] = 255;
        c[3][0] = c[3][1] = c[3][2] = 0, c[3][3] = fmt == 1 ? 0 : 255;
    } else {
        for (int k = 0; k < 3; k++) {
            c[2][k] = (c[0][k] * 2 + c[1][k] + 1) / 3;
            c[3][k] = (c[1][k] * 2 + c[0][k] + 1) / 3;
        }
        c[2][3] = c[3][3] = 255;
    }
    uint64_t abits = 0;
    for (int k = 0; k < 8; k++) {
        abits |= (uint64_t)b[k] << (8 * k);
    }
    uint32_t a5[8] = { 0 };
    if (fmt == 3) {
        uint32_t a0 = b[0], a1 = b[1];
        if (a0 <= a1) {
            uint32_t t[8] = { a0, a1, (4 * a0 + a1 + 2) / 5, (3 * a0 + 2 * a1 + 2) / 5,
                              (2 * a0 + 3 * a1 + 2) / 5, (a0 + 4 * a1 + 2) / 5, 0, 255 };
            memcpy(a5, t, sizeof(a5));
        } else {
            uint32_t t[8] = { a0, a1, (6 * a0 + a1 + 3) / 7, (5 * a0 + 2 * a1 + 3) / 7,
                              (4 * a0 + 3 * a1 + 3) / 7, (3 * a0 + 4 * a1 + 3) / 7,
                              (2 * a0 + 5 * a1 + 3) / 7, (a0 + 6 * a1 + 3) / 7 };
            memcpy(a5, t, sizeof(a5));
        }
    }
    for (int i = 0; i < 16; i++) {
        uint32_t cs = (sel >> (2 * i)) & 3u, a;
        if (fmt == 2) {
            a = (uint32_t)((abits >> (4 * i)) & 0xFu);
            a |= a << 4;
        } else if (fmt == 3) {
            a = a5[((abits >> 16) >> (3 * i)) & 7u];
        } else if (fmt == 0) {
            a = 255;
        } else {
            a = c[cs][3];
        }
        px[i][0] = (uint8_t)c[cs][0], px[i][1] = (uint8_t)c[cs][1];
        px[i][2] = (uint8_t)c[cs][2], px[i][3] = (uint8_t)a;
    }
}

static size_t s3_level_size(uint32_t f, uint32_t w, uint32_t h)
{
    size_t bw = (w + 3) / 4, bh = (h + 3) / 4;
    switch (f) {
    case S3_TF_BC1: case S3_TF_BC1A: return bw * bh * 8;
    case S3_TF_BC2: case S3_TF_BC3: return bw * bh * 16;
    case S3_TF_RGBA8: return (size_t)w * h * 4;
    default: return (size_t)w * h;
    }
}

/* A TEXTURE record: every level, as the GPU takes them where it can */
static void s3_texture(const struct ros_s3_record *r, const uint8_t *data)
{
    uint32_t w = r->p[0] & 0xFFFFu, h = r->p[0] >> 16;
    uint32_t f = r->p[1] & 0xFFu, n = (r->p[1] >> 8) & 0xFFu;
    if (!r->id || r->id >= ROS_S3_TEX_IDS || !w || !h || w > 16384 || h > 16384 || f >= S3_TF_N) {
        return;
    }
    if (n < 1) {
        n = 1;
    }
    size_t need = 0;
    uint32_t lw = w, lh = h, levels = 0;
    for (uint32_t k = 0; k < n && k < 15; k++, levels++) {
        need += s3_level_size(f, lw, lh);
        lw = lw > 1 ? lw / 2 : 1, lh = lh > 1 ? lh / 2 : 1;
    }
    if (need > r->size) {
        return;                         /* short: not drawn */
    }
    bool bcf = f <= S3_TF_BC3;
    bool gpu = bcf && s3.bc && !s3.cpu_bc && !(w & 3) && !(h & 3);
    MTLPixelFormat pf = !gpu ? MTLPixelFormatRGBA8Unorm
                      : f == S3_TF_BC2 ? MTLPixelFormatBC2_RGBA
                      : f == S3_TF_BC3 ? MTLPixelFormatBC3_RGBA : MTLPixelFormatBC1_RGBA;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:w
                                                                                 height:h mipmapped:NO];
    td.mipmapLevelCount = levels;
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [m.device newTextureWithDescriptor:td];
    if (!t) {
        return;
    }
    const uint8_t *src = data;
    uint8_t *rgba = NULL;
    lw = w, lh = h;
    for (uint32_t k = 0; k < levels; k++) {
        size_t ls = s3_level_size(f, lw, lh);
        MTLRegion rg = MTLRegionMake2D(0, 0, lw, lh);
        if (gpu) {
            [t replaceRegion:rg mipmapLevel:k withBytes:src
                 bytesPerRow:((lw + 3) / 4) * (f >= S3_TF_BC2 ? 16 : 8)];
        } else {
            rgba = realloc(rgba, (size_t)lw * lh * 4);
            if (!rgba) {
                break;
            }
            if (bcf) {
                uint32_t bw = (lw + 3) / 4, bh = (lh + 3) / 4, bs = f >= S3_TF_BC2 ? 16 : 8;
                uint8_t px[16][4];
                for (uint32_t by = 0; by < bh; by++) {
                    for (uint32_t bx = 0; bx < bw; bx++) {
                        s3_bc_block((int)f, src + ((size_t)by * bw + bx) * bs, px);
                        for (uint32_t y = 0; y < 4; y++) {
                            for (uint32_t x = 0; x < 4; x++) {
                                uint32_t gx = bx * 4 + x, gy = by * 4 + y;
                                if (gx < lw && gy < lh) {
                                    memcpy(rgba + ((size_t)gy * lw + gx) * 4, px[y * 4 + x], 4);
                                }
                            }
                        }
                    }
                }
            } else if (f == S3_TF_RGBA8) {
                memcpy(rgba, src, (size_t)lw * lh * 4);
            } else {
                for (size_t i = 0; i < (size_t)lw * lh; i++) {      /* R8: as the twin, a a a a */
                    memset(rgba + i * 4, src[i], 4);
                }
            }
            [t replaceRegion:rg mipmapLevel:k withBytes:rgba bytesPerRow:lw * 4];
        }
        src += ls;
        lw = lw > 1 ? lw / 2 : 1, lh = lh > 1 ? lh / 2 : 1;
    }
    free(rgba);
    S3Tex *x = &s3.tex[r->id];
    [x->t release];
    x->t = t;
    x->w = (uint16_t)w, x->h = (uint16_t)h, x->levels = (uint8_t)levels;
    x->wrap = (r->p[2] & ROS_S3_T_WRAP) != 0;
    x->opaque = f == S3_TF_BC1;
}

/* A MESH record: its vertices and its indices, each in a buffer of its own */
static void s3_mesh(const struct ros_s3_record *r, const uint8_t *data)
{
    uint32_t nv = r->p[0], ni = r->p[1];
    size_t vbytes = (size_t)nv * sizeof(struct ros_s3_vertex);
    if (!r->id || r->id >= ROS_S3_MESH_IDS || !nv || !ni || nv > 65536 || ni % 3
        || vbytes + (size_t)ni * 2 > r->size) {
        return;
    }
    /* the indices kept inside the vertices: a bad one is made 0 */
    uint16_t *idx = malloc((size_t)ni * 2);
    if (!idx) {
        return;
    }
    memcpy(idx, data + vbytes, (size_t)ni * 2);
    for (uint32_t i = 0; i < ni; i++) {
        if (idx[i] >= nv) {
            idx[i] = 0;
        }
    }
    S3Mesh *x = &s3.mesh[r->id];
    [x->vb release];
    [x->ib release];
    x->vb = [m.device newBufferWithBytes:data length:vbytes options:MTLResourceStorageModeShared];
    x->ib = [m.device newBufferWithBytes:idx length:(size_t)ni * 2 options:MTLResourceStorageModeShared];
    free(idx);
    x->nidx = ni;
    x->nbones = r->p[2] > 254 ? 254 : r->p[2];
}

/* The pages' new records, as far as the live block says each page goes */
static void s3_assets(const struct ros_scene3d *h)
{
    const MetalSurfacesView *v = &surf.view;
    uint32_t np = h->npages > ROS_S3_PAGES ? ROS_S3_PAGES : h->npages;

    for (uint32_t i = 0; i < np; i++) {
        const struct ros_s3_page_ref *pr = &h->page[i];
        if (!pr->backing || pr->offset == 0xFFFFFFFFu || pr->size < ROS_S3_PAGE_DATA
            || pr->size > ROS_S3_PAGE_MAX || (uint64_t)pr->offset + pr->size > v->region_size) {
            continue;
        }
        const uint8_t *base = (const uint8_t *)v->region + pr->offset;
        const struct ros_s3_page *pg = (const struct ros_s3_page *)base;
        if (pg->magic != ROS_S3_PAGE_MAGIC || pg->version != ROS_S3_PAGE_VERSION || pg->epoch != h->epoch
            || pg->index != i) {
            continue;
        }
        if (s3.page_epoch[i] != h->epoch) {
            s3.page_epoch[i] = h->epoch;
            s3.page_done[i] = 0;
        }
        uint32_t nrec = pg->nrecords > ROS_S3_RECORDS ? ROS_S3_RECORDS : pg->nrecords;
        while (s3.page_done[i] < nrec) {
            struct ros_s3_record r;
            memcpy(&r, &pg->rec[s3.page_done[i]], sizeof(r));
            if (r.offset < ROS_S3_PAGE_DATA || (uint64_t)r.offset + r.size > pr->size) {
                break;                  /* not there yet in what the live block names */
            }
            const uint8_t *d = base + r.offset;
            switch (r.kind) {
            case S3_R_MESH:
                s3_mesh(&r, d);
                break;
            case S3_R_TEXTURE:
                s3_texture(&r, d);
                break;
            case S3_R_SKIN:
                if (r.id && r.id < ROS_S3_SKIN_IDS) {
                    memcpy(s3.skin[r.id], r.p, sizeof(s3.skin[r.id]));
                    s3.skin_have[r.id] = true;
                }
                break;
            case S3_R_ALPHAMAP:
                /* the twin's: the record with the least texture id */
                if (r.p[0] && r.p[0] < ROS_S3_TEX_IDS && r.p[2] && (!s3.am.have || r.p[0] < s3.am.tex)) {
                    s3.am.tex = (uint16_t)r.p[0], s3.am.stride = r.p[2], s3.am.shapes = r.p[3];
                    s3.am.have = true;
                }
                break;
            case S3_R_TERRAIN:
                if (r.id && r.id < ROS_S3_TERRAIN_IDS && r.size >= 16) {
                    s3.terr[r.id].diffuse = (uint16_t)r.p[0];
                    s3.terr[r.id].base = r.p[2];
                    memcpy(s3.terr[r.id].m, d, 16);
                    s3.terr[r.id].have = true;
                }
                break;
            default:
                break;
            }
            s3.page_done[i]++;
            s3.assets++;
        }
    }
}

/* The live block under its seqlock: the header, the terrain when its gen
 * moved, the instances and bones; their textures into the ring's next.
 * Returns whether there is a whole frame to draw. */
static bool s3_take(const uint8_t *blk, uint32_t size)
{
    const struct ros_scene3d *b = (const struct ros_scene3d *)blk;
    uint32_t f0;

    memcpy(&f0, blk + offsetof(struct ros_scene3d, frame), 4);
    if ((f0 & 1u) || b->magic != ROS_S3_MAGIC || b->version < ROS_S3_VERSION_MIN || b->version > ROS_S3_VERSION) {
        return s3.have;
    }
    if (s3.have && f0 == s3.frame) {
        return true;
    }
    memcpy(s3.hdr_next, blk, 2048);
    const struct ros_scene3d *n = (const struct ros_scene3d *)s3.hdr_next;
    uint32_t ni = n->ninstances > ROS_S3_INSTANCES ? ROS_S3_INSTANCES : n->ninstances;
    uint32_t nb = n->nbones > ROS_S3_BONES ? ROS_S3_BONES : n->nbones;
    bool epoch = !s3.have || n->epoch != s3.epoch;
    bool terrain = epoch || n->gen[S3_TERRAIN] != s3.gen[S3_TERRAIN];
    bool splats = epoch || n->gen[S3_SPLATS] != s3.gen[S3_SPLATS];
    bool los = epoch || n->gen[S3_LOS] != s3.gen[S3_LOS];
    bool v2 = n->version >= 2 && size >= sizeof(struct ros_scene3d);  /* else no HUD: nothing read
                                                                          past the LOS grid */
    bool hud = v2 && (epoch || n->gen[S3_HUD] != s3.gen[S3_HUD]);
    bool canvas = v2 && (epoch || n->gen[S3_CANVAS] != s3.gen[S3_CANVAS]);
    if (!v2) {
        s3.cur->hud.nquads = 0;
    }
    uint32_t ns = 0;
    /* the instances and bones into the next of the ring (the GPU may still
     * read the last); the terrain in place (it changes when a map loads) */
    int ring = (s3.ring + 1) % S3_RING;
    uint32_t irows = (ni * 4 + S3_INST_W - 1) / S3_INST_W, brows = (nb * 3 + S3_BONE_W - 1) / S3_BONE_W;
    if (irows) {
        [s3.inst[ring] replaceRegion:MTLRegionMake2D(0, 0, S3_INST_W, irows) mipmapLevel:0
                           withBytes:blk + offsetof(struct ros_scene3d, inst) bytesPerRow:S3_INST_W * 16];
    }
    if (brows) {
        [s3.bones[ring] replaceRegion:MTLRegionMake2D(0, 0, S3_BONE_W, brows) mipmapLevel:0
                            withBytes:blk + offsetof(struct ros_scene3d, bone) bytesPerRow:S3_BONE_W * 16];
    }
    memcpy(s3.cur->inst, b->inst, (size_t)ni * sizeof(b->inst[0]));
    if (terrain) {
        memcpy(s3.cur->height, b->height, sizeof(b->height));
        memcpy(s3.cur->tile, b->tile, sizeof(b->tile));
    }
    if (splats) {
        memcpy(s3.cur->splat_first, b->splat_first, sizeof(b->splat_first));
        ns = s3.cur->splat_first[ROS_S3_PATCHES * ROS_S3_PATCHES];
        ns = ns > ROS_S3_SPLATS ? ROS_S3_SPLATS : ns;
        memcpy(s3.cur->splat, b->splat, (size_t)ns * 4);
    }
    if (los) {
        memcpy(s3.cur->los, b->los, sizeof(b->los));
    }
    if (hud) {
        memcpy(&s3.cur->hud, &b->hud, sizeof(b->hud));
        uint32_t nq = s3.cur->hud.nquads > ROS_S3_QUADS ? ROS_S3_QUADS : s3.cur->hud.nquads;
        s3.cur->hud.nquads = nq;
        memcpy(s3.cur->quad, b->quad, (size_t)nq * sizeof(b->quad[0]));
    }
    if (canvas) {
        uint32_t ch = b->hud.canvas_h > ROS_S3_CANVAS ? ROS_S3_CANVAS : b->hud.canvas_h;
        memcpy(s3.cur->canvas, b->canvas, (size_t)ch * sizeof(b->canvas[0]));
    }
    uint32_t after;
    memcpy(&after, blk + offsetof(struct ros_scene3d, frame), 4);
    if (after != f0) {
        return s3.have;                 /* torn: the last whole frame stands (its instances
                                           in cur may be the new ones: drawn next frame) */
    }
    if (epoch && s3.have) {
        s3_drop();
    }
    memcpy(s3.cur, s3.hdr_next, 2048);
    if (terrain) {
        [s3.heights replaceRegion:MTLRegionMake2D(0, 0, ROS_S3_HSTRIDE, ROS_S3_VERTS) mipmapLevel:0
                        withBytes:s3.cur->height bytesPerRow:ROS_S3_HSTRIDE * 2];
    }
    if (los) {
        [s3.los replaceRegion:MTLRegionMake2D(0, 0, ROS_S3_HSTRIDE, ROS_S3_VERTS) mipmapLevel:0
                    withBytes:s3.cur->los bytesPerRow:ROS_S3_HSTRIDE];
    }
    if (canvas) {
        [s3.canvas replaceRegion:MTLRegionMake2D(0, 0, ROS_S3_CANVAS, ROS_S3_CANVAS) mipmapLevel:0
                       withBytes:s3.cur->canvas bytesPerRow:ROS_S3_CANVAS * 4];
    }
    if (hud) {
        uint32_t nq = s3.cur->hud.nquads;
        [s3.hudq[ring] release];
        s3.hudq[ring] = nq ? [m.device newBufferWithBytes:s3.cur->quad length:(size_t)nq * sizeof(s3.cur->quad[0])
                                                  options:MTLResourceStorageModeShared] : nil;
        for (int k = 0; k < S3_RING; k++) {
            if (k != ring) {
                [s3.hudq[k] release];
                s3.hudq[k] = nil;
            }
        }
        s3.hud_ring = ring;
    }
    if (splats) {
        [s3.splats release];
        s3.splats = ns ? [m.device newBufferWithBytes:s3.cur->splat length:(size_t)ns * 4
                                              options:MTLResourceStorageModeShared] : nil;
    }
    s3.ring = ring;
    memcpy(s3.gen, n->gen, sizeof(s3.gen));
    s3.epoch = n->epoch;
    s3.frame = f0;
    s3.have = true;
    return true;
}

static void s3_unpack8(uint32_t w, float o[4])
{
    for (int k = 0; k < 4; k++) {
        o[k] = (float)((w >> (8 * k)) & 255u) / 255.0f;
    }
}

/* Before the frame's main pass: the scene drawn into its own picture the
 * size of the icon's box in the target's pixels, when anything moved */
static void surf_scene3d_render(id<MTLCommandBuffer> cb, CGSize target, uint32_t xres, uint32_t yres)
{
    const MetalSurfacesView *v = &surf.view;
    struct ros_surf_scene3d d;
    const MetalIcon *ic = s3_first(&d);
    double sx = target.width / MAX(xres, 1u), sy = target.height / MAX(yres, 1u);

    if (!ic) {
        return;
    }
    s3_setup();
    if (!s3.ok || !s3_take((const uint8_t *)v->region + d.offset, d.size)) {
        return;
    }
    const struct ros_scene3d *h = s3.cur;
    s3_assets(h);
    uint32_t w = (uint32_t)lround(ic->w * sx), hh = (uint32_t)lround(ic->h * sy);
    if (w < 1 || hh < 1 || w > 16384 || hh > 16384) {
        return;
    }
    if (s3.color && s3.drawn_frame == s3.frame && s3.drawn_w == w && s3.drawn_h == hh
        && s3.drawn_assets == s3.assets) {
        return;                         /* drawn already */
    }
    if (!s3.color || s3.color.width != w || s3.color.height != hh) {
        [s3.color release];
        [s3.zbuf release];
        s3.color = gp_texture(MTLPixelFormatRGBA8Unorm, w, hh,
                              MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
        MTLTextureDescriptor *zd = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:w height:hh mipmapped:NO];
        zd.usage = MTLTextureUsageRenderTarget;
        zd.storageMode = MTLStorageModePrivate;
        s3.zbuf = [m.device newTextureWithDescriptor:zd];
    }
    if (!s3.color || !s3.zbuf) {
        return;
    }
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    float sky[4];
    for (int k = 0; k < 3; k++) {
        float c = h->env.sky_rgb[k] < 0 ? 0 : h->env.sky_rgb[k] > 1 ? 1 : h->env.sky_rgb[k];
        sky[k] = roundf(c * 255.0f) / 255.0f;
    }
    rp.colorAttachments[0].texture = s3.color;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(sky[0], sky[1], sky[2], 1);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.depthAttachment.texture = s3.zbuf;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = 1.0;
    rp.depthAttachment.storeAction = MTLStoreActionDontCare;
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setDepthStencilState:s3.depth];
    [enc setCullMode:MTLCullModeNone];
    S3U u;
    memset(&u, 0, sizeof(u));
    memcpy(u.vp, h->view.viewproj, sizeof(u.vp));
    memcpy(u.sun_dir, h->env.sun_dir, 16);
    memcpy(u.sun_rgb, h->env.sun_rgb, 16);
    memcpy(u.amb, h->env.ambient_rgb, 16);
    u.vsize[0] = (float)w, u.vsize[1] = (float)hh;
    u.tint[0] = u.tint[1] = u.tint[2] = u.tint[3] = 1.0f;
    u.player[0] = u.player[1] = u.player[2] = u.object[0] = u.object[1] = u.object[2] = 1.0f;
    /* the terrain (SCENE3D.md 6.3): the visible patches' sides, then their
     * opaque splats, then their blends, a run of one texture a draw */
    uint32_t patches = h->patches > ROS_S3_PATCHES ? ROS_S3_PATCHES : h->patches;
    if (patches && !(h->flags & ROS_S3_F_NO_TERRAIN) && s3.splats) {
        uint32_t n = patches * ROS_S3_PATCH, ns = (uint32_t)(s3.splats.length / 4);
        uint32_t nt = h->nterrains > ROS_S3_TERRAINS ? ROS_S3_TERRAINS : h->nterrains;
        uint8_t back[4];
        memcpy(back, &h->back_rgba, 4);
        for (int k = 0; k < 3; k++) {
            u.side[k] = back[k] / 255.0f;
        }
        u.side[3] = h->env.water[2] != 0.0f ? h->env.water[0] : 0.0f;
        u.terr[0] = (int32_t)(n + 1);
        S3Tex *amt = s3.am.have && s3.tex[s3.am.tex].t ? &s3.tex[s3.am.tex] : NULL;
        if (amt) {
            u.am[0] = (float)s3.am.stride, u.am[1] = (float)amt->w, u.am[2] = (float)s3.am.shapes;
        }
        [enc setVertexTexture:s3.heights atIndex:0];
        [enc setFragmentTexture:s3.los atIndex:1];
        if (amt) {
            [enc setFragmentTexture:amt->t atIndex:2];
            [enc setFragmentSamplerState:amt->wrap ? s3.repeat : s3.clamp atIndex:1];
        } else {
            [enc setFragmentSamplerState:s3.clamp atIndex:1];
        }
        [enc setVertexBuffer:s3.splats offset:0 atIndex:1];
        for (int pass = 0; pass < 3; pass++) {
            [enc setRenderPipelineState:pass == 0 ? s3.side : pass == 1 ? s3.splat : s3.blend];
            [enc setDepthStencilState:pass == 2 ? s3.depth_blend : s3.depth];
            for (uint32_t pz = 0; pz < patches; pz++) {
                for (uint32_t px = 0; px < patches; px++) {
                    if (!(h->patch_vis[pz] >> px & 1u)) {
                        continue;
                    }
                    if (pass == 0) {
                        int32_t e[4][4];
                        int ne = 0;
                        if (px == 0) {
                            int32_t a[4] = { 0, (int32_t)(pz * ROS_S3_PATCH), 0, 1 };
                            memcpy(e[ne++], a, 16);
                        }
                        if (px == patches - 1) {
                            int32_t a[4] = { (int32_t)n, (int32_t)(pz * ROS_S3_PATCH), 0, 1 };
                            memcpy(e[ne++], a, 16);
                        }
                        if (pz == 0) {
                            int32_t a[4] = { (int32_t)(px * ROS_S3_PATCH), 0, 1, 0 };
                            memcpy(e[ne++], a, 16);
                        }
                        if (pz == patches - 1) {
                            int32_t a[4] = { (int32_t)(px * ROS_S3_PATCH), (int32_t)n, 1, 0 };
                            memcpy(e[ne++], a, 16);
                        }
                        for (int k = 0; k < ne; k++) {
                            memcpy(u.edge, e[k], 16);
                            [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
                            [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
                            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0
                                    vertexCount:ROS_S3_PATCH * 6];
                        }
                        continue;
                    }
                    uint32_t p = pz * ROS_S3_PATCHES + px;
                    uint32_t a = h->splat_first[p], b = h->splat_first[p + 1];
                    a = a > ns ? ns : a, b = b > ns ? ns : b;
                    u.terr[2] = (int32_t)px, u.terr[3] = (int32_t)pz;
                    for (uint32_t k = a; k < b;) {
                        uint32_t w = s3.cur->splat[k], shape = (w >> 8) & 15u, tex = w >> 16;
                        bool solid = shape == ROS_S3_SHAPE_SOLID;
                        /* the run: one texture, one kind */
                        uint32_t e = k + 1;
                        while (e < b && (solid || shape < s3.am.shapes) && (s3.cur->splat[e] >> 16) == tex
                               && (((s3.cur->splat[e] >> 8) & 15u) == ROS_S3_SHAPE_SOLID) == solid
                               && (solid || ((s3.cur->splat[e] >> 8) & 15u) < s3.am.shapes)) {
                            e++;
                        }
                        uint32_t first = k;
                        k = e;
                        if (solid != (pass == 1)) {
                            continue;
                        }
                        S3Tex *t = NULL;
                        float mtx[4] = { 0.25f, 0, 0, 0.25f };
                        if (tex < nt) {
                            uint16_t id = h->terrain[tex];
                            if (id && id < ROS_S3_TERRAIN_IDS && s3.terr[id].have) {
                                memcpy(mtx, s3.terr[id].m, 16);
                                uint16_t di = s3.terr[id].diffuse;
                                if (di && di < ROS_S3_TEX_IDS && s3.tex[di].t) {
                                    t = &s3.tex[di];
                                }
                            }
                        }
                        if (!solid && (!t || !amt || shape >= s3.am.shapes)) {
                            continue;           /* a blend with nothing to blend */
                        }
                        memcpy(u.tm, mtx, 16);
                        u.mode[0] = solid ? 0 : 256, u.mode[1] = t != NULL, u.mode[2] = 0;
                        if (t) {
                            u.tsize[0] = t->w, u.tsize[1] = t->h, u.tsize[2] = t->levels - 1;
                            [enc setFragmentTexture:t->t atIndex:0];
                            [enc setFragmentSamplerState:t->wrap ? s3.repeat : s3.clamp atIndex:0];
                        }
                        [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
                        [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
                        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:first * 6
                                vertexCount:(e - first) * 6];
                    }
                }
            }
        }
        [enc setDepthStencilState:s3.depth];
        memset(u.edge, 0, sizeof(u.edge));
    }
    /* the models: a draw an instance */
    uint32_t ni = h->ninstances > ROS_S3_INSTANCES ? ROS_S3_INSTANCES : h->ninstances;
    if (ni) {
        [enc setRenderPipelineState:s3.model];
        [enc setFragmentTexture:s3.los atIndex:1];
        [enc setVertexTexture:s3.inst[s3.ring] atIndex:1];
        [enc setVertexTexture:s3.bones[s3.ring] atIndex:2];
        for (uint32_t k = 0; k < ni; k++) {
            const struct ros_s3_instance *in = &s3.cur->inst[k];
            if (!in->mesh || in->mesh >= ROS_S3_MESH_IDS || !s3.mesh[in->mesh].vb) {
                continue;
            }
            const S3Mesh *me = &s3.mesh[in->mesh];
            S3Tex *t = NULL;
            uint32_t shading = 0;
            float obj[4] = { 1, 1, 1, 1 };
            if (in->skin && in->skin < ROS_S3_SKIN_IDS && s3.skin_have[in->skin]) {
                const uint32_t *sk = s3.skin[in->skin];
                uint16_t base = (uint16_t)(sk[0] & 0xFFFFu);
                shading = sk[2] & 0xFFFFu;
                s3_unpack8(sk[3], obj);
                if (base && base < ROS_S3_TEX_IDS && s3.tex[base].t) {
                    t = &s3.tex[base];
                }
            }
            s3_unpack8(h->player[in->player % ROS_S3_PLAYERS], u.player);
            memcpy(u.object, obj, 16);
            s3_unpack8(in->shading, u.tint);
            u.tint[3] = in->alpha / 255.0f;
            u.mode[0] = (int32_t)(shading | (in->flags & ROS_S3_I_NO_LOS ? 4096u : 0u));
            u.mode[1] = t != NULL, u.mode[2] = t ? t->opaque : 0;
            u.mode[3] = (int32_t)k;
            u.terr[1] = (int32_t)me->nbones;
            if (t) {
                u.tsize[0] = t->w, u.tsize[1] = t->h, u.tsize[2] = t->levels - 1;
                [enc setFragmentTexture:t->t atIndex:0];
                [enc setFragmentSamplerState:t->wrap ? s3.repeat : s3.clamp atIndex:0];
            }
            [enc setVertexBuffer:me->vb offset:0 atIndex:1];
            [enc setVertexBytes:&u length:sizeof(u) atIndex:0];
            [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:me->nidx
                             indexType:MTLIndexTypeUInt16 indexBuffer:me->ib indexBufferOffset:0];
        }
    }
    /* the HUD over it all: quads in order, a draw for each run of one texture */
    uint32_t nq = h->hud.nquads > ROS_S3_QUADS ? ROS_S3_QUADS : h->hud.nquads;
    id<MTLBuffer> hq = s3.hudq[s3.hud_ring];
    if (nq && hq && hq.length >= (size_t)nq * sizeof(struct ros_s3_quad) && h->hud.width && h->hud.height) {
        S3HudU hu;
        memset(&hu, 0, sizeof(hu));
        hu.size[0] = h->hud.width, hu.size[1] = h->hud.height;
        [enc setRenderPipelineState:s3.hud];
        [enc setDepthStencilState:s3.depth_none];
        [enc setVertexBuffer:hq offset:0 atIndex:1];
        [enc setFragmentSamplerState:s3.clamp atIndex:0];
        for (uint32_t k = 0; k < nq;) {
            uint16_t tex = s3.cur->quad[k].tex;
            uint32_t e = k + 1;
            while (e < nq && s3.cur->quad[e].tex == tex) {
                e++;
            }
            id<MTLTexture> t = nil;
            if (tex == ROS_S3_HUD_CANVAS) {
                t = s3.canvas;
            } else if (tex && tex < ROS_S3_TEX_IDS) {
                t = s3.tex[tex].t;
            }
            if (tex && !t) {
                k = e;              /* a texture not yet uploaded: those quads wait */
                continue;
            }
            hu.mode[0] = t != nil;
            if (t) {
                [enc setFragmentTexture:t atIndex:0];
            }
            [enc setVertexBytes:&hu length:sizeof(hu) atIndex:0];
            [enc setFragmentBytes:&hu length:sizeof(hu) atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:k * 6 vertexCount:(e - k) * 6];
            k = e;
        }
    }
    [enc endEncoding];
    s3.drawn_frame = s3.frame, s3.drawn_w = w, s3.drawn_h = hh, s3.drawn_assets = s3.assets;
    if (!s3.drawn_any || ++s3.shown % 900 == 0) {
        metal_log("scene3d: frame %u, %ux%u, %u instances, %u bones, epoch %u", s3.frame, w, hh, ni,
                  h->nbones, h->epoch);
        s3.drawn_any = true;
    }
}

/* The scene's picture in the icon's visible rectangles */
static void surf_scene3d(id<MTLRenderCommandEncoder> enc, bool shot, CGSize target,
                         uint32_t xres, uint32_t yres)
{
    const MetalSurfacesView *v = &surf.view;
    struct ros_surf_scene3d d;
    const MetalIcon *ic = s3_first(&d);
    double sx = target.width / MAX(xres, 1u), sy = target.height / MAX(yres, 1u);
    S3ShowU u;

    if (!ic || !s3.ok || !s3.color) {
        return;
    }
    memset(&u, 0, sizeof(u));
    u.box[0] = ic->x * sx, u.box[1] = ic->y * sy, u.box[2] = ic->w * sx, u.box[3] = ic->h * sy;
    u.target[0] = target.width, u.target[1] = target.height;
    [enc setRenderPipelineState:shot ? s3.show_shot : s3.show];
    [enc setFragmentTexture:s3.color atIndex:0];
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
            /* the rate the clock asks (a smooth second hand) while the
             * window has the keyboard; once a second behind other
             * windows, where 30 frames a second would only cost power */
            double s = v->icon[i].rate > 1 && [m.window isKeyWindow]
                       ? 1.0 / MIN(v->icon[i].rate, 60u) : 1.0;

            step = step ? MIN(step, s) : s;
        }
        if (v->icon[i].kind == ROSGD_SURF_TILES && v->icon[i].nrects && v->icon[i].rate) {
            /* a tile map's rich drawing moves (water): at the rate the box
             * asks, while the window has the keyboard; a few frames a
             * second behind other windows */
            double s = 1.0 / ([m.window isKeyWindow] ? MIN(v->icon[i].rate, 60u) : 4u);

            step = step ? MIN(step, s) : s;
        }
        if (v->icon[i].kind == ROSGD_SURF_TOWER && v->icon[i].nrects && v->icon[i].rate) {
            /* a tower's smooth and rich drawings move (the lamps, fire):
             * as a tile map's */
            double s = 1.0 / ([m.window isKeyWindow] ? MIN(v->icon[i].rate, 60u) : 4u);

            step = step ? MIN(step, s) : s;
        }
        if (v->icon[i].kind == ROSGD_SURF_SCENE3D && v->icon[i].nrects && v->icon[i].rate) {
            /* a scene asks to be drawn unasked (water, P5): as a tower */
            double s = 1.0 / ([m.window isKeyWindow] ? MIN(v->icon[i].rate, 60u) : 4u);

            step = step ? MIN(step, s) : s;
        }
        if (v->icon[i].kind == ROSGD_SURF_GAMEPANE && v->icon[i].nrects) {
            /* a game pane is drawn at sixty whatever the window's focus:
             * its block changes without the Wimp knowing, so no damage
             * would ever say so, and a pane running has priority */
            step = step ? MIN(step, 1.0 / 60) : 1.0 / 60;
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
    if (n && !surf.tex_mode) {
        [enc setRenderPipelineState:surf.pipe];
        [enc setVertexBuffer:qb offset:0 atIndex:0];
        [enc setVertexBytes:screen length:sizeof(screen) atIndex:1];
        [enc setFragmentBuffer:qb offset:0 atIndex:0];
        [enc setFragmentBuffer:surf.region offset:0 atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
              instanceCount:n];
    } else if (n) {
        /* each surface in turn: its shadow, then its pixels from its copy
         * (surf_take made it this list's); a page from guest RAM */
        [enc setVertexBuffer:qb offset:0 atIndex:0];
        [enc setVertexBytes:screen length:sizeof(screen) atIndex:1];
        [enc setFragmentBuffer:qb offset:0 atIndex:0];
        for (uint32_t i = 0; i < surf.view.n; i++) {
            const MetalSurface *e = &surf.view.s[i];
            id<MTLTexture> t = nil;

            if (sq_shn[i]) {
                [enc setRenderPipelineState:surf.pipe];
                [enc setFragmentBuffer:surf.region offset:0 atIndex:1];
                [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
                      instanceCount:sq_shn[i] baseInstance:sq_sh0[i]];
            }
            if (!sq_pxn[i]) {
                continue;
            }
            if (e->format == 0 && (t = surf_texture_of(e)) != nil) {
                [enc setRenderPipelineState:surf.tex_pipe];
                [enc setFragmentTexture:t atIndex:0];
            } else {
                [enc setRenderPipelineState:surf.pipe];
                [enc setFragmentBuffer:surf.region offset:0 atIndex:1];
            }
            [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
                  instanceCount:sq_pxn[i] baseInstance:sq_px0[i]];
        }
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
/* metal-browser.m's: the host channel and the browser's panels */

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

    pthread_mutex_lock(&surf.mx);       /* the board's view and the list: the taker's too */
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
    surf.on = have_fb && metal_glue_surfaces_view(&surf.taken) && surf_build() && surf_region();
    if (have_fb) {
        surf_take();                    /* a new list's damage copied, then acked */
    }
    if (surf.on) {
        surf.view = surf.taken;         /* this frame's list, and the copies are its */
    }
    pthread_mutex_unlock(&surf.mx);
    /* the box's browser, the Mac's WebKit: its lines answered and its
     * panels placed, a frame presented or not */
    metal_browser_tick(m.window, [m.window contentView], have_fb ? &surf.view : NULL,
                       have_fb ? v.xres : 0, have_fb ? v.yres : 0);

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
            && (!surf.on || (surf.view.frame == surf.drawn_frame
                             && surf.view.pages == surf.drawn_pages)) && !surf_face_due()) {
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
        surf.drawn_pages = surf.on ? surf.view.pages : 0;
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
    pthread_mutex_lock(&surf.mx);       /* the copies stay as this frame's list has them */
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
    /* a game pane's picture first, in a pass of its own: the main pass
       then shows it like any other layer */
    if (have_fb) {
        surf_gamepane_render(cb);
        surf_scene3d_render(cb, target, v.xres, v.yres);
    }
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
        surf_gamepanes(enc, target, v.xres, v.yres);
        surf_tiles(enc, false, target, v.xres, v.yres);
        surf_tower(enc, false, target, v.xres, v.yres);
        surf_scene3d(enc, false, target, v.xres, v.yres);
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
        if (metal_browser_shown()) {
            /* a page on the screen: presented in one transaction with the
             * pages' places and masks, so that a page and the desktop round
             * it change in the same refresh */
            m.layer.presentsWithTransaction = YES;
            [cb commit];
            [cb waitUntilScheduled];
            [CATransaction begin];
            [CATransaction setDisableActions:YES];
            metal_browser_place();
            [drawable present];
            [CATransaction commit];
        } else {
            m.layer.presentsWithTransaction = NO;
            [cb presentDrawable:drawable];
            [cb commit];
        }
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
    pthread_mutex_unlock(&surf.mx);
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

    /* (and with the box's surfaces, whatever the backdrop: the pass is
     * where the window's own icons -- the clock, the meter -- go in) */
    if ((backdrop.mode != METAL_BACKDROP_OFF || surf.on)
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
            surf_gamepane_render(cb);
            surf_scene3d_render(cb, CGSizeMake(fb.xres, fb.yres), fb.xres, fb.yres);
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
            surf_gamepanes(senc, CGSizeMake(fb.xres, fb.yres), fb.xres, fb.yres);
            surf_tiles(senc, true, CGSizeMake(fb.xres, fb.yres), fb.xres, fb.yres);
            surf_tower(senc, true, CGSizeMake(fb.xres, fb.yres), fb.xres, fb.yres);
            surf_scene3d(senc, true, CGSizeMake(fb.xres, fb.yres), fb.xres, fb.yres);
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
    metal_key_held([e keyCode], true);
    metal_glue_key(true, [e keyCode]);
}

- (void)keyUp:(NSEvent *)e
{
    if ([e keyCode] == 105) {
        return;
    }
    metal_key_held([e keyCode], false);
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

/* A click on the desktop takes the keyboard back from a page */
static void metal_view_take_keys(void)
{
    if ([m.window firstResponder] != m.view) {
        [m.window makeFirstResponder:m.view];
    }
}

/* The keyboard gone to a page: the guest hears every key it holds go up,
 * or RISC OS repeats one for ever */
- (BOOL)resignFirstResponder
{
    metal_release_modifiers();
    metal_release_keys();
    return YES;
}

/* Every press and release carries its own place: Cocoa sends no
 * mouseMoved: just before a click when the pointer came in from another
 * window, or the window was brought forward by the click, or the move was
 * quicker than its events, so the guest would put the press where the
 * pointer last was -- and a stroke drawn from it starts off the page, with
 * a line out to where the button really went down. */
- (void)mouseDown:(NSEvent *)e
{
    metal_view_take_keys();
    mouse.held_left = metal_button_for(e, 0);
    if (metal_debug()) {
        metal_log("view: mouseDown -> button %d (flags %#lx)",
                  mouse.held_left, (unsigned long)[e modifierFlags]);
    }
    metal_mouse_move(e);            /* where it was pressed, first */
    metal_button(mouse.held_left, true);
}

- (void)mouseUp:(NSEvent *)e
{
    metal_mouse_move(e);
    metal_button(mouse.held_left, false);
    metal_mods_restore(e);
}

- (void)rightMouseDown:(NSEvent *)e
{
    metal_view_take_keys();
    mouse.held_right = metal_button_for(e, 2);
    if (metal_debug()) {
        metal_log("view: rightMouseDown -> button %d (flags %#lx)",
                  mouse.held_right, (unsigned long)[e modifierFlags]);
    }
    metal_mouse_move(e);
    metal_button(mouse.held_right, true);
}

- (void)rightMouseUp:(NSEvent *)e
{
    metal_mouse_move(e);
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
    metal_view_take_keys();
    metal_mouse_move(e);
    metal_button(1, true);
}

- (void)otherMouseUp:(NSEvent *)e
{
    if ([e buttonNumber] != 2) {
        return;
    }
    metal_mouse_move(e);
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
    metal_release_keys();           /* and no key it will never see go up */
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
            if (!surf.taker) {
                /* the taker from the first pass, frame or none: it keeps
                 * the presenter's beat, which a window that is not drawing
                 * (minimised, hidden) would otherwise leave stale or never
                 * write -- and the Wimp would make no window a surface */
                pthread_t t;

                surf.taker = pthread_create(&t, NULL, surf_taker, NULL) == 0;
                if (surf.taker) {
                    pthread_detach(t);
                }
            }
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
                /* nothing new: half a frame's wait -- but a list the box
                 * publishes meanwhile is taken at once, since the Wimp
                 * draws nothing more till it is (ROSGD_PRESENT_SNAPSHOT) */
                for (int i = 0; i < 8 && !metal_glue_surfaces_pending(); i++) {
                    usleep(1000);
                }
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

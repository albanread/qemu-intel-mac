/*
 * metal-virtio-gpu.c — the Metal front end's board for a virtio-gpu
 * machine: the ROSGD box (A7232ToolChain/rosgd), a RISC OS personality
 * on Linux.
 *
 * The picture is the resource that scanout 0 shows, read straight out
 * of guest RAM each frame, as the Pi's board reads the bcm2835
 * framebuffer.  A virtio-gpu resource's backing is ordinary guest pages,
 * and QEMU holds a host pointer to each (res->iov), so the window needs
 * no RESOURCE_FLUSH, no TRANSFER_TO_HOST and no copy in the device: the
 * guest draws into its pages and the next frame shows them.
 *
 * Linux's virtio-gpu driver declares every scanout XRGB8888.  ROSGD's
 * screen is in whatever format the RISC OS mode asks for, so ROSGD
 * describes it in a screen block, in the last 4 KB of the same buffer
 * (rosgd/include/rosgd/screen.h): size, pitch, depth, byte order, pan
 * and palette, as the Pi's mailbox describes its framebuffer.  The
 * decode shaders take it from there.  A buffer without the block --
 * Linux's own console, before ROSGD takes the screen -- is shown as the
 * XRGB8888 it declares.
 *
 * The pointer is the scanout's cursor, which Linux's DRM cursor plane
 * sends down the cursor queue: a 64x64 ARGB image and a position.
 *
 * Reading the backing takes the BQL only to see whether the resource
 * has changed; the bytes are copied without it, since guest RAM stays
 * mapped whatever the guest does with its resources.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/iov.h"
#include "qemu/main-loop.h"
#include "qom/object.h"
#include "hw/virtio/virtio-gpu.h"
#include "ui/console.h"
#include "ui/metal-board.h"

/* ---- the screen block: rosgd/include/rosgd/screen.h, the other side ---- */

#define ROSGD_SCREEN_MAGIC  0x44475352u     /* "RSGD" */
#define ROSGD_SCREEN_BLOCK  4096u           /* the last 4 KB of the buffer */

typedef struct RosgdScreenBlock {
    uint32_t magic;
    uint32_t version;           /* 1 */
    uint32_t generation;        /* moves on every mode change */
    uint32_t xres, yres;        /* visible pixels */
    uint32_t pitch;             /* bytes per row */
    uint32_t bpp;               /* 1, 2, 4, 8, 16 or 32 */
    uint32_t pixo;              /* 32 bpp: 1 red in the low byte, 0 blue */
    uint32_t xoffset, yoffset;  /* pan, pixels and rows */
    uint32_t reserved[6];
    uint32_t palette[256];      /* 0x00BBGGRR */
} RosgdScreenBlock;

QEMU_BUILD_BUG_ON(sizeof(RosgdScreenBlock) != 64 + 1024);

/* A mode, as read from the block or assumed from the scanout. */
typedef struct VgMode {
    uint32_t xres, yres, pitch, bpp, pixo, xoffset, yoffset;
    uint32_t block_gen;
    bool described;             /* from a screen block */
} VgMode;

/* The backing of the resource scanout 0 shows: a copy of its iovec,
 * taken under the BQL whenever the resource or its pages change. */
typedef struct VgBacking {
    uint32_t res_id;
    const void *res;            /* identity only: never dereferenced */
    const struct iovec *src;    /* likewise */
    struct iovec *iov;
    unsigned niov;
    size_t total;
    uint32_t scan_w, scan_h;    /* the scanout's own size */
} VgBacking;

static VirtIOGPU *vg_device(void)
{
    Object *obj = object_resolve_path_type("", TYPE_VIRTIO_GPU, NULL);

    return obj ? VIRTIO_GPU(obj) : NULL;
}

static bool vg_present(void)
{
    return vg_device() != NULL;
}

/*
 * Bring b up to date with scanout 0.  BQL held.  False when nothing is
 * being scanned out, or its resource has no backing in guest RAM.
 */
static bool vg_backing_sync(VirtIOGPU *g, VgBacking *b)
{
    struct virtio_gpu_scanout *s = &g->parent_obj.scanout[0];
    struct virtio_gpu_simple_resource *res;

    if (!s->resource_id) {
        return false;
    }
    res = virtio_gpu_find_resource(g, s->resource_id);
    if (!res || !res->iov || !res->iov_cnt) {
        return false;
    }
    b->scan_w = s->width;
    b->scan_h = s->height;
    if (res->resource_id == b->res_id && res == b->res
        && res->iov == b->src && res->iov_cnt == b->niov) {
        return true;
    }
    g_free(b->iov);
    b->iov = g_memdup2(res->iov, sizeof(struct iovec) * res->iov_cnt);
    b->niov = res->iov_cnt;
    b->src = res->iov;
    b->res = res;
    b->res_id = res->resource_id;
    b->total = iov_size(b->iov, b->niov);
    return true;
}

/*
 * The mode, from the screen block if there is a sane one, else the
 * scanout as the XRGB8888 it declares.  pal gets the palette (1 KB).
 */
static bool vg_mode(const VgBacking *b, VgMode *m, uint32_t *pal)
{
    RosgdScreenBlock blk;
    size_t need;

    memset(m, 0, sizeof(*m));
    if (b->total > ROSGD_SCREEN_BLOCK
        && iov_to_buf(b->iov, b->niov, b->total - ROSGD_SCREEN_BLOCK,
                      &blk, sizeof(blk)) == sizeof(blk)
        && le32_to_cpu(blk.magic) == ROSGD_SCREEN_MAGIC
        && le32_to_cpu(blk.version) == 1) {
        m->xres = le32_to_cpu(blk.xres);
        m->yres = le32_to_cpu(blk.yres);
        m->pitch = le32_to_cpu(blk.pitch);
        m->bpp = le32_to_cpu(blk.bpp);
        m->pixo = le32_to_cpu(blk.pixo) ? 1 : 0;
        m->xoffset = le32_to_cpu(blk.xoffset);
        m->yoffset = le32_to_cpu(blk.yoffset);
        m->block_gen = le32_to_cpu(blk.generation);
        m->described = true;
        for (unsigned i = 0; i < 256; i++) {
            pal[i] = le32_to_cpu(blk.palette[i]);
        }
        need = (size_t)m->pitch * (m->yres + m->yoffset);
        switch (m->bpp) {
        case 1: case 2: case 4: case 8: case 16: case 32:
            break;
        default:
            return false;
        }
        return m->xres && m->yres && m->xres <= 16384 && m->yres <= 16384
               && (uint64_t)(m->xres + m->xoffset) * m->bpp <= (uint64_t)m->pitch * 8
               && need <= b->total - ROSGD_SCREEN_BLOCK;
    }
    /* Undescribed: what Linux declared, XRGB8888 -- blue in the low byte. */
    m->xres = b->scan_w;
    m->yres = b->scan_h;
    m->bpp = 32;
    m->pixo = 0;
    m->pitch = b->scan_w * 4;
    memset(pal, 0, 256 * 4);
    return m->xres && m->yres
           && (size_t)m->pitch * m->yres <= b->total;
}

/* ---- the frame's view (UI thread) --------------------------------------- */

static struct {
    VirtIOGPU *g;
    VgBacking b;
    VgMode mode;                /* last shown, to see a mode change */
    const void *mode_res;
    uint32_t generation;
    uint8_t *stage;             /* the frame's bytes, contiguous */
    size_t stage_len;
    uint32_t pal[256];
} view;

static int vg_fb_view(MetalFbView *out)
{
    bool locked = !bql_locked();
    bool ok;
    VgMode m;
    size_t need;

    if (locked) {
        bql_lock();
    }
    if (!view.g) {
        view.g = vg_device();
    }
    ok = view.g && vg_backing_sync(view.g, &view.b);
    if (locked) {
        bql_unlock();
    }
    if (!ok || !vg_mode(&view.b, &m, view.pal)) {
        return 0;
    }

    if (view.mode_res != view.b.res || m.xres != view.mode.xres
        || m.yres != view.mode.yres || m.pitch != view.mode.pitch
        || m.bpp != view.mode.bpp || m.pixo != view.mode.pixo
        || m.yoffset != view.mode.yoffset
        || m.block_gen != view.mode.block_gen
        || m.described != view.mode.described || !view.generation) {
        view.generation++;
        view.mode = m;
        view.mode_res = view.b.res;
        metal_log("virtio-gpu: resource %u, %ux%u %u bpp%s, pitch %u, "
                  "%s", view.b.res_id, m.xres, m.yres, m.bpp,
                  m.bpp == 32 ? (m.pixo ? " red-low" : " blue-low") : "",
                  m.pitch, m.described ? "described" : "as XRGB8888");
    }

    need = (size_t)m.pitch * (m.yres + m.yoffset);
    if (need > view.stage_len) {
        g_free(view.stage);
        view.stage = g_malloc(need);
        view.stage_len = need;
    }
    iov_to_buf(view.b.iov, view.b.niov, 0, view.stage, need);

    out->xres = m.xres;
    out->yres = m.yres;
    out->xoffset = m.xoffset;
    out->yoffset = m.yoffset;
    out->pitch = m.pitch;
    out->bpp = m.bpp;
    out->pixo = m.pixo;
    out->rows = m.yres + m.yoffset;
    out->generation = view.generation;
    out->gbase = 0;             /* scattered pages: no one guest address */
    out->fb = view.stage;
    out->palette = view.pal;
    return 1;
}

static void vg_fb_done(void)
{
    g_free(view.stage);
    g_free(view.b.iov);
    memset(&view, 0, sizeof(view));
}

/* ---- the pointer: the scanout's cursor ---------------------------------- */

#define VG_CURSOR 64            /* virtio-gpu's cursors are 64x64 */

static struct {
    uint32_t argb[VG_CURSOR * VG_CURSOR];
    int32_t x, y, hot_x, hot_y;
    bool visible;
    uint32_t generation;
} cur;

static int vg_cursor_view(MetalCursorView *out)
{
    bool locked = !bql_locked();
    struct virtio_gpu_scanout *s;
    bool visible, changed = false;
    int32_t x = 0, y = 0, hx = 0, hy = 0;

    if (!view.g) {
        return 0;
    }
    if (locked) {
        bql_lock();
    }
    s = &view.g->parent_obj.scanout[0];
    visible = s->cursor.resource_id && s->current_cursor
              && s->current_cursor->width == VG_CURSOR
              && s->current_cursor->height == VG_CURSOR;
    if (visible) {
        /* The position is the cursor plane's corner, and the hot spot
         * comes beside it: Linux's virtio-gpu sends crtc_x/crtc_y and
         * the plane's HOTSPOT_X/Y (drm_plane.c, "hotspot properties";
         * virtgpu_plane.c), the pointer being the corner plus the hot
         * spot -- which is what lets a console draw the pointer where
         * the host's mouse is before the guest has moved it. */
        x = (int32_t)s->cursor.pos.x;
        y = (int32_t)s->cursor.pos.y;
        hx = (int32_t)s->current_cursor->hot_x;
        hy = (int32_t)s->current_cursor->hot_y;
        if (memcmp(cur.argb, s->current_cursor->data, sizeof(cur.argb))) {
            memcpy(cur.argb, s->current_cursor->data, sizeof(cur.argb));
            changed = true;
        }
    }
    if (locked) {
        bql_unlock();
    }
    if (changed || visible != cur.visible || x != cur.x || y != cur.y
        || hx != cur.hot_x || hy != cur.hot_y || !cur.generation) {
        cur.generation++;
        cur.visible = visible;
        cur.x = x;
        cur.y = y;
        cur.hot_x = hx;
        cur.hot_y = hy;
    }

    out->generation = cur.generation;
    out->stale = false;
    out->visible = cur.visible;
    out->x = cur.x;
    out->y = cur.y;
    out->hot_x = cur.hot_x;
    out->hot_y = cur.hot_y;
    out->has_hot = true;
    out->w = VG_CURSOR;
    out->h = VG_CURSOR;
    out->img_w = VG_CURSOR;
    out->img_h = VG_CURSOR;
    /* Measured in the mode's own pixels. */
    out->disp_w = view.mode.xres;
    out->disp_h = view.mode.yres;
    out->argb = cur.argb;
    return 1;
}

/* ---- the screendump (BQL held) ------------------------------------------ */

static int vg_capture(MetalCapture *c, const char **errmsg)
{
    VirtIOGPU *g = vg_device();
    VgBacking b = { 0 };
    uint32_t pal[256];
    uint32_t bypp, rowlen;
    VgMode m;

    if (!g) {
        *errmsg = "no virtio-gpu device";
        return -1;
    }
    if (!vg_backing_sync(g, &b) || !vg_mode(&b, &m, pal)) {
        g_free(b.iov);
        *errmsg = "nothing is being scanned out";
        return -1;
    }
    bypp = (m.bpp + 7) / 8;
    rowlen = m.xres * bypp;
    c->raw = g_malloc_n(m.yres, rowlen);
    for (uint32_t y = 0; y < m.yres; y++) {
        /* Sub-byte depths read a whole row's bytes from the row's start,
         * as the Pi's capture does; the pan is whole bytes at 8 bpp up. */
        size_t off = (size_t)(m.yoffset + y) * m.pitch
                     + (size_t)m.xoffset * m.bpp / 8;
        size_t len = MIN((size_t)rowlen, b.total - off);

        memset(c->raw + (size_t)y * rowlen, 0, rowlen);
        iov_to_buf(b.iov, b.niov, off, c->raw + (size_t)y * rowlen, len);
    }
    c->pal = NULL;
    if (m.bpp <= 8) {
        c->pal = g_memdup2(pal, sizeof(pal));
    }
    c->xres = m.xres;
    c->yres = m.yres;
    c->bpp = m.bpp;
    c->pixo = m.pixo;
    c->generation = view.generation;
    g_free(b.iov);
    return 0;
}

static const MetalBoard vg_board = {
    .name           = "virtio-gpu",
    .system         = "ROSGD",
    .machine        = "virtio-gpu (QEMU)",
    .present        = vg_present,
    .fb_view        = vg_fb_view,
    .fb_done        = vg_fb_done,
    .cursor_view    = vg_cursor_view,
    .capture        = vg_capture,
};

static void metal_virtio_gpu_register(void)
{
    metal_board_register(&vg_board);
}
type_init(metal_virtio_gpu_register);

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
#include "monitor/monitor.h"
#include "system/address-spaces.h"
#include "system/memory.h"
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
    size_t block_off;           /* where the block is in the buffer */
} VgMode;

/*
 * ---- block v2 and the backings region (design 39) ----------------------
 *
 * Block v2 is the Wimp's surfaces list (screen.h, design 37): its own
 * area of the screen's buffer, found through the surfaces table's
 * v2_offset, a seqlock round it.  Version 3 adds the backings region:
 * one physically contiguous stretch of guest RAM, its base and size in
 * the header, each window's pixels at an offset in it.  The host maps
 * the region once and draws the backings where they lie.
 *
 * The host says it is there -- that it draws the surfaces -- in the
 * block's presenter area (PRES): until it does, the Wimp makes no
 * backings and draws to the screen as before.  METAL_SURFACES says
 * whether it does: on (the window composites the list), off (the screen
 * as it is, as before design 39), or probe (PRES written and the lists
 * taken, nothing drawn from them: sprint 1's test of the region).
 */
#define ROSGD_SCREEN_PRESENT    3200u
#define ROSGD_PRESENT_MAGIC     0x53455250u     /* "PRES" */
#define ROSGD_PRESENT_SURFACES  1u
#define ROSGD_PRESENT_WINDOWS_ONLY 2u           /* the clock's and the meter's faces are
                                                   their tasks' own, not drawn here */
#define ROSGD_PRESENT_REGION    4u              /* backings are drawn only from the region */
#define ROSGD_SCREEN_SURF       (ROSGD_SCREEN_PRESENT + 16u)
#define ROSGD_V2_MAGIC          0x32565352u     /* "RSV2" */
#define ROSGD_V2_SIZE           65536u
#define ROSGD_V2_DAMAGE         256u
#define ROSGD_V2_SURFACES       128u
#define ROSGD_V2_RECTS          2048u
#define ROSGD_V2_SDAMAGE        1024u
#define ROSGD_V2_NO_OFFSET      0xFFFFFFFFu
#define ROSGD_SURF_WINDOW       3u
#define ROSGD_SURF_PAGE         4u

typedef struct RosgdPresent {
    uint32_t magic, state, beat, caps;
} RosgdPresent;

typedef struct RosgdV2Rect {
    int16_t x0, y0, x1, y1;
} RosgdV2Rect;

typedef struct RosgdV2Surface {
    uint32_t kind, param, window, frame;
    int16_t x, y, w, h;
    uint32_t rects, nrects, damage, ndamage;
    uint32_t backing, stride;
    int16_t shadow_dx, shadow_dy, shadow_blur, shadow_strength;
    uint32_t shadow_rects, shadow_nrects;
    uint8_t data[32];
} RosgdV2Surface;

typedef struct RosgdV2 {
    uint32_t magic, version, frame, full, ndamage, ack, nsurfaces, look;
    RosgdV2Rect damage[ROSGD_V2_DAMAGE];
    RosgdV2Surface surface[ROSGD_V2_SURFACES];
    RosgdV2Rect rect[ROSGD_V2_RECTS];
    RosgdV2Rect sdamage[ROSGD_V2_SDAMAGE];
    uint64_t shm_base;          /* version 3: the region, guest-physical */
    uint32_t shm_size, shm_used;
    uint32_t offset[ROSGD_V2_SURFACES];
} RosgdV2;

QEMU_BUILD_BUG_ON(sizeof(RosgdV2Surface) != 96);
QEMU_BUILD_BUG_ON(offsetof(RosgdV2, shm_base) != 38944);
QEMU_BUILD_BUG_ON(sizeof(RosgdV2) > ROSGD_V2_SIZE);

/* The backing of the resource scanout 0 shows: a copy of its iovec,
 * taken under the BQL whenever the resource or its pages change. */
typedef struct VgBacking {
    uint32_t res_id;
    const void *res;            /* identity only: never dereferenced */
    const struct iovec *src;    /* likewise */
    struct iovec *iov;
    unsigned niov;
    size_t total;               /* the backing's pages, whole */
    size_t res_bytes;           /* the resource's own extent, unrounded */
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
    /* The screen block sits in the last 4 KB of the buffer the guest
     * allocated -- a dumb buffer of this resource's own width x height,
     * its bytes exactly width * height * 4 (Linux's virtio-gpu dumbs are
     * 32 bpp).  The backing's pages are rounded up to whole pages beyond
     * that, so the block is NOT in the backing's last 4 KB whenever the
     * buffer's own end falls inside its last page (1280 x 800's is 3
     * pages short); measure from the resource's extent, not the pages'. */
    b->res_bytes = MIN((uint64_t)res->width * res->height * 4, b->total);
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
    /* The block sits in the last 4 KB of the buffer the guest allocated:
     * a dumb buffer of this resource's own width x height, whose bytes are
     * exactly width * height * 4 (Linux's virtio-gpu dumbs are 32 bpp).
     * The backing's pages run on to whole pages past that end, so the
     * block is NOT at the backing's own last 4 KB whenever the buffer's
     * end falls inside its last page (1280 x 800's is 3 pages short);
     * measure from the resource's extent, falling back to the pages'. */
    for (unsigned i = 0; i < 2; i++) {
        size_t end = i == 0 ? b->res_bytes : b->total;

        if (i == 1 && end == b->res_bytes) {
            break;                  /* the same place twice */
        }
        if (end <= ROSGD_SCREEN_BLOCK) {
            break;
        }
        if (iov_to_buf(b->iov, b->niov, end - ROSGD_SCREEN_BLOCK,
                       &blk, sizeof(blk)) != sizeof(blk)
            || le32_to_cpu(blk.magic) != ROSGD_SCREEN_MAGIC
            || le32_to_cpu(blk.version) != 1) {
            continue;
        }
        m->xres = le32_to_cpu(blk.xres);
        m->yres = le32_to_cpu(blk.yres);
        m->pitch = le32_to_cpu(blk.pitch);
        m->bpp = le32_to_cpu(blk.bpp);
        m->pixo = le32_to_cpu(blk.pixo) ? 1 : 0;
        m->xoffset = le32_to_cpu(blk.xoffset);
        m->yoffset = le32_to_cpu(blk.yoffset);
        m->block_gen = le32_to_cpu(blk.generation);
        m->described = true;
        m->block_off = end - ROSGD_SCREEN_BLOCK;
        for (unsigned j = 0; j < 256; j++) {
            pal[j] = le32_to_cpu(blk.palette[j]);
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
               && need <= end - ROSGD_SCREEN_BLOCK;
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

/* Block v2's place in the buffer, from the surfaces table: 0 if none */
static size_t vg_v2_offset(const VgBacking *b, const VgMode *m)
{
    uint32_t off = 0, magic = 0;

    if (!m->described
        || iov_to_buf(b->iov, b->niov, m->block_off + ROSGD_SCREEN_SURF + 8,
                      &off, 4) != 4) {
        return 0;
    }
    off = le32_to_cpu(off);
    if (!off || (size_t)off + ROSGD_V2_SIZE > m->block_off
        || iov_to_buf(b->iov, b->niov, off, &magic, 4) != 4
        || le32_to_cpu(magic) != ROSGD_V2_MAGIC) {
        return 0;
    }
    return off;
}

/* The region, mapped to a host pointer once: guest RAM on the virt board
 * is one block, so a contiguous guest-physical range is one host range.
 * Remapped only if the box publishes another. */
static struct {
    uint64_t base, size;
    void *host;
} region;

static void *vg_region(uint64_t base, uint64_t size)
{
    hwaddr len = size;

    if (region.host && region.base == base && region.size == size) {
        return region.host;
    }
    if (region.host) {
        address_space_unmap(&address_space_memory, region.host, region.size,
                            true, region.size);
        region.host = NULL;
    }
    if (!base || !size) {
        return NULL;
    }
    region.host = address_space_map(&address_space_memory, base, &len, true,
                                    MEMTXATTRS_UNSPECIFIED);
    if (region.host && len != size) {
        address_space_unmap(&address_space_memory, region.host, len, true, 0);
        region.host = NULL;
    }
    region.base = base;
    region.size = size;
    metal_log("box-v2: region &%" PRIx64 " +%" PRIu64 " MB -> host %p",
              base, size >> 20, region.host);
    return region.host;
}

/* METAL_SURFACES: 0 off, 1 probe, 2 on */
enum { VG_SURF_OFF, VG_SURF_PROBE, VG_SURF_ON };

static int vg_surfaces_mode(void)
{
    static int mode = -1;

    if (mode < 0) {
        const char *e = getenv("METAL_SURFACES");

        mode = !e || !strcmp(e, "off") ? VG_SURF_OFF
             : !strcmp(e, "probe") ? VG_SURF_PROBE : VG_SURF_ON;
        metal_log("box-v2: METAL_SURFACES=%s", mode == VG_SURF_ON ? "on: the window "
                  "composites the surfaces" : mode == VG_SURF_PROBE ? "probe: PRES "
                  "written, nothing drawn from the surfaces" : "off: the screen as it is");
    }
    return mode;
}

/* The presenter's area, written at every frame the window looks at (a
 * beat each time); in probe mode every list is also taken (the ack), so
 * that the Wimp makes its backings in the region with nothing drawing
 * them -- in on mode the ack is the surfaces view's */
static void vg_announce(const VgBacking *b, const VgMode *m)
{
    static uint32_t beat;
    int mode = vg_surfaces_mode();
    RosgdPresent p;
    size_t v2;
    uint32_t frame;

    if (mode == VG_SURF_OFF || !m->described) {
        return;
    }
    p.magic = cpu_to_le32(ROSGD_PRESENT_MAGIC);
    p.state = cpu_to_le32(2);
    p.beat = cpu_to_le32(++beat);
    p.caps = cpu_to_le32(ROSGD_PRESENT_SURFACES | ROSGD_PRESENT_REGION |
                         (mode == VG_SURF_ON ? ROSGD_PRESENT_WINDOWS_ONLY : 0));
    iov_from_buf(b->iov, b->niov, m->block_off + ROSGD_SCREEN_PRESENT,
                 &p, sizeof(p));
    if (mode != VG_SURF_PROBE) {
        return;
    }
    v2 = vg_v2_offset(b, m);
    if (v2 && iov_to_buf(b->iov, b->niov, v2 + offsetof(RosgdV2, frame),
                         &frame, 4) == 4 && !(le32_to_cpu(frame) & 1)) {
        iov_from_buf(b->iov, b->niov, v2 + offsetof(RosgdV2, ack), &frame, 4);
    }
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
    /* Power: whether the picture changed, so that the window presents a
     * frame only when there is something new (metal.m).  A compositor
     * (rosgd.display=compositor) shows each new picture by flipping to
     * another resource, which costs nothing to see; a screen drawn in
     * place is compared with the last copy, chunk by chunk, and only the
     * changed chunks copied -- every frame while it changes, and once
     * still for a tenth of a second, one frame in six. */
    uint32_t content;           /* MetalFbView.content */
    const void *content_res;    /* the resource the stage was copied from */
    uint32_t copied_gen;        /* the mode generation it was copied in */
    uint32_t drawn_pal[256];    /* the palette last seen: a change is new */
    unsigned still, tick;
} view;

/* the stage brought up to date with the guest's bytes; whether any changed */
static bool vg_stage_sync(size_t need)
{
    enum { CHUNK = 16384 };
    uint8_t buf[CHUNK];
    bool changed = false;

    for (size_t off = 0; off < need; off += CHUNK) {
        size_t n = need - off < CHUNK ? need - off : CHUNK;

        iov_to_buf(view.b.iov, view.b.niov, off, buf, n);
        if (memcmp(view.stage + off, buf, n) != 0) {
            memcpy(view.stage + off, buf, n);
            changed = true;
        }
    }
    return changed;
}

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
    vg_announce(&view.b, &m);

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
        view.content_res = NULL;        /* all of it copied, below */
    }
    view.tick++;
    if (view.content_res != view.b.res || view.copied_gen != view.generation) {
        /* another resource: a compositor's new frame, or a new mode */
        iov_to_buf(view.b.iov, view.b.niov, 0, view.stage, need);
        view.content_res = view.b.res;
        view.copied_gen = view.generation;
        view.content++;
        view.still = 0;
    } else if (view.still < 6 || view.tick % 6 == 0) {
        /* the same resource, drawn in place: looked at */
        if (vg_stage_sync(need)) {
            view.content++;
            view.still = 0;
        } else {
            view.still++;
        }
    }

    out->xres = m.xres;
    out->yres = m.yres;
    out->xoffset = m.xoffset;
    out->yoffset = m.yoffset;
    out->pitch = m.pitch;
    out->bpp = m.bpp;
    out->pixo = m.pixo;
    out->rows = m.yres + m.yoffset;
    if (memcmp(view.drawn_pal, view.pal, sizeof(view.pal)) != 0) {
        memcpy(view.drawn_pal, view.pal, sizeof(view.pal));
        view.content++;                 /* a palette mode's colours */
    }
    out->generation = view.generation;
    out->content = view.content ? view.content : 1;
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

/* ---- the surfaces (design 39): block v2's list, for metal.m to draw ----- */

/*
 * The list, taken whole under its seqlock: a copy of block v2 while its
 * frame is even and the same after.  Only a new frame is taken; a torn or
 * odd one leaves `out` as the last good list.  Each window with its pixels
 * in the region becomes a surface, its rectangles and its shadow's copied
 * into the view's pool.  The frame taken is acked, which lets the Wimp
 * begin its next list afresh: there is no upload to wait for, the window
 * reading the backings where they lie.
 */
static int vg_surfaces_view(MetalSurfacesView *out)
{
    static RosgdV2 *v;
    static uint32_t taken;
    static int have;
    uint32_t f0, f1;
    size_t off;
    void *host;

    if (vg_surfaces_mode() != VG_SURF_ON || !view.g || !view.mode.described) {
        return 0;
    }
    off = vg_v2_offset(&view.b, &view.mode);
    if (!off) {
        return have = 0;
    }
    if (iov_to_buf(view.b.iov, view.b.niov, off + offsetof(RosgdV2, frame), &f0, 4) != 4) {
        return have;
    }
    f0 = le32_to_cpu(f0);
    if ((f0 & 1) || (have && f0 == taken)) {
        return have;                    /* being written, or as it was */
    }
    if (!v) {
        v = g_malloc(sizeof(*v));
    }
    iov_to_buf(view.b.iov, view.b.niov, off, v, sizeof(*v));
    smp_rmb();
    if (iov_to_buf(view.b.iov, view.b.niov, off + offsetof(RosgdV2, frame), &f1, 4) != 4
        || le32_to_cpu(f1) != f0 || v->frame != f0) {
        return have;                    /* torn: the next frame's */
    }
    taken = f0;
    iov_from_buf(view.b.iov, view.b.niov, off + offsetof(RosgdV2, ack), &f0, 4);
    host = NULL;
    if (v->version >= 3 && v->shm_size) {
        bool locked = !bql_locked();

        if (locked) {
            bql_lock();
        }
        host = vg_region(v->shm_base, v->shm_size);
        if (locked) {
            bql_unlock();
        }
    }
    out->frame = f0;
    out->region = host;
    out->region_size = host ? v->shm_size : 0;
    out->look = v->look;
    out->n = 0;
    out->nrects = 0;
    for (uint32_t k = 0; k < v->nsurfaces && k < ROSGD_V2_SURFACES; k++) {
        const RosgdV2Surface *e = &v->surface[k];
        MetalSurface *d;
        uint32_t o = v->offset[k];

        if (e->kind != ROSGD_SURF_WINDOW || !e->backing || o == ROSGD_V2_NO_OFFSET || !host
            || e->w <= 0 || e->h <= 0 || !e->stride
            || (uint64_t)o + (uint64_t)e->stride * (uint64_t)e->h > v->shm_size) {
            continue;
        }
        d = &out->s[out->n++];
        d->x = e->x, d->y = e->y, d->w = e->w, d->h = e->h;
        d->offset = o;
        d->stride = e->stride;
        d->radius = e->data[1];
        d->shadow_dx = e->shadow_dx, d->shadow_dy = e->shadow_dy;
        d->shadow_blur = e->shadow_blur, d->shadow_strength = e->shadow_strength;
        d->rects = out->nrects;
        for (uint32_t r = 0; r < e->nrects && e->rects + r < ROSGD_V2_RECTS
             && out->nrects < METAL_SURF_RECTS; r++) {
            const RosgdV2Rect *q = &v->rect[e->rects + r];

            out->rect[out->nrects][0] = q->x0, out->rect[out->nrects][1] = q->y0;
            out->rect[out->nrects][2] = q->x1, out->rect[out->nrects][3] = q->y1;
            out->nrects++;
        }
        d->nrects = out->nrects - d->rects;
        d->shadow_rects = out->nrects;
        for (uint32_t r = 0; e->shadow_strength > 0 && r < e->shadow_nrects
             && e->shadow_rects + r < ROSGD_V2_RECTS && out->nrects < METAL_SURF_RECTS; r++) {
            const RosgdV2Rect *q = &v->rect[e->shadow_rects + r];

            out->rect[out->nrects][0] = q->x0, out->rect[out->nrects][1] = q->y0;
            out->rect[out->nrects][2] = q->x1, out->rect[out->nrects][3] = q->y1;
            out->nrects++;
        }
        d->shadow_nrects = out->nrects - d->shadow_rects;
    }
    have = out->n > 0;
    return have;
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
    .surfaces_view  = vg_surfaces_view,
    .capture        = vg_capture,
};

/* ---- info box-v2: the list, and the region read through the mapping ---- */

static const char *vg_kind(uint32_t k)
{
    switch (k) {
    case 1: return "clock";
    case 2: return "monitor";
    case ROSGD_SURF_WINDOW: return "window";
    case ROSGD_SURF_PAGE: return "page";
    default: return "?";
    }
}

/*
 * The header, the region, and each entry; for each entry with pixels in
 * the region, its first four words read through the host's mapping and,
 * beside them, through QEMU's memory API from the guest-physical
 * address: the two views must agree, with nothing copied between them.
 */
static void hmp_info_box_v2(Monitor *mon, const QDict *qdict)
{
    VirtIOGPU *g = vg_device();
    VgBacking b = { 0 };
    uint32_t pal[256];
    VgMode m;
    size_t off;
    RosgdV2 *v;
    uint8_t *host;
    unsigned agree = 0, differ = 0;

    if (!g || !vg_backing_sync(g, &b) || !vg_mode(&b, &m, pal)) {
        monitor_printf(mon, "box-v2: nothing is being scanned out\n");
        g_free(b.iov);
        return;
    }
    off = vg_v2_offset(&b, &m);
    if (!off) {
        monitor_printf(mon, "box-v2: no block v2 in the screen's buffer "
                       "(%ux%u, %u bpp%s)\n", m.xres, m.yres, m.bpp,
                       m.described ? "" : ", no screen block");
        g_free(b.iov);
        return;
    }
    v = g_malloc(sizeof(*v));
    iov_to_buf(b.iov, b.niov, off, v, sizeof(*v));
    monitor_printf(mon, "box-v2: version %u, frame %u, ack %u, %u surfaces, "
                   "look &%x, damage %s%u\n",
                   v->version, v->frame, v->ack, v->nsurfaces, v->look,
                   v->full ? "full, " : "", v->ndamage);
    host = v->version >= 3 ? vg_region(v->shm_base, v->shm_size) : NULL;
    if (v->version >= 3 && v->shm_size) {
        monitor_printf(mon, "region: &%" PRIx64 ", %u MB, %u KB carved, "
                       "host %p\n", v->shm_base, v->shm_size >> 20,
                       v->shm_used >> 10, host);
    } else {
        monitor_printf(mon, "region: none\n");
    }
    for (uint32_t k = 0; k < v->nsurfaces && k < ROSGD_V2_SURFACES; k++) {
        const RosgdV2Surface *e = &v->surface[k];
        uint32_t o = v->version >= 3 ? v->offset[k] : ROSGD_V2_NO_OFFSET;

        monitor_printf(mon, "%3u %-7s win &%08x at %d,%d %dx%d", k,
                       vg_kind(e->kind), e->window, e->x, e->y, e->w, e->h);
        if (e->kind == ROSGD_SURF_WINDOW) {
            monitor_printf(mon, " backing %u stride %u", e->backing,
                           e->stride);
        }
        if (o != ROSGD_V2_NO_OFFSET && host && (uint64_t)o + 16 <= v->shm_size) {
            uint32_t mapped[4], phys[4];

            memcpy(mapped, host + o, sizeof(mapped));
            address_space_read(&address_space_memory, v->shm_base + o,
                               MEMTXATTRS_UNSPECIFIED, phys, sizeof(phys));
            monitor_printf(mon, " offset &%x: %08x %08x %08x %08x %s", o,
                           mapped[0], mapped[1], mapped[2], mapped[3],
                           memcmp(mapped, phys, sizeof(mapped)) ? "DIFFER"
                                                                 : "agree");
            if (memcmp(mapped, phys, sizeof(mapped))) {
                differ++;
            } else {
                agree++;
            }
        }
        monitor_printf(mon, "\n");
    }
    if (agree || differ) {
        monitor_printf(mon, "region reads: %u agree, %u differ\n", agree,
                       differ);
    }
    g_free(v);
    g_free(b.iov);
}

static void metal_virtio_gpu_register(void)
{
    metal_board_register(&vg_board);
    monitor_register_hmp("box-v2", true, hmp_info_box_v2);
}
type_init(metal_virtio_gpu_register);

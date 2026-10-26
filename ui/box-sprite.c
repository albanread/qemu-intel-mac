/* SPDX-License-Identifier: MIT
 * box-sprite.c -- RISC OS sprite files as RGBA8 pictures, for the
 * backdrop layers: QEMU's Metal renderer on the Mac (ui/metal.m) and the
 * GPU presenter on the PC (rosgd/ports/present), design 37, 3.5.
 *
 * The disc's wallpapers (Resources/Wallpapers) are sprite files, ,ff9 on
 * the host side.  A sprite file is a sprite area without its first word:
 * the count, the offset (+4) of the first sprite, the offset of the free
 * space.  Each sprite is a 44-byte header -- next, name[12], width in
 * words - 1, height - 1, first and last bit used, image and mask
 * offsets, the mode -- then the palette, if any (pairs of &BBGGRR00
 * words), the image and the mask.  Read here:
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
 * theme's Sprites11 has) two to one.  Moved out of metal.m unchanged. */
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "box-sprite.h"

#define SPR_MIN(a, b) ((a) < (b) ? (a) : (b))
#define SPR_MAX(a, b) ((a) > (b) ? (a) : (b))

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
        uint32_t entries = (SPR_MIN(img, has_mask ? msk : img) - SPR_MIN(img, 44u)) / 8;
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
    *dpi = SPR_MAX(sm.xdpi, sm.ydpi);
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
 * theme's Tools file does), the one that suits the display: the densest
 * up to 180 dpi when dense, else the one nearest 90 dpi. */
uint8_t *box_sprite_decode_file(const uint8_t *f, size_t len, bool dense,
                                size_t *w, size_t *h, int *dpi)
{
    size_t at, best_at = 0;
    bool ok = true;
    uint32_t count, first, i;
    char base[13] = "";
    int best = -1;

    if (!f || len < 12 + 44) {
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
            int dp = SPR_MAX(sm.xdpi, sm.ydpi);
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

uint8_t *box_sprite_load(const char *path, bool dense, size_t *w, size_t *h, int *dpi)
{
    FILE *fp = fopen(path, "rb");
    uint8_t *f, *out;
    long len;

    if (!fp) {
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0 || (len = ftell(fp)) <= 0 || len > (64L << 20)) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    f = malloc((size_t)len);
    if (!f || fread(f, 1, (size_t)len, fp) != (size_t)len) {
        free(f);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    out = box_sprite_decode_file(f, (size_t)len, dense, w, h, dpi);
    free(f);
    return out;
}

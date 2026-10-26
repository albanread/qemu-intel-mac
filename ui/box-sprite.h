/* SPDX-License-Identifier: MIT
 * box-sprite.h -- RISC OS sprite files as RGBA8 pictures (box-sprite.c),
 * shared by QEMU's Metal renderer and the PC's GPU presenter. */
#ifndef BOX_SPRITE_H
#define BOX_SPRITE_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* A sprite file's picture, as RGBA8 rows from the top, square pixels,
 * masked pixels laid on the key colour #B7C0B4; NULL if it cannot be
 * read.  dense prefers a 180 dpi sibling (NAME11) where the file has
 * one.  *dpi is the picture's density: 90 is a RISC OS pixel.  The
 * caller frees the result. */
uint8_t *box_sprite_load(const char *path, bool dense, size_t *w, size_t *h, int *dpi);
uint8_t *box_sprite_decode_file(const uint8_t *f, size_t len, bool dense,
                                size_t *w, size_t *h, int *dpi);
#endif

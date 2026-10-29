/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Alban Read */
/* scene3d.h -- a 3D scene the presenter draws (ROS_SURF_SCENE3D).
 *
 * PROPOSED: kept in the Antiquity repository (docs/scene3d.h) until the
 * presenters are wired; it then moves to rosgd/include/rosgd/scene3d.h
 * unchanged.  docs/SCENE3D.md is the design: what every field means, how
 * the presenters draw it, and the CPU twin.  Like tower.h it is MIT, and
 * nothing in it comes from 0 A.D.'s code: it is a data format.
 *
 * A game (Antiquity: 0 A.D.'s engine in Rust) publishes its scene as data
 * and the presenter draws it: the PC's presenter in GLES 3.0, the Mac's
 * host in Metal.  The guest never touches a GPU.
 *
 * TWO KINDS OF BLOCK, each in backings of its own (modules/wimp/backing.c):
 *
 *  - THE LIVE BLOCK (struct ros_scene3d, about 9.9 MB): the camera, the
 *    light, the player colours, the terrain (heights, tiles, the splats it
 *    is drawn with and the LOS grid that shades it), the visible patches,
 *    the visible model instances and the bone matrices they are skinned
 *    with.  Some of it changes every frame, the rest rarely; each section
 *    has a generation.
 *
 *  - ASSET PAGES (struct ros_s3_page and its records, up to 16 MB each, up
 *    to 8 of them): meshes, textures (in the DDS's own block form, BC1-BC3,
 *    or RGBA8), skins (which textures and shading a mesh is drawn with),
 *    terrain textures.  A page is APPEND-ONLY within an epoch: records are
 *    added at its end and never changed, so the broker copies only the new
 *    tail and the presenter uploads only the new records.  A new map (a new
 *    asset set) is a new epoch: every page starts again empty.
 *
 *  - THE HUD, at the live block's end (version 2): a list of 2D quads
 *    drawn over the scene in order, each a rectangle of a texture from the
 *    pages (icons, portraits, a font's glyphs) or of the HUD's own canvas
 *    (the minimap: an RGBA picture the engine redraws as it likes), tinted
 *    and blended.  The engine lays the HUD out; the presenter only draws
 *    rectangles.
 *
 * THE VERSION.  Version 2 added the HUD at the live block's end and changed
 * nothing before it.  A broker that copies the HUD tells the engine so
 * ("scene3d LIVE PAGES 2"); an engine told only "scene3d LIVE PAGES" writes
 * version 1 and no HUD, so that a box built before keeps drawing the scene.
 * Brokers and presenters take either: a version 1 block is read only to
 * ROS_S3_V1_SIZE and has no HUD.  Pages stay at ROS_S3_PAGE_VERSION.
 *
 * The live block names the pages (page[]: the broker writes each backing
 * and offset, the engine each size and gen).  The surface's 32 bytes
 * (struct ros_surf_scene3d, Wimp_Extend &5204 R5) name the live block only.
 * On the PC a block is at byte 60 of /dev/rosgd-backing-<id>; on the Mac at
 * its offset in the backings region.  The icon's box on the screen is the
 * viewport: the scene fills it.
 *
 * THE SEQLOCK is `frame`, as tiles.h's and tower.h's: odd while the live
 * block is being written, even when it may be read.  A reader that sees an
 * odd frame, or a different one either side of its copy, draws the last
 * whole frame.  A reader copies and uploads only the sections whose gen[]
 * moved (the terrain by patch row, patch_gen[]), only the first ninstances
 * instances and nbones bone matrices, and of each page only the records
 * past those it has.
 *
 * MATRICES are float, in two forms.  A 4x4 (view, proj) is sixteen floats
 * in GL's order, column after column (m[c*4 + r]).  A 3x4 (an instance's
 * transform, a bone) is three rows of four floats: row r is (m_r0, m_r1,
 * m_r2, m_r3), so that p' = (dot(row0, p), dot(row1, p), dot(row2, p)) for
 * p = (x, y, z, 1); one RGBA32F texel a row.
 *
 * World space is 0 A.D.'s: x east, y up, z north, in the engine's units (a
 * tile is 4; a man is about 3.9 tall); the terrain's vertex (i, j) is at
 * x = 4i, z = 4j, y = height * ROS_S3_HEIGHT_SCALE.  World and eye space
 * are left-handed (the eye looks along +z).  Clip space is GL's (z -1 to
 * 1, y up, w = eye z); a Metal presenter maps z to (z + w) / 2 itself.
 *
 * TEXTURES are addressed as GL addresses them: the first row of a level's
 * data is v = 0.  The engine has already turned the meshes' v over (as
 * 0 A.D.'s ModelRenderer::BuildUV does), so no presenter flips anything.
 *
 * NOTHING HERE IS TRUSTED BY THE PRESENTER.  Every count is clamped, every
 * index masked or checked against its table, every record's bytes kept
 * inside its page.
 */
#ifndef ROSGD_SCENE3D_H
#define ROSGD_SCENE3D_H

#include <stdint.h>

#define ROS_SURF_SCENE3D      9u                /* after ROS_SURF_BATTERY (8) */
#define ROS_S3_MAGIC          0x44334353u       /* "SC3D" */
#define ROS_S3_PAGE_MAGIC     0x50334353u       /* "SC3P" */
#define ROS_S3_VERSION        2u                /* the live block's: 2 carries the HUD */
#define ROS_S3_VERSION_MIN    1u                /* 1: the same block without the HUD */
#define ROS_S3_PAGE_VERSION   1u                /* an asset page's (unchanged by 2) */
#define ROS_S3_V1_SIZE        8408616u          /* a version 1 live block's bytes: to los[]'s end */

/* ---- sizes ---------------------------------------------------------------- */

#define ROS_S3_PATCH        16                  /* tiles a patch side (0 A.D.'s PATCH_SIZE) */
#define ROS_S3_PATCHES      32                  /* patches a map side at most: 512 tiles, "giant" */
#define ROS_S3_TILES        512                 /* tiles a map side at most */
#define ROS_S3_VERTS        513                 /* height rows (and heights a row) at most */
#define ROS_S3_HSTRIDE      520                 /* heights a row in the block */
#define ROS_S3_TERRAINS     256                 /* terrain textures one map uses */
#define ROS_S3_INSTANCES    16384               /* a 2048 x 32 texture, 4 texels each */
#define ROS_S3_BONES        49152               /* a 1536 x 96 RGBA32F texture, 3 texels each */
#define ROS_S3_PLAYERS      16                  /* 0 gaia, 1-8 players; the rest spare */
#define ROS_S3_PAGES        8
#define ROS_S3_PAGE_MAX     (16u << 20)         /* a page's bytes at most */
#define ROS_S3_RECORDS      2048                /* records a page */
#define ROS_S3_NONE         0xFFFFFFFFu         /* "no bone matrices" in an instance */
#define ROS_S3_SPLATS       786432              /* terrain quads: 3 a tile of a giant map */

#define ROS_S3_UNITS        4.0f                /* metres a tile (TERRAIN_TILE_SIZE) */
#define ROS_S3_HEIGHT_SCALE (1.0f / 92.0f)      /* metres a height unit (HEIGHT_SCALE) */

/* asset id spaces: 1 .. N-1, 0 is "none" */
#define ROS_S3_MESH_IDS     8192
#define ROS_S3_TEX_IDS      16384
#define ROS_S3_SKIN_IDS     8192
#define ROS_S3_TERRAIN_IDS  1024

/* ---- the live block's sections, for gen[] -------------------------------- */

enum {
    S3_SCENE,           /* view, env, player colours, the header's counts */
    S3_TLIST,           /* terrain[]: the map's terrain textures */
    S3_TERRAIN,         /* height[] and tile[], whole; patch_gen[] says which rows */
    S3_INST,            /* inst[0 .. ninstances) */
    S3_BONES,           /* bone[0 .. nbones) */
    S3_SPLATS,          /* splat_first[] and splat[] */
    S3_LOS,             /* los[] */
    S3_HUD,             /* hud and quad[0 .. hud.nquads) */
    S3_CANVAS,          /* canvas[0 .. hud.canvas_h) rows */
    S3_NSECT
};

/* ---- the header's flags ---------------------------------------------------- */

#define ROS_S3_F_WIREFRAME  1u                  /* debugging: draw edges only */
#define ROS_S3_F_NO_TERRAIN 2u                  /* models only (the actor viewer) */

/* ---- quality: tiles.h's numbers -------------------------------------------- */
/* 0 classic (lit, no shadows), 1 smooth (shadow map, fog), 2 rich (normal,
 * specular and AO maps, post effects).  Shared with ROS_TILES_Q_*. */

/* ---- a tile (RG16UI, 512 x 512) -------------------------------------------- */
/* tex: index into terrain[] (not an asset id), 0xFFFF none (drawn black);
 * prio: the tile's blending priority as a rank (the engine maps 0 A.D.'s
 * u32 priorities to ranks, keeping their order; equal stays equal). */
struct ros_s3_tile { uint16_t tex, prio; };

/* ---- a splat: one quad of the terrain (docs/SCENE3D.md 6.2) ---------------
 * bits 0-3   i, the tile's column in its patch
 * bits 4-7   j, its row
 * bits 8-11  shape: 0-13 a blend shape of the ALPHAMAP record, drawn
 *            blended; ROS_S3_SHAPE_SOLID the tile's own texture, opaque
 * bits 12-14 orientation, ROS_S3_O_*: from the tile's own (s, t) in 0..1
 *            (s east, t north), (a, b) = SWAP ? (t, s) : (s, t), then
 *            MIRROR_A a = 1 - a, MIRROR_B b = 1 - b; the shape is sampled
 *            at (u0 + a (u1 - u0), b)
 * bits 16-31 the texture: an index into terrain[], 0xFFFF none */
#define ROS_S3_SHAPE_SOLID  15u
#define ROS_S3_O_SWAP       1u
#define ROS_S3_O_MIRROR_A   2u
#define ROS_S3_O_MIRROR_B   4u

/* ---- an instance (4 texels, RGBA32UI with floats as bits) ------------------ */
#define ROS_S3_I_CAST       0x01u               /* casts a shadow (smooth and rich) */
#define ROS_S3_I_NO_LOS     0x02u               /* drawn whatever the fog of war says */
#define ROS_S3_I_SILHOUETTE 0x04u               /* drawn as a silhouette where hidden */
#define ROS_S3_I_SELECTED   0x08u               /* spare for P5 (rings are overlays) */

struct ros_s3_instance {
    float    m[3][4];       /* model to world, rows.  Ignored when bones is not
                               ROS_S3_NONE: the bone matrices are in world space */
    uint16_t mesh, skin;    /* asset ids; a skin 0 draws the mesh plain grey */
    uint32_t bones;         /* first of the mesh's matrices in bone[], or ROS_S3_NONE */
    uint32_t shading;       /* RGBA8 0xAABBGGRR, multiplies the lit colour (CModel's
                               shading colour: white, or grey under the fog of war) */
    uint8_t  player;        /* index into player[] */
    uint8_t  flags;         /* ROS_S3_I_* */
    uint8_t  alpha;         /* 255 opaque; less fades the instance (P5) */
    uint8_t  reserved;
};

/* ---- the camera ------------------------------------------------------------ */
struct ros_s3_view {
    float view[16];         /* world to eye, column-major */
    float proj[16];         /* eye to clip, column-major, GL's clip space */
    float viewproj[16];     /* proj x view, as the engine multiplied it */
    float eye[4];           /* the camera's position, w 1 */
    float near_z, far_z;    /* metres */
    float fov_y, aspect;    /* radians; width / height of the viewport */
    uint32_t reserved[8];   /* 256 bytes */
};

/* ---- the light and the environment (0 A.D.'s CLightEnv and the map's) ------ */
struct ros_s3_env {
    float sun_dir[4];       /* the way the sunlight travels, unit, w 0 */
    float sun_rgb[4];       /* the sun's colour, w 0 */
    float ambient_rgb[4];   /* the ambient colour, w 0 */
    float fog_rgb[4];       /* the fog's colour, w 0 */
    float fog[4];           /* factor, max (the map's thickness), 0, 0 */
    float post[4];          /* brightness, contrast, saturation, bloom */
    float water_rgb[4];     /* w: murkiness */
    float water_tint[4];    /* w: waviness */
    float water[4];         /* height (metres), wind angle, 0 (no water) or 1, 0 */
    float sky_rgb[4];       /* the clear colour where no sky is drawn */
    float time[4];          /* seconds since the map began, then 0, 0, 0 */
    uint16_t sky[8];        /* the sky set's textures (P5), 0 none */
    uint32_t reserved[16];  /* 256 bytes */
};

/* ---- where an asset page is ------------------------------------------------ */
struct ros_s3_page_ref {
    uint32_t backing;       /* the broker's: the page's backing, 0 none */
    uint32_t offset;        /* the broker's: its offset in the backings region (the Mac) */
    uint32_t size;          /* the engine's: bytes used, header included */
    uint32_t gen;           /* the engine's: moves whenever records are added */
};

/* ---- the HUD (version 2) --------------------------------------------------- */
/* The HUD is drawn after the scene, over it, quad by quad in the list's
 * order, alpha-blended (source alpha, one minus source alpha), with no depth
 * test.  Its coordinates are its own: (0, 0) is the icon's top left and
 * (hud.width, hud.height) its bottom right, y down; the presenter scales
 * them to the icon's box.  A quad's texture coordinates are 0..1 across its
 * texture (v = 0 the first row), sampled linearly and clamped.  The
 * canvas's are 0..1 across all ROS_S3_CANVAS texels, used or not. */

#define ROS_S3_QUADS        8192
#define ROS_S3_CANVAS       512                 /* the canvas's side: RGBA8, 512 x 512 */
#define ROS_S3_HUD_CANVAS   0xFFFFu             /* a quad's tex: the canvas */

/* a quad's flags */
#define ROS_S3_Q_COVERAGE   0x0001u             /* the texel's red is a coverage (a glyph,
                                                   R8): the colour is rgba with its alpha
                                                   times that coverage */
#define ROS_S3_Q_GREY       0x0002u             /* the texel made grey first (a disabled
                                                   button): luma = (r*77 + g*150 + b*29) / 256 */
#define ROS_S3_Q_GRADIENT   0x0004u             /* rgba at the top edge, rgba2 at the bottom,
                                                   between them linearly */

struct ros_s3_quad {
    float    x0, y0, x1, y1;    /* the rectangle, in the HUD's units: x0 < x1, y0 < y1 */
    float    u0, v0, u1, v1;    /* the texture's rectangle (u1 < u0 mirrors) */
    uint16_t tex;               /* a TEXTURE id; 0 none (a plain rgba rectangle);
                                   ROS_S3_HUD_CANVAS the canvas */
    uint16_t flags;             /* ROS_S3_Q_* */
    uint32_t rgba;              /* 0xAABBGGRR, multiplies the texel (straight alpha) */
    uint32_t rgba2;             /* the bottom edge's, with ROS_S3_Q_GRADIENT */
    uint32_t reserved;
};

struct ros_s3_hud {
    uint32_t nquads;            /* quad[] used, 0 .. ROS_S3_QUADS; 0 no HUD */
    uint16_t width, height;     /* the HUD's units across and down the icon's box */
    uint16_t canvas_w, canvas_h;/* the canvas's texels used (the rest undefined) */
    uint32_t flags;             /* none yet */
    uint32_t reserved[12];      /* 64 bytes */
};

/* ---- the live block -------------------------------------------------------- */

struct ros_scene3d {
    uint32_t magic, version;
    uint32_t frame;                 /* the seqlock */
    uint32_t epoch;                 /* the asset set this frame was made for; pages of
                                       another epoch are not drawn from */
    uint16_t patches;               /* the map's patches a side, 0 .. 32; 0 no terrain */
    uint16_t nterrains;             /* terrain[] entries used */
    uint16_t npages;                /* page[] entries used */
    uint8_t  quality;               /* ROS_TILES_Q_* */
    uint8_t  rate;                  /* frames a second the presenter redraws unasked (water) */
    uint32_t ninstances;            /* inst[] used, 0 .. ROS_S3_INSTANCES */
    uint32_t nbones;                /* bone[] used, 0 .. ROS_S3_BONES */
    uint32_t flags;                 /* ROS_S3_F_* */
    uint32_t back_rgba;             /* 0xAABBGGRR: past the map's edge */
    uint32_t reserved0[2];
    uint32_t gen[16];               /* each section's generation, S3_SCENE .. */
    uint32_t patch_gen[ROS_S3_PATCHES];   /* each patch row's: heights rows 16r .. 16r+16
                                             and tile rows 16r .. 16r+15 */
    struct ros_s3_page_ref page[ROS_S3_PAGES];
    uint32_t patch_vis[ROS_S3_PATCHES];   /* this frame's visible patches: bit px of
                                             row pz (S3_SCENE) */
    uint32_t reserved1[4];          /* the header is 512 bytes */
    struct ros_s3_view view;                                    /*       256 */
    struct ros_s3_env  env;                                     /*       256 */
    uint32_t player[ROS_S3_PLAYERS];  /* RGBA8 0xAABBGGRR */    /*        64 */
    uint16_t terrain[ROS_S3_TERRAINS];/* TERRAIN record ids */  /*       512 */
    uint32_t reserved2[112];                                    /*       448 */
    uint16_t height[ROS_S3_VERTS][ROS_S3_HSTRIDE];              /*   533,520 */
    struct ros_s3_tile tile[ROS_S3_TILES][ROS_S3_TILES];        /* 1,048,576 */
    struct ros_s3_instance inst[ROS_S3_INSTANCES];              /* 1,048,576 */
    float bone[ROS_S3_BONES][3][4];                             /* 2,359,296 */
    uint32_t splat_first[ROS_S3_PATCHES * ROS_S3_PATCHES + 1];  /*     4,100 */
                                    /* patch (px, pz)'s splats are splat[first[p] ..
                                       first[p + 1]), p = pz * 32 + px; the last is the
                                       total.  Each patch's opaque splats come first */
    uint32_t reserved3[3];
    uint32_t splat[ROS_S3_SPLATS];                              /* 3,145,728 */
    uint8_t  los[ROS_S3_VERTS][ROS_S3_HSTRIDE];                 /*   266,760 */
                                    /* a vertex's light, 0 black .. 255 lit (R8, linear,
                                       clamped): the map's edge (P3) and the fog of war */
    uint32_t reserved4[2];                                      /*         8 */
    struct ros_s3_hud hud;                  /* version 2 */     /*        64 */
    struct ros_s3_quad quad[ROS_S3_QUADS];                      /*   393,216 */
    uint32_t canvas[ROS_S3_CANVAS][ROS_S3_CANVAS];              /* 1,048,576 */
                                    /* RGBA8 0xAABBGGRR, rows from the top */
};

_Static_assert(sizeof(struct ros_s3_tile) == 4, "a tile is 4 bytes");
_Static_assert(sizeof(struct ros_s3_instance) == 64, "an instance is 64 bytes");
_Static_assert(sizeof(struct ros_s3_view) == 256, "the view is 256 bytes");
_Static_assert(sizeof(struct ros_s3_env) == 256, "the environment is 256 bytes");
_Static_assert(__builtin_offsetof(struct ros_scene3d, gen) == 48, "gen at 48");
_Static_assert(__builtin_offsetof(struct ros_scene3d, page) == 240, "pages at 240");
_Static_assert(__builtin_offsetof(struct ros_scene3d, view) == 512, "the header is 512 bytes");
_Static_assert(__builtin_offsetof(struct ros_scene3d, player) == 1024, "player colours");
_Static_assert(__builtin_offsetof(struct ros_scene3d, terrain) == 1088, "terrain list");
_Static_assert(__builtin_offsetof(struct ros_scene3d, height) == 2048, "heights");
_Static_assert(__builtin_offsetof(struct ros_scene3d, tile) == 535568, "tiles");
_Static_assert(__builtin_offsetof(struct ros_scene3d, inst) == 1584144, "instances");
_Static_assert(__builtin_offsetof(struct ros_scene3d, bone) == 2632720, "bones");
_Static_assert(__builtin_offsetof(struct ros_scene3d, patch_vis) == 368, "visible patches");
_Static_assert(__builtin_offsetof(struct ros_scene3d, splat_first) == 4992016, "splats");
_Static_assert(__builtin_offsetof(struct ros_scene3d, splat) == 4996128, "the splat pool");
_Static_assert(__builtin_offsetof(struct ros_scene3d, los) == 8141856, "the LOS grid");
_Static_assert(sizeof(struct ros_s3_quad) == 48, "a quad is 48 bytes");
_Static_assert(sizeof(struct ros_s3_hud) == 64, "the HUD's header is 64 bytes");
_Static_assert(__builtin_offsetof(struct ros_scene3d, hud) == 8408624, "the HUD");
_Static_assert(__builtin_offsetof(struct ros_scene3d, quad) == 8408688, "the quads");
_Static_assert(__builtin_offsetof(struct ros_scene3d, canvas) == 8801904, "the canvas");
_Static_assert(sizeof(struct ros_scene3d) == 9850480, "the live block is 9,850,480 bytes");

/* ---- an asset page --------------------------------------------------------- */

enum {
    S3_R_NONE,
    S3_R_MESH,          /* p: nverts, nindices, nbones (the palette is nbones + 1), flags */
    S3_R_TEXTURE,       /* p: width | height << 16, format | levels << 8, flags */
    S3_R_SKIN,          /* p: base | norm << 16, spec | ao << 16, shading (ROS_S3_K_*),
                              object colour RGBA8 */
    S3_R_TERRAIN,       /* p: diffuse | norm << 16, spec, base colour RGBA8,
                              its ALPHAMAP's id (0 the first, standard), 0.  Its 16 bytes: the
                              texture matrix, float u = a x + b z, v = c x + d z
                              from world x, z: a, b, c, d */
    S3_R_ALPHAMAP,      /* p: texture id, shape side, shape stride (side + 4), shapes (14) */
    S3_R_NKINDS
};

/* a mesh's flags (p[3]) */
#define ROS_S3_M_SKINNED    1u                  /* its vertices name bones */

/* a mesh's vertex: 48 bytes.  bone[i] 255 ends the list; bone == nbones is
 * the bind-shape slot (the instance's own transform, kept last in its
 * matrices); weight[] are unorm8 and sum to 255. */
struct ros_s3_vertex {
    float   pos[3];
    float   norm[3];
    float   uv0[2];         /* the first UV set, v already turned over */
    float   uv1[2];         /* the second (the AO map's), else the first again */
    uint8_t bone[4];
    uint8_t weight[4];
};
/* after the vertices, nindices uint16 indices, three a triangle,
 * front faces counter-clockwise as 0 A.D. winds them, padded to 16 bytes. */

/* texture formats (p[1] bits 0-7); the data is every level, largest first,
 * each tightly packed (BC: 4x4 blocks, rows of blocks; RGBA8: rows) */
enum { S3_TF_BC1, S3_TF_BC1A, S3_TF_BC2, S3_TF_BC3, S3_TF_RGBA8, S3_TF_R8, S3_TF_N };

/* a texture's flags (p[2]) */
#define ROS_S3_T_WRAP       1u                  /* repeat (models, terrain); else clamp */
#define ROS_S3_T_ALPHA      2u                  /* its alpha means something */

/* a skin's shading (p[2] bits 0-15) */
#define ROS_S3_K_PLAYER     0x0001u             /* base alpha: 0 player colour, 1 the texture */
#define ROS_S3_K_OBJECT     0x0002u             /* base alpha: 0 object colour, 1 the texture */
#define ROS_S3_K_TRANS      0x0004u             /* alpha tested (and blended in rich) */
#define ROS_S3_K_WIND       0x0008u             /* sways (trees, crops) */
#define ROS_S3_K_GLOW       0x0010u             /* self-lit by the specular map's alpha */
#define ROS_S3_K_PARALLAX   0x0020u             /* the normal map's alpha is a height */
#define ROS_S3_K_WATER      0x0040u             /* model_water / waterfall: drawn as water (P5) */
#define ROS_S3_K_NOSHADOW   0x0080u             /* receives no shadow */

struct ros_s3_record {
    uint16_t kind, id;      /* S3_R_*; the id in the kind's space */
    uint32_t offset;        /* its bytes, from the page's start; 16-aligned */
    uint32_t size;
    uint32_t p[5];          /* by kind, above */
};

struct ros_s3_page {
    uint32_t magic, version;
    uint32_t epoch;         /* the live block's epoch it belongs to */
    uint32_t index;         /* 0 .. 7 */
    uint32_t size;          /* bytes used, header included (= the live block's page[].size) */
    uint32_t nrecords;
    uint32_t gen;
    uint32_t reserved[9];   /* 64 bytes */
    struct ros_s3_record rec[ROS_S3_RECORDS];
    /* then the records' bytes, from ROS_S3_PAGE_DATA */
};

#define ROS_S3_PAGE_DATA    ((uint32_t)sizeof(struct ros_s3_page))

_Static_assert(sizeof(struct ros_s3_vertex) == 48, "a vertex is 48 bytes");
_Static_assert(sizeof(struct ros_s3_record) == 32, "a record is 32 bytes");
_Static_assert(__builtin_offsetof(struct ros_s3_page, rec) == 64, "the page header is 64 bytes");
_Static_assert(sizeof(struct ros_s3_page) == 65600, "a page's data starts at 65,600");

/* ---- the surface's 32 bytes (Wimp_Extend &5204 R5) ------------------------- */

struct ros_surf_scene3d {
    uint32_t backing, offset, size, frame;      /* the live block */
    uint32_t epoch;                             /* its epoch, so a presenter can drop
                                                   the last map's assets at once */
    uint32_t reserved[3];
};

_Static_assert(sizeof(struct ros_surf_scene3d) == 32, "the surface's data is 32 bytes");

#endif

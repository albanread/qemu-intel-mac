/*
 * Human Monitor rdb-* stubs, for machines without the RISC OS debugger
 *
 * The rdb-* commands (hw/misc/rdb.c, DEBUGDESIGN.md) are registered in
 * hmp-commands.hx for every target, but rdb.c is built only with
 * CONFIG_RASPI.  Any other target -- ROSGD's x86-64 guest among them --
 * would otherwise fail to link.  As a stub these link only where rdb.c is
 * absent, and they answer rather than assert: the commands still appear in
 * the monitor's table, so a user can type them.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "monitor/hmp.h"
#include "monitor/monitor.h"

#define RDB_STUB(cmd) \
    void hmp_rdb_##cmd(Monitor *mon, const QDict *qdict) \
    { \
        monitor_printf(mon, "rdb-" #cmd ": the RISC OS debugger is only " \
                       "built for Raspberry Pi machines\n"); \
    }

RDB_STUB(stop)
RDB_STUB(cont)
RDB_STUB(step)
RDB_STUB(regs)
RDB_STUB(dis)
RDB_STUB(mem)
RDB_STUB(memw)
RDB_STUB(fill)
RDB_STUB(regw)
RDB_STUB(bp)
RDB_STUB(bpc)
RDB_STUB(wp)
RDB_STUB(list)

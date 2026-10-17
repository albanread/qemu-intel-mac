/*
 * metal-browser.m — the ROSGD box's browser engine, the Mac's own WebKit
 * (RISCOSGrandDesign design 39, sprint 5).
 *
 * On a PC, !Browser (WaylandWindows' browser part, in RISC OS) steers a
 * WPE WebKit program in the box's Linux root.  On the Mac the window does
 * it with WKWebView: video decodes in hardware, WebGL runs on Metal, and
 * nothing of the engine is ours to keep.  !Browser is unchanged: it speaks
 * the same lines (rosgd ports/browser/rosgd-browser.c) over the host
 * channel, the last 64 KB of the backings region (rosgd
 * include/rosgd/hostchan.h), which the board file maps once.  Here, as
 * the compositor would, the window makes each browser window ("wm new"),
 * and answers each one's lines as the engine would.
 *
 * Each view is in a borderless child window of the QEMU window, placed at
 * frame time where the Wimp puts the window's page (its ROS_SURF_PAGE
 * entry in block v2) and masked to what of it can be seen, so a RISC OS
 * window in front of it covers it.  The child window is a real window:
 * clicks, keys and scrolling go to the page natively.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>
#import <WebKit/WebKit.h>
#include <objc/runtime.h>

#include "ui/metal.h"

/* ---- the host channel: rosgd include/rosgd/hostchan.h, the other side ---- */

#define HC_SIZE          65536u
#define HC_MAGIC         0x48435352u
#define HC_RING          32736u
#define HC_HOST_BROWSER  1u

typedef struct HcRing {
    uint32_t head, tail;
    uint8_t data[HC_RING];
} HcRing;

typedef struct HostChan {
    uint32_t magic, box_gen, host_caps, host_beat;
    HcRing to_host, to_box;
} HostChan;

_Static_assert(sizeof(HostChan) <= HC_SIZE, "host channel room");

static HostChan *chan;
static uint32_t chan_gen;

static void hc_say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

static void hc_say(const char *fmt, ...)
{
    char line[4096];
    va_list ap;
    int n;
    HcRing *r;
    uint32_t head, tail;

    if (!chan) {
        return;
    }
    va_start(ap, fmt);
    n = vsnprintf(line, sizeof(line) - 1, fmt, ap);
    va_end(ap);
    if (n < 0) {
        return;
    }
    if (n > (int)sizeof(line) - 2) {
        n = (int)sizeof(line) - 2;
    }
    for (int i = 0; i < n; i++) {
        if (line[i] == '\n' || line[i] == '\r') {
            line[i] = ' ';
        }
    }
    line[n++] = '\n';
    r = &chan->to_box;
    head = r->head;
    tail = __atomic_load_n(&r->tail, __ATOMIC_ACQUIRE);
    if (HC_RING - (head - tail) < (uint32_t)n) {
        metal_log("browser: the box is not reading its lines: one dropped");
        return;
    }
    for (int i = 0; i < n; i++) {
        r->data[(head + (uint32_t)i) % HC_RING] = (uint8_t)line[i];
    }
    __atomic_store_n(&r->head, head + (uint32_t)n, __ATOMIC_RELEASE);
}

/* the box's next whole line into out: true, or false for none */
static bool hc_take(char *out, size_t max)
{
    HcRing *r = &chan->to_host;
    uint32_t tail = r->tail, head = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
    uint32_t n = 0;
    size_t k;

    while (tail + n != head && r->data[(tail + n) % HC_RING] != '\n') {
        n++;
    }
    if (tail + n == head) {
        return false;
    }
    k = n < max - 1 ? n : max - 1;
    for (size_t i = 0; i < k; i++) {
        out[i] = (char)r->data[(tail + (uint32_t)i) % HC_RING];
    }
    out[k] = 0;
    __atomic_store_n(&r->tail, tail + n + 1, __ATOMIC_RELEASE);
    return true;
}

/* ---- a browser window ------------------------------------------------------ */

#define START_PAGE "https://www.riscosopen.org/"
#define HB_FIRST_ID 4096u           /* above any compositor's ids */

/* where the page's scroll bars go and how it says where it is: the WPE
 * engine's style and script (rosgd-browser.c), and the link under the
 * pointer for !Browser's menus */
static NSString *const hb_css =
    @"::-webkit-scrollbar{display:none!important;width:0!important;height:0!important}"
    @"html{scrollbar-width:none!important}";
static NSString *const hb_js =
    @"(function(){var t=0;"
    @"function r(){t=0;var d=document.documentElement,b=document.body;"
    @"var h=Math.max(d?d.scrollHeight:0,b?b.scrollHeight:0,innerHeight);"
    @"window.webkit.messageHandlers.rosgd.postMessage('scroll '+Math.round(scrollY)+' '+h+' '+innerHeight);}"
    @"function q(){if(!t)t=requestAnimationFrame(r);}"
    @"addEventListener('scroll',q,{passive:true});addEventListener('resize',q);"
    @"addEventListener('load',q);addEventListener('DOMContentLoaded',function(){"
    @"try{new ResizeObserver(q).observe(document.documentElement);}catch(e){}q();});q();"
    @"var last='';addEventListener('mouseover',function(e){var a=e.target.closest?e.target.closest('a[href]'):null;"
    @"var i=e.target.tagName=='IMG'?e.target.src:'';var k=(a?a.href:'')+'|'+i;if(k==last)return;last=k;"
    @"var m=window.webkit.messageHandlers.rosgd;m.postMessage('hit');"
    @"if(a)m.postMessage('hitlink '+a.href);if(i)m.postMessage('hitimage '+i);},true);})();";

@class HBWindow;
static void hb_close(HBWindow *w, bool tell);
static HBWindow *hb_new(NSString *url, WKWebViewConfiguration *cfg);

/* a borderless window that takes the keyboard when the page is clicked */
@interface HBPanel : NSWindow
@end
@implementation HBPanel
- (BOOL)canBecomeKeyWindow { return YES; }
- (BOOL)canBecomeMainWindow { return NO; }
@end

@interface HBWindow : NSObject <WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler,
                                WKDownloadDelegate>
@property(nonatomic) uint32_t ident;
@property(nonatomic, retain) HBPanel *panel;
@property(nonatomic, retain) NSView *box;           /* the panel's content: the mask's */
@property(nonatomic, retain) WKWebView *view;
@property(nonatomic, retain) CAShapeLayer *mask;
@property(nonatomic) bool shown;
@property(nonatomic) NSRect placed;                 /* the panel's frame last set */
@property(nonatomic) uint32_t mask_sig;             /* the mask's rectangles, hashed */
@property(nonatomic, copy) NSString *scroll;        /* the page's last report */
@property(nonatomic, copy) NSString *wanted;        /* the address asked for */
@property(nonatomic, copy) NSString *find;
@end

static NSMutableArray<HBWindow *> *windows;
static uint32_t next_id = HB_FIRST_ID;
static NSWindow *parent_window;

static HBWindow *hb_by_id(uint32_t id)
{
    for (HBWindow *w in windows) {
        if (w.ident == id) {
            return w;
        }
    }
    return nil;
}

@implementation HBWindow

- (void)say:(NSString *)line
{
    hc_say("b%u %s", self.ident, line.UTF8String);
}

- (void)sayState
{
    NSString *u = self.view.URL.absoluteString ?: self.wanted ?: @"";
    NSString *t = self.view.title.length ? self.view.title : (u.length ? u : @"Browser");

    [self say:[@"url " stringByAppendingString:u]];
    [self say:[@"title " stringByAppendingString:t]];
    [self say:[NSString stringWithFormat:@"nav %d %d", self.view.canGoBack, self.view.canGoForward]];
    [self say:[NSString stringWithFormat:@"zoom %d", (int)lround(self.view.pageZoom * 100)]];
    [self say:self.view.loading ? @"load started" : @"load finished"];
    if (self.scroll) {
        [self say:[@"scroll " stringByAppendingString:self.scroll]];
    }
}

- (void)observeValueForKeyPath:(NSString *)key ofObject:(id)o change:(NSDictionary *)c
                       context:(void *)ctx
{
    if ([key isEqualToString:@"title"]) {
        NSString *t = self.view.title.length ? self.view.title
                                             : (self.view.URL.absoluteString ?: @"Browser");
        [self say:[@"title " stringByAppendingString:t]];
        hc_say("wm title %u %s", self.ident, t.UTF8String);
    } else if ([key isEqualToString:@"URL"]) {
        if (self.view.URL) {
            self.wanted = self.view.URL.absoluteString;
        }
        [self say:[@"url " stringByAppendingString:self.wanted ?: @""]];
        [self say:[NSString stringWithFormat:@"nav %d %d", self.view.canGoBack,
                   self.view.canGoForward]];
    } else if ([key isEqualToString:@"estimatedProgress"]) {
        [self say:[NSString stringWithFormat:@"progress %d",
                   (int)lround(self.view.estimatedProgress * 100)]];
    } else if ([key isEqualToString:@"canGoBack"] || [key isEqualToString:@"canGoForward"]) {
        [self say:[NSString stringWithFormat:@"nav %d %d", self.view.canGoBack,
                   self.view.canGoForward]];
    }
}

/* ---- navigation --------------------------------------------------------- */

- (void)webView:(WKWebView *)v didStartProvisionalNavigation:(WKNavigation *)n
{
    [self say:@"load started"];
}

- (void)webView:(WKWebView *)v didFinishNavigation:(WKNavigation *)n
{
    [self say:@"load finished"];
    [self sayState];
}

- (void)failed:(NSError *)e
{
    /* Stop, and a load that a download or a new load took over, are the
     * browser working, not failing (rosgd-browser.c's rule) */
    if ([e.domain isEqualToString:NSURLErrorDomain] && e.code == NSURLErrorCancelled) {
        [self say:@"load finished"];
        return;
    }
    if ([e.domain isEqualToString:@"WebKitErrorDomain"] && (e.code == 102 || e.code == 204)) {
        [self say:@"load finished"];
        return;
    }
    [self say:@"load finished"];
    [self say:[@"error " stringByAppendingString:e.localizedDescription ?: @"the page did not come"]];
}

- (void)webView:(WKWebView *)v didFailProvisionalNavigation:(WKNavigation *)n withError:(NSError *)e
{
    [self failed:e];
}

- (void)webView:(WKWebView *)v didFailNavigation:(WKNavigation *)n withError:(NSError *)e
{
    [self failed:e];
}

- (void)webViewWebContentProcessDidTerminate:(WKWebView *)v
{
    [self say:@"load finished"];
    [self say:@"error This page stopped: its web process has gone. Reload tries it again"];
}

/* a response the view cannot show is a download, to the share's Downloads */
- (void)webView:(WKWebView *)v decidePolicyForNavigationResponse:(WKNavigationResponse *)r
                                                 decisionHandler:(void (^)(WKNavigationResponsePolicy))h
{
    h(r.canShowMIMEType ? WKNavigationResponsePolicyAllow : WKNavigationResponsePolicyDownload);
}

- (void)webView:(WKWebView *)v navigationResponse:(WKNavigationResponse *)r
                               didBecomeDownload:(WKDownload *)d
{
    d.delegate = self;
}

- (void)webView:(WKWebView *)v navigationAction:(WKNavigationAction *)a
                             didBecomeDownload:(WKDownload *)d
{
    d.delegate = self;
}

- (void)download:(WKDownload *)d decideDestinationUsingResponse:(NSURLResponse *)r
                                             suggestedFilename:(NSString *)name
                                             completionHandler:(void (^)(NSURL *))h
{
    const char *share = metal_glue_share_path();
    NSString *dir, *base, *path;
    NSFileManager *fm = [NSFileManager defaultManager];

    if (!share) {
        [self say:@"download failed: the box has no share"];
        h(nil);
        return;
    }
    dir = [[NSString stringWithUTF8String:share] stringByAppendingPathComponent:@"Downloads"];
    [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
    base = [(name.length ? name : @"download") stringByReplacingOccurrencesOfString:@"/"
                                                                         withString:@"_"];
    path = [dir stringByAppendingPathComponent:base];
    for (int n = 1; [fm fileExistsAtPath:path] && n < 1000; n++) {
        NSString *stem = base.stringByDeletingPathExtension, *ext = base.pathExtension;
        NSString *nm = ext.length ? [NSString stringWithFormat:@"%@-%d.%@", stem, n, ext]
                                  : [NSString stringWithFormat:@"%@-%d", stem, n];
        path = [dir stringByAppendingPathComponent:nm];
    }
    [self say:[@"download started " stringByAppendingString:path.lastPathComponent]];
    objc_setAssociatedObject(d, "rosgd-name", path.lastPathComponent,
                             OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    h([NSURL fileURLWithPath:path]);
}

- (void)downloadDidFinish:(WKDownload *)d
{
    NSString *n = objc_getAssociatedObject(d, "rosgd-name");
    [self say:[@"download done " stringByAppendingString:n ?: @"?"]];
}

- (void)download:(WKDownload *)d didFailWithError:(NSError *)e resumeData:(NSData *)r
{
    NSString *n = objc_getAssociatedObject(d, "rosgd-name");
    [self say:[NSString stringWithFormat:@"download failed %@: %@", n ?: @"?",
               e.localizedDescription]];
}

/* window.open and target="_blank": a browser window of its own */
- (WKWebView *)webView:(WKWebView *)v createWebViewWithConfiguration:(WKWebViewConfiguration *)c
                   forNavigationAction:(WKNavigationAction *)a windowFeatures:(WKWindowFeatures *)f
{
    HBWindow *w = hb_new(nil, c);

    return w.view;
}

- (void)webViewDidClose:(WKWebView *)v
{
    hb_close(self, true);
}

/* ---- the page's own news ------------------------------------------------ */

- (void)userContentController:(WKUserContentController *)u didReceiveScriptMessage:(WKScriptMessage *)m
{
    NSString *s = [m.body isKindOfClass:[NSString class]] ? m.body : nil;

    if (!s) {
        return;
    }
    if ([s hasPrefix:@"scroll "]) {
        self.scroll = [s substringFromIndex:7];
    }
    [self say:s];
}

@end

/* ---- making and ending ------------------------------------------------------- */

static HBWindow *hb_new(NSString *url, WKWebViewConfiguration *cfg)
{
    HBWindow *w = [[HBWindow alloc] init];
    WKWebViewConfiguration *c = cfg ? [cfg retain] : [[WKWebViewConfiguration alloc] init];
    WKUserContentController *ucc;
    WKUserScript *css, *js;

    if (!cfg) {
        c.websiteDataStore = [WKWebsiteDataStore defaultDataStore];
        c.preferences.javaScriptCanOpenWindowsAutomatically = NO;
        if (@available(macOS 11.0, *)) {
            c.defaultWebpagePreferences.allowsContentJavaScript = YES;
        }
    }
    ucc = c.userContentController;
    css = [[WKUserScript alloc] initWithSource:
              [NSString stringWithFormat:@"(function(){var s=document.createElement('style');"
                                          "s.textContent='%@';document.documentElement.appendChild(s);})();",
                                          hb_css]
                                 injectionTime:WKUserScriptInjectionTimeAtDocumentStart
                              forMainFrameOnly:YES];
    js = [[WKUserScript alloc] initWithSource:hb_js
                                injectionTime:WKUserScriptInjectionTimeAtDocumentStart
                             forMainFrameOnly:YES];
    [ucc addUserScript:css];
    [ucc addUserScript:js];
    [ucc addScriptMessageHandler:w name:@"rosgd"];
    [css release];
    [js release];

    w.ident = next_id++;
    w.panel = [[[HBPanel alloc] initWithContentRect:NSMakeRect(0, 0, 640, 480)
                                          styleMask:NSWindowStyleMaskBorderless
                                            backing:NSBackingStoreBuffered
                                              defer:NO] autorelease];
    w.panel.releasedWhenClosed = NO;
    w.panel.hasShadow = NO;
    w.panel.opaque = NO;
    w.panel.backgroundColor = [NSColor clearColor];
    w.box = [[[NSView alloc] initWithFrame:NSMakeRect(0, 0, 640, 480)] autorelease];
    w.box.wantsLayer = YES;
    w.mask = [CAShapeLayer layer];
    w.box.layer.mask = w.mask;
    w.panel.contentView = w.box;
    w.view = [[[WKWebView alloc] initWithFrame:w.box.bounds configuration:c] autorelease];
    w.view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    w.view.navigationDelegate = w;
    w.view.UIDelegate = w;
    w.view.allowsBackForwardNavigationGestures = YES;
    w.view.customUserAgent = nil;
    [w.box addSubview:w.view];
    for (NSString *k in @[ @"title", @"URL", @"estimatedProgress", @"canGoBack", @"canGoForward" ]) {
        [w.view addObserver:w forKeyPath:k options:0 context:NULL];
    }
    [c release];
    if (!windows) {
        windows = [[NSMutableArray alloc] init];
    }
    [windows addObject:w];
    [w release];
    metal_log("browser: window %u made%s%s", w.ident, url ? ": " : "", url ? url.UTF8String : "");
    /* the box makes the Wimp window ("wm new"), opens it, and its page's
     * place then comes in block v2: until then the panel stays hidden */
    hc_say("wm new %u 1024 640 0 Browser", w.ident);
    if (!cfg) {
        NSURL *u = [NSURL URLWithString:url.length ? url : @START_PAGE];

        w.wanted = u.absoluteString;
        if (u.isFileURL) {
            [w.view loadFileURL:u allowingReadAccessToURL:u.URLByDeletingLastPathComponent];
        } else if (u) {
            [w.view loadRequest:[NSURLRequest requestWithURL:u]];
        }
    }
    return w;
}

static void hb_close(HBWindow *w, bool tell)
{
    uint32_t id = w.ident;

    for (NSString *k in @[ @"title", @"URL", @"estimatedProgress", @"canGoBack", @"canGoForward" ]) {
        [w.view removeObserver:w forKeyPath:k];
    }
    [w.view.configuration.userContentController removeScriptMessageHandlerForName:@"rosgd"];
    [w.view stopLoading];
    w.view.navigationDelegate = nil;
    w.view.UIDelegate = nil;
    if (w.panel.parentWindow) {
        [w.panel.parentWindow removeChildWindow:w.panel];
    }
    [w.panel orderOut:nil];
    [w.panel close];
    [windows removeObject:w];
    metal_log("browser: window %u closed", id);
    if (tell) {
        hc_say("wm gone %u", id);
    }
}

/* ---- the box's lines ----------------------------------------------------------- */

static void hb_line(char *l)
{
    if (!strncmp(l, "wm ", 3)) {
        char cmd[16] = "";
        unsigned id = 0;
        int n = 0;

        l += 3;
        if (!strncmp(l, "open", 4) && (l[4] == 0 || l[4] == ' ')) {
            hb_new(l[4] ? [NSString stringWithUTF8String:l + 5] : nil, nil);
            return;
        }
        if (sscanf(l, "%15s %u%n", cmd, &id, &n) < 2) {
            return;
        }
        HBWindow *w = hb_by_id(id);
        if (!w) {
            return;
        }
        if (!strcmp(cmd, "close")) {
            hb_close(w, true);
        } else if (!strcmp(cmd, "focus")) {
            /* the Wimp gave the window the caret (a click on its page, or
             * its title): the page takes the keyboard */
            if (w.shown) {
                [w.panel makeKeyWindow];
                [w.panel makeFirstResponder:w.view];
            }
        }
        /* configure, motion, button, axis, key, leave: the panel is a real
         * window, its input its own, and its size follows its place */
        return;
    }
    if (l[0] != 'b') {
        return;
    }
    char *e;
    unsigned long id = strtoul(l + 1, &e, 10);
    HBWindow *w = hb_by_id((uint32_t)id);
    if (!w || *e != ' ') {
        return;
    }
    l = e + 1;
    NSString *arg = nil;
    char *sp = strchr(l, ' ');
    if (sp) {
        *sp = 0;
        arg = [NSString stringWithUTF8String:sp + 1];
    }
    if (!strcmp(l, "go") && arg.length) {
        NSURL *u = [NSURL URLWithString:arg];

        if (!u.scheme) {
            u = [NSURL URLWithString:[@"https://" stringByAppendingString:arg]];
        }
        w.wanted = u.absoluteString;
        if (u.isFileURL) {
            [w.view loadFileURL:u allowingReadAccessToURL:u.URLByDeletingLastPathComponent];
        } else if (u) {
            [w.view loadRequest:[NSURLRequest requestWithURL:u]];
        }
    } else if (!strcmp(l, "back")) {
        [w.view goBack];
    } else if (!strcmp(l, "forward")) {
        [w.view goForward];
    } else if (!strcmp(l, "reload")) {
        if (w.view.URL) {
            [w.view reload];
        } else if (w.wanted) {
            [w.view loadRequest:[NSURLRequest requestWithURL:[NSURL URLWithString:w.wanted]]];
        }
    } else if (!strcmp(l, "stop")) {
        [w.view stopLoading];
        [w say:@"load finished"];
    } else if (!strcmp(l, "state")) {
        [w sayState];
    } else if (!strcmp(l, "scrollto") && arg) {
        [w.view evaluateJavaScript:[NSString stringWithFormat:@"window.scrollTo(scrollX,%d)",
                                    arg.intValue] completionHandler:nil];
    } else if (!strcmp(l, "zoom") && arg) {
        int p = MAX(25, MIN(500, arg.intValue));

        w.view.pageZoom = p / 100.0;
        [w say:[NSString stringWithFormat:@"zoom %d", p]];
    } else if (!strcmp(l, "newwindow")) {
        hb_new(arg, nil);
    } else if (!strcmp(l, "find") || !strcmp(l, "findnext") || !strcmp(l, "findprev")) {
        if (!strcmp(l, "find")) {
            w.find = arg ?: @"";
        }
        if (@available(macOS 11.0, *)) {
            WKFindConfiguration *fc = [[[WKFindConfiguration alloc] init] autorelease];

            fc.backwards = !strcmp(l, "findprev");
            fc.wraps = YES;
            [w.view findString:w.find ?: @"" withConfiguration:fc
             completionHandler:^(WKFindResult *r) {
                 [w say:r.matchFound ? @"find found" : @"find notfound"];
             }];
        }
    } else if (!strcmp(l, "findstop")) {
        [w.view evaluateJavaScript:@"window.getSelection().removeAllRanges()" completionHandler:nil];
    } else if (!strcmp(l, "save")) {
        NSURL *u = arg.length ? [NSURL URLWithString:arg] : w.view.URL;

        if (u) {
            [w.view startDownloadUsingRequest:[NSURLRequest requestWithURL:u]
                            completionHandler:^(WKDownload *d) { d.delegate = w; }];
        }
    }
}

/* ---- each frame -------------------------------------------------------------- */

/* The panels where the Wimp puts the pages: guest pixels, over the view's
 * points (the scale pass stretches the screen over the whole view), to
 * the screen; each masked to what of its page is seen, and hidden while
 * its page is not on the screen at all */
static void hb_place(NSWindow *win, NSView *view, const MetalSurfacesView *sv,
                     uint32_t xres, uint32_t yres)
{
    NSRect vb = view.bounds;
    double sx = vb.size.width / MAX(xres, 1u), sy = vb.size.height / MAX(yres, 1u);

    for (HBWindow *w in [[windows copy] autorelease]) {
        int pi = -1;

        for (uint32_t k = 0; sv && k < sv->nplaces; k++) {
            if (sv->place[k].id == w.ident) {
                pi = (int)k;
            }
        }
        if (pi < 0 || !sv->place[pi].nrects || sv->place[pi].w <= 0 || sv->place[pi].h <= 0) {
            if (w.shown) {
                [win removeChildWindow:w.panel];
                [w.panel orderOut:nil];
                w.shown = false;
            }
            continue;
        }
        double gx = sv->place[pi].x, gy = sv->place[pi].y;
        double gw = sv->place[pi].w, gh = sv->place[pi].h;
        NSRect r = NSMakeRect(gx * sx, vb.size.height - (gy + gh) * sy, gw * sx, gh * sy);
        NSRect onwin = [view convertRect:r toView:nil];
        NSRect screen = [win convertRectToScreen:onwin];
        uint32_t sig = 2166136261u;

        if (!NSEqualRects(screen, w.placed)) {
            [w.panel setFrame:screen display:NO];
            w.placed = screen;
            sig = 0;                    /* the mask again, at the new size */
        }
        /* the mask: the seen rectangles, in the panel's points, y up */
        for (uint32_t k = 0; k < sv->place[pi].nrects; k++) {
            const int16_t *q = sv->rect[sv->place[pi].rects + k];

            for (int j = 0; j < 4; j++) {
                sig = (sig ^ (uint16_t)q[j]) * 16777619u;
            }
        }
        if (sig != w.mask_sig) {
            CGMutablePathRef path = CGPathCreateMutable();

            for (uint32_t k = 0; k < sv->place[pi].nrects; k++) {
                const int16_t *q = sv->rect[sv->place[pi].rects + k];
                CGRect c = CGRectMake((q[0] - gx) * sx, (gy + gh - q[3]) * sy,
                                      (q[2] - q[0]) * sx, (q[3] - q[1]) * sy);

                CGPathAddRect(path, NULL, c);
            }
            [CATransaction begin];
            [CATransaction setDisableActions:YES];
            w.mask.frame = w.box.layer.bounds;
            w.mask.path = path;
            [CATransaction commit];
            CGPathRelease(path);
            w.mask_sig = sig;
        }
        if (!w.shown) {
            [win addChildWindow:w.panel ordered:NSWindowAbove];
            [w.panel orderFront:nil];
            w.shown = true;
        }
    }
}

/*
 * Called at every pass of the window's loop, a frame presented or not:
 * the channel looked at (the host's caps and beat written, the box's lines
 * answered) and the panels placed.  region is the board's mapping of the
 * backings region, NULL until the box publishes one.
 */
void metal_browser_tick(NSWindow *win, NSView *view, const MetalSurfacesView *sv,
                        uint32_t xres, uint32_t yres)
{
    static char line[4096];
    HostChan *c;

    parent_window = win;
    if (!sv || !sv->region || sv->region_size < (128u << 20) + HC_SIZE) {
        return;
    }
    c = (HostChan *)((uint8_t *)sv->region + sv->region_size - HC_SIZE);
    if (__atomic_load_n(&c->magic, __ATOMIC_ACQUIRE) != HC_MAGIC) {
        return;
    }
    if (chan != c || c->box_gen != chan_gen) {
        /* the box set the channel up afresh (it started again): every
         * window it knew of is gone */
        chan = c;
        chan_gen = c->box_gen;
        for (HBWindow *w in [[windows copy] autorelease]) {
            hb_close(w, false);
        }
        metal_log("browser: the box's channel, generation %u: the Mac's WebKit answers !Browser",
                  chan_gen);
    }
    __atomic_store_n(&c->host_caps, HC_HOST_BROWSER, __ATOMIC_RELEASE);
    __atomic_store_n(&c->host_beat, c->host_beat + 1, __ATOMIC_RELEASE);
    for (int n = 0; n < 64 && hc_take(line, sizeof(line)); n++) {
        hb_line(line);
    }
    hb_place(win, view, sv, xres, yres);
}

#include <ApplicationServices/ApplicationServices.h>
#include <string.h>
#include <unistd.h>

static CGKeyCode keycode(const char *key) {
    if (!strcmp(key, "a")) return 0;
    if (!strcmp(key, "c")) return 8;
    if (!strcmp(key, "v")) return 9;
    if (!strcmp(key, "l")) return 37;
    if (!strcmp(key, "enter")) return 36;
    if (!strcmp(key, "pagedown")) return 121;
    if (!strcmp(key, "pageup")) return 116;
    if (!strcmp(key, "home")) return 115;
    if (!strcmp(key, "end")) return 119;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    CGKeyCode code = keycode(argv[1]);
    CGEventFlags flags = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "cmd")) flags |= kCGEventFlagMaskCommand;
        if (!strcmp(argv[i], "shift")) flags |= kCGEventFlagMaskShift;
        if (!strcmp(argv[i], "ctrl")) flags |= kCGEventFlagMaskControl;
        if (!strcmp(argv[i], "alt")) flags |= kCGEventFlagMaskAlternate;
    }
    CGEventRef down = CGEventCreateKeyboardEvent(NULL, code, true);
    CGEventRef up = CGEventCreateKeyboardEvent(NULL, code, false);
    if (!down || !up) return 2;
    CGEventSetFlags(down, flags);
    CGEventSetFlags(up, flags);
    CGEventPost(kCGSessionEventTap, down);
    usleep(80000);
    CGEventPost(kCGSessionEventTap, up);
    CFRelease(down);
    CFRelease(up);
    usleep(150000);
    return 0;
}

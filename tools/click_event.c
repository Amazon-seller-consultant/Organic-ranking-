#include <ApplicationServices/ApplicationServices.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 3) return 1;
    double x = atof(argv[1]);
    double y = atof(argv[2]);
    CGPoint point = CGPointMake(x, y);
    CGWarpMouseCursorPosition(point);
    usleep(50000);
    CGEventRef move = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, point, kCGMouseButtonLeft);
    CGEventRef down = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseDown, point, kCGMouseButtonLeft);
    CGEventRef up = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseUp, point, kCGMouseButtonLeft);
    if (!move || !down || !up) return 2;
    CGEventPost(kCGSessionEventTap, move);
    usleep(50000);
    CGEventPost(kCGSessionEventTap, down);
    usleep(80000);
    CGEventPost(kCGSessionEventTap, up);
    CFRelease(move);
    CFRelease(down);
    CFRelease(up);
    return 0;
}

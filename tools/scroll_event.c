#include <ApplicationServices/ApplicationServices.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int amount = argc > 1 ? atoi(argv[1]) : -500;
    double x = argc > 2 ? atof(argv[2]) : 450;
    double y = argc > 3 ? atof(argv[3]) : 450;
    CGWarpMouseCursorPosition(CGPointMake(x, y));
    usleep(50000);
    CGEventRef event = CGEventCreateScrollWheelEvent(NULL, kCGScrollEventUnitPixel, 1, amount);
    if (!event) return 1;
    CGEventPost(kCGHIDEventTap, event);
    CFRelease(event);
    usleep(150000);
    return 0;
}

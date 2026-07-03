#include <ApplicationServices/ApplicationServices.h>
#include <stdio.h>

int main(void) {
    CGEventRef event = CGEventCreate(NULL);
    CGPoint p = CGEventGetLocation(event);
    printf("%.0f %.0f\n", p.x, p.y);
    CFRelease(event);
    return 0;
}

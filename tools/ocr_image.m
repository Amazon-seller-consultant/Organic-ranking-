#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <AppKit/AppKit.h>
#import <ImageIO/ImageIO.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) return 1;
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        NSURL *url = [NSURL fileURLWithPath:path];
        CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
        if (!source) return 2;
        CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(source);
        if (!cgImage) return 3;

        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] initWithCompletionHandler:^(VNRequest *req, NSError *error) {
            if (error) {
                fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
                return;
            }
            NSArray *results = req.results;
            NSMutableArray *rows = [NSMutableArray array];
            for (VNRecognizedTextObservation *obs in results) {
                VNRecognizedText *candidate = [[obs topCandidates:1] firstObject];
                if (!candidate) continue;
                NSDictionary *row = @{
                    @"text": candidate.string ?: @"",
                    @"x": @(obs.boundingBox.origin.x),
                    @"y": @(obs.boundingBox.origin.y),
                    @"w": @(obs.boundingBox.size.width),
                    @"h": @(obs.boundingBox.size.height)
                };
                [rows addObject:row];
            }
            NSData *json = [NSJSONSerialization dataWithJSONObject:rows options:0 error:nil];
            fwrite(json.bytes, 1, json.length, stdout);
        }];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.revision = VNRecognizeTextRequestRevision3;
        request.minimumTextHeight = 0.0;
        request.usesLanguageCorrection = NO;
        request.recognitionLanguages = @[@"en-US"];

        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:cgImage options:@{}];
        NSError *err = nil;
        [handler performRequests:@[request] error:&err];
        if (err) {
            fprintf(stderr, "%s\n", err.localizedDescription.UTF8String);
            CGImageRelease(cgImage);
            return 4;
        }
        CGImageRelease(cgImage);
    }
    return 0;
}

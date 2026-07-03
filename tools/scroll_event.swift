import CoreGraphics
import Foundation

let amount = Int32(CommandLine.arguments.dropFirst().first ?? "-500") ?? -500
guard let event = CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 1, wheel1: amount, wheel2: 0, wheel3: 0) else {
    exit(1)
}
event.post(tap: .cghidEventTap)
usleep(150_000)

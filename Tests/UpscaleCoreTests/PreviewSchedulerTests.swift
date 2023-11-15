import Foundation
import Testing

@testable import UpscaleCore

/// Last-request-wins scheduling for previews.
///
/// A pan produces requests faster than they can be served. What matters is not that the scheduler
/// is fast but that it is **honest about what it abandoned**: a superseded request must report
/// itself cancelled rather than silently stop emitting, or a UI observing it shows a preview that
/// appears to hang forever.
struct PreviewSchedulerTests {
    /// A request that blocks until released, so ordering is deterministic rather than timed.
    actor Gate {
        private var continuations: [CheckedContinuation<Void, Never>] = []
        private var opened = false

        func wait() async {
            if opened { return }
            await withCheckedContinuation { continuations.append($0) }
        }

        func open() {
            opened = true
            for continuation in continuations { continuation.resume() }
            continuations.removeAll()
        }
    }

    @Test("scheduling a second request cancels the first, and says so")
    func secondRequestSupersedesTheFirst() async {
        let scheduler = Scheduling.PreviewScheduler()
        let gate = Gate()

        let first = await scheduler.schedule { _ in await gate.wait() }
        _ = try? first.begin()
        let second = await scheduler.schedule { _ in await gate.wait() }

        // The abandoned request reports its own fate; a UI watching it is not left guessing.
        #expect(first.snapshot.state == .failed(.cancelled))
        // Distinguished from a real failure: the UI shows a neutral notice, not an error.
        guard case .failed(let reason) = first.snapshot.state else {
            Issue.record("expected .failed, got \(first.snapshot.state)")
            return
        }
        #expect(reason.isCancellation)
        #expect(second.snapshot.state != .failed(.cancelled))

        await gate.open()
    }

    @Test("the superseded request's task is actually cancelled, not merely relabelled")
    func supersededTaskIsCancelled() async {
        let scheduler = Scheduling.PreviewScheduler()
        let observed = Observed()

        _ = await scheduler.schedule { _ in
            while !Task.isCancelled { await Task.yield() }
            await observed.record()
        }
        _ = await scheduler.schedule { _ in }

        // The first closure only records once its task is genuinely cancelled.
        for _ in 0..<1000 where await observed.count == 0 { await Task.yield() }
        #expect(await observed.count == 1, "the superseded task never observed cancellation")
    }

    actor Observed {
        private(set) var count = 0
        func record() { count += 1 }
    }

    @Test("cancelling with no replacement leaves nothing in flight")
    func cancelCurrentClearsTheSlot() async {
        let scheduler = Scheduling.PreviewScheduler()
        let gate = Gate()

        let job = await scheduler.schedule { _ in await gate.wait() }
        #expect(await scheduler.isBusy)

        await scheduler.cancelCurrent()

        #expect(!(await scheduler.isBusy))
        #expect(job.snapshot.state == .failed(.cancelled))
        await gate.open()
    }

    /// The subtle one. A superseded task may finish *after* its replacement started; if it cleared
    /// the slot then, the scheduler would believe nothing is running while a preview computes.
    @Test("a superseded task finishing late does not clear its successor's slot")
    func lateFinisherDoesNotClearSuccessor() async {
        let scheduler = Scheduling.PreviewScheduler()
        let firstGate = Gate()
        let secondGate = Gate()

        _ = await scheduler.schedule { _ in await firstGate.wait() }
        _ = await scheduler.schedule { _ in await secondGate.wait() }

        // Let the FIRST, already-superseded closure run to completion now.
        await firstGate.open()
        for _ in 0..<200 { await Task.yield() }

        #expect(await scheduler.isBusy, "the second request is still in flight and must be tracked")
        await secondGate.open()
    }

    @Test("superseded ids are recorded newest first and bounded")
    func supersededHistoryIsBounded() async {
        let scheduler = Scheduling.PreviewScheduler()
        let gate = Gate()
        var ids: [Scheduling.JobID] = []

        for _ in 0..<12 {
            ids.append(await scheduler.schedule { _ in await gate.wait() }.id)
        }

        let history = await scheduler.recentlySuperseded
        #expect(history.count == 8, "history must not grow without bound")
        #expect(history.first == ids[10], "newest superseded first")
        await gate.open()
    }
}

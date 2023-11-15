import Foundation

extension Scheduling {
    /// Serialises preview requests, last request wins.
    ///
    /// A pan generates requests far faster than they can be served. Letting them overlap is worse
    /// than useless: they contend for the ANE, so *every* preview gets slower, and the one the user
    /// is waiting for finishes last. Only the newest request is worth computing, because it is the
    /// only one describing where the user is now looking.
    ///
    /// **Cancellation is cooperative and lands at a tile boundary.** `TilingEngine` checks at the
    /// top of its tile loop, so a superseded request stops within one tile — around 146 ms for x4v3
    /// at tile 512 on an iPhone 12 Pro. That is the honest latency; it is not instant, and a caller
    /// that needs it to be should debounce before calling rather than expect this to.
    ///
    /// The previous job is told it was cancelled through its own `JobTracker`, so a UI observing
    /// that job sees `.failed(.cancelled)` rather than a task that simply stopped emitting — the
    /// difference between a screen that explains itself and one that appears to hang.
    public actor PreviewScheduler {
        /// The request in flight, if any.
        private var current: (task: Task<Void, Never>, tracker: JobTracker)?

        /// Requests that have been superseded, newest first. Bounded; diagnostics only.
        private var supersededIDs: [JobID] = []
        private let supersededLimit = 8

        public init() {}

        /// Whether a request is currently in flight. Diagnostics and tests; not a lock.
        public var isBusy: Bool { current != nil }

        /// IDs of recently superseded requests, newest first.
        public var recentlySuperseded: [JobID] { supersededIDs }

        /// Replace whatever is running with `work`.
        ///
        /// Returns the tracker for the new request so a caller can observe it. The previous
        /// request, if any, is cancelled first — **and told so**, so its own snapshot reads
        /// `.failed(.cancelled)`.
        ///
        /// `work` receives the tracker it should report through. It is not given a cancellation
        /// flag: `Task.isCancelled` and `Task.checkCancellation()` are the mechanism, and the
        /// engine already honours them.
        @discardableResult
        public func schedule(
            id: JobID = JobID(),
            _ work: @escaping @Sendable (JobTracker) async -> Void
        ) -> JobTracker {
            supersede()

            let tracker = JobTracker(id: id)
            let task = Task { [weak self] in
                await work(tracker)
                await self?.clearIfCurrent(id)
            }
            current = (task, tracker)
            return tracker
        }

        /// Cancel the in-flight request without starting another — the user let go, or the view
        /// went away.
        public func cancelCurrent() {
            supersede()
        }

        private func supersede() {
            guard let existing = current else { return }
            // Told, then cancelled. A tracker asked to cancel after its task is already gone would
            // still be correct, but reporting first means a UI observing the old job never sees a
            // gap where it is neither running nor finished.
            _ = existing.tracker.requestCancellation()
            _ = existing.tracker.fail(.cancelled)
            existing.task.cancel()

            supersededIDs.insert(existing.tracker.id, at: 0)
            if supersededIDs.count > supersededLimit { supersededIDs.removeLast() }
            current = nil
        }

        /// Clear only if the finishing task is still the current one.
        ///
        /// A task that has been superseded must not clear its successor's slot: it may finish
        /// *after* the replacement started, and clearing then would leave the scheduler believing
        /// nothing is running while a preview is still computing.
        private func clearIfCurrent(_ id: JobID) {
            guard current?.tracker.id == id else { return }
            current = nil
        }
    }
}

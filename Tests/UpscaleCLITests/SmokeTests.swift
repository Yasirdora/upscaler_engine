import Testing

@testable import UpscaleCLI

/// Smoke tests for the `UpscaleCLI` executable target.
///
/// `Entry.run` returns its exit status and takes its output sinks as parameters, so the basic
/// contract (`swift run UpscaleCLI --version` prints a version string and exits 0) is asserted here
/// without spawning a process.
@Test("`--version` prints the version line and exits 0")
func versionFlagPrintsVersionAndExitsZero() {
    var printed: [String] = []
    let status = Entry.run(arguments: ["--version"], out: { printed.append($0) }, err: { _ in })

    #expect(status == 0)
    #expect(printed == [Version.line])
}

@Test("an unknown argument is a usage error, not a silent success")
func unknownArgumentExitsTwo() {
    var errors: [String] = []
    let status = Entry.run(arguments: ["--nope"], out: { _ in }, err: { errors.append($0) })

    #expect(status == 2)
    #expect(errors.first?.contains("--nope") == true)
}

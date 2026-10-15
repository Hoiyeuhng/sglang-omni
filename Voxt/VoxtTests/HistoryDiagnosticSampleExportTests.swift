import XCTest
@testable import Voxt

final class HistoryDiagnosticSampleExportTests: XCTestCase {
    func testTextOnly() throws { try HistoryDiagnosticSampleChecks.textOnly() }
    func testAudioOnly() throws { try HistoryDiagnosticSampleChecks.audioOnly() }
    func testBothWithExpectedTranscription() throws { try HistoryDiagnosticSampleChecks.bothAndExpected() }
    func testNoSelectionWritesNothing() throws { try HistoryDiagnosticSampleChecks.nothingSelected() }
    func testSelectedMissingAudioFailsAtomically() throws { try HistoryDiagnosticSampleChecks.missingAudio() }
    func testDestinationConflictsPreserveExistingFiles() throws { try HistoryDiagnosticSampleChecks.destinationConflicts() }
    func testCopyFailureRemovesPartialSample() throws { try HistoryDiagnosticSampleChecks.copyFailureCleanup() }
    func testCancellationRemovesPartialSample() throws { try HistoryDiagnosticSampleChecks.cancelCleanup() }
    func testPublishFailureRemovesPartialSample() throws { try HistoryDiagnosticSampleChecks.publishFailureCleanup() }
    func testCleanupFailureDisclosesPartialLocation() throws { try HistoryDiagnosticSampleChecks.cleanupFailureDisclosure() }
    func testSourcePathsRejectTraversalSymlinksAndDirectories() throws { try HistoryDiagnosticSampleChecks.sourcePathBoundaries() }
    func testOpenSourceSurvivesRetentionAndReplacement() throws { try HistoryDiagnosticSampleChecks.sourceUnlinked() }
    func testMutatedSourceCannotPublish() throws { try HistoryDiagnosticSampleChecks.sourceMutated() }
    func testManifestAndBundleExcludeUnselectedPrivateValues() throws { try HistoryDiagnosticSampleChecks.manifestPrivacy() }
    func testTaskCancellationDuringBackgroundCopy() async throws { try await HistoryDiagnosticSampleChecks.taskCancellation() }
    func testRetainedAudioPreviewDoesNotAutoplay() throws { try HistoryDiagnosticSampleChecks.retainedAudioPreview() }
    func testResolvedArchiveRemainsAvailableAfterDirectorySwitch() throws { try HistoryDiagnosticSampleChecks.previousAudioLocation() }
    func testInheritedACLDoesNotExposeStagingOrFinalSample() throws { try HistoryDiagnosticSampleChecks.inheritedACLIsolation() }
    func testPrivatePermissionFailureRemovesPartialSample() throws { try HistoryDiagnosticSampleChecks.permissionFailureCleanup() }
    func testFinalHistoryTextIsNotIdentifiedAsRawASR() throws { try HistoryDiagnosticSampleChecks.storedFinalTextOrigin() }
    func testPrivatePermissions() throws { try HistoryDiagnosticSampleChecks.privatePermissions() }
    func testSyntheticAudioProcessingToExport() throws { try HistoryDiagnosticSampleChecks.syntheticAudioRoundtrip() }
}

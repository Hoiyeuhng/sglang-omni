// BrowserContextResolver.swift
// Provides Browser Context Resolver for app lifecycle and routing.

import Foundation
import AppKit
import ApplicationServices

extension AppDelegate {
    func isBrowserBundleID(_ bundleID: String?) -> Bool {
        guard let bundleID else { return false }
        return supportedBrowserBundleIDs().contains(bundleID)
    }

    func activeBrowserTabURL(frontmostBundleID: String?) -> String? {
        guard let frontmostBundleID else { return nil }
        if let deniedUntil = browserAutomationDeniedUntilByBundleID[frontmostBundleID],
           deniedUntil > Date() {
            return nil
        }
        guard NSRunningApplication.runningApplications(withBundleIdentifier: frontmostBundleID)
            .contains(where: { !$0.isTerminated }) else {
            VoxtLog.model(.init(.contextReadFailed, outcome: .failed, failure: .runtime))
            return nil
        }
        guard let provider = browserScriptProvider(for: frontmostBundleID) else { return nil }
        if let scriptedURL = runAppleScriptCandidates(provider.scripts) {
            browserAutomationDeniedUntilByBundleID.removeValue(forKey: frontmostBundleID)
            return scriptedURL
        }
        if let axURL = activeBrowserTabURLFromAccessibility(frontmostBundleID: frontmostBundleID) {
            VoxtLog.model(.init(.contextReadCompleted, outcome: .success))
            return axURL
        }
        return nil
    }

    func browserScriptProvider(for bundleID: String) -> BrowserScriptProvider? {
        switch bundleID {
        case "com.apple.Safari", "com.apple.SafariTechnologyPreview":
            return BrowserScriptProvider(
                name: "Safari",
                scripts: [
                    "tell application id \"\(bundleID)\" to get URL of front document",
                    "tell application id \"\(bundleID)\" to get URL of current tab of front window",
                    "tell application \"Safari\" to get URL of front document"
                ]
            )
        case "com.google.Chrome":
            return BrowserScriptProvider(
                name: "Google Chrome",
                scripts: [
                    "tell application id \"com.google.Chrome\" to get the URL of active tab of front window",
                    "tell application \"Google Chrome\" to get the URL of active tab of front window"
                ]
            )
        case "com.microsoft.edgemac":
            return BrowserScriptProvider(
                name: "Microsoft Edge",
                scripts: [
                    "tell application id \"com.microsoft.edgemac\" to get the URL of active tab of front window",
                    "tell application \"Microsoft Edge\" to get the URL of active tab of front window"
                ]
            )
        case "com.brave.Browser":
            return BrowserScriptProvider(
                name: "Brave Browser",
                scripts: [
                    "tell application id \"com.brave.Browser\" to get the URL of active tab of front window",
                    "tell application \"Brave Browser\" to get the URL of active tab of front window"
                ]
            )
        case "company.thebrowser.Browser":
            return BrowserScriptProvider(
                name: "Arc",
                scripts: [
                    "tell application id \"company.thebrowser.Browser\" to get the URL of active tab of front window",
                    "tell application id \"company.thebrowser.Browser\" to get the URL of active tab of window 1",
                    "tell application \"Arc\" to get the URL of active tab of front window"
                ]
            )
        default:
            guard let customDisplayName = customBrowserDisplayName(for: bundleID) else {
                return nil
            }
            return BrowserScriptProvider(
                name: customDisplayName,
                scripts: scriptsForCustomBrowser(bundleID: bundleID, displayName: customDisplayName)
            )
        }
    }

    func supportedBrowserBundleIDs() -> Set<String> {
        var bundleIDs: Set<String> = [
            "com.apple.Safari",
            "com.apple.SafariTechnologyPreview",
            "com.google.Chrome",
            "com.microsoft.edgemac",
            "com.brave.Browser",
            "company.thebrowser.Browser"
        ]
        for browser in loadStoredCustomBrowsers() where !browser.bundleID.isEmpty {
            bundleIDs.insert(browser.bundleID)
        }
        return bundleIDs
    }

    func customBrowserDisplayName(for bundleID: String) -> String? {
        loadStoredCustomBrowsers().first { $0.bundleID == bundleID }?.displayName
    }

    func loadStoredCustomBrowsers() -> [StoredCustomBrowser] {
        guard let json = UserDefaults.standard.string(forKey: AppPreferenceKey.appBranchCustomBrowsers),
              let data = json.data(using: .utf8),
              let decoded = try? JSONDecoder().decode([StoredCustomBrowser].self, from: data) else {
            return []
        }
        return decoded
    }

    func scriptsForCustomBrowser(bundleID: String, displayName: String) -> [String] {
        BrowserAutomationScriptBuilder.customBrowserRuntimeScripts(
            bundleID: bundleID,
            displayName: displayName
        )
    }

    func runAppleScriptCandidates(_ sources: [String]) -> String? {
        var lastError: NSDictionary?
        for (index, source) in sources.enumerated() {
            var executionError: NSDictionary?
            let startedNanoseconds = DispatchTime.now().uptimeNanoseconds
            if let output = runAppleScript(source, error: &executionError, logFailure: false, timeout: 0.8),
               !output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let elapsedMs = Int((DispatchTime.now().uptimeNanoseconds - startedNanoseconds) / 1_000_000)
                if index > 0 {
                    VoxtLog.model(.init(.contextReadCompleted, outcome: .success, durationMilliseconds: UInt64(max(0, elapsedMs))))
                }
                return output
            }
            if let executionError {
                let elapsedMs = Int((DispatchTime.now().uptimeNanoseconds - startedNanoseconds) / 1_000_000)
                VoxtLog.model(.init(.contextReadFailed, outcome: .failed, failure: .runtime, durationMilliseconds: UInt64(max(0, elapsedMs))))
                lastError = executionError
                if let errorNumber = executionError["NSAppleScriptErrorNumber"] as? Int {
                    if errorNumber == -1743 || errorNumber == -10004 {
                        if let frontmostBundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier {
                            browserAutomationDeniedUntilByBundleID[frontmostBundleID] = Date().addingTimeInterval(300)
                        }
                    }
                    if errorNumber == -600 {
                        break
                    }
                }
            } else {
                let elapsedMs = Int((DispatchTime.now().uptimeNanoseconds - startedNanoseconds) / 1_000_000)
                VoxtLog.model(.init(.contextReadCompleted, outcome: .empty, durationMilliseconds: UInt64(max(0, elapsedMs))))
            }
        }
        if lastError != nil {
            VoxtLog.model(.init(.contextReadFailed, outcome: .failed, failure: .runtime))
        }
        return nil
    }

    func runAppleScript(
        _ source: String,
        error: inout NSDictionary?,
        logFailure: Bool = true,
        timeout: TimeInterval? = nil
    ) -> String? {
        let wrappedSource: String
        if let timeout, timeout > 0 {
            let seconds = max(1, Int(ceil(timeout)))
            wrappedSource = """
            with timeout of \(seconds) seconds
            \(source)
            end timeout
            """
        } else {
            wrappedSource = source
        }

        guard let script = NSAppleScript(source: wrappedSource) else { return nil }
        guard let output = script.executeAndReturnError(&error).stringValue else {
            if logFailure, error != nil {
                VoxtLog.model(.init(.contextReadFailed, outcome: .failed, failure: .runtime))
            }
            return nil
        }
        return output
    }

    func activeBrowserTabURLFromAccessibility(frontmostBundleID: String) -> String? {
        guard AccessibilityPermissionManager.isTrusted() else {
            VoxtLog.model(.init(.contextReadFailed, outcome: .failed, failure: .permission))
            return nil
        }
        guard let app = NSWorkspace.shared.frontmostApplication,
              app.bundleIdentifier == frontmostBundleID
        else {
            VoxtLog.model(.init(.contextReadFailed, outcome: .failed, failure: .runtime))
            return nil
        }

        let appElement = AXUIElementCreateApplication(app.processIdentifier)
        var focusedWindowValue: CFTypeRef?
        let focusedStatus = AXUIElementCopyAttributeValue(
            appElement,
            kAXFocusedWindowAttribute as CFString,
            &focusedWindowValue
        )
        if focusedStatus == .success,
           let focusedWindow = focusedWindowValue,
           let url = axDocumentURL(from: focusedWindow) {
            return url
        } else if focusedStatus != .success {
            VoxtLog.model(.init(.contextReadFailed, outcome: .failed, failure: .runtime, errorCode: Int(focusedStatus.rawValue)))
        }

        var mainWindowValue: CFTypeRef?
        let mainStatus = AXUIElementCopyAttributeValue(
            appElement,
            kAXMainWindowAttribute as CFString,
            &mainWindowValue
        )
        if mainStatus == .success,
           let mainWindow = mainWindowValue {
            return axDocumentURL(from: mainWindow)
        }
        VoxtLog.model(.init(.contextReadFailed, outcome: .failed, failure: .runtime, errorCode: Int(mainStatus.rawValue)))
        return nil
    }

    func axDocumentURL(from windowRef: CFTypeRef) -> String? {
        guard CFGetTypeID(windowRef) == AXUIElementGetTypeID() else { return nil }
        let windowElement = unsafeBitCast(windowRef, to: AXUIElement.self)
        var documentValue: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(
            windowElement,
            kAXDocumentAttribute as CFString,
            &documentValue
        )
        guard status == .success, let documentValue else {
            VoxtLog.model(.init(.contextReadFailed, outcome: .failed, failure: .runtime, errorCode: Int(status.rawValue)))
            return nil
        }
        return documentValue as? String
    }
}

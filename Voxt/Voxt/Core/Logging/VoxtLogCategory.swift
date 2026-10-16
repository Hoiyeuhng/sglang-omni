// VoxtLogCategory.swift
// Defines logging categories used by Voxt diagnostics.

nonisolated enum VoxtLogCategory: String, Codable, CaseIterable, Sendable {
    case app
    case audio
    case dictionary
    case history
    case hotkey
    case input
    case asr
    case llm
    case model
    case meeting
    case network
    case settings
    case persistence
    case translation
    case update
    case security
}

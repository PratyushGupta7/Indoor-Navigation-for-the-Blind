"use client"

import { createContext, useContext, useEffect, useState, useMemo } from "react"

type VoiceSpeed = "slow" | "normal" | "fast" | "very-fast"

interface SettingsContextType {
  ttsVolume: number
  setTtsVolume: (v: number) => void
  hapticFeedbackIntensity: number
  setHapticFeedbackIntensity: (v: number) => void
  detectionDistance: number
  setDetectionDistance: (v: number) => void
  voiceSpeed: VoiceSpeed
  setVoiceSpeed: (v: VoiceSpeed) => void
  showOverlays: boolean
  setShowOverlays: (v: boolean) => void
}

const SettingsContext = createContext<SettingsContextType | undefined>(undefined)

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [ttsVolume, setTtsVolume] = useState(75)
  const [hapticFeedbackIntensity, setHapticFeedbackIntensity] = useState(80)
  const [detectionDistance, setDetectionDistance] = useState(3)
  const [voiceSpeed, setVoiceSpeed] = useState<VoiceSpeed>("normal")
  const [showOverlays, setShowOverlays] = useState(true)

  // load from localStorage once
  useEffect(() => {
    if (typeof window === "undefined") return
    try {
      const raw = localStorage.getItem("oculon-settings")
      if (!raw) return
      const parsed = JSON.parse(raw) as Partial<Record<keyof SettingsContextType, any>>
      parsed.ttsVolume != null && setTtsVolume(parsed.ttsVolume)
      parsed.hapticFeedbackIntensity != null && setHapticFeedbackIntensity(parsed.hapticFeedbackIntensity)
      parsed.detectionDistance != null && setDetectionDistance(parsed.detectionDistance)
      parsed.voiceSpeed && setVoiceSpeed(parsed.voiceSpeed)
      parsed.showOverlays != null && setShowOverlays(parsed.showOverlays)
    } catch (e) {
      console.warn("Could not load settings:", e)
    }
  }, [])

  // save whenever any value changes
  useEffect(() => {
    if (typeof window === "undefined") return
    try {
      localStorage.setItem(
        "oculon-settings",
        JSON.stringify({ ttsVolume, hapticFeedbackIntensity, detectionDistance, voiceSpeed, showOverlays })
      )
    } catch (e) {
      console.warn("Could not save settings:", e)
    }
  }, [ttsVolume, hapticFeedbackIntensity, detectionDistance, voiceSpeed, showOverlays])

  const value = useMemo(
    () => ({
      ttsVolume,
      setTtsVolume,
      hapticFeedbackIntensity,
      setHapticFeedbackIntensity,
      detectionDistance,
      setDetectionDistance,
      voiceSpeed,
      setVoiceSpeed,
      showOverlays,
      setShowOverlays,
    }),
    [ttsVolume, hapticFeedbackIntensity, detectionDistance, voiceSpeed, showOverlays]
  )

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>
}

export function useSettings() {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error("useSettings must be used within SettingsProvider")
  return ctx
}

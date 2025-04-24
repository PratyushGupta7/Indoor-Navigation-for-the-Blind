"use client"

import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useSettings } from "@/hooks/use-settings"
import { ArrowLeft, Volume2, Vibrate } from "lucide-react"

export default function SettingsPage() {
  const router = useRouter()
  const settings = useSettings()

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <div className="flex items-center mb-8">
        <Button variant="ghost" onClick={() => router.back()} className="mr-2 p-2">
          <ArrowLeft className="w-6 h-6" />
        </Button>
        <h1 className="text-2xl font-bold">Settings</h1>
      </div>

      <div className="space-y-8">
        {/* TTS Volume */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Volume2 className="w-6 h-6" />
            <span>TTS Volume</span>
          </div>
          <span className="text-right">{settings.ttsVolume}%</span>
        </div>
        <Slider
          value={[settings.ttsVolume]}
          min={0}
          max={100}
          step={1}
          onValueChange={(value) => settings.setTtsVolume(value[0])}
          className="bg-purple-900/20"
        />

        {/* Haptic Feedback Intensity */}
        <div className="flex items-center justify-between mt-8">
          <div className="flex items-center gap-3">
            <Vibrate className="w-6 h-6" />
            <span>Haptic Feedback Intensity</span>
          </div>
          <span className="text-right">{settings.hapticFeedbackIntensity}%</span>
        </div>
        <Slider
          value={[settings.hapticFeedbackIntensity]}
          min={0}
          max={100}
          step={1}
          onValueChange={(value) => settings.setHapticFeedbackIntensity(value[0])}
          className="bg-purple-900/20"
        />

        {/* Detection Distance */}
        <div className="flex items-center justify-between mt-8">
          <span>Detection Distance (meters)</span>
          <span className="text-right">{settings.detectionDistance}m</span>
        </div>
        <Slider
          value={[settings.detectionDistance]}
          min={1}
          max={10}
          step={1}
          onValueChange={(value) => settings.setDetectionDistance(value[0])}
          className="bg-purple-900/20"
        />

        {/* Voice Speed */}
        <div className="flex items-center justify-between mt-8">
          <span>Voice Speed</span>
          <div className="w-64">
            <Select value={settings.voiceSpeed} onValueChange={settings.setVoiceSpeed}>
              <SelectTrigger className="bg-gray-800 border-gray-700">
                <SelectValue placeholder="Select speed" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="slow">Slow</SelectItem>
                <SelectItem value="normal">Normal</SelectItem>
                <SelectItem value="fast">Fast</SelectItem>
                <SelectItem value="very-fast">Very Fast</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Show Visual Overlays */}
        <div className="flex items-center justify-between mt-8">
          <span>Show Visual Overlays</span>
          <Switch
            checked={settings.showOverlays}
            onCheckedChange={settings.setShowOverlays}
            className="bg-gray-700 data-[state=checked]:bg-purple-500"
          />
        </div>
      </div>
    </div>
  )
}

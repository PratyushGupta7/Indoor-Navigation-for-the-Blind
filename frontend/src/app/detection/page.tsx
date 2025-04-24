// src/app/detection/page.tsx
"use client"

import { useRouter } from "next/navigation"
import { useEffect, useRef, useState } from "react"
import { Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useSettings } from "@/hooks/use-settings"

interface DetectedObject {
  label: string
  confidence: number
  bbox: number[]
  center: number[]
  size: number[]
  closeness: number
}

interface PredictionResponse {
  objects: DetectedObject[]
  waypoints: number[][]
  instructions: string[]
}

export default function DetectionPage() {
  const router = useRouter()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [hasPermission, setHasPermission] = useState<boolean>(false)
  const settings = useSettings()
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([])

  // Persist last instruction/time across renders
  const lastInstructionRef = useRef<string>("")
  const lastInstructionTimeRef = useRef<number>(0)

  // Speech synthesis
  const speechSynthRef = useRef<SpeechSynthesis | null>(null)
  useEffect(() => {
    if (typeof window !== "undefined") {
      speechSynthRef.current = window.speechSynthesis
    }
  }, [])

    // Request camera access and start stream
  const requestCamera = async () => {
    try {
      const constraints = { video: { facingMode: "environment" } }
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = mediaStream
      setHasPermission(true)
    } catch (err) {
      console.error("Camera permission denied or error:", err)
      setHasPermission(false)
    }
  }

  // On permission granted, assign stream to video and play, assign stream to video and play
  useEffect(() => {
    if (hasPermission && streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current
      videoRef.current
        .play()
        .catch((err) => console.error("Error playing video after permission:", err))
    }
  }, [hasPermission])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  // Draw overlays
  useEffect(() => {
    if (!hasPermission) return
    const canvas = canvasRef.current!
    const video = videoRef.current!
    const ctx = canvas.getContext("2d")!
    const colors: Record<string, string> = {
      person: "#FF0000", car: "#00FF00", bicycle: "#0000FF",
      dog: "#FFFF00", chair: "#FF00FF", default: "#00FFFF",
    }
    let rafId: number
    const drawLoop = () => {
      if (video.videoWidth && video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        if (settings.showOverlays) {
          detectedObjects.forEach((obj) => {
            const [x, y, x2, y2] = obj.bbox
            const w = x2 - x, h = y2 - y
            const color = colors[obj.label.toLowerCase()] || colors.default
            ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.strokeRect(x, y, w, h)
            const label = `${obj.label} ${Math.round(obj.confidence * 100)}%`
            const textW = ctx.measureText(label).width
            ctx.fillStyle = color; ctx.globalAlpha = 0.7
            ctx.fillRect(x, y > 20 ? y - 25 : y + h, textW + 10, 20)
            ctx.globalAlpha = 1.0; ctx.fillStyle = "#000"; ctx.font = "16px Arial"
            ctx.fillText(label, x + 5, y > 20 ? y - 10 : y + h + 15)
          })
        }
      }
      rafId = requestAnimationFrame(drawLoop)
    }
    drawLoop()
    return () => cancelAnimationFrame(rafId)
  }, [hasPermission, settings.showOverlays, detectedObjects])

  // Capture & send frames every 0.5s
  useEffect(() => {
    if (!hasPermission) return
    const interval = setInterval(captureAndSendFrame, 500)
    captureAndSendFrame()
    return () => clearInterval(interval)
  }, [hasPermission])

  const captureAndSendFrame = async () => {
    const video = videoRef.current!
    if (video.readyState !== 4) return
    const temp = document.createElement("canvas")
    temp.width = video.videoWidth; temp.height = video.videoHeight
    const tmpCtx = temp.getContext("2d")!
    tmpCtx.drawImage(video, 0, 0)
    try {
      const blob = await new Promise<Blob>((res) => temp.toBlob((b) => res(b!), "image/jpeg", 0.8))
      const form = new FormData()
      form.append("file", blob, "frame.jpg")
      const res = await fetch("/api/predict", { method: "POST", body: form })
      if (!res.ok) { console.error("API", res.status, await res.text()); return }
      const data: PredictionResponse = await res.json()
      setDetectedObjects(data.objects)
      handleInstruction(data.instructions[0])
    } catch (e) {
      console.error("Error sending frame", e)
    }
  }

  const handleInstruction = (inst?: string) => {
    if (!inst || inst === "No path computed") return
    const now = Date.now()
    if (inst !== lastInstructionRef.current || now - lastInstructionTimeRef.current > 5000) {
      const synth = speechSynthRef.current!
      synth.cancel()
      const utt = new SpeechSynthesisUtterance(inst)
      utt.volume = settings.ttsVolume / 100
      switch (settings.voiceSpeed) {
        case "slow": utt.rate = 0.8; break
        case "normal": utt.rate = 1.0; break
        case "fast": utt.rate = 1.2; break
        case "very-fast": utt.rate = 1.5; break
      }
      synth.speak(utt)
      lastInstructionRef.current = inst
      lastInstructionTimeRef.current = now
      if (settings.hapticFeedbackIntensity > 0 && navigator.vibrate) {
        navigator.vibrate(Math.floor((settings.hapticFeedbackIntensity / 100) * 200))
      }
    }
  }

  const stopDetection = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    router.push("/")
  }

  // Render
  if (!hasPermission) {
    return (
      <div className="flex items-center justify-center h-screen bg-black">
        <Button onClick={requestCamera} className="px-6 py-3 bg-purple-600 text-white">
          Enable Camera & Start
        </Button>
      </div>
    )
  }

  return (
    <div className="relative h-screen w-full bg-black overflow-hidden">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover z-10"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full z-20 pointer-events-none"
      />
      <div className="absolute bottom-8 left-0 right-0 flex justify-center z-30">
        <Button
          onClick={stopDetection}
          className="rounded-full px-8 py-6 bg-red-600 hover:bg-red-700 text-white"
        >
          Stop Detection
        </Button>
      </div>
      <Button
        onClick={() => router.push("/settings")}
        variant="ghost"
        className="absolute bottom-8 right-8 text-white z-30"
      >
        <Settings className="w-6 h-6" />
      </Button>
    </div>
  )
}

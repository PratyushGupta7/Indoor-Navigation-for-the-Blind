"use client"

import { useRouter } from "next/navigation"
import { Camera } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useEffect, useState } from "react"

export default function Home() {
  const router = useRouter()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-6 bg-black text-white">
      <div className="w-full h-full flex flex-col items-center justify-center gap-4">
        <div className="flex-1 flex flex-col items-center justify-center gap-2">
          <h1 className="text-6xl font-bold text-center">Oculon</h1>
          <p className="text-xl text-center">Visual Assistance App</p>
        </div>

        <div className="flex-1 flex flex-col items-center justify-center gap-6">
          <Button
            onClick={() => router.push("/detection")}
            className="w-32 h-32 rounded-3xl bg-purple-400 hover:bg-purple-500"
          >
            <Camera className="w-12 h-12 text-black" />
          </Button>
          <p className="text-center text-gray-300">Tap the button above to start obstacle detection</p>
        </div>

        <div className="mt-auto">
          <Button
            onClick={() => router.push("/settings")}
            variant="outline"
            className="rounded-full px-6 py-2 bg-purple-400 hover:bg-purple-500 border-none text-black"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="mr-2"
            >
              <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path>
              <circle cx="12" cy="12" r="3"></circle>
            </svg>
            Settings
          </Button>
        </div>
      </div>
    </main>
  )
}

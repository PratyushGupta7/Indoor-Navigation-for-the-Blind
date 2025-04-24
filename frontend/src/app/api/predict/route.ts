import { NextResponse } from 'next/server'

// Proxy API route to forward multipart/form-data to external prediction endpoint
export async function POST(request: Request) {
  try {
    // Parse incoming form-data (expects field name "file")
    const formData = await request.formData()

    // Forward to external prediction endpoint
    const externalRes = await fetch(
      'http://192.168.45.162:8000/predict',
      {
        method: 'POST',
        body: formData,
      }
    )

    // Read response text
    const contentType = externalRes.headers.get('Content-Type') || 'application/json'
    const body = await externalRes.text()

    return new NextResponse(body, {
      status: externalRes.status,
      headers: { 'Content-Type': contentType },
    })
  } catch (err) {
    console.error('Error in /api/predict proxy:', err)
    return NextResponse.json({ error: 'Proxy error' }, { status: 500 })
  }
}

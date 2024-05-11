// Placeholder function for next step. Paste over this in the next step.
export async function enableCam() {
  const videoElement = document.getElementById("webcam");
  const constraints = {
    video: true,
    frameRate: { ideal: 60 },
  };

  // Activate the webcam stream.
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  videoElement.srcObject = stream;
}

// Check if webcam access is supported.
export function getUserMediaSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

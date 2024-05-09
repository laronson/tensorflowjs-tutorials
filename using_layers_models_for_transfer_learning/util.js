const VIDEO = document.getElementById("webcam");
const ENABLE_CAM_BUTTON = document.getElementById("enableCam");

export function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

export async function enableCam() {
  if (hasGetUserMedia()) {
    //getUserMedia parameters
    const userMediaParameters = {
      video: true,
      width: 640,
      height: 480,
    };

    //activate webcam
    const stream = await navigator.mediaDevices.getUserMedia(
      userMediaParameters
    );
    VIDEO.srcObject = stream;
    VIDEO.addEventListener("loadeddata", function () {
      ENABLE_CAM_BUTTON.classList.add("removed");
    });
  } else {
    console.warn("getUserMedia() is not supported by your browser");
  }
}

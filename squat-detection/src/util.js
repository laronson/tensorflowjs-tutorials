export function addImageToScreen() {
  const canvasElement = document.getElementById("imageCanvas");
  const imageElement = document.getElementById("sampleImage");

  const canvasCtx = canvasElement.getContext("2d");
  canvasCtx.drawImage(imageElement, 0, 0);
}

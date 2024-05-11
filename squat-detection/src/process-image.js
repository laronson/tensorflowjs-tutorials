export async function processImage(detector, canvasElement) {
  const poses = await detector.estimatePoses(canvasElement, {
    flipHorizontal: false,
  });

  return poses;
}

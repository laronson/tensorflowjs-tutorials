import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
// Register WebGL backend.
import "@tensorflow/tfjs-backend-webgl";
let detector;

async function createDetector() {
  const model = poseDetection.SupportedModels.BlazePose;
  const detectorConfig = {
    runtime: "tfjs",
    enableSmoothing: true, //set to false if you are using this with a static image as smoothing is not needed
    modelType: "lite", //lite: small & most optimized, heavy: largest and most accurate, full: combination of both (default full)
  };
  detector = await poseDetection.createDetector(model, detectorConfig);
}

async function processImage() {
  const canvasElement = document.getElementById("imageCanvas");
  const canvasCtx = canvasElement.getContext("2d");

  const poses = await detector.estimatePoses(canvasElement, {
    flipHorizontal: false,
  });
  console.log({ poses });

  poses[0].keypoints.forEach((pose) => {
    canvasCtx.beginPath();
    canvasCtx.fillStyle = "blue";
    canvasCtx.fillRect(pose.x, pose.y, 5, 5);
    canvasCtx.stroke();
  });
}

function addImageToScreen() {
  const canvasElement = document.getElementById("imageCanvas");
  const imageElement = document.getElementById("sampleImage");

  const canvasCtx = canvasElement.getContext("2d");
  canvasCtx.drawImage(imageElement, 0, 0);
}

async function app() {
  console.log("WAITING FOR TF TO BE READY");
  await tf.ready();
  console.log("TF IS READY");
  await createDetector();

  addImageToScreen();

  const processMeButton = document.getElementById("processButton");
  processMeButton.addEventListener("click", processImage);
}

app();

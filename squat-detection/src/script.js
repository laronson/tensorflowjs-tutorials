import { enableCam, getUserMediaSupported } from "./enable-webcam.js";
import { normalizeSquatData } from "./normalize-squat-data.js";
import { preparePoseModel, prepareSquatCountModel } from "./prepare-model.js";
import { processImage } from "./process-image.js";
import * as tf from "@tensorflow/tfjs";

let isRecording = false;
let isRecordingTrainingData = false;
let isRecordingSquat = false;
let shouldUseSquatCountModel = false;
let squatCount = 0;
let alreadyCounted = false;

const videoElement = document.getElementById("webcam");
const videoCanvas = document.getElementById("videoCanvas");
const videoContext = videoCanvas.getContext("2d");
const squatLabel = document.getElementById("sqautLabel");
const squatConfidenceLabel = document.getElementById("sqautConfidence");
const squatCountLabel = document.getElementById("squatCount");
const squatCountTrainingInputs = [];
const squatCountTrainingOutputs = [];

async function renderContent(detector, squatCountModel) {
  if (isRecording) {
    const poses = await processImage(detector, videoCanvas);
    videoContext.drawImage(videoElement, 0, 0);

    if (poses && poses.length === 1) {
      let isDatasetValid = true;
      const keypoints = poses[0].keypoints;
      for (let i = 0; i < keypoints.length; i++) {
        videoContext.fillStyle = "green";
        if (keypoints[i].score < 0.5) {
          isDatasetValid = false;
          videoContext.fillStyle = "red";
        }

        const circle = new Path2D();
        circle.arc(keypoints[i].x, keypoints[i].y, 5, 0, 2 * Math.PI);
        videoContext.fill(circle);
        videoContext.stroke(circle);
      }

      if (isDatasetValid) {
        saveData(keypoints);
      }

      if (shouldUseSquatCountModel && isDatasetValid) {
        const result = await predictSquat(squatCountModel, keypoints);
        squatConfidenceLabel.innerText = result[0];
        if (result[0] > 0.9) {
          if (!alreadyCounted) {
            squatCount++;
            alreadyCounted = true;
          }
          squatLabel.innerText = "SQUATTING";
        } else {
          alreadyCounted = false;
          squatLabel.innerText = "NOT SQUATTING";
        }
        squatCountLabel.innerText = `SQUAT COUNT: ${squatCount}`;
      }
    }
  }
}

async function renderLoop(detector, squatCountModel) {
  await renderContent(detector, squatCountModel);
  requestAnimationFrame(function () {
    renderLoop(detector, squatCountModel);
  });
}

function saveData(keyPoints) {
  if (isRecordingTrainingData) {
    const isSquatting = isRecordingSquat ? 1 : 0;
    squatCountTrainingInputs.push(keyPoints.map(({ x, y }) => ({ x, y })));
    squatCountTrainingOutputs.push(isSquatting);
  }
}

async function trainSquatCountModel(squatCountModel) {
  console.log(squatCountTrainingInputs);
  console.log(squatCountTrainingOutputs);
  const normalizedInputs = normalizeSquatData(squatCountTrainingInputs);
  console.log(normalizedInputs);
  const inputTensor = tf.tensor2d(normalizedInputs);
  const outputTensor = tf.tensor1d(squatCountTrainingOutputs);

  await squatCountModel.fit(inputTensor, outputTensor, {
    validationSplit: 0.2,
    shuffle: true,
    batchSize: 32,
    epochs: 50,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`epoch: ${epoch} loss: ${logs.loss}`);
      },
    },
  });

  inputTensor.dispose();
  outputTensor.dispose();
}

function predictSquat(squatCountModel, data) {
  return tf.tidy(() => {
    const normalizedData = normalizeSquatData([data]);
    console.log(normalizedData);
    const prediction = squatCountModel.predict(tf.tensor2d(normalizedData));
    return prediction.dataSync();
  });
}

async function app() {
  const { model, detector } = await preparePoseModel();
  const { squatCountModel } = prepareSquatCountModel();

  const enableWebCamButton = document.getElementById("enableWebCamButton");
  if (getUserMediaSupported() && model) {
    enableWebCamButton.addEventListener("click", async () => {
      enableCam();
      enableWebCamButton.setAttribute("disabled", true);
      isRecording = true;

      renderLoop(detector, squatCountModel);
    });
  }

  const saveTrainingButton = document.getElementById(
    "recordTrainingDataButton"
  );
  saveTrainingButton.addEventListener("click", () => {
    if (!isRecordingTrainingData) {
      isRecordingTrainingData = true;
      saveTrainingButton.innerText = "Stop Recording Training Data";
    } else {
      isRecordingTrainingData = false;
      saveTrainingButton.innerText = "Record Training Data";
    }
  });

  const squatDataButton = document.getElementById("recordSquattingDataButton");
  squatDataButton.addEventListener("click", () => {
    if (!isRecordingSquat) {
      isRecordingSquat = true;
      squatDataButton.innerText = "Stop Recording Squatting Data";
    } else {
      isRecordingSquat = false;
      squatDataButton.innerText = "Record Squatting Data";
    }
  });

  const trainModelButton = document.getElementById("trainModelButton");
  trainModelButton.addEventListener("click", async () => {
    await trainSquatCountModel(squatCountModel);
    console.log("done training");
  });

  const useSquatCountModelButton = document.getElementById(
    "useSquadModelButton"
  );
  useSquatCountModelButton.addEventListener("click", () => {
    shouldUseSquatCountModel = true;
  });
}

app();

import { loadMobileNetFeatureModel } from "./loadMobileNetFeatureModel.js";
import { defineClassificationModelHead } from "./defineClassificationModelHead.js";
import { enableCam } from "./util.js";
import { gatherData } from "./dataGather.js";
import {
  MOBILE_NET_INPUT_HEIGHT,
  MOBILE_NET_INPUT_WIDTH,
} from "./constants.js";
import { train } from "./train.js";
import { predict } from "./predict.js";

let videoPlaying = false;
let gatheringAnimationFrameRequestId = null;
let predictionAnimationFrameRequestId = null;
const trainingDataInputs = [];
const trainingDataOutputs = [];
const exampleCount = [];

/**
 * Setup UI functionality
 */
const ENABLE_CAM_BUTTON = document.getElementById("enableCam");
const RESET_BUTTON = document.getElementById("reset");
const TRAIN_BUTTON = document.getElementById("train");
const STATUS = document.getElementById("status");

ENABLE_CAM_BUTTON.addEventListener("click", () => {
  enableCam();
  videoPlaying = true;
});
let dataCollectorButtons = document.querySelectorAll("button.dataCollector");
const CLASS_NAMES = [];
dataCollectorButtons.forEach((button) => {
  CLASS_NAMES.push(button.getAttribute("data-name"));
});

TRAIN_BUTTON.addEventListener("click", async () => {
  console.log("training");
  window.cancelAnimationFrame(predictionAnimationFrameRequestId);
  await train(
    trainingDataInputs,
    trainingDataOutputs,
    CLASS_NAMES,
    classificationModel
  );
  predictLoop();
});
RESET_BUTTON.addEventListener("click", reset);

/**********************************************************************************************************************
 * Model setup
 **********************************************************************************************************************/
const mobileNet = await loadMobileNetFeatureModel(
  MOBILE_NET_INPUT_HEIGHT,
  MOBILE_NET_INPUT_WIDTH
);
const classificationModel = defineClassificationModelHead(CLASS_NAMES.length);

/**********************************************************************************************************************
 * Data Gathering buttons and functions
 **********************************************************************************************************************/

for (let i = 0; i < dataCollectorButtons.length; i++) {
  //no arrow function here because an arrow function binds this to the script and not the button
  dataCollectorButtons[i].addEventListener("mousedown", function () {
    const gatheringClassNumber = parseInt(this.getAttribute("data-1hot"));
    if (videoPlaying) {
      gatherDataLoop(gatheringClassNumber, mobileNet);
    }
  });
  dataCollectorButtons[i].addEventListener("mouseup", function () {
    window.cancelAnimationFrame(gatheringAnimationFrameRequestId); //bug: keeps recording if scroll off element before mouse up
  });
}

function gatherDataLoop(classNumber, mobileNet) {
  const { imageFeatures } = gatherData(
    mobileNet,
    MOBILE_NET_INPUT_HEIGHT,
    MOBILE_NET_INPUT_WIDTH
  );

  //Push the new training data gathered and predicted in the gatherData() function and add it to the training data
  //input and output datasets
  trainingDataInputs.push(imageFeatures);
  trainingDataOutputs.push(classNumber);

  //increase the current count of examples for the class we are gathering for.  Set to 1 if not previously defined
  if (exampleCount[classNumber] === undefined) {
    exampleCount[classNumber] = 0;
  }
  exampleCount[classNumber]++;

  STATUS.innerText = "";
  for (let n = 0; n < CLASS_NAMES.length; n++) {
    STATUS.innerText += CLASS_NAMES[n] + "data count" + exampleCount[n];
  }

  //Recursively call this function when a new animation is rendered on the screen.  This recursive loop will end once
  //The user takes their mouse off the gather button and the recursive calls cease.
  gatheringAnimationFrameRequestId = window.requestAnimationFrame(function () {
    gatherDataLoop(classNumber, mobileNet);
  });
}

/**********************************************************************************************************************
 * Predict Loop function
 **********************************************************************************************************************/
function predictLoop() {
  const { highestIndex, confidence } = predict(
    MOBILE_NET_INPUT_HEIGHT,
    MOBILE_NET_INPUT_WIDTH,
    mobileNet,
    classificationModel
  );

  STATUS.innerText = `Prediction ${CLASS_NAMES[highestIndex]} with ${Math.floor(
    confidence * 100
  )}% confidence`;

  predictionAnimationFrameRequestId = window.requestAnimationFrame(predictLoop);
}

/**********************************************************************************************************************
 * Reset function
 **********************************************************************************************************************/
function reset() {
  console.log("resetting");
  window.cancelAnimationFrame(predictionAnimationFrameRequestId);
  exampleCount.splice(0);
  trainingDataInputs.forEach((trainingDataTensor) =>
    trainingDataTensor.dispose()
  );

  trainingDataInputs.splice(0);
  trainingDataOutputs.splice(0);
  console.log(`tensors in memory: ${tf.memory().numTensors}`);
}

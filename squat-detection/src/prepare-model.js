import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
// Register WebGL backend.
import "@tensorflow/tfjs-backend-webgl";

export async function preparePoseModel() {
  await tf.ready();
  const model = poseDetection.SupportedModels.BlazePose;

  const detectorConfig = {
    runtime: "tfjs",
    enableSmoothing: true, //set to false if you are using this with a static image as smoothing is not needed
    modelType: "lite", //lite: small & most optimized, heavy: largest and most accurate, full: combination of both (default full)
  };
  const detector = await poseDetection.createDetector(model, detectorConfig);

  return { model, detector };
}

export function prepareSquatCountModel() {
  const squatCountModel = tf.sequential();

  squatCountModel.add(
    tf.layers.dense({ inputShape: [66], units: 256, activation: "relu" })
  );

  squatCountModel.add(tf.layers.dense({ units: 256, activation: "relu" }));

  squatCountModel.add(tf.layers.dense({ units: 1 }));

  squatCountModel.compile({
    optimizer: tf.train.sgd(0.01),
    loss: "meanSquaredError",
    metrics: ["accuracy"],
  });

  return { squatCountModel };
}

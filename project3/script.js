const MODEL_PATH =
  "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4";
const EXAMPLE_IMG = document.getElementById("exampleImage");

let movementModel = undefined;
let cocoSsdModel = undefined;

// async function loadAndRunModel() {
//   movement = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });

//   //create a tensor from an image and then print the shape (height and width) of the image
//   //NOTE: Tensors always print height then width rather than width then height
//   let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
//   console.log(imageTensor.shape);

//   let exampleInputTensor = tf.zeros([1, 192, 192, 3], "int32");

//   let tensorOutput = movement.predict(exampleInputTensor);
//   let arrayOutput = await tensorOutput.array();
//   console.log(arrayOutput);
// }

async function loadCocoSsdModel() {
  cocoSsdModel = await cocoSsd.load();
  console.log("Loaded coco model");
}

async function loadMovementModel() {
  movementModel = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });
}

function getImageTensor() {
  let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
  console.log(imageTensor.shape);

  return {
    tensor: imageTensor,
    height: imageTensor.shape[0],
    width: imageTensor.shape[1],
  };
}

async function detectHuman(img) {
  const cocoPredictions = await cocoSsdModel.detect(img);
  console.log(cocoPredictions);
  const humanPrediction = cocoPredictions.find(
    (prediction) => prediction.class === "person"
  );
  return humanPrediction;
}

function resizeHumanImage(imageTensor, humanPrediction) {
  const humanBbox = humanPrediction.bbox;
  const imgWidth = Math.ceil(humanBbox[2]);
  const imgHeight = Math.ceil(humanBbox[3]);
  const xExpansionToSquare = Math.abs((imgWidth - imgHeight) / 2);

  const imgStartX = Math.ceil(humanBbox[0]) - xExpansionToSquare;
  console.log({
    humanBbox,
    imgWidth,
    imgHeight,
    imgStartX,
    xExpansionToSquare,
    imgShape: imageTensor.shape,
  });

  const cropStartPoint = [Math.ceil(humanBbox[1]), imgStartX, 0];
  const desiredCropSize = [imgHeight, imgHeight, 3];

  const croppedTensor = tf.slice(imageTensor, cropStartPoint, desiredCropSize);
  console.log(croppedTensor.shape);

  const resizedTensor = tf.image
    .resizeBilinear(croppedTensor, [192, 192], true)
    .toInt();
  console.log(resizedTensor.shape);
  return resizedTensor;
}

async function predictMovement(imageTensor) {
  const tensorOutput = await movementModel.predict(tf.expandDims(imageTensor));
  console.log(await tensorOutput.array());
}

await loadCocoSsdModel();
await loadMovementModel();

const image = getImageTensor();
const humanPrediction = await detectHuman(image.tensor);
const resizedTensorImage = resizeHumanImage(image.tensor, humanPrediction);
predictMovement(resizedTensorImage);

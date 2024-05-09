const MODEL_JSON_URL =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/spam/model.json";

export async function loadModel() {
  const model = await tf.loadLayersModel("./enhanced-model/model.json");
  return model;
}

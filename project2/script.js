const MODEL_PATH =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json";
let model = undefined;

async function saveModelLocally() {
  //you can save your model in local storage so the load time for this site is faster next time the page is loaded
  //this could also be used to grant offline capabilities
  await model.save("localstorage://demo/housingModal");

  //check if modal already exists in localstorage.  If it does, use the local storage url instead of the web url
  console.log(JSON.stringify(await tf.io.listModels()));
}

async function loadModel() {
  model = await tf.loadLayersModel(MODEL_PATH);
  model.summary();
  saveModelLocally();

  //batch of 1
  const input = tf.tensor2d([[870]]);

  //batch of 3
  const inputBatch = tf.tensor2d([[500], [1100], [970]]);

  const result = model.predict(input);
  const resultBatch = model.predict(inputBatch);

  //print results using tensor.print().  We could also use tensor.array() to convert back to a js array and print that
  //way
  result.print();
  resultBatch.print();

  //Because there is no automatic garbage collection of tensors in tensorflow we must explicitly dispose of our
  //tensors manually
  input.dispose();
  inputBatch.dispose();
  result.dispose();
  resultBatch.dispose();
}

loadModel();

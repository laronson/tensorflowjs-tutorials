export async function predict(model, inputTensor) {
  if (model === undefined) {
    console.warn("Model has not been initialized");
    return;
  }

  var results = await model.predict(inputTensor);
  results.print();

  /**
   * return the results of the prediction from this function by using the dataSync function.  This returns the data
   * that is stored in the results tensor under the data attribute.  In this case, because we are working with an NLP
   * (it may be the case that this works for other types of models too), the data array returned from dataSync is used
   * to show the probability of if the input data is true for the given inputs (like a softmax 1hot encoded array).
   * In this case, the values in the labels file will result in an output in the format of [false(forSpam),true(forSpam)]
   * Therefore, we can check against either one of these values to see the percent likelyhood of the comment being spam
   */
  return results.dataSync();
}

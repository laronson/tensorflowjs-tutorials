const PREDICTION_ELEMENT = document.getElementById("prediction");

export async function evaluate(model, INPUTS, OUTPUTS) {
  //Generate a index value used to select a random input value from the INPUTS array
  const OFFSET = Math.floor(Math.random() * INPUTS.length);

  let answer = tf.tidy(function () {
    //Create a 1D tensor from the selected input.  Remember, a greyscale image is just a 1D array of int values
    //from 0-255
    let newInput = tf.tensor1d(INPUTS[OFFSET]);

    //Call model.predict on the selected value.  HOWEVER, we have to remember that the model expects a batch of values
    //(more than one image to predict for).  Therefore, we need to use the expandDims() function to convert the input
    //we have selected into a 2d tensor.  Therefore, we are turning the input of type number[] into number[][].
    let output = model.predict(newInput.expandDims());
    output.print();

    //Return the index of the highest value in the output softMax array using the argMax() function.  Because the output
    //of the model was a batch result as a 2d tensor, we must call the squeeze function to remove the outermost array
    //in the tensor.
    //NOTE* Instead of using the squeeze function to get the innermost array, you could set the depth of where you want
    //to grab the output from as a parameter to the argMax function.
    return output.squeeze().argMax();
  });

  //get the index returned from the prediction using the array function.  Because the index correlates to the value we
  //are using as our answer (aka 1 for an image that contains a one and 2 for an image that contains a 2) we can simply
  //print out the index as our predicted output.  If we were trying to classify images of animals we would need a lookup
  //to indicate which index correlates to which animal we are trying to classify
  const answerIndex = await answer.array();

  PREDICTION_ELEMENT.innerText = answerIndex;
  PREDICTION_ELEMENT.setAttribute(
    "class",
    answerIndex === OUTPUTS[OFFSET] ? "correct" : "wrong"
  );

  answer.dispose();
  return { OFFSET };
}

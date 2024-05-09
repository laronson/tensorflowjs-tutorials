const CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext("2d", { willReadFrequently: true });

export function drawImage(digitImage) {
  // get the image data stored in the canavas on the page specifying what part of the image you want to get with the
  //parameters getImageData(xIdx:number, yIdx:number, width:number, height:number)
  var imageData = CTX.getImageData(0, 0, 28, 28);

  //Iterate through every pixel of the digitalImage and set the corresponding pixel on the page's canvas element based off
  //of the value of the pixel in the digitImage pixel array.  Remember, the digitImage values were normalized for the model
  //so you need to convert them back to normal pixel values by multiplying them by 255
  for (let i = 0; i < digitImage.length; i++) {
    imageData.data[i * 4] = digitImage[i] * 255; //Red Channel
    imageData.data[i * 4 + 1] = digitImage[i] * 255; //Green Channel
    imageData.data[i * 4 + 2] = digitImage[i] * 255; //Blue Channel
    imageData.data[i * 4 + 3] = 255; // Alpha transparency Channel - set to 255 to not make any pixel transparent
  }

  // Render the updated array of data to the canvas
  CTX.putImageData(imageData, 0, 0);
}

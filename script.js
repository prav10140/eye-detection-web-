let model;
const webcam = document.getElementById("webcam");
const predictionText = document.getElementById("prediction");

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        webcam.srcObject = stream;
        webcam.addEventListener("loadeddata", resolve);
      })
      .catch(reject);
  });
}

async function predict() {
  const videoTensor = tf.browser.fromPixels(webcam, 1)
    .resizeNearestNeighbor([24, 24])
    .expandDims(0)
    .div(255.0);
  
  const prediction = model.predict(videoTensor);
  const result = (await prediction.argMax(1).data())[0];
  
  predictionText.textContent = result === 1 ? "👁 Open Eye" : "😴 Closed Eye";
  
  requestAnimationFrame(predict);
}

async function main() {
  model = await tf.loadLayersModel("tfjs_model/model.json");
  await setupWebcam();
  predict();
}

main();

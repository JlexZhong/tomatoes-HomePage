
let model; 
let trainingDataSets = [0, 0];
 
const loadModel = async () => {
    console.log("Model is Loading...");
    const model = await tf.loadLayersModel("model_js_incep/model.json");
    console.log(model.summary());
    console.log("Model Loaded successfully!");
    return model;
};

// 调整大小并归一化处理
function preprocess(imgData)
{
return tf.tidy(()=>{
    //convert the image data to a tensor 
    let tensor = tf.browser.fromPixels(imgData, numChannels= 3);
    //resize to 299 x 299 resized = [150,150,3]
    const resized = tf.image.resizeBilinear(tensor, [150, 150]).toFloat();
    // Normalize the image 
    const offset = tf.scalar(255.0);
    const normalized = tf.scalar(1.0).sub(resized.div(offset));
    //We add a dimension to get a batch shape 
    const batched = normalized.expandDims(0);
    console.log(batched.shape)
    // batched = [1,150,150,3]
    return batched;
})
}

async function _Predict() {
    //extract the image data 
    console.log("获取图像");
    const imgData = context.getImageData(0,0,299,299);
    // 预测,这将返回一个规模为「N, 100」的概率。
    console.log("加载模型.....");
    console.log("加载成功，正在预测");
    const batched = preprocess(imgData);
    const pred = model.predict(batched).dataSync();
    console.log(pred)
};

async function Predict() {
    const predictedClass = tf.tidy(() => {
        console.log("获取图像");
        const imgData = context.getImageData(0,0,299,299);
        const batched = preprocess(imgData);
        const predictions = model.predict(batched);
        console.log(predictions.shape)
        const label = tf.argMax(predictions, axis=1).dataSync()[0]
        console.log(label)
        return predictions.as1D().argMax();
      });
    const classId = (await predictedClass.data())[0];
    console.log(classId)
    document.getElementById("prediction").innerText = classId ;   
    predictedClass.dispose();
    
}

function startPredict(){
    Predict()
}



async function init(){
    model = await loadModel();

}

init()
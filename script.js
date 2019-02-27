async function classify(sentence){

	const session = new InferenceSession();
	const url = './models/IMDB-Bidir.onnx';
	await session.loadModel(url);

	console.log(sentence)
}


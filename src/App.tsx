import React, { Component } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as tf from '@tensorflow/tfjs';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
/* import SquirtleData from './data/squirtle/squirtle';
import CharmanderData from './data/charmander/charmander';
import BulbasaurData from './data/bulbasaur/bulbasaur'; */
import model from './model.json';
import './App.css';

/**
 * A simple React App to for classifying Pokemons
 *
 * @author iberatkaya
 */

interface Props {

};

/**
 * State of the React App
 * 
 * @property {string} image The file path of the image
 * @property {boolean} loading To check if the models have loaded
 * @property {mobilenet.MobileNet} modelMobilenet The MobileNet model 
 * @property {knnClassifier.KNNClassifier} classifier The KNN Classifier
 * @property {Array<object>} mobilenetPred The predictions of the MobileNet model,
 * @property {object} knnPred The predictions of the KNN model,
 * @property {boolean} scanned To check if the image was scanned
 */

interface State {
  image: string,
  loading: boolean,
  modelMobilenet: mobilenet.MobileNet | null,
  mobilenetPred: Array<object>,
  knnPred: {
    label: string;
    classIndex: number;
    confidences: {
      [label: string]: number;
    };
  } | null,
  scanned: boolean,
  classifier: knnClassifier.KNNClassifier | null
}

/**
 * Types used for downloading and importing the dataset
 */

type Dataset = {
  [classId: number]: tf.Tensor<tf.Rank.R2>
};

type DatasetObjectEntry = {
  classId: number,
  data: number[],
  shape: [number, number]
};

type DatasetObject = DatasetObjectEntry[];

class App extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      image: '',
      loading: true,
      knnPred: null,
      modelMobilenet: null,
      mobilenetPred: [],
      scanned: false,
      classifier: null
    }
    // this.trainData(BulbasaurData, CharmanderData, SquirtleData);
  }

  /**
   * Function inserting the images to the DOM
   * 
   * @param {object} BulbasaurData The image paths of the images
   * @param {object} CharmanderData The image paths of the images
   * @param {object} SquirtleData The image paths of the images
   */

  trainData = (BulbasaurData: any, CharmanderData: any, SquirtleData: any) => {
    let arr = [Object.values(BulbasaurData), Object.values(CharmanderData), Object.values(SquirtleData)].flat()
    for (let i = 0; i < arr.length; i++) {
      const im = new Image()
      im.src = arr[i] as string;
      im.width = 300;
      im.height = 300;
      im.className = "hidden";
      document.body.appendChild(im);
      im.id = "img" + (i + 1).toString()
    }
  }


  /**
   * Function to train the model using the dataset
   * 
   * @param {object} BulbasaurData The image paths of the images
   * @param {object} CharmanderData The image paths of the images
   * @param {object} SquirtleData The image paths of the images
   */


  trainModel = async (BulbasaurData: any, CharmanderData: any, SquirtleData: any) => {
    const modelMobilenet = await mobilenet.load({ version: 2, alpha: 1 });
    const classifier = knnClassifier.create();
    let arr = [Object.values(BulbasaurData), Object.values(CharmanderData), Object.values(SquirtleData)].flat()
    for (let i = 0; i < arr.length; i++) {
      let image = tf.browser.fromPixels(document.getElementById("img" + (i + 1).toString()) as HTMLImageElement);
      const inferred = modelMobilenet.infer(image);
      if (i <= 100)
        classifier.addExample(inferred, 0);
      else if (i > 100 && i <= 200)
        classifier.addExample(inferred, 1);
      else
        classifier.addExample(inferred, 2);
    }
    this.saveClassifier(classifier);
    this.setState({ modelMobilenet, classifier, loading: false }, () => console.log(this.state.classifier?.getNumClasses()));
  }



  async componentDidMount() {
    /*  await this.trainModel(BulbasaurData, CharmanderData, SquirtleData);
     this.saveClassifier(this.state.classifier!); */  
    const modelMobilenet = await mobilenet.load();
    const classifier = this.loadClassifier()
    console.log(classifier.getNumClasses());
    this.setState({ classifier, modelMobilenet, loading: false });
  }

  /**
   * Function to convert Dataset to Tensor
   * @param {DatasetObject} datasetObject The dataset object
   * @returns {Array.<Tensor>} The Tensor array
   */

  fromDatasetObject = (datasetObject: DatasetObject): Dataset => {
    return Object.entries(datasetObject).reduce((result: Dataset, [indexString, { data, shape }]) => {
      const tensor = tf.tensor2d(data, shape);
      const index = Number(indexString);
      result[index] = tensor;
      return result;
    }, {});
  }

  /**
 * Function to load the classifier
 * @returns {knnClassifier.KNNClassifier} The classifier
 */

  loadClassifier = (): knnClassifier.KNNClassifier => {
    const classifier: knnClassifier.KNNClassifier = new knnClassifier.KNNClassifier();
    console.log(model);
    // @ts-ignore
    const dataset = this.fromDatasetObject(model);
    classifier.setClassifierDataset(dataset);
    return classifier;
  }

  /**
   * Function to convert Dataset to Tensor
   * @param {Dataset} dataset The dataset
   * @returns {Promise<DatasetObject>} The dataset object
   */

  toDatasetObject = async (dataset: Dataset): Promise<DatasetObject> => {
    const result: DatasetObject = await Promise.all(
      Object.entries(dataset).map(async ([classId, value], index) => {
        const data = await value.data();
        return {
          classId: Number(classId),
          data: Array.from(data),
          shape: value.shape
        };
      })
    );
    return result;
  };

  /**
   * Function to save the classifier
   * @param {knnClassifier.KNNClassifier} classifier The KNN Classifier
   */


  saveClassifier = async (classifier: knnClassifier.KNNClassifier) => {
    const dataset = classifier.getClassifierDataset();
    const datasetOjb: DatasetObject = await this.toDatasetObject(dataset);
    const jsonStr = JSON.stringify(datasetOjb);
    this.handleSaveToPC(jsonStr);
  }

  /**
   * Downlaods the model.json
   * @param {jsonData} string The JSON Data
   */

  handleSaveToPC = (jsonData: string) => {
    const fileData = jsonData;
    const blob = new Blob([fileData], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = 'model.json';
    link.href = url;
    link.click();
  }

  onChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    let file = URL.createObjectURL(e.target.files![0]);
    this.setState({ image: file })
  }

  /**
   * The prediction used while training
   */

  trainPred = async () => {
    const modelMobilenet = this.state.modelMobilenet;
    const classifier = this.state.classifier;
    const predmobilenet = await modelMobilenet!.classify(this.refs.image as HTMLImageElement);
    const activation = modelMobilenet!.infer(this.refs.image as HTMLImageElement);
    const predclass = await classifier!.predictClass(activation);
    this.setState({ mobilenetPred: predmobilenet, knnPred: predclass, scanned: true });
  }


  render() {
    return (
      <div>
        <Navbar bg="success" variant="dark" expand="lg">
          <Navbar.Brand style={{ color: '#eee' }} href="#home">Pokemon Classification</Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="ml-auto">
              <Nav.Link style={{ color: '#eee' }} target="_blank" rel="noopener noreferrer" href="https://github.com/iberatkaya">GitHub</Nav.Link>
              <Nav.Link style={{ color: '#eee' }} target="_blank" rel="noopener noreferrer" href="https://linkedin.com/in/ibrahim-berat-kaya">LinkedIn</Nav.Link>
              <Nav.Link style={{ color: '#eee' }} target="_blank" rel="noopener noreferrer" href="https://www.npmjs.com/package/@tensorflow-models/mobilenet">MobileNet</Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Navbar>

        <div className="container">
          <div className="row">
            <div className="col-lg-12 text-center">
              <h2 className="m-3">Pokemon Classification</h2>
            </div>
          </div>
        </div>

        {this.state.loading ?
          <div className="container text-center">
            <div className="spinner-border text-danger mb-3" role="status">
            </div>
            <div className="lead">Loading Models...</div>
          </div>
          :
          <div>
            {
              this.state.image === '' ?
                <div className="container justify-center align-items-center">
                  <div className="text-center">
                    <p className="lead" style={{ fontSize: '1.1rem' }}>Upload your Pokemon image to classify it. Images are clasified with Tensorflow.js and MobileNet using transfer learning. Currently only Bulbasaur, Charmander, and Squirtle are classified. </p>
                  </div>
                  <form className={"w-75 form"}>
                    <div className="input-group">
                      <div className="input-group-prepend">
                        <span className="input-group-text">Upload</span>
                      </div>
                      <div className="custom-file">
                        <input value={this.state.image} onChange={this.onChange} accept="image/*" type="file" className="custom-file-input" />
                        <label className="custom-file-label">Choose image</label>
                      </div>
                    </div>
                  </form>
                </div>
                :
                <div className="container-fluid">
                  <div className="row">
                    <div className="col-lg-12">
                      <div className="text-center">
                        <img alt="input" ref="image" style={{ maxWidth: '60%' }} className="img-responsive" src={this.state.image}></img>
                      </div>
                      <p></p> {/* Moves button to bottom of image */}
                      {!this.state.scanned ?
                        <div className="text-center">
                          <button className="btn btn-outline-primary" onClick={async () => {
                            const modelMobilenet = this.state.modelMobilenet;
                            const classifier = this.state.classifier;
                            const predmobilenet = await modelMobilenet!.classify(this.refs.image as HTMLImageElement);
                            const activation = modelMobilenet!.infer(this.refs.image as HTMLImageElement);
                            const predclass = await classifier!.predictClass(activation);
                            console.log(predclass);
                            this.setState({ mobilenetPred: predmobilenet, knnPred: predclass, scanned: true });

                          }}>Classify</button>
                        </div>
                        :
                        <div className="container">
                          <ul className="list-group mb-4">
                            <li className="list-group-item disabled">KNN Predictions</li>
                            <li className="list-group-item">Bulbasaur - Probability: {(this.state.knnPred!.confidences["0"] * 100).toFixed(2)}%</li>
                            <li className="list-group-item">Charmander - Probability: {(this.state.knnPred!.confidences["1"] * 100).toFixed(2)}%</li>
                            <li className="list-group-item">Squirtle - Probability: {(this.state.knnPred!.confidences["2"] * 100).toFixed(2)}%</li>
                          </ul>
                          <div className="text-center">
                            <button className="btn btn-outline-primary" onClick={async () => {
                              this.setState({ image: '', mobilenetPred: [], knnPred: null, scanned: false })
                            }}>Classify New Image</button>
                          </div>
                        </div>
                      }
                    </div>
                  </div>
                </div>
            }
          </div>
        }
      </div>
    )
  }
}

export default App

import React, { Component } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as tf from '@tensorflow/tfjs';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import {isMobile} from 'react-device-detect';
import SquirtleData from './data/squirtle/squirtle';
import CharmanderData from './data/charmander/charmander';
import BulbasaurData from './data/bulbasaur/bulbasaur';
import './App.css';

/**
 * A simple React App to test image recognition models MobileNet and COCOSSD
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
    let arr = isMobile ? [Object.values(BulbasaurData).filter((i, index) => {if(index < 20) return i;}), Object.values(CharmanderData).filter((i, index) => {if(index < 20) return i;}), Object.values(SquirtleData).filter((i, index) => {if(index < 20) return i;})].flat() : [Object.values(BulbasaurData), Object.values(CharmanderData), Object.values(SquirtleData)].flat()
    for (let i = 0; i < arr.length; i++) {
      const im = new Image()
      im.src = arr[i]!;
      im.width = 180;
      im.height = 180;
      im.className = "hidden";
      document.body.appendChild(im);
      im.id = "img" + (i + 1).toString()
    }
  }

  async componentDidMount() {
    const modelMobilenet = await mobilenet.load({ version: 2, alpha: 1 });
    const classifier = knnClassifier.create();
    let arr = isMobile ? [Object.values(BulbasaurData).filter((i, index) => {if(index < 20) return i;}), Object.values(CharmanderData).filter((i, index) => {if(index < 20) return i;}), Object.values(SquirtleData).filter((i, index) => {if(index < 20) return i;})].flat() : [Object.values(BulbasaurData), Object.values(CharmanderData), Object.values(SquirtleData)].flat()
    for (let i = 0; i < arr.length; i++) {
      let image = tf.browser.fromPixels(document.getElementById("img" + (i + 1).toString()) as HTMLImageElement);
      const inferred = modelMobilenet.infer(image);
      if (i <= (isMobile ? 19 : 100))
        classifier.addExample(inferred, 'Balbasaur');
      else if ((i > (isMobile ? 19 : 100) && i <= (isMobile ? 38 : 200)))
        classifier.addExample(inferred, 'Charmander');
      else
        classifier.addExample(inferred, 'Squirtle');
    }
    this.setState({ modelMobilenet, classifier, loading: false }, () => console.log(this.state.classifier?.getNumClasses()));
  }


  onChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    let file = URL.createObjectURL(e.target.files![0]);
    this.setState({ image: file })
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
                    <p className="lead" style={{fontSize: '1.1rem'}}>Upload your Pokemon image to classify it. Images are clasified with Tensorflow.js and MobileNet using transfer learning. Currently only Bulbasaur, Charmander, and Squirtle are classified. </p>
                  </div>
                  <form className={isMobile ? "form m-2" : "w-75 form"}>
                    { isMobile ? <p className="text-muted text-center">For better performance, view on a desktop.</p> : <div></div> }
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
                            this.setState({ mobilenetPred: predmobilenet, knnPred: predclass, scanned: true });

                          }}>Classify</button>
                        </div>
                        :
                        <div className="container">
                          <ul className="list-group mb-4">
                            <li className="list-group-item disabled">KNN</li>
                            <li className="list-group-item">Prediction: Balbasaur - Probability: {this.state.knnPred!.confidences.Balbasaur}</li>
                            <li className="list-group-item">Prediction: Charmander - Probability: {this.state.knnPred!.confidences.Charmander}</li>
                            <li className="list-group-item">Prediction: Squirtle - Probability: {this.state.knnPred!.confidences.Squirtle}</li>
                          </ul>
                          <div className="text-center">
                            <button className="btn btn-outline-primary" onClick={async () => {
                              this.setState({ image: '', mobilenetPred: [], scanned: false })
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

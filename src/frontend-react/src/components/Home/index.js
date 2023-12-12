import React, { useEffect, useRef, useState } from 'react';
import { withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';

import DataService from "../../services/DataService";
import styles from './styles';

const Home = (props) => {
    const { classes } = props;

    console.log("================================== Home ======================================");

    const inputFile = useRef(null);
    const textInput = useRef(null);

    // Component States
    const [image, setImage] = useState(null);
    const [text, setText] = useState(null); 
    const [prediction, setPrediction] = useState(null);

    // Setup Component
    useEffect(() => {

    }, []);

    // Handlers
    const handleImageUploadClick = () => {
        inputFile.current.click();
    }
    const handleOnChange = (event) => {
        console.log(event.target.files);
        const file = event.target.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onloadend = () => {
                setImage(reader.result);
            };

            reader.readAsDataURL(file);
        }
    }

    const handlePredictClick = () => {
        const formData = new FormData();
        formData.append("image", inputFile.current.files[0]);
        formData.append("text", text);
    
        DataService.Predict(formData)
            .then(function (response) {
                console.log(response.data);
                setPrediction(response.data);
            })
            .catch(function (error) {
                console.error("Prediction error:", error);
            });
    };


    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxWidth="md" className={classes.container}>
                    {prediction && (
                        <Typography variant="h4" gutterBottom align='center'>
                        {prediction["Fake Likelihood"]==="High" &&
                            <span className={classes.false}>{"Fake news risk: " + prediction["Fake Likelihood"] + " (" + 100*prediction["Fake Probability"] + "%)"}</span>
                        }
                        {prediction["Fake Likelihood"]==="Low" &&
                            <span className={classes.true}>{"Fake news risk: " + prediction["Fake Likelihood"] + " (" + 100*prediction["Fake Probability"] + "%)"}</span>
                        }
                        </Typography>
                    )}

                    <div className={classes.help}>Please enter the text that you want to verify and an accompanying image: </div>
                    <div className={classes.textInputContainer}>
                        <div className={classes.help1}>Step 1: Enter potential fake news: </div>
                        <textarea
                            rows={5} 
                            wrap="hard"
                            className={classes.textInput}
                            placeholder={"Type text here..."}
                            onChange={(event) => setText(event.target.value)}
                            ref={textInput}
                        />
                        <div className={classes.help1}>Step 2: Enter accompanying image: </div>
                        <input
                            type="file"
                            accept="image/*"
                            capture="camera"
                            autoComplete="off"
                            tabIndex="-1"
                            className={classes.fileInput}
                            ref={inputFile}
                            onChange={(event) => handleOnChange(event)}
                        />
    
                        <button className={classes.uploadButton} onClick={() => handleImageUploadClick()}>
                            Upload Image
                        </button>

                        {image && (
                        
                            <div>
                                <div className={classes.help1}>Image preview:</div>
                                <img className={classes.preview} src={image} alt="Preview" style={{ alignSelf: 'center' }}/>
                                
                            </div>
                            
                        )}
    
                        <button className={classes.predictButton} onClick={() => handlePredictClick()}>
                            Is it fake news?
                        </button>
                    </div>
    
                    {/* <div className={classes.help}>Enter fake news and optionally upload an accompanying image...</div> */}
                </Container>
            </main>
        </div>
    )};
    
export default withStyles(styles)(Home);
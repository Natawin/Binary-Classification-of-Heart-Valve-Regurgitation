## Overview

This project aims to develop an AI-powered diagnostic support system for detecting heart valve regurgitation using **phonocardiogram (PCG)** recordings. By leveraging digital heart sound technology, the tool provides a screening method for early detection, especially in regions with limited medical resources. The system is designed to be integrated with medical devices such as a digital stethoscope for practical healthcare applications.

## Problem Statement

Heart valve regurgitation occurs when blood leaks backward through a heart valve, which, if left undiagnosed, can lead to complications such as heart failure or arrhythmias. Traditional diagnosis often requires echocardiography, which is not always available in rural or resource-limited areas.

This project addresses the problem by building a shared AI classification model capable of evaluating multiple heart valves (mitral, aortic, tricuspid, and pulmonary) using phonocardiogram data.

## Features

- Shared Model Design: A single binary classifier trained across multiple heart valves.

- AI-based Detection: Identifies whether a phonocardiogram signal indicates valve regurgitation.

- Streamlit Web Application (Demo): Provides an accessible, user-friendly interface for testing with prepared example data.

- Scalable for Medical Use: Designed to integrate with digital stethoscopes for real-world healthcare applications.

## Deployment

You can try the tool directly via the deployed Streamlit Demo App here:
👉 [Heart Valve Regurgitation Classifier](https://binary-classification-of-heart-valve-regurgitation-zjfe6sgsybs.streamlit.app/)

⚠️ Note: The current Streamlit version is a demo for demonstration purposes only.
Users cannot upload their own recordings but can explore the model using the example data provided.

## How the Model Works

**Input**: Example phonocardiogram signals are used.

**Preprocessing**: Signals are normalized and transformed into features suitable for analysis.

**Classification**: The AI model predicts whether regurgitation is present in the given valve recording.

**Output**: Results indicate Normal or Regurgitation Detected, along with a confidence score.
Example Output
```bash
Valve: Mitral
Prediction: Normal
```

or
```bash
Valve: Aortic
Prediction: Abnormal
```
## Limitations

- The current version is a demo that only works with prepared example data and does not yet support real file uploads.

- The accuracy depends heavily on the quality of the phonocardiogram signals.

- The system cannot provide 100% reliable or clinically validated results. It is designed to serve as a decision-support tool for healthcare professionals, not as a final medical diagnosis.

- Further validation with larger and more diverse datasets is required before real-world deployment.

## Future Work

- Expand dataset coverage for improved model generalization.

- Enable real file upload and live signal recording through Streamlit.

- Enhance mobile integration for real-time screening in remote areas.

- Collaborate with medical professionals for clinical validation and deployment.

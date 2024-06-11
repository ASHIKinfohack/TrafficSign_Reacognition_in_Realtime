Traffic Sign Recognition and Auditory Alert System

The Traffic Sign Recognition and Auditory Alert System is a real-time application designed to enhance road safety by accurately identifying traffic signs and providing auditory alerts to drivers. Utilizing Convolutional Neural Networks (CNNs) for detection and a Text-to-Speech (TTS) module for notifications, this system ensures drivers are promptly informed about traffic signs, improving their situational awareness and decision-making.
Features

    🛣️ Real-time Traffic Sign Detection: Continuously processes video frames to detect various traffic signs.
    🗣️ Auditory Alerts: Announces detected traffic signs using a Text-to-Speech (TTS) engine.
    🎥 Live Video Feed Processing: Captures and processes frames from a live video feed.
    🌐 High Accuracy: Utilizes a trained CNN model to achieve high detection accuracy.
    📈 Robust Performance: Effective in diverse environmental conditions including low light and adverse weather.

Technologies Used

    Backend: TensorFlow, OpenCV
    Frontend: Tkinter (for GUI)
    Machine Learning: Convolutional Neural Networks (CNNs)
    Audio Processing: pyttsx3 (for TTS)
    Version Control: Git, GitHub


Usage

    Start the application: Launch the application to begin real-time traffic sign detection.
    Live video feed: The system captures video frames, processes them, and announces detected traffic signs.
    Quit the application: Use the 'QUIT' button to stop the application safely.

Preprocessing Techniques

    Grayscale Conversion: Converts images to grayscale to reduce complexity.
    Resizing: Resizes images to 32x32 pixels for model input.
    Normalization: Normalizes pixel values to [0, 1] range.

Text-to-Speech Engine Configuration

    Voice Selection: Configured to use a female voice.
    Speed and Volume: Adjustable settings to ensure clear and timely alerts.

Video Capture Setup

    Camera Selection: Uses the default camera for video capture.
    Resolution and Frame Rate: Configured for real-time processing without lag.

Evaluation Metrics

    Accuracy: Measures the overall correctness of the system's predictions.
    Precision: Proportion of correctly predicted traffic signs out of all identified instances.
    Recall: Measures the system's ability to detect all instances of a particular traffic sign class.
    F1-Score: Combines precision and recall for a comprehensive performance assessment.

Results and Discussion

    The system demonstrates high accuracy in real-time traffic sign detection.
    Effectively provides auditory alerts to enhance driver awareness and safety.
    Robust performance under various environmental conditions, including low light and adverse weather.

Conclusion and Future Work

    Conclusion: The Traffic Sign Recognition system significantly improves driver safety and navigation through effective real-time traffic sign detection and auditory alerts.
    Future Work: Enhancements will focus on improving adaptability to various environmental conditions and integration with other vehicle systems.

Acknowledgements

    Special thanks to the contributors and the open-source community for their valuable resources and support.

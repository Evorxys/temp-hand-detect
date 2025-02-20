const videoElement = document.getElementById('video');
const predictionElement = document.getElementById('prediction');

// Ensure DOM elements exist
document.addEventListener('DOMContentLoaded', () => {
    if (!videoElement || !predictionElement) {
        console.error('❌ Required elements not found');
        return;
    }

    // Initialize MediaPipe Hands
    const hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`;
        }
    });

    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    hands.onResults(onResults);

    // Setup Camera
    const camera = new Camera(videoElement, {
        onFrame: async () => {
            try {
                await hands.send({ image: videoElement });
            } catch (error) {
                console.error('❌ Hand detection error:', error);
            }
        },
        width: 640,
        height: 480
    });

    camera.start()
        .catch(error => {
            console.error('❌ Camera start error:', error);
            predictionElement.textContent = 'Camera error: ' + error.message;
        });
});

// Process detected hand landmarks
async function onResults(results) {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];

        // Flatten landmarks for prediction
        const flattenedLandmarks = landmarks.flatMap(landmark => 
            [landmark.x, landmark.y, landmark.z]
        );

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ landmarks: flattenedLandmarks })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            predictionElement.textContent = 
                `${data.gesture} (${(data.confidence * 100).toFixed(2)}%)`;
        } catch (error) {
            console.error('❌ Prediction error:', error);
            predictionElement.textContent = 'Error in prediction';
        }
    }
}

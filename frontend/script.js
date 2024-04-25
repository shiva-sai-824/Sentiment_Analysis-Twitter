function analyzeSentiment() {
    var textInput = document.getElementById("textInput").value.trim();

    if (textInput === "") {
        document.getElementById("result").innerText = "Please provide input text";
        return;
    }

    var requestData = { text: textInput };
    console.log('Request Data:', requestData); // Log request payload

    fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Assuming the response from the backend contains the sentiment analysis result
            console.log('Response Data:', data); // Log response data
            document.getElementById("result").innerText = "Sentiment: " + data.sentiment;
            document.getElementById("textInput").value = "";
        })
        .catch(error => {
            console.error('There was a problem with your fetch operation:', error);
            document.getElementById("result").innerText = "Failed to analyze sentiment";
        });
}

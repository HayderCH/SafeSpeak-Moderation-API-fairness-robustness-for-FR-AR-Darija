# SafeSpeak Web Interface

A beautiful, modern web interface for the SafeSpeak multilingual toxicity detection API.

## 🚀 Quick Start

### Prerequisites

- SafeSpeak API running on `http://127.0.0.1:8002`
- Python HTTP server for serving the interface

### Start the Interface

1. **Start the SafeSpeak API** (if not already running):

   ```bash
   cd safespeak-project
   .\.venv\Scripts\activate
   python -c "import uvicorn; from scripts.safespeak_api import app; uvicorn.run(app, host='127.0.0.1', port=8002)"
   ```

2. **Start the Web Interface Server**:

   ```bash
   cd safespeak-project
   python -m http.server 8081
   ```

3. **Open your browser** and go to:
   ```
   http://127.0.0.1:8081/interface.html
   ```

## 🎨 Features

### ✨ Modern UI

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Beautiful Interface**: Gradient backgrounds, smooth animations
- **Real-time Feedback**: Loading states and error handling

### 🔍 Text Analysis

- **Multilingual Support**: English, French, Arabic, Spanish, German, and more
- **Real-time Classification**: Instant toxicity detection
- **Confidence Scores**: Visual confidence indicators
- **Language Detection**: Automatic language identification

### 📊 Detailed Results

- **Prediction Badges**: Clear SAFE/TOXIC indicators
- **Confidence Visualization**: Progress bar showing model confidence
- **Processing Metrics**: Response time and request tracking
- **Language Information**: Detected language display

### 🎯 Example Texts

- **Pre-loaded Examples**: Quick test cases for different languages
- **Safe & Toxic Samples**: Both positive and negative examples
- **Multilingual Content**: Examples in multiple languages

## 🛠️ Technical Details

### API Integration

- **Endpoint**: `POST /predict`
- **Request Format**: `{"text": "your text here"}`
- **Response Format**: JSON with prediction, confidence, and metadata

### Supported Languages

- English (en)
- French (fr)
- Arabic (ar)
- Spanish (es)
- German (de)
- And many more...

### Keyboard Shortcuts

- **Ctrl/Cmd + Enter**: Analyze text quickly

## 📱 Usage Instructions

1. **Enter Text**: Type or paste text in the input area
2. **Click Analyze**: Press the "Analyze Text" button or use Ctrl+Enter
3. **View Results**: See prediction, confidence, and language information
4. **Try Examples**: Click example buttons for quick testing

## 🔧 Troubleshooting

### API Connection Issues

- Ensure SafeSpeak API is running on port 8002
- Check browser console for network errors
- Verify CORS settings if accessing from different domain

### Interface Not Loading

- Make sure HTTP server is running on port 8081
- Check that `interface.html` exists in the project root
- Try refreshing the page

### Model Not Responding

- Check API health endpoint: `http://127.0.0.1:8002/health`
- Verify model files are present in `results/` directory
- Check API logs for error messages

## 🎨 Customization

### Styling

The interface uses modern CSS with:

- CSS Grid and Flexbox for layouts
- CSS custom properties for theming
- Smooth transitions and animations
- Mobile-responsive design

### API Configuration

Update the `API_BASE_URL` constant in the JavaScript to point to your API:

```javascript
const API_BASE_URL = "http://your-api-host:port";
```

## 📄 Files

- `interface.html` - Main web interface
- `scripts/safespeak_api.py` - FastAPI backend
- `docs/` - Documentation and guides

## 🤝 Contributing

Feel free to improve the interface by:

- Adding more example texts
- Improving the UI/UX design
- Adding new features like batch processing
- Supporting additional languages

## 📄 License

This interface is part of the SafeSpeak project. See project documentation for licensing details.

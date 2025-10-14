# Voice Cloning Studio - Frontend

React-based web application for voice cloning with real-time progress tracking and audio visualization.

## Features

- **Audio Upload**: Drag-and-drop audio file upload with validation
- **Real-time Progress**: Server-Sent Events (SSE) for live training/conversion progress
- **Waveform Visualization**: Interactive audio player with WaveSurfer.js
- **Quality Reports**: Audio quality analysis with SNR and duration metrics
- **Responsive Design**: Mobile-first design with Tailwind CSS
- **State Management**: Zustand for lightweight, performant state management

## Tech Stack

- **Framework**: React 18 with Vite
- **State Management**: Zustand
- **HTTP Client**: Axios
- **Audio Visualization**: WaveSurfer.js
- **UI Components**: Tailwind CSS + Custom Components
- **File Upload**: React Dropzone

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Development

The app will be available at `http://localhost:5173` (default Vite port).

## Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── AudioUpload.jsx  # Drag-and-drop file upload
│   │   ├── ProgressBar.jsx  # Real-time progress display
│   │   ├── WaveformPlayer.jsx # Audio player with waveform
│   │   └── QualityReport.jsx # Audio quality metrics
│   ├── pages/               # Page components
│   │   ├── TrainingFlow.jsx # Voice model training workflow
│   │   └── ConversionFlow.jsx # Audio conversion workflow
│   ├── store/               # State management
│   │   └── appStore.js      # Zustand store
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── public/                  # Static assets
├── index.html               # HTML template
├── package.json             # Dependencies
├── vite.config.js           # Vite configuration
└── tailwind.config.js       # Tailwind CSS configuration
```

## Usage

### Training a Voice Model

1. Upload 5-10 minutes of audio containing your voice
2. Review the audio quality report
3. Enter a model name
4. Start training (30-40 minutes on RTX 3070)
5. Monitor progress in real-time

### Converting Audio

1. After training completes, upload target audio
2. Monitor conversion progress (2-5 minutes)
3. Listen to the result with the waveform player
4. Download the converted audio

## API Integration

The frontend connects to the backend API at `http://localhost:8000/api`.

### Key Endpoints

- `POST /api/upload/training-audio` - Upload training audio
- `POST /api/upload/target-audio` - Upload target audio
- `POST /api/jobs/train` - Start training job
- `POST /api/jobs/convert` - Start conversion job
- `GET /api/stream/progress/{job_id}` - SSE progress stream
- `GET /api/download/{job_id}` - Download result

## Configuration

Edit `src/store/appStore.js` to change the API URL:

```javascript
const API_URL = 'http://localhost:8000/api';
```

## Building for Production

```bash
npm run build
```

The production build will be in the `dist/` directory. Serve it with any static file server.

## Troubleshooting

### Backend Connection Issues

- Ensure backend is running on `http://localhost:8000`
- Check browser console for CORS errors
- Verify API endpoints are accessible

### SSE Connection Drops

- Check backend SSE implementation
- Verify network stability
- Refresh page to reconnect

### Audio Playback Issues

- Ensure audio format is supported (WAV, MP3, M4A, FLAC)
- Check browser audio permissions
- Try a different browser

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

Part of the Voice Cloning Studio project.

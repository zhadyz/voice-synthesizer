# Quick Start Guide - Frontend

## Prerequisites

1. **Node.js 16+** installed
2. **Backend API** running on `http://localhost:8000`

## Installation & Setup

### Step 1: Navigate to Frontend Directory

```bash
cd frontend
```

### Step 2: Install Dependencies

```bash
npm install
```

This installs:
- React 18
- Vite (build tool)
- Zustand (state management)
- Axios (HTTP client)
- WaveSurfer.js (audio visualization)
- React Dropzone (file uploads)
- Tailwind CSS (styling)

### Step 3: Start Development Server

```bash
npm run dev
```

The app will be available at: **http://localhost:5173**

## Usage

### Training a Voice Model

1. Open http://localhost:5173 in your browser
2. Upload 5-10 minutes of audio containing your voice
3. Review the audio quality report
4. Enter a model name (e.g., "My Voice")
5. Click "Start Training"
6. Monitor progress in real-time (takes 30-40 minutes on RTX 3070)

### Converting Audio

1. After training completes, you'll automatically move to the conversion step
2. Upload any audio file you want to convert
3. Monitor conversion progress (2-5 minutes)
4. Listen to the result using the waveform player
5. Download the converted audio

## Available Scripts

```bash
# Development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build locally
npm run preview
```

## Configuration

### Change API URL

Edit `src/store/appStore.js`:

```javascript
const API_URL = 'http://localhost:8000/api'; // Change this
```

## Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── AudioUpload.jsx
│   │   ├── ProgressBar.jsx
│   │   ├── WaveformPlayer.jsx
│   │   └── QualityReport.jsx
│   ├── pages/               # Page components
│   │   ├── TrainingFlow.jsx
│   │   └── ConversionFlow.jsx
│   ├── store/               # State management
│   │   └── appStore.js
│   ├── App.jsx              # Main app
│   └── main.jsx             # Entry point
└── package.json
```

## Troubleshooting

### Port Already in Use

If port 5173 is in use:

```bash
# Kill the process or specify a different port
npm run dev -- --port 3000
```

### Backend Connection Failed

1. Verify backend is running: `curl http://localhost:8000/api/health`
2. Check browser console for CORS errors
3. Ensure API URL in `appStore.js` is correct

### Build Fails

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### SSE Connection Drops

- Refresh the page to reconnect
- Check backend logs for SSE errors
- Verify network stability

## Features

- **Drag & Drop Upload**: Intuitive file upload
- **Real-time Progress**: SSE streaming for live updates
- **Waveform Player**: Interactive audio visualization
- **Quality Reports**: SNR and duration analysis
- **Responsive Design**: Works on mobile and desktop
- **Error Handling**: Clear error messages

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Next Steps

1. **Customize UI**: Edit components in `src/components/`
2. **Add Features**: Extend `appStore.js` for new functionality
3. **Deploy**: Build with `npm run build` and serve `dist/` folder

For detailed documentation, see `frontend/README.md`

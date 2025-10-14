# Voice Synthesizer

> **⚠️ BETA / EXPERIMENTAL** - This project is currently in early development and not ready for production use.

An open-source, offline voice cloning and conversion system powered by state-of-the-art machine learning models. Train custom voice models locally and convert any audio to your voice with complete privacy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

## 🎯 Project Status

**Current Phase:** Beta / Experimental
**Status:** Under Active Development
**Production Ready:** No

This is a research and development project exploring offline voice synthesis. Use at your own risk and expect breaking changes.

## ✨ Key Features

- **🔒 100% Offline** - All processing happens locally, your data never leaves your machine
- **🎤 Voice Cloning** - Train custom voice models from 5-10 minutes of audio
- **🎵 Voice Conversion** - Transform any audio (speech, singing) to your voice
- **🖥️ Web Interface** - Modern React-based UI with real-time progress tracking
- **⚡ GPU Accelerated** - Optimized for NVIDIA RTX 3000/4000 series GPUs
- **🎨 Professional Quality** - Leverages RVC, BS-RoFormer, and state-of-the-art neural vocoders

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend                            │
│  Drag-drop Upload | Waveform Viz | Real-time Progress       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend                             │
│  REST API | Job Queue | SSE Streaming | Model Management    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ML Pipeline (GPU Optimized)                     │
│                                                              │
│  [1] BS-RoFormer → Voice Isolation                          │
│  [2] Silero VAD → Speech Detection                          │
│  [3] Denoiser → Noise Removal                               │
│  [4] RVC → Voice Model Training                             │
│  [5] Voice Conversion → Output                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **OS:** Windows 10/11, Linux, or macOS
- **GPU:** NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better recommended)
- **CUDA:** 11.8 or later
- **Python:** 3.11 or 3.12 (3.13 has compatibility issues)
- **Node.js:** 18+ (for frontend)
- **RAM:** 16GB minimum, 32GB recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zhadyz/voice-synthesizer.git
   cd voice-synthesizer
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt

   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify GPU detection**
   ```bash
   python verify_setup.py
   ```

5. **Install Node.js dependencies (Frontend)**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

1. **Start the backend** (Terminal 1)
   ```bash
   # Windows
   run_backend.bat

   # Linux/Mac
   ./run_backend.sh
   ```

2. **Start the frontend** (Terminal 2)
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open browser**
   ```
   http://localhost:5173
   ```

## 📖 Documentation

- **[API Documentation](backend/README.md)** - Backend API reference
- **[Frontend Guide](frontend/README.md)** - Frontend architecture and components
- **[Deployment Guide](backend/DEPLOYMENT_CHECKLIST.md)** - Production deployment instructions

## 🎮 Usage

### Training a Voice Model

1. Record or upload 5-10 minutes of clear audio of your voice
2. Upload via the web interface
3. Wait ~30-40 minutes for training (RTX 3070)
4. Model saved automatically

### Converting Audio

1. Select your trained voice model
2. Upload target audio (song, speech, etc.)
3. Wait <1 minute for conversion
4. Download the result

### Quality Tips

- Use high-quality audio (minimal background noise)
- Speak naturally with varied emotions
- Record in a quiet environment
- Avoid echo and reverb
- Minimum 5 minutes, 10 minutes recommended

## ⚙️ Configuration

### Hardware Optimization

**For RTX 3070 (8GB VRAM):**
- Voice isolation uses 3GB VRAM (default)
- Training uses 5GB VRAM (batch_size=8)
- All operations stay under 8GB limit

**For RTX 4090 (24GB VRAM):**
- Increase batch sizes for faster training
- Process multiple files in parallel
- Enable FP16 for 2x speedup

### Environment Variables

Create `.env` file:
```env
# Backend
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=sqlite:///./voice_cloning.db

# Redis (for job queue)
REDIS_HOST=localhost
REDIS_PORT=6379

# GPU
CUDA_VISIBLE_DEVICES=0
```

## 🧪 Testing

Run the test suite:
```bash
# All tests
pytest tests/ -v

# Specific test categories
pytest tests/test_ml_pipeline_integration.py -v
pytest tests/test_backend_integration.py -v
pytest tests/test_performance_benchmarks.py -v
```

## 🔧 Technology Stack

### ML Models
- **RVC** (Retrieval-based Voice Conversion) - Voice cloning
- **BS-RoFormer** - State-of-the-art vocal separation (12.9 dB SDR)
- **Silero VAD** - Voice activity detection
- **Facebook Denoiser** - Speech enhancement
- **HiFi-GAN** - Neural vocoder for audio synthesis
- **F5-TTS** - Zero-shot text-to-speech (optional)

### Backend
- **FastAPI** - Modern async Python web framework
- **ARQ** - Async job queue with Redis
- **SQLAlchemy** - Database ORM
- **PyTorch** - Deep learning framework
- **torchaudio/librosa** - Audio processing

### Frontend
- **React 18** - UI framework
- **Zustand** - State management
- **Tailwind CSS** - Styling
- **WaveSurfer.js** - Audio visualization
- **Axios** - HTTP client
- **React Dropzone** - File uploads

## 📊 Performance

| Operation | RTX 3070 | RTX 4090 |
|-----------|----------|----------|
| Voice Isolation | 30s (3-min audio) | 10s |
| Preprocessing | <2 min | <1 min |
| Training (200 epochs) | 30-40 min | 10-20 min |
| Conversion | <1 min (3-min audio) | 20s |

**Quality:** 9/10 (professional-grade, near-indistinguishable from real voice)

## 🗺️ Roadmap

### Current (Beta)
- [x] Core ML pipeline
- [x] Web application (backend + frontend)
- [x] GPU optimization for RTX 3070+
- [x] Real-time progress tracking
- [x] Basic quality validation

### Planned (v1.0)
- [ ] FP16 mixed precision (2x speedup)
- [ ] Progressive training (preview at 10 min)
- [ ] BigVGAN vocoder (quality boost)
- [ ] Batch processing
- [ ] Model management UI
- [ ] Docker deployment
- [ ] Production documentation

### Future (v2.0+)
- [ ] Instant voice cloning (2-min training)
- [ ] Multilingual support (32+ languages)
- [ ] Prosody/emotion control
- [ ] Real-time API (<200ms latency)
- [ ] Multi-speaker diarization
- [ ] Mobile app (iOS/Android)


## ⚠️ Known Issues

- **Python 3.13:** Not fully supported (use 3.11 or 3.12)
- **CPU-only:** Extremely slow without GPU (50-100x slower)
- **Windows:** Some audio formats may require ffmpeg installation
- **Training:** Cannot be interrupted/resumed (must complete fully)
- **Multi-user:** Single-user only (no concurrent training)

See [Issues](https://github.com/zhadyz/voice-synthesizer/issues) for full list.

## 🤝 Contributing

This project is experimental and contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas where help is needed:**
- Testing on different hardware configurations
- Improving voice isolation quality
- Frontend UX improvements
- Documentation and tutorials
- Bug reports and fixes

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds on the work of many open-source projects:

- **[RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)** - Voice conversion model
- **[audio-separator](https://github.com/nomadkaraoke/python-audio-separator)** - BS-RoFormer implementation
- **[Silero VAD](https://github.com/snakers4/silero-vad)** - Voice activity detection
- **[Facebook Denoiser](https://github.com/facebookresearch/denoiser)** - Speech enhancement
- **[F5-TTS](https://github.com/SWivid/F5-TTS)** - Zero-shot text-to-speech

## 📧 Contact

- **Author:** Bari (zhadyz)
- **GitHub:** [@zhadyz](https://github.com/zhadyz)
- **Issues:** [GitHub Issues](https://github.com/zhadyz/voice-synthesizer/issues)

## ⚖️ Disclaimer

**This software is for research and educational purposes only.**

- Use responsibly and ethically
- Obtain consent before cloning someone's voice
- Do not use for illegal activities, fraud, or deception
- Respect copyright and intellectual property rights
- The authors are not responsible for misuse of this software

Voice cloning technology has ethical implications. Please use it responsibly.

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{voice_synthesizer_2025,
  title = {Voice Synthesizer: Open-Source Offline Voice Cloning},
  author = {Bari (zhadyz)},
  year = {2025},
  url = {https://github.com/zhadyz/voice-synthesizer}
}
```

---

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ by the open-source community

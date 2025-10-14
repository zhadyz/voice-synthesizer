# Contributing to Voice Synthesizer

Thank you for your interest in contributing to Voice Synthesizer! This document provides guidelines and instructions for contributing to the project.

## 🚀 Project Status

**Current Phase:** Beta / Experimental
This project is in active development. We welcome contributions but expect breaking changes and rapid iteration.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and different perspectives
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discriminatory language, or personal attacks
- Trolling, insulting comments, or spam
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## 🤝 How Can I Contribute?

### 1. Testing & Bug Reports

**High Priority:**
- Test on different hardware configurations (GPUs, CPUs)
- Test with various audio formats and quality levels
- Report bugs with detailed reproduction steps
- Verify fixes and provide feedback

### 2. Documentation

- Improve README clarity
- Add tutorials and guides
- Document API endpoints
- Create video tutorials
- Translate documentation

### 3. Code Contributions

**Areas Needing Help:**

**ML Pipeline:**
- Improve voice isolation quality
- Optimize training speed
- Reduce memory usage
- Add support for additional models

**Backend:**
- Add user authentication
- Implement model versioning
- Add batch processing
- Improve error handling

**Frontend:**
- Enhance UX/UI design
- Add mobile responsiveness
- Implement dark mode
- Add accessibility features

**Infrastructure:**
- Docker optimization
- Kubernetes deployment
- CI/CD pipeline
- Performance monitoring

### 4. Research & Experimentation

- Test new ML models (Seed-VC, GenVC, etc.)
- Experiment with optimization techniques
- Benchmark performance improvements
- Evaluate quality metrics

## 🛠️ Development Setup

### Prerequisites

- Python 3.11 or 3.12
- Node.js 18+
- NVIDIA GPU with CUDA 11.8+
- Git
- 16GB+ RAM

### Fork & Clone

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/yourusername/voice-synthesizer.git
cd voice-synthesizer

# Add upstream remote
git remote add upstream https://github.com/originalauthor/voice-synthesizer.git
```

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dev dependencies

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Run verification
python verify_setup.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
# Backend tests
pytest tests/ -v

# Frontend tests
cd frontend
npm test

# Linting
flake8 src/
black src/ --check
```

## 📐 Coding Standards

### Python (Backend/ML)

- **Style:** Follow PEP 8
- **Formatter:** Black (line length 100)
- **Linter:** Flake8
- **Type Hints:** Use type hints for function signatures
- **Docstrings:** Google-style docstrings for all public functions

```python
def process_audio(audio_path: str, sample_rate: int = 22050) -> np.ndarray:
    """
    Process audio file and return waveform.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default: 22050)

    Returns:
        Audio waveform as numpy array

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio format is unsupported
    """
    pass
```

### JavaScript/React (Frontend)

- **Style:** ESLint with React configuration
- **Formatter:** Prettier
- **Components:** Functional components with hooks
- **Naming:** PascalCase for components, camelCase for functions

```javascript
// Good
function AudioUploader({ onUpload, maxSize = 100 }) {
  const [file, setFile] = useState(null);

  return <div>...</div>;
}

// Avoid
function audioUploader(props) {
  let file = null;
  return <div>...</div>;
}
```

### Git Commit Messages

```
type(scope): brief description

Detailed explanation of changes (optional)

Fixes #issue_number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(ml): add FP16 inference support

Implements mixed precision training and inference using PyTorch AMP.
Reduces VRAM usage by 40% and increases speed by 2x.

Fixes #42
```

## 🧪 Testing

### Test Requirements

- All new features must have tests
- Bug fixes should include regression tests
- Maintain >80% code coverage
- Tests must pass before merging

### Writing Tests

```python
# tests/test_feature.py
import pytest
from src.module import function

def test_function_basic():
    """Test basic functionality"""
    result = function(input_data)
    assert result == expected_output

def test_function_edge_case():
    """Test edge case handling"""
    with pytest.raises(ValueError):
        function(invalid_input)
```

### Running Specific Tests

```bash
# Single test file
pytest tests/test_ml_pipeline.py -v

# Single test function
pytest tests/test_ml_pipeline.py::test_voice_isolation -v

# With coverage
pytest --cov=src tests/
```

## 🔄 Pull Request Process

### Before Submitting

1. ✅ Create an issue describing the problem/feature
2. ✅ Fork the repository and create a feature branch
3. ✅ Make your changes with clear commit messages
4. ✅ Add/update tests for your changes
5. ✅ Run tests and ensure they pass
6. ✅ Update documentation if needed
7. ✅ Ensure code follows style guidelines

### Branch Naming

```
feature/add-batch-processing
fix/gpu-memory-leak
docs/update-installation-guide
refactor/optimize-voice-isolation
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings

## Related Issues
Fixes #issue_number
```

### Review Process

1. Automated checks run (tests, linting)
2. Maintainer reviews code
3. Requested changes addressed
4. Approved and merged

**Review Timeline:** Expect response within 3-7 days.

## 🐛 Issue Reporting

### Before Opening an Issue

- Search existing issues to avoid duplicates
- Check if it's listed in Known Issues (README)
- Try the latest version
- Gather system information

### Issue Template

```markdown
**Description:**
Clear description of the issue

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: Windows 10 / Ubuntu 22.04 / macOS 13
- Python Version: 3.11.5
- PyTorch Version: 2.1.2+cu118
- GPU: NVIDIA RTX 3070
- VRAM: 8GB

**Logs:**
```
Paste relevant error messages or logs
```

**Screenshots:**
If applicable, add screenshots

**Additional Context:**
Any other relevant information
```

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested
- `wontfix`: Will not be worked on

## 🎯 Development Priorities

### High Priority
1. GPU memory optimization
2. Training speed improvements
3. Quality enhancement (prosody, emotion)
4. Documentation and tutorials

### Medium Priority
1. Multi-user support
2. Model management UI
3. Batch processing
4. Docker deployment improvements

### Future
1. Instant voice cloning (Seed-VC)
2. Multilingual support
3. Real-time API
4. Mobile app

## 💬 Communication

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** General questions and ideas
- **Pull Requests:** Code contributions

**Response Time:** Maintainers typically respond within 3-7 days.

## 📚 Additional Resources

- [Project Roadmap](PHASE_5_S_PLUS_OPTIMIZATION.md)
- [Architecture Overview](PHASE_2_CORE_PIPELINE.md)
- [API Documentation](backend/README.md)
- [Development Guides](docs/)

## 🙏 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- CONTRIBUTORS.md file
- Release notes

Thank you for contributing to Voice Synthesizer! 🎉

---

Questions? Open a [Discussion](https://github.com/yourusername/voice-synthesizer/discussions) or reach out via Issues.

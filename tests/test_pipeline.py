"""
Comprehensive Pipeline Testing
Tests each module individually and end-to-end workflow
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.voice_isolator import VoiceIsolator
from src.preprocessing.speech_enhancer import SpeechEnhancer
from src.preprocessing.quality_validator import QualityValidator
from src.training.rvc_trainer import RVCTrainer
from src.inference.voice_converter import VoiceConverter
from src.pipeline.voice_cloning_pipeline import VoiceCloningPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_voice_isolator(audio_path: str):
    """Test voice isolation module"""
    logger.info("=" * 60)
    logger.info("TEST 1: Voice Isolator")
    logger.info("=" * 60)

    try:
        isolator = VoiceIsolator()
        vocals = isolator.isolate_vocals(audio_path)
        isolator.cleanup()

        logger.info(f"✓ Voice isolation successful: {vocals}")
        return True, vocals
    except Exception as e:
        logger.error(f"✗ Voice isolation failed: {e}")
        return False, None


def test_speech_enhancer(audio_path: str):
    """Test speech enhancement module"""
    logger.info("=" * 60)
    logger.info("TEST 2: Speech Enhancer")
    logger.info("=" * 60)

    try:
        enhancer = SpeechEnhancer()
        output = "outputs/clean/test_clean.wav"
        enhancer.extract_clean_speech(audio_path, output)

        logger.info(f"✓ Speech enhancement successful: {output}")
        return True, output
    except Exception as e:
        logger.error(f"✗ Speech enhancement failed: {e}")
        return False, None


def test_quality_validator(audio_path: str):
    """Test quality validation module"""
    logger.info("=" * 60)
    logger.info("TEST 3: Quality Validator")
    logger.info("=" * 60)

    try:
        validator = QualityValidator()
        report = validator.generate_report(audio_path)
        print(report)

        logger.info("✓ Quality validation successful")
        return True
    except Exception as e:
        logger.error(f"✗ Quality validation failed: {e}")
        return False


def test_preprocessing_pipeline(audio_path: str):
    """Test complete preprocessing pipeline"""
    logger.info("=" * 60)
    logger.info("TEST 4: Complete Preprocessing Pipeline")
    logger.info("=" * 60)

    try:
        pipeline = VoiceCloningPipeline()
        result = pipeline.preprocess_training_audio(audio_path, "test_user")

        logger.info(f"✓ Preprocessing pipeline successful")
        logger.info(f"  Clean audio: {result['clean_audio_path']}")
        logger.info(f"  Valid: {result['quality_report']['valid']}")
        logger.info(f"  Time: {result['preprocessing_time']:.1f}s")
        return True, result
    except Exception as e:
        logger.error(f"✗ Preprocessing pipeline failed: {e}")
        return False, None


def test_rvc_trainer(audio_path: str):
    """Test RVC training (requires clean audio)"""
    logger.info("=" * 60)
    logger.info("TEST 5: RVC Trainer (OPTIONAL - SLOW)")
    logger.info("=" * 60)

    try:
        trainer = RVCTrainer()
        model_path = trainer.train_from_audio(audio_path, "test_model", epochs=10)

        logger.info(f"✓ RVC training successful: {model_path}")
        return True, model_path
    except Exception as e:
        logger.error(f"✗ RVC training failed: {e}")
        logger.error("This is expected if RVC is not fully configured")
        return False, None


def test_voice_converter(model_path: str, audio_path: str):
    """Test voice conversion"""
    logger.info("=" * 60)
    logger.info("TEST 6: Voice Converter")
    logger.info("=" * 60)

    try:
        converter = VoiceConverter(model_path=model_path)
        output = "outputs/converted/test_converted.wav"
        converter.convert_voice(audio_path, output)

        logger.info(f"✓ Voice conversion successful: {output}")
        return True
    except Exception as e:
        logger.error(f"✗ Voice conversion failed: {e}")
        return False


def run_all_tests(test_audio: str, skip_training: bool = True):
    """Run all tests in sequence"""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 CORE PIPELINE - COMPREHENSIVE TEST SUITE")
    logger.info("=" * 60 + "\n")

    results = {
        'voice_isolator': False,
        'speech_enhancer': False,
        'quality_validator': False,
        'preprocessing_pipeline': False,
        'rvc_trainer': False,
        'voice_converter': False
    }

    # Test 1: Voice Isolator
    success, vocals = test_voice_isolator(test_audio)
    results['voice_isolator'] = success

    if not success:
        logger.error("Voice isolator failed. Cannot proceed.")
        return results

    # Test 2: Speech Enhancer
    success, clean_audio = test_speech_enhancer(vocals)
    results['speech_enhancer'] = success

    # Test 3: Quality Validator
    if clean_audio:
        success = test_quality_validator(clean_audio)
        results['quality_validator'] = success

    # Test 4: Complete Preprocessing Pipeline
    success, preprocess_result = test_preprocessing_pipeline(test_audio)
    results['preprocessing_pipeline'] = success

    # Test 5: RVC Training (optional, slow)
    if not skip_training and preprocess_result:
        success, model_path = test_rvc_trainer(preprocess_result['clean_audio_path'])
        results['rvc_trainer'] = success

        # Test 6: Voice Converter
        if success and model_path:
            success = test_voice_converter(model_path, test_audio)
            results['voice_converter'] = success

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test:30} {status}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <test_audio.mp3> [--with-training]")
        print()
        print("Examples:")
        print("  python test_pipeline.py test_audio/sample.mp3")
        print("  python test_pipeline.py test_audio/sample.mp3 --with-training")
        sys.exit(1)

    test_audio = sys.argv[1]
    skip_training = "--with-training" not in sys.argv

    results = run_all_tests(test_audio, skip_training=skip_training)

    # Exit with error code if any test failed
    if not all(results.values()):
        sys.exit(1)

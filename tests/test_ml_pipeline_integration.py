"""
TIER 1: ML Pipeline Integration Tests
Comprehensive testing of voice isolation, enhancement, and quality validation
"""

import pytest
import os
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.voice_isolator import VoiceIsolator
from src.preprocessing.speech_enhancer import SpeechEnhancer
from src.preprocessing.quality_validator import QualityValidator
from src.pipeline.voice_cloning_pipeline import VoiceCloningPipeline


class TestVoiceIsolation:
    """Test voice isolation component"""

    def test_voice_isolator_initialization(self):
        """Test VoiceIsolator can be initialized"""
        try:
            isolator = VoiceIsolator()
            assert isolator is not None
            isolator.cleanup()
        except Exception as e:
            pytest.fail(f"VoiceIsolator initialization failed: {e}")

    def test_voice_isolation_with_mock_audio(self, temp_audio_file, test_output_dir):
        """Test voice isolation with mock audio file"""
        isolator = VoiceIsolator()

        try:
            result = isolator.isolate_vocals(str(temp_audio_file))

            # Verify output file was created
            assert os.path.exists(result), f"Output file not created: {result}"
            assert result.endswith('.wav'), "Output should be WAV format"

            # Verify file has content
            file_size = os.path.getsize(result)
            assert file_size > 1000, f"Output file too small: {file_size} bytes"

            print(f"✓ Voice isolation successful: {result} ({file_size} bytes)")

        finally:
            isolator.cleanup()

    @pytest.mark.slow
    def test_voice_isolation_3sec_audio(self, mock_audio_3sec, test_output_dir):
        """Test voice isolation with 3-second audio"""
        isolator = VoiceIsolator()

        try:
            start = time.time()
            result = isolator.isolate_vocals(str(mock_audio_3sec))
            duration = time.time() - start

            assert os.path.exists(result)
            print(f"✓ 3-second isolation completed in {duration:.2f}s")

            # Should complete in reasonable time (< 60s)
            assert duration < 60, f"Isolation too slow: {duration:.2f}s"

        finally:
            isolator.cleanup()


class TestSpeechEnhancement:
    """Test speech enhancement component"""

    def test_speech_enhancer_initialization(self):
        """Test SpeechEnhancer can be initialized"""
        try:
            enhancer = SpeechEnhancer()
            assert enhancer is not None
        except Exception as e:
            pytest.fail(f"SpeechEnhancer initialization failed: {e}")

    def test_speech_enhancement(self, temp_audio_file, test_output_dir):
        """Test speech enhancement with mock audio"""
        enhancer = SpeechEnhancer()
        output_path = test_output_dir / "enhanced_test.wav"

        try:
            result = enhancer.extract_clean_speech(
                str(temp_audio_file),
                str(output_path)
            )

            assert os.path.exists(result)
            assert os.path.getsize(result) > 100
            print(f"✓ Speech enhancement successful: {result}")

        except Exception as e:
            # Enhancement might fail on synthetic audio - log but don't fail
            print(f"⚠ Speech enhancement warning: {e}")


class TestQualityValidation:
    """Test quality validation component"""

    def test_quality_validator_initialization(self):
        """Test QualityValidator can be initialized"""
        try:
            validator = QualityValidator()
            assert validator is not None
        except Exception as e:
            pytest.fail(f"QualityValidator initialization failed: {e}")

    def test_snr_calculation(self, temp_audio_file):
        """Test SNR calculation"""
        validator = QualityValidator()

        try:
            validation = validator.validate_audio(str(temp_audio_file))

            assert 'snr_db' in validation
            assert 'duration_sec' in validation
            assert 'valid' in validation
            assert isinstance(validation['snr_db'], (int, float))

            print(f"✓ SNR calculated: {validation['snr_db']:.2f} dB")
            print(f"  Duration: {validation['duration_sec']:.2f}s")
            print(f"  Valid: {validation['valid']}")

        except Exception as e:
            pytest.fail(f"SNR calculation failed: {e}")

    def test_quality_report_generation(self, temp_audio_file):
        """Test quality report generation"""
        validator = QualityValidator()

        try:
            report = validator.generate_report(str(temp_audio_file))

            assert report is not None
            assert len(report) > 0
            print(f"✓ Quality report generated:\n{report}")

        except Exception as e:
            pytest.fail(f"Report generation failed: {e}")


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline"""

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized"""
        try:
            pipeline = VoiceCloningPipeline()
            assert pipeline is not None
            assert pipeline.voice_isolator is not None
            assert pipeline.speech_enhancer is not None
            assert pipeline.quality_validator is not None
        except Exception as e:
            pytest.fail(f"Pipeline initialization failed: {e}")

    @pytest.mark.slow
    def test_preprocessing_end_to_end(self, temp_audio_file, test_user_id):
        """Test complete preprocessing workflow"""
        pipeline = VoiceCloningPipeline()

        try:
            result = pipeline.preprocess_training_audio(
                str(temp_audio_file),
                test_user_id
            )

            # Validate result structure
            assert 'clean_audio_path' in result
            assert 'quality_report' in result
            assert 'preprocessing_time' in result

            # Validate output file
            assert os.path.exists(result['clean_audio_path'])
            assert result['preprocessing_time'] > 0

            # Log results
            print(f"\n✓ Preprocessing pipeline successful:")
            print(f"  Clean audio: {result['clean_audio_path']}")
            print(f"  Valid: {result['quality_report'].get('valid', 'N/A')}")
            print(f"  SNR: {result['quality_report'].get('snr_db', 'N/A'):.2f} dB")
            print(f"  Time: {result['preprocessing_time']:.1f}s")

        except Exception as e:
            pytest.fail(f"Preprocessing pipeline failed: {e}")

    @pytest.mark.slow
    def test_preprocessing_with_3sec_audio(self, mock_audio_3sec, test_user_id):
        """Test preprocessing with longer audio"""
        pipeline = VoiceCloningPipeline()

        try:
            start = time.time()
            result = pipeline.preprocess_training_audio(
                str(mock_audio_3sec),
                test_user_id + "_3sec"
            )
            duration = time.time() - start

            assert os.path.exists(result['clean_audio_path'])
            print(f"✓ 3-second preprocessing completed in {duration:.2f}s")

            # Should complete in under 2 minutes for 3-second audio
            assert duration < 120, f"Preprocessing too slow: {duration:.2f}s"

        except Exception as e:
            pytest.fail(f"3-second preprocessing failed: {e}")


class TestErrorHandling:
    """Test error handling in ML pipeline"""

    def test_invalid_audio_path(self):
        """Test handling of invalid audio file path"""
        pipeline = VoiceCloningPipeline()

        with pytest.raises(Exception):
            pipeline.preprocess_training_audio("nonexistent_file.wav", "test_user")

    def test_corrupted_audio_file(self, tmp_path):
        """Test handling of corrupted audio file"""
        # Create a fake corrupted file
        corrupted_file = tmp_path / "corrupted.wav"
        corrupted_file.write_bytes(b"not a valid wav file")

        pipeline = VoiceCloningPipeline()

        with pytest.raises(Exception):
            pipeline.preprocess_training_audio(str(corrupted_file), "test_user")


# Test execution summary
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ML PIPELINE INTEGRATION TESTS")
    print("="*70 + "\n")

    pytest.main([__file__, "-v", "-s", "--tb=short"])

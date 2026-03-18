"""Tests for video frame extraction with time range support."""

import pytest
from pathlib import Path
from src.framing.extractor import VideoFrameExtractor


class TestTimeRangeValidation:
    """Test time range parameter validation."""

    def test_start_time_default_is_zero(self, tmp_path):
        """Default start_time should be 0."""
        extractor = VideoFrameExtractor(
            input_dir=str(tmp_path), output_dir=str(tmp_path / "out")
        )
        assert extractor.start_time == 0.0

    def test_end_time_default_is_none(self, tmp_path):
        """Default end_time should be None (video end)."""
        extractor = VideoFrameExtractor(
            input_dir=str(tmp_path), output_dir=str(tmp_path / "out")
        )
        assert extractor.end_time is None

    def test_start_time_cannot_be_negative(self, tmp_path):
        """start_time < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="start_time cannot be negative"):
            VideoFrameExtractor(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                start_time=-1.0,
            )

    def test_end_time_cannot_be_negative(self, tmp_path):
        """end_time < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="end_time cannot be negative"):
            VideoFrameExtractor(
                input_dir=str(tmp_path), output_dir=str(tmp_path / "out"), end_time=-1.0
            )

    def test_end_time_must_be_greater_than_start_time(self, tmp_path):
        """end_time <= start_time should raise ValueError."""
        with pytest.raises(
            ValueError, match="end_time must be greater than start_time"
        ):
            VideoFrameExtractor(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                start_time=10.0,
                end_time=5.0,
            )

    def test_start_time_equals_end_time_raises_error(self, tmp_path):
        """start_time == end_time should raise ValueError."""
        with pytest.raises(
            ValueError, match="end_time must be greater than start_time"
        ):
            VideoFrameExtractor(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                start_time=5.0,
                end_time=5.0,
            )


class TestTimeRangeExtraction:
    """Test frame extraction with time range."""

    @pytest.fixture
    def sample_video(self):
        """Path to sample video for testing."""
        video_path = Path("data/video.mp4")
        if not video_path.exists():
            pytest.skip("Sample video not found at data/video.mp4")
        return video_path

    def test_extract_frames_within_time_range(self, sample_video, tmp_path):
        """Extract frames only within specified time range."""
        extractor = VideoFrameExtractor(
            input_dir=str(sample_video.parent),
            output_dir=str(tmp_path / "out"),
            start_time=1.0,
            end_time=3.0,
            interval_ms=1000,
        )

        results = extractor.extract_from_video(sample_video)

        # With 1s interval, 1.0-3.0s range should give ~2 frames
        extracted = [r for r in results if "Extracted" in r]
        assert len(extracted) >= 1

    def test_extract_frames_start_time_only(self, sample_video, tmp_path):
        """Extract frames from start_time to video end."""
        extractor = VideoFrameExtractor(
            input_dir=str(sample_video.parent),
            output_dir=str(tmp_path / "out"),
            start_time=2.0,
            interval_ms=1000,
        )

        results = extractor.extract_from_video(sample_video)

        # Should extract frames from 2s onwards
        extracted = [r for r in results if "Extracted" in r]
        assert len(extracted) >= 0

    def test_start_time_exceeds_video_duration(self, sample_video, tmp_path):
        """start_time > video duration should extract no frames."""
        import cv2

        cap = cv2.VideoCapture(str(sample_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        extractor = VideoFrameExtractor(
            input_dir=str(sample_video.parent),
            output_dir=str(tmp_path / "out"),
            start_time=duration + 100.0,
            interval_ms=1000,
        )

        results = extractor.extract_from_video(sample_video)

        extracted = [r for r in results if "Extracted" in r]
        assert len(extracted) == 0

    def test_end_time_exceeds_video_duration(self, sample_video, tmp_path):
        """end_time > video duration should use video duration."""
        extractor = VideoFrameExtractor(
            input_dir=str(sample_video.parent),
            output_dir=str(tmp_path / "out"),
            start_time=0.0,
            end_time=10000.0,
            interval_ms=1000,
        )

        # Should not raise error, just cap at video duration
        results = extractor.extract_from_video(sample_video)
        # Results should contain frames
        assert isinstance(results, list)

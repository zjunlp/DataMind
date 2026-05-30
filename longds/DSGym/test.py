"""Simple test for download_competition_data function."""

import tempfile
import shutil
from pathlib import Path

from dsgym.datasets.loaders.kaggle_downloader import KaggleChallengeDownloader


def test_download_competition_data():
    """Test downloading a small Kaggle competition dataset."""
    # Use a temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = KaggleChallengeDownloader(download_dir=tmpdir)
        
        # Test with a well-known, small competition
        competition_name = "titanic"
        # competition_name = "LANL-Earthquake-Prediction"
        result = downloader.download_competition_data(competition_name)
        
        print(f"\nResult: {result}")
        
        # Check the result
        assert result["status"] in ["success", "partial_success"], f"Download failed: {result}"
        assert result["competition_name"] == competition_name
        assert "directory" in result
        
        # Verify files were downloaded
        competition_dir = Path(result["directory"])
        assert competition_dir.exists(), f"Directory not created: {competition_dir}"
        
        downloaded_files = list(competition_dir.iterdir())
        print(f"Downloaded files: {[f.name for f in downloaded_files]}")
        
        assert len(downloaded_files) > 0, "No files were downloaded"
        
        print("\n✓ Test passed!")


def test_kaggle_submission():
    """Test submitting a file to Kaggle competition."""
    from dsgym.eval.metrics.dspredict.dspredict_metric import KaggleSubmissionMetric
    
    # Initialize the metric
    metric = KaggleSubmissionMetric(timeout_minutes=10, online=True)
    
    # Test submission
    submission_path = "./submissions/container_002/playground-series-s3e11_2_20260201_211224_submission.csv"
    challenge_name = "playground-series-s3e11"
    
    result = metric.evaluate(
        prediction=submission_path,
        ground_truth=None,
        query=None,
        extra_info={"challenge_name": challenge_name},
    )
    
    print(f"\nResult: {result}")
    print(f"Score: {result.score}")
    
    print("\n✓ Test completed!")


if __name__ == "__main__":
    # test_download_competition_data()
    test_kaggle_submission()

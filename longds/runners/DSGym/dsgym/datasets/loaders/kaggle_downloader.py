import os
import json
import zipfile
from pathlib import Path
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
from typing import Optional
from bs4 import BeautifulSoup
import requests
from dsgym.datasets.config import REPO_ROOT
class KaggleScraper:
    def __init__(self):
        self.session = requests.Session()
    
    def scrape_overview_and_description(self, competition_url: str) -> tuple[str, str]:
        try:
            response = self.session.get(competition_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            overview_section = soup.find('div', {'class': 'competition-overview'})
            overview_text = overview_section.get_text(strip=True) if overview_section else ""
            
            data_section = soup.find('div', {'id': 'data'})
            data_text = data_section.get_text(strip=True) if data_section else ""
            
            return overview_text, data_text
        except Exception as e:
            return "", ""


class KaggleChallengeDownloader:
    def __init__(self, download_dir: str = "./kaggle_challenges"):
        self.client = KaggleClient()

        self.scraper = KaggleScraper()
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
    def extract_competition_name(self, challenge_input: str) -> str:
        if challenge_input.startswith("https://www.kaggle.com/competitions/"):
            return challenge_input.split("/competitions/")[1].split("/")[0]
        elif challenge_input.startswith("https://www.kaggle.com/c/"):
            return challenge_input.split("/c/")[1].split("/")[0]
        else:
            return challenge_input
    
    def download_competition_data(self, competition_name: str) -> dict:
        print(f"Downloading competition: {competition_name}")
        
        competition_dir = self.download_dir / competition_name
        competition_dir.mkdir(exist_ok=True)
        
        try:
            try:
                competition_url = f"https://www.kaggle.com/competitions/{competition_name}"
                overview_cleaned, data_cleaned = self.scraper.scrape_overview_and_description(competition_url)
                
                if overview_cleaned or data_cleaned:
                    final_description = f"Challenge description:\n{overview_cleaned}\n\nData description:\n{data_cleaned}"
                    with open(competition_dir / "description.txt", "w", encoding='utf-8') as f:
                        f.write(final_description)
                    print(f"  ✓ Description saved (overview + data)")
                else:
                    print(f"  ⚠ Could not scrape description")
            except Exception as e:
                print(f"  ⚠ Description scraping failed: {str(e)}")
            
            try:
                download_request = ApiDownloadDataFilesRequest()
                download_request.competition_name = competition_name
                
                dataset_response = self.client.competitions.competition_api_client.download_data_files(download_request)
                
                dataset_file = competition_dir / f"{competition_name}.zip"
                with open(dataset_file, "wb") as f:
                    if hasattr(dataset_response, 'content'):
                        f.write(dataset_response.content)
                    elif hasattr(dataset_response, 'read'):
                        f.write(dataset_response.read())
                    else:
                        f.write(dataset_response)
                
                print(f"  ✓ Dataset downloaded: {dataset_file}")
                
                try:
                    with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
                        zip_ref.extractall(competition_dir)
                    print(f"  ✓ Dataset extracted to: {competition_dir}")
                    
                    dataset_file.unlink()
                    print(f"  ✓ Zip file removed")
                    
                    extracted_files = [f.name for f in competition_dir.iterdir() if f.is_file() and f.name != "description.txt"]
                    
                    base_files = []
                    if (competition_dir / "description.txt").exists():
                        base_files.append("description.txt")
                    
                    return {
                        "status": "success",
                        "competition_name": competition_name,
                        "directory": str(competition_dir),
                        "files_downloaded": base_files + extracted_files
                    }
                except zipfile.BadZipFile:
                    print(f"  ⚠ Could not extract zip file (may be corrupted or not a zip)")
                    
                    base_files = []
                    if (competition_dir / "description.txt").exists():
                        base_files.append("description.txt")
                    base_files.append(f"{competition_name}.zip")
                    
                    return {
                        "status": "success",
                        "competition_name": competition_name,
                        "directory": str(competition_dir),
                        "files_downloaded": base_files
                    }
                
            except Exception as e:
                print(f"  ⚠ Could not download dataset: {str(e)}")
                
                partial_files = []
                if (competition_dir / "description.txt").exists():
                    partial_files.append("description.txt")
                
                return {
                    "status": "partial_success",
                    "competition_name": competition_name,
                    "directory": str(competition_dir),
                    "files_downloaded": partial_files,
                    "error": f"Dataset download failed: {str(e)}"
                }
                
        except Exception as e:
            print(f"  ✗ Failed to download {competition_name}: {str(e)}")
            return {
                "status": "failed",
                "competition_name": competition_name,
                "error": str(e)
            }
    
    def download_challenges(self, challenge_list: list[str]) -> dict:
        results = []
        successful_downloads = 0
        failed_downloads = 0
        
        print(f"Starting download of {len(challenge_list)} challenges...")
        print(f"Download directory: {self.download_dir.absolute()}")
        print("-" * 50)
        
        for i, challenge in enumerate(challenge_list, 1):
            print(f"[{i}/{len(challenge_list)}] Processing: {challenge}")
            
            competition_name = self.extract_competition_name(challenge)
            result = self.download_competition_data(competition_name)
            results.append(result)
            
            if result["status"] in ["success", "partial_success"]:
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            print()
        
        summary = {
            "total_challenges": len(challenge_list),
            "successful_downloads": successful_downloads,
            "failed_downloads": failed_downloads,
            "results": results,
            "download_directory": str(self.download_dir.absolute())
        }
        
        summary_file = self.download_dir / "download_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print("=" * 50)
        print("DOWNLOAD SUMMARY")
        print("=" * 50)
        print(f"Total challenges: {summary['total_challenges']}")
        print(f"Successful downloads: {summary['successful_downloads']}")
        print(f"Failed downloads: {summary['failed_downloads']}")
        print(f"Summary saved to: {summary_file}")
        
        return summary


def download_from_file(file_path: str, download_dir: str = "./kaggle_challenges") -> dict:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    challenge_list = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    downloader = KaggleChallengeDownloader(download_dir)
    return downloader.download_challenges(challenge_list)


def download_from_list(challenge_list: list[str], download_dir: str = "./kaggle_challenges") -> dict:
    downloader = KaggleChallengeDownloader(download_dir)
    return downloader.download_challenges(challenge_list)

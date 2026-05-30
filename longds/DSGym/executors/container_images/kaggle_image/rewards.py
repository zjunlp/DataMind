import os
import time
import statistics
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiGetSubmissionRequest,
    ApiGetLeaderboardRequest,
    ApiStartSubmissionUploadRequest,
    ApiListSubmissionsRequest
)
from kagglesdk.competitions.types.submission_status import SubmissionStatus

class KaggleSubmissionManager:
    def __init__(self, username=None):
        self.client = KaggleClient()
        self.username = username or "federicobianchi2"

    def submit_file(self, competition_name, file_path, description=""):
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        upload_request = ApiStartSubmissionUploadRequest()
        upload_request.competition_name = competition_name
        upload_request.content_length = file_size
        upload_request.file_name = file_name
        
        upload_response = self.client.competitions.competition_api_client.start_submission_upload(upload_request)
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
            import requests
            requests.put(upload_response.create_url, data=file_data)
        
        submit_request = ApiCreateSubmissionRequest()
        submit_request.competition_name = competition_name
        submit_request.blob_file_tokens = upload_response.token
        submit_request.submission_description = description
        
        response = self.client.competitions.competition_api_client.create_submission(submit_request)
        return response.ref

    def get_submission_score(self, submission_ref):
        request = ApiGetSubmissionRequest()
        request.ref = submission_ref
        
        submission = self.client.competitions.competition_api_client.get_submission(request)
        return {
            'public_score': submission.public_score,
            'private_score': submission.private_score,
            'status': submission.status
        }

    def wait_for_submission(self, submission_ref, timeout_minutes=30):
        print(f"Waiting for submission {submission_ref} to complete...")
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            score_info = self.get_submission_score(submission_ref)
            status = score_info['status']
            
            if status == SubmissionStatus.COMPLETE:
                print(f"Submission completed! Score: {score_info['public_score']}")
                return score_info
            elif status == SubmissionStatus.ERROR:
                print("Submission failed with error")
                return score_info
            
            print(f"Status: {status.name}, waiting...")
            time.sleep(30)
        
        print(f"Timeout after {timeout_minutes} minutes")
        return self.get_submission_score(submission_ref)

    def get_user_submissions(self, competition_name):
        request = ApiListSubmissionsRequest()
        request.competition_name = competition_name
        
        response = self.client.competitions.competition_api_client.list_submissions(request)
        return response.submissions

    def get_leaderboard_scores(self, competition_name):
        # Get user's own submissions to identify all team names they've used
        user_submissions = self.get_user_submissions(competition_name)
        print(f"Found {len(user_submissions)} user submissions")
        
        user_teams = set()
        for sub in user_submissions:
            print(f"User submission: team_name='{sub.team_name}', submitted_by='{sub.submitted_by}', ref={sub.ref}")
            if sub.team_name:
                user_teams.add(sub.team_name)
            if sub.submitted_by:
                user_teams.add(sub.submitted_by)
        user_teams.add(self.username)
        
        print(f"Filtering out submissions from teams: {user_teams}")
        
        # Get leaderboard
        request = ApiGetLeaderboardRequest()
        request.competition_name = competition_name
        
        response = self.client.competitions.competition_api_client.get_leaderboard(request)
        print(f"Leaderboard has {len(response.submissions)} entries")
        
        all_scores = []
        user_scores = []
        
        for submission in response.submissions:
            if submission.score:
                score_val = float(submission.score)
                all_scores.append(score_val)
                
                # Check if this submission belongs to the user
                if submission.team_name in user_teams:
                    print(f"Found user submission on leaderboard: team='{submission.team_name}', score={score_val}")
                    user_scores.append(score_val)
        
        other_scores = [s for s in all_scores if s not in user_scores]
        
        return {
            'all_scores': all_scores,
            'other_scores': other_scores,
            'user_scores': user_scores,
            'user_teams_found': list(user_teams),
            'normalized_mean': statistics.mean(other_scores) if other_scores else 0,
            'normalized_median': statistics.median(other_scores) if other_scores else 0
        }

    def get_leaderboard_stats(self, competition_name):
        request = ApiGetLeaderboardRequest()
        request.competition_name = competition_name
        
        response = self.client.competitions.competition_api_client.get_leaderboard(request)
        print(f"Leaderboard has {len(response.submissions)} entries")
        
        public_scores = []
        private_scores = []
        
        for submission in response.submissions:
            if submission.score:
                try:
                    public_score = float(submission.score)
                    public_scores.append(public_score)
                except:
                    pass
        
        return {
            'public_scores': public_scores,
            'private_scores': private_scores,
            'public_mean': statistics.mean(public_scores) if public_scores else 0,
            'public_median': statistics.median(public_scores) if public_scores else 0,
            'private_mean': statistics.mean(private_scores) if private_scores else 0,
            'private_median': statistics.median(private_scores) if private_scores else 0,
            'total_submissions': len(response.submissions)
        }

def test_leaderboard_filtering():
    # Test just the leaderboard filtering without submitting
    manager = KaggleSubmissionManager()
    competition = "playground-series-s5e5"
    
    print("Testing leaderboard filtering...")
    leaderboard_stats = manager.get_leaderboard_scores(competition)
    print(f"Found {len(leaderboard_stats['user_scores'])} of your submissions")
    print(f"Normalized stats from {len(leaderboard_stats['other_scores'])} other submissions")
    print(f"Mean: {leaderboard_stats['normalized_mean']:.6f}")
    print(f"Median: {leaderboard_stats['normalized_median']:.6f}")
    
    # Also check if we can find the user's submissions even if not on leaderboard
    print("\n--- Checking user's submission history ---")
    user_subs = manager.get_user_submissions(competition)
    print(f"User has {len(user_subs)} total submissions")
    for sub in user_subs[-5:]:  # Show last 5
        print(f"  Submission {sub.ref}: {sub.public_score} (status: {sub.status.name})")

def main():
    # Use default Kaggle credentials from ~/.kaggle/kaggle.json
    manager = KaggleSubmissionManager()
    
    competition = "playground-series-s5e5"
    
    print("=== Testing Submission Workflow ===")
    
    # Submit a file
    print("\n1. Submitting file...")
    submission_ref = manager.submit_file(competition, "sample_submission.csv", "Test submission from script")
    print(f"✅ Submitted with ref: {submission_ref}")
    
    # Wait for submission to complete
    print("\n2. Waiting for submission to complete...")
    score_info = manager.wait_for_submission(submission_ref, timeout_minutes=10)
    print(f"✅ Final score info: {score_info}")
    
    # Get leaderboard statistics
    print("\n3. Getting leaderboard statistics...")
    stats = manager.get_leaderboard_stats(competition)
    print(f"📊 Leaderboard Stats ({stats['total_submissions']} submissions):")
    print(f"   Public scores - Mean: {stats['public_mean']:.6f}, Median: {stats['public_median']:.6f}")
    if stats['private_scores']:
        print(f"   Private scores - Mean: {stats['private_mean']:.6f}, Median: {stats['private_median']:.6f}")
    else:
        print("   Private scores not yet available")
    
    print("\n=== Complete! ===")

if __name__ == "__main__":
    main()
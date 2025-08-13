# visual_asset_manager.py - Manages searching and downloading video assets from Pexels API

import os
from typing import List, Optional
import uuid

import requests


class VisualAssetManager:
    """
    Manages searching and downloading video assets from the Pexels API.
    """

    def __init__(self, pexels_api_key: str):
        """
        Initialize the VisualAssetManager.

        Args:
            pexels_api_key: Pexels API key for authentication
        """
        if not pexels_api_key:
            raise ValueError("Pexels API key is required.")

        self.api_key = pexels_api_key
        self.search_url = "https://api.pexels.com/videos/search"
        self.headers = {"Authorization": pexels_api_key}

    def search_videos(self, query: str, per_page: int = 1) -> List[str]:
        """
        Searches for videos on Pexels based on a query.
        Returns a list of high-quality video download URLs.

        Args:
            query: Search query for video content
            per_page: Number of videos to return (default: 1)

        Returns:
            List of video download URLs
        """
        params = {"query": query, "per_page": per_page, "orientation": "landscape"}

        try:
            response = requests.get(
                self.search_url, headers=self.headers, params=params
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            video_urls = []

            for video in data.get("videos", []):
                # Find the highest quality video link
                video_files = video.get("video_files", [])
                if video_files:
                    best_link = max(video_files, key=lambda x: x.get("width", 0))
                    if best_link:
                        video_urls.append(best_link["link"])

            return video_urls

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Pexels API: {e}")
            return []

    def download_video(self, video_url: str, save_folder: str) -> Optional[str]:
        """
        Downloads a video from a URL and saves it to a specified folder.
        Returns the full path to the saved file.

        Args:
            video_url: URL of the video to download
            save_folder: Folder path where to save the video

        Returns:
            Full path to the saved video file, or None if download failed
        """
        try:
            video_res = requests.get(video_url, stream=True)
            video_res.raise_for_status()

            # Generate a unique filename
            save_path = os.path.join(save_folder, f"{uuid.uuid4()}.mp4")

            with open(save_path, "wb") as f:
                for chunk in video_res.iter_content(chunk_size=8192):
                    f.write(chunk)

            return save_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading video from {video_url}: {e}")
            return None

    def get_video_info(self, query: str) -> List[dict]:
        """
        Gets detailed information about videos matching a query.

        Args:
            query: Search query for video content

        Returns:
            List of video information dictionaries
        """
        params = {"query": query, "per_page": 5, "orientation": "landscape"}

        try:
            response = requests.get(
                self.search_url, headers=self.headers, params=params
            )
            response.raise_for_status()

            data = response.json()
            videos_info = []

            for video in data.get("videos", []):
                video_info = {
                    "id": video.get("id"),
                    "width": video.get("width"),
                    "height": video.get("height"),
                    "duration": video.get("duration"),
                    "url": video.get("url"),
                    "image": video.get("image"),
                    "user": video.get("user", {}).get("name", "Unknown"),
                    "video_files": [],
                }

                # Get available video file qualities
                for video_file in video.get("video_files", []):
                    file_info = {
                        "quality": video_file.get("quality", "unknown"),
                        "width": video_file.get("width"),
                        "height": video_file.get("height"),
                        "file_type": video_file.get("file_type"),
                        "link": video_file.get("link"),
                    }
                    video_info["video_files"].append(file_info)

                videos_info.append(video_info)

            return videos_info

        except requests.exceptions.RequestException as e:
            print(f"Error getting video info from Pexels API: {e}")
            return []

    def cleanup_downloads(self, folder_path: str) -> None:
        """
        Removes all downloaded video files from a folder.

        Args:
            folder_path: Path to the folder containing downloaded videos
        """
        try:
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".mp4"):
                        file_path = os.path.join(folder_path, filename)
                        os.remove(file_path)
                        print(f"Removed: {file_path}")
                print(f"Cleanup complete for folder: {folder_path}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

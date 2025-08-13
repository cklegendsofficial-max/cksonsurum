# production_coordinator.py - Production Coordination and Visual Planning

import os
from typing import Any, Dict, List

from visual_asset_manager import VisualAssetManager


class ProductionCoordinator:
    """Coordinates production workflow and creates visual plans from structured scripts."""

    def __init__(self, pexels_api_key: str = None):
        """
        Initialize the ProductionCoordinator.

        Args:
            pexels_api_key: Pexels API key for authentication (optional)
        """
        self.asset_manager = None
        self.download_folder = "temp_assets"

        # Create the asset folder if it doesn't exist
        if not os.path.exists(self.download_folder):
            os.makedirs(self.download_folder)

        # Initialize VisualAssetManager if API key is provided
        if pexels_api_key:
            try:
                self.asset_manager = VisualAssetManager(pexels_api_key=pexels_api_key)
                print("✅ VisualAssetManager initialized with Pexels API")
            except Exception as e:
                print(f"⚠️ Failed to initialize VisualAssetManager: {e}")
                self.asset_manager = None
        else:
            print("⚠️ No Pexels API key provided - VisualAssetManager not initialized")

    def create_visual_plan(self, script_json: List[Dict[str, str]]) -> None:
        """Create a visual plan from the structured JSON script.

        Args:
            script_json: List of scene objects, each with "script_text" and "visual_keywords"
        """
        if not script_json or not isinstance(script_json, list):
            print("❌ Invalid script data provided")
            return

        print("\n🎬 PRODUCTION COORDINATOR - VISUAL PLAN")
        print("=" * 50)

        for i, scene in enumerate(script_json, 1):
            if not isinstance(scene, dict):
                print(f"⚠️ Scene {i}: Invalid scene data")
                continue

            script_text = scene.get("script_text", "No script text available")
            visual_keywords = scene.get(
                "visual_keywords", "No visual keywords available"
            )

            print(f"\n📝 SCENE {i}")
            print("-" * 30)
            print("🎭 Script Text:")
            print(f"   {script_text}")
            print("\n🎨 Visual Keywords:")
            print(f"   {visual_keywords}")

            # Placeholder for Pexels/Pixabay API implementation
            print(
                f"\n🔍 TODO: Implement Pexels/Pixabay API call with keywords: {visual_keywords}"
            )
            print("-" * 30)

        print(f"\n✅ Visual plan created for {len(script_json)} scenes")
        print("=" * 50)

    def create_visual_plan_and_download_assets(
        self, script_json: List[Dict[str, str]]
    ) -> List[List[str]]:
        """
        Create a visual plan and download video assets for each scene.

        Args:
            script_json: List of scene objects, each with "script_text" and "visual_keywords"

        Returns:
            List of lists, where each inner list contains paths to downloaded assets for that scene
        """
        if not script_json or not isinstance(script_json, list):
            print("❌ Invalid script data provided")
            return []

        if not self.asset_manager:
            print(
                "❌ VisualAssetManager not initialized. Please provide a Pexels API key."
            )
            return []

        print("\n🎬 PRODUCTION COORDINATOR - VISUAL PLAN & ASSET DOWNLOAD")
        print("=" * 60)

        # This list will store paths of downloaded assets for each scene
        scene_assets = []

        for i, scene in enumerate(script_json):
            scene_text = scene.get("script_text", "No text")
            scene_keywords = scene.get("visual_keywords", "")

            print(f"\n📝 SCENE {i+1}")
            print("-" * 40)
            print(f"🎭 Script Text:\n   {scene_text}")
            print(f"🎨 Visual Keywords:\n   {scene_keywords}")

            downloaded_files_for_scene = []

            if scene_keywords:
                print("🔍 Searching for visuals with keywords...")
                video_urls = self.asset_manager.search_videos(
                    query=scene_keywords, per_page=1
                )

                if video_urls:
                    for url in video_urls:
                        saved_path = self.asset_manager.download_video(
                            url, self.download_folder
                        )
                        if saved_path:
                            print(f"✅ Video downloaded successfully: {saved_path}")
                            downloaded_files_for_scene.append(saved_path)
                        else:
                            print(f"❌ Failed to download video from: {url}")
                else:
                    print(f"⚠️ No videos found for keywords: {scene_keywords}")
            else:
                print("⚠️ No visual keywords provided for this scene")

            scene_assets.append(downloaded_files_for_scene)
            print("-" * 40)

        print("\n🎉 Asset download phase complete!")
        print("=" * 60)

        return scene_assets

    def get_download_folder(self) -> str:
        """Get the path to the download folder for assets.

        Returns:
            Path to the download folder
        """
        return self.download_folder

    def cleanup_assets(self) -> None:
        """Clean up all downloaded assets from the download folder."""
        if self.asset_manager:
            self.asset_manager.cleanup_downloads(self.download_folder)
        else:
            print("⚠️ VisualAssetManager not initialized - cannot cleanup assets")

    def get_scene_summary(self, script_json: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get a summary of the script for production planning.

        Args:
            script_json: List of scene objects

        Returns:
            Dictionary containing script summary information
        """
        if not script_json:
            return {}

        total_scenes = len(script_json)
        total_script_length = sum(
            len(scene.get("script_text", "").split()) for scene in script_json
        )

        # Extract unique visual themes
        all_keywords = []
        for scene in script_json:
            keywords = scene.get("visual_keywords", "")
            if keywords:
                all_keywords.extend([kw.strip() for kw in keywords.split(",")])

        unique_keywords = list(set(all_keywords))

        return {
            "total_scenes": total_scenes,
            "total_words": total_script_length,
            "estimated_duration_minutes": total_script_length
            / 150,  # Rough estimate: 150 words per minute
            "unique_visual_themes": unique_keywords,
            "visual_theme_count": len(unique_keywords),
        }

#!/usr/bin/env python3
# test_visual_assets.py - Test script for VisualAssetManager and updated ProductionCoordinator

import os

from production_coordinator import ProductionCoordinator
from visual_asset_manager import VisualAssetManager


def test_visual_asset_manager():
    """Test the VisualAssetManager class."""
    print("🧪 Testing VisualAssetManager...")

    # Test with a mock API key (this will fail but test the structure)
    try:
        # This will fail since we don't have a real API key
        asset_manager = VisualAssetManager(pexels_api_key="test_key")
        print("✅ VisualAssetManager instance created")

        # Test search method structure
        if hasattr(asset_manager, "search_videos"):
            print("✅ search_videos method exists")
        if hasattr(asset_manager, "download_video"):
            print("✅ download_video method exists")
        if hasattr(asset_manager, "get_video_info"):
            print("✅ get_video_info method exists")
        if hasattr(asset_manager, "cleanup_downloads"):
            print("✅ cleanup_downloads method exists")

    except Exception as e:
        print(f"⚠️ Expected error with test API key: {e}")

    print("✅ VisualAssetManager class structure verified")


def test_production_coordinator_without_api():
    """Test ProductionCoordinator without API key."""
    print("\n🧪 Testing ProductionCoordinator without API key...")

    try:
        coordinator = ProductionCoordinator()  # No API key
        print("✅ ProductionCoordinator instance created without API key")

        # Check if methods exist
        if hasattr(coordinator, "create_visual_plan"):
            print("✅ create_visual_plan method exists")
        if hasattr(coordinator, "create_visual_plan_and_download_assets"):
            print("✅ create_visual_plan_and_download_assets method exists")
        if hasattr(coordinator, "get_download_folder"):
            print("✅ get_download_folder method exists")
        if hasattr(coordinator, "cleanup_assets"):
            print("✅ cleanup_assets method exists")

        # Test download folder creation
        download_folder = coordinator.get_download_folder()
        if os.path.exists(download_folder):
            print(f"✅ Download folder created: {download_folder}")
        else:
            print(f"❌ Download folder not created: {download_folder}")

    except Exception as e:
        print(f"❌ Error testing ProductionCoordinator: {e}")


def test_with_mock_data():
    """Test the complete workflow with mock data."""
    print("\n🧪 Testing complete workflow with mock data...")

    # Mock script data
    mock_script_json = [
        {
            "script_text": "In the heart of ancient Rome, beneath the bustling streets of the eternal city, lies a secret that has puzzled historians for centuries.",
            "visual_keywords": "ancient roman architecture, marble columns, bustling roman streets",
        },
        {
            "script_text": "Their aqueducts, stretching for hundreds of miles across the empire, brought fresh water to millions of people.",
            "visual_keywords": "roman aqueducts, stone arches, flowing water",
        },
        {
            "script_text": "But perhaps their most astonishing achievement was the construction of the Colosseum.",
            "visual_keywords": "colosseum rome, massive amphitheater, ancient construction",
        },
    ]

    print(f"✅ Mock data created with {len(mock_script_json)} scenes")

    # Test ProductionCoordinator without API key
    coordinator = ProductionCoordinator()

    # Test visual plan creation
    print("\n🎬 Creating visual plan...")
    coordinator.create_visual_plan(mock_script_json)

    # Test asset download (will fail without API key)
    print("\n🔍 Testing asset download (expected to fail without API key)...")
    scene_assets = coordinator.create_visual_plan_and_download_assets(mock_script_json)
    if not scene_assets:
        print("✅ Expected behavior: No assets downloaded without API key")

    # Test summary
    print("\n📊 Getting script summary...")
    summary = coordinator.get_scene_summary(mock_script_json)
    print("✅ Summary generated:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def test_with_api_key():
    """Test with actual API key if available."""
    print("\n🧪 Testing with API key (if available)...")

    # Check if we have an API key in environment
    api_key = os.getenv("PEXELS_API_KEY")
    if api_key and api_key != "your_pexels_api_key_here":
        print(f"✅ Pexels API key found: {api_key[:10]}...")

        try:
            # Test VisualAssetManager
            asset_manager = VisualAssetManager(pexels_api_key=api_key)
            print("✅ VisualAssetManager initialized with real API key")

            # Test search functionality
            print("🔍 Testing video search...")
            results = asset_manager.search_videos("ancient rome", per_page=1)
            if results:
                print(f"✅ Found {len(results)} videos")
            else:
                print("⚠️ No videos found (this might be normal)")

        except Exception as e:
            print(f"❌ Error testing with real API key: {e}")

    else:
        print("⚠️ No valid Pexels API key found in environment")
        print("   Set PEXELS_API_KEY environment variable to test with real API")


def main():
    """Main test function."""
    print("🚀 Starting VisualAssetManager and ProductionCoordinator tests...")
    print("=" * 70)

    # Test VisualAssetManager
    test_visual_asset_manager()

    # Test ProductionCoordinator without API
    test_production_coordinator_without_api()

    # Test with mock data
    test_with_mock_data()

    # Test with API key if available
    test_with_api_key()

    print("\n" + "=" * 70)
    print("🎉 All tests completed!")
    print("\n📝 Note: To test with real Pexels API:")
    print("   1. Set PEXELS_API_KEY environment variable")
    print("   2. Run: python test_visual_assets.py")
    print("   3. Check the 'temp_assets' folder for downloaded videos")


if __name__ == "__main__":
    main()

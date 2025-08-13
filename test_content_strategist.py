#!/usr/bin/env python3
# test_content_strategist.py - Test script for ContentStrategist and ProductionCoordinator

from content_strategist import ContentStrategist
from production_coordinator import ProductionCoordinator


def test_content_strategist():
    """Test the ContentStrategist class."""
    print("🧪 Testing ContentStrategist...")

    # Create instance
    strategist = ContentStrategist()
    print("✅ ContentStrategist instance created")

    # Test data
    video_idea = {"title": "The Mystery of Ancient Roman Engineering"}
    channel_name = "test_channel"

    # Generate script
    print("\n📝 Generating script...")
    script_json = strategist.write_script(video_idea, channel_name)

    if script_json:
        print(f"✅ Script generated successfully with {len(script_json)} scenes")
        print("\n📋 Script Preview:")
        for i, scene in enumerate(script_json[:3], 1):  # Show first 3 scenes
            print(f"\nScene {i}:")
            print(f"  Script: {scene.get('script_text', 'N/A')[:100]}...")
            print(f"  Visual Keywords: {scene.get('visual_keywords', 'N/A')}")
        return script_json
    else:
        print("❌ Script generation failed")
        return None


def test_production_coordinator(script_json):
    """Test the ProductionCoordinator class."""
    if not script_json:
        print("⚠️ Skipping ProductionCoordinator test - no script data")
        return

    print("\n🧪 Testing ProductionCoordinator...")

    # Create instance
    coordinator = ProductionCoordinator()
    print("✅ ProductionCoordinator instance created")

    # Create visual plan
    print("\n🎬 Creating visual plan...")
    coordinator.create_visual_plan(script_json)

    # Get summary
    print("\n📊 Getting script summary...")
    summary = coordinator.get_scene_summary(script_json)
    print("✅ Summary generated:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main():
    """Main test function."""
    print("🚀 Starting ContentStrategist and ProductionCoordinator tests...")
    print("=" * 60)

    # Test ContentStrategist
    script_json = test_content_strategist()

    # Test ProductionCoordinator
    test_production_coordinator(script_json)

    print("\n" + "=" * 60)
    print("🎉 Tests completed!")


if __name__ == "__main__":
    main()

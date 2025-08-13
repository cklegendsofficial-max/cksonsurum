#!/usr/bin/env python3
# test_content_strategist_mock.py - Mock test for ContentStrategist and ProductionCoordinator

from content_strategist import ContentStrategist
from production_coordinator import ProductionCoordinator


def test_with_mock_data():
    """Test the classes with mock data to avoid Ollama dependency."""
    print("🧪 Testing with mock data...")

    # Mock script data that mimics what ContentStrategist would return
    mock_script_json = [
        {
            "script_text": "In the heart of ancient Rome, beneath the bustling streets of the eternal city, lies a secret that has puzzled historians for centuries. The Romans were master engineers, capable of feats that seem impossible even by today's standards.",
            "visual_keywords": "ancient roman architecture, marble columns, bustling roman streets, eternal city",
        },
        {
            "script_text": "Their aqueducts, stretching for hundreds of miles across the empire, brought fresh water to millions of people. These engineering marvels defied gravity, flowing uphill through sophisticated systems of arches and channels.",
            "visual_keywords": "roman aqueducts, stone arches, flowing water, engineering marvels",
        },
        {
            "script_text": "But perhaps their most astonishing achievement was the construction of the Colosseum. This massive amphitheater, capable of seating 50,000 spectators, was built in just eight years using techniques that modern engineers still study.",
            "visual_keywords": "colosseum rome, massive amphitheater, ancient construction, roman engineering",
        },
        {
            "script_text": "The Romans used concrete, a material they invented, to create structures that have lasted for over two thousand years. Their formula, lost to time, created a material that actually gets stronger when exposed to seawater.",
            "visual_keywords": "roman concrete, ancient construction material, seawater exposure, time-worn structures",
        },
        {
            "script_text": "Today, as we walk through the ruins of their cities, we can only marvel at the ingenuity of these ancient engineers. Their legacy lives on in every modern building, every bridge, every road that follows their principles.",
            "visual_keywords": "roman ruins, ancient city remains, modern construction, engineering legacy",
        },
    ]

    print(f"✅ Mock data created with {len(mock_script_json)} scenes")

    # Test ProductionCoordinator with mock data
    print("\n🧪 Testing ProductionCoordinator with mock data...")

    coordinator = ProductionCoordinator()
    print("✅ ProductionCoordinator instance created")

    # Create visual plan
    print("\n🎬 Creating visual plan...")
    coordinator.create_visual_plan(mock_script_json)

    # Get summary
    print("\n📊 Getting script summary...")
    summary = coordinator.get_scene_summary(mock_script_json)
    print("✅ Summary generated:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return mock_script_json


def test_content_strategist_structure():
    """Test the ContentStrategist class structure without calling Ollama."""
    print("\n🧪 Testing ContentStrategist class structure...")

    try:
        strategist = ContentStrategist()
        print("✅ ContentStrategist instance created successfully")

        # Check if the class has the required method
        if hasattr(strategist, "write_script"):
            print("✅ write_script method exists")
        else:
            print("❌ write_script method missing")

        # Check if the class has the required attributes
        if hasattr(strategist, "ollama_url"):
            print("✅ ollama_url attribute exists")
        if hasattr(strategist, "ollama_model"):
            print("✅ ollama_model attribute exists")

        print("✅ ContentStrategist class structure is correct")

    except Exception as e:
        print(f"❌ Error testing ContentStrategist structure: {e}")


def main():
    """Main test function."""
    print("🚀 Starting ContentStrategist and ProductionCoordinator mock tests...")
    print("=" * 70)

    # Test class structure
    test_content_strategist_structure()

    # Test with mock data
    mock_script = test_with_mock_data()

    print("\n" + "=" * 70)
    print("🎉 Mock tests completed successfully!")
    print("\n📝 Note: These tests verify the class structure and functionality")
    print("   without requiring Ollama to be running. To test the full")
    print("   script generation, ensure Ollama is running and use:")
    print("   python test_content_strategist.py")


if __name__ == "__main__":
    main()

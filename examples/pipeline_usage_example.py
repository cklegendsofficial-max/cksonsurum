#!/usr/bin/env python3
"""
Pipeline Usage Example - Niche Resolution Helpers

This example demonstrates how to use the niche_from_channel helper function
and the get_topics_by_channel method for automatic niche resolution.
"""

from improved_llm_handler import ImprovedLLMHandler, niche_from_channel
from config import TIER1_GEOS, DEFAULT_TIMEFRAMES

def main():
    """Demonstrate pipeline usage with automatic niche resolution."""
    
    # Initialize the handler
    handler = ImprovedLLMHandler()
    
    # Example 1: Using niche_from_channel helper directly
    print("=== Example 1: Direct helper usage ===")
    channel_name = "CKDrive"
    niche = niche_from_channel(channel_name)
    print(f"Channel: {channel_name} -> Niche: {niche}")
    
    # Get topics using the resolved niche
    topics = handler.get_topics_resilient(niche, timeframe="today 1-m", geo="US")
    print(f"Found {len(topics)} topics for {niche}")
    if topics:
        print(f"Sample topics: {topics[:3]}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Using the convenience method get_topics_by_channel
    print("=== Example 2: Convenience method usage ===")
    channel_name = "cklegends"
    topics = handler.get_topics_by_channel(channel_name, geo="GB")
    print(f"Channel: {channel_name} -> Automatically resolved to niche")
    print(f"Found {len(topics)} topics")
    if topics:
        print(f"Sample topics: {topics[:3]}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Batch processing multiple channels
    print("=== Example 3: Batch processing ===")
    channels = ["CKIronWill", "CKFinanceCore", "CKCombat"]
    
    for channel in channels:
        niche = niche_from_channel(channel)
        topics = handler.get_topics_by_channel(channel, timeframe="now 7-d")
        print(f"{channel} -> {niche}: {len(topics)} topics")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Pipeline integration pattern
    print("=== Example 4: Pipeline integration pattern ===")
    
    def process_channel_pipeline(channel_name: str, timeframe: str = None, geo: str = None):
        """
        Example pipeline function showing the recommended usage pattern.
        
        Args:
            channel_name: Channel name (e.g., "CKDrive", "cklegends")
            timeframe: Optional timeframe override
            geo: Optional geo override
        """
        # Automatic niche resolution
        niche = niche_from_channel(channel_name)
        
        # Get topics with resilient fallback
        topics = handler.get_topics_resilient(niche, timeframe=timeframe, geo=geo)
        
        # Use topics in your pipeline logic
        print(f"Pipeline processing {channel_name} ({niche}): {len(topics)} topics available")
        
        # Example: Generate viral ideas using the topics
        if topics:
            # Your pipeline logic here...
            print(f"  - Top 3 topics: {topics[:3]}")
            print(f"  - Ready for content generation")
        else:
            print(f"  - No topics available, using fallback")
        
        return topics
    
    # Test the pipeline function
    process_channel_pipeline("CKDrive", geo="US")
    process_channel_pipeline("cklegends", timeframe="today 3-m")

if __name__ == "__main__":
    print("🚀 Pipeline Usage Examples - Niche Resolution Helpers")
    print("=" * 60)
    main()
    print("\n✅ All examples completed successfully!")

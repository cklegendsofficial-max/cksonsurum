# pytrends_offline.py - Offline PyTrends Simulation Module

from datetime import datetime, timedelta
import json
import os
import random
from typing import Any, Dict, List


class PyTrendsOffline:
    """Offline PyTrends simulation with local trend caching"""

    def __init__(self, cache_file: str = "trends_cache.json"):
        self.cache_file = cache_file
        self.trends_cache = {}
        self.load_cache()

        # Simulated trending topics for different niches
        self.niche_trends = {
            "history": [
                "ancient civilizations",
                "archaeological discoveries",
                "historical mysteries",
                "lost cities",
                "ancient technology",
                "historical figures",
                "battle strategies",
                "ancient religions",
                "historical artifacts",
                "ancient architecture",
            ],
            "motivation": [
                "personal development",
                "success mindset",
                "goal achievement",
                "self-discipline",
                "mental strength",
                "career growth",
                "life transformation",
                "positive thinking",
                "habits formation",
                "mindset shift",
            ],
            "finance": [
                "investment strategies",
                "wealth building",
                "financial freedom",
                "passive income",
                "cryptocurrency",
                "stock market",
                "real estate",
                "business growth",
                "financial planning",
                "economic trends",
            ],
            "automotive": [
                "electric vehicles",
                "autonomous driving",
                "car technology",
                "performance cars",
                "classic cars",
                "car maintenance",
                "racing",
                "car reviews",
                "automotive industry",
                "future of transportation",
            ],
            "combat_sports": [
                "martial arts",
                "boxing techniques",
                "MMA strategies",
                "self-defense",
                "fighting skills",
                "combat training",
                "warrior mindset",
                "fight analysis",
                "combat history",
                "martial arts philosophy",
            ],
        }

    def load_cache(self):
        """Load cached trends from JSON file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, encoding="utf-8") as f:
                    self.trends_cache = json.load(f)
                print(f"✅ Loaded {len(self.trends_cache)} cached trends")
            else:
                self.trends_cache = {}
                print("📁 No trends cache found, starting fresh")
        except Exception as e:
            print(f"⚠️ Cache loading failed: {e}")
            self.trends_cache = {}

    def save_cache(self):
        """Save current trends cache to JSON file"""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.trends_cache, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved {len(self.trends_cache)} trends to cache")
        except Exception as e:
            print(f"❌ Cache saving failed: {e}")

    def get_trending_topics(
        self, niche: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Get trending topics for a specific niche (offline simulation)"""
        try:
            # Check if we have recent cached data for this niche
            cache_key = f"{niche}_{datetime.now().strftime('%Y%m%d')}"

            if cache_key in self.trends_cache:
                print(f"📊 Using cached trends for {niche}")
                return self.trends_cache[cache_key]

            # Generate simulated trending topics
            print(f"🔍 Generating offline trends for {niche}...")

            # Get base topics for the niche
            base_topics = self.niche_trends.get(
                niche, ["general topic", "trending content"]
            )

            # Generate trending variations
            trending_topics = []
            for i in range(max_results):
                base_topic = random.choice(base_topics)

                # Add trending modifiers
                modifiers = [
                    "latest",
                    "viral",
                    "breaking",
                    "shocking",
                    "amazing",
                    "incredible",
                    "unbelievable",
                    "controversial",
                    "revolutionary",
                    "game-changing",
                ]

                modifier = random.choice(modifiers)
                topic = f"{modifier} {base_topic}"

                # Generate trend data
                trend_data = {
                    "topic": topic,
                    "niche": niche,
                    "trend_score": random.uniform(0.5, 1.0),
                    "growth_rate": random.uniform(0.1, 0.8),
                    "search_volume": random.randint(1000, 100000),
                    "timestamp": datetime.now().isoformat(),
                    "source": "offline_simulation",
                }

                trending_topics.append(trend_data)

            # Cache the results
            self.trends_cache[cache_key] = trending_topics
            self.save_cache()

            print(f"✅ Generated {len(trending_topics)} offline trends for {niche}")
            return trending_topics

        except Exception as e:
            print(f"❌ Offline trend generation failed: {e}")
            return []

    def get_related_topics(self, main_topic: str, niche: str) -> List[str]:
        """Get related topics for a main topic"""
        try:
            base_topics = self.niche_trends.get(niche, [])

            # Generate related topics
            related = []
            for base in base_topics[:5]:  # Top 5 related
                if base != main_topic:
                    related.append(base)

            # Add some generic related terms
            generic_terms = ["trending", "viral", "latest", "breaking", "news"]
            for term in generic_terms:
                if len(related) < 8:  # Max 8 related topics
                    related.append(f"{main_topic} {term}")

            return related[:8]

        except Exception as e:
            print(f"❌ Related topics generation failed: {e}")
            return []

    def get_trending_queries(self, niche: str, max_results: int = 15) -> List[str]:
        """Get trending search queries for a niche"""
        try:
            topics = self.get_trending_topics(niche, max_results)

            # Convert topics to search queries
            queries = []
            for topic in topics:
                query = topic["topic"].replace(" ", " ")
                queries.append(query)

            return queries

        except Exception as e:
            print(f"❌ Trending queries generation failed: {e}")
            return []

    def get_niche_trends_summary(self, niche: str) -> Dict[str, Any]:
        """Get a summary of trends for a specific niche"""
        try:
            topics = self.get_trending_topics(niche, 10)

            if not topics:
                return {"niche": niche, "status": "no_data"}

            # Calculate summary statistics
            avg_trend_score = sum(t["trend_score"] for t in topics) / len(topics)
            avg_growth_rate = sum(t["growth_rate"] for t in topics) / len(topics)
            total_search_volume = sum(t["search_volume"] for t in topics)

            # Get top trending topic
            top_topic = max(topics, key=lambda x: x["trend_score"])

            summary = {
                "niche": niche,
                "total_topics": len(topics),
                "average_trend_score": round(avg_trend_score, 3),
                "average_growth_rate": round(avg_growth_rate, 3),
                "total_search_volume": total_search_volume,
                "top_trending_topic": top_topic["topic"],
                "top_trend_score": top_topic["trend_score"],
                "timestamp": datetime.now().isoformat(),
                "source": "offline_simulation",
            }

            return summary

        except Exception as e:
            print(f"❌ Niche trends summary failed: {e}")
            return {"niche": niche, "status": "error", "message": str(e)}

    def simulate_trend_evolution(
        self, topic: str, days: int = 7
    ) -> List[Dict[str, Any]]:
        """Simulate how a trend evolves over time"""
        try:
            evolution = []
            base_score = random.uniform(0.3, 0.7)

            for day in range(days):
                # Simulate trend evolution with some randomness
                day_score = base_score + random.uniform(-0.2, 0.3)
                day_score = max(0.0, min(1.0, day_score))  # Clamp between 0 and 1

                evolution_data = {
                    "day": day + 1,
                    "date": (datetime.now() - timedelta(days=days - day - 1)).strftime(
                        "%Y-%m-%d"
                    ),
                    "trend_score": round(day_score, 3),
                    "search_volume": int(
                        base_score * 50000 + random.uniform(-10000, 20000)
                    ),
                    "status": (
                        "rising"
                        if day_score > base_score
                        else "declining"
                        if day_score < base_score
                        else "stable"
                    ),
                }

                evolution.append(evolution_data)

            return evolution

        except Exception as e:
            print(f"❌ Trend evolution simulation failed: {e}")
            return []

    def get_cross_niche_trends(
        self, niches: List[str], max_per_niche: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get trends across multiple niches"""
        try:
            cross_niche_data = {}

            for niche in niches:
                if niche in self.niche_trends:
                    topics = self.get_trending_topics(niche, max_per_niche)
                    cross_niche_data[niche] = topics
                else:
                    cross_niche_data[niche] = []

            return cross_niche_data

        except Exception as e:
            print(f"❌ Cross-niche trends failed: {e}")
            return {}

    def clear_cache(self):
        """Clear the trends cache"""
        try:
            self.trends_cache = {}
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            print("🗑️ Trends cache cleared")
        except Exception as e:
            print(f"❌ Cache clearing failed: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache"""
        try:
            total_entries = len(self.trends_cache)
            niches = list(set(key.split("_")[0] for key in self.trends_cache.keys()))

            stats = {
                "total_cached_entries": total_entries,
                "cached_niches": niches,
                "cache_file_size": (
                    os.path.getsize(self.cache_file)
                    if os.path.exists(self.cache_file)
                    else 0
                ),
                "last_updated": datetime.now().isoformat(),
            }

            return stats

        except Exception as e:
            print(f"❌ Cache stats failed: {e}")
            return {"error": str(e)}


# Convenience functions for backward compatibility
def get_trending_topics(niche: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Backward compatibility function"""
    offline_trends = PyTrendsOffline()
    return offline_trends.get_trending_topics(niche, max_results)


def get_related_topics(main_topic: str, niche: str) -> List[str]:
    """Backward compatibility function"""
    offline_trends = PyTrendsOffline()
    return offline_trends.get_related_topics(main_topic, niche)


# Test the module
if __name__ == "__main__":
    print("🧪 Testing PyTrends Offline Module...")

    # Create offline trends instance
    offline_trends = PyTrendsOffline()

    # Test different niches
    test_niches = ["history", "motivation", "finance"]

    for niche in test_niches:
        print(f"\n📊 Testing {niche} niche...")

        # Get trending topics
        topics = offline_trends.get_trending_topics(niche, 5)
        print(f"✅ Generated {len(topics)} trending topics")

        # Get summary
        summary = offline_trends.get_niche_trends_summary(niche)
        print(
            f"📈 Summary: {summary['total_topics']} topics, avg score: {summary['average_trend_score']}"
        )

        # Get related topics for first topic
        if topics:
            related = offline_trends.get_related_topics(topics[0]["topic"], niche)
            print(f"🔗 Related topics: {len(related)} found")

    # Test trend evolution
    print("\n📈 Testing trend evolution...")
    evolution = offline_trends.simulate_trend_evolution("ancient civilizations", 7)
    print(f"✅ Generated {len(evolution)} days of trend evolution")

    # Get cache stats
    stats = offline_trends.get_cache_stats()
    print(
        f"📊 Cache stats: {stats['total_cached_entries']} entries, {len(stats['cached_niches'])} niches"
    )

    print("\n🎉 PyTrends Offline Module test completed!")

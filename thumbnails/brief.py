#!/usr/bin/env python3
"""
Thumbnail Brief Generator
Creates one-sentence art briefs for video thumbnails
"""

import logging
import os
from typing import Any, Dict, Optional


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("thumbnail_brief")


def generate_thumbnail_brief(
    video_title: str,
    channel_niche: str,
    script_structure: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a one-sentence art brief for video thumbnail.

    Args:
        video_title: Title of the video
        channel_niche: Niche of the channel (e.g., "finance", "gaming")
        script_structure: Optional script structure with hook/promise/proof/preview

    Returns:
        One-sentence art brief for thumbnail creation
    """
    try:
        # Extract key elements from script structure if available
        hook_content = ""
        if script_structure and "hook" in script_structure:
            hook_content = script_structure["hook"].get("content", "")

        # Create thumbnail brief based on niche and content
        if channel_niche.lower() in ["finance", "business", "economics"]:
            if "crisis" in video_title.lower() or "crash" in video_title.lower():
                return f"Create a dramatic thumbnail showing financial crisis with red charts, falling graphs, and urgent typography for '{video_title}'"
            elif "success" in video_title.lower() or "profit" in video_title.lower():
                return f"Design an inspiring thumbnail with green charts, upward trends, and motivational elements for '{video_title}'"
            else:
                return f"Create a professional finance thumbnail with charts, graphs, and business imagery for '{video_title}'"

        elif channel_niche.lower() in ["gaming", "esports"]:
            if "win" in video_title.lower() or "victory" in video_title.lower():
                return f"Design an epic gaming thumbnail with victory effects, glowing elements, and triumphant imagery for '{video_title}'"
            elif "fail" in video_title.lower() or "lose" in video_title.lower():
                return f"Create a humorous gaming thumbnail with fail reactions, funny faces, and comedic elements for '{video_title}'"
            else:
                return f"Design a dynamic gaming thumbnail with action scenes, vibrant colors, and gaming elements for '{video_title}'"

        elif channel_niche.lower() in ["tech", "technology", "ai"]:
            if (
                "ai" in video_title.lower()
                or "artificial intelligence" in video_title.lower()
            ):
                return f"Create a futuristic AI thumbnail with neural networks, glowing circuits, and sci-fi elements for '{video_title}'"
            elif "hack" in video_title.lower() or "security" in video_title.lower():
                return f"Design a cybersecurity thumbnail with code, lock symbols, and mysterious elements for '{video_title}'"
            else:
                return f"Create a modern tech thumbnail with sleek design, digital elements, and innovation imagery for '{video_title}'"

        elif channel_niche.lower() in ["history", "documentary"]:
            if (
                "ancient" in video_title.lower()
                or "civilization" in video_title.lower()
            ):
                return f"Design a historical thumbnail with ancient ruins, mysterious artifacts, and archaeological elements for '{video_title}'"
            elif "war" in video_title.lower() or "battle" in video_title.lower():
                return f"Create an epic historical thumbnail with dramatic battle scenes, period costumes, and intense imagery for '{video_title}'"
            else:
                return f"Create a documentary-style thumbnail with historical imagery, vintage elements, and storytelling visuals for '{video_title}'"

        elif channel_niche.lower() in ["science", "education"]:
            if "space" in video_title.lower() or "universe" in video_title.lower():
                return f"Design a cosmic thumbnail with galaxies, planets, and space exploration imagery for '{video_title}'"
            elif (
                "experiment" in video_title.lower()
                or "discovery" in video_title.lower()
            ):
                return f"Create a scientific thumbnail with lab equipment, formulas, and discovery elements for '{video_title}'"
            else:
                return f"Create an educational thumbnail with clear visuals, diagrams, and learning elements for '{video_title}'"

        else:
            # Generic thumbnail brief
            if hook_content:
                # Use hook content to create more specific brief
                hook_words = hook_content.split()[:5]  # First 5 words
                hook_text = " ".join(hook_words)
                return f"Design an engaging thumbnail that captures the essence of '{hook_text}' for '{video_title}' with vibrant colors and compelling imagery"
            else:
                return f"Create an eye-catching thumbnail with bold typography, vibrant colors, and engaging imagery for '{video_title}'"

    except Exception as e:
        logger.error(f"Error generating thumbnail brief: {e}")
        return f"Create an engaging thumbnail with bold design and compelling visuals for '{video_title}'"


def generate_thumbnail_brief_from_script(script_data: Dict[str, Any]) -> str:
    """
    Generate thumbnail brief from complete script data.

    Args:
        script_data: Complete script data dictionary

    Returns:
        One-sentence art brief for thumbnail creation
    """
    try:
        video_title = script_data.get("video_title", "Unknown Video")

        # Extract channel niche from metadata if available
        channel_niche = "general"
        if "metadata" in script_data:
            metadata = script_data["metadata"]
            if "visual_prevention" in metadata:
                # Extract niche from visual_prevention field
                visual_prevention = metadata["visual_prevention"]
                if "niche" in visual_prevention:
                    start = visual_prevention.find("'") + 1
                    end = visual_prevention.rfind("'")
                    if start > 0 and end > start:
                        channel_niche = visual_prevention[start:end]

        # Get script structure
        script_structure = script_data.get("script_structure", {})

        return generate_thumbnail_brief(video_title, channel_niche, script_structure)

    except Exception as e:
        logger.error(f"Error generating thumbnail brief from script: {e}")
        return "Create an engaging thumbnail with bold design and compelling visuals for the video"


def save_thumbnail_brief(brief: str, output_dir: str, video_title: str) -> str:
    """
    Save thumbnail brief to a file.

    Args:
        brief: Thumbnail brief text
        output_dir: Output directory path
        video_title: Video title for filename

    Returns:
        Path to saved brief file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create safe filename
        safe_title = "".join(
            c for c in video_title if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_title = safe_title.replace(" ", "_")

        # Save brief to file
        brief_file = os.path.join(output_dir, f"{safe_title}_thumbnail_brief.txt")

        with open(brief_file, "w", encoding="utf-8") as f:
            f.write(f"THUMBNAIL BRIEF FOR: {video_title}\n")
            f.write("=" * 50 + "\n\n")
            f.write(brief + "\n\n")
            f.write("=" * 50 + "\n")
            f.write("Generated by Enhanced Master Director\n")

        logger.info(f"Thumbnail brief saved to: {brief_file}")
        return brief_file

    except Exception as e:
        logger.error(f"Error saving thumbnail brief: {e}")
        return ""


if __name__ == "__main__":
    # Test the thumbnail brief generator
    test_title = "The Hidden Truth About AI That Will Shock You"
    test_niche = "technology"

    brief = generate_thumbnail_brief(test_title, test_niche)
    print(f"Generated brief: {brief}")

    # Test with script structure
    test_script = {
        "video_title": test_title,
        "script_structure": {
            "hook": {
                "content": "Artificial intelligence is secretly controlling your life right now",
                "visual_query": "AI surveillance camera",
                "timing_seconds": 0,
                "engagement_hook": "Shocking AI revelation",
            }
        },
    }

    brief_from_script = generate_thumbnail_brief_from_script(test_script)
    print(f"Brief from script: {brief_from_script}")

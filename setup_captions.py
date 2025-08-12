#!/usr/bin/env python3
"""
Captions System Setup Script
Automatically installs required dependencies for multi-language captions.
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ffmpeg is already installed")
            return True
    except FileNotFoundError:
        pass

    print("⚠️  ffmpeg not found")
    print("   Windows: Download from https://www.gyan.dev/ffmpeg/builds/")
    print("   macOS: brew install ffmpeg")
    print("   Ubuntu: sudo apt-get install ffmpeg")
    return False


def main():
    """Main setup function."""
    print("🚀 Captions System Setup")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

    # Check ffmpeg
    ffmpeg_ok = check_ffmpeg()

    # Install Python packages
    print("\n📦 Installing Python packages...")

    packages = ["openai-whisper", "torch", "transformers", "sentencepiece"]

    for package in packages:
        if not run_command(f"pip install {package} --upgrade", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")

    # Set environment variable
    print("\n🔧 Setting up environment...")
    try:
        os.environ["WHISPER_MODEL"] = "base"
        print("✅ WHISPER_MODEL set to 'base'")
    except Exception as e:
        print(f"⚠️  Could not set WHISPER_MODEL: {e}")

    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import whisper

        print("✅ Whisper imported successfully")
    except ImportError:
        print("❌ Whisper import failed")
        return False

    try:
        import transformers

        print("✅ Transformers imported successfully")
    except ImportError:
        print("❌ Transformers import failed")
        return False

    # Final status
    print("\n" + "=" * 50)
    if ffmpeg_ok:
        print("🎉 Captions system setup completed successfully!")
        print("   You can now use auto_captions.py")
    else:
        print("⚠️  Setup completed with warnings")
        print("   Please install ffmpeg manually for full functionality")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

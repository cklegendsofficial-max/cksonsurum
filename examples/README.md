# Pipeline Niche Resolution Helpers

This directory contains examples and documentation for using the automatic niche resolution helpers in your video creation pipeline.

## 🎯 Quick Start

### Basic Usage

```python
from improved_llm_handler import ImprovedLLMHandler, niche_from_channel

# Initialize handler
handler = ImprovedLLMHandler()

# Automatic niche resolution
niche = niche_from_channel("CKDrive")  # Returns "automotive"
topics = handler.get_topics_resilient(niche, timeframe="today 1-m", geo="US")
```

### Convenience Method

```python
# Even simpler - automatic niche resolution
topics = handler.get_topics_by_channel("CKDrive", geo="US")
# Automatically resolves "CKDrive" -> "automotive" and gets topics
```

## 🔧 Available Helpers

### 1. `niche_from_channel(channel_name: str) -> str`

**Purpose**: Automatically resolves channel name to niche

**Examples**:
- `"CKDrive"` → `"automotive"`
- `"cklegends"` → `"history"`
- `"CKIronWill"` → `"motivation"`
- `"CKFinanceCore"` → `"finance"`
- `"CKCombat"` → `"combat"`

**Usage**:
```python
# In your pipeline
niche = niche_from_channel(channel_name)
topics = handler.get_topics_resilient(niche, timeframe=timeframe, geo=geo)
```

### 2. `handler.get_topics_by_channel(channel_name, timeframe, geo)`

**Purpose**: One-line method to get topics by channel name

**Parameters**:
- `channel_name`: Channel name (e.g., "CKDrive", "cklegends")
- `timeframe`: Optional timeframe (e.g., "today 1-m", "now 7-d")
- `geo`: Optional geographic location (e.g., "US", "GB", "CA")

**Returns**: List of trending topics for the channel's niche

**Usage**:
```python
# Simple one-liner
topics = handler.get_topics_by_channel("CKDrive", geo="US")
```

## 🚀 Pipeline Integration Patterns

### Pattern 1: Direct Helper Usage

```python
def process_channel_pipeline(channel_name: str, timeframe: str = None, geo: str = None):
    # Automatic niche resolution
    niche = niche_from_channel(channel_name)

    # Get topics with resilient fallback
    topics = handler.get_topics_resilient(niche, timeframe=timeframe, geo=geo)

    # Use topics in your pipeline logic
    return topics
```

### Pattern 2: Convenience Method

```python
def process_channel_pipeline(channel_name: str, timeframe: str = None, geo: str = None):
    # One-line topic retrieval with automatic niche resolution
    topics = handler.get_topics_by_channel(channel_name, timeframe=timeframe, geo=geo)

    # Use topics in your pipeline logic
    return topics
```

### Pattern 3: Batch Processing

```python
def process_multiple_channels(channels: list, timeframe: str = "today 1-m"):
    results = {}

    for channel in channels:
        niche = niche_from_channel(channel)
        topics = handler.get_topics_by_channel(channel, timeframe=timeframe)
        results[channel] = {"niche": niche, "topics": topics}

    return results

# Usage
channels = ["CKDrive", "cklegends", "CKIronWill"]
results = process_multiple_channels(channels)
```

## 📍 Supported Niche Mappings

| Channel Name | Niche | Description |
|--------------|-------|-------------|
| `CKDrive` | `automotive` | Car reviews, EV tech, racing |
| `cklegends` | `history` | Ancient mysteries, archaeology |
| `CKIronWill` | `motivation` | Personal development, discipline |
| `CKFinanceCore` | `finance` | Investing, markets, crypto |
| `CKCombat` | `combat` | MMA, boxing, martial arts |

## 🔄 Fallback System

The system includes a robust fallback mechanism:

1. **Online PyTrends** (6-second timeout)
2. **Offline trending topics** (if available)
3. **Rich seed topics** (26+ topics per niche)
4. **7-day deduplication** with backfill

## 📝 Example Files

- `pipeline_usage_example.py` - Complete working examples
- `README.md` - This documentation

## 🧪 Testing

Run the examples to verify everything works:

```bash
cd examples
python pipeline_usage_example.py
```

## 💡 Best Practices

1. **Always use helpers**: Don't hardcode niche strings
2. **Case-insensitive**: Channel names work in any case
3. **Fallback ready**: The system handles failures gracefully
4. **Batch processing**: Use for multiple channels efficiently
5. **Consistent naming**: Stick to the established channel naming convention

## 🚨 Error Handling

The helpers include robust error handling:

- Invalid channel names fall back to "history"
- Network failures fall back to offline/seed topics
- Timeouts are handled gracefully
- All operations are logged for debugging

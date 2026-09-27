---
name: video-silence-remover
description: Cut silent segments out of a video by detecting low audio levels and re-encoding the remaining clips.
---
# Remove silence from video

Measure the audio loudness per frame window, mark windows below the threshold as silent, and concatenate the rest.

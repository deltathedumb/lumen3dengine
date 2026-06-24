"""Audio -- sound effects (WAV) and music (OGG/MP3) via SDL2_mixer.

The engine-layer sound API for lumen3dengine:

    audio = Audio()
    audio.init()

    shoot_snd = audio.load_sound("shoot.wav")
    audio.play_sound(shoot_snd)

    audio.play_music("theme.ogg", loops=-1)  # -1 = loop forever
    audio.stop_music()

Sound effects (WAV) are played on numbered channels (up to max_channels,
default 16).  Music is a single global track; calling play_music while
music is playing stops the current track first.

Volumes are 0..128 (matching SDL2_mixer's MIX_MAX_VOLUME convention) so
scripts that read SDL docs feel at home, but set_master_volume(0.0..1.0)
is also available for the more common float-in-0-1 preference.
"""
from __future__ import annotations

import _audio_sdl as _mix


MAX_VOLUME: int = 128


class Sound:
    """A loaded WAV chunk.  Obtain via Audio.load_sound(), not directly."""
    _chunk: int
    _volume: int

    def __init__(self, chunk: int) -> None:
        self._chunk = chunk
        self._volume = MAX_VOLUME


class Music:
    """A loaded music track.  Obtain via Audio.load_music(), not directly."""
    _mus: int

    def __init__(self, mus: int) -> None:
        self._mus = mus


class Audio:
    """Top-level audio manager.  One instance per game (usually held by a
    GameLoop or created once at module level).  Call init() before any other
    method; call shutdown() on exit (or just let the process end -- SDL_mixer
    cleans up on exit anyway)."""

    _open: int
    max_channels: int

    def __init__(self) -> None:
        self._open = 0
        self.max_channels = 16

    def init(self) -> int:
        """Opens the audio device at 44100 Hz, stereo, 16-bit signed.
        Returns 0 on success, nonzero on SDL_mixer error."""
        if self._open == 1:
            return 0
        result: int = _mix.open(
            _mix.MIX_DEFAULT_FREQ,
            _mix.MIX_DEFAULT_FORMAT,
            _mix.MIX_DEFAULT_CHANNELS,
            2048,
        )
        if result == 0:
            self._open = 1
        return result

    def shutdown(self) -> None:
        if self._open == 1:
            _mix.close()
            self._open = 0

    def load_sound(self, path: str) -> Sound:
        """Load a WAV file from disk. Returns a Sound handle; keep it alive
        (don't let it go out of scope) for as long as you want to play it."""
        chunk: int = _mix.load_wav(path)
        return Sound(chunk)

    def free_sound(self, snd: Sound) -> None:
        """Release a loaded WAV chunk. Don't play it after calling this."""
        _mix.free(snd._chunk)
        snd._chunk = 0

    def play_sound(self, snd: Sound) -> int:
        """Play a sound effect once on the first available channel.
        Returns the channel number used, or -1 on error."""
        return _mix.play(snd._chunk, -1, 0)

    def play_sound_on(self, snd: Sound, channel: int) -> int:
        """Play a sound effect on a specific channel (0..max_channels-1),
        stopping whatever was playing there. Returns channel or -1."""
        return _mix.play(snd._chunk, channel, 0)

    def play_sound_loop(self, snd: Sound, loops: int) -> int:
        """Play a sound effect `loops` times (-1 = loop forever).
        Returns the channel used, or -1 on error."""
        return _mix.play(snd._chunk, -1, loops)

    def stop_sound(self, channel: int) -> None:
        """Stop a sound effect playing on `channel`.  Pass -1 to stop all."""
        _mix.halt(channel)

    def is_playing(self, channel: int) -> int:
        """Returns 1 if `channel` has a sound playing, 0 otherwise."""
        return _mix.playing(channel)

    def set_sound_volume(self, snd: Sound, volume: int) -> None:
        """Set a sound's volume (0..128).  Affects future play_sound calls."""
        snd._volume = volume
        _mix.volume(-1, volume)

    def load_music(self, path: str) -> Music:
        """Load a music track (OGG, MP3, WAV, MOD, etc.) from disk."""
        mus: int = _mix.load_mus(path)
        return Music(mus)

    def free_music(self, mus: Music) -> None:
        """Release a loaded music track."""
        _mix.free_mus(mus._mus)
        mus._mus = 0

    def play_music(self, mus: Music, loops: int) -> None:
        """Start playing a music track.  loops=-1 loops forever, loops=0
        plays once, loops=N plays N+1 times (SDL2_mixer convention).
        Stops any currently-playing music first."""
        if _mix.playing_mus() == 1:
            _mix.halt_mus()
        _mix.play_mus(mus._mus, loops)

    def stop_music(self) -> None:
        """Stop the currently-playing music track."""
        _mix.halt_mus()

    def is_music_playing(self) -> int:
        """Returns 1 if music is currently playing."""
        return _mix.playing_mus()

    def set_music_volume(self, volume: int) -> None:
        """Set global music volume (0..128)."""
        _mix.vol_mus(volume)

    def set_master_volume(self, v: float) -> None:
        """Set master sound+music volume from a 0.0..1.0 float."""
        vol: int = int(v * float(MAX_VOLUME))
        if vol < 0:
            vol = 0
        if vol > MAX_VOLUME:
            vol = MAX_VOLUME
        _mix.volume(-1, vol)
        _mix.vol_mus(vol)

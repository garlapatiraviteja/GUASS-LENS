"""

Image + video I/O utilities — pure Python / NumPy, no Pillow, no cv2.
"""

import numpy as np
import os
import zlib
import struct


# ─────────────────────────────────────────────────────────────────────────────
# PNG
# ─────────────────────────────────────────────────────────────────────────────

def _save_png(pixels: np.ndarray, path: str) -> None:
    H, W, C = pixels.shape

    def chunk(name, data):
        c = name + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)

    raw = b''
    for row in pixels:
        raw += b'\x00' + row.tobytes()

    png  = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw, 9))
    png += chunk(b'IEND', b'')

    with open(path, 'wb') as f:
        f.write(png)


def _unfilter_png(raw: bytes, H: int, row_bytes: int, bpp: int) -> np.ndarray:
    """Undo PNG scanline filters 0-4 (None/Sub/Up/Average/Paeth)."""
    out  = np.zeros((H, row_bytes), dtype=np.uint8)
    prev = np.zeros(row_bytes, dtype=np.int32)
    pos  = 0
    for y in range(H):
        ft   = raw[pos]
        line = np.frombuffer(raw, dtype=np.uint8, count=row_bytes, offset=pos + 1).astype(np.int32)
        pos += 1 + row_bytes
        cur  = np.zeros(row_bytes, dtype=np.int32)
        if ft == 0:
            cur = line
        elif ft == 1:
            for c in range(bpp):
                cur[c::bpp] = np.cumsum(line[c::bpp]) % 256
        elif ft == 2:
            cur = (line + prev) % 256
        elif ft == 3:
            for i in range(row_bytes):
                a = cur[i - bpp] if i >= bpp else 0
                cur[i] = (line[i] + ((a + prev[i]) >> 1)) % 256
        elif ft == 4:
            for i in range(row_bytes):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (line[i] + pr) % 256
        else:
            raise ValueError(f"Unsupported PNG filter type {ft}")
        out[y] = cur
        prev = cur
    return out


def _load_png(path: str) -> np.ndarray:
    """Load a PNG as float32 H x W x 3 in [0,1].  Uses Pillow when available (any PNG
    variant); otherwise a pure-NumPy decoder for 8-bit gray/RGB/RGBA, non-interlaced."""
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    except ImportError:
        pass
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', "Not a PNG"
    pos, ihdr, idat = 8, None, b''
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos+4])[0]
        name, body = data[pos+4:pos+8], data[pos+8:pos+8+length]
        pos += 12 + length
        if name == b'IHDR':
            ihdr = body
        elif name == b'IDAT':
            idat += body
    W, H, depth, ctype, _, _, interlace = struct.unpack('>IIBBBBB', ihdr)
    channels = {0: 1, 2: 3, 6: 4}.get(ctype)
    if depth != 8 or channels is None or interlace != 0:
        raise ValueError("Pure-NumPy PNG decoder supports 8-bit gray/RGB/RGBA, non-interlaced "
                         "only; install Pillow for other PNG variants.")
    rows = _unfilter_png(zlib.decompress(idat), H, W * channels, channels)
    px = rows.reshape(H, W, channels)
    px = np.repeat(px, 3, axis=2) if channels == 1 else px[..., :3]
    return px.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# BMP
# ─────────────────────────────────────────────────────────────────────────────

def _save_bmp(pixels: np.ndarray, path: str) -> None:
    H, W, _ = pixels.shape
    row_size  = (W * 3 + 3) & ~3
    pad       = row_size - W * 3
    img_size  = row_size * H
    file_size = 54 + img_size

    header  = b'BM'
    header += struct.pack('<I', file_size)
    header += struct.pack('<HH', 0, 0)
    header += struct.pack('<I', 54)
    header += struct.pack('<IiiHHIIiiII', 40, W, -H, 1, 24, 0, img_size, 0, 0, 0, 0)

    with open(path, 'wb') as f:
        f.write(header)
        for row in pixels:
            bgr = row[:, ::-1].tobytes()
            f.write(bgr + b'\x00' * pad)


def _load_bmp(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        data = f.read()

    offset   = struct.unpack_from('<I', data, 10)[0]
    W        = struct.unpack_from('<i', data, 18)[0]
    H_raw    = struct.unpack_from('<i', data, 22)[0]
    H        = abs(H_raw)
    top_down = H_raw < 0

    row_size = (W * 3 + 3) & ~3
    pixels   = np.zeros((H, W, 3), dtype=np.uint8)

    for y in range(H):
        src_y = y if top_down else (H - 1 - y)
        start = offset + src_y * row_size
        row   = np.frombuffer(data[start:start + W * 3], dtype=np.uint8).reshape(W, 3)
        pixels[y] = row[:, ::-1]

    return pixels.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Public image API
# ─────────────────────────────────────────────────────────────────────────────

def save_image(img: np.ndarray, path: str) -> None:
    """Save float32 H×W×3 [0,1] image to PNG or BMP."""
    arr    = np.clip(img, 0.0, 1.0)
    pixels = (arr * 255.0).round().astype(np.uint8)
    ext    = os.path.splitext(path)[1].lower()
    if ext == '.bmp':
        _save_bmp(pixels, path)
    else:
        _save_png(pixels, path)


def load_image(path: str):
    """Load PNG or BMP → float32 H×W×3 [0,1]. Returns None if not found."""
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == '.bmp':
        return _load_bmp(path)
    else:
        return _load_png(path)


# ─────────────────────────────────────────────────────────────────────────────
# Video frame loader
# ─────────────────────────────────────────────────────────────────────────────

def load_video_frames(path: str, max_frames: int = 2000,
                      every_n: int = 1) -> list:
    """
    Extract frames from a video file → list of float32 H×W×3 [0,1] arrays.

    Strategy
    --------
    1. Try opencv (cv2)  — fastest, handles mp4/avi/mov/mkv/webm.
    2. Try imageio/ffmpeg — fallback if cv2 not installed.
    3. If the path is actually a directory of PNG/BMP frames, load them
       directly — useful for testing with pre-extracted frames.
    4. Returns [] if nothing works, so the caller can handle gracefully.
    """

    # ── Strategy 3: directory of image frames ────────────────────────────
    if os.path.isdir(path):
        exts   = {'.png', '.bmp', '.jpg', '.jpeg'}
        files  = sorted(f for f in os.listdir(path)
                        if os.path.splitext(f)[1].lower() in exts)
        frames = []
        for i, fname in enumerate(files):
            if i % every_n != 0:
                continue
            img = load_image(os.path.join(path, fname))
            if img is not None:
                frames.append(img)
            if len(frames) >= max_frames:
                break
        return frames

    # ── Strategy 1: opencv ───────────────────────────────────────────────
    try:
        import cv2
        cap    = cv2.VideoCapture(path)
        frames = []
        idx    = 0
        while cap.isOpened() and len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every_n == 0:
                # cv2 gives BGR uint8 → convert to RGB float32
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb.astype(np.float32) / 255.0)
            idx += 1
        cap.release()
        if frames:
            return frames
    except ImportError:
        pass

    # ── Strategy 2: imageio + ffmpeg ─────────────────────────────────────
    try:
        import imageio
        reader = imageio.get_reader(path, 'ffmpeg')
        frames = []
        for idx, frame in enumerate(reader):
            if len(frames) >= max_frames:
                break
            if idx % every_n == 0:
                frames.append(frame.astype(np.float32) / 255.0)
        reader.close()
        if frames:
            return frames
    except (ImportError, Exception):
        pass

    print(f"[io_utils] Warning: could not read video '{path}'. "
          f"Install opencv-python or imageio[ffmpeg].")
    return []